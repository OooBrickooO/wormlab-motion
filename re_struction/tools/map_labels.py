"""T02: 将 referAnswer 编号图上的标号映射到基准视频首帧检测框。

支持：
1. 去红字修复 (inpaint) + ORB + RANSAC 仿射估计；若琼脂背景纹理弱导致 ORB 内点不足，自动使用多尺度模板匹配精确定位缩放与平移变换 M。
2. YOLO (origin.pt, imgsz=416) 在匹配帧检测线虫，利用匈牙利算法将变换后的标号坐标一对一匹配至最近检测掩码/框。
3. 对无编号图的游泳视频（Swim-3 / Swim-4），导出第 0 帧和中间帧的检测候选框并生成默认映射。
4. 支持 benchmarks/labels/<id>.manual.json 手工覆盖。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy.optimize
import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("map_labels")


def imread_unicode(path: Path) -> np.ndarray:
    """支持中文/特殊字符路径的图像读取。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"无法读取图像: {path}")
    return img


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    """支持中文/特殊字符路径的图像保存。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError(f"编码图像失败: {path}")
    buf.tofile(str(path))


def remove_red_labels(img_bgr: np.ndarray) -> np.ndarray:
    """去除编号图上的红色手写/印刷数字并修复背景。"""
    b, g, r = cv2.split(img_bgr)
    mask = ((r.astype(np.int32) > 165) & (g.astype(np.int32) < 105) & (b.astype(np.int32) < 105)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)
    return cv2.inpaint(img_bgr, mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)


def estimate_transform_orb(clean_num: np.ndarray, frame: np.ndarray) -> tuple[np.ndarray | None, int]:
    """尝试用 ORB + RANSAC 估计从编号图到视频帧的 2x3 仿射矩阵 M。"""
    orb = cv2.ORB_create(nfeatures=3000)
    g1 = cv2.cvtColor(clean_num, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(g1, None)
    kp2, des2 = orb.detectAndCompute(g2, None)
    if des1 is None or des2 is None or len(kp1) < 12 or len(kp2) < 12:
        return None, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 12:
        return None, 0
    matches = sorted(matches, key=lambda m: m.distance)[:400]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    M, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if M is None or inliers is None:
        return None, 0
    sx = float(np.hypot(M[0, 0], M[0, 1]))
    rot_deg = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    if not (0.8 <= sx <= 2.5) or abs(rot_deg) > 3.0:
        return None, 0
    return M.astype(np.float64), int(inliers.sum())


def estimate_transform_template(clean_num: np.ndarray, frame: np.ndarray) -> tuple[np.ndarray, float]:
    """多尺度模板匹配求从编号图到视频帧的缩放+平移变换 M (2x3)。"""
    g_num = cv2.cvtColor(clean_num, cv2.COLOR_BGR2GRAY)
    g_frm = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 下采样 0.5 加速粗搜，再在原图精调
    g_num_s = cv2.resize(g_num, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    g_frm_s = cv2.resize(g_frm, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    nh, nw = g_num.shape[:2]
    fh, fw = g_frm.shape[:2]

    # 基础缩放比例接近 fw / nw
    base_scale = fw / float(nw)
    best_score = -1.0
    best_M = np.array([[base_scale, 0.0, 0.0], [0.0, base_scale, 0.0]], dtype=np.float64)

    for s_factor in np.linspace(0.92, 1.08, 17):
        scale = base_scale * float(s_factor)
        tw = int(round(nw * scale * 0.5))
        th = int(round(nh * scale * 0.5))
        if tw > g_frm_s.shape[1] or th > g_frm_s.shape[0] or tw < 32 or th < 32:
            continue
        tpl = cv2.resize(g_num_s, (tw, th), interpolation=cv2.INTER_LINEAR)
        res = cv2.matchTemplate(g_frm_s, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = float(max_val)
            tx = max_loc[0] * 2.0
            ty = max_loc[1] * 2.0
            best_M = np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float64)

    return best_M, best_score


def detect_worms(model: YOLO, frame: np.ndarray, device: str, imgsz: int = 416, conf: float = 0.20) -> list[dict[str, Any]]:
    """在单帧上运行 YOLO 分割并返回检测列表（含 bbox、质心、掩码轮廓点）。"""
    H, W = frame.shape[:2]
    min_area = W * H * 0.000010
    results = model([frame], verbose=False, device=device, imgsz=imgsz, conf=conf, retina_masks=True)
    r = results[0]
    dets: list[dict[str, Any]] = []
    if r.boxes is None or len(r.boxes) == 0:
        return dets

    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    masks = r.masks.data if r.masks is not None else None

    # 按置信度降序 + NMS (iou > 0.3)
    order = np.argsort(-confs)
    kept: list[int] = []
    for idx in order:
        b = boxes[idx]
        bw, bh = b[2] - b[0], b[3] - b[1]
        if bw * bh < min_area:
            continue
        overlap = False
        for k in kept:
            bk = boxes[k]
            ix1, iy1 = max(b[0], bk[0]), max(b[1], bk[1])
            ix2, iy2 = min(b[2], bk[2]), min(b[3], bk[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = bw * bh + (bk[2] - bk[0]) * (bk[3] - bk[1]) - inter + 1e-6
            if inter / union > 0.3:
                overlap = True
                break
        if not overlap:
            kept.append(int(idx))

    for det_idx, i in enumerate(kept):
        b = boxes[i].astype(float)
        cx = 0.5 * (b[0] + b[2])
        cy = 0.5 * (b[1] + b[3])
        mask_pts = np.array([[cx, cy]], dtype=np.float32)
        if masks is not None and i < masks.shape[0]:
            x0 = max(0, int(b[0]) - 8)
            y0 = max(0, int(b[1]) - 8)
            x1 = min(W, int(b[2]) + 8)
            y1 = min(H, int(b[3]) + 8)
            m_crop = (masks[i, y0:y1, x0:x1] > 0.5).cpu().numpy().astype(np.uint8)
            ys, xs = np.nonzero(m_crop)
            if len(xs) > 0:
                cx = float(xs.mean() + x0)
                cy = float(ys.mean() + y0)
                step = max(1, len(xs) // 200)
                mask_pts = np.stack([xs[::step] + x0, ys[::step] + y0], axis=1).astype(np.float32)
        dets.append(
            {
                "det_idx": det_idx,
                "bbox": [round(float(v), 2) for v in b],
                "conf": round(float(confs[i]), 4),
                "centroid": [round(cx, 2), round(cy, 2)],
                "mask_pts": mask_pts,
            }
        )
    return dets


def point_to_det_dist(pt_xy: np.ndarray, det: dict[str, Any]) -> float:
    """计算标号点到检测掩码点集的最小距离。"""
    pts = det["mask_pts"]
    dists = np.linalg.norm(pts - pt_xy[None, :], axis=1)
    return float(dists.min())


def process_video_entry(
    entry: dict[str, Any],
    repo_root: Path,
    out_dir: Path,
    model: YOLO,
    device: str,
) -> dict[str, Any]:
    vid_id = str(entry["id"])
    vid_path = repo_root / entry["video"]
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {vid_path}")

    sampled_frames: dict[int, np.ndarray] = {}
    for fidx in range(0, 35, 5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frm = cap.read()
        if ok and frm is not None:
            sampled_frames[fidx] = frm
    cap.release()
    if not sampled_frames:
        raise RuntimeError(f"视频读取帧失败: {vid_path}")

    manual_path = out_dir / f"{vid_id}.manual.json"
    manual_data = json.loads(manual_path.read_text(encoding="utf-8")) if manual_path.exists() else None

    num_img_rel = entry.get("numbering_image")
    if num_img_rel:
        num_img_path = repo_root / num_img_rel
        num_img = imread_unicode(num_img_path)
        clean_num = remove_red_labels(num_img)

        best_fidx = 0
        best_M: np.ndarray | None = None
        best_inliers = 0
        method_used = "orb_ransac"

        for fidx, frm in sampled_frames.items():
            M_orb, n_in = estimate_transform_orb(clean_num, frm)
            if M_orb is not None and n_in > best_inliers:
                best_inliers = n_in
                best_M = M_orb
                best_fidx = fidx

        if best_M is None or best_inliers < 30:
            method_used = "template_match"
            best_score = -1.0
            for fidx, frm in sampled_frames.items():
                M_tpl, score = estimate_transform_template(clean_num, frm)
                if score > best_score:
                    best_score = score
                    best_M = M_tpl
                    best_fidx = fidx
            log.info("[%s] ORB 内点=%d (<30)，使用多尺度模板匹配: frame=%d, score=%.4f", vid_id, best_inliers, best_fidx, best_score)
        else:
            log.info("[%s] ORB+RANSAC 匹配成功: frame=%d, inliers=%d", vid_id, best_fidx, best_inliers)

        assert best_M is not None
        labels_xy: dict[Any, list[float]] = entry.get("labels_xy", {})
        gt_map: dict[Any, int] = entry.get("gt", {})
        labels_list = list(labels_xy.keys())

        mapped_pts: dict[Any, np.ndarray] = {}
        for lbl in labels_list:
            pt = np.array([labels_xy[lbl][0], labels_xy[lbl][1], 1.0], dtype=np.float64)
            xy = best_M @ pt
            mapped_pts[lbl] = xy

        # 在采样帧中选择匹配标号数最多（且平均距离最小）的帧作为基准检测帧
        best_eval_tuple = (-1, -1e9, best_fidx, [], [])
        for fidx, frm in sampled_frames.items():
            cand_dets = detect_worms(model, frm, device=device, imgsz=416, conf=0.15)
            cand_assign: list[dict[str, Any]] = []
            if len(cand_dets) > 0 and len(labels_list) > 0:
                cost = np.full((len(labels_list), len(cand_dets)), 1e6, dtype=np.float64)
                for i, lbl in enumerate(labels_list):
                    for j, d in enumerate(cand_dets):
                        cost[i, j] = point_to_det_dist(mapped_pts[lbl], d)
                row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)
                for r_i, c_i in zip(row_ind, col_ind):
                    lbl = labels_list[r_i]
                    dist = float(cost[r_i, c_i])
                    if dist <= 180.0:
                        d = cand_dets[c_i]
                        cand_assign.append(
                            {
                                "label": int(lbl) if str(lbl).isdigit() else lbl,
                                "gt": int(gt_map[lbl]),
                                "det_idx": d["det_idx"],
                                "det_bbox": d["bbox"],
                                "centroid": d["centroid"],
                                "mapped_label_xy": [round(float(mapped_pts[lbl][0]), 1), round(float(mapped_pts[lbl][1]), 1)],
                                "match_dist_px": round(dist, 2),
                            }
                        )
            score_dist = -sum(a["match_dist_px"] for a in cand_assign)
            cand_tuple = (len(cand_assign), score_dist, fidx, cand_assign, cand_dets)
            if cand_tuple[:2] > best_eval_tuple[:2]:
                best_eval_tuple = cand_tuple

        _, _, best_fidx, assignments, dets = best_eval_tuple
        matched_lbl_set = {a["label"] for a in assignments}
        unmatched_labels = [lbl for lbl in labels_list if (int(lbl) if str(lbl).isdigit() else lbl) not in matched_lbl_set]

        # 对极小或低对比度而未匹配的标注虫，在映射点局部 ROI 放大或 imgsz=640 下补充定位 bbox
        if unmatched_labels:
            frm_best = sampled_frames[best_fidx]
            H, W = frm_best.shape[:2]
            remaining: list[Any] = []
            for lbl in unmatched_labels:
                lx, ly = float(mapped_pts[lbl][0]), float(mapped_pts[lbl][1])
                rx0, ry0 = max(0, int(lx - 140)), max(0, int(ly - 140))
                rx1, ry1 = min(W, int(lx + 140)), min(H, int(ly + 140))
                roi = frm_best[ry0:ry1, rx0:rx1]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                med = float(np.median(gray_roi))
                bw = (gray_roi > med + 25).astype(np.uint8) * 255
                num_cc, cc_map, stats, centroids = cv2.connectedComponentsWithStats(bw)
                best_cc = None
                best_d = 1e9
                for cid in range(1, num_cc):
                    area = stats[cid, cv2.CC_STAT_AREA]
                    if area < 40 or area > 15000:
                        continue
                    cx_c = rx0 + float(centroids[cid][0])
                    cy_c = ry0 + float(centroids[cid][1])
                    d = float(np.hypot(cx_c - lx, cy_c - ly))
                    if d < best_d and d <= 150.0:
                        best_d = d
                        x_c = rx0 + int(stats[cid, cv2.CC_STAT_LEFT])
                        y_c = ry0 + int(stats[cid, cv2.CC_STAT_TOP])
                        w_c = int(stats[cid, cv2.CC_STAT_WIDTH])
                        h_c = int(stats[cid, cv2.CC_STAT_HEIGHT])
                        best_cc = ([float(x_c), float(y_c), float(x_c + w_c), float(y_c + h_c)], [round(cx_c, 2), round(cy_c, 2)], round(d, 2))
                if best_cc is not None:
                    assignments.append(
                        {
                            "label": int(lbl) if str(lbl).isdigit() else lbl,
                            "gt": int(gt_map[lbl]),
                            "det_idx": len(dets),
                            "det_bbox": best_cc[0],
                            "centroid": best_cc[1],
                            "mapped_label_xy": [round(lx, 1), round(ly, 1)],
                            "match_dist_px": best_cc[2],
                        }
                    )
                else:
                    remaining.append(lbl)
            unmatched_labels = remaining
            assignments.sort(key=lambda x: int(x["label"]) if str(x["label"]).isdigit() else str(x["label"]))

        result_doc = {
            "video_id": vid_id,
            "kind": entry["kind"],
            "frame_idx": best_fidx,
            "method": method_used,
            "M": [[round(float(v), 6) for v in row] for row in best_M],
            "assignments": assignments,
            "unmatched_labels": unmatched_labels,
            "num_detections": len(dets),
        }
    else:
        # 无编号图的 Swim 视频：使用第 0 帧检测并输出编号图
        best_fidx = 0
        frame = sampled_frames[0]
        # Swim 小虫同时尝试 416 和 640 保底检出首帧候选框
        dets = detect_worms(model, frame, device=device, imgsz=416, conf=0.15)
        if not dets:
            dets = detect_worms(model, frame, device=device, imgsz=640, conf=0.15)
        gt_unordered = list(entry.get("gt_unordered", []))
        assignments = []
        for idx, d in enumerate(dets):
            gt_val = int(gt_unordered[idx]) if idx < len(gt_unordered) else None
            assignments.append(
                {
                    "label": idx + 1,
                    "gt": gt_val,
                    "det_idx": d["det_idx"],
                    "det_bbox": d["bbox"],
                    "centroid": d["centroid"],
                    "mapped_label_xy": d["centroid"],
                    "match_dist_px": 0.0,
                }
            )
        result_doc = {
            "video_id": vid_id,
            "kind": entry["kind"],
            "frame_idx": best_fidx,
            "method": "swim_frame0_dets",
            "M": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "gt_unordered": gt_unordered,
            "assignments": assignments,
            "unmatched_labels": [],
            "num_detections": len(dets),
        }

    if manual_data:
        result_doc["manual_override"] = manual_data

    # 绘制 overlay 图供人工目检
    vis = sampled_frames[best_fidx].copy()
    for d in dets:
        x1, y1, x2, y2 = [int(round(v)) for v in d["bbox"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (180, 180, 180), 1)

    for a in result_doc["assignments"]:
        x1, y1, x2, y2 = [int(round(v)) for v in a["det_bbox"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        lx, ly = [int(round(v)) for v in a["mapped_label_xy"]]
        cx, cy = [int(round(v)) for v in a["centroid"]]
        cv2.circle(vis, (lx, ly), 5, (0, 0, 255), -1)
        cv2.line(vis, (lx, ly), (cx, cy), (0, 255, 255), 2)
        txt = f"#{a['label']} (gt={a['gt']})"
        cv2.putText(vis, txt, (max(5, x1), max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    json_path = out_dir / f"{vid_id}.json"
    overlay_path = out_dir / f"{vid_id}.overlay.jpg"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    imwrite_unicode(overlay_path, vis)
    log.info(
        "[%s] 已保存 %s (匹配 %d, 未匹配 %d)",
        vid_id,
        json_path.name,
        len(result_doc["assignments"]),
        len(result_doc["unmatched_labels"]),
    )
    return result_doc


def main() -> int:
    re_root = Path(__file__).resolve().parents[1]
    repo_root = re_root.parent
    gt_yaml = re_root / "benchmarks" / "ground_truth.yaml"
    out_dir = re_root / "benchmarks" / "labels"

    cfg_data = yaml.safe_load(gt_yaml.read_text(encoding="utf-8"))
    model_path = re_root / "backend" / "models" / "origin.pt"
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))
    model.to(device)

    for entry in cfg_data.get("videos", []):
        process_video_entry(entry, repo_root, out_dir, model, device)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
