"""丢失已确认轨迹的滞后召回（416×416 ROI YOLO + 面积匹配静止背景差分，防偷框）。"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from wormlab.config import RecoveryCfg
from wormlab.geometry import iou_xyxy
from wormlab.schemas import Detection, TrackOut

from .tracker import Tracker


def compute_static_median_gray(
    video_path: str | Path,
    total_frames: int,
    n_samples: int = 20,
    cancel_check: Callable[[], bool] | None = None,
) -> np.ndarray | None:
    """均匀抽样若干帧计算逐像素静态中值灰度背景。"""
    p = Path(video_path)
    if not p.exists() or int(total_frames) < 5:
        return None
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return None
    try:
        count = max(3, min(int(n_samples), int(total_frames)))
        indices = np.linspace(0, max(0, int(total_frames) - 1), count, dtype=int)
        samples: list[np.ndarray] = []
        for idx in indices:
            if cancel_check is not None and cancel_check():
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frm = cap.read()
            if ok and frm is not None:
                samples.append(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY))
        if len(samples) < 3:
            return None
        stack = np.stack(samples, axis=0)
        return np.partition(stack, len(samples) // 2, axis=0)[len(samples) // 2]
    finally:
        cap.release()


def _box_center(box: np.ndarray) -> tuple[float, float]:
    return (
        0.5 * (float(box[0]) + float(box[2])),
        0.5 * (float(box[1]) + float(box[3])),
    )


def _passes_non_steal_guard(
    cand_box: np.ndarray,
    self_tid: int,
    self_center: tuple[float, float],
    busy_boxes: list[np.ndarray],
    other_centers: list[tuple[int, tuple[float, float], float]],
) -> bool:
    for bb in busy_boxes:
        if iou_xyxy(cand_box, bb) > 0.15:
            return False
    cx, cy = _box_center(cand_box)
    d_self = math.hypot(cx - self_center[0], cy - self_center[1])
    for otid, (ox, oy), o_span in other_centers:
        if otid == self_tid:
            continue
        d_other = math.hypot(cx - ox, cy - oy)
        if d_other < 0.9 * d_self and d_other < max(40.0, o_span):
            return False
    return True


def _pick_roi_yolo_candidate(
    cands: list[Detection],
    self_tid: int,
    pred_box: np.ndarray,
    ref_area: float,
    busy_boxes: list[np.ndarray],
    other_centers: list[tuple[int, tuple[float, float], float]],
) -> Detection | None:
    px, py = _box_center(pred_box)
    bw = max(1.0, float(pred_box[2] - pred_box[0]))
    bh = max(1.0, float(pred_box[3] - pred_box[1]))
    max_dist = max(80.0, 1.5 * max(bw, bh))

    best: Detection | None = None
    best_score = float("inf")
    for d in cands:
        cx, cy = _box_center(d.bbox)
        dist = math.hypot(cx - px, cy - py)
        if dist > max_dist:
            continue
        if d.mask is not None:
            area = float(d.mask.sum())
            if area < max(15.0, 0.35 * ref_area) or area > 2.2 * ref_area:
                continue
        if not _passes_non_steal_guard(d.bbox, self_tid, (px, py), busy_boxes, other_centers):
            continue
        score = dist - 10.0 * float(d.conf)
        if score < best_score:
            best_score = score
            best = d
    return best


def _find_bg_sub_candidate(
    frame_gray: np.ndarray,
    bg_gray: np.ndarray,
    self_tid: int,
    pred_box: np.ndarray,
    ref_area: float,
    thresholds: list[int],
    busy_boxes: list[np.ndarray],
    other_centers: list[tuple[int, tuple[float, float], float]],
) -> Detection | None:
    h, w = frame_gray.shape[:2]
    px, py = _box_center(pred_box)
    bw = max(1.0, float(pred_box[2] - pred_box[0]))
    bh = max(1.0, float(pred_box[3] - pred_box[1]))
    rad = max(70, int(round(1.5 * max(bw, bh))))

    rx0 = max(0, int(round(px)) - rad)
    ry0 = max(0, int(round(py)) - rad)
    rx1 = min(w, int(round(px)) + rad)
    ry1 = min(h, int(round(py)) + rad)
    if rx1 <= rx0 + 4 or ry1 <= ry0 + 4:
        return None

    diff = cv2.absdiff(bg_gray[ry0:ry1, rx0:rx1], frame_gray[ry0:ry1, rx0:rx1])
    min_a = max(20.0, 0.50 * ref_area)
    max_a = max(min_a + 1.0, 1.75 * ref_area)

    for thr in thresholds:
        fg = (diff >= int(thr)).astype(np.uint8)
        n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        best: tuple[float, Detection] | None = None
        for lbl in range(1, n_lbl):
            area = float(stats[lbl, cv2.CC_STAT_AREA])
            if area < min_a or area > max_a:
                continue
            lx = int(stats[lbl, cv2.CC_STAT_LEFT])
            ly = int(stats[lbl, cv2.CC_STAT_TOP])
            lw = int(stats[lbl, cv2.CC_STAT_WIDTH])
            lh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            gx0, gy0 = rx0 + lx, ry0 + ly
            gx1, gy1 = gx0 + lw, gy0 + lh
            cbox = np.array([float(gx0), float(gy0), float(gx1), float(gy1)], dtype=np.float32)
            if not _passes_non_steal_guard(cbox, self_tid, (px, py), busy_boxes, other_centers):
                continue
            sub_mask = (labels[ly : ly + lh, lx : lx + lw] == lbl).astype(bool)
            cand = Detection(
                bbox=cbox,
                conf=0.20,
                mask=sub_mask,
                mask_origin=(gx0, gy0),
                det_idx=-1,
            )
            cx, cy = _box_center(cbox)
            score = abs(area - ref_area) + 0.5 * math.hypot(cx - px, cy - py)
            if best is None or score < best[0]:
                best = (score, cand)
        if best is not None:
            return best[1]
    return None


def recover_missing_confirmed_tracks(
    frame_bgr: np.ndarray,
    bg_gray: np.ndarray | None,
    tracker: Tracker,
    detector: Any,
    outs: list[TrackOut],
    dets: list[Detection],
    ref_areas: dict[int, float],
    rec_cfg: RecoveryCfg,
    frame_idx: int,
) -> tuple[list[TrackOut], list[Detection]]:
    """对当前帧丢失的已确认轨迹依次尝试 ROI YOLO 与背景差分召回，并原地更新输出与检测列表。"""
    lost_confirmed = [t for t in tracker.tracks if t.confirmed and t.lost_frames >= 1]
    if not lost_confirmed:
        return outs, dets

    updated_outs = list(outs)
    updated_dets = list(dets)
    out_idx_by_tid = {o.track_id: idx for idx, o in enumerate(updated_outs)}

    busy_boxes: list[np.ndarray] = [np.asarray(o.bbox, dtype=np.float32) for o in updated_outs if not o.is_ghost and o.confirmed]
    other_centers: list[tuple[int, tuple[float, float], float]] = []
    for t in tracker.tracks:
        if not t.confirmed:
            continue
        tb = t.get_state()
        other_centers.append((t.track_id, _box_center(tb), max(float(tb[2] - tb[0]), float(tb[3] - tb[1]))))

    frame_gray: np.ndarray | None = None
    has_roi_yolo = hasattr(detector, "infer_roi")

    for track in lost_confirmed:
        pred_box = track.last_predicted_bbox if track.last_predicted_bbox is not None else track.get_state()
        bw = max(1.0, float(pred_box[2] - pred_box[0]))
        bh = max(1.0, float(pred_box[3] - pred_box[1]))
        ref_area = ref_areas.get(track.uid, 0.40 * bw * bh)
        px, py = _box_center(pred_box)

        rec_det: Detection | None = None
        if has_roi_yolo:
            roi_cands = detector.infer_roi(
                frame_bgr,
                center_xy=(px, py),
                roi_size=int(rec_cfg.roi_size),
                min_confidence=float(rec_cfg.roi_confidence),
            )
            rec_det = _pick_roi_yolo_candidate(roi_cands, track.track_id, pred_box, ref_area, busy_boxes, other_centers)

        if rec_det is None and rec_cfg.bg_sub_enabled and bg_gray is not None:
            if frame_gray is None:
                frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            rec_det = _find_bg_sub_candidate(
                frame_gray=frame_gray,
                bg_gray=bg_gray,
                self_tid=track.track_id,
                pred_box=pred_box,
                ref_area=ref_area,
                thresholds=[int(x) for x in rec_cfg.bg_thresholds],
                busy_boxes=busy_boxes,
                other_centers=other_centers,
            )

        if rec_det is None:
            continue

        new_det_idx = len(updated_dets)
        final_det = Detection(
            bbox=rec_det.bbox,
            conf=rec_det.conf,
            mask=rec_det.mask,
            mask_origin=rec_det.mask_origin,
            det_idx=new_det_idx,
        )
        updated_dets.append(final_det)
        rec_out = tracker.apply_recovered_detection(track.track_id, final_det, det_idx=new_det_idx, frame_idx=frame_idx)
        if rec_out is None:
            continue
        if track.track_id in out_idx_by_tid:
            updated_outs[out_idx_by_tid[track.track_id]] = rec_out
        else:
            out_idx_by_tid[track.track_id] = len(updated_outs)
            updated_outs.append(rec_out)
        busy_boxes.append(np.asarray(rec_out.bbox, dtype=np.float32))

    return updated_outs, updated_dets
