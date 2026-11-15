"""T04: 旧系统基线导出脚本 (dump_legacy_baseline.py)。

只读导入原工程 backend/worm_processing.py，在 benchmarks/ground_truth.yaml 的基准视频上运行旧版
EnhancedWormTracker + process_track_single，并将旧轨迹与 benchmarks/labels/*.json 中的标号对齐，
输出 benchmarks/baseline_legacy.json。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("dump_legacy_baseline")


def run_legacy_on_video(
    wp: Any,
    model: Any,
    video_path: Path,
    label_doc: dict[str, Any],
    max_frames: int | None = None,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None and max_frames > 0:
        total_frames = min(total_frames, max_frames)

    assignments = label_doc.get("assignments", [])
    init_count = max(len(assignments), int(label_doc.get("num_detections", len(assignments)))) or None
    tracker = wp.EnhancedWormTracker(
        fps=fps,
        frame_size=(width, height),
        sub_video_dir=None,
        max_worm_id=init_count,
        head_config=[],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8
    frame_idx = 0
    t0 = time.perf_counter()

    # 记录每条旧 tid 在前 45 帧内的质心，用于与标号对齐
    early_centroids: dict[int, list[tuple[int, float, float]]] = {}

    while frame_idx < total_frames:
        batch: list[np.ndarray] = []
        for _ in range(batch_size):
            if frame_idx + len(batch) >= total_frames:
                break
            ret, frm = cap.read()
            if not ret or frm is None:
                break
            batch.append(frm)
        if not batch:
            break

        batch_results = model(
            batch,
            verbose=False,
            device=device,
            retina_masks=True,
            imgsz=416,
            conf=wp.MIN_CONFIDENCE,
        )
        for i, res in enumerate(batch_results):
            cur_idx = frame_idx + i
            detections = wp.sv.Detections.from_ultralytics(res)
            if len(detections) > 0:
                x1, y1, x2, y2 = detections.xyxy[:, 0], detections.xyxy[:, 1], detections.xyxy[:, 2], detections.xyxy[:, 3]
                areas = (x2 - x1) * (y2 - y1)
                conf = detections.confidence if detections.confidence is not None else np.ones(len(detections))
                keep = (areas >= tracker.MIN_AREA) & (conf >= tracker.MIN_CONFIDENCE)
                detections = detections[keep]
            masks = detections.mask if hasattr(detections, "mask") else None
            tracker.update_tracking(detections, masks, cur_idx, frame=batch[i])
            if cur_idx <= 45:
                for trk in tracker.tracks:
                    tid = int(trk.track_id)
                    hist = tracker.track_history.get(tid)
                    if hist and len(hist) > 0:
                        cx, cy = float(hist[-1][0]), float(hist[-1][1])
                        early_centroids.setdefault(tid, []).append((cur_idx, cx, cy))

        frame_idx += len(batch)

    cap.release()
    elapsed = time.perf_counter() - t0
    log.info(
        "视频 %s 处理完成: %d 帧, 耗时 %.1fs (%.1f fps), 轨迹数=%d",
        video_path.name,
        frame_idx,
        elapsed,
        frame_idx / max(1e-3, elapsed),
        len(tracker.cycle_counts),
    )

    # 汇总所有旧轨迹
    all_tracks_out: dict[str, Any] = {}
    for tid, info in sorted(tracker.cycle_counts.items(), key=lambda kv: int(kv[0])):
        tid_int = int(tid)
        hist = list(tracker.track_history.get(tid, []))
        first_xy = [round(float(hist[0][0]), 2), round(float(hist[0][1]), 2)] if hist else None
        all_tracks_out[str(tid_int)] = {
            "track_id": tid_int,
            "c_count": int(info.get("c_count", 0)),
            "s_count": int(info.get("s_count", 0)),
            "cycles": int(info.get("cycles", 0)),
            "swings": int(info.get("swings", 0)),
            "motion_type": str(info.get("motion_type", "unknown")),
            "avg_freq": round(float(info.get("avg_freq", 0.0)), 4),
            "activity_score": round(float(info.get("activity_score", 0.0)), 4),
            "first_centroid": first_xy,
        }

    # 标号 -> 旧 tid 匹配
    label_mapping: dict[str, Any] = {}
    used_tids: set[int] = set()
    ref_fidx = int(label_doc.get("frame_idx", 0))

    for a in assignments:
        lbl = str(a["label"])
        lcx, lcy = float(a["centroid"][0]), float(a["centroid"][1])
        best_tid: int | None = None
        best_dist = 1e9
        for tid_int, pts in early_centroids.items():
            if tid_int in used_tids:
                continue
            # 取距离 ref_fidx 最近的帧的质心
            pts_sorted = sorted(pts, key=lambda p: abs(p[0] - ref_fidx))
            _, tx, ty = pts_sorted[0]
            d = float(np.hypot(tx - lcx, ty - lcy))
            if d < best_dist:
                best_dist = d
                best_tid = tid_int
        if best_tid is not None and best_dist <= 160.0:
            used_tids.add(best_tid)
            tinfo = all_tracks_out.get(str(best_tid), {})
            label_mapping[lbl] = {
                "legacy_tid": best_tid,
                "gt": a.get("gt"),
                "c_count": int(tinfo.get("c_count", 0)),
                "s_count": int(tinfo.get("s_count", 0)),
                "cycles": int(tinfo.get("cycles", 0)),
                "swings": int(tinfo.get("swings", 0)),
                "motion_type": tinfo.get("motion_type", "unknown"),
                "match_dist_px": round(best_dist, 2),
            }
        else:
            label_mapping[lbl] = {
                "legacy_tid": None,
                "gt": a.get("gt"),
                "c_count": 0,
                "s_count": 0,
                "cycles": 0,
                "swings": 0,
                "motion_type": "unmatched",
                "match_dist_px": None,
            }

    return {
        "frames_processed": frame_idx,
        "fps": round(fps, 2),
        "elapsed_s": round(elapsed, 2),
        "labels": label_mapping,
        "all_tracks": all_tracks_out,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出旧系统 worm_processing 基线 JSON")
    parser.add_argument("--max-frames", type=int, default=None, help="可选限制最大帧数（默认跑完整视频）")
    args = parser.parse_args(argv)

    re_root = Path(__file__).resolve().parents[1]
    repo_root = re_root.parent
    legacy_backend = repo_root / "backend"
    sys.path.insert(0, str(legacy_backend))

    import worm_processing as wp  # type: ignore[import-not-found]
    from ultralytics import YOLO

    model_path = re_root / "backend" / "models" / "origin.pt"
    model = YOLO(str(model_path))
    if torch.cuda.is_available():
        model.to("cuda")

    gt_yaml = re_root / "benchmarks" / "ground_truth.yaml"
    labels_dir = re_root / "benchmarks" / "labels"
    cfg_data = yaml.safe_load(gt_yaml.read_text(encoding="utf-8"))

    baseline_out: dict[str, Any] = {}
    for entry in cfg_data.get("videos", []):
        vid_id = str(entry["id"])
        vid_path = repo_root / entry["video"]
        lbl_path = labels_dir / f"{vid_id}.json"
        lbl_doc = json.loads(lbl_path.read_text(encoding="utf-8")) if lbl_path.exists() else {}
        log.info("正在运行旧系统基线: %s (%s) ...", vid_id, vid_path.name)
        baseline_out[vid_id] = run_legacy_on_video(wp, model, vid_path, lbl_doc, max_frames=args.max_frames)

    out_path = re_root / "benchmarks" / "baseline_legacy.json"
    out_path.write_text(json.dumps(baseline_out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("旧系统基线已保存至: %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
