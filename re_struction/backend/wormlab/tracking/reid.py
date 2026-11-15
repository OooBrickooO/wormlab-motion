"""04 §4: 丢失轨迹关联门限计算与第三阶段视觉残留重识别 (Re-ID)。"""
from __future__ import annotations

import numpy as np

from wormlab.config import TrackingCfg
from wormlab.schemas import Detection

from . import cost as cost_fn
from .track import Track


def match_gate(track: Track, cfg: TrackingCfg, frame_wh: tuple[int, int]) -> float:
    """关联搜索半径：目标丢失越久搜索半径越大，在 ghost_reid_max_lost 帧后封顶。"""
    grown = min(float(track.lost_frames), float(cfg.ghost_reid_max_lost))
    base = float(cfg.max_speed) * (1.0 + grown / 20.0)
    limit = max(0.18 * float(max(frame_wh[0], frame_wh[1])), float(cfg.max_speed) * 1.5)
    return float(min(base, limit))


def reid_stage3(
    tracks: list[Track],
    pred_bboxes: list[np.ndarray],
    track_features: list[np.ndarray],
    dets: list[Detection],
    det_sizes: list[float],
    det_features: list[np.ndarray],
    unmatched_tracks: list[int],
    unmatched_dets: list[int],
    cfg: TrackingCfg,
    frame_wh: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """阶段 3：针对 lost_frames >= 3 的丢失轨迹按丢失时长降序执行宽松找回匹配。"""
    rem_tracks = list(unmatched_tracks)
    rem_dets = list(unmatched_dets)
    stage3_matches: list[tuple[int, int]] = []

    recovery_tracks = sorted(
        [t for t in rem_tracks if tracks[t].lost_frames >= 3],
        key=lambda t: -tracks[t].lost_frames,
    )

    for t in recovery_tracks:
        if t not in rem_tracks:
            continue
        track = tracks[t]
        pred_bbox = pred_bboxes[t]
        gate = match_gate(track, cfg, frame_wh)
        size_prev = float((pred_bbox[2] - pred_bbox[0]) + (pred_bbox[3] - pred_bbox[1]))

        best_d = -1
        best_cost = 1e6
        for d in list(rem_dets):
            det_bbox = dets[d].bbox
            dist = cost_fn.center_dist(pred_bbox, det_bbox)
            if dist > gate:
                continue
            size_det = float(det_sizes[d])
            ratio = max(size_prev, size_det) / (min(size_prev, size_det) + 1e-6)
            if ratio > 2.5:
                continue
            app_dist = cost_fn.appearance_dist(track_features[t], det_features[d])
            if app_dist > 8.0:
                continue
            c_val = (
                0.55 * min(dist / max(1e-6, gate), 1.0)
                + 0.25 * min(abs(ratio - 1.0), 1.0)
                + 0.20 * min(app_dist / 5.0, 1.0)
            )
            if c_val < best_cost:
                best_cost = c_val
                best_d = d

        if best_d >= 0 and best_cost < 0.85:
            stage3_matches.append((t, best_d))
            rem_tracks.remove(t)
            rem_dets.remove(best_d)

    return stage3_matches, rem_dets, rem_tracks
