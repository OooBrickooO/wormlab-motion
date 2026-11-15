"""04 §4: 三阶段匈牙利与残留重识别关联算法 (associate)。

与旧实现 EnhancedWormTracker.associate_detections_to_tracks 保持 1:1 逻辑与权重一致。
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from wormlab.config import TrackingCfg
from wormlab.schemas import Detection

from . import cost as cost_fn
from .features import extract_features
from .reid import match_gate, reid_stage3
from .track import Track


def _extract_det_feature(det: Detection) -> np.ndarray:
    if det.mask is None:
        return np.zeros(11, dtype=np.float64)
    ox, oy = det.mask_origin
    b = det.bbox
    bbox_in_crop = np.array([b[0] - ox, b[1] - oy, b[2] - ox, b[3] - oy], dtype=np.float64)
    return extract_features(det.mask, bbox_in_crop)


def associate(
    tracks: list[Track],
    dets: list[Detection],
    cfg: TrackingCfg,
    frame_wh: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[int], list[int], list[np.ndarray]]:
    """执行三阶段检测-轨迹关联，返回 (matches, unmatched_dets, unmatched_tracks, det_features)。"""
    num_tracks = len(tracks)
    num_dets = len(dets)
    if num_tracks == 0 or num_dets == 0:
        return [], list(range(num_dets)), list(range(num_tracks)), [np.zeros(11, dtype=np.float64) for _ in range(num_dets)]

    # 每个轨迹恰好预测一次
    pred_bboxes = [t.predict() for t in tracks]

    det_centers: list[tuple[float, float]] = []
    det_sizes: list[float] = []
    det_features: list[np.ndarray] = []

    for d in dets:
        b = d.bbox
        det_centers.append(((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0))
        det_sizes.append(float((b[2] - b[0]) + (b[3] - b[1])))
        det_features.append(_extract_det_feature(d))

    track_features = [t.get_feature() for t in tracks]
    cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float64)

    w_iou = float(cfg.weights.iou)
    w_mot = float(cfg.weights.motion)
    w_dir = float(cfg.weights.direction)
    w_app = float(cfg.weights.appearance)

    # ── 阶段 1：主匈牙利匹配 ──
    for t_idx in range(num_tracks):
        track = tracks[t_idx]
        pred_bbox = pred_bboxes[t_idx]
        gate = match_gate(track, cfg, frame_wh)

        for d_idx in range(num_dets):
            det_bbox = dets[d_idx].bbox
            c_dist = cost_fn.center_dist(pred_bbox, det_bbox)
            if c_dist > gate:
                cost_matrix[t_idx, d_idx] = 1e6
                continue

            iou_val = cost_fn.iou(pred_bbox, det_bbox)
            iou_c = 1.0 - iou_val
            mot_c = cost_fn.motion_cost(track, det_bbox, det_sizes[d_idx], max_speed=cfg.max_speed)
            dir_c = cost_fn.direction_cost(track, det_centers[d_idx])
            app_d = cost_fn.appearance_dist(track_features[t_idx], det_features[d_idx])
            app_c = min(app_d / 5.0, 1.0)
            conf_pen = (1.0 - float(dets[d_idx].conf)) * 0.1

            combined = (
                w_iou * iou_c
                + w_mot * mot_c
                + w_dir * dir_c
                + w_app * app_c
                + conf_pen
            )
            if iou_val < float(cfg.iou_match_thresh) and track.lost_frames == 0:
                combined += 0.5

            cost_matrix[t_idx, d_idx] = combined

    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    matches: list[tuple[int, int]] = []
    used_tracks: set[int] = set()
    used_dets: set[int] = set()

    for r_i, c_i in zip(row_idx, col_idx):
        if cost_matrix[r_i, c_i] < float(cfg.cost_threshold):
            matches.append((int(r_i), int(c_i)))
            used_tracks.add(int(r_i))
            used_dets.add(int(c_i))

    unmatched_tracks = [t for t in range(num_tracks) if t not in used_tracks]
    unmatched_dets = [d for d in range(num_dets) if d not in used_dets]

    # ── 阶段 2：高 IoU 二次匹配 ──
    for t_idx in list(unmatched_tracks):
        best_d = -1
        best_cost = 1e6
        gate = match_gate(tracks[t_idx], cfg, frame_wh)
        for d_idx in unmatched_dets:
            if d_idx in used_dets:
                continue
            iou_val = cost_fn.iou(pred_bboxes[t_idx], dets[d_idx].bbox)
            if iou_val < float(cfg.iou_secondary):
                continue
            if cost_fn.center_dist(pred_bboxes[t_idx], dets[d_idx].bbox) > gate:
                continue
            mot_c = cost_fn.motion_cost(tracks[t_idx], dets[d_idx].bbox, det_sizes[d_idx], max_speed=cfg.max_speed)
            app_d = cost_fn.appearance_dist(track_features[t_idx], det_features[d_idx])
            app_c = min(app_d / 5.0, 1.0)
            c_val = 0.3 * (1.0 - iou_val) + 0.4 * mot_c + 0.3 * app_c
            if c_val < best_cost:
                best_cost = c_val
                best_d = d_idx

        if best_d != -1 and best_cost < 1.2:
            matches.append((t_idx, best_d))
            used_tracks.add(t_idx)
            used_dets.add(best_d)

    unmatched_tracks = [t for t in range(num_tracks) if t not in used_tracks]
    unmatched_dets = [d for d in range(num_dets) if d not in used_dets]

    # ── 阶段 3：丢失轨迹视觉残留重识别 (Re-ID) ──
    s3_matches, unmatched_dets, unmatched_tracks = reid_stage3(
        tracks=tracks,
        pred_bboxes=pred_bboxes,
        track_features=track_features,
        dets=dets,
        det_sizes=det_sizes,
        det_features=det_features,
        unmatched_tracks=unmatched_tracks,
        unmatched_dets=unmatched_dets,
        cfg=cfg,
        frame_wh=frame_wh,
    )
    matches.extend(s3_matches)

    return matches, unmatched_dets, unmatched_tracks, det_features
