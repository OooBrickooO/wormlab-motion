"""04 §3.4: 跟踪关联代价纯函数集 (IoU、中心距离、方向一致性、运动一致性、外观距离)。"""
from __future__ import annotations

import math

import numpy as np

from .track import Track


def iou(bbox1: np.ndarray | list[float], bbox2: np.ndarray | list[float]) -> float:
    x1 = max(float(bbox1[0]), float(bbox2[0]))
    y1 = max(float(bbox1[1]), float(bbox2[1]))
    x2 = min(float(bbox1[2]), float(bbox2[2]))
    y2 = min(float(bbox1[3]), float(bbox2[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (float(bbox1[2]) - float(bbox1[0])) * (float(bbox1[3]) - float(bbox1[1]))
    area2 = (float(bbox2[2]) - float(bbox2[0])) * (float(bbox2[3]) - float(bbox2[1]))
    union = area1 + area2 - inter
    return float(inter / union) if union > 0.0 else 0.0


def center_dist(bbox1: np.ndarray | list[float], bbox2: np.ndarray | list[float]) -> float:
    cx1 = (float(bbox1[0]) + float(bbox1[2])) / 2.0
    cy1 = (float(bbox1[1]) + float(bbox1[3])) / 2.0
    cx2 = (float(bbox2[0]) + float(bbox2[2])) / 2.0
    cy2 = (float(bbox2[1]) + float(bbox2[3])) / 2.0
    return float(math.hypot(cx1 - cx2, cy1 - cy2))


def appearance_dist(feat1: np.ndarray, feat2: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(feat1, dtype=np.float64) - np.asarray(feat2, dtype=np.float64)))


def direction_cost(track: Track, det_center: tuple[float, float] | np.ndarray) -> float:
    v = track.get_motion_vector()
    norm_v = float(np.linalg.norm(v))
    if norm_v < 1e-6:
        return 0.0
    v_dir = v / norm_v
    pred_center = np.array(track.get_predicted_center(), dtype=np.float64)
    delta = np.asarray(det_center, dtype=np.float64) - pred_center
    norm_d = float(np.linalg.norm(delta))
    if norm_d < 1e-6:
        return 0.0
    delta_dir = delta / norm_d
    cos_sim = float(np.dot(v_dir, delta_dir))
    return float((1.0 - cos_sim) / 2.0)


def motion_cost(
    track: Track,
    det_bbox: np.ndarray | list[float],
    det_size: float | None = None,
    max_speed: float = 50.0,
) -> float:
    if len(track.history) < 2:
        return 0.0
    _, _, w_prev, h_prev = track.history[-1]
    cx_det = (float(det_bbox[0]) + float(det_bbox[2])) / 2.0
    cy_det = (float(det_bbox[1]) + float(det_bbox[3])) / 2.0
    w_det = float(det_bbox[2]) - float(det_bbox[0])
    h_det = float(det_bbox[3]) - float(det_bbox[1])
    if det_size is None:
        det_size = w_det + h_det

    cx_pred, cy_pred = track.get_predicted_center()
    vel_diff = math.hypot(cx_det - cx_pred, cy_det - cy_pred)
    vel_cost = min(vel_diff / (float(max_speed) + 1e-6), 1.0)

    size_prev = float(w_prev + h_prev)
    size_curr = float(w_det + h_det)
    size_ratio = max(size_curr, size_prev) / (min(size_curr, size_prev) + 1e-6)
    size_cost = min(abs(size_ratio - 1.0) / 0.5, 1.0)

    return float(0.5 * vel_cost + 0.5 * size_cost)
