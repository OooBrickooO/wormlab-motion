"""01 §5: 通用几何工具函数。"""
from __future__ import annotations

import numpy as np


def arc_length(pts: np.ndarray) -> np.ndarray:
    """计算折线累积弧长数组 (n,)，首项为 0。"""
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or len(arr) == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(arr) == 1:
        return np.zeros((1,), dtype=np.float32)
    diffs = np.diff(arr, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    return cum.astype(np.float32)


def polyline_length(pts: np.ndarray) -> float:
    """计算折线总弧长。"""
    s = arc_length(pts)
    return float(s[-1]) if len(s) > 0 else 0.0


def resample_polyline(pts: np.ndarray, n: int) -> np.ndarray:
    """按弧长等距线性插值重采样折线至 n 个点，返回 (n, 2) float32。"""
    arr = np.asarray(pts, dtype=np.float64)
    if len(arr) == 0:
        return np.zeros((n, 2), dtype=np.float32)
    if len(arr) == 1 or n <= 1:
        return np.repeat(arr[:1], max(1, n), axis=0).astype(np.float32)

    s = arc_length(arr).astype(np.float64)
    total = float(s[-1])
    if total <= 1e-8:
        return np.repeat(arr[:1], n, axis=0).astype(np.float32)

    s_norm = s / total
    target = np.linspace(0.0, 1.0, n, dtype=np.float64)
    xs = np.interp(target, s_norm, arr[:, 0])
    ys = np.interp(target, s_norm, arr[:, 1])
    return np.stack([xs, ys], axis=1).astype(np.float32)


def tangent_angles(pts: np.ndarray) -> np.ndarray:
    """计算折线各段切线角 atan2(dy, dx) 并沿弧长 unwrap，返回 (n-1,) float32。"""
    arr = np.asarray(pts, dtype=np.float64)
    if len(arr) < 2:
        return np.zeros((0,), dtype=np.float32)
    diffs = np.diff(arr, axis=0)
    raw = np.arctan2(diffs[:, 1], diffs[:, 0])
    return np.unwrap(raw).astype(np.float32)


def wrap_pi(x: float | np.ndarray) -> float | np.ndarray:
    """将角度折叠到 (-π, π] 区间。"""
    res = (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi
    # 把 -π 映射到 +π 以满足 (-π, π]
    if np.ndim(res) == 0:
        val = float(res)
        return float(np.pi) if np.isclose(val, -np.pi) else val
    res = np.where(np.isclose(res, -np.pi), np.pi, res)
    return res


def iou_xyxy(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    """计算两个 xyxy 边界框的交并比 (IoU)。"""
    ax1, ay1, ax2, ay2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-9 else 0.0
