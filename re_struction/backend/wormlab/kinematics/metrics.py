"""03 §8: 单帧运动学记录 (PerFrame)、实时快照 (LiveCounts) 与轨迹终稿报告 (TrackReport)。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from wormlab.schemas import Behavior, Quality


@dataclass
class PerFrame:
    """Pass B 渲染与逐帧日志所需的单帧运动学状态。"""

    frame_idx: int
    behavior: Behavior
    s_count: int
    c_cycles: int
    c_bends: int
    c_turn: int
    c_count: int
    wave_ok: bool
    phi: float | None
    c: float
    R: float
    quality: Quality
    activity_score: float = 0.0
    live_freq: float = 0.0
    avg_freq: float = 0.0
    bends_per_min: float = 0.0


@dataclass
class LiveCounts:
    """Pass A 实时推流使用的轨迹当前累计计数快照。"""

    s_count: int
    c_cycles: int
    c_bends: int
    c_turn_count: int
    c_count: int
    c_unit: str
    behavior: Behavior
    motion_type: str
    activity_score: float
    is_static: bool
    live_freq: float
    avg_freq: float
    cycles: int
    swings: int
    bends_per_min: float
    is_bidirectional: bool = False


@dataclass
class TrackReport:
    """单条轨迹的完整运动学汇总报告。"""

    s_count: int
    s_cycles_float: float
    c_cycles: int
    c_bends: int
    c_turn_count: int
    c_count: int
    c_unit: str
    rejected: dict[str, int]
    aborted_runs: int
    mode_time: dict[str, float]
    wave_eps_median: float
    observed_seconds: float
    freq_s_hz: float
    freq_c_hz: float
    avg_freq_legacy: float
    activity_score: float
    motion_type_legacy: str
    n_ok: int
    n_suspect: int
    n_bad: int
    n_missing: int
    spatial: dict[str, Any] = field(default_factory=dict)


def compute_spatial_stats(centers: list[np.ndarray], frame_height: int = 1080) -> dict[str, Any]:
    """按旧版 build_id_pose_summary 规范计算轨迹空间统计（左下角笛卡尔坐标换算仅在此汇总字典提供）。"""
    if len(centers) < 2:
        return {}
    pts = np.asarray(centers, dtype=np.float64)
    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    total_path = float(np.sum(seg_lens))
    net_disp = float(np.hypot(pts[-1, 0] - pts[0, 0], pts[-1, 1] - pts[0, 1]))
    h = float(frame_height)
    start_bl = [round(float(pts[0, 0]), 2), round(float(h - pts[0, 1]), 2)]
    end_bl = [round(float(pts[-1, 0]), 2), round(float(h - pts[-1, 1]), 2)]
    x_vals = pts[:, 0]
    y_vals = pts[:, 1]
    return {
        "total_path_length_px": round(total_path, 2),
        "net_displacement_px": round(net_disp, 2),
        "start_centroid_bl": start_bl,
        "end_centroid_bl": end_bl,
        "bbox_coverage_bl": {
            "x_min": round(float(x_vals.min()), 2),
            "y_min": round(float(h - y_vals.max()), 2),
            "x_max": round(float(x_vals.max()), 2),
            "y_max": round(float(h - y_vals.min()), 2),
        },
    }
