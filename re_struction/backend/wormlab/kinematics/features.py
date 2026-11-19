"""03 §2: 单帧中心线运动学特征提取 (Feat)。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wormlab.config import KinCfg

from .modes import compute_total_turn_c, project_trig_modes


@dataclass
class Feat:
    """单帧中心线几何与模态特征。"""

    L: float  # 弧长 (px)
    chord: float  # 首尾直线弦长 (px)
    R: float  # 舒展比 chord / L ∈ [0, 1]
    theta: np.ndarray  # (32,) 沿 s 展开的切线角 (未减均值)
    th: np.ndarray  # (32,) 扣除空间均值的切线角 θ̃
    a1: float  # 余弦模态系数
    a2: float  # 正弦模态系数
    c: float  # 总转角 (尾20% - 头20%, rad)
    u: np.ndarray  # (2,) 指向头部的单位体轴向量
    center: np.ndarray  # (2,) 中心线质心坐标
    tangent_mid: np.ndarray | None = None  # (2,) 中段 25%~75% 指向头部的单位切向量（弧线爬行路径积分用）


def compute_features(dense: np.ndarray | None, cfg: KinCfg) -> Feat | None:
    """从 (33, 2) 稠密中心线计算单帧运动学特征；若输入无效返回 None。"""
    if dense is None or len(dense) < 4:
        return None
    pts = np.asarray(dense, dtype=np.float64)
    if not np.all(np.isfinite(pts)):
        return None

    diffs = np.diff(pts, axis=0)
    segs = np.hypot(diffs[:, 0], diffs[:, 1])
    L = float(segs.sum())
    if L <= 1e-6:
        return None

    head_to_tail_vec = pts[0] - pts[-1]
    chord = float(np.hypot(head_to_tail_vec[0], head_to_tail_vec[1]))
    R = float(np.clip(chord / L, 0.0, 1.0))

    theta = np.unwrap(np.arctan2(diffs[:, 1], diffs[:, 0])).astype(np.float64)
    th = theta - float(np.mean(theta))

    a1, a2 = project_trig_modes(th, k_body=cfg.k_body)
    c = compute_total_turn_c(theta)

    if chord > 1e-6:
        u = (head_to_tail_vec / chord).astype(np.float64)
    else:
        # 首尾重合时退化使用前半身指向头的方向
        mid_vec = pts[0] - pts[len(pts) // 2]
        n_mid = float(np.hypot(mid_vec[0], mid_vec[1]))
        u = (mid_vec / max(1e-6, n_mid)).astype(np.float64)

    center = pts.mean(axis=0).astype(np.float64)

    # 中段 25%~75% 指向头部的单位切向量：避开头端觅食摆动与尾端卷曲噪声（08_SPEC S4）
    n_pts = len(pts)
    i_head = max(0, n_pts // 4)
    i_tail = min(n_pts - 1, (3 * n_pts) // 4)
    mid_vec = pts[i_head] - pts[i_tail]
    mid_norm = float(np.hypot(mid_vec[0], mid_vec[1]))
    tangent_mid = (mid_vec / mid_norm).astype(np.float64) if mid_norm > 1e-6 else u.astype(np.float64)

    return Feat(
        L=L,
        chord=chord,
        R=R,
        theta=theta.astype(np.float32),
        th=th.astype(np.float32),
        a1=a1,
        a2=a2,
        c=c,
        u=u.astype(np.float32),
        center=center.astype(np.float32),
        tangent_mid=tangent_mid.astype(np.float32),
    )
