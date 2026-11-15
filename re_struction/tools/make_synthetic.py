"""T09 (07 §1): 合成线虫中心线序列与二值掩码渲染生成器。

无需人工标注即可对姿态层 (Pose) 与运动学层 (Kinematics) 进行解析真值验证。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

N_SEG = 32
S_MID = (np.arange(N_SEG, dtype=np.float64) + 0.5) / N_SEG


@dataclass
class Scn:
    """合成场景参数配置。"""

    kind: str
    duration_s: float = 20.0
    fps: float = 30.0
    length_px: float = 120.0
    width_px: float = 7.0
    A: float = 0.8  # 行波切线角幅度 (rad)
    f: float = 0.5  # 波频率 (Hz)
    k: float = 1.0  # 每体长波数
    disp_per_cycle: float = 0.4  # 每周期沿前进方向的体长位移数
    jitter_px: float = 0.0  # 中心线高斯抖动标准差 (px)
    heading: float = 0.0  # 前进方向角 (rad)
    mask_noise: bool = False  # 是否在渲染掩码时叠加随机形态学微扰
    seed: int = 0


def build_dense_from_theta(theta: np.ndarray, length_px: float, center_pos: np.ndarray) -> np.ndarray:
    """由 32 段切线角 (head→tail) 积分生成 33 点中心线，并平移使质心位于 center_pos。"""
    seg = float(length_px) / float(N_SEG)
    d = np.stack([np.cos(theta), np.sin(theta)], axis=1) * seg
    pts = np.vstack([np.zeros((1, 2), dtype=np.float64), np.cumsum(d, axis=0)])
    pts = pts - pts.mean(axis=0, keepdims=True) + np.asarray(center_pos, dtype=np.float64)[None, :]
    return pts.astype(np.float32)


def make_series(scn: Scn) -> tuple[np.ndarray, dict[str, Any]]:
    """生成 (T, 33, 2) float32 中心线序列（dense[0]=head）与期望真值字典 expected。"""
    T = max(2, int(round(float(scn.duration_s) * float(scn.fps))))
    t = np.arange(T, dtype=np.float64) / float(scn.fps)
    L = float(scn.length_px)
    A = float(scn.A)
    f = float(scn.f)
    k = float(scn.k)
    disp = float(scn.disp_per_cycle)

    th0 = float(scn.heading) + np.pi  # 切线从 head→tail 指向后方，前进方向为 heading
    fwd = np.array([np.cos(scn.heading), np.sin(scn.heading)], dtype=np.float64)
    origin_xy = np.array([300.0, 300.0], dtype=np.float64)

    theta = np.zeros((T, N_SEG), dtype=np.float64)
    centers = np.zeros((T, 2), dtype=np.float64)
    expected: dict[str, Any] = {
        "s_count": 0,
        "s_tol": 0,
        "direction": 0,
        "c_cycles": 0,
        "c_bends": 0,
        "c_tol": 0,
        "turn_count": 0,
    }

    kind = scn.kind
    if kind in ("crawl_fwd", "treadmill", "wrong_dir", "wobble", "standing", "head_wiggle"):
        for i, tt in enumerate(t):
            if kind == "wobble":
                ph = 0.4 * 2.0 * np.pi * np.sin(2.0 * np.pi * 0.2 * tt)
            else:
                ph = 2.0 * np.pi * f * tt

            if kind == "standing":
                th = A * np.sin(2.0 * np.pi * k * S_MID) * np.cos(2.0 * np.pi * f * tt)
            elif kind == "head_wiggle":
                th = np.where(
                    S_MID < 0.25,
                    0.6 * np.sin(2.0 * np.pi * 1.5 * tt) * np.sin(np.pi * S_MID / 0.25),
                    0.0,
                )
            else:
                th = A * np.sin(2.0 * np.pi * k * S_MID - ph)
            theta[i] = th0 + th

        v_map = {
            "crawl_fwd": disp,
            "treadmill": 0.0,
            "wrong_dir": -disp,
            "wobble": disp,
            "standing": disp,
            "head_wiggle": disp,
        }
        v_rel = v_map[kind]
        centers = origin_xy[None, :] + (t[:, None] * (v_rel * L * f)) * fwd[None, :]
        if kind == "crawl_fwd":
            n_cyc = int(round(scn.duration_s * f))
            expected.update({"s_count": n_cyc, "s_tol": 1, "direction": +1})

    elif kind == "crawl_bwd":
        for i, tt in enumerate(t):
            theta[i] = th0 + A * np.sin(2.0 * np.pi * k * S_MID + 2.0 * np.pi * f * tt)
        centers = origin_xy[None, :] - (t[:, None] * (disp * L * f)) * fwd[None, :]
        # 甲方规则：向后 S 不计入 s_count，记入 rejected.backward
        expected.update({"s_count": 0, "s_tol": 0, "direction": -1, "min_backward_rejected": 8})

    elif kind in ("fwd_then_back", "fwd_25_then_back"):
        half = scn.duration_s / 2.0
        cur_pos = origin_xy.copy()
        for i, tt in enumerate(t):
            if tt < half:
                theta[i] = th0 + A * np.sin(2.0 * np.pi * k * S_MID - 2.0 * np.pi * f * tt)
                cur_pos = cur_pos + fwd * (disp * L * f / scn.fps)
            else:
                tb = tt - half
                theta[i] = th0 + A * np.sin(
                    2.0 * np.pi * k * S_MID - 2.0 * np.pi * f * half + 2.0 * np.pi * f * tb
                )
                cur_pos = cur_pos - fwd * (disp * L * f / scn.fps)
            centers[i] = cur_pos
        fwd_done = int(np.floor(half * f))
        expected.update({"s_count": fwd_done, "s_tol": 0, "direction": +1, "min_aborted": 1})

    elif kind == "static_noise":
        theta[:] = th0
        centers[:] = origin_xy[None, :]

    elif kind in ("swim_thrash", "swim_small_amp"):
        # c(t) = theta[-6:].mean() - theta[:6].mean()
        # 对线性曲率 theta = th0 + kappa*(s - 0.5)，mean(s[-6:]) - mean(s[:6]) = 26/32 = 0.8125
        # 为使 |c| 峰值等于 target_c，令 kappa_amp = target_c / 0.8125
        f_swim = 2.0 if scn.f == 0.5 else scn.f
        target_c = 1.5 if kind == "swim_thrash" else 0.5
        kappa_amp = target_c / 0.8125
        for i, tt in enumerate(t):
            kappa = kappa_amp * np.sin(2.0 * np.pi * f_swim * tt)
            theta[i] = th0 + kappa * (S_MID - 0.5)
        centers[:] = origin_xy[None, :]
        if kind == "swim_thrash":
            full_cyc = int(round(scn.duration_s * f_swim))
            expected.update({"c_cycles": full_cyc, "c_bends": 2 * full_cyc, "c_tol": 2})

    elif kind == "omega":
        # 每 4s 产生一次深弯使 R: 0.95 -> 0.38 -> 0.95
        period_s = 4.0
        n_turns = int(scn.duration_s // period_s)
        for i, tt in enumerate(t):
            tau = (tt % period_s) / period_s  # [0, 1)
            # 在 tau ∈ [0.3, 0.7] 期间弯曲至 R≈0.38
            if 0.25 <= tau <= 0.75:
                u_turn = np.sin(np.pi * (tau - 0.25) / 0.50)
                bend_a = 0.5 + 3.4 * u_turn
            else:
                bend_a = 0.5
            theta[i] = th0 + bend_a * (S_MID - 0.5)
        centers[:] = origin_xy[None, :]
        expected.update({"turn_count": n_turns})

    else:
        raise ValueError(f"未知的合成场景类型: {kind}")

    dense = np.stack([build_dense_from_theta(theta[i], L, centers[i]) for i in range(T)], axis=0)
    rng = np.random.default_rng(scn.seed)
    if scn.jitter_px > 0.0:
        dense = dense + rng.normal(0.0, float(scn.jitter_px), size=dense.shape).astype(np.float32)
    elif kind == "static_noise":
        dense = dense + rng.normal(0.0, 0.5, size=dense.shape).astype(np.float32)

    return dense.astype(np.float32), expected


def render_mask(
    dense33: np.ndarray,
    width_px: float = 7.0,
    pad: int = 12,
    mask_noise: bool = False,
    extra_dense33: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """将 33 点中心线渲染为局部 bool 掩码与原点 (x0, y0)。可选传入第二条虫的中心线构造粘连 (merge) 掩码。"""
    pts_all = dense33 if extra_dense33 is None else np.vstack([dense33, extra_dense33])
    x0 = int(np.floor(pts_all[:, 0].min())) - int(pad) - int(np.ceil(width_px))
    y0 = int(np.floor(pts_all[:, 1].min())) - int(pad) - int(np.ceil(width_px))
    x1 = int(np.ceil(pts_all[:, 0].max())) + int(pad) + int(np.ceil(width_px))
    y1 = int(np.ceil(pts_all[:, 1].max())) + int(pad) + int(np.ceil(width_px))

    w = max(24, x1 - x0)
    h = max(24, y1 - y0)
    canvas = np.zeros((h, w), dtype=np.uint8)
    thickness = max(2, int(round(float(width_px))))

    offset = np.array([[x0, y0]], dtype=np.float32)
    p1 = np.round(dense33 - offset).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [p1], isClosed=False, color=255, thickness=thickness, lineType=cv2.LINE_8)

    if extra_dense33 is not None:
        p2 = np.round(extra_dense33 - offset).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [p2], isClosed=False, color=255, thickness=thickness, lineType=cv2.LINE_8)

    if mask_noise:
        rng = np.random.default_rng(42)
        edge = cv2.Canny(canvas, 50, 150) > 0
        noise_flip = rng.random(canvas.shape) < 0.15
        canvas[edge & noise_flip] = 255 - canvas[edge & noise_flip]

    return canvas > 0, (x0, y0)
