"""T14 单测：验证 SwingCounter 同时输出 c_bends 与 c_cycles。"""
from __future__ import annotations

import numpy as np

from wormlab.config import load_config
from wormlab.kinematics.swing import SwingCounter
from wormlab.schemas import Quality


def test_swing_counter_bends_and_cycles() -> None:
    cfg = load_config("swim_liquid").kinematics.swing
    fps = 30.0
    t = np.arange(int(10.0 * fps)) / fps

    # 1. c = 1.5 * sin(2π * 2t), 10s -> 20 个完整周期 (cycles=20±1, bends=40±1)
    c_sig = 1.5 * np.sin(2.0 * np.pi * 2.0 * t)
    sc = SwingCounter(cfg, fps=fps)
    for i, val in enumerate(c_sig):
        sc.step(i, float(val), valid=True)
    assert abs(sc.bends - 40) <= 1
    assert abs(sc.cycles - 20) <= 1

    # 2. 幅度 0.5 (< bend_high=0.9) -> 均为 0
    sc_small = SwingCounter(cfg, fps=fps)
    for i, val in enumerate(0.5 * np.sin(2.0 * np.pi * 2.0 * t)):
        sc_small.step(i, float(val), valid=True)
    assert sc_small.bends == 0
    assert sc_small.cycles == 0

    # 3. 叠加 ±0.25 高频抖动 -> 迟滞防抖保证计数仍为 40±1 / 20±1
    rng = np.random.default_rng(7)
    noisy = c_sig + rng.uniform(-0.25, 0.25, size=len(c_sig))
    sc_noisy = SwingCounter(cfg, fps=fps)
    for i, val in enumerate(noisy):
        sc_noisy.step(i, float(val), valid=True)
    assert abs(sc_noisy.bends - 40) <= 1
    assert abs(sc_noisy.cycles - 20) <= 1

    # 4. 离散序列 +, -, +（间隔足够） -> bends=3, cycles=1
    sc_seq = SwingCounter(cfg, fps=fps)
    sc_seq.step(0, 1.2, valid=True)
    sc_seq.step(10, -1.2, valid=True)
    sc_seq.step(20, 1.2, valid=True)
    assert sc_seq.bends == 3
    assert sc_seq.cycles == 1


def test_swing_flip_immunity_and_biased_prominence() -> None:
    cfg = load_config("swim_liquid").kinematics.swing
    fps = 30.0

    # 1. 头尾单帧 180° 翻转免疫：连续同侧弯曲期间某帧 u 反向且 c 变号 -> 不产生假翻转计数
    sc_flip = SwingCounter(cfg, fps=fps)
    u_fwd = np.array([1.0, 0.0], dtype=np.float32)
    u_rev = np.array([-1.0, 0.0], dtype=np.float32)
    for i in range(12):
        if i in (4, 8):
            sc_flip.step(i, -1.2, valid=True, u_curr=u_rev)
        else:
            sc_flip.step(i, 1.2, valid=True, u_curr=u_fwd)
    assert sc_flip.bends == 1

    # 2. 姿态偏置下的峰谷突出度计数：+0.45 + 0.85 * sin(2π * 2t) 范围 [-0.40, +1.30]
    t = np.arange(int(5.0 * fps)) / fps
    biased = 0.45 + 0.85 * np.sin(2.0 * np.pi * 2.0 * t)
    sc_bias = SwingCounter(cfg, fps=fps)
    for i, val in enumerate(biased):
        sc_bias.step(i, float(val), valid=True)
    assert sc_bias.cycles >= 8


def test_swing_continuous_rotation_no_false_flip() -> None:
    """08_SPEC C2: 游泳同时以 20°/帧 匀速整体旋转，新版 1 帧对齐参考绝不产生假翻转。"""
    cfg = load_config("swim_liquid").kinematics.swing
    fps = 30.0
    t = np.arange(int(10.0 * fps)) / fps
    c_sig = 1.4 * np.sin(2.0 * np.pi * 1.8 * t)

    sc = SwingCounter(cfg, fps=fps)
    for i, val in enumerate(c_sig):
        ang = np.deg2rad(20.0 * i)  # 连续整体旋转（远超 360°）
        u = np.array([float(np.cos(ang)), float(np.sin(ang))], dtype=np.float32)
        sc.step(i, float(val), valid=True, u_curr=u)

    assert sc.flip_corrections == 0  # 旧 EMA 实现在连续旋转下会误触发大量假翻转
    assert abs(sc.bends - 36) <= 2  # 1.8 Hz * 10s -> 36 bends
    assert abs(sc.cycles - 18) <= 1


def test_swim_shape_counter_ellipse_aspect() -> None:
    """08_SPEC C4: 合成 25px 小虫 C 形摆动掩码序列，免骨架计数误差 <= 1；静止死卷输出 0。"""
    import cv2

    from wormlab.kinematics.swim_shape import count_swim_from_shape
    from wormlab.pose.centerline import _compute_shape_moments
    from wormlab.schemas import Pose

    fps = 30.0
    duration_s = 6.0
    n_frames = int(duration_s * fps)
    f_swim = 1.5  # Hz
    body_len = 25.0
    width_px = 4
    n_seg = 40

    def _render_arc_mask(turn_total: float, center: tuple[float, float] = (60.0, 60.0)) -> np.ndarray:
        """把给定总转角的圆弧中心线渲染为小画布二值掩码。"""
        s = np.linspace(0.0, 1.0, n_seg)
        theta = turn_total * (s - 0.5)
        ds = body_len / (n_seg - 1)
        xs = center[0] + np.cumsum(np.cos(theta) * ds)
        ys = center[1] + np.cumsum(np.sin(theta) * ds)
        canvas = np.zeros((120, 120), dtype=np.uint8)
        pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=False, color=255, thickness=width_px, lineType=cv2.LINE_8)
        return canvas > 0

    poses: list[Pose] = []
    for i in range(n_frames):
        tt = i / fps
        turn_total = 2.4 * np.sin(2.0 * np.pi * f_swim * tt)  # 直 <-> C 形摆动
        mask = _render_arc_mask(turn_total)
        aspect, ecc, angle, skew = _compute_shape_moments(mask)
        poses.append(
            Pose(
                dense=None,
                length=body_len,
                chord=body_len * 0.8,
                width=float(width_px),
                area=float(mask.sum()),
                quality=Quality.OK,
                ell_aspect=aspect,
                ell_ecc=ecc,
                ell_angle=angle,
                ell_skew_minor=skew,
                roi_motion=3.0,  # 明确活跃
            )
        )

    cfg = load_config("swim_liquid").kinematics.swing
    c_cycles, c_bends, peaks = count_swim_from_shape(poses, fps=fps, cfg=cfg)
    expected_bends = int(round(duration_s * f_swim * 2))  # 每周期 2 个单侧 C 弯波峰
    assert abs(c_bends - expected_bends) <= 1
    assert abs(c_cycles - expected_bends // 2) <= 1
    assert len(peaks) == c_bends

    # 静止死卷：恒定深 C 形 + roi_motion 低于阈值 -> 0
    dead_poses: list[Pose] = []
    for i in range(n_frames):
        mask = _render_arc_mask(2.4)
        aspect, ecc, angle, skew = _compute_shape_moments(mask)
        dead_poses.append(
            Pose(
                dense=None,
                length=body_len,
                chord=body_len * 0.2,
                width=float(width_px),
                area=float(mask.sum()),
                quality=Quality.OK,
                ell_aspect=aspect,
                ell_ecc=ecc,
                ell_angle=angle,
                ell_skew_minor=skew,
                roi_motion=0.2,  # 低于 shape_motion_min=1.5
            )
        )
    d_cycles, d_bends, _ = count_swim_from_shape(dead_poses, fps=fps, cfg=cfg)
    assert d_cycles == 0
    assert d_bends == 0
