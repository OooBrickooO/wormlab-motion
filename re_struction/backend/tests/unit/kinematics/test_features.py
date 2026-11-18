"""T11 单测：验证 compute_features 对直线、半圆弧与解析正弦波的模态投影和符号不变量。"""
from __future__ import annotations

import numpy as np
import pytest

from wormlab.config import load_config
from wormlab.kinematics.features import compute_features
from wormlab.kinematics.modes import modal_phase
from wormlab.schemas import N_DENSE


def _dense_from_theta(theta: np.ndarray, L: float = 120.0) -> np.ndarray:
    seg = L / float(len(theta))
    d = np.stack([np.cos(theta), np.sin(theta)], axis=1) * seg
    pts = np.vstack([np.zeros((1, 2)), np.cumsum(d, axis=0)])
    return pts.astype(np.float32)


def test_features_straight_line() -> None:
    cfg = load_config("crawl_agar").kinematics
    pts = np.stack([np.linspace(120.0, 0.0, N_DENSE), np.zeros(N_DENSE)], axis=1).astype(np.float32)
    f = compute_features(pts, cfg)
    assert f is not None
    assert f.L == pytest.approx(120.0, rel=1e-4)
    assert f.R == pytest.approx(1.0, rel=1e-4)
    assert abs(f.a1) < 1e-5 and abs(f.a2) < 1e-5
    assert abs(f.c) < 1e-5
    # u 指向 head (120, 0)
    assert np.allclose(f.u, [1.0, 0.0], atol=1e-5)


def test_features_semicircle() -> None:
    cfg = load_config("crawl_agar").kinematics
    s_mid = (np.arange(32) + 0.5) / 32.0
    # 总转角为 π 的半圆弧，混合 cos(πs) 模态与去尖端首尾段差约为 0.65*2.5 + 0.35*(22/32)*π
    theta = np.pi * (s_mid - 0.5)
    pts = _dense_from_theta(theta, L=120.0)
    f = compute_features(pts, cfg)
    assert f is not None
    assert f.R == pytest.approx(2.0 / np.pi, rel=0.02)
    assert f.c == pytest.approx(0.65 * 2.5 + 0.35 * np.pi * (22.0 / 32.0), rel=0.02)


def test_features_analytic_traveling_wave_sign_invariant() -> None:
    cfg = load_config("crawl_agar").kinematics
    s_mid = (np.arange(32) + 0.5) / 32.0
    A = 0.8
    # 波从头向尾传播: θ(s, t) = π + A * sin(2π s - ω t)
    phis: list[float] = []
    for step in range(8):
        wt = step * 0.3
        theta = np.pi + A * np.sin(2.0 * np.pi * s_mid - wt)
        pts = _dense_from_theta(theta, L=120.0)
        f = compute_features(pts, cfg)
        assert f is not None
        assert np.hypot(f.a1, f.a2) == pytest.approx(A, rel=0.02)
        phis.append(modal_phase(f.a1, f.a2))

    # 相位 Φ 随时间单调递增
    unwrapped = np.unwrap(phis)
    diffs = np.diff(unwrapped)
    assert np.all(diffs > 0.25)


def test_non_integer_k_orthogonality_and_curved_crawl_parity() -> None:
    from wormlab.kinematics.counter import KinematicsEngine
    from wormlab.kinematics.modes import project_trig_modes
    from wormlab.schemas import Pose, Quality

    # 1. 非整数波数 k_body=1.25 正交化基不应向常曲率 (static C-bend) 泄漏明显分量
    s_mid = (np.arange(32) + 0.5) / 32.0
    th_const_c = 1.6 * (s_mid - 0.5)
    th_const_c -= np.mean(th_const_c)
    a1_c, a2_c = project_trig_modes(th_const_c, k_body=1.25)
    assert np.hypot(a1_c, a2_c) < 0.70

    # 2. 带有静曲偏移 + 1.5s 遮挡间隙的短波长前向爬行应准确计数且支持流式/批量一致性
    fps = 30.0
    n_frames = int(12.0 * fps)
    k = 1.15
    f = 0.5
    L = 120.0
    cfg = load_config("crawl_agar").kinematics
    poses: list[Pose | None] = []
    pos = np.array([200.0, 200.0], dtype=np.float32)
    for i in range(n_frames):
        t = i / fps
        if 150 <= i < 195:  # 1.5s 遮挡间隙
            poses.append(None)
            pos = pos + np.array([0.45 * L * f / fps, 0.0], dtype=np.float32)
            continue
        theta = np.pi + 0.75 * (s_mid - 0.5) + 0.9 * np.sin(2.0 * np.pi * k * s_mid - 2.0 * np.pi * f * t)
        pos = pos + np.array([0.45 * L * f / fps, 0.0], dtype=np.float32)
        base = _dense_from_theta(theta, L=L)
        dense = (base - np.mean(base, axis=0, keepdims=True) + pos).astype(np.float32)
        poses.append(Pose(dense=dense, length=L, chord=0.8 * L, width=6.0, area=600.0, quality=Quality.OK))

    rep_batch, pf_batch = KinematicsEngine.run(poses, fps=fps, cfg=cfg)
    assert rep_batch.s_count >= 4

    eng = KinematicsEngine(cfg=cfg, fps=fps)
    for idx, p in enumerate(poses):
        eng.push(idx, p)
    rep_stream, pf_stream = eng.finalize()
    assert rep_batch.s_count == rep_stream.s_count
    assert len(pf_batch) == len(pf_stream)
