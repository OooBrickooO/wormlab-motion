"""T14 单测：验证 SwingCounter 同时输出 c_bends 与 c_cycles。"""
from __future__ import annotations

import numpy as np

from wormlab.config import load_config
from wormlab.kinematics.swing import SwingCounter


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
