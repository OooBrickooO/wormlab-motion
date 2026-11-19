"""T14 单测：验证 TurnCounter 迟滞深弯计数。"""
from __future__ import annotations

import pytest

from wormlab.config import load_config
from wormlab.kinematics.turns import TurnCounter


def test_turn_counter_hysteresis_and_cooldown() -> None:
    cfg = load_config("crawl_agar").kinematics.turn
    fps = 10.0
    tc = TurnCounter(cfg, fps=fps)

    # 3 次 0.9 -> 0.4(保持 0.5s=5帧) -> 0.9，每次间隔 4s (40帧 >= cooldown=25帧) -> count=3
    frame = 0
    for _ in range(3):
        for _ in range(20):
            tc.step(frame, 0.90, valid=True)
            frame += 1
        for _ in range(6):
            tc.step(frame, 0.40, valid=True)
            assert tc.in_curl is True
            frame += 1
        for _ in range(25):
            tc.step(frame, 0.90, valid=True)
            assert tc.in_curl is False
            frame += 1

    assert tc.count == 3

    # 只降到 0.65 (> R_enter=0.58) -> 不触发
    tc2 = TurnCounter(cfg, fps=fps)
    for i in range(10):
        tc2.step(i, 0.90, valid=True)
    for i in range(10, 20):
        tc2.step(i, 0.65, valid=True)
    for i in range(20, 30):
        tc2.step(i, 0.90, valid=True)
    assert tc2.count == 0

    # INIT 阶段直接从 0.4 开始，回升到 0.9 时不计为一次折返
    tc_init = TurnCounter(cfg, fps=fps)
    for i in range(10):
        tc_init.step(i, 0.40, valid=True)
    for i in range(10, 20):
        tc_init.step(i, 0.90, valid=True)
    assert tc_init.count == 0


def test_turn_counter_timeout_release_and_min_peak_c() -> None:
    cfg = load_config("crawl_agar").kinematics.turn
    fps = 10.0

    # 1. CURL 超时自动释放，防止永久锁死下游 S 计数
    tc_to = TurnCounter(cfg, fps=fps)
    for i in range(10):
        tc_to.step(i, 0.90, valid=True, c_val=0.1)
    for i in range(10, 10 + tc_to.max_curl_frames + 3):
        tc_to.step(i, 0.58, valid=True, c_val=1.6)
    assert tc_to.in_curl is False
    assert tc_to.count == 1

    # 2. 传入 c_val 但 |c| 未达 min_peak_c_rad（如掩码缺角导致 R 下降）时不计为有效弯
    tc_low_c = TurnCounter(cfg, fps=fps)
    for i in range(10):
        tc_low_c.step(i, 0.90, valid=True, c_val=0.1)
    for i in range(10, 16):
        tc_low_c.step(i, 0.55, valid=True, c_val=0.3)
    for i in range(16, 25):
        tc_low_c.step(i, 0.90, valid=True, c_val=0.1)
    assert tc_low_c.count == 0


def test_turn_counter_semicircle_c_bend() -> None:
    """08_SPEC C1: 180° 半圆 C 形（R=0.64, c=2.5 rad）经 R_c_enter/c_enter_rad 联合判定准确触发；
    直虫掩码缺角（R=0.64 但 c=0.3 rad）不误计。"""
    cfg = load_config("crawl_agar").kinematics.turn
    assert cfg.R_c_enter == pytest.approx(0.75)
    assert cfg.c_enter_rad == pytest.approx(1.65)
    fps = 10.0

    # 1. 标准 C 形弯：R=0.64 > R_enter=0.58（非深 Ω 弯），但 |c|=2.5 >= c_enter_rad=1.65 -> 触发
    tc = TurnCounter(cfg, fps=fps)
    for i in range(10):
        tc.step(i, 0.90, valid=True, c_val=0.1)
    for i in range(10, 16):
        tc.step(i, 0.64, valid=True, c_val=2.5)
        assert tc.in_curl is True
    for i in range(16, 26):
        tc.step(i, 0.90, valid=True, c_val=0.1)
    assert tc.count == 1

    # 2. 直虫掩码缺角：R=0.64 但 |c|=0.3 -> 既非深弯也非 C 形弯 -> 输出 0
    tc2 = TurnCounter(cfg, fps=fps)
    for i in range(10):
        tc2.step(i, 0.90, valid=True, c_val=0.1)
    for i in range(10, 16):
        tc2.step(i, 0.64, valid=True, c_val=0.3)
        assert tc2.in_curl is False
    for i in range(16, 26):
        tc2.step(i, 0.90, valid=True, c_val=0.1)
    assert tc2.count == 0
