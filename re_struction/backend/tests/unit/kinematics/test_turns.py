"""T14 单测：验证 TurnCounter 迟滞深弯计数。"""
from __future__ import annotations

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

    # 只降到 0.65 (> R_enter=0.55) -> 不触发
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
