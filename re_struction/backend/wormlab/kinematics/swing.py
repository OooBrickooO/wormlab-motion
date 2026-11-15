"""03 §5: 游泳摆动迟滞过零计数器 (SwingCounter) —— 同时输出完整周期 (c_cycles) 与单次弯曲 (c_bends)。"""
from __future__ import annotations

from wormlab.config import SwingCfg


class SwingCounter:
    """对总转角 c(t) 做迟滞双侧高位计数，同时维护 c_bends 与 c_cycles。"""

    def __init__(self, cfg: SwingCfg, fps: float, gap_hold_s: float = 0.30) -> None:
        self.cfg = cfg
        self.fps = float(fps)
        self.min_flip_frames = max(1, int(round(float(cfg.min_flip_interval_s) * float(fps))))
        self.gap_hold_frames = max(1, int(round(float(gap_hold_s) * float(fps))))

        self.side: int = 0
        self.entries: int = 0
        self.flips: int = 0
        self.last_flip: int = -(10**9)
        self.last_high_frame: int = -(10**9)
        self.invalid_run: int = 0

    def step(self, i: int, c: float, valid: bool) -> None:
        if not valid:
            self.invalid_run += 1
            if self.invalid_run > self.gap_hold_frames:
                self.side = 0
            return

        self.invalid_run = 0
        thr = float(self.cfg.bend_high)
        new_side = 1 if c >= thr else (-1 if c <= -thr else 0)
        if new_side != 0:
            self.last_high_frame = int(i)

        if new_side == 0 or new_side == self.side:
            return

        if self.side != 0:
            if (int(i) - self.last_flip) < self.min_flip_frames:
                return
            self.flips += 1
            self.last_flip = int(i)

        self.entries += 1
        self.side = new_side

    @property
    def bends(self) -> int:
        """单次弯曲次数 (c_bends)：左弯、右弯每次进入高位各算 1 次。"""
        return int(self.entries)

    @property
    def cycles(self) -> int:
        """完整往返周期数 (c_cycles)：左弯 + 右弯 = 1 次完整来回。"""
        return int(self.entries // 2)
