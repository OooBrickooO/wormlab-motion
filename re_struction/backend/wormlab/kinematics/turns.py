"""03 §6: 爬行深弯 / Ω 折返计数器 (TurnCounter)。"""
from __future__ import annotations

from wormlab.config import TurnCfg


class TurnCounter:
    """基于舒展比 R = D/L 的迟滞深弯状态机 (INIT -> EXT -> CURL -> EXT)。"""

    def __init__(self, cfg: TurnCfg, fps: float) -> None:
        self.cfg = cfg
        self.fps = float(fps)
        self.min_dwell_frames = max(1, int(round(float(cfg.min_dwell_s) * float(fps))))
        self.cooldown_frames = max(1, int(round(float(cfg.cooldown_s) * float(fps))))

        self.state: str = "INIT"
        self.curl_start: int = 0
        self.last_turn: int = -(10**9)
        self.count: int = 0

    def step(self, i: int, R: float, valid: bool) -> None:
        if not valid:
            return
        r_val = float(R)
        r_enter = float(self.cfg.R_enter)
        r_exit = float(self.cfg.R_exit)

        if self.state == "INIT":
            if r_val >= r_exit:
                self.state = "EXT"
            return

        if self.state == "EXT":
            if r_val <= r_enter:
                self.state = "CURL"
                self.curl_start = int(i)
        elif self.state == "CURL":
            if r_val >= r_exit:
                if (int(i) - self.curl_start) >= self.min_dwell_frames and (
                    int(i) - self.last_turn
                ) >= self.cooldown_frames:
                    self.count += 1
                    self.last_turn = int(i)
                self.state = "EXT"

    @property
    def in_curl(self) -> bool:
        return self.state == "CURL"
