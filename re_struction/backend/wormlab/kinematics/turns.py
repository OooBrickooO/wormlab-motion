"""03 §6: 爬行深弯 / Ω 折返计数器 (TurnCounter)。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wormlab.config import TurnCfg


@dataclass(frozen=True)
class TurnEvent:
    """一次被计入的 C/Ω 深弯事件。"""

    start_frame: int
    end_frame: int


class TurnCounter:
    """基于舒展比 R = D/L 的迟滞深弯状态机 (INIT -> EXT -> CURL -> EXT)。

    - CURL 驻留超过 max_curl_s 自动释放回 EXT，绝不永久锁死下游 S 计数；
    - 若调用方传入 c_val，则 CURL 期间 |c| 峰值需达到 min_peak_c_rad 才计数，
      以区分真实 C 弯与掩码残缺导致的弦长缩短；未传 c_val 时不卡该门槛（兼容旧调用）。
    """

    def __init__(self, cfg: TurnCfg, fps: float) -> None:
        self.cfg = cfg
        self.fps = float(fps)
        self.min_dwell_frames = max(1, int(round(float(cfg.min_dwell_s) * float(fps))))
        self.cooldown_frames = max(1, int(round(float(cfg.cooldown_s) * float(fps))))
        self.max_curl_frames = max(10, int(round(float(getattr(cfg, "max_curl_s", 2.0)) * float(fps))))
        self.min_peak_c_rad = float(getattr(cfg, "min_peak_c_rad", 0.85))
        self.gap_hold_frames = max(1, int(round(0.30 * float(fps))))

        self.state: str = "INIT"
        self.curl_start: int = 0
        self.last_turn: int = -(10**9)
        self.count: int = 0
        self.invalid_run: int = 0
        self.peak_c_in_curl: float = 0.0
        self.events: list[TurnEvent] = []

    def _try_count(self, i: int) -> None:
        dwell = int(i) - self.curl_start
        if dwell < self.min_dwell_frames or self.peak_c_in_curl < self.min_peak_c_rad:
            return
        if (int(i) - self.last_turn) < self.cooldown_frames:
            return
        self.count += 1
        self.last_turn = int(i)
        self.events.append(TurnEvent(start_frame=self.curl_start, end_frame=int(i)))

    def _release(self) -> None:
        self.state = "EXT"
        self.peak_c_in_curl = 0.0

    def step(self, i: int, R: float, valid: bool, c_val: float | None = None) -> None:
        if not getattr(self.cfg, "enabled", True):
            return
        r_val = float(R)
        if not valid or not np.isfinite(r_val):
            self.invalid_run += 1
            if self.state == "CURL" and self.invalid_run > self.gap_hold_frames:
                self._release()
            return
        self.invalid_run = 0

        r_enter = float(self.cfg.R_enter)
        r_exit = float(self.cfg.R_exit)
        c_abs = abs(float(c_val)) if (c_val is not None and np.isfinite(c_val)) else float("inf")

        if self.state == "INIT":
            if r_val >= r_exit:
                self.state = "EXT"
            return

        if self.state == "EXT":
            if r_val <= r_enter:
                self.state = "CURL"
                self.curl_start = int(i)
                self.peak_c_in_curl = c_abs
            return

        # state == "CURL"
        self.peak_c_in_curl = max(self.peak_c_in_curl, c_abs)
        dwell = int(i) - self.curl_start
        if dwell > self.max_curl_frames:
            # 超时强制释放，防止永久锁死 S 运动计数；转入 INIT 要求先舒展到 R_exit 才可再次进入 CURL，
            # 避免长时间保持卷曲被按 max_curl_s 周期性重复计数
            self._try_count(i)
            self._release()
            self.state = "INIT"
            return
        if r_val >= r_exit:
            self._try_count(i)
            self._release()

    @property
    def in_curl(self) -> bool:
        return self.state == "CURL"
