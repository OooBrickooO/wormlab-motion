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
        # C 形弯联合触发阈值（08_SPEC C1）：均匀圆弧 180° 半圆 R=2/π≈0.637，
        # 单靠 R_enter≈0.58 需弯曲 >=196° 已是 Ω 环；R_c_enter + c_enter_rad 联合判定标准 145°~180° C 形
        self.r_c_enter = getattr(cfg, "R_c_enter", None)
        self.c_enter_rad = getattr(cfg, "c_enter_rad", None)

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

    def _r_exit_eff(self) -> float:
        """C 形弯触发阈值较深时，退出阈值同步抬高（如 R_c_enter=0.75 -> R_exit=0.82）。"""
        r_exit = float(self.cfg.R_exit)
        if self.r_c_enter is not None:
            r_exit = max(r_exit, float(self.r_c_enter) + 0.08)
        return r_exit

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
        r_exit_eff = self._r_exit_eff()
        c_abs = abs(float(c_val)) if (c_val is not None and np.isfinite(c_val)) else float("inf")

        # EXT -> CURL 触发：深 Ω 弯 (R <= R_enter) 或 标准 C 形弯 (R <= R_c_enter 且 |c| >= c_enter_rad)
        deep_trigger = r_val <= r_enter
        c_bend_trigger = bool(
            self.r_c_enter is not None
            and self.c_enter_rad is not None
            and c_val is not None
            and np.isfinite(c_val)
            and r_val <= float(self.r_c_enter)
            and abs(float(c_val)) >= float(self.c_enter_rad)
        )

        if self.state == "INIT":
            if r_val >= r_exit_eff:
                self.state = "EXT"
            return

        if self.state == "EXT":
            if deep_trigger or c_bend_trigger:
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
        if r_val >= r_exit_eff:
            self._try_count(i)
            self._release()

    @property
    def in_curl(self) -> bool:
        return self.state == "CURL"
