"""03 §5: 游泳摆动迟滞过零计数器 (SwingCounter) —— 同时输出完整周期 (c_cycles) 与单次弯曲 (c_bends)。

Mode C1 增强：
1. 头尾 180° 翻转免疫：若传入当前帧首尾单位向量 u_curr，且其与平滑参考方向 u_ref 点积 < -0.15
   （说明 orient() 在该帧翻转了头尾），则自动对 c 取反，杜绝单帧头尾翻转引发的假计数；
2. 双模式触发：保留 ±bend_high 对称迟滞翻转，同时增加峰谷相对突出度 (prominence) 检测，
   使存在姿态偏置（如 +1.2 ↔ −0.3）的摆动在回摆幅度足够且跨过近期中位基线时同样计数。
"""
from __future__ import annotations

import numpy as np

from wormlab.config import SwingCfg


class SwingCounter:
    """对总转角 c(t) 做迟滞双侧高位计数，同时维护 c_bends 与 c_cycles。"""

    def __init__(self, cfg: SwingCfg, fps: float, gap_hold_s: float = 0.30) -> None:
        self.cfg = cfg
        self.fps = float(fps)
        self.min_flip_frames = max(1, int(round(float(cfg.min_flip_interval_s) * float(fps))))
        self.gap_hold_frames = max(1, int(round(float(gap_hold_s) * float(fps))))
        self.bias_win_frames = max(10, int(round(2.0 * float(fps))))

        self.side: int = 0
        self.entries: int = 0
        self.flips: int = 0
        self.last_flip: int = -(10**9)
        self.last_high_frame: int = -(10**9)
        self.invalid_run: int = 0
        self.extreme: float = 0.0
        self.u_ref: np.ndarray | None = None
        self._recent_c: list[float] = []

    def _fix_head_tail_flip(self, c_val: float, u_curr: np.ndarray | None) -> float:
        if u_curr is None:
            return c_val
        u_arr = np.asarray(u_curr, dtype=np.float64).reshape(-1)
        if u_arr.shape[0] != 2 or not np.all(np.isfinite(u_arr)):
            return c_val
        u_norm = float(np.linalg.norm(u_arr))
        if u_norm <= 1e-6:
            return c_val
        u_unit = u_arr / u_norm
        if self.u_ref is None:
            self.u_ref = u_unit.copy()
            return c_val
        if float(np.dot(u_unit, self.u_ref)) < -0.15:
            c_val = -c_val
            u_unit = -u_unit
        ref = 0.85 * self.u_ref + 0.15 * u_unit
        self.u_ref = ref / max(float(np.linalg.norm(ref)), 1e-9)
        return c_val

    def step(self, i: int, c: float, valid: bool, u_curr: np.ndarray | None = None) -> None:
        if not valid or not np.isfinite(c):
            self.invalid_run += 1
            if self.invalid_run > self.gap_hold_frames:
                self.side = 0
                self.u_ref = None
            return
        self.invalid_run = 0

        c_val = self._fix_head_tail_flip(float(c), u_curr)

        self._recent_c.append(c_val)
        if len(self._recent_c) > self.bias_win_frames:
            self._recent_c.pop(0)
        c_bias = float(np.median(self._recent_c)) if len(self._recent_c) >= 10 else 0.0

        thr = float(self.cfg.bend_high)
        prom_thr = 1.55 * thr  # 峰谷突出度阈值

        # 标准对称迟滞判定
        new_side = 1 if c_val >= thr else (-1 if c_val <= -thr else 0)
        if new_side != 0:
            self.last_high_frame = int(i)

        # 追踪当前侧极值，支持偏置姿态下的峰谷突出度翻转
        if self.side == 1:
            self.extreme = max(self.extreme, c_val)
            if new_side == 0 and (self.extreme - c_val) >= prom_thr and (c_val - c_bias) <= -0.35 * thr:
                new_side = -1
        elif self.side == -1:
            self.extreme = min(self.extreme, c_val)
            if new_side == 0 and (c_val - self.extreme) >= prom_thr and (c_val - c_bias) >= 0.35 * thr:
                new_side = 1

        if new_side == 0 or new_side == self.side:
            return

        if self.side != 0:
            if (int(i) - self.last_flip) < self.min_flip_frames:
                return
            self.flips += 1
            self.last_flip = int(i)

        self.entries += 1
        self.side = new_side
        self.extreme = c_val

    @property
    def bends(self) -> int:
        """单次弯曲次数 (c_bends)：左弯、右弯每次进入高位各算 1 次。"""
        return int(self.entries)

    @property
    def cycles(self) -> int:
        """完整往返周期数 (c_cycles)：左弯 + 右弯 = 1 次完整来回。"""
        return int(self.entries // 2)
