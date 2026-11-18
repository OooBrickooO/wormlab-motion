"""03 §4: S 形运动单调相位累加、净位移校验与反向运动清零计数器 (SRunCounter)。

严格遵守甲方业务定义：
1. 仅统计向前行波且沿头部方向产生净位移 (>= min_disp_ratio * L_ref) 的完整 2π 周期；
2. 向后完整周期完全不计入 s_count，仅记入诊断字段 rejected['backward']；
3. 一旦本轮未完成周期相位进度回退为负 (< -reverse_phase_eps_rad) 且质心沿体轴实际后退 (>= reverse_disp_ratio * L_ref)，
   立即作废当前未完成的半截周期并重新计数，已完成并计入的 s_count 保留；
4. 仅头部轻微摆动由上游 wave_gate (amp_post/amp_ant) 拦截，不进入本计数器。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wormlab.config import SRunCfg
from wormlab.geometry import wrap_pi


@dataclass(frozen=True)
class SEvent:
    """完成或中止的 S 周期事件记录。"""

    frame_idx: int
    direction: int
    disp_px: float
    reason: str = "completed"


class SRunCounter:
    """单调相位 run 追踪与 S 周期计数器。"""

    def __init__(self, cfg: SRunCfg, fps: float) -> None:
        self.cfg = cfg
        self.fps = float(fps)
        self.gap_hold_frames = max(1, int(round(float(cfg.gap_hold_s) * float(fps))))

        self.active: bool = False
        self.direction: int = 0
        self.phi_prev: float | None = None
        self.prog: float = 0.0
        self.anchor_prog: float = 0.0
        self.anchor_center: np.ndarray = np.zeros(2, dtype=np.float64)
        self.axis_sum: np.ndarray = np.zeros(2, dtype=np.float64)
        self.last_center: np.ndarray | None = None
        self.gap: int = 0

        self.s_count: int = 0
        self.rejected: dict[str, int] = {"no_disp": 0, "wrong_dir": 0, "backward": 0}
        self.aborted: int = 0
        self.events: list[SEvent] = []

    def _start(self, phi: float, center: np.ndarray) -> None:
        self.active = True
        self.direction = 0
        self.prog = 0.0
        self.phi_prev = float(phi)
        self.anchor_prog = 0.0
        self.anchor_center = np.asarray(center, dtype=np.float64).copy()
        self.last_center = self.anchor_center.copy()
        self.axis_sum = np.zeros(2, dtype=np.float64)
        self.gap = 0

    def _abort(self, reason: str) -> None:
        if self.active:
            self.aborted += 1
            self.active = False
            self.direction = 0
            self.prog = 0.0
            self.anchor_prog = 0.0
            self.phi_prev = None
            self.gap = 0

    def force_abort(self, reason: str = "forced") -> None:
        """外部强制中止当前未完成周期（如进入深弯 CURL 或连续 BAD 帧超时）。"""
        self._abort(reason)

    def freeze(self) -> None:
        """短暂掉门限或遮挡期间冻结进度而不清零，待恢复后重新对齐相位。"""
        if self.active and self.gap < 1:
            self.gap = 1

    def settle_partial(self, i: int, L_ref: float, tail_frac: float = 0.65) -> bool:
        """
        当一次正向爬行因停顿 (gate_lost) 或视频结束 (EOF) 终止时，
        若当前未完成周期已累积 >= tail_frac * 2π（如 65% 周期）且顺向推进位移按比例达标，
        则结算为 +1 次有效 S 周期，避免 1.7~1.9 周期被硬截断为 1。
        """
        if not self.active or self.direction <= 0 or self.last_center is None:
            return False
        rem_frac = (self.prog - self.anchor_prog) / (2.0 * np.pi)
        if rem_frac < float(tail_frac):
            return False
        axis_cur = self._unit(self.axis_sum)
        p_cur = float(np.dot(self.last_center - self.anchor_center, axis_cur))
        req_disp = float(self.cfg.min_disp_ratio) * float(tail_frac) * float(L_ref)
        if p_cur >= req_disp:
            self.s_count += 1
            self.events.append(SEvent(frame_idx=int(i), direction=+1, disp_px=p_cur, reason="tail_settled"))
            self.anchor_prog += 2.0 * np.pi
            self.anchor_center = self.last_center.copy()
            self.axis_sum = np.zeros(2, dtype=np.float64)
            return True
        return False

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        n = float(np.hypot(v[0], v[1]))
        return v / n if n > 1e-9 else np.array([1.0, 0.0], dtype=np.float64)

    def step(
        self,
        i: int,
        gate_ok: bool,
        phi: float | None,
        center: np.ndarray | None,
        u: np.ndarray | None,
        L_ref: float,
    ) -> None:
        if not gate_ok or phi is None or center is None or u is None:
            if self.active:
                self.gap += 1
                if self.gap > self.gap_hold_frames:
                    self._abort("gate_lost")
            return

        if self.active and self.gap > 0:
            # 门限短暂掉出后恢复：冻结期间的相位跳变不计，重新对齐上一帧相位
            self.phi_prev = float(phi)
            self.gap = 0

        if not self.active or self.phi_prev is None:
            self._start(float(phi), center)
            return

        dphi = float(wrap_pi(float(phi) - self.phi_prev))
        self.phi_prev = float(phi)
        self.axis_sum = self.axis_sum + np.asarray(u, dtype=np.float64)
        self.last_center = np.asarray(center, dtype=np.float64).copy()

        if self.direction == 0:
            self.prog += dphi
            if abs(self.prog) >= float(self.cfg.dir_lock_rad):
                self.direction = 1 if self.prog > 0 else -1
                self.prog = abs(self.prog)
                self.anchor_prog = 0.0
            return

        self.prog += float(self.direction) * dphi
        cyc = self.prog - self.anchor_prog
        axis = self._unit(self.axis_sum)
        c_arr = np.asarray(center, dtype=np.float64)
        p = float(np.dot(c_arr - self.anchor_center, axis))

        # 反向运动清零规则：本轮相位进度为负 且 质心沿体轴向反方向移动超阈值 -> 作废当前未完成周期
        if (
            cyc < -float(self.cfg.reverse_phase_eps_rad)
            and int(np.sign(p)) == -self.direction
            and abs(p) >= float(self.cfg.reverse_disp_ratio) * float(L_ref)
        ):
            self._abort("reversal")
            self._start(float(phi), c_arr)
            return

        two_pi = 2.0 * np.pi
        while self.prog - self.anchor_prog >= two_pi:
            axis_cur = self._unit(self.axis_sum)
            p_cur = float(np.dot(c_arr - self.anchor_center, axis_cur))
            ok_dir = int(np.sign(p_cur)) == self.direction
            ok_disp = abs(p_cur) >= float(self.cfg.min_disp_ratio) * float(L_ref)

            if ok_dir and ok_disp and self.direction > 0:
                self.s_count += 1
                self.events.append(SEvent(frame_idx=int(i), direction=+1, disp_px=p_cur, reason="completed"))
            elif ok_dir and ok_disp and self.direction < 0:
                self.rejected["backward"] += 1
            else:
                key = "wrong_dir" if not ok_dir else "no_disp"
                self.rejected[key] += 1

            self.anchor_prog += two_pi
            self.anchor_center = c_arr.copy()
            self.axis_sum = np.zeros(2, dtype=np.float64)

    @property
    def s_cycles_float(self) -> float:
        """含当前正向未完成周期小数部分的浮点周期数（供诊断用）。"""
        partial = 0.0
        if self.active and self.direction > 0:
            partial = float(np.clip((self.prog - self.anchor_prog) / (2.0 * np.pi), 0.0, 0.99))
        return float(self.s_count) + partial
