"""03 §4: S 形运动单调相位累加、净位移校验与反向运动清零计数器 (SRunCounter)。

严格遵守甲方业务定义：
1. 仅统计向前行波且沿头部方向产生净位移 (>= min_disp_ratio * L_ref) 的完整 2π 周期；
2. 向后完整周期完全不计入 s_count，仅记入诊断字段 rejected['backward']；
3. 一旦本轮未完成周期相位进度回退为负 (< -reverse_phase_eps_rad) 且质心沿体轴实际后退 (>= reverse_disp_ratio * L_ref)，
   立即作废当前未完成的半截周期并重新计数，已完成并计入的 s_count 保留；
4. 仅头部轻微摆动由上游 wave_gate (amp_post/amp_ant) 拦截，不进入本计数器。

08_SPEC S3/S4 修复：
- S3: freeze() 改为逐帧递增 gap 计数；短间隙 (<= phase_bridge_s) 恢复时若方向一致且相位跳变
  < 0.65π，则接续冻结期间真实推进的相位与位移，而非静默丢弃；
- S4: 逐帧沿中段切线积分路径推进位移 path_disp（弧线拐弯不再被首尾直线弦压扁），
  可由 cfg.use_path_disp=False 回退旧版直线弦投影。
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
        self.phase_bridge_frames = max(1, int(round(float(getattr(cfg, "phase_bridge_s", 0.45)) * float(fps))))
        self.use_path_disp = bool(getattr(cfg, "use_path_disp", True))

        self.active: bool = False
        self.direction: int = 0
        self.phi_prev: float | None = None
        self.prog: float = 0.0
        self.anchor_prog: float = 0.0
        self.anchor_center: np.ndarray = np.zeros(2, dtype=np.float64)
        self.axis_sum: np.ndarray = np.zeros(2, dtype=np.float64)
        self.last_center: np.ndarray | None = None
        self.gap: int = 0
        self.path_disp: float = 0.0  # 当前周期内沿体轴切线累计的有符号推进弧长 (px)
        self.prev_center: np.ndarray | None = None

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
        self.path_disp = 0.0
        self.prev_center = self.anchor_center.copy()

    def _abort(self, reason: str) -> None:
        if self.active:
            self.aborted += 1
            self.active = False
            self.direction = 0
            self.prog = 0.0
            self.anchor_prog = 0.0
            self.phi_prev = None
            self.gap = 0
            self.path_disp = 0.0
            self.prev_center = None

    def force_abort(self, reason: str = "forced") -> None:
        """外部强制中止当前未完成周期（如进入深弯 CURL 或连续 BAD 帧超时）。"""
        self._abort(reason)

    def freeze(self) -> None:
        """短暂掉门限或遮挡期间冻结进度而不清零；逐帧递增 gap 以精确度量冻结时长。"""
        if self.active:
            self.gap += 1

    def _disp_eff(self, center: np.ndarray) -> float:
        """当前轮推进位移度量：path_disp（切向路径积分）或旧版直线弦投影。"""
        if self.use_path_disp:
            return float(self.path_disp)
        if self.last_center is None:
            return 0.0
        axis_cur = self._unit(self.axis_sum)
        return float(np.dot(np.asarray(center, dtype=np.float64) - self.anchor_center, axis_cur))

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
        p_cur = self._disp_eff(self.last_center)
        req_disp = float(self.cfg.min_disp_ratio) * float(tail_frac) * float(L_ref)
        if p_cur >= req_disp:
            self.s_count += 1
            self.events.append(SEvent(frame_idx=int(i), direction=+1, disp_px=p_cur, reason="tail_settled"))
            self.anchor_prog += 2.0 * np.pi
            self.anchor_center = self.last_center.copy()
            self.axis_sum = np.zeros(2, dtype=np.float64)
            self.path_disp = 0.0
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
        tangent_mid: np.ndarray | None = None,
    ) -> None:
        if not gate_ok or phi is None or center is None or u is None:
            if self.active:
                self.gap += 1
                if self.gap > self.gap_hold_frames:
                    self._abort("gate_lost")
            return

        c_arr = np.asarray(center, dtype=np.float64)
        t_vec = np.asarray(tangent_mid if tangent_mid is not None else u, dtype=np.float64)
        t_unit = self._unit(t_vec)

        if not self.active or self.phi_prev is None:
            self._start(float(phi), c_arr)
            return

        dphi = float(wrap_pi(float(phi) - self.phi_prev))
        bridged_gap = 1

        if self.gap > 0:
            # 关键修复 S3: 短间隙 (<= phase_bridge_frames) 恢复时，若方向一致且相位跳变合理，
            # 接续这段相位与位移，而非静默丢弃冻结期间真实推进的相位！
            can_bridge = (
                self.gap <= self.phase_bridge_frames
                and abs(dphi) < 0.65 * np.pi
                and (self.direction == 0 or (self.direction * dphi) > 0.0)
            )
            bridged_gap = self.gap + 1
            self.gap = 0
            self.phi_prev = float(phi)
            if not can_bridge:
                # 超长间隙或发生大相位跳变，只重锚不计入，保持 prev_center 更新
                self.prev_center = c_arr.copy()
                self.last_center = c_arr.copy()
                return
            # can_bridge=True：继续落入下方正常累加 dphi 与 path_disp

        else:
            self.phi_prev = float(phi)

        # 关键修复 S4: 逐帧切向位移积分（弧线拐弯不再被直线弦压扁；原地抖动正负抵消依然≈0）
        if self.prev_center is not None:
            step_vec = c_arr - self.prev_center
            # 限制单帧异常跳变（如追踪闪跳 > 0.45 * L_ref；跨隙桥接时按 bridged_gap 适度放宽），防脏帧注入假位移
            max_step = min(1.5, 0.45 * float(bridged_gap)) * max(20.0, float(L_ref))
            if float(np.linalg.norm(step_vec)) <= max_step:
                self.path_disp += float(np.dot(step_vec, t_unit))
        self.prev_center = c_arr.copy()
        self.last_center = c_arr.copy()
        self.axis_sum = self.axis_sum + np.asarray(u, dtype=np.float64)

        if self.direction == 0:
            self.prog += dphi
            if abs(self.prog) >= float(self.cfg.dir_lock_rad):
                self.direction = 1 if self.prog > 0 else -1
                self.prog = abs(self.prog)
                self.anchor_prog = 0.0
            return

        self.prog += float(self.direction) * dphi
        cyc = self.prog - self.anchor_prog

        # 当前轮位移度量选择：path_disp 或 chord_disp
        p_eff = self._disp_eff(c_arr)

        # 反向运动清零规则：本轮相位进度为负 且 质心沿体轴向反方向移动超阈值 -> 作废当前未完成周期
        if (
            cyc < -float(self.cfg.reverse_phase_eps_rad)
            and int(np.sign(p_eff)) == -self.direction
            and abs(p_eff) >= float(self.cfg.reverse_disp_ratio) * float(L_ref)
        ):
            self._abort("reversal")
            self._start(float(phi), c_arr)
            return

        two_pi = 2.0 * np.pi
        while self.prog - self.anchor_prog >= two_pi:
            p_cur = self._disp_eff(c_arr)
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
            self.path_disp = 0.0

    @property
    def s_cycles_float(self) -> float:
        """含当前正向未完成周期小数部分的浮点周期数（供诊断用）。"""
        partial = 0.0
        if self.active and self.direction > 0:
            partial = float(np.clip((self.prog - self.anchor_prog) / (2.0 * np.pi), 0.0, 0.99))
        return float(self.s_count) + partial
