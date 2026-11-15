"""03 §3: 行波门限判别器 (wave_gate) —— 区分全身行波与驻波/原地抖动/仅头部试探。"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wormlab.config import WaveCfg

from .features import Feat


@dataclass(frozen=True)
class GateResult:
    """滑窗行波门限评估结果。"""

    ok: bool
    eps: float  # 相平面椭圆度/圆度 ε = sqrt(λ_min / λ_max) ∈ [0, 1]
    r_med: float  # 模态幅度中位数 median(hypot(a1, a2))
    amp_ant: float  # 前半身切线角时间振荡幅度 (rad)
    amp_post: float  # 后半身切线角时间振荡幅度 (rad)

    @staticmethod
    def off() -> GateResult:
        return GateResult(ok=False, eps=0.0, r_med=0.0, amp_ant=0.0, amp_post=0.0)


def wave_gate(
    window_feats: list[Feat | None],
    cfg: WaveCfg,
    expected_len: int | None = None,
) -> GateResult:
    """在滑窗特征列表上判定是否满足全身行波门限。

    关键设计（已在数学原型中验证）：
    计算前后半身振幅比时，每帧扣除沿体轴 s 的中位数 `np.median(f.theta)` 而非均值，
    防止仅头部试探摆动 (s < 0.25) 通过均值减法泄漏污染后半身。
    """
    valid_feats = [f for f in window_feats if f is not None]
    total_slots = expected_len if expected_len is not None else len(window_feats)
    min_needed = max(4, int(math.ceil(float(cfg.min_valid_frac) * float(max(1, total_slots)))))
    if len(valid_feats) < min_needed:
        return GateResult.off()

    A = np.array([[f.a1, f.a2] for f in valid_feats], dtype=np.float64)
    r_med = float(np.median(np.hypot(A[:, 0], A[:, 1])))

    cov = np.cov(A.T)
    ev = np.linalg.eigvalsh(cov)
    l_min = max(0.0, float(ev[0]))
    l_max = max(1e-12, float(ev[1]))
    eps = float(math.sqrt(l_min / l_max))

    # 每帧减沿 s 的中位数，再减窗口时间均值
    th_mat = np.stack([f.theta.astype(np.float64) - float(np.median(f.theta)) for f in valid_feats], axis=0)
    th_mat = th_mat - th_mat.mean(axis=0, keepdims=True)
    half_seg = th_mat.shape[1] // 2
    amp_ant = float(math.sqrt(float(np.mean(np.var(th_mat[:, :half_seg], axis=0)))))
    amp_post = float(math.sqrt(float(np.mean(np.var(th_mat[:, half_seg:], axis=0)))))

    ok = bool(
        r_med >= float(cfg.r_min)
        and eps >= float(cfg.eps_min)
        and amp_post >= float(cfg.post_ratio_min) * amp_ant
        and amp_post >= float(cfg.amp_post_min)
    )
    return GateResult(ok=ok, eps=eps, r_med=r_med, amp_ant=amp_ant, amp_post=amp_post)
