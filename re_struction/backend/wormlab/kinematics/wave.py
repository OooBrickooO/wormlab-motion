"""03 §3: 行波门限判别器 (wave_gate) —— 区分全身行波与驻波/原地抖动/仅头部试探。

08_SPEC S2 升级：除原相平面圆度 ε 判据外，新增瞬时角动量 L_z = a1·ȧ2 − a2·ȧ1 的
方向一致性 C_wave = |ΣLz| / Σ|Lz| 与加权角速度 ω̄ = ΣLz / Σr² 双判据：
- 驻波 / 原地线性抖动：a1, a2 共线，L_z ≡ 0；
- 行波（含慢速 0.15 Hz 或刚起步 0.5s 短爆发）：L_z 瞬时恒同号，短窗内一致性 ≈ 0.8–1.0。
gate_mode: "eps"（旧版兼容默认）| "angular_momentum" | "hybrid"。
"""
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
    coherence: float = 0.0  # 角动量方向一致性 |ΣLz| / Σ|Lz| ∈ [0, 1]
    mean_omega: float = 0.0  # r² 加权平均角速度 ΣLz / Σr² (rad/s，带符号)

    @staticmethod
    def off() -> GateResult:
        return GateResult(ok=False, eps=0.0, r_med=0.0, amp_ant=0.0, amp_post=0.0)


def _angular_momentum_stats(window_feats: list[Feat | None], fps: float) -> tuple[float, float]:
    """按原始时间索引计算角动量统计 (coherence, mean_omega)；尊重中间缺失间隔 dt_k。"""
    valid_pairs = [(idx, f) for idx, f in enumerate(window_feats) if f is not None]
    lz_list: list[float] = []
    r2_list: list[float] = []
    for (i0, f0), (i1, f1) in zip(valid_pairs[:-1], valid_pairs[1:]):
        dt = float(i1 - i0) / max(1e-6, float(fps))
        if dt <= 0.0 or dt > 0.35:
            continue
        a1_m = 0.5 * (float(f0.a1) + float(f1.a1))
        a2_m = 0.5 * (float(f0.a2) + float(f1.a2))
        da1 = (float(f1.a1) - float(f0.a1)) / dt
        da2 = (float(f1.a2) - float(f0.a2)) / dt
        lz_list.append(a1_m * da2 - a2_m * da1)
        r2_list.append(a1_m * a1_m + a2_m * a2_m)

    if not lz_list:
        return 0.0, 0.0
    lz_arr = np.asarray(lz_list, dtype=np.float64)
    r2_arr = np.asarray(r2_list, dtype=np.float64)
    coherence = float(abs(np.sum(lz_arr)) / (np.sum(np.abs(lz_arr)) + 1e-9))
    mean_omega = float(np.sum(lz_arr) / (np.sum(r2_arr) + 1e-9))
    return coherence, mean_omega


def wave_gate(
    window_feats: list[Feat | None],
    cfg: WaveCfg,
    expected_len: int | None = None,
    fps: float = 30.0,
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
    r_all = np.hypot(A[:, 0], A[:, 1])
    # 兼顾中位数与窗口内前 35% 活跃帧波幅，防止短促 S 波进出滑窗时被前后静止帧稀释掐断
    r_med = max(float(np.median(r_all)), 0.85 * float(np.percentile(r_all, 65)))

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

    base_amp_ok = bool(
        r_med >= float(cfg.r_min)
        and amp_post >= float(cfg.post_ratio_min) * amp_ant
        and amp_post >= float(cfg.amp_post_min)
    )

    coherence, mean_omega = _angular_momentum_stats(window_feats, fps)

    mode = str(getattr(cfg, "gate_mode", "eps") or "eps")
    if mode == "angular_momentum":
        travel_ok = coherence >= float(cfg.coherence_min) and abs(mean_omega) >= float(cfg.omega_min_rad_s)
    elif mode == "hybrid":
        travel_ok = eps >= float(cfg.eps_min) or (
            coherence >= float(cfg.coherence_min) and abs(mean_omega) >= float(cfg.omega_min_rad_s)
        )
    else:  # "eps"（默认，向后兼容）
        travel_ok = eps >= float(cfg.eps_min)

    return GateResult(
        ok=bool(base_amp_ok and travel_ok),
        eps=eps,
        r_med=r_med,
        amp_ant=amp_ant,
        amp_post=amp_post,
        coherence=coherence,
        mean_omega=mean_omega,
    )
