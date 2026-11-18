"""03 §2–§3: 行波三角模态投影、总转角 c(t) 与相位计算。"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=64)
def _trig_basis(m: int, k_body: float) -> tuple[np.ndarray, np.ndarray]:
    """生成长度 m 的正余弦投影基；非整数波数 k_body 做去均值与 Gram-Schmidt 正交化以防静态弯折 DC 泄漏。"""
    s_mid = (np.arange(m, dtype=np.float64) + 0.5) / float(m)
    arg = 2.0 * np.pi * float(k_body) * s_mid
    c = np.cos(arg)
    sn = np.sin(arg)
    if abs(k_body - round(k_body)) <= 1e-6:
        scale = 2.0 / float(m)
        return c * scale, sn * scale

    c = c - np.mean(c)
    sn = sn - np.mean(sn)
    c_norm = max(1e-12, float(np.linalg.norm(c)))
    c = c / c_norm
    sn = sn - float(np.dot(sn, c)) * c
    sn_norm = max(1e-12, float(np.linalg.norm(sn)))
    sn = sn / sn_norm
    scale = math.sqrt(2.0 / float(m))
    return c * scale, sn * scale


def project_trig_modes(th: np.ndarray, k_body: float = 1.0) -> tuple[float, float]:
    """将去均值切线角序列 th (M,) 投影至一阶正余弦行波基，返回 (a1, a2)。"""
    m = len(th)
    if m == 0:
        return 0.0, 0.0
    c_basis, s_basis = _trig_basis(m, float(k_body))
    a1 = float(np.dot(th, c_basis))
    a2 = float(np.dot(th, s_basis))
    return a1, a2


def compute_total_turn_c(theta: np.ndarray) -> float:
    """计算鲁棒 C 弯曲模态 c(t) (rad)。

    结合全躯干 cos(πs) 对称模态（等价于对曲率按 sin(πs) 加权积分，中心权重最大、两端为 0）
    与去尖端（避开首尾约 6% 端帽噪声点）的首尾 20% 段切线角差，消除掩码两端圆帽毛刺。
    均匀圆弧下两项与原 "尾 20% − 头 20%" 定义同号同量纲。
    """
    th_arr = np.asarray(theta, dtype=np.float64)
    n = len(th_arr)
    if n < 2:
        return 0.0
    if n < 4:
        return float(th_arr[-1] - th_arr[0])
    th_zero = th_arr - float(np.mean(th_arr))
    s_mid = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    # 均匀圆弧 θ = Δθ·s 时，−1.25·π·mean(θ̃·cos(πs)) = 2.5Δθ/π ≈ 0.80·Δθ，与原 0.80·Δθ 量纲严格 1:1 对齐
    c_cos = float(-1.25 * np.pi * np.mean(th_zero * np.cos(np.pi * s_mid)))

    pad = max(1, int(round(0.06 * n)))
    m = max(1, int(round(0.20 * n)))
    head_seg = th_arr[pad : pad + m]
    tail_seg = th_arr[max(pad, n - pad - m) : n - pad]
    c_end = float(np.mean(tail_seg) - np.mean(head_seg))
    return 0.65 * c_cos + 0.35 * c_end


def modal_phase(a1: float, a2: float) -> float:
    """返回行波模态相位 Φ = atan2(a2, a1) ∈ [-π, π]。"""
    return float(math.atan2(float(a2), float(a1)))


def eigenworm_basis(*args: object, **kwargs: object) -> np.ndarray:
    """数据驱动 eigenworm 正交基（D4' 预留接口）。"""
    raise NotImplementedError("eigenworm_basis 为 D4' 预留接口，当前固定使用三角基。")
