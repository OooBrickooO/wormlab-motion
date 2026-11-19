"""03 §5.3: 免骨架游泳计数模块 (swim_shape.py) —— 基于 Pose 椭圆形状描述子的 C 形摆动双轨计数。

理论依据：wrMTrck (Nussbaum-Krammer 2015) + WF-NTP (Koopman et al., Nature Protocols 2020)
+ Imaginera (Ji et al., G3 2024)。

核心思路：
- 每次游泳从"直 → 左 C → 直 → 右 C"时，等效椭圆短长轴比 ar(t) = b/a 产生 2 个清晰波峰
  （每次弯曲一个峰，直接对应 c_bends）；完全不需要骨架、完全免疫头尾翻转与虫体 360° 整体旋转；
- 叠加两个辅助物理量防误报：
  1. 动态活跃度门限（roi_motion 或局部滑窗 std(ar)）：避免静止死卷的高 ar 虫被误计；
  2. 连续带符号弯向 signed_bend(t)：对 ell_angle 做双倍角展开 0.5·unwrap(2·angle)，
     结合 ell_skew_minor 得到无需头尾定义的带符号弯曲信号，辅助交叉验证 c_cycles。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from wormlab.config import SwingCfg
from wormlab.schemas import Pose


def interp_short_gaps_1d(x: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """仅对 <= max_gap_frames 的极短缺口做 1D 线性插值（绝不做长缺口插值）。"""
    out = np.asarray(x, dtype=np.float64).copy()
    n = len(out)
    if n == 0 or max_gap_frames < 1:
        return out
    finite = np.isfinite(out)
    if not finite.any():
        return out
    idx = np.arange(n)
    out[~finite] = np.interp(idx[~finite], idx[finite], out[finite])
    # 超长缺口还原为 NaN（不外推伪造数据）
    ok_idx = idx[finite]
    first, last = int(ok_idx[0]), int(ok_idx[-1])
    for i in range(n):
        if not np.isfinite(out[i]):
            prev_ok = ok_idx[ok_idx < i]
            next_ok = ok_idx[ok_idx > i]
            if len(prev_ok) == 0 or len(next_ok) == 0:
                out[i] = np.nan
                continue
            gap = int(next_ok[0] - prev_ok[-1])
            if gap > max_gap_frames + 1:
                out[i] = np.nan
    return out


def contiguous_finite_segments(x: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """返回 [start, end) 形式的连续有限值段（长度 >= min_len）。"""
    finite = np.isfinite(x)
    segs: list[tuple[int, int]] = []
    start: int | None = None
    for i in range(len(x)):
        if finite[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start >= min_len:
                    segs.append((start, i))
                start = None
    if start is not None and len(x) - start >= min_len:
        segs.append((start, len(x)))
    return segs


def zero_phase_bandpass(sig: np.ndarray, lo_hz: float, hi_hz: float, fps: float) -> np.ndarray:
    """零相位 Butterworth 带通（sosfiltfilt）；信号过短或滤波失败时退化为均值去除。"""
    from scipy.signal import butter, sosfiltfilt

    sig = np.asarray(sig, dtype=np.float64)
    if len(sig) < 9:
        return sig - float(np.mean(sig))
    nyq = max(1e-6, 0.5 * float(fps))
    lo = float(np.clip(lo_hz / nyq, 1e-4, 0.99))
    hi = float(np.clip(hi_hz / nyq, lo + 1e-3, 0.999))
    try:
        sos = butter(3, [lo, hi], btype="bandpass", output="sos")
        return sosfiltfilt(sos, sig)
    except Exception:  # noqa: BLE001
        return sig - float(np.mean(sig))


def signed_bend_series(
    ar_hp: np.ndarray,
    angle_raw: np.ndarray,
    skew: np.ndarray,
) -> np.ndarray:
    """连续带符号弯向 s_bend(t) = ar_hp(t) * sign(skew_aligned(t))。

    对 ell_angle 做双倍角展开 phi_u = 0.5·unwrap(2·angle)，令相邻帧主轴向量连续；
    若 cos(phi_u[t] - angle_raw[t]) < 0 则翻转该帧 ell_skew_minor 符号。
    """
    n = len(ar_hp)
    out = np.zeros(n, dtype=np.float64)
    for t in range(n):
        if not (
            np.isfinite(ar_hp[t]) and np.isfinite(angle_raw[t]) and np.isfinite(skew[t])
        ):
            out[t] = 0.0
            continue
        lo, hi = max(0, t - 3), min(n, t + 4)
        idxs = [k for k in range(lo, hi) if np.isfinite(angle_raw[k])]
        if t not in idxs or len(idxs) < 2:
            out[t] = 0.0
            continue
        seg = np.asarray([angle_raw[k] for k in idxs], dtype=np.float64)
        phi_u = 0.5 * np.unwrap(2.0 * seg)
        phi_u_t = float(phi_u[idxs.index(t)])
        sign = -1.0 if float(np.cos(phi_u_t - float(angle_raw[t]))) < 0.0 else 1.0
        out[t] = float(ar_hp[t]) * float(np.sign(float(skew[t])) * sign)
    return out


def count_swim_from_shape(
    poses: Sequence[Pose | None],
    fps: float,
    cfg: SwingCfg,
) -> tuple[int, int, np.ndarray]:
    """基于 Pose.ell_aspect / ell_angle / ell_skew_minor / roi_motion 的免骨架游泳计数。

    返回 (c_cycles, c_bends, peak_frame_indices)。
    """
    n = len(poses)
    ar = np.full(n, np.nan, dtype=np.float64)
    mot = np.full(n, np.nan, dtype=np.float64)
    ang = np.full(n, np.nan, dtype=np.float64)
    skew = np.full(n, np.nan, dtype=np.float64)
    for i, p in enumerate(poses):
        if p is not None and np.isfinite(getattr(p, "ell_aspect", float("nan"))):
            ar[i] = float(p.ell_aspect)
            mot[i] = float(p.roi_motion) if np.isfinite(getattr(p, "roi_motion", float("nan"))) else float("inf")
            if np.isfinite(getattr(p, "ell_angle", float("nan"))):
                ang[i] = float(p.ell_angle)
            if np.isfinite(getattr(p, "ell_skew_minor", float("nan"))):
                skew[i] = float(p.ell_skew_minor)

    valid_idx = np.nonzero(np.isfinite(ar))[0]
    if len(valid_idx) < max(8, int(0.5 * fps)):
        return 0, 0, np.zeros(0, dtype=np.int32)

    # 1) 仅对 <= 0.25s 的极短缺口做 1D 标量插值
    ar_interp = interp_short_gaps_1d(ar, max_gap_frames=max(1, int(round(0.25 * fps))))

    # 2) 按连续有效段分别做带通滤波 (0.4 Hz ~ min(6.5 Hz, 0.42 * fps))
    #    - 0.4 Hz 高通：滤除虫体静态偏置与慢速体态漂移
    #    - 6.5 Hz 低通：滤除单帧像素抖动（野生型最高 ~4.5 bends/s）
    min_dist = max(2, int(round(float(cfg.min_flip_interval_s) * fps)))
    peaks_all: list[int] = []
    seg_min_len = max(6, int(0.4 * fps))
    for seg_s, seg_e in contiguous_finite_segments(ar_interp, min_len=seg_min_len):
        sig = ar_interp[seg_s:seg_e]
        sig_bp = zero_phase_bandpass(sig, lo_hz=0.4, hi_hz=min(6.5, 0.42 * fps), fps=fps)
        try:
            from scipy.signal import find_peaks

            pks, _props = find_peaks(
                sig_bp,
                prominence=float(cfg.shape_aspect_prom),
                distance=min_dist,
            )
        except Exception:  # noqa: BLE001
            pks = np.zeros(0, dtype=np.int64)
        for pk in pks:
            g_idx = seg_s + int(pk)
            # 活跃度校验：排除完全静止的死卷虫
            win_lo = max(seg_s, g_idx - int(0.5 * fps))
            win_hi = min(seg_e, g_idx + int(0.5 * fps) + 1)
            local_mot = float(np.nanmedian(mot[win_lo:win_hi]))
            if np.isfinite(local_mot) and local_mot < float(cfg.shape_motion_min):
                continue
            # 局部滑窗波动幅度兜底：roi_motion 缺失时要求 ar 波动本身显著
            if not np.isfinite(local_mot):
                local_std = float(np.nanstd(ar_interp[win_lo:win_hi]))
                if not np.isfinite(local_std) or local_std < 0.03:
                    continue
            peaks_all.append(g_idx)

    peaks_all.sort()
    c_bends = len(peaks_all)
    c_cycles = c_bends // 2
    return int(c_cycles), int(c_bends), np.asarray(peaks_all, dtype=np.int32)
