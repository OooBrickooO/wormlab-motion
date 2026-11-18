"""02 §5: 稠密中心线头尾朝向对齐（用户首帧头点、帧间连续性与卷曲段自适应消歧）。"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Sequence

import numpy as np

from wormlab.geometry import wrap_pi


def orient(
    dense: np.ndarray,
    prev_dense: np.ndarray | None = None,
    user_head_xy: tuple[float, float] | None = None,
    margin: float = 0.9,
) -> np.ndarray:
    """对齐中心线方向使 dense[0] 始终为头部。

    优先级：
    1. 若提供 user_head_xy（首帧人工头标记），令距离 user_head_xy 更近的端点作为 dense[0]。
    2. 若提供上一有效帧 prev_dense，比较正反序列逐点平均欧氏距离，满足 d_rev < margin * d_fwd 时翻转。
    3. 否则保持原样返回。
    """
    arr = np.asarray(dense, dtype=np.float32)
    if len(arr) < 2:
        return arr

    if user_head_xy is not None:
        head_pt = np.array([float(user_head_xy[0]), float(user_head_xy[1])], dtype=np.float32)
        d0 = float(np.linalg.norm(arr[0] - head_pt))
        d1 = float(np.linalg.norm(arr[-1] - head_pt))
        if d1 < d0:
            return np.ascontiguousarray(arr[::-1])
        return arr

    if prev_dense is not None and prev_dense.shape == arr.shape:
        # 去质心后对齐，避免虫体跨越帧间位移/遮挡间隙后头尾反转
        arr_c = arr - np.mean(arr, axis=0, keepdims=True)
        prev_c = prev_dense - np.mean(prev_dense, axis=0, keepdims=True)
        d_f = float(np.mean(np.linalg.norm(arr_c - prev_c, axis=1)))
        arr_rev = arr[::-1]
        arr_rev_c = arr_rev - np.mean(arr_rev, axis=0, keepdims=True)
        d_r = float(np.mean(np.linalg.norm(arr_rev_c - prev_c, axis=1)))
        if d_r < float(margin) * d_f:
            return np.ascontiguousarray(arr_rev)

    return arr


def reorient_track_poses(poses: list, margin: float = 0.9) -> list:
    """对单条轨迹按时间顺序重新校验帧间头尾连续性（跳过无效/BAD 帧）。"""
    from dataclasses import replace

    from wormlab.schemas import Quality

    out: list = []
    prev: np.ndarray | None = None
    for p in poses:
        if p is None or getattr(p, "dense", None) is None or getattr(p, "quality", None) == Quality.BAD:
            out.append(p)
            continue
        aligned = orient(p.dense, prev_dense=prev, margin=margin)
        prev = aligned
        out.append(replace(p, dense=aligned))
    return out


def reorient_track_poses_coil_aware(
    poses: Sequence[Any],
    fps: float,
    cfg: Any,
    margin: float = 0.9,
    coil_ratio: float = 0.45,
    long_gap_s: float = 1.5,
    preserve_first_head: bool = False,
) -> list[Any]:
    """按深卷曲 (Omega) 或长缺失间隔切分非卷曲片段，逐段重排并利用行进波/质心推进自动纠偏头尾。"""
    from wormlab.kinematics.counter import KinematicsEngine
    from wormlab.kinematics.modes import modal_phase, project_trig_modes
    from wormlab.schemas import Quality

    n = len(poses)
    if n == 0:
        return list(poses)

    long_gap = max(10, int(round(float(long_gap_s) * max(1.0, float(fps)))))
    is_break = [False] * n
    none_run = 0
    coil_run = 0

    for i, p in enumerate(poses):
        if p is None or getattr(p, "dense", None) is None or len(p.dense) < 2:
            none_run += 1
            coil_run = 0
            if none_run >= long_gap:
                for k in range(i - none_run + 1, i + 1):
                    is_break[k] = True
        else:
            none_run = 0
            ratio = float(p.chord) / max(1e-6, float(p.length))
            if ratio < float(coil_ratio):
                coil_run += 1
                if coil_run >= 2:
                    is_break[i - 1] = True
                    is_break[i] = True
            else:
                coil_run = 0

    segs: list[list[int]] = []
    cur: list[int] = []
    for i in range(n):
        if not is_break[i] and poses[i] is not None:
            cur.append(i)
        elif is_break[i] and cur:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)

    if not segs:
        return reorient_track_poses(list(poses), margin=margin)

    out: list[Any] = list(poses)
    s_k = float(getattr(getattr(cfg, "s_run", None), "k_body", 1.15))

    for si, idxs in enumerate(segs):
        s_start = 0 if si == 0 else (segs[si - 1][-1] + idxs[0]) // 2 + 1
        s_end = n if si == len(segs) - 1 else (idxs[-1] + segs[si + 1][0]) // 2 + 1
        sub_re = reorient_track_poses(list(poses[s_start:s_end]), margin=margin)
        # preserve_first_head：首段默认信任人工/首帧头标；但若首段呈现典型"正向爬行但头尾标反"
        # （无任何计入的 S 周期、存在 backward 拒绝且综合得分显著为负），仍允许自动翻转纠正
        locked_head = bool(preserve_first_head and si == 0)
        rel_local = [
            i - s_start
            for i in idxs
            if sub_re[i - s_start] is not None and sub_re[i - s_start].quality != Quality.BAD
        ]
        if len(rel_local) >= 4:
            lengths = [float(sub_re[k].length) for k in rel_local if float(sub_re[k].length) > 1.0]
            l_med = float(np.median(lengths)) if lengths else 100.0
            net_fwd = 0.0
            for j in range(1, len(rel_local)):
                p0, p1 = sub_re[rel_local[j - 1]], sub_re[rel_local[j]]
                u0 = p0.dense[0] - p0.dense[-1]
                un = float(np.linalg.norm(u0))
                if un > 1e-6:
                    net_fwd += float(np.dot(p1.dense.mean(axis=0) - p0.dense.mean(axis=0), u0 / un))

            eng_s = KinematicsEngine(cfg=cfg, fps=fps)
            for k in rel_local:
                eng_s.push(k, sub_re[k])
            eng_s.finalize()

            dphi_tot = 0.0
            prev_ph: float | None = None
            for feat in eng_s.feats:
                if feat is None:
                    prev_ph = None
                    continue
                ph = modal_phase(*project_trig_modes(feat.th, k_body=s_k))
                if prev_ph is not None:
                    dphi_tot += float(wrap_pi(ph - prev_ph))
                prev_ph = ph

            score = (net_fwd / max(1.0, l_med)) + 0.35 * (dphi_tot / (2.0 * math.pi))
            if locked_head:
                do_flip = bool(
                    eng_s.s.s_count == 0 and int(eng_s.s.rejected.get("backward", 0)) >= 1 and score < -0.25
                )
            else:
                do_flip = score < 0.0
            if do_flip:
                sub_re = [
                    None if (p is None or getattr(p, "dense", None) is None) else replace(p, dense=p.dense[::-1].copy())
                    for p in sub_re
                ]

        for k, p in enumerate(sub_re):
            out[s_start + k] = p
    return out


def bridge_short_pose_gaps(
    poses: Sequence[Any],
    fps: float,
    max_gap_s: float = 0.50,
) -> list[Any]:
    """对不超过 max_gap_s 的短时检测/姿态缺口按端序对齐后线性插值中心线，维持跨缺口相位连续。"""
    from wormlab.schemas import Quality

    max_bridge = int(round(float(max_gap_s) * max(1.0, float(fps))))
    out = list(poses)
    if max_bridge <= 0:
        return out

    last_ok: int | None = None
    for i, p in enumerate(out):
        if p is not None and getattr(p, "dense", None) is not None and getattr(p, "quality", None) != Quality.BAD:
            if last_ok is not None and 1 < (i - last_ok) <= max_bridge + 1:
                p0 = out[last_ok]
                span = i - last_ok
                d0 = np.asarray(p0.dense, dtype=np.float32)
                d1 = np.asarray(p.dense, dtype=np.float32)
                if float(np.dot(d0[0] - d0[-1], d1[0] - d1[-1])) < 0.0:
                    d1 = d1[::-1]
                for k in range(last_ok + 1, i):
                    alpha = float(k - last_ok) / float(span)
                    dk = ((1.0 - alpha) * d0 + alpha * d1).astype(np.float32)
                    chord = float(np.linalg.norm(dk[0] - dk[-1])) if len(dk) >= 2 else 0.0
                    out[k] = replace(p0, dense=dk, chord=chord, quality=Quality.SUSPECT)
            last_ok = i
    return out
