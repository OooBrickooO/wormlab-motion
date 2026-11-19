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


def _tremor_vote(denses: list) -> int:
    """头部高频觅食微颤先验投票（Tierpsy 多线索思路）。

    对前后各 15% 躯干切线角做高频时间差分方差比较：var(diff(theta[:, :k])) vs var(diff(theta[:, -k:]))。
    返回 +1（dense[0] 侧更像觅食头部）、-1（反侧更像头部）或 0（无法判定）。
    """
    th_rows: list[np.ndarray] = []
    for d in denses:
        arr = np.asarray(d, dtype=np.float64)
        if arr.ndim != 2 or len(arr) < 8:
            return 0
        diffs = np.diff(arr, axis=0)
        th = np.unwrap(np.arctan2(diffs[:, 1], diffs[:, 0]))
        th_rows.append(th - float(np.median(th)))
    if len(th_rows) < 3:
        return 0
    m = len(th_rows[0])
    if any(len(r) != m for r in th_rows):
        return 0
    th_mat = np.stack(th_rows, axis=0)
    dth = np.diff(th_mat, axis=0)
    if dth.shape[0] < 2:
        return 0
    k = max(1, int(round(0.15 * m)))
    v_front = float(np.var(dth[:, :k]))
    v_rear = float(np.var(dth[:, -k:]))
    denom = max(1e-9, max(v_front, v_rear))
    if denom <= 1e-9:
        return 0
    ratio = v_front / max(1e-9, v_rear)
    if 0.8 <= ratio <= 1.25:
        return 0
    return 1 if v_front > v_rear else -1


def reorient_track_poses_coil_aware(
    poses: Sequence[Any],
    fps: float,
    cfg: Any,
    margin: float = 0.9,
    coil_ratio: float = 0.45,
    long_gap_s: float = 1.5,
    preserve_first_head: bool = False,
) -> list[Any]:
    """按深卷曲 (Omega)、长缺失间隔或单帧方向突跳切分非卷曲片段，逐段重排并利用
    行进波/质心推进 + 全局主导方向 + 头部微颤先验自动纠偏头尾（08_SPEC S6）。

    - 真实短暂倒车 (< 3s) 不再被孤立按 score < 0 强翻成假前进；
    - 浅卷 (R ∈ [0.45, 0.65]) 翻头由单帧方向突跳切点兜底切段；
    - 未人工标头时由"头部觅食微颤先验 + 全轨迹主导推进方向"联合投票确定全局正方向。
    """
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

    # 单帧方向突跳切点：连续两有效帧间体轴点积 < -0.5（>120° 无过渡突转）视为潜在翻头点
    for i in range(1, n):
        p0, p1 = poses[i - 1], poses[i]
        if (
            p0 is None
            or p1 is None
            or getattr(p0, "dense", None) is None
            or getattr(p1, "dense", None) is None
            or len(p0.dense) < 2
            or len(p1.dense) < 2
        ):
            continue
        u0 = np.asarray(p0.dense[0], dtype=np.float64) - np.asarray(p0.dense[-1], dtype=np.float64)
        u1 = np.asarray(p1.dense[0], dtype=np.float64) - np.asarray(p1.dense[-1], dtype=np.float64)
        n0 = float(np.linalg.norm(u0))
        n1 = float(np.linalg.norm(u1))
        if n0 > 1e-6 and n1 > 1e-6 and float(np.dot(u0 / n0, u1 / n1)) < -0.5:
            is_break[i] = True

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

    seg_infos: list[dict[str, Any]] = []
    for si, idxs in enumerate(segs):
        s_start = 0 if si == 0 else (segs[si - 1][-1] + idxs[0]) // 2 + 1
        s_end = n if si == len(segs) - 1 else (idxs[-1] + segs[si + 1][0]) // 2 + 1
        sub_re = reorient_track_poses(list(poses[s_start:s_end]), margin=margin)
        locked_head = bool(preserve_first_head and si == 0)
        rel_local = [
            i - s_start
            for i in idxs
            if sub_re[i - s_start] is not None and sub_re[i - s_start].quality != Quality.BAD
        ]
        score = 0.0
        dur_s = float(max(1, idxs[-1] - idxs[0] + 1)) / max(1.0, float(fps))
        tremor = 0
        eng_s = None
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
            tremor = _tremor_vote([sub_re[k].dense for k in rel_local if sub_re[k].dense is not None])

        seg_infos.append(
            {
                "si": si,
                "s_start": s_start,
                "sub_re": sub_re,
                "rel_local": rel_local,
                "score": score,
                "dur_s": dur_s,
                "tremor": tremor,
                "locked": locked_head,
                "eng": eng_s,
            }
        )

    # 全轨迹主导方向联合投票：运动学得分（按时长加权） + 头部微颤先验（半权）
    motion_vote = float(sum(info["score"] * info["dur_s"] for info in seg_infos))
    tremor_vote = float(sum(info["tremor"] * info["dur_s"] for info in seg_infos))
    global_reversed = (motion_vote + 0.5 * tremor_vote) < 0.0

    for info in seg_infos:
        si = info["si"]
        score = float(info["score"])
        dur_s = float(info["dur_s"])
        sub_re = info["sub_re"]
        do_flip = False
        if info["locked"]:
            # preserve_first_head：首段默认信任人工/首帧头标；但若首段呈现典型"正向爬行但头尾标反"
            # （无任何计入的 S 周期、存在 backward 拒绝且综合得分显著为负），仍允许自动翻转纠正
            eng_s = info["eng"]
            do_flip = bool(
                eng_s is not None
                and eng_s.s.s_count == 0
                and int(eng_s.s.rejected.get("backward", 0)) >= 1
                and score < -0.25
            )
        elif preserve_first_head:
            # 跨卷曲恢复段：区分"短时真实倒车"与"整段头尾装反"（08_SPEC S6）
            # - 持续 >= 3.0s 且 score < -0.25（线虫罕见持续倒车超 3s）-> 翻转；
            # - 全轨迹绝大多数帧一致反向 -> 翻转；
            # - 短暂倒车逃逸（< 3s）参考全局/后继方向，不翻转。
            do_flip = bool((dur_s >= 3.0 and score < -0.25) or (global_reversed and score < 0.0))
        else:
            if dur_s >= 2.0:
                # 长段：自身运动学得分足够可靠，沿用 score < 0 判据
                do_flip = bool(score < 0.0)
            else:
                # 短段：不再孤立按 score < 0 乱翻，跟随全局主导方向（含微颤先验）投票
                do_flip = bool(global_reversed)

        if do_flip:
            sub_re = [
                None if (p is None or getattr(p, "dense", None) is None) else replace(p, dense=p.dense[::-1].copy())
                for p in sub_re
            ]

        s_start = info["s_start"]
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
