"""04 §8: 离线轨迹片段缝合与交叉 ID 交换修正 (stitch.py，默认关闭)。"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from wormlab.config import TrackingCfg


def _row_center(row: dict[str, Any]) -> np.ndarray:
    if "centroid" in row and row["centroid"] is not None:
        return np.asarray(row["centroid"], dtype=np.float64)[:2]
    b = np.asarray(row["bbox"], dtype=np.float64)[:4]
    return np.array([0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])], dtype=np.float64)


def _row_length(row: dict[str, Any]) -> float:
    if "length" in row and row["length"] is not None and float(row["length"]) > 0.0:
        return float(row["length"])
    b = np.asarray(row["bbox"], dtype=np.float64)[:4]
    return float(math.hypot(b[2] - b[0], b[3] - b[1]))


def _connection_cost(
    pre_rows: list[dict[str, Any]],
    post_rows: list[dict[str, Any]],
    t_c: int,
) -> float:
    """计算将 pre_rows (t <= t_c) 连接到 post_rows (t > t_c) 的位置外推残差 + 体长差 + 特征差代价。"""
    if len(pre_rows) < 2 or len(post_rows) < 1:
        return 1e6

    r_prev1 = pre_rows[-2]
    r_prev2 = pre_rows[-1]
    dt_pre = max(1, int(r_prev2["frame_idx"]) - int(r_prev1["frame_idx"]))
    c1 = _row_center(r_prev1)
    c2 = _row_center(r_prev2)
    vel = (c2 - c1) / float(dt_pre)

    l_pre = float(np.median([_row_length(r) for r in pre_rows[-5:]]))
    l_post = float(np.median([_row_length(r) for r in post_rows[:5]]))
    scale = max(20.0, 0.5 * (l_pre + l_post))

    pos_residuals: list[float] = []
    for r_post in post_rows[:5]:
        dt_post = max(1, int(r_post["frame_idx"]) - int(r_prev2["frame_idx"]))
        c_extrap = c2 + vel * float(dt_post)
        c_obs = _row_center(r_post)
        pos_residuals.append(float(np.linalg.norm(c_obs - c_extrap)) / scale)

    pos_cost = float(np.mean(pos_residuals))
    len_cost = abs(l_post - l_pre) / scale

    feat_cost = 0.0
    if "feature" in r_prev2 and "feature" in post_rows[0] and r_prev2["feature"] is not None and post_rows[0]["feature"] is not None:
        f1 = np.asarray(r_prev2["feature"], dtype=np.float64)
        f2 = np.asarray(post_rows[0]["feature"], dtype=np.float64)
        feat_cost = float(np.linalg.norm(f1 - f2)) * 0.2

    return pos_cost + 0.5 * len_cost + feat_cost


def stitch(
    rows_by_track: dict[int, list[dict[str, Any]]],
    cfg: TrackingCfg,
    force: bool = False,
) -> dict[int, int]:
    """检测两条轨迹接触点 t_c 后的 ID 交换并在原地修正后续片段，返回受影响轨迹的最新映射 {old_id: new_id}。

    默认在 cfg.stitch.enabled == False 且 force == False 时直接返回恒等映射。
    """
    tids = sorted(rows_by_track.keys())
    remap: dict[int, int] = {tid: tid for tid in tids}
    if not force and not bool(cfg.stitch.enabled):
        return remap

    if len(tids) < 2:
        return remap

    # 按 frame_idx 排序每条轨迹
    for tid in tids:
        rows_by_track[tid] = sorted(rows_by_track[tid], key=lambda r: int(r["frame_idx"]))

    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            tid_a, tid_b = tids[i], tids[j]
            rows_a = rows_by_track[tid_a]
            rows_b = rows_by_track[tid_b]
            if len(rows_a) < 6 or len(rows_b) < 6:
                continue

            # 若两条轨迹均带有首帧用户标签锚点，禁止互相替换身份
            has_anchor_a = any(r.get("user_label") is not None or r.get("head_xy") is not None for r in rows_a)
            has_anchor_b = any(r.get("user_label") is not None or r.get("head_xy") is not None for r in rows_b)
            if has_anchor_a and has_anchor_b:
                continue

            by_frame_a = {int(r["frame_idx"]): r for r in rows_a}
            by_frame_b = {int(r["frame_idx"]): r for r in rows_b}
            common_frames = sorted(set(by_frame_a.keys()) & set(by_frame_b.keys()))
            if not common_frames:
                continue

            # 寻找最近接触帧 t_c
            best_tc: int | None = None
            min_dist = 1e9
            for f in common_frames:
                ra, rb = by_frame_a[f], by_frame_b[f]
                d = float(np.linalg.norm(_row_center(ra) - _row_center(rb)))
                max_l = max(_row_length(ra), _row_length(rb), 20.0)
                if d < 0.5 * max_l and d < min_dist:
                    min_dist = d
                    best_tc = f

            if best_tc is None:
                continue

            pre_a = [r for r in rows_a if int(r["frame_idx"]) <= best_tc]
            post_a = [r for r in rows_a if int(r["frame_idx"]) > best_tc]
            pre_b = [r for r in rows_b if int(r["frame_idx"]) <= best_tc]
            post_b = [r for r in rows_b if int(r["frame_idx"]) > best_tc]

            if len(pre_a) < 2 or len(pre_b) < 2 or len(post_a) < 2 or len(post_b) < 2:
                continue

            cost_keep = _connection_cost(pre_a, post_a, best_tc) + _connection_cost(pre_b, post_b, best_tc)
            cost_swap = _connection_cost(pre_a, post_b, best_tc) + _connection_cost(pre_b, post_a, best_tc)

            if cost_swap < 0.6 * cost_keep:
                for r in post_a:
                    r["track_id"] = tid_b
                for r in post_b:
                    r["track_id"] = tid_a
                rows_by_track[tid_a] = pre_a + post_b
                rows_by_track[tid_b] = pre_b + post_a
                remap[tid_a] = tid_b
                remap[tid_b] = tid_a

    return remap
