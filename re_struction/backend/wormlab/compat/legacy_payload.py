"""05 §5 & API_CONTRACT §4: 新旧字段映射与左下角坐标换算兼容层 (legacy_payload.py)。

注意：本模块是唯一允许执行 y_bl = frame_h - y 左下角原点换算的模块。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from wormlab.kinematics.behavior import behavior_to_legacy_motion_type
from wormlab.schemas import Behavior, Pose, Quality


def motion_type_legacy(behavior: Behavior | str) -> str:
    if isinstance(behavior, Behavior):
        return behavior_to_legacy_motion_type(behavior)
    try:
        return behavior_to_legacy_motion_type(Behavior(str(behavior)))
    except ValueError:
        return "unknown"


def _pt_bl(xy: tuple[float, float] | np.ndarray, frame_h: int) -> dict[str, float] | None:
    x_f, y_f = float(xy[0]), float(xy[1])
    if not (np.isfinite(x_f) and np.isfinite(y_f)):
        return None
    return {"x": round(x_f, 2), "y": round(float(frame_h) - y_f, 2)}


def to_legacy_track(
    tid: int,
    label: str | None,
    behavior_state: Behavior | str,
    report_like: Any,
    pose: Pose | None,
    frame_h: int,
    is_ghost: bool = False,
    lost_frames: int = 0,
    head_kpt_idx: int | None = None,
    counted: bool = True,
    c_unit: str = "cycle",
    observed_sec: float = 0.0,
) -> dict[str, Any]:
    """将单帧实时/终稿状态组装为 API_CONTRACT §4 定义的 TrackLive 字典。"""
    beh_str = behavior_state.value if isinstance(behavior_state, Behavior) else str(behavior_state)
    m_type = motion_type_legacy(beh_str)
    is_stat = beh_str == Behavior.STATIC.value

    if not counted:
        c_cnt = 0
        s_cnt = 0
        c_cyc = 0
        c_bnd = 0
        s_cyc = 0.0
        freq_s = 0.0
        freq_c = 0.0
        act_score = 0.0
        avg_f = 0.0
        live_f = 0.0
        bpm = 0.0
    else:
        s_cnt = int(getattr(report_like, "s_count", 0))
        c_cyc = int(getattr(report_like, "c_cycles", 0))
        c_bnd = int(getattr(report_like, "c_bends", 0))
        turn_cnt = int(getattr(report_like, "c_turn_count", 0))
        if hasattr(report_like, "c_count"):
            c_cnt = int(getattr(report_like, "c_count", 0))
        else:
            c_cnt = (c_bnd if str(c_unit) == "bend" else c_cyc) + turn_cnt
        s_cyc = float(getattr(report_like, "s_cycles", float(s_cnt)))
        freq_s = float(getattr(report_like, "freq_s_hz", 0.0))
        freq_c = float(getattr(report_like, "freq_c_hz", 0.0))
        act_score = float(getattr(report_like, "activity_score", 0.0))

        total_cs = c_cnt + s_cnt
        obs_s = float(observed_sec) if observed_sec > 0.0 else float(getattr(report_like, "observed_sec", 0.0))
        if obs_s > 1e-3:
            avg_f = float(total_cs) / obs_s
        else:
            avg_f = max(freq_s, freq_c)
        live_f = float(getattr(report_like, "live_freq", avg_f))
        bpm = avg_f * 60.0

    centroid_bl: dict[str, float] | None = None
    head_bl: dict[str, float] | None = None
    keypoints_bl: list[dict[str, float]] | None = None
    pose_q_str = Quality.MISSING.name

    if pose is not None:
        pose_q_str = pose.quality.name if isinstance(pose.quality, Quality) else str(pose.quality)
        if pose.dense is not None and len(pose.dense) > 0 and np.isfinite(pose.dense).all():
            c_xy = np.mean(pose.dense, axis=0)
            centroid_bl = _pt_bl(c_xy, frame_h)
            head_bl = _pt_bl(pose.dense[0], frame_h)
            pts11 = pose.pts11
            if pts11 is not None:
                k_list: list[dict[str, float]] = []
                for pt in pts11:
                    p_bl = _pt_bl(pt, frame_h)
                    if p_bl is not None:
                        k_list.append(p_bl)
                keypoints_bl = k_list if len(k_list) == 11 else None

    total_cycles = int(c_cnt + s_cnt)
    return {
        "track_id": int(tid),
        "user_label": label,
        "c_count": int(c_cnt),
        "s_count": int(s_cnt),
        "is_bidirectional": False,
        "motion_type": m_type,
        "activity_score": round(act_score, 4),
        "is_static": bool(is_stat),
        "is_ghost": bool(is_ghost),
        "lost_frames": int(lost_frames),
        "live_freq": round(live_f, 4),
        "avg_freq": round(avg_f, 4),
        "cycles": total_cycles,
        "swings": total_cycles,
        "bends_per_min": round(bpm, 2),
        "centroid_bl": centroid_bl,
        "head_bl": head_bl,
        "head_kpt_idx": head_kpt_idx,
        "keypoints_bl": keypoints_bl,
        # 新增可选字段
        "behavior": beh_str,
        "s_cycles": round(s_cyc, 3),
        "freq_s_hz": round(freq_s, 4),
        "freq_c_hz": round(freq_c, 4),
        "pose_quality": pose_q_str,
        "c_cycles": int(c_cyc),
        "c_bends": int(c_bnd),
        "c_unit": str(c_unit),
        "counted": bool(counted),
        "late_entry": not bool(counted),
    }


def to_legacy_summary(
    reports_by_tid: dict[int, Any],
    labels_by_tid: dict[int, str | None] | None = None,
    counted_by_tid: dict[int, bool] | None = None,
    c_unit: str = "cycle",
    include_late: bool = False,
) -> list[dict[str, Any]]:
    """按 track_id 升序组装 API_CONTRACT §4 定义的 TrackSummary 列表。"""
    labels = labels_by_tid or {}
    counted_map = counted_by_tid or {}
    summaries: list[dict[str, Any]] = []

    for tid in sorted(reports_by_tid.keys()):
        is_cnt = bool(counted_map.get(tid, True))
        if not is_cnt and not include_late:
            continue
        rep = reports_by_tid[tid]
        lbl = labels.get(tid)

        if not is_cnt:
            c_cnt = 0
            s_cnt = 0
            c_cyc = 0
            c_bnd = 0
            turn_cnt = 0
            s_cyc = 0.0
            freq_s = 0.0
            freq_c = 0.0
            avg_f = 0.0
            bpm = 0.0
            act_score = 0.0
            beh_str = Behavior.STATIC.value
        else:
            s_cnt = int(getattr(rep, "s_count", 0))
            c_cyc = int(getattr(rep, "c_cycles", 0))
            c_bnd = int(getattr(rep, "c_bends", 0))
            turn_cnt = int(getattr(rep, "c_turn_count", 0))
            if str(c_unit) == "bend":
                c_cnt = c_bnd + turn_cnt
            else:
                c_cnt = c_cyc + turn_cnt
            s_cyc = float(
                getattr(
                    rep,
                    "s_cycles_float",
                    getattr(rep, "s_cycles", float(s_cnt)),
                )
            )
            freq_s = float(getattr(rep, "freq_s_hz", 0.0))
            freq_c = float(getattr(rep, "freq_c_hz", 0.0))
            act_score = float(getattr(rep, "activity_score", 0.0))

            mode_time = getattr(rep, "mode_time", {}) or {}
            obs_s = float(sum(float(v) for v in mode_time.values()))
            total_cs = c_cnt + s_cnt
            avg_f = (float(total_cs) / obs_s) if obs_s > 1e-3 else max(freq_s, freq_c)
            bpm = avg_f * 60.0

            legacy_mt = getattr(rep, "motion_type_legacy", None)
            if isinstance(legacy_mt, str) and legacy_mt:
                m_type = legacy_mt
                is_stat = m_type == "static"
            else:
                dom_beh = getattr(rep, "dominant_behavior", Behavior.STATIC)
                beh_str = dom_beh.value if isinstance(dom_beh, Behavior) else str(dom_beh)
                m_type = motion_type_legacy(beh_str)
                is_stat = beh_str == Behavior.STATIC.value

        total_cycles = int(c_cnt + s_cnt)
        sp_raw = getattr(rep, "spatial", None)
        if sp_raw is not None and hasattr(sp_raw, "total_displacement_px"):
            sp_stats = {
                "total_displacement_px": round(float(sp_raw.total_displacement_px), 2),
                "total_displacement_bl": round(float(sp_raw.total_displacement_bl), 2),
                "net_displacement_px": round(float(sp_raw.net_displacement_px), 2),
                "net_displacement_bl": round(float(sp_raw.net_displacement_bl), 2),
                "tortuosity": round(float(sp_raw.tortuosity), 2),
                "avg_speed_bl_per_s": round(float(sp_raw.avg_speed_bl_per_s), 2),
                "activity_score": round(float(sp_raw.activity_score), 3),
                "mean_body_length_px": round(float(sp_raw.mean_body_length_px), 1),
            }
        elif isinstance(sp_raw, dict) and sp_raw:
            sp_stats = sp_raw
        else:
            sp_stats = getattr(rep, "spatial_stats", None) or {
            "total_path_length_px": 0.0,
            "net_displacement_px": 0.0,
            "start_centroid_bl": [0.0, 0.0],
            "end_centroid_bl": [0.0, 0.0],
            "bbox_coverage_bl": {"x_min": 0.0, "y_min": 0.0, "x_max": 0.0, "y_max": 0.0},
        }

        summaries.append(
            {
                "track_id": int(tid),
                "user_label": lbl,
                "c_count": int(c_cnt),
                "s_count": int(s_cnt),
                "is_bidirectional": False,
                "motion_type": m_type,
                "activity_score": round(act_score, 4),
                "is_static": bool(is_stat),
                "live_freq": round(avg_f, 4),
                "avg_freq": round(avg_f, 4),
                "cycles": total_cycles,
                "swings": total_cycles,
                "bends_per_min": round(bpm, 2),
                "spatial_stats": sp_stats,
                # 新增扩展字段
                "s_cycles": round(s_cyc, 3),
                "c_cycles": int(c_cyc),
                "c_bends": int(c_bnd),
                "c_unit": str(c_unit),
                "c_turn_count": int(turn_cnt),
                "freq_s_hz": round(freq_s, 4),
                "freq_c_hz": round(freq_c, 4),
                "mode_time": getattr(rep, "mode_time", {}),
                "rejected": getattr(rep, "rejected", {}),
                "counted": bool(is_cnt),
                "late_entry": not bool(is_cnt),
            }
        )

    return summaries
