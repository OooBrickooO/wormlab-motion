"""04 §5: 首帧人工头标记对齐与手动补框植入 (apply_head_config)。"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from .track import Track

if TYPE_CHECKING:
    from .tracker import Tracker


def _extract_head_xy(cfg_item: dict[str, Any]) -> tuple[float, float] | None:
    if "head_xy" in cfg_item and cfg_item["head_xy"] is not None:
        pt = cfg_item["head_xy"]
        return float(pt[0]), float(pt[1])
    if "x" in cfg_item and "y" in cfg_item:
        return float(cfg_item["x"]), float(cfg_item["y"])
    kpts = cfg_item.get("keypoints")
    h_idx = int(cfg_item.get("head_kpt_idx", 0) or 0)
    if kpts and isinstance(kpts, (list, tuple)) and len(kpts) > 0:
        idx = min(max(0, h_idx), len(kpts) - 1)
        kp = kpts[idx]
        if isinstance(kp, dict):
            return float(kp["x"]), float(kp["y"])
        if isinstance(kp, (list, tuple)) and len(kp) >= 2:
            return float(kp[0]), float(kp[1])
    return None


def _extract_centroid(cfg_item: dict[str, Any]) -> tuple[float, float] | None:
    c_pos = cfg_item.get("centroid")
    if isinstance(c_pos, (list, tuple)) and len(c_pos) >= 2:
        return float(c_pos[0]), float(c_pos[1])
    if "centroid_x" in cfg_item and "centroid_y" in cfg_item:
        return float(cfg_item["centroid_x"]), float(cfg_item["centroid_y"])
    bbox = cfg_item.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return 0.5 * (float(bbox[0]) + float(bbox[2])), 0.5 * (float(bbox[1]) + float(bbox[3]))
    if "x" in cfg_item and "y" in cfg_item:
        return float(cfg_item["x"]), float(cfg_item["y"])
    return None


def apply_head_config(tracker: Tracker, head_config: list[dict[str, Any]] | dict[str, Any]) -> None:
    """将用户首帧标注应用至当前轨迹列表，并对未检出的人工框选虫植入初始轨迹。"""
    if not head_config or not tracker.tracks:
        return

    cfgs: list[dict[str, Any]] = (
        head_config if isinstance(head_config, list) else list(head_config.values())
    )
    radius = float(tracker.cfg.head_match_radius_px)

    # 1. 匹配已有检测轨迹
    for item in cfgs:
        u_lbl = item.get("user_label") or item.get("label")
        h_idx = int(item.get("head_kpt_idx", 0) or 0)
        c_pos = _extract_centroid(item)

        best_t: Track | None = None
        best_dist = 1e6
        for t in tracker.tracks:
            if t.user_label is not None:
                continue
            bbox = t.get_state()
            cx = 0.5 * (float(bbox[0]) + float(bbox[2]))
            cy = 0.5 * (float(bbox[1]) + float(bbox[3]))
            d = math.hypot(cx - c_pos[0], cy - c_pos[1]) if c_pos is not None else 0.0
            if d < best_dist:
                best_dist = d
                best_t = t

        if best_t is not None and best_dist < radius:
            best_t.user_label = str(u_lbl) if u_lbl is not None else f"Worm-{best_t.track_id}"
            best_t.confirmed = True
            best_t.head_kpt_idx = h_idx
            best_t.head_xy = _extract_head_xy(item)

    # 2. 对首帧未匹配且带 bbox 的手动框选目标，植入初始 Kalman 轨迹
    for item in cfgs:
        u_lbl = item.get("user_label") or item.get("label")
        already_matched = any(t.user_label == u_lbl for t in tracker.tracks) if u_lbl is not None else False
        if already_matched or not item.get("bbox"):
            continue

        m_bbox = np.asarray(item["bbox"], dtype=np.float32)
        m_id = tracker.id_pool.alloc()
        if m_id is None:
            # 受限模式下池满：从后往前抢占一个未确认且无用户标签的临时轨迹 ID
            for cand in reversed(tracker.tracks):
                if cand.user_label is None and not cand.confirmed:
                    tracker.tracks.remove(cand)
                    tracker.id_pool.release(cand.track_id)
                    m_id = tracker.id_pool.alloc()
                    break
        if m_id is None:
            continue

        uid = tracker.alloc_uid()
        m_track = Track(
            track_id=m_id,
            bbox=m_bbox,
            uid=uid,
            first_frame=0,
            conf=1.0,
        )
        m_track.user_label = str(u_lbl) if u_lbl is not None else f"Worm-{m_id}"
        m_track.confirmed = True
        m_track.head_kpt_idx = int(item.get("head_kpt_idx", 0) or 0)
        m_track.head_xy = _extract_head_xy(item)
        if item.get("keypoints"):
            kpts = np.asarray(item["keypoints"], dtype=np.float32)
            if m_track.head_kpt_idx == 10 and len(kpts) == 11:
                kpts = kpts[::-1].copy()
            m_track.feed_pose(kpts)

        tracker.tracks.append(m_track)
        cx = 0.5 * (float(m_bbox[0]) + float(m_bbox[2]))
        cy = 0.5 * (float(m_bbox[1]) + float(m_bbox[3]))
        tracker.track_history[m_id].append((cx, cy))
