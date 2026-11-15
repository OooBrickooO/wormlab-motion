"""05 §3.5: 后进入虫判定与按 uid 实例分组辅助 (late_entry.py)。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from wormlab.schemas import FrameRecord, Pose


@dataclass
class UidTrackGroup:
    """单个轨迹实例 (uid) 的帧记录集合与计数资格。"""

    uid: int
    track_id: int
    first_frame: int
    counted: bool
    late_entry: bool
    user_label: str | None = None
    head_kpt_idx: int | None = None
    records: list[FrameRecord] = field(default_factory=list)

    def as_pose_list(self, total_frames: int) -> list[Pose | None]:
        """构造完整 0..total_frames-1 长度的姿态序列（缺失帧或幽灵帧为 None）。"""
        poses: list[Pose | None] = [None] * int(total_frames)
        for r in self.records:
            if 0 <= r.frame_idx < total_frames and not r.is_ghost:
                poses[r.frame_idx] = r.pose
        return poses


def is_counted(first_frame: int, fps: float, cfg_or_grace: Any = 1.0) -> bool:
    """判定首次出现在 first_frame 的轨迹实例是否属于初始虫（可计数）。"""
    if hasattr(cfg_or_grace, "counting"):
        grace_s = float(cfg_or_grace.counting.late_entry_grace_s)
    elif hasattr(cfg_or_grace, "late_entry_grace_s"):
        grace_s = float(cfg_or_grace.late_entry_grace_s)
    else:
        grace_s = float(cfg_or_grace)
    return int(first_frame) <= int(round(grace_s * float(fps)))


def partition_tracks_by_uid(
    records: list[FrameRecord],
    fps: float,
    grace_s: float = 1.0,
) -> dict[int, UidTrackGroup]:
    """将长格式缓存记录按 uid 分组，并标记每个实例的 counted / late_entry 状态。"""
    groups: dict[int, UidTrackGroup] = {}
    for rec in records:
        uid = int(rec.uid) if int(rec.uid) > 0 else int(rec.track_id)
        grp = groups.get(uid)
        if grp is None:
            f0 = int(rec.first_frame) if int(rec.first_frame) >= 0 else int(rec.frame_idx)
            cnt_flag = is_counted(f0, fps=fps, cfg_or_grace=grace_s)
            grp = UidTrackGroup(
                uid=uid,
                track_id=int(rec.track_id),
                first_frame=f0,
                counted=cnt_flag,
                late_entry=not cnt_flag,
                user_label=rec.user_label,
                head_kpt_idx=rec.head_kpt_idx,
            )
            groups[uid] = grp
        else:
            if rec.frame_idx < grp.first_frame:
                grp.first_frame = int(rec.frame_idx)
                grp.counted = is_counted(grp.first_frame, fps=fps, cfg_or_grace=grace_s)
                grp.late_entry = not grp.counted
            if grp.user_label is None and rec.user_label is not None:
                grp.user_label = rec.user_label
            if grp.head_kpt_idx is None and rec.head_kpt_idx is not None:
                grp.head_kpt_idx = rec.head_kpt_idx
        grp.records.append(rec)

    for grp in groups.values():
        grp.records.sort(key=lambda r: r.frame_idx)
    return groups


def select_primary_uid_per_track_id(
    uid_groups: dict[int, UidTrackGroup],
) -> tuple[dict[int, UidTrackGroup], list[int]]:
    """当同一 track_id 对应多个 uid（ID 池回收复用）时，汇总优先保留 counted 且持续帧数最长的 uid。"""
    by_tid: dict[int, list[UidTrackGroup]] = {}
    for grp in uid_groups.values():
        by_tid.setdefault(grp.track_id, []).append(grp)

    selected: dict[int, UidTrackGroup] = {}
    dropped_uids: list[int] = []
    for tid, cand_list in by_tid.items():
        if len(cand_list) == 1:
            selected[tid] = cand_list[0]
            continue
        cand_sorted = sorted(
            cand_list,
            key=lambda g: (1 if g.counted else 0, len(g.records), -g.first_frame),
            reverse=True,
        )
        selected[tid] = cand_sorted[0]
        for d in cand_sorted[1:]:
            dropped_uids.append(d.uid)
    return selected, dropped_uids
