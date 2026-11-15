"""T19: associator + reid 单元测试。"""
from __future__ import annotations

import numpy as np

from wormlab.config import TrackingCfg
from wormlab.schemas import Detection
from wormlab.tracking.associator import associate
from wormlab.tracking.track import Track
from wormlab.tracking.tracker import Tracker


def test_associator_matches_active_and_reids_lost() -> None:
    cfg = TrackingCfg()
    t1 = Track(track_id=1, bbox=np.array([100.0, 100.0, 140.0, 140.0], dtype=np.float32), uid=1, first_frame=0)
    t2 = Track(track_id=2, bbox=np.array([300.0, 300.0, 340.0, 340.0], dtype=np.float32), uid=2, first_frame=0)
    t2.lost_frames = 5  # 走二级重关联

    d1 = Detection(bbox=np.array([102.0, 101.0, 142.0, 141.0], dtype=np.float32), conf=0.9)
    d2 = Detection(bbox=np.array([310.0, 305.0, 350.0, 345.0], dtype=np.float32), conf=0.85)

    matches, unmatched_dets, unmatched_tracks, _ = associate([t1, t2], [d1, d2], cfg, (640, 480))
    matched_pairs = dict(matches)
    assert matched_pairs.get(0) == 0
    assert matched_pairs.get(1) == 1
    assert unmatched_dets == []
    assert unmatched_tracks == []


def test_tracker_id_cap_and_ghost_lifecycle() -> None:
    cfg = TrackingCfg(persistence_sec=1.0, ghost_overlay_sec=0.2, max_lost_sec_legacy=1.0, confirm_hits=2)
    tracker = Tracker(cfg=cfg, fps=10.0, frame_wh=(640, 480), max_worm_id=2)

    dets_f0 = [
        Detection(bbox=np.array([50.0, 50.0, 90.0, 90.0], dtype=np.float32), conf=0.95),
        Detection(bbox=np.array([200.0, 200.0, 240.0, 240.0], dtype=np.float32), conf=0.85),
        Detection(bbox=np.array([400.0, 400.0, 440.0, 440.0], dtype=np.float32), conf=0.50),
    ]
    outs0 = tracker.update(dets_f0, frame_idx=0)
    # 虫数上限为 2，且按置信度降序分配 ID 1 和 2
    assert len(outs0) == 2
    assert [o.track_id for o in outs0] == [1, 2]

    # 下一帧只有第 1 只虫检测到，第 2 只进入 ghost
    outs1 = tracker.update([dets_f0[0]], frame_idx=1)
    assert len(outs1) == 2
    ghost = [o for o in outs1 if o.is_ghost]
    assert len(ghost) == 1 and ghost[0].track_id == 2
