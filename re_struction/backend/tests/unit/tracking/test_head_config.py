"""T20: head_config 抢占、标签与手绘点注入单元测试。"""
from __future__ import annotations

import numpy as np

from wormlab.config import TrackingCfg
from wormlab.schemas import Detection
from wormlab.tracking.tracker import Tracker


def test_head_config_matches_by_centroid_and_sets_head_xy() -> None:
    cfg = TrackingCfg()
    head_config = [
        {"user_label": "N2-A", "centroid": [110.0, 110.0], "head_xy": [105.0, 105.0], "head_kpt_idx": 0},
        {"user_label": "N2-B", "centroid": [310.0, 310.0], "head_xy": [305.0, 305.0], "head_kpt_idx": 10},
    ]
    tracker = Tracker(cfg=cfg, fps=10.0, frame_wh=(640, 480), head_config=head_config)

    dets = [
        Detection(bbox=np.array([90.0, 90.0, 130.0, 130.0], dtype=np.float32), conf=0.95),
        Detection(bbox=np.array([290.0, 290.0, 330.0, 330.0], dtype=np.float32), conf=0.80),
    ]
    outs = tracker.update(dets, frame_idx=0)
    by_id = {o.track_id: o for o in outs}
    assert set(by_id.keys()) == {1, 2}
    assert by_id[1].user_label == "N2-A"
    assert by_id[2].user_label == "N2-B"

    t1 = tracker.get_track(1)
    assert t1 is not None and t1.head_xy is not None
    assert np.allclose(t1.head_xy, [105.0, 105.0])
    assert t1.head_kpt_idx == 0


def test_head_config_creates_manual_track_when_unmatched() -> None:
    cfg = TrackingCfg()
    head_config = [
        {
            "user_label": "ManualWorm",
            "bbox": [480.0, 480.0, 520.0, 520.0],
            "head_xy": [485.0, 485.0],
            "head_kpt_idx": 0,
        },
    ]
    tracker = Tracker(cfg=cfg, fps=10.0, frame_wh=(640, 480), head_config=head_config, max_worm_id=1)
    # 检测框远离 (500,500) 超过 250px -> 抢占未确认轨迹 ID=1 并创建手动轨迹
    dets = [
        Detection(bbox=np.array([20.0, 20.0, 60.0, 60.0], dtype=np.float32), conf=0.9),
    ]
    outs = tracker.update(dets, frame_idx=0)
    t1 = tracker.get_track(1)
    assert t1 is not None
    assert t1.user_label == "ManualWorm"
    assert any(o.track_id == 1 and o.user_label == "ManualWorm" for o in outs)
