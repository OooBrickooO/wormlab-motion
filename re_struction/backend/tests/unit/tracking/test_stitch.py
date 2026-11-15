"""T22 (04 §8): 离线轨迹交叉 ID 交换缝合器单元测试。"""
from __future__ import annotations

import numpy as np

from wormlab.config import StitchCfg, TrackingCfg
from wormlab.tracking.stitch import stitch


def _make_crossing_swapped_tracks() -> dict[int, list[dict]]:
    """构造两条匀速直线交叉运动、但在 t=10 帧后发生 ID 交换的合成轨迹。"""
    rows_1: list[dict] = []
    rows_2: list[dict] = []
    for f in range(21):
        # 真实物理轨迹 A: 从左向右匀速移动 y=100, x=50 + 10*f
        pos_a = np.array([50.0 + 10.0 * f, 100.0])
        # 真实物理轨迹 B: 从上向下匀速移动 x=150, y=0 + 10*f (在 f=10 时与 A 在 (150,100) 相交)
        pos_b = np.array([150.0, 0.0 + 10.0 * f])

        if f <= 10:
            c1, c2 = pos_a, pos_b
        else:
            # f > 10 发生 ID 交换：track 1 拿到了物理 B 的位置，track 2 拿到了物理 A 的位置
            c1, c2 = pos_b, pos_a

        rows_1.append(
            {
                "frame_idx": f,
                "track_id": 1,
                "centroid": c1.copy(),
                "bbox": np.array([c1[0] - 15, c1[1] - 15, c1[0] + 15, c1[1] + 15], dtype=np.float32),
                "length": 60.0,
            }
        )
        rows_2.append(
            {
                "frame_idx": f,
                "track_id": 2,
                "centroid": c2.copy(),
                "bbox": np.array([c2[0] - 15, c2[1] - 15, c2[0] + 15, c2[1] + 15], dtype=np.float32),
                "length": 60.0,
            }
        )
    return {1: rows_1, 2: rows_2}


def test_stitch_disabled_by_default() -> None:
    cfg = TrackingCfg()
    assert cfg.stitch.enabled is False
    tracks = _make_crossing_swapped_tracks()
    remap = stitch(tracks, cfg)
    assert remap == {1: 1, 2: 2}
    # 末尾帧未发生交换修正
    assert np.allclose(tracks[1][-1]["centroid"], [150.0, 200.0])


def test_stitch_fixes_crossing_id_swap_when_enabled() -> None:
    cfg = TrackingCfg(stitch=StitchCfg(enabled=True))
    tracks = _make_crossing_swapped_tracks()
    remap = stitch(tracks, cfg)
    assert remap == {1: 2, 2: 1}
    # 修正后 track 1 全程沿 y=100 水平向右运动，末尾帧在 (250, 100)
    assert np.allclose(tracks[1][-1]["centroid"], [250.0, 100.0])
    assert np.allclose(tracks[2][-1]["centroid"], [150.0, 200.0])
