"""T24 (05 §1): CacheWriter / CacheReader 读写往返与后进入虫判定单元测试。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from wormlab.pipeline.cache import CacheReader, CacheWriter
from wormlab.pipeline.late_entry import is_counted, partition_tracks_by_uid, select_primary_uid_per_track_id
from wormlab.schemas import N_DENSE, Pose, Quality, TrackOut


def test_cache_writer_reader_roundtrip(tmp_path: Path) -> None:
    writer = CacheWriter(tmp_path)

    dense_0 = np.column_stack([np.linspace(100.0, 160.0, N_DENSE), np.linspace(200.0, 210.0, N_DENSE)]).astype(np.float32)
    contour_0 = np.array([[98.0, 198.0], [162.0, 198.0], [162.0, 212.0], [98.0, 212.0]], dtype=np.float32)
    pose_0 = Pose(
        dense=dense_0,
        length=61.0,
        chord=60.8,
        width=6.5,
        area=400.0,
        quality=Quality.OK,
        flags=frozenset({"loop", "upscaled"}),
        contour=contour_0,
    )
    to_0 = TrackOut(
        track_id=1,
        bbox=np.array([95.0, 195.0, 165.0, 215.0], dtype=np.float32),
        det_idx=0,
        is_ghost=False,
        lost_frames=0,
        confirmed=True,
        user_label="N2-1",
        head_kpt_idx=0,
        head_xy=(100.0, 200.0),
        conf=0.92,
        first_frame=0,
        uid=1,
    )
    writer.append(frame_idx=0, track_out=to_0, pose=pose_0)

    # 第二行：ghost 帧，pose=None（含 NaN 与空轮廓）
    to_1 = TrackOut(
        track_id=1,
        bbox=np.array([96.0, 196.0, 166.0, 216.0], dtype=np.float32),
        det_idx=None,
        is_ghost=True,
        lost_frames=1,
        confirmed=True,
        user_label="N2-1",
        head_kpt_idx=0,
        head_xy=None,
        conf=0.18,
        first_frame=0,
        uid=1,
    )
    writer.append(frame_idx=1, track_out=to_1, pose=None)

    # 第三行：后进入的虫 (first_frame=45 > 1.0s * 30fps)
    to_late = TrackOut(
        track_id=2,
        bbox=np.array([300.0, 300.0, 350.0, 340.0], dtype=np.float32),
        det_idx=0,
        is_ghost=False,
        lost_frames=0,
        confirmed=False,
        user_label=None,
        head_kpt_idx=None,
        head_xy=None,
        conf=0.80,
        first_frame=45,
        uid=2,
    )
    writer.append(frame_idx=45, track_out=to_late, pose=pose_0)

    writer.close({"fps": 30.0, "width": 640, "height": 480, "total_frames": 60})

    reader = CacheReader(tmp_path)
    assert reader.meta["fps"] == 30.0
    assert len(reader.records) == 3

    r0 = reader.frame_rows(0)[0]
    assert r0.track_id == 1 and r0.uid == 1 and not r0.is_ghost
    assert r0.user_label == "N2-1" and r0.head_kpt_idx == 0
    assert r0.pose is not None and r0.pose.dense is not None
    assert np.allclose(r0.pose.dense, dense_0)
    assert r0.pose.flags == frozenset({"loop", "upscaled"})
    assert r0.pose.contour is not None and np.allclose(r0.pose.contour, contour_0)

    r1 = reader.frame_rows(1)[0]
    assert r1.is_ghost and r1.pose is None

    # 验证 late_entry 判定与分组
    groups = partition_tracks_by_uid(reader.records, fps=30.0, grace_s=1.0)
    assert groups[1].counted is True and groups[1].late_entry is False
    assert groups[2].counted is False and groups[2].late_entry is True
    assert is_counted(30, fps=30.0, cfg_or_grace=1.0) is True
    assert is_counted(31, fps=30.0, cfg_or_grace=1.0) is False

    selected, dropped = select_primary_uid_per_track_id(groups)
    assert set(selected.keys()) == {1, 2}
    assert dropped == []


def test_shape_descriptor_roundtrip_and_legacy_npz(tmp_path: Path) -> None:
    """08_SPEC Workstream A: 5 个免骨架形状字段无损往返；旧版缺列 .npz 自动回退 NaN。"""
    writer = CacheWriter(tmp_path)
    dense_0 = np.column_stack(
        [np.linspace(100.0, 160.0, N_DENSE), np.linspace(200.0, 210.0, N_DENSE)]
    ).astype(np.float32)
    pose_0 = Pose(
        dense=dense_0,
        length=61.0,
        chord=60.8,
        width=6.5,
        area=400.0,
        quality=Quality.OK,
        ell_aspect=0.42,
        ell_ecc=0.91,
        ell_angle=0.35,
        ell_skew_minor=-0.87,
        roi_motion=2.7,
    )
    to_0 = TrackOut(
        track_id=1,
        bbox=np.array([95.0, 195.0, 165.0, 215.0], dtype=np.float32),
        det_idx=0,
        is_ghost=False,
        lost_frames=0,
        confirmed=True,
        user_label=None,
        head_kpt_idx=None,
        head_xy=None,
        conf=0.9,
        first_frame=0,
        uid=1,
    )
    writer.append(frame_idx=0, track_out=to_0, pose=pose_0)
    writer.close({"fps": 30.0, "width": 640, "height": 480, "total_frames": 2})

    reader = CacheReader(tmp_path)
    r0 = reader.frame_rows(0)[0]
    assert r0.pose is not None
    assert r0.pose.ell_aspect == np.float32(0.42).item() or abs(r0.pose.ell_aspect - 0.42) < 1e-6
    assert abs(r0.pose.ell_ecc - 0.91) < 1e-6
    assert abs(r0.pose.ell_angle - 0.35) < 1e-6
    assert abs(r0.pose.ell_skew_minor - (-0.87)) < 1e-6
    assert abs(r0.pose.roi_motion - 2.7) < 1e-6

    # 旧版 tracks.npz（无 5 个形状列）-> CacheReader 自动填充 NaN 且不报错
    npz_path = tmp_path / "tracks.npz"
    with np.load(npz_path, allow_pickle=False) as data:
        legacy = {k: data[k] for k in data.files if k not in {"ell_aspect", "ell_ecc", "ell_angle", "ell_skew_minor", "roi_motion"}}
    np.savez_compressed(npz_path, **legacy)
    reader_legacy = CacheReader(tmp_path)
    rl = reader_legacy.frame_rows(0)[0]
    assert rl.pose is not None
    assert not np.isfinite(rl.pose.ell_aspect)
    assert not np.isfinite(rl.pose.ell_ecc)
    assert not np.isfinite(rl.pose.ell_angle)
    assert not np.isfinite(rl.pose.ell_skew_minor)
    assert not np.isfinite(rl.pose.roi_motion)
