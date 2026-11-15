"""T25 (05 §5): compat/legacy_payload.py 字段集合与左下角坐标换算单元测试。"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from wormlab.compat import to_legacy_summary, to_legacy_track
from wormlab.schemas import Behavior, N_DENSE, Pose, Quality

REQUIRED_TRACK_LIVE_KEYS: set[str] = {
    "track_id",
    "user_label",
    "c_count",
    "s_count",
    "is_bidirectional",
    "motion_type",
    "activity_score",
    "is_static",
    "is_ghost",
    "lost_frames",
    "live_freq",
    "avg_freq",
    "cycles",
    "swings",
    "bends_per_min",
    "centroid_bl",
    "head_bl",
    "head_kpt_idx",
    "keypoints_bl",
    "behavior",
    "s_cycles",
    "freq_s_hz",
    "freq_c_hz",
    "pose_quality",
    "c_cycles",
    "c_bends",
    "c_unit",
    "counted",
    "late_entry",
}

REQUIRED_TRACK_SUMMARY_KEYS: set[str] = {
    "track_id",
    "user_label",
    "c_count",
    "s_count",
    "is_bidirectional",
    "motion_type",
    "activity_score",
    "is_static",
    "live_freq",
    "avg_freq",
    "cycles",
    "swings",
    "bends_per_min",
    "spatial_stats",
    "s_cycles",
    "c_cycles",
    "c_bends",
    "c_unit",
    "c_turn_count",
    "freq_s_hz",
    "freq_c_hz",
    "mode_time",
    "rejected",
    "counted",
    "late_entry",
}


def test_to_legacy_track_and_late_entry_zeroing() -> None:
    dense = np.column_stack([np.linspace(100.0, 200.0, N_DENSE), np.full(N_DENSE, 120.0)]).astype(np.float32)
    pose = Pose(dense=dense, length=100.0, chord=100.0, width=6.0, area=600.0, quality=Quality.OK)
    snap = SimpleNamespace(
        s_count=3,
        c_cycles=2,
        c_bends=4,
        c_turn_count=1,
        c_count=3,
        s_cycles=3.2,
        freq_s_hz=0.5,
        freq_c_hz=0.0,
        activity_score=0.75,
        live_freq=0.6,
    )

    live = to_legacy_track(
        tid=1,
        label="N2",
        behavior_state=Behavior.CRAWL,
        report_like=snap,
        pose=pose,
        frame_h=480,
        is_ghost=False,
        lost_frames=0,
        head_kpt_idx=0,
        counted=True,
        c_unit="cycle",
        observed_sec=10.0,
    )
    assert REQUIRED_TRACK_LIVE_KEYS.issubset(set(live.keys()))
    assert live["is_bidirectional"] is False
    assert live["s_count"] == 3
    assert live["c_cycles"] == 2 and live["c_bends"] == 4
    # 左下角坐标 y_bl = 480 - 120 = 360
    assert live["head_bl"] == {"x": 100.0, "y": 360.0}
    assert len(live["keypoints_bl"]) == 11

    # 后进入的虫：counted=False -> 计数全部置 0
    late_live = to_legacy_track(
        tid=2,
        label=None,
        behavior_state=Behavior.CRAWL,
        report_like=snap,
        pose=pose,
        frame_h=480,
        counted=False,
    )
    assert late_live["counted"] is False and late_live["late_entry"] is True
    assert late_live["s_count"] == 0 and late_live["c_count"] == 0 and late_live["cycles"] == 0


def test_to_legacy_summary_schema_and_filtering() -> None:
    rep1 = SimpleNamespace(
        s_count=4,
        c_cycles=5,
        c_bends=10,
        c_turn_count=1,
        s_cycles=4.1,
        freq_s_hz=0.4,
        freq_c_hz=1.2,
        activity_score=0.88,
        dominant_behavior=Behavior.THRASH,
        mode_time={"thrash": 10.0},
        rejected={"no_disp": 1, "wrong_dir": 0, "backward": 0},
        spatial_stats={
            "total_path_length_px": 250.0,
            "net_displacement_px": 95.0,
            "start_centroid_bl": [100.0, 300.0],
            "end_centroid_bl": [195.0, 300.0],
            "bbox_coverage_bl": {"x_min": 80.0, "y_min": 280.0, "x_max": 220.0, "y_max": 320.0},
        },
    )
    summaries_cycle = to_legacy_summary(
        reports_by_tid={1: rep1, 2: rep1},
        labels_by_tid={1: "W1", 2: "LateWorm"},
        counted_by_tid={1: True, 2: False},
        c_unit="cycle",
        include_late=False,
    )
    # 默认不包含后进入虫 (tid=2)
    assert len(summaries_cycle) == 1
    item = summaries_cycle[0]
    assert REQUIRED_TRACK_SUMMARY_KEYS.issubset(set(item.keys()))
    assert item["c_count"] == 6  # c_cycles(5) + turn(1)
    assert item["c_cycles"] == 5 and item["c_bends"] == 10

    summaries_bend = to_legacy_summary(
        reports_by_tid={1: rep1},
        labels_by_tid={1: "W1"},
        counted_by_tid={1: True},
        c_unit="bend",
    )
    assert summaries_bend[0]["c_count"] == 11  # c_bends(10) + turn(1)


def test_to_legacy_summary_reads_track_report_fields() -> None:
    from wormlab.kinematics.metrics import TrackReport

    rep = TrackReport(
        s_count=2,
        s_cycles_float=2.345,
        c_cycles=0,
        c_bends=0,
        c_turn_count=0,
        c_count=0,
        c_unit="cycle",
        rejected={"no_disp": 0, "wrong_dir": 0, "backward": 0},
        aborted_runs=0,
        mode_time={"crawl": 8.0},
        wave_eps_median=0.45,
        observed_seconds=10.0,
        freq_s_hz=0.2,
        freq_c_hz=0.0,
        avg_freq_legacy=0.2,
        activity_score=0.05,
        motion_type_legacy="static",
        n_ok=300,
        n_suspect=0,
        n_bad=0,
        n_missing=0,
        spatial={
            "total_path_length_px": 180.0,
            "net_displacement_px": 95.0,
            "start_centroid_bl": [10.0, 20.0],
            "end_centroid_bl": [105.0, 20.0],
            "bbox_coverage_bl": {"x_min": 5.0, "y_min": 15.0, "x_max": 110.0, "y_max": 25.0},
        },
    )
    items = to_legacy_summary(reports_by_tid={1: rep}, labels_by_tid={1: "W0"}, counted_by_tid={1: True})
    assert len(items) == 1
    assert items[0]["s_cycles"] == 2.345
    assert items[0]["is_static"] is True
    assert items[0]["spatial_stats"]["total_path_length_px"] == 180.0
    assert items[0]["spatial_stats"]["net_displacement_px"] == 95.0
