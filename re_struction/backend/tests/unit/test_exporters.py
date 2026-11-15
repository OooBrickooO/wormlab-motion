"""T25 (05 §4, §6): exporters (JSON / CSV / ZIP) 与 Pass B 端到端单元测试。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from zipfile import ZipFile

from server.exporters import csv_report, summary_json, zip_pack
from wormlab.config import load_config
from wormlab.pipeline import JobSpec, PassB


def test_exporters_json_csv_zip(tmp_path: Path) -> None:
    sample_summary = [
        {
            "track_id": 1,
            "user_label": "Worm-A",
            "c_count": 6,
            "s_count": 3,
            "is_bidirectional": False,
            "motion_type": "sinusoidal",
            "activity_score": 0.85,
            "is_static": False,
            "live_freq": 0.45,
            "avg_freq": 0.45,
            "cycles": 9,
            "swings": 9,
            "bends_per_min": 27.0,
            "c_cycles": 6,
            "c_bends": 12,
            "spatial_stats": {
                "total_path_length_px": 180.5,
                "net_displacement_px": 75.2,
                "start_centroid_bl": [50.0, 200.0],
                "end_centroid_bl": [125.2, 200.0],
                "bbox_coverage_bl": {"x_min": 40.0, "y_min": 180.0, "x_max": 140.0, "y_max": 220.0},
            },
        }
    ]

    j_path = summary_json.write(tmp_path / "test_summary.json", sample_summary)
    loaded = json.loads(j_path.read_text(encoding="utf-8"))
    assert len(loaded) == 1 and loaded[0]["track_id"] == 1

    csv_name = csv_report.write(tmp_path, "test_stem", sample_summary)
    csv_path = tmp_path / csv_name
    assert csv_path.exists()
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == csv_report.CSV_HEADER
    assert rows[1][0] == "1"
    assert rows[1][5] == "否"
    assert rows[1][-2:] == ["6", "12"]

    sub_dir = tmp_path / "sub_videos"
    sub_dir.mkdir()
    (sub_dir / "worm_1_cycles_9.mp4").write_bytes(b"fake_mp4_bytes")
    z_path = zip_pack.write(tmp_path / "sub.zip", sub_dir)
    with ZipFile(z_path, "r") as zf:
        assert zf.namelist() == ["worm_1_cycles_9.mp4"]


def test_pass_b_on_move3_cache(tmp_path: Path) -> None:
    re_root = Path(__file__).resolve().parents[3]
    cache_dir = re_root / "data" / "cache" / "Move-3"
    if not (cache_dir / "tracks.npz").exists():
        return

    cfg = load_config("crawl_agar")
    job = JobSpec(
        task_id="test_move3_b",
        video_path="",
        profile="crawl_agar",
        c_unit="cycle",
        result_dir=str(tmp_path),
    )
    res = PassB(cfg).run(cache_dir=cache_dir, job=job, render_videos=False)
    assert Path(res.summary_path).exists()
    assert Path(res.csv_path).exists()
    assert Path(res.zip_path).exists()
    assert len(res.id_pose_summary) > 0
