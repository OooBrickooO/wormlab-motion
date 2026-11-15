"""T15 单测：验证 KinematicsEngine 流式逐帧 push()+snapshot()+finalize() 与批量 run() 完全一致且计数单调不减。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RE_ROOT = Path(__file__).resolve().parents[4]
if str(RE_ROOT) not in sys.path:
    sys.path.insert(0, str(RE_ROOT))

from tools.make_synthetic import Scn, make_series  # noqa: E402
from wormlab.config import load_config  # noqa: E402
from wormlab.kinematics import KinematicsEngine  # noqa: E402
from wormlab.schemas import Pose, Quality  # noqa: E402


def _dense_to_poses(dense_seq: np.ndarray) -> list[Pose]:
    poses: list[Pose] = []
    for d in dense_seq:
        chord = float(np.linalg.norm(d[0] - d[-1]))
        poses.append(Pose(dense=d, length=120.0, chord=chord, width=7.0, area=840.0, quality=Quality.OK))
    return poses


def test_engine_streaming_push_equals_batch_run() -> None:
    cfg = load_config("crawl_agar").kinematics
    dense_seq, _ = make_series(Scn(kind="crawl_fwd", duration_s=12.0, fps=30.0))
    poses = _dense_to_poses(dense_seq)

    # (a) 一次性 batch run
    rep_batch, pf_batch = KinematicsEngine.run(poses, fps=30.0, cfg=cfg)

    # (b) 逐帧 push + 途中频繁调用 snapshot()
    eng = KinematicsEngine(cfg=cfg, fps=30.0)
    snap_s_counts: list[int] = []
    for i, p in enumerate(poses):
        eng.push(i, p)
        snap = eng.snapshot()
        snap_s_counts.append(snap.s_count)

    rep_stream, pf_stream = eng.finalize()

    assert rep_stream == rep_batch
    assert pf_stream == pf_batch

    # 任意时刻 snapshot() 计数单调不减且 <= 最终计数
    assert all(a <= b for a, b in zip(snap_s_counts, snap_s_counts[1:]))
    assert snap_s_counts[-1] <= rep_stream.s_count
