"""T15 合成全场景端到端运动学测试 (test_kin_synth.py)。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RE_ROOT = Path(__file__).resolve().parents[3]
if str(RE_ROOT) not in sys.path:
    sys.path.insert(0, str(RE_ROOT))

from tools.make_synthetic import Scn, make_series  # noqa: E402
from wormlab.config import load_config  # noqa: E402
from wormlab.kinematics import KinematicsEngine  # noqa: E402
from wormlab.schemas import Pose, Quality  # noqa: E402


def _to_poses(dense_seq: np.ndarray) -> list[Pose]:
    out: list[Pose] = []
    for d in dense_seq:
        diffs = np.diff(d, axis=0)
        L = float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())
        chord = float(np.linalg.norm(d[0] - d[-1]))
        out.append(Pose(dense=d, length=L, chord=chord, width=7.0, area=840.0, quality=Quality.OK))
    return out


@pytest.mark.parametrize("fps", [30.0, 10.0])
def test_kin_synth_crawl_profiles(fps: float) -> None:
    cfg = load_config("crawl_agar").kinematics

    # crawl_fwd -> s_count ≈ 10 (±1)
    d_fwd, exp_fwd = make_series(Scn(kind="crawl_fwd", duration_s=20.0, fps=fps))
    rep_fwd, _ = KinematicsEngine.run(_to_poses(d_fwd), fps=fps, cfg=cfg)
    assert abs(rep_fwd.s_count - exp_fwd["s_count"]) <= exp_fwd["s_tol"]
    assert rep_fwd.motion_type_legacy == "sinusoidal"

    # crawl_bwd -> s_count == 0, rejected.backward >= 7
    d_bwd, _ = make_series(Scn(kind="crawl_bwd", duration_s=20.0, fps=fps))
    rep_bwd, _ = KinematicsEngine.run(_to_poses(d_bwd), fps=fps, cfg=cfg)
    assert rep_bwd.s_count == 0
    assert rep_bwd.rejected["backward"] >= 7

    # head_wiggle / treadmill / wrong_dir / standing / static_noise -> s_count == 0
    for zero_kind in ("head_wiggle", "treadmill", "wrong_dir", "standing", "static_noise"):
        d_z, _ = make_series(Scn(kind=zero_kind, duration_s=15.0, fps=fps))
        rep_z, _ = KinematicsEngine.run(_to_poses(d_z), fps=fps, cfg=cfg)
        assert rep_z.s_count == 0, f"{zero_kind} (fps={fps}) s_count={rep_z.s_count} != 0"

    # omega -> c_turn_count == 3
    d_om, exp_om = make_series(Scn(kind="omega", duration_s=12.0, fps=fps))
    rep_om, _ = KinematicsEngine.run(_to_poses(d_om), fps=fps, cfg=cfg)
    assert rep_om.c_turn_count == exp_om["turn_count"]


@pytest.mark.parametrize("fps", [30.0, 15.0])
def test_kin_synth_swim_profiles_both_units(fps: float) -> None:
    cfg = load_config("swim_liquid").kinematics

    d_sw, exp_sw = make_series(Scn(kind="swim_thrash", duration_s=10.0, fps=fps, f=2.0))
    poses = _to_poses(d_sw)

    rep_cyc, _ = KinematicsEngine.run(poses, fps=fps, cfg=cfg, c_unit="cycle")
    rep_bnd, _ = KinematicsEngine.run(poses, fps=fps, cfg=cfg, c_unit="bend")

    assert abs(rep_cyc.c_cycles - exp_sw["c_cycles"]) <= exp_sw["c_tol"]
    assert abs(rep_cyc.c_bends - exp_sw["c_bends"]) <= 2 * exp_sw["c_tol"]
    assert rep_cyc.c_count == rep_cyc.c_cycles
    assert rep_bnd.c_count == rep_bnd.c_bends

    # 小幅摆动 (< bend_high) -> 0
    d_sm, _ = make_series(Scn(kind="swim_small_amp", duration_s=8.0, fps=fps, f=2.0))
    rep_sm, _ = KinematicsEngine.run(_to_poses(d_sm), fps=fps, cfg=cfg)
    assert rep_sm.c_cycles == 0
    assert rep_sm.c_bends == 0
