"""T13 单测：对照 proto_kinematics.py 在 30 fps 与 10 fps 下覆盖全部 12 类合成场景。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RE_ROOT = Path(__file__).resolve().parents[4]
if str(RE_ROOT) not in sys.path:
    sys.path.insert(0, str(RE_ROOT))

from tools.make_synthetic import Scn, make_series  # noqa: E402
from wormlab.config import load_config  # noqa: E402
from wormlab.kinematics.features import Feat, compute_features  # noqa: E402
from wormlab.kinematics.modes import modal_phase  # noqa: E402
from wormlab.kinematics.s_runs import SRunCounter  # noqa: E402
from wormlab.kinematics.wave import wave_gate  # noqa: E402


def _run_s_counter_on_dense(
    dense_seq: np.ndarray,
    fps: float,
    missing_slice: tuple[int, int] | None = None,
) -> SRunCounter:
    cfg = load_config("crawl_agar").kinematics
    feats: list[Feat | None] = []
    for idx, d in enumerate(dense_seq):
        if missing_slice is not None and missing_slice[0] <= idx < missing_slice[1]:
            feats.append(None)
        else:
            feats.append(compute_features(d, cfg))

    half = max(1, int(round(cfg.wave.win_s * fps / 2.0)))
    counter = SRunCounter(cfg.s_run, fps=fps)

    for i, f in enumerate(feats):
        lo, hi = max(0, i - half), min(len(feats), i + half + 1)
        win = feats[lo:hi]
        g = wave_gate(win, cfg.wave, expected_len=2 * half + 1) if f is not None else wave_gate([], cfg.wave, expected_len=2 * half + 1)
        valid_Ls = [x.L for x in feats[max(0, i - int(round(2.0 * fps))) : i + 1] if x is not None]
        L_ref = float(np.median(valid_Ls)) if valid_Ls else 120.0
        phi = modal_phase(f.a1, f.a2) if f is not None else None
        counter.step(
            i=i,
            gate_ok=g.ok,
            phi=phi,
            center=f.center if f is not None else None,
            u=f.u if f is not None else None,
            L_ref=L_ref,
        )
    return counter


@pytest.mark.parametrize("fps", [30.0, 10.0])
def test_s_run_all_synthetic_scenarios(fps: float) -> None:
    # 1. 前进 10 周期 -> s_count = 10 ± 1
    d_fwd, _ = make_series(Scn(kind="crawl_fwd", duration_s=20.0, fps=fps))
    c_fwd = _run_s_counter_on_dense(d_fwd, fps)
    assert abs(c_fwd.s_count - 10) <= 1

    # 2. 后退 10 周期 -> s_count = 0, rejected.backward >= 8
    d_bwd, _ = make_series(Scn(kind="crawl_bwd", duration_s=20.0, fps=fps))
    c_bwd = _run_s_counter_on_dense(d_bwd, fps)
    assert c_bwd.s_count == 0
    assert c_bwd.rejected["backward"] >= 8

    # 3. 原地跑步机 -> s_count = 0, rejected.no_disp + wrong_dir >= 8
    d_trd, _ = make_series(Scn(kind="treadmill", duration_s=20.0, fps=fps))
    c_trd = _run_s_counter_on_dense(d_trd, fps)
    assert c_trd.s_count == 0
    assert (c_trd.rejected["no_disp"] + c_trd.rejected["wrong_dir"]) >= 8

    # 4. 波向前但质心后退 -> s_count = 0, rejected.wrong_dir >= 8
    d_wd, _ = make_series(Scn(kind="wrong_dir", duration_s=20.0, fps=fps))
    c_wd = _run_s_counter_on_dense(d_wd, fps)
    assert c_wd.s_count == 0
    assert c_wd.rejected["wrong_dir"] >= 8

    # 5. 仅头部试探摆动 -> s_count = 0
    d_hw, _ = make_series(Scn(kind="head_wiggle", duration_s=20.0, fps=fps))
    c_hw = _run_s_counter_on_dense(d_hw, fps)
    assert c_hw.s_count == 0

    # 6. 相位来回晃 ±0.4 周期 -> s_count = 0
    d_wb, _ = make_series(Scn(kind="wobble", duration_s=20.0, fps=fps))
    c_wb = _run_s_counter_on_dense(d_wb, fps)
    assert c_wb.s_count == 0

    # 7. 前进 3.5 周期后反向 3.5 周期 (14s) -> s_count = 3, rejected.backward >= 1, aborted >= 1
    d_fb35, _ = make_series(Scn(kind="fwd_then_back", duration_s=14.0, fps=fps))
    c_fb35 = _run_s_counter_on_dense(d_fb35, fps)
    assert c_fb35.s_count == 3
    assert c_fb35.rejected["backward"] >= 1
    assert c_fb35.aborted >= 1

    # 8. 前进 2.5 周期后反向 2.5 周期 (10s) -> s_count = 2（半截作废，已完成保留）
    d_fb25, _ = make_series(Scn(kind="fwd_25_then_back", duration_s=10.0, fps=fps))
    c_fb25 = _run_s_counter_on_dense(d_fb25, fps)
    assert c_fb25.s_count == 2
    assert c_fb25.aborted >= 1

    # 9. 驻波 -> s_count = 0
    d_st, _ = make_series(Scn(kind="standing", duration_s=20.0, fps=fps))
    c_st = _run_s_counter_on_dense(d_st, fps)
    assert c_st.s_count == 0

    # 10. 静止噪声 + jitter -> s_count = 0
    d_ns, _ = make_series(Scn(kind="static_noise", duration_s=20.0, fps=fps, jitter_px=1.0))
    c_ns = _run_s_counter_on_dense(d_ns, fps)
    assert c_ns.s_count == 0

    # 11. 行波中途缺失 0.2s (<= gap_hold=0.3s) -> 计数仍为 10 ± 1
    gap_short = max(1, int(round(0.2 * fps)))
    mid_idx = len(d_fwd) // 2
    c_gap_short = _run_s_counter_on_dense(d_fwd, fps, missing_slice=(mid_idx, mid_idx + gap_short))
    assert abs(c_gap_short.s_count - 10) <= 1

    # 12. 行波中途缺失 1.0s (> gap_hold) -> run 被 abort，总数 <= 10 且 aborted >= 1
    gap_long = int(round(1.0 * fps))
    c_gap_long = _run_s_counter_on_dense(d_fwd, fps, missing_slice=(mid_idx, mid_idx + gap_long))
    assert c_gap_long.s_count <= 10
    assert c_gap_long.aborted >= 1
