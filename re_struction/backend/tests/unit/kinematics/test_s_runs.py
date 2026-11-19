"""T13 单测：对照 proto_kinematics.py 在 30 fps 与 10 fps 下覆盖全部 12 类合成场景。"""
from __future__ import annotations

from dataclasses import replace

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
        g = (
            wave_gate(win, cfg.wave, expected_len=2 * half + 1, fps=fps)
            if f is not None
            else wave_gate([], cfg.wave, expected_len=2 * half + 1, fps=fps)
        )
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


def test_s_run_mid_pause_settle_partial() -> None:
    cfg = load_config("crawl_agar").kinematics.s_run
    fps = 30.0
    counter = SRunCounter(cfg, fps=fps)
    u = np.array([1.0, 0.0], dtype=np.float32)
    L_ref = 100.0
    # 模拟推进 0.85 个周期（5.34 rad）后停顿，顺向推进 12px (> 0.08 * 0.65 * 100 = 5.2px)
    n_steps = 40
    for i in range(n_steps):
        phi = -np.pi + (i / (n_steps - 1)) * (1.7 * np.pi)
        phi_wrapped = float(np.arctan2(np.sin(phi), np.cos(phi)))
        center = np.array([12.0 * i / (n_steps - 1), 0.0], dtype=np.float32)
        counter.step(i=i, gate_ok=True, phi=phi_wrapped, center=center, u=u, L_ref=L_ref)
    assert counter.s_count == 0
    settled = counter.settle_partial(n_steps, L_ref=L_ref, tail_frac=0.65)
    assert settled is True
    assert counter.s_count == 1


def test_s_run_freeze_bridges_short_gap_without_dropping_phase() -> None:
    """08_SPEC S3: 2 周期爬行中间插入 3 次 0.15s freeze 闪烁，相位应被桥接而非丢弃。"""
    cfg = load_config("crawl_agar").kinematics.s_run
    assert cfg.phase_bridge_s >= 0.15
    fps = 30.0
    counter = SRunCounter(cfg, fps=fps)
    u = np.array([1.0, 0.0], dtype=np.float32)
    L_ref = 100.0

    total_frames = int(4.4 * fps)  # 0.5 Hz -> 2.2 个完整周期（留出完整结算余量）
    freeze_len = max(1, int(round(0.15 * fps)))
    freeze_starts = [30, 60, 90]

    for i in range(total_frames):
        if any(s <= i < s + freeze_len for s in freeze_starts):
            counter.freeze()
            continue
        t = i / fps
        phi = 2.0 * np.pi * 0.5 * t
        phi_w = float(np.arctan2(np.sin(phi), np.cos(phi)))
        # 每周期沿体轴推进 0.4 * L (40px >> min_disp_ratio * L_ref = 8px)
        center = np.array([40.0 * 0.5 * t, 0.0], dtype=np.float32)
        counter.step(i=i, gate_ok=True, phi=phi_w, center=center, u=u, L_ref=L_ref)

    assert counter.s_count == 2
    assert counter.rejected["no_disp"] == 0
    assert counter.rejected["wrong_dir"] == 0


def test_s_run_curved_path_displacement() -> None:
    """08_SPEC S4: 沿强弯曲圆弧爬行 2 周期，use_path_disp=True 时切向路径积分准确计数。"""
    cfg = load_config("crawl_agar").kinematics.s_run
    assert cfg.use_path_disp is True
    fps = 30.0
    L_ref = 100.0
    v = 20.0  # px/s -> 每周期 (0.5Hz) 推进 40px 弧长
    radius = 7.0  # 每周期转 40/7 ≈ 5.7 rad，端点直线弦仅 ≈4px (< min_disp_ratio*L=8px)

    def _make_counter() -> SRunCounter:
        return SRunCounter(cfg, fps=fps)

    def _run(counter: SRunCounter) -> SRunCounter:
        total_frames = int(4.4 * fps)  # 2.2 个完整周期
        for i in range(total_frames):
            t = i / fps
            theta_arc = v * t / radius
            cx, cy = 300.0 + radius * np.cos(theta_arc), 300.0 + radius * np.sin(theta_arc)
            center = np.array([cx, cy], dtype=np.float32)
            # 体轴指向路径前进切线方向（逆时针圆弧）
            tangent = np.array([-np.sin(theta_arc), np.cos(theta_arc)], dtype=np.float32)
            phi = 2.0 * np.pi * 0.5 * t
            phi_w = float(np.arctan2(np.sin(phi), np.cos(phi)))
            counter.step(
                i=i,
                gate_ok=True,
                phi=phi_w,
                center=center,
                u=tangent,
                L_ref=L_ref,
                tangent_mid=tangent,
            )
        return counter

    counter = _run(_make_counter())
    assert counter.s_count == 2
    assert counter.rejected["no_disp"] == 0

    # 对照组：旧版首尾直线弦投影在强弯弧下会把每周期位移压到阈值之下
    cfg_chord = replace(cfg, use_path_disp=False)
    counter_chord = _run(SRunCounter(cfg_chord, fps=fps))
    assert counter_chord.s_count <= 1
    assert counter_chord.rejected["no_disp"] >= 1
