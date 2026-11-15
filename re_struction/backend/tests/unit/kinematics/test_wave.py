"""T12 单测：验证 wave_gate 对全身行波、驻波、仅头部试探摆动及小噪声的判别。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

RE_ROOT = Path(__file__).resolve().parents[4]
if str(RE_ROOT) not in sys.path:
    sys.path.insert(0, str(RE_ROOT))

from tools.make_synthetic import Scn, make_series  # noqa: E402
from wormlab.config import load_config  # noqa: E402
from wormlab.kinematics.features import compute_features  # noqa: E402
from wormlab.kinematics. modes import eigenworm_basis  # noqa: E402
from wormlab.kinematics.wave import wave_gate  # noqa: E402


def _eval_gate_on_scn(kind: str, A: float = 0.8) -> tuple[bool, float, float, float, float]:
    cfg = load_config("crawl_agar").kinematics
    scn = Scn(kind=kind, duration_s=3.0, fps=30.0, A=A, f=0.5)
    dense_seq, _ = make_series(scn)
    feats = [compute_features(d, cfg) for d in dense_seq]
    res = wave_gate(feats, cfg.wave, expected_len=len(feats))
    return res.ok, res.eps, res.r_med, res.amp_ant, res.amp_post


def test_wave_gate_traveling_vs_standing_vs_head_wiggle() -> None:
    # 1. 全身行波 -> ok=True, 圆度 ε 接近 1
    ok_fwd, eps_fwd, r_fwd, _, _ = _eval_gate_on_scn("crawl_fwd", A=0.8)
    assert ok_fwd is True
    assert eps_fwd > 0.85
    assert r_fwd > 0.70

    # 2. 驻波 (线性往复) -> ε 接近 0, ok=False
    ok_std, eps_std, _, _, _ = _eval_gate_on_scn("standing", A=0.8)
    assert ok_std is False
    assert eps_std < 0.15

    # 3. 只有头部试探摆动 (s < 0.25) -> 后半身振幅接近 0, ok=False
    ok_hw, _, _, amp_ant_hw, amp_post_hw = _eval_gate_on_scn("head_wiggle")
    assert ok_hw is False
    assert amp_ant_hw > 0.15
    assert amp_post_hw < 0.05

    # 4. 小幅度噪声 (< r_min) -> ok=False
    ok_small, _, r_small, _, _ = _eval_gate_on_scn("crawl_fwd", A=0.10)
    assert ok_small is False
    assert r_small < 0.20

    with pytest.raises(NotImplementedError):
        eigenworm_basis()
