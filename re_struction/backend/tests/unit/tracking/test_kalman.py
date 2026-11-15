"""T18 单测：KalmanBoxFilter 与旧实现 KalmanBoxTracker 数值对拍 (容差 1e-5)。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
LEGACY_BACKEND = REPO_ROOT / "backend"
if str(LEGACY_BACKEND) not in sys.path:
    sys.path.insert(0, str(LEGACY_BACKEND))

import worm_processing as wp  # type: ignore[import-not-found]  # noqa: E402
from wormlab.tracking.kalman import KalmanBoxFilter  # noqa: E402


def test_kalman_parity_with_legacy() -> None:
    rng = np.random.default_rng(123)
    init_bbox = np.array([100.0, 200.0, 160.0, 240.0], dtype=np.float32)

    old_kf = wp.KalmanBoxTracker(init_bbox, track_id=1)
    new_kf = KalmanBoxFilter(init_bbox)

    assert np.allclose(old_kf.get_state(), new_kf.get_state(), atol=1e-5)

    for step in range(25):
        p_old = old_kf.predict()
        p_new = new_kf.predict()
        assert np.allclose(p_old, p_new, atol=1e-5)

        if step % 3 != 0:
            shift = np.array([step * 2.5, step * 1.2, step * 2.5, step * 1.2], dtype=np.float32)
            noise = rng.normal(0.0, 1.5, size=4).astype(np.float32)
            meas_box = init_bbox + shift + noise
            old_kf.update(meas_box)
            new_kf.update(meas_box)

        assert np.allclose(old_kf.get_state(), new_kf.get_state(), atol=1e-5)
        assert np.allclose(old_kf.get_velocity(), new_kf.get_velocity(), atol=1e-5)
        assert np.allclose(old_kf.get_predicted_center(), new_kf.get_predicted_center(), atol=1e-5)
