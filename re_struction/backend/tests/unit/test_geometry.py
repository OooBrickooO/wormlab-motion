"""T05 单测：通用几何函数与 Pose.pts11。"""
from __future__ import annotations

import numpy as np
import pytest

from wormlab.geometry import (
    arc_length,
    iou_xyxy,
    polyline_length,
    resample_polyline,
    tangent_angles,
    wrap_pi,
)
from wormlab.schemas import N_DENSE, N_UI, Pose, Quality


def test_straight_line_geometry() -> None:
    pts = np.stack([np.linspace(0.0, 100.0, 21), np.zeros(21)], axis=1).astype(np.float32)
    assert polyline_length(pts) == pytest.approx(100.0, rel=1e-5)
    resampled = resample_polyline(pts, N_DENSE)
    assert resampled.shape == (N_DENSE, 2)
    assert resampled[0, 0] == pytest.approx(0.0, abs=1e-4)
    assert resampled[-1, 0] == pytest.approx(100.0, abs=1e-4)

    th = tangent_angles(resampled)
    assert th.shape == (N_DENSE - 1,)
    assert np.allclose(th, 0.0, atol=1e-5)


def test_semicircle_arc_length_and_tangents() -> None:
    r = 50.0
    t = np.linspace(0.0, np.pi, 500)
    pts = np.stack([r * np.cos(t), r * np.sin(t)], axis=1).astype(np.float32)
    expected_len = np.pi * r
    assert polyline_length(pts) == pytest.approx(expected_len, rel=1e-3)

    s = arc_length(pts)
    assert s[0] == pytest.approx(0.0)
    assert s[-1] == pytest.approx(expected_len, rel=1e-3)


def test_wrap_pi_and_iou() -> None:
    assert wrap_pi(0.0) == pytest.approx(0.0)
    assert wrap_pi(3.0 * np.pi) == pytest.approx(np.pi)
    assert wrap_pi(-np.pi) == pytest.approx(np.pi)
    arr = np.array([-3.0 * np.pi / 2.0, 3.0 * np.pi / 2.0])
    w = wrap_pi(arr)
    assert np.allclose(w, [np.pi / 2.0, -np.pi / 2.0], atol=1e-6)

    box_a = np.array([0.0, 0.0, 10.0, 10.0])
    box_b = np.array([5.0, 0.0, 15.0, 10.0])
    # inter=50, union=150 -> 1/3
    assert iou_xyxy(box_a, box_b) == pytest.approx(1.0 / 3.0, rel=1e-5)
    assert iou_xyxy(box_a, np.array([20.0, 20.0, 30.0, 30.0])) == pytest.approx(0.0)


def test_pose_pts11_property() -> None:
    dense = np.stack([np.linspace(10.0, 110.0, N_DENSE), np.linspace(20.0, 70.0, N_DENSE)], axis=1).astype(np.float32)
    pose = Pose(dense=dense, length=111.8, chord=111.8, width=6.0, area=650.0, quality=Quality.OK)
    pts11 = pose.pts11
    assert pts11 is not None
    assert pts11.shape == (N_UI, 2)
    assert np.allclose(pts11[0], [10.0, 20.0], atol=1e-4)
    assert np.allclose(pts11[-1], [110.0, 70.0], atol=1e-4)
