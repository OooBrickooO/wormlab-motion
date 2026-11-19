"""T07 单测：合成直线/弧/S/C/Ω 掩码的中心线提取、端帽精度与 loop 标记验证。"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from wormlab.config import load_config
from wormlab.geometry import polyline_length, resample_polyline
from wormlab.pose.centerline import extract_pose
from wormlab.pose.mask_refine import refine_mask
from wormlab.schemas import N_DENSE, Quality


def _render_curve_mask(pts: np.ndarray, width_px: float = 8.0, pad: int = 16) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    """将给定中心线渲染为二值掩码（圆帽），返回 (mask, origin, global_pts33)。"""
    pts33 = resample_polyline(pts, N_DENSE)
    x0 = int(np.floor(pts33[:, 0].min())) - pad
    y0 = int(np.floor(pts33[:, 1].min())) - pad
    x1 = int(np.ceil(pts33[:, 0].max())) + pad
    y1 = int(np.ceil(pts33[:, 1].max())) + pad
    w = max(32, x1 - x0)
    h = max(32, y1 - y0)

    canvas = np.zeros((h, w), dtype=np.uint8)
    local_pts = np.round(pts33 - np.array([[x0, y0]], dtype=np.float32)).astype(np.int32)
    thickness = max(2, int(round(width_px)))
    cv2.polylines(canvas, [local_pts.reshape(-1, 1, 2)], isClosed=False, color=255, thickness=thickness, lineType=cv2.LINE_8)
    return canvas > 0, (x0, y0), pts33


def _endpoint_error(pred_dense: np.ndarray, gt_dense: np.ndarray) -> float:
    """计算预测中心线两端点与真值两端点的最小配对误差（消除无先验时的正反方向）。"""
    err_fwd = max(
        float(np.linalg.norm(pred_dense[0] - gt_dense[0])),
        float(np.linalg.norm(pred_dense[-1] - gt_dense[-1])),
    )
    err_rev = max(
        float(np.linalg.norm(pred_dense[0] - gt_dense[-1])),
        float(np.linalg.norm(pred_dense[-1] - gt_dense[0])),
    )
    return min(err_fwd, err_rev)


@pytest.mark.parametrize("shape_kind", ["line", "arc", "s_wave", "c_shape", "omega"])
def test_extract_pose_shapes_accuracy(shape_kind: str) -> None:
    cfg = load_config("crawl_agar").pose
    s = np.linspace(0.0, 1.0, 200)
    L_true = 140.0
    ds = L_true / (len(s) - 1)

    if shape_kind == "line":
        theta = np.zeros_like(s)
    elif shape_kind == "arc":
        theta = 1.2 * (s - 0.5)
    elif shape_kind == "s_wave":
        theta = 0.8 * np.sin(2.0 * np.pi * s)
    elif shape_kind == "c_shape":
        theta = 2.6 * (s - 0.5)
    elif shape_kind == "omega":
        theta = 2.1 * np.sin(np.pi * (s - 0.5))
    else:
        raise ValueError(shape_kind)

    xs = 200.0 + np.cumsum(np.cos(theta) * ds)
    ys = 200.0 + np.cumsum(np.sin(theta) * ds)
    gt_curve = np.stack([xs, ys], axis=1).astype(np.float32)

    mask, origin, gt_dense = _render_curve_mask(gt_curve, width_px=8.0)
    pose = extract_pose(mask, origin, cfg)

    assert pose.dense is not None
    assert pose.dense.shape == (N_DENSE, 2)
    gt_len = polyline_length(gt_dense)
    rel_err = abs(pose.length - gt_len) / gt_len

    tol = 0.08 if shape_kind in ("c_shape", "omega") else 0.05
    assert rel_err < tol, f"{shape_kind}: length={pose.length:.2f}, gt={gt_len:.2f}, rel_err={rel_err:.4f}"

    ep_err = _endpoint_error(pose.dense, gt_dense)
    assert ep_err < 2.0, f"{shape_kind}: endpoint error {ep_err:.2f} px >= 2.0 px"


def test_omega_self_touch_loop_flag() -> None:
    cfg = load_config("crawl_agar").pose
    # 构造首尾相接形成空洞的闭环 Ω 掩码
    canvas = np.zeros((160, 160), dtype=np.uint8)
    cv2.circle(canvas, (80, 80), 38, 255, thickness=9)
    mask = canvas > 0
    pose = extract_pose(mask, (10, 20), cfg)
    assert pose.dense is not None
    assert "loop" in pose.flags


def test_tiny_mask_and_upscaled_flag() -> None:
    cfg = load_config("swim_liquid").pose
    # 空掩码 / 小于 min_mask_px (12px, 08_SPEC §6 swim_liquid)
    tiny = np.zeros((30, 30), dtype=bool)
    tiny[10:12, 10:13] = True
    p_tiny = extract_pose(tiny, (0, 0), cfg)
    assert p_tiny.dense is None
    assert p_tiny.quality == Quality.MISSING
    assert "empty" in p_tiny.flags

    # 50px 小虫触发 upscaled
    pts = np.stack([np.linspace(10.0, 55.0, 50), np.full(50, 25.0)], axis=1).astype(np.float32)
    mask, origin, gt_dense = _render_curve_mask(pts, width_px=5.0)
    p_small = extract_pose(mask, origin, cfg)
    assert p_small.dense is not None
    assert "upscaled" in p_small.flags
    assert abs(p_small.length - 45.0) / 45.0 < 0.05
    assert _endpoint_error(p_small.dense, gt_dense) < 2.0


def test_mask_refine_interface() -> None:
    cfg_on = load_config("crawl_agar", overrides={"pose": {"mask_refine": {"enabled": True}}}).pose
    gray = np.full((60, 100), 40, dtype=np.uint8)
    mask = np.zeros((60, 100), dtype=bool)
    mask[20:32, 15:85] = True
    gray[21:31, 16:84] = 220
    refined = refine_mask(gray, mask, cfg_on)
    assert refined.shape == mask.shape
    assert refined.dtype == bool
    assert int(refined.sum()) > 0


def test_extract_pose_fills_shape_moments() -> None:
    """08_SPEC Workstream A: extract_pose 必须填充免骨架椭圆形状描述子（含 fallback PCA 路径）。"""
    import math

    cfg = load_config("crawl_agar").pose
    s = np.linspace(0.0, 1.0, 200)
    L_true = 140.0
    ds = L_true / (len(s) - 1)

    def _render(theta: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        xs = 200.0 + np.cumsum(np.cos(theta) * ds)
        ys = 200.0 + np.cumsum(np.sin(theta) * ds)
        gt_curve = np.stack([xs, ys], axis=1).astype(np.float32)
        mask, origin, _ = _render_curve_mask(gt_curve, width_px=8.0)
        return mask, origin

    # 1. 细直虫：ell_aspect 远小于 0.35，偏心率接近 1
    mask_line, origin_line = _render(np.zeros_like(s))
    p_line = extract_pose(mask_line, origin_line, cfg)
    assert p_line.dense is not None
    assert math.isfinite(p_line.ell_aspect)
    assert math.isfinite(p_line.ell_ecc)
    assert math.isfinite(p_line.ell_angle)
    assert math.isfinite(p_line.ell_skew_minor)
    assert p_line.ell_aspect < 0.35
    assert p_line.ell_ecc > 0.9

    # 2. 深弯 C 虫（约 229° 总转角）：等效椭圆短长轴比显著跃升，ell_aspect > 0.50
    mask_c, origin_c = _render(4.0 * (s - 0.5))
    p_c = extract_pose(mask_c, origin_c, cfg)
    assert p_c.dense is not None
    assert p_c.ell_aspect > 0.50

    # 3. 180° 半圆 C 虫：解析值 b/a = sqrt((1/2-1/π)/(1/2)) ≈ 0.44（忽略端帽），显著高于直虫
    mask_semi, origin_semi = _render(np.pi * (s - 0.5))
    p_semi = extract_pose(mask_semi, origin_semi, cfg)
    assert p_semi.dense is not None
    assert p_semi.ell_aspect > 0.40

    # 4. 极小掩码 (< 6px)：形状描述子回退 NaN
    tiny = np.zeros((30, 30), dtype=bool)
    tiny[5, 5:7] = True
    p_tiny2 = extract_pose(tiny, (0, 0), cfg)
    assert not math.isfinite(p_tiny2.ell_aspect)
