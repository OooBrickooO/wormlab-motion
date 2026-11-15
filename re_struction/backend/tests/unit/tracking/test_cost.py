"""T18 单测：外观特征提取与各项关联代价函数与旧实现数值对拍 (容差 1e-5)。"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
LEGACY_BACKEND = REPO_ROOT / "backend"
if str(LEGACY_BACKEND) not in sys.path:
    sys.path.insert(0, str(LEGACY_BACKEND))

import worm_processing as wp  # type: ignore[import-not-found]  # noqa: E402
from wormlab.tracking import cost as new_cost  # noqa: E402
from wormlab.tracking.features import extract_features  # noqa: E402
from wormlab.tracking.track import Track  # noqa: E402


def test_features_and_cost_parity_with_legacy() -> None:
    old_tracker = wp.EnhancedWormTracker(fps=30, frame_size=(640, 480))

    # 构造带椭圆虫体的整帧掩码与裁剪掩码
    full_mask = np.zeros((480, 640), dtype=bool)
    canvas = np.zeros((480, 640), dtype=np.uint8)
    cv2.ellipse(canvas, (200, 150), (35, 8), 25.0, 0, 360, 255, -1)
    full_mask = canvas > 0

    bbox = np.array([165.0, 130.0, 235.0, 170.0], dtype=np.float32)
    x0, y0 = 150, 115
    mask_crop = full_mask[y0:185, x0:250]
    bbox_in_crop = bbox - np.array([x0, y0, x0, y0], dtype=np.float32)

    feat_old = old_tracker.extract_features(full_mask, bbox)
    feat_new = extract_features(mask_crop, bbox_in_crop)
    assert np.allclose(feat_old, feat_new, atol=1e-5)

    # 对拍 IoU、中心距离、运动一致性与方向代价
    b2 = np.array([175.0, 135.0, 248.0, 176.0], dtype=np.float32)
    assert np.isclose(old_tracker.compute_iou(bbox, b2), new_cost.iou(bbox, b2), atol=1e-5)
    assert np.isclose(old_tracker.compute_center_distance(bbox, b2), new_cost.center_dist(bbox, b2), atol=1e-5)

    t_old = wp.KalmanBoxTracker(bbox, feat_old, track_id=1)
    t_new = Track(track_id=1, bbox=bbox, uid=1, first_frame=0, feature=feat_new)
    for step_b in (
        np.array([168.0, 132.0, 238.0, 172.0], dtype=np.float32),
        np.array([172.0, 134.0, 242.0, 174.0], dtype=np.float32),
    ):
        t_old.predict()
        t_new.predict()
        t_old.update(step_b, feat_old)
        t_new.update(step_b, feat_new)

    t_old.predict()
    t_new.predict()

    mc_old = old_tracker.compute_motion_consistency_cost(t_old, b2)
    mc_new = new_cost.motion_cost(t_new, b2, max_speed=50.0)
    assert np.isclose(mc_old, mc_new, atol=1e-5)

    det_c = ((b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0)
    dc_old = old_tracker.compute_direction_cost(t_old, det_c)
    dc_new = new_cost.direction_cost(t_new, det_c)
    assert np.isclose(dc_old, dc_new, atol=1e-5)
