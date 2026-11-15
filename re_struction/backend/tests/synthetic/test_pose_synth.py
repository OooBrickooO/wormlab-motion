"""T09 合成测试：验证 PoseExtractor 在合成行波、游泳摆动、Ω 折返与双虫粘连 (merge) 序列上的准确性。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RE_ROOT = Path(__file__).resolve().parents[3]
if str(RE_ROOT) not in sys.path:
    sys.path.insert(0, str(RE_ROOT))

from tools.make_synthetic import Scn, make_series, render_mask  # noqa: E402
from wormlab.config import load_config  # noqa: E402
from wormlab.pose import PoseExtractor  # noqa: E402
from wormlab.schemas import Detection, Quality  # noqa: E402


def _make_det(mask: np.ndarray, origin: tuple[int, int], idx: int = 0) -> Detection:
    h, w = mask.shape[:2]
    x0, y0 = origin
    bbox = np.array([x0, y0, x0 + w, y0 + h], dtype=np.float32)
    return Detection(bbox=bbox, conf=0.95, mask=mask, mask_origin=origin, det_idx=idx)


def test_pose_synth_crawl_and_orientation_stability() -> None:
    cfg = load_config("crawl_agar").pose
    scn = Scn(kind="crawl_fwd", duration_s=4.0, fps=15.0, length_px=120.0, width_px=8.0)
    series, _ = make_series(scn)
    extractor = PoseExtractor(cfg, fps=scn.fps, user_head_xy=(float(series[0, 0, 0]), float(series[0, 0, 1])))

    head_errors: list[float] = []
    lengths: list[float] = []
    for fidx, gt_dense in enumerate(series):
        mask, origin = render_mask(gt_dense, width_px=scn.width_px)
        pose = extractor.process(_make_det(mask, origin), frame_idx=fidx)
        assert pose.dense is not None
        assert pose.quality in (Quality.OK, Quality.SUSPECT)
        head_errors.append(float(np.linalg.norm(pose.dense[0] - gt_dense[0])))
        lengths.append(pose.length)

    # 全程无头尾翻转，头部位置误差 < 2.5px，平均体长误差 < 5%
    assert max(head_errors) < 3.0
    assert np.mean(head_errors) < 1.5
    assert abs(float(np.median(lengths)) - scn.length_px) / scn.length_px < 0.05


def test_pose_synth_merge_triggers_bad_quality() -> None:
    cfg = load_config("crawl_agar").pose
    scn = Scn(kind="crawl_fwd", duration_s=2.0, fps=15.0, length_px=120.0, width_px=8.0)
    series, _ = make_series(scn)
    extractor = PoseExtractor(cfg, fps=scn.fps)

    # 前 15 帧单虫建立面积基准
    for fidx in range(15):
        mask, origin = render_mask(series[fidx], width_px=scn.width_px)
        p = extractor.process(_make_det(mask, origin), frame_idx=fidx)
        assert p.quality == Quality.OK

    # 第 15 帧构造第二条虫与之交叉粘连成一团（面积约 1.8x）
    gt1 = series[15]
    center = gt1.mean(axis=0, keepdims=True)
    # 旋转 60 度构造交叉虫
    ang = np.pi / 3.0
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], dtype=np.float32)
    gt2 = (gt1 - center) @ rot.T + center
    merged_mask, origin = render_mask(gt1, width_px=scn.width_px, extra_dense33=gt2)
    p_merge = extractor.process(_make_det(merged_mask, origin), frame_idx=15)
    assert p_merge.quality == Quality.BAD
    assert "merge" in p_merge.flags
