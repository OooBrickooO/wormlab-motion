"""T21 (04 §7): 新旧 Tracker 在真实检测流 (300 帧 × 3 视频) 上的回放对拍集成测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import supervision as sv

ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "backend"))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import worm_processing as legacy_wp  # noqa: E402
from dump_detections import unpack_detections  # noqa: E402

from wormlab.config import TrackingCfg  # noqa: E402
from wormlab.schemas import Detection  # noqa: E402
from wormlab.tracking.cost import iou  # noqa: E402
from wormlab.tracking.tracker import Tracker  # noqa: E402


def _restore_full_masks(dets: list[Detection], width: int, height: int) -> np.ndarray | None:
    if not dets:
        return None
    full_masks = np.zeros((len(dets), height, width), dtype=np.uint8)
    for idx, d in enumerate(dets):
        if d.mask is not None and d.mask.size > 0:
            ox, oy = int(d.mask_origin[0]), int(d.mask_origin[1])
            mh, mw = d.mask.shape[:2]
            y1 = max(0, oy)
            x1 = max(0, ox)
            y2 = min(height, oy + mh)
            x2 = min(width, ox + mw)
            if y2 > y1 and x2 > x1:
                full_masks[idx, y1:y2, x1:x2] = (d.mask[: y2 - y1, : x2 - x1] > 0).astype(np.uint8)
    return full_masks


@pytest.mark.slow
@pytest.mark.parametrize("video_id", ["10x", "1.6x-1", "Move-3"])
def test_tracker_replay_parity(video_id: str) -> None:
    cache_path = ROOT / "data" / "cache" / f"dets_{video_id}.npz"
    if not cache_path.exists():
        pytest.skip(f"未找到检测缓存 {cache_path}，请先运行 tools/dump_detections.py")

    frames_dets, fps, width, height = unpack_detections(cache_path)
    cfg = TrackingCfg()

    # 实例化旧跟踪器与新跟踪器
    old_tracker = legacy_wp.EnhancedWormTracker(
        fps=fps,
        frame_size=(width, height),
        sub_video_dir=None,
        max_worm_id=None,
        head_config=[],
    )
    # 仅对拍纯跟踪器（禁用旧跟踪器内部耦合的耗时骨架化，使两者均测试跟踪+外观特征+卡尔曼关联）
    old_tracker._extract_centerline_in_window = lambda *args, **kwargs: (None, None)

    new_tracker = Tracker(
        cfg=cfg,
        fps=fps,
        frame_wh=(width, height),
        max_worm_id=None,
        head_config=[],
    )

    total_assignments = 0
    matched_assignments = 0

    for f_idx, dets in enumerate(frames_dets):
        if len(dets) > 0:
            xyxy = np.vstack([d.bbox for d in dets]).astype(np.float32)
            confs = np.array([d.conf for d in dets], dtype=np.float32)
            sv_dets = sv.Detections(xyxy=xyxy, confidence=confs)
            full_masks = _restore_full_masks(dets, width, height)
        else:
            sv_dets = sv.Detections.empty()
            full_masks = None

        old_out = old_tracker.update_tracking(sv_dets, full_masks, f_idx, frame=None)
        new_out = new_tracker.update(dets, frame_idx=f_idx)

        old_boxes = np.asarray(old_out.xyxy, dtype=np.float32) if len(old_out) > 0 else np.zeros((0, 4), dtype=np.float32)
        old_tids = (
            np.asarray(old_out.tracker_id, dtype=int)
            if (len(old_out) > 0 and old_out.tracker_id is not None)
            else np.zeros(0, dtype=int)
        )

        # 逐目标对比新旧 bbox -> track_id 分配
        for n_item in new_out:
            total_assignments += 1
            best_iou = 0.0
            best_old_tid = -1
            for ob, otid in zip(old_boxes, old_tids):
                ov = iou(n_item.bbox, ob)
                if ov > best_iou:
                    best_iou = ov
                    best_old_tid = int(otid)
            if best_iou >= 0.85 and best_old_tid == n_item.track_id:
                matched_assignments += 1

    agreement = matched_assignments / max(1, total_assignments)
    assert agreement >= 0.99, f"{video_id} 新旧跟踪器 ID 一致率 {agreement:.4f} < 0.99 ({matched_assignments}/{total_assignments})"
