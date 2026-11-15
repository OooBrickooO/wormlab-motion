"""T06 单测与集成自检：Detector、filter_detections、apply_nms 及 Swim-4 首帧检测。"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from wormlab.config import load_config
from wormlab.detection import Detector, apply_nms, filter_detections
from wormlab.schemas import Detection


def _make_fake_result(boxes: np.ndarray, confs: np.ndarray, masks: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=torch.from_numpy(boxes.astype(np.float32)),
            conf=torch.from_numpy(confs.astype(np.float32)),
        ),
        masks=SimpleNamespace(data=torch.from_numpy(masks.astype(np.float32))),
    )


def test_filter_and_nms_boundaries() -> None:
    cfg = load_config("crawl_agar")
    frame_wh = (1920, 1080)  # min_area = 1920*1080*0.000015 = 31.104

    d_small = Detection(bbox=np.array([0, 0, 4, 4], dtype=np.float32), conf=0.9, mask=None, mask_origin=(0, 0), det_idx=0)
    d_low_conf = Detection(bbox=np.array([0, 0, 50, 50], dtype=np.float32), conf=0.15, mask=None, mask_origin=(0, 0), det_idx=1)
    d_good_hi = Detection(bbox=np.array([100, 100, 160, 160], dtype=np.float32), conf=0.85, mask=None, mask_origin=(0, 0), det_idx=2)
    d_good_dup = Detection(bbox=np.array([105, 105, 162, 162], dtype=np.float32), conf=0.60, mask=None, mask_origin=(0, 0), det_idx=3)
    d_good_2 = Detection(bbox=np.array([400, 400, 460, 460], dtype=np.float32), conf=0.75, mask=None, mask_origin=(0, 0), det_idx=4)

    filt = filter_detections([d_small, d_low_conf, d_good_hi, d_good_dup, d_good_2], frame_wh, cfg)
    assert len(filt) == 3

    kept = apply_nms(filt, iou_thr=0.3)
    assert len(kept) == 2
    assert kept[0].conf == 0.85
    assert kept[1].conf == 0.75
    assert [d.det_idx for d in kept] == [0, 1]


def test_detector_with_fake_model_and_tiled() -> None:
    cfg = load_config("crawl_agar")
    h, w = 200, 300
    mask_full = np.zeros((2, h, w), dtype=np.float32)
    mask_full[0, 50:80, 60:110] = 1.0
    mask_full[1, 52:82, 62:112] = 1.0  # 重叠框，应被 NMS 抑制

    boxes = np.array([[60, 50, 110, 80], [62, 52, 112, 82]], dtype=np.float32)
    confs = np.array([0.9, 0.5], dtype=np.float32)

    def fake_model(frames, **kwargs):
        return [_make_fake_result(boxes, confs, mask_full) for _ in frames]

    det = Detector("dummy.pt", cfg, device="cpu", _model_override=fake_model)
    frm = np.zeros((h, w, 3), dtype=np.uint8)
    out = det.infer([frm])
    assert len(out) == 1
    assert len(out[0]) == 1
    d0 = out[0][0]
    assert d0.mask is not None
    assert d0.mask.dtype == bool
    # 裁剪后的掩码尺寸远小于整帧
    assert d0.mask.shape[0] < h and d0.mask.shape[1] < w
    assert int(d0.mask.sum()) == 30 * 50


def test_detector_swim4_frame0_real_model() -> None:
    re_root = Path(__file__).resolve().parents[4]
    vid_path = re_root / "referAnswer" / "Swim-4.mp4"
    model_path = re_root / "backend" / "models" / "origin.pt"

    cap = cv2.VideoCapture(str(vid_path))
    ok, frame0 = cap.read()
    cap.release()
    assert ok and frame0 is not None

    cfg = load_config("swim_liquid")
    detector = Detector(str(model_path), cfg)
    dets_batch = detector.infer([frame0])
    assert len(dets_batch) == 1
    assert len(dets_batch[0]) == 1
    assert dets_batch[0][0].mask is not None
    assert dets_batch[0][0].conf >= cfg.detection.min_confidence


def test_tiled_seam_unclipped_beats_clipped_and_infer_roi() -> None:
    from wormlab.config import load_config

    cfg = load_config(
        "swim_liquid",
        overrides={
            "detection": {
                "tile": {"enabled": True, "size": 200, "overlap": 0.25, "infer_imgsz": 160, "seam_ios_thresh": 0.55},
                "min_area_ratio": 0.0001,
            }
        },
    )
    seen_imgsz: list[int] = []

    def fake_model(crops, **kwargs):
        seen_imgsz.append(int(kwargs.get("imgsz", 0)))
        out = []
        # 200x300 宽度配合 tile=200 overlap=0.25 生成两个水平 tile：x0=0 和 x0=100
        for idx, _ in enumerate(crops):
            m = np.zeros((1, 200, 200), dtype=np.float32)
            if idx == 0:
                # 瓦片 0 上在右边缘 x=199 截断，置信度较高 0.90
                m[0, 80:110, 150:200] = 1.0
                boxes = np.array([[150, 80, 200, 110]], dtype=np.float32)
                confs = np.array([0.90], dtype=np.float32)
            else:
                # 瓦片 1 (x0=100) 覆盖完整蠕虫 x=150..240，置信度略低 0.72
                m[0, 80:110, 50:140] = 1.0
                boxes = np.array([[50, 80, 140, 110]], dtype=np.float32)
                confs = np.array([0.72], dtype=np.float32)
            out.append(_make_fake_result(boxes, confs, m))
        return out

    det = Detector("dummy.pt", cfg, device="cpu", _model_override=fake_model)
    frm = np.zeros((200, 300, 3), dtype=np.uint8)
    dets = det.infer([frm])[0]
    assert seen_imgsz == [160]
    assert len(dets) == 1
    # 未被截断的瓦片 1 结果（x2=240）必须胜过更高 conf 的截断半条虫（x2=200）
    assert float(dets[0].bbox[2]) >= 235.0
    assert abs(float(dets[0].conf) - 0.72) < 1e-4

    # ROI 推理验证全局坐标偏移正确
    roi_dets = det.infer_roi(frm, center_xy=(190.0, 95.0), roi_size=120, min_confidence=0.12)
    assert len(roi_dets) == 1
    assert float(roi_dets[0].bbox[0]) >= 130.0
