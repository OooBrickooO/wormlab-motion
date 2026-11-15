"""01 §4: 检测过滤与非极大值抑制 (NMS)。"""
from __future__ import annotations

from dataclasses import replace

from wormlab.config import Config, DetectionCfg
from wormlab.geometry import iou_xyxy
from wormlab.schemas import Detection


def filter_detections(
    dets: list[Detection],
    frame_wh: tuple[int, int],
    cfg: Config | DetectionCfg,
    min_confidence_override: float | None = None,
) -> list[Detection]:
    """按 bbox 面积比例与最小置信度过滤检测结果（可选 max_area_ratio 剔除培养皿反光等大面积误检）。"""
    det_cfg = cfg.detection if isinstance(cfg, Config) else cfg
    w, h = int(frame_wh[0]), int(frame_wh[1])
    frame_area = float(w * h)
    min_area = frame_area * float(det_cfg.min_area_ratio)
    max_ratio = getattr(det_cfg, "max_area_ratio", None)
    max_area = frame_area * float(max_ratio) if max_ratio is not None else float("inf")
    min_conf = float(min_confidence_override) if min_confidence_override is not None else float(det_cfg.min_confidence)

    kept: list[Detection] = []
    for d in dets:
        bw = max(0.0, float(d.bbox[2]) - float(d.bbox[0]))
        bh = max(0.0, float(d.bbox[3]) - float(d.bbox[1]))
        area = bw * bh
        if min_area <= area <= max_area and float(d.conf) >= min_conf:
            kept.append(d)
    return [replace(d, det_idx=idx) for idx, d in enumerate(kept)]


def apply_nms(dets: list[Detection], iou_thr: float = 0.3) -> list[Detection]:
    """按置信度降序执行 bbox IoU 非极大值抑制，并重排 det_idx。"""
    if len(dets) <= 1:
        return [replace(d, det_idx=idx) for idx, d in enumerate(dets)]

    sorted_dets = sorted(dets, key=lambda d: float(d.conf), reverse=True)
    kept: list[Detection] = []
    for cand in sorted_dets:
        if any(iou_xyxy(cand.bbox, prev.bbox) > iou_thr for prev in kept):
            continue
        kept.append(cand)
    return [replace(d, det_idx=idx) for idx, d in enumerate(kept)]
