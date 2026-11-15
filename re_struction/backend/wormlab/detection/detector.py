"""01 §4: YOLO 分割批推理、GPU 局部掩码裁剪、OOM 自动降级与可选切片推理。"""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import replace
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from wormlab.config import Config
from wormlab.geometry import iou_xyxy
from wormlab.schemas import Detection

from .filters import apply_nms, filter_detections

log = logging.getLogger(__name__)


def _mask_overlap_metrics(a: Detection, b: Detection) -> tuple[float, float]:
    """计算两个局部掩码在整帧坐标系下的 (mask_iou, mask_ios)。"""
    box_iou = iou_xyxy(a.bbox, b.bbox)
    if box_iou <= 0.0:
        return 0.0, 0.0
    if a.mask is None or b.mask is None:
        return box_iou, box_iou

    ax0, ay0 = a.mask_origin
    ah, aw = a.mask.shape[:2]
    ax1, ay1 = ax0 + aw, ay0 + ah

    bx0, by0 = b.mask_origin
    bh, bw = b.mask.shape[:2]
    bx1, by1 = bx0 + bw, by0 + bh

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0, 0.0

    sub_a = a.mask[iy0 - ay0 : iy1 - ay0, ix0 - ax0 : ix1 - ax0]
    sub_b = b.mask[iy0 - by0 : iy1 - by0, ix0 - bx0 : ix1 - bx0]
    inter = int(np.logical_and(sub_a, sub_b).sum())
    if inter == 0:
        return 0.0, 0.0
    area_a = max(1, int(a.mask.sum()))
    area_b = max(1, int(b.mask.sum()))
    union = max(1, area_a + area_b - inter)
    return float(inter / union), float(inter / min(area_a, area_b))


def _mask_iou_global(a: Detection, b: Detection) -> float:
    """计算两个局部掩码在整帧坐标系下的掩码 IoU（用于切片跨块合并）。"""
    return _mask_overlap_metrics(a, b)[0]


def _is_inner_tile_seam(
    bbox_global: np.ndarray,
    x0: int,
    y0: int,
    tile_size: int,
    frame_wh: tuple[int, int],
    tol_px: float = 2.0,
) -> bool:
    """判断全局坐标下的检测框是否贴近切片内部拼接边界（被瓦片边缘截断）。"""
    w_frm, h_frm = frame_wh
    x1, y1, x2, y2 = float(bbox_global[0]), float(bbox_global[1]), float(bbox_global[2]), float(bbox_global[3])
    if x0 > 0 and x1 <= float(x0) + tol_px:
        return True
    if y0 > 0 and y1 <= float(y0) + tol_px:
        return True
    if (x0 + tile_size) < w_frm and x2 >= float(x0 + tile_size) - tol_px:
        return True
    if (y0 + tile_size) < h_frm and y2 >= float(y0 + tile_size) - tol_px:
        return True
    return False


class Detector:
    """YOLO 实例分割推理器（无模块级全局状态）。"""

    def __init__(
        self,
        model_path: str,
        cfg: Config,
        device: str | None = None,
        _model_override: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self._infer_lock = threading.Lock()
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                log.warning("CUDA 不可用，Detector 回退至 CPU 运行。")
        else:
            self.device = device

        if _model_override is not None:
            self.model = _model_override
        else:
            self.model = YOLO(str(model_path))
            self.model.to(self.device)

        self.imgsz: int | None = None
        if isinstance(cfg.detection.imgsz, int):
            self.imgsz = int(cfg.detection.imgsz)
            if _model_override is None:
                self._warmup(self.imgsz)

    def _warmup(self, imgsz: int) -> None:
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        try:
            self.model(
                [dummy],
                verbose=False,
                device=self.device,
                retina_masks=True,
                imgsz=imgsz,
                conf=self.cfg.detection.min_confidence,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Detector warm-up 异常 (已忽略): %s", exc)

    def _ensure_imgsz(self, frame: np.ndarray) -> int:
        if self.imgsz is not None:
            return self.imgsz
        h, w = frame.shape[:2]
        if str(self.cfg.detection.imgsz).lower() == "native":
            self.imgsz = int(math.ceil(max(w, h) / 32.0) * 32)
        else:
            self.imgsz = int(self.cfg.detection.imgsz)
        self._warmup(self.imgsz)
        return self.imgsz

    def _run_model_with_oom_guard(
        self,
        frames: list[np.ndarray],
        imgsz: int,
        conf_thresh: float | None = None,
    ) -> list[Any]:
        conf = float(conf_thresh) if conf_thresh is not None else float(self.cfg.detection.min_confidence)
        try:
            with self._infer_lock:
                return list(
                    self.model(
                        frames,
                        verbose=False,
                        device=self.device,
                        retina_masks=True,
                        imgsz=imgsz,
                        conf=conf,
                    )
                )
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not self.cfg.detection.oom_fallback:
                raise
            if len(frames) > 1:
                mid = len(frames) // 2
                log.warning("CUDA OOM: batch=%d 拆分为 %d + %d 重试", len(frames), mid, len(frames) - mid)
                return self._run_model_with_oom_guard(frames[:mid], imgsz, conf) + self._run_model_with_oom_guard(
                    frames[mid:], imgsz, conf
                )
            if imgsz > 416:
                new_sz = max(416, (imgsz // 2 // 32) * 32)
                log.warning("CUDA OOM (batch=1): imgsz 从 %d 降至 %d 重试", imgsz, new_sz)
                self.imgsz = new_sz
                return self._run_model_with_oom_guard(frames, new_sz, conf)
            raise

    def _parse_single_result(self, res: Any, frame_wh: tuple[int, int], offset_xy: tuple[int, int] = (0, 0)) -> list[Detection]:
        w_frm, h_frm = frame_wh
        ox, oy = offset_xy
        if res.boxes is None or len(res.boxes.xyxy) == 0:
            return []

        boxes_t = res.boxes.xyxy
        confs_t = res.boxes.conf
        boxes = boxes_t.detach().cpu().numpy().astype(np.float32) if hasattr(boxes_t, "detach") else np.asarray(boxes_t, dtype=np.float32)
        confs = confs_t.detach().cpu().numpy().astype(np.float32) if hasattr(confs_t, "detach") else np.asarray(confs_t, dtype=np.float32)
        masks_t = res.masks.data if (getattr(res, "masks", None) is not None and res.masks.data is not None) else None

        margin_ratio = float(self.cfg.detection.mask_margin_ratio)
        raw_dets: list[Detection] = []
        n = len(boxes)

        for i in range(n):
            b_local = boxes[i]
            bw = max(1.0, float(b_local[2] - b_local[0]))
            bh = max(1.0, float(b_local[3] - b_local[1]))
            margin = max(8, int(round(margin_ratio * max(bw, bh))))

            mask_crop: np.ndarray | None = None
            origin_x = int(max(0, math.floor(float(b_local[0])) - margin)) + ox
            origin_y = int(max(0, math.floor(float(b_local[1])) - margin)) + oy

            if masks_t is not None and i < len(masks_t):
                mh, mw = int(masks_t.shape[1]), int(masks_t.shape[2])
                lx0 = max(0, int(math.floor(float(b_local[0]))) - margin)
                ly0 = max(0, int(math.floor(float(b_local[1]))) - margin)
                lx1 = min(mw, int(math.ceil(float(b_local[2]))) + margin)
                ly1 = min(mh, int(math.ceil(float(b_local[3]))) + margin)
                # 在 GPU 上直接切片裁剪后再搬回 CPU
                sub_t = masks_t[i, ly0:ly1, lx0:lx1] > 0.5
                mask_crop = sub_t.detach().cpu().numpy().astype(bool) if hasattr(sub_t, "detach") else np.asarray(sub_t, dtype=bool)
                origin_x = lx0 + ox
                origin_y = ly0 + oy

            b_global = np.array(
                [
                    np.clip(b_local[0] + ox, 0.0, float(w_frm)),
                    np.clip(b_local[1] + oy, 0.0, float(h_frm)),
                    np.clip(b_local[2] + ox, 0.0, float(w_frm)),
                    np.clip(b_local[3] + oy, 0.0, float(h_frm)),
                ],
                dtype=np.float32,
            )
            raw_dets.append(
                Detection(
                    bbox=b_global,
                    conf=float(confs[i]),
                    mask=mask_crop,
                    mask_origin=(origin_x, origin_y),
                    det_idx=i,
                )
            )
        return raw_dets

    def _resolve_tile_imgsz(self) -> int:
        t_cfg = self.cfg.detection.tile
        if getattr(t_cfg, "infer_imgsz", None) is not None:
            return int(t_cfg.infer_imgsz)
        if self.imgsz is not None:
            return int(self.imgsz)
        cfg_sz = self.cfg.detection.imgsz
        return int(cfg_sz) if isinstance(cfg_sz, int) else 416

    def _infer_tiled(self, frame: np.ndarray, min_confidence: float | None = None) -> list[Detection]:
        h, w = frame.shape[:2]
        t_cfg = self.cfg.detection.tile
        tsize = int(t_cfg.size)
        stride = max(1, int(round(tsize * (1.0 - float(t_cfg.overlap)))))

        xs = list(range(0, max(1, w - tsize + 1), stride))
        if not xs or xs[-1] + tsize < w:
            xs.append(max(0, w - tsize))
        ys = list(range(0, max(1, h - tsize + 1), stride))
        if not ys or ys[-1] + tsize < h:
            ys.append(max(0, h - tsize))

        crops: list[np.ndarray] = []
        offsets: list[tuple[int, int]] = []
        for y0 in ys:
            for x0 in xs:
                crop = frame[y0 : y0 + tsize, x0 : x0 + tsize]
                if crop.shape[0] < tsize or crop.shape[1] < tsize:
                    pad_b = max(0, tsize - crop.shape[0])
                    pad_r = max(0, tsize - crop.shape[1])
                    crop = cv2.copyMakeBorder(crop, 0, pad_b, 0, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                crops.append(crop)
                offsets.append((x0, y0))

        infer_sz = self._resolve_tile_imgsz()
        results = self._run_model_with_oom_guard(crops, infer_sz, conf_thresh=min_confidence)
        # 在各瓦片独立过滤置信度与面积，同时保留该候选是否贴近瓦片内部拼缝的标记
        cand_items: list[tuple[bool, Detection]] = []
        for res, off in zip(results, offsets):
            tile_dets = filter_detections(
                self._parse_single_result(res, (w, h), offset_xy=off),
                (w, h),
                self.cfg,
                min_confidence_override=min_confidence,
            )
            for d in tile_dets:
                clipped = _is_inner_tile_seam(d.bbox, off[0], off[1], tsize, (w, h))
                cand_items.append((not clipped, d))

        # 完整未截断候选优先于边界截断候选，同优先级按置信度降序
        cand_items.sort(key=lambda pair: (pair[0], float(pair[1].conf)), reverse=True)
        seam_ios_thr = float(getattr(t_cfg, "seam_ios_thresh", 0.55))
        nms_iou_thr = float(self.cfg.detection.nms_iou)

        kept: list[Detection] = []
        for _, cand in cand_items:
            suppress = False
            for prev in kept:
                if iou_xyxy(cand.bbox, prev.bbox) > nms_iou_thr:
                    suppress = True
                    break
                m_iou, m_ios = _mask_overlap_metrics(cand, prev)
                if m_iou > 0.5 or m_ios > seam_ios_thr:
                    suppress = True
                    break
            if not suppress:
                kept.append(cand)
        return [replace(d, det_idx=idx) for idx, d in enumerate(kept)]

    def infer_roi(
        self,
        frame: np.ndarray,
        center_xy: tuple[float, float],
        roi_size: int = 416,
        min_confidence: float = 0.12,
    ) -> list[Detection]:
        """以预测质心为中心裁剪固定尺寸局部窗口执行高分辨率 YOLO 召回。"""
        h, w = frame.shape[:2]
        rs = max(64, int(roi_size))
        rw = min(rs, w)
        rh = min(rs, h)
        cx, cy = float(center_xy[0]), float(center_xy[1])
        x0 = max(0, min(w - rw, int(round(cx - rw * 0.5))))
        y0 = max(0, min(h - rh, int(round(cy - rh * 0.5))))
        crop = frame[y0 : y0 + rh, x0 : x0 + rw]
        if rw < rs or rh < rs:
            crop = cv2.copyMakeBorder(crop, 0, max(0, rs - rh), 0, max(0, rs - rw), cv2.BORDER_CONSTANT, value=(0, 0, 0))
        results = self._run_model_with_oom_guard([crop], rs, conf_thresh=min_confidence)
        if not results:
            return []
        raw = self._parse_single_result(results[0], (w, h), offset_xy=(x0, y0))
        filt = filter_detections(raw, (w, h), self.cfg, min_confidence_override=min_confidence)
        return apply_nms(filt, self.cfg.detection.nms_iou)

    def infer(self, frames: list[np.ndarray], min_confidence: float | None = None) -> list[list[Detection]]:
        """批量推理视频帧，每帧返回过滤并经过 NMS 的 Detection 列表。"""
        if not frames:
            return []

        t_cfg = self.cfg.detection.tile
        fh, fw = frames[0].shape[:2]
        use_tile = bool(t_cfg.enabled) or (bool(getattr(t_cfg, "auto", False)) and max(fw, fh) > int(t_cfg.size))
        if use_tile:
            return [self._infer_tiled(f, min_confidence=min_confidence) for f in frames]

        imgsz = self._ensure_imgsz(frames[0])
        results = self._run_model_with_oom_guard(frames, imgsz, conf_thresh=min_confidence)
        out: list[list[Detection]] = []
        for frm, res in zip(frames, results):
            h, w = frm.shape[:2]
            raw = self._parse_single_result(res, (w, h), offset_xy=(0, 0))
            filt = filter_detections(raw, (w, h), self.cfg, min_confidence_override=min_confidence)
            nms_dets = apply_nms(filt, self.cfg.detection.nms_iou)
            out.append(nms_dets)
        return out
