"""02 §7: 可选局部自适应掩码精修（默认关闭）。"""
from __future__ import annotations

import cv2
import numpy as np

from wormlab.config import PoseCfg


def refine_mask(
    gray_crop: np.ndarray | None,
    mask_crop: np.ndarray,
    cfg: PoseCfg,
) -> np.ndarray:
    """在膨胀 2px 的 YOLO 掩码区域内执行局部 Otsu 精修并求交；异常变化时自动回退原掩码。"""
    m = np.asarray(mask_crop, dtype=bool)
    if not cfg.mask_refine.enabled or gray_crop is None or gray_crop.shape[:2] != m.shape[:2]:
        return m

    orig_area = int(m.sum())
    if orig_area < cfg.min_mask_px:
        return m

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(m.astype(np.uint8) * 255, kernel, iterations=1) > 0
    roi_vals = gray_crop[dilated]
    if len(roi_vals) < 16:
        return m

    # 判断虫体相对于背景是亮还是暗
    inner_mean = float(gray_crop[m].mean())
    ring_mask = np.logical_and(dilated, ~m)
    outer_mean = float(gray_crop[ring_mask].mean()) if np.any(ring_mask) else float(gray_crop.mean())

    otsu_thr, _ = cv2.threshold(roi_vals.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if inner_mean >= outer_mean:
        local_bin = gray_crop >= otsu_thr
    else:
        local_bin = gray_crop <= otsu_thr

    refined = np.logical_and(m, local_bin)
    ref_area = int(refined.sum())
    if ref_area == 0:
        return m

    inter = int(np.logical_and(refined, m).sum())
    union = int(np.logical_or(refined, m).sum())
    iou = float(inter) / float(max(1, union))
    area_ratio = float(ref_area) / float(orig_area)

    if iou < 0.70 or not (0.80 <= area_ratio <= 1.20):
        return m
    return refined
