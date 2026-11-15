"""04 §3.3: 11 维外观形态学特征提取 (extract_features)。

与旧实现 EnhancedWormTracker.extract_features 逐项等价，支持裁剪后的局部掩码与整帧掩码。
"""
from __future__ import annotations

import cv2
import numpy as np


def extract_features(
    mask_crop: np.ndarray | None,
    bbox_in_crop: np.ndarray | list[float],
) -> np.ndarray:
    """提取 11 维外观特征：[log_area, log_perimeter, circularity, aspect_ratio, hu_0..hu_6]。"""
    if mask_crop is None or mask_crop.size == 0:
        return np.zeros(11, dtype=np.float64)

    m = mask_crop
    if m.ndim >= 3:
        m = np.any(m > 0, axis=0)

    x1, y1, x2, y2 = [float(v) for v in bbox_in_crop[:4]]
    w = x2 - x1
    h = y2 - y1
    aspect_ratio = w / (h + 1e-6)

    roi_x1 = max(0, int(np.floor(x1)) - 2)
    roi_y1 = max(0, int(np.floor(y1)) - 2)
    roi_x2 = min(int(m.shape[1]), int(np.ceil(x2)) + 2)
    roi_y2 = min(int(m.shape[0]), int(np.ceil(y2)) + 2)
    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
        return np.zeros(11, dtype=np.float64)

    roi = m[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.dtype != np.uint8:
        roi_u8 = (roi > 0).astype(np.uint8) * 255
    else:
        roi_u8 = roi

    area = int(np.sum(roi_u8 > 0))
    if area < 50:
        return np.zeros(11, dtype=np.float64)

    log_area = float(np.log1p(area))
    contours, _ = cv2.findContours(roi_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        log_perimeter = 0.0
        circularity = 0.0
        hu = np.zeros(7, dtype=np.float64)
    else:
        cnt = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(cnt, True))
        log_perimeter = float(np.log1p(perimeter))
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter + 1e-6))
        moments = cv2.moments(cnt)
        hu_raw = cv2.HuMoments(moments).flatten()
        hu = np.sign(hu_raw) * np.log1p(np.abs(hu_raw))

    return np.concatenate(([log_area, log_perimeter, circularity, aspect_ratio], hu)).astype(np.float64)
