"""02 §7: 高清原图 ROI 前景精修（08_SPEC Workstream A）。

突破 YOLOv8-seg `104×104` 低清掩码分辨率天花板：
1. 路径 1（首选）：原图 ROI 静态中值背景差分 + MAD 自适应阈值 + YOLO 掩码膨胀先验约束，
   重建像素级锐利虫体前景（对小虫 C 形内凹不糊平）；
2. 路径 2（回退）：无背景图时使用局部 ring 对比度极性 + 自适应阈值；
3. 护栏：面积比 / IoU 异常时整体回退原掩码，绝不引入碎裂或邻虫粘连。
"""
from __future__ import annotations

import cv2
import numpy as np

from wormlab.config import PoseCfg


def keep_largest_cc_overlapping(fg: np.ndarray, seed_mask: np.ndarray) -> np.ndarray | None:
    """在候选前景中保留与种子掩码 (YOLO 糊掩码) 重叠最大的单一连通域，防止合并邻近另一条虫。"""
    n_cc, labels = cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)
    if n_cc <= 1:
        return fg.copy() if int(fg.sum()) > 0 else None
    best_label = 0
    best_overlap = 0
    for lbl in range(1, n_cc):
        overlap = int(np.logical_and(labels == lbl, seed_mask).sum())
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = lbl
    if best_label == 0:
        return None
    return labels == best_label


def _dilate_prior(m: np.ndarray, r: int) -> np.ndarray:
    r = max(1, int(r))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m.astype(np.uint8) * 255, kernel, iterations=1) > 0


def refine_mask(
    gray_crop: np.ndarray | None,
    mask_crop: np.ndarray,
    cfg: PoseCfg,
    bg_crop: np.ndarray | None = None,
) -> np.ndarray:
    """在原图 ROI 内重建锐利前景：背景差分优先，局部对比度回退；异常时自动回退原掩码。"""
    m = np.asarray(mask_crop, dtype=bool)
    if not cfg.mask_refine.enabled or gray_crop is None or gray_crop.shape[:2] != m.shape[:2]:
        return m

    orig_area = int(m.sum())
    if orig_area < cfg.min_mask_px:
        return m

    prior = _dilate_prior(m, cfg.mask_refine.prior_dilate_px)

    cand: np.ndarray | None = None
    # 路径 1（首选）: 静态中值背景差分 —— 对细小 C 形虫最锐利，且不依赖 YOLO 糊轮廓边界
    if (
        cfg.mask_refine.use_bg_sub
        and bg_crop is not None
        and bg_crop.shape[:2] == gray_crop.shape[:2]
    ):
        diff = cv2.absdiff(gray_crop, bg_crop).astype(np.float32)
        ring = prior & (~m)
        bg_vals = diff[ring] if int(np.count_nonzero(ring)) >= 16 else diff[prior]
        if bg_vals.size >= 16:
            med = float(np.median(bg_vals))
            mad = float(np.median(np.abs(bg_vals - med))) * 1.4826
            thr = max(float(cfg.mask_refine.bg_min_diff), med + float(cfg.mask_refine.bg_mad_k) * mad)
            fg = (diff >= thr) & prior
            cand = keep_largest_cc_overlapping(fg, seed_mask=m)

    # 路径 2（回退）: 无背景图时，用局部 ring 估计虫体极性（暗虫/亮虫）与自适应阈值
    # （不再与糊掩码 m 强求交！避免旧版 np.logical_and(m, local_bin) 把凹槽边界重新抹平）
    if cand is None or int(cand.sum()) < cfg.min_mask_px:
        ring = prior & (~m)
        if int(np.count_nonzero(ring)) >= 16:
            ring_vals = gray_crop[ring].astype(np.float32)
            bg_med = float(np.median(gray_crop[ring]))
            bg_mad = float(np.median(np.abs(ring_vals - bg_med))) * 1.4826
            in_med = float(np.median(gray_crop[m]))
            delta = max(float(cfg.mask_refine.bg_min_diff), 3.0 * bg_mad)
            local_bin = (gray_crop <= bg_med - delta) if in_med < bg_med else (gray_crop >= bg_med + delta)
            cand = keep_largest_cc_overlapping(local_bin & prior, seed_mask=m)

    if cand is None or int(cand.sum()) < cfg.min_mask_px:
        return m

    # 护栏：防止把邻近另一条虫合并进来或缩成噪点
    area_ratio = float(cand.sum()) / float(max(1, orig_area))
    inter = int(np.logical_and(cand, m).sum())
    union = int(np.logical_or(cand, m).sum())
    iou = float(inter) / float(max(1, union))
    if not (cfg.mask_refine.area_ratio_lo <= area_ratio <= cfg.mask_refine.area_ratio_hi):
        return m
    if iou < cfg.mask_refine.min_iou_fallback:
        return m
    return cand
