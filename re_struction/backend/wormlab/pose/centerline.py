"""02 §2–§4: 掩码预处理、小虫 ROI 放大、填洞、端帽延伸、33 点中心线重采样与平滑外轮廓提取。"""
from __future__ import annotations

import math

import cv2
import numpy as np
import scipy.ndimage
from scipy.ndimage import gaussian_filter1d

from wormlab.config import PoseCfg
from wormlab.geometry import polyline_length, resample_polyline
from wormlab.schemas import N_DENSE, Pose, Quality

from .skeleton import extract_skeleton_path


def _compute_shape_moments(mask: np.ndarray) -> tuple[float, float, float, float]:
    """从二值前景计算等效椭圆长短轴比 b/a、偏心率、主轴角、沿短轴 3 阶标准化偏度。

    特点：完全免骨架，对 10~60px 极小虫与 C 形弯曲高度稳定（08_SPEC Workstream A）。
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 6:
        return float("nan"), float("nan"), float("nan"), float("nan")
    dx = xs.astype(np.float64) - float(xs.mean())
    dy = ys.astype(np.float64) - float(ys.mean())
    cxx = float(np.mean(dx * dx))
    cyy = float(np.mean(dy * dy))
    cxy = float(np.mean(dx * dy))

    tr = cxx + cyy
    det_term = math.sqrt(max(0.0, (cxx - cyy) ** 2 + 4.0 * cxy * cxy))
    lam1 = max(1e-6, 0.5 * (tr + det_term))   # 长轴方差
    lam2 = max(1e-6, 0.5 * (tr - det_term))   # 短轴方差
    aspect = float(math.sqrt(lam2 / lam1))    # b / a ∈ (0, 1]：直虫≈0.15~0.30，C弯跃升至 0.45~0.85
    ecc = float(math.sqrt(max(0.0, 1.0 - lam2 / lam1)))
    angle = float(0.5 * math.atan2(2.0 * cxy, cxx - cyy))  # 主轴角 ∈ [-π/2, π/2]

    # 沿短轴方向 v_minor = (-sin(angle), cos(angle)) 投影计算 3 阶标准化偏度
    # 物理意义：C 形弧的质心偏向凹侧，导致短轴投影呈显著非对称偏态；
    # 符号受主轴 ±π 不定性影响，后续在时间轴上对 angle 做 unwrap(2*angle)/2 即可得到连续带符号弯向
    v_min_x, v_min_y = -math.sin(angle), math.cos(angle)
    proj_minor = dx * v_min_x + dy * v_min_y
    std_minor = math.sqrt(lam2)
    skew_minor = float(np.mean(proj_minor ** 3) / (std_minor ** 3 + 1e-9))
    return aspect, ecc, angle, skew_minor


def _missing_pose(flags: set[str], area: float = 0.0) -> Pose:
    return Pose(
        dense=None,
        length=0.0,
        chord=0.0,
        width=0.0,
        area=float(area),
        quality=Quality.MISSING,
        flags=frozenset(flags),
        contour=None,
        branch_ratio=0.0,
    )


def _fallback_pca_axis(mask: np.ndarray, n_pts: int = N_DENSE) -> tuple[np.ndarray, float]:
    """当骨架点数不足时，使用掩码像素 PCA 主轴 2%~98% 分位点生成中心线。"""
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    mean_pt = pts.mean(axis=0)
    centered = pts - mean_pt
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, int(np.argmax(eigvals))]
    proj = centered @ principal
    p_lo, p_hi = np.percentile(proj, [2.0, 98.0])
    p0 = mean_pt + p_lo * principal
    p1 = mean_pt + p_hi * principal
    t = np.linspace(0.0, 1.0, n_pts)[:, None]
    line = (1.0 - t) * p0[None, :] + t * p1[None, :]
    dt = scipy.ndimage.distance_transform_edt(mask)
    width = float(2.0 * np.median(dt[ys, xs])) if len(xs) > 0 else 2.0
    return line, width


def _extend_endcap_single(
    ps: np.ndarray,
    at_start: bool,
    dt_grid: np.ndarray,
    mask_grid: np.ndarray,
    gain: float,
) -> np.ndarray:
    """沿端部向外切线补偿尖端骨架内缩（仅当端点距离掩码前沿大于局部内切圆半径 dt 时补偿）。"""
    n = len(ps)
    k = min(n - 1, max(3, int(round(0.05 * n))))
    if at_start:
        end_pt = ps[0]
        vec = ps[0] - ps[k]
    else:
        end_pt = ps[-1]
        vec = ps[-1] - ps[-1 - k]

    norm = float(np.hypot(vec[0], vec[1]))
    if norm <= 1e-8:
        return ps
    t_dir = vec / norm

    h, w = mask_grid.shape[:2]
    ey = int(np.clip(round(float(end_pt[1])), 0, h - 1))
    ex = int(np.clip(round(float(end_pt[0])), 0, w - 1))
    local_r = float(dt_grid[ey, ex])

    # 探测沿切线到掩码外边界的距离 d_bound
    step = 0.5
    d_bound = 0.0
    max_probe = max(local_r * 3.0, 4.0)
    dist = step
    while dist <= max_probe:
        cand = end_pt + t_dir * dist
        cx, cy = int(round(float(cand[0]))), int(round(float(cand[1])))
        if 0 <= cx < w and 0 <= cy < h and mask_grid[cy, cx]:
            d_bound = dist
            dist += step
        else:
            break

    # 若掩码端部为标准半圆帽，骨架端点已在圆心，此时 d_bound ≈ local_r，无需重复外扩；
    # 若掩码端部呈锥形尖头（真实线虫头尾）或因毛刺分叉提前终止，d_bound > local_r，补偿差值。
    shortfall = max(0.0, d_bound - local_r)
    ext = min(d_bound, shortfall * float(gain))
    if ext < 0.5:
        return ps

    new_pt = end_pt + t_dir * ext
    if at_start:
        return np.vstack([new_pt[None, :], ps])
    return np.vstack([ps, new_pt[None, :]])


def _extract_smooth_contour(mask_original: np.ndarray, origin: tuple[int, int]) -> np.ndarray | None:
    """提取平滑外轮廓折线 (K<=80, 2) float32，供渲染层使用。"""
    bin_f = (cv2.GaussianBlur(mask_original.astype(np.float32), (5, 5), 1.0) >= 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bin_f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 20.0:
        return None
    approx = cv2.approxPolyDP(largest, epsilon=0.7, closed=True).reshape(-1, 2).astype(np.float32)
    if len(approx) > 80:
        idx = np.linspace(0, len(approx) - 1, 80).round().astype(int)
        approx = approx[idx]
    ox, oy = float(origin[0]), float(origin[1])
    approx[:, 0] += ox
    approx[:, 1] += oy
    return approx


def _open_closed_ring(mask: np.ndarray) -> np.ndarray:
    """当 Ω 弯首尾相接形成大空洞闭环时，在距离变换最薄处切开闭环以恢复完整周长骨架。"""
    dt = scipy.ndimage.distance_transform_edt(mask)
    sk = extract_skeleton_Pixels(mask)
    ys, xs = np.nonzero(sk)
    if len(xs) == 0:
        return mask
    # 取距离变换最薄的骨架点切断
    min_idx = int(np.argmin(dt[ys, xs]))
    cy, cx = int(ys[min_idx]), int(xs[min_idx])
    r_cut = max(2, int(round(float(dt[cy, cx]) + 1.5)))
    opened = mask.copy().astype(np.uint8)
    cv2.circle(opened, (cx, cy), r_cut, 0, -1)
    return opened.astype(bool)


def extract_skeleton_Pixels(mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize

    return skeletonize(mask)


def extract_pose(
    det_mask: np.ndarray | None,
    origin: tuple[int, int],
    cfg: PoseCfg,
) -> Pose:
    """从局部二值掩码提取 33 点稠密中心线与几何度量（永不抛异常）。"""
    try:
        if det_mask is None or det_mask.size == 0:
            return _missing_pose({"empty"}, area=0.0)

        m_raw = np.asarray(det_mask, dtype=bool)
        area0 = int(m_raw.sum())
        if area0 < int(cfg.min_mask_px):
            return _missing_pose({"empty"}, area=float(area0))

        flags: set[str] = set()

        # 2.1 最大连通域与碎片检测
        n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(m_raw.astype(np.uint8), connectivity=8)
        if n_cc <= 1:
            return _missing_pose({"empty"}, area=float(area0))

        areas_cc = stats[1:, cv2.CC_STAT_AREA]
        order_cc = np.argsort(-areas_cc)
        keep_label = int(order_cc[0]) + 1
        largest_area = float(areas_cc[order_cc[0]])
        if len(areas_cc) >= 2:
            second_area = float(areas_cc[order_cc[1]])
            if second_area > float(cfg.fragment_ratio) * largest_area:
                flags.add("fragment")

        m_main = labels == keep_label
        bw = int(stats[keep_label, cv2.CC_STAT_WIDTH])
        bh = int(stats[keep_label, cv2.CC_STAT_HEIGHT])
        long_side = max(bw, bh)

        # 2.2 小虫体 ROI 放大
        if long_side < int(cfg.upscale_below_px):
            scale = min(float(cfg.max_upscale), float(cfg.upscale_target_px) / float(max(1, long_side)))
            blurred = cv2.GaussianBlur(m_main.astype(np.float32), (0, 0), 0.8)
            resized = cv2.resize(blurred, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            m_work = resized > 0.5
            flags.add("upscaled")
        else:
            scale = 1.0
            m_work = m_main

        # 2.3 空洞检测与填洞（仅填充微小噪声孔洞，保留 Ω 弯自接触环路并由骨架解环处理）
        filled = scipy.ndimage.binary_fill_holes(m_work)
        hole_px = int(filled.sum()) - int(m_work.sum())
        filled_px = max(1, int(filled.sum()))
        hole_frac = float(hole_px) / float(filled_px)
        has_loop = False
        if hole_frac > float(cfg.loop_hole_frac):
            flags.add("loop")
            has_loop = True
        else:
            m_work = filled

        # 3. 骨架测地路径
        dt_work = scipy.ndimage.distance_transform_edt(m_work).astype(np.float32)
        path_xy, path_dt, branch_ratio = extract_skeleton_path(
            m_work,
            center_penalty=cfg.center_penalty,
            has_loop=has_loop,
        )

        if path_xy is None or len(path_xy) < 4:
            # 3.3 回退主轴
            flags.add("fallback")
            ps, width_scaled = _fallback_pca_axis(m_work, n_pts=cfg.n_dense)
            branch_ratio = 0.0
        else:
            if branch_ratio > float(cfg.branch_flag_ratio):
                flags.add("branch")
            # 4.1 高斯平滑（固定两端点避免收缩）
            sigma = max(1.5, float(cfg.smooth_sigma_frac) * len(path_xy))
            ps = gaussian_filter1d(path_xy, sigma=sigma, axis=0, mode="nearest")
            ps[0] = path_xy[0]
            ps[-1] = path_xy[-1]

            # 4.2 端帽延伸
            if cfg.endcap:
                ps = _extend_endcap_single(ps, at_start=True, dt_grid=dt_work, mask_grid=m_work, gain=cfg.endcap_gain)
                ps = _extend_endcap_single(ps, at_start=False, dt_grid=dt_work, mask_grid=m_work, gain=cfg.endcap_gain)

            width_scaled = float(2.0 * np.mean(path_dt)) if len(path_dt) > 0 else 2.0

        # 4.3 按弧长等距重采样至 N_DENSE (33) 点
        dense_local = resample_polyline(ps, int(cfg.n_dense))
        length_px = polyline_length(ps) / scale
        chord_px = float(np.linalg.norm(dense_local[0] - dense_local[-1])) / scale
        width_px = width_scaled / scale

        ox, oy = float(origin[0]), float(origin[1])
        dense_global = dense_local / float(scale)
        dense_global[:, 0] += ox
        dense_global[:, 1] += oy
        dense_global = dense_global.astype(np.float32)

        contour = _extract_smooth_contour(m_main, origin)

        # 免骨架形状描述子：即使在骨架回退 (fallback PCA) 路径也必须填充
        ell_aspect, ell_ecc, ell_angle, ell_skew_minor = _compute_shape_moments(m_main)

        return Pose(
            dense=dense_global,
            length=float(length_px),
            chord=float(chord_px),
            width=float(width_px),
            area=float(area0),
            quality=Quality.OK,
            flags=frozenset(flags),
            contour=contour,
            branch_ratio=float(branch_ratio),
            ell_aspect=ell_aspect,
            ell_ecc=ell_ecc,
            ell_angle=ell_angle,
            ell_skew_minor=ell_skew_minor,
        )
    except Exception as exc:  # noqa: BLE001
        import logging

        logging.getLogger(__name__).warning("extract_pose 捕获异常并安全回退 MISSING: %s", exc)
        return _missing_pose({"empty"}, area=0.0)
