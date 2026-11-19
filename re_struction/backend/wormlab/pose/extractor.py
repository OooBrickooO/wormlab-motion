"""02 §1: 单条轨迹的有状态姿态提取器 (PoseExtractor)。"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import replace

from wormlab.config import PoseCfg
from wormlab.schemas import Detection, Pose, Quality

from .centerline import extract_pose
from .mask_refine import _dilate_prior, refine_mask
from .orientation import orient
from .quality import PoseQualityJudge


class PoseExtractor:
    """每条轨迹独立的姿态提取与朝向/质量追踪器。"""

    def __init__(
        self,
        cfg: PoseCfg,
        fps: float,
        user_head_xy: tuple[float, float] | None = None,
    ) -> None:
        self.cfg = cfg
        self.fps = float(fps)
        self.user_head_xy = user_head_xy
        self.prev_dense: np.ndarray | None = None
        self.last_valid_frame: int | None = None
        self.reorient_gap_frames = max(1, int(round(float(cfg.reorient_gap_s) * float(fps))))
        self.judge = PoseQualityJudge(cfg, fps)

    def set_user_head_xy(self, head_xy: tuple[float, float] | None) -> None:
        """设置人工指定的头部坐标（将在下一次有效提取时生效一次）。"""
        self.user_head_xy = head_xy

    def _crop_with_prior(
        self,
        gray: np.ndarray | None,
        x0: int,
        y0: int,
        mh: int,
        mw: int,
    ) -> tuple[np.ndarray | None, int, int]:
        """按 mask_origin 向外扩边 prior_dilate_px 裁切灰度图，返回 (crop, ex0, ey0)。"""
        if gray is None:
            return None, x0, y0
        r = max(1, int(getattr(self.cfg.mask_refine, "prior_dilate_px", 8)))
        gh, gw = gray.shape[:2]
        ex0 = int(max(0, x0 - r))
        ey0 = int(max(0, y0 - r))
        ex1 = int(min(gw, x0 + mw + r))
        ey1 = int(min(gh, y0 + mh + r))
        if ex1 <= ex0 or ey1 <= ey0:
            return None, x0, y0
        return gray[ey0:ey1, ex0:ex1], ex0, ey0

    def _compute_roi_motion(
        self,
        mask: np.ndarray,
        x0: int,
        y0: int,
        frame_gray: np.ndarray | None,
        prev_frame_gray: np.ndarray | None,
    ) -> float:
        """ROI 帧间像素活跃度 mean(|gray_t - gray_{t-1}|[dilated_prior])，区分"真 C 摆动"与"瘫痪僵卷"。"""
        if frame_gray is None or prev_frame_gray is None:
            return float("nan")
        mh, mw = mask.shape[:2]
        gh, gw = frame_gray.shape[:2]
        if prev_frame_gray.shape[:2] != (gh, gw):
            return float("nan")
        x1 = int(min(gw, x0 + mw))
        y1 = int(min(gh, y0 + mh))
        x0c = int(max(0, x0))
        y0c = int(max(0, y0))
        if x1 <= x0c or y1 <= y0c:
            return float("nan")
        mask_crop = mask[0 : y1 - y0c, 0 : x1 - x0c]
        if int(mask_crop.sum()) < 1:
            return float("nan")
        gray_crop = frame_gray[y0c:y1, x0c:x1]
        prev_crop = prev_frame_gray[y0c:y1, x0c:x1]
        r = max(1, int(getattr(self.cfg.mask_refine, "prior_dilate_px", 8)))
        prior = _dilate_prior(mask_crop, r)
        if not np.any(prior):
            return float("nan")
        diff = cv2.absdiff(gray_crop, prev_crop).astype(np.float32)
        return float(diff[prior].mean())

    def process(
        self,
        det: Detection,
        frame_gray: np.ndarray | None = None,
        bg_gray: np.ndarray | None = None,
        prev_frame_gray: np.ndarray | None = None,
        frame_idx: int | None = None,
    ) -> Pose:
        mask = det.mask
        origin_xy = det.mask_origin
        roi_motion_val = float("nan")

        if mask is not None and self.cfg.mask_refine.enabled:
            x0, y0 = det.mask_origin
            mh, mw = mask.shape[:2]
            gray_crop, ex0, ey0 = self._crop_with_prior(frame_gray, x0, y0, mh, mw)
            bg_crop, _, _ = self._crop_with_prior(bg_gray, x0, y0, mh, mw)
            prev_crop, _, _ = self._crop_with_prior(prev_frame_gray, x0, y0, mh, mw)

            if gray_crop is not None:
                # 将原掩码嵌入外扩画布（掩码原点同步平移到 (ex0, ey0)，08_SPEC Workstream A）
                canvas = np.zeros(gray_crop.shape[:2], dtype=bool)
                oy0, ox0 = y0 - ey0, x0 - ex0
                canvas[oy0 : oy0 + mh, ox0 : ox0 + mw] = mask
                mask = refine_mask(gray_crop, canvas, self.cfg, bg_crop=bg_crop)
                origin_xy = (ex0, ey0)

        if mask is not None:
            roi_motion_val = self._compute_roi_motion(mask, origin_xy[0], origin_xy[1], frame_gray, prev_frame_gray)

        pose = extract_pose(mask, origin_xy, self.cfg)

        if pose.dense is not None:
            aligned = orient(pose.dense, prev_dense=self.prev_dense, user_head_xy=self.user_head_xy)
            self.user_head_xy = None
            flags = set(pose.flags)
            if (
                frame_idx is not None
                and self.last_valid_frame is not None
                and (frame_idx - self.last_valid_frame) > self.reorient_gap_frames
            ):
                flags.add("reoriented_after_gap")
            pose = replace(pose, dense=aligned, flags=frozenset(flags))
            pose = replace(pose, roi_motion=roi_motion_val)

        pose = self.judge.judge(pose)
        if pose.dense is not None and pose.quality != Quality.BAD:
            self.prev_dense = pose.dense
            if frame_idx is not None:
                self.last_valid_frame = int(frame_idx)
        return pose
