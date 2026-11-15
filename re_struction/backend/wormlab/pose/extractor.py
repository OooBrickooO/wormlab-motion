"""02 §1: 单条轨迹的有状态姿态提取器 (PoseExtractor)。"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from wormlab.config import PoseCfg
from wormlab.schemas import Detection, Pose, Quality

from .centerline import extract_pose
from .mask_refine import refine_mask
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

    def process(
        self,
        det: Detection,
        frame_gray: np.ndarray | None = None,
        frame_idx: int | None = None,
    ) -> Pose:
        mask = det.mask
        if mask is not None and self.cfg.mask_refine.enabled and frame_gray is not None:
            x0, y0 = det.mask_origin
            mh, mw = mask.shape[:2]
            gray_crop = frame_gray[y0 : y0 + mh, x0 : x0 + mw]
            mask = refine_mask(gray_crop, mask, self.cfg)

        pose = extract_pose(mask, det.mask_origin, self.cfg)
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

        pose = self.judge.judge(pose)
        if pose.dense is not None and pose.quality != Quality.BAD:
            self.prev_dense = pose.dense
            if frame_idx is not None:
                self.last_valid_frame = int(frame_idx)
        return pose
