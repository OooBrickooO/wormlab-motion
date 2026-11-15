"""01 §1–§2: 核心常量、枚举与数据类定义。

所有图像坐标均为左上角原点 (x 向右, y 向下, float32)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum

import numpy as np

from .geometry import resample_polyline

N_DENSE: int = 33
N_UI: int = 11

FLAG_BITS: tuple[str, ...] = (
    "loop",
    "branch",
    "fragment",
    "fallback",
    "empty",
    "merge",
    "clipped",
    "upscaled",
    "reoriented_after_gap",
    "length_jump",
    "aspect",
    "width",
    "endpoint_jump",
)


def encode_flags(flags: frozenset[str] | set[str]) -> int:
    val = 0
    for i, name in enumerate(FLAG_BITS):
        if name in flags:
            val |= 1 << i
    return val


def decode_flags(mask: int) -> frozenset[str]:
    m = int(mask)
    return frozenset(name for i, name in enumerate(FLAG_BITS) if (m & (1 << i)))


class Quality(IntEnum):
    """单帧姿态质量等级。"""

    OK = 0
    SUSPECT = 1
    BAD = 2
    MISSING = 3


class Behavior(str, Enum):
    """逐帧行为状态枚举。"""

    STATIC = "static"
    CRAWL = "crawl"
    THRASH = "thrash"
    TURN = "turn"
    MOVING = "moving"
    INVALID = "invalid"


@dataclass
class Detection:
    """单帧内单个检测目标。"""

    bbox: np.ndarray  # (4,) xyxy float32，整帧坐标
    conf: float = 1.0
    mask: np.ndarray | None = None  # 裁剪后的 bool 掩码 (h, w)
    mask_origin: tuple[int, int] = (0, 0)  # (x0, y0) 裁剪掩码左上角在整帧中的坐标
    det_idx: int = 0


@dataclass
class Pose:
    """单帧稠密中心线姿态与几何度量。"""

    dense: np.ndarray | None  # (33, 2) float32 整帧坐标，head=0；失败为 None
    length: float  # 弧长 (px，已含端帽修正)
    chord: float  # 首尾直线距离 (px)
    width: float  # 平均体宽 (px)
    area: float  # 原始掩码面积 (px²)
    quality: Quality = Quality.MISSING
    flags: frozenset[str] = field(default_factory=frozenset)
    contour: np.ndarray | None = None  # (K, 2) float32 平滑外轮廓折线，K<=80
    branch_ratio: float = 0.0  # 骨架分支像素占比（供 PoseQualityJudge 使用）

    @property
    def pts11(self) -> np.ndarray | None:
        """按弧长在 dense 上等距重采样得到的 11 个兼容关键点。"""
        if self.dense is None or len(self.dense) < 2:
            return None
        return resample_polyline(self.dense, N_UI)


@dataclass
class TrackOut:
    """Tracker.update() 每帧每条活跃/残留轨迹的输出。"""

    track_id: int
    bbox: np.ndarray  # (4,) xyxy float32
    det_idx: int | None  # None 表示 ghost 帧（无本帧检测）
    is_ghost: bool
    lost_frames: int
    confirmed: bool
    user_label: str | None
    head_kpt_idx: int | None
    head_xy: tuple[float, float] | None
    conf: float
    first_frame: int  # 该轨迹实例首次出现的帧号
    uid: int  # 轨迹实例唯一递增编号（永不复用）


@dataclass
class FrameRecord:
    """Pass A 缓存中的一行记录（长格式）。"""

    frame_idx: int
    track_id: int
    bbox: np.ndarray  # (4,) float32
    conf: float
    is_ghost: bool
    pose: Pose | None
    centroid: tuple[float, float]
    uid: int = 0
    first_frame: int = 0
    user_label: str | None = None
    head_kpt_idx: int | None = None
