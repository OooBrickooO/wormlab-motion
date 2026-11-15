"""04 §3.2: 单条活跃/残留跟踪轨迹状态类 (Track)。"""
from __future__ import annotations

from collections import deque

import numpy as np

from .kalman import KalmanBoxFilter


class Track:
    """封装卡尔曼滤波器、外观特征历史、运动方向向量、子视频稳定窗口及实例唯一标识 uid。"""

    def __init__(
        self,
        track_id: int,
        bbox: np.ndarray | list[float],
        uid: int,
        first_frame: int = 0,
        feature: np.ndarray | None = None,
        conf: float = 1.0,
    ) -> None:
        self.track_id: int = int(track_id)
        self.uid: int = int(uid)
        self.first_frame: int = int(first_frame)
        self.conf: float = float(conf)

        self.kf = KalmanBoxFilter(bbox, dt=1.0)
        b = np.asarray(bbox, dtype=np.float64)
        cx = float((b[0] + b[2]) / 2.0)
        cy = float((b[1] + b[3]) / 2.0)
        w = float(b[2] - b[0])
        h = float(b[3] - b[1])

        self.history: deque[tuple[float, float, float, float]] = deque(maxlen=30)
        self.history.append((cx, cy, w, h))

        self.feature: np.ndarray = (
            np.asarray(feature, dtype=np.float64).copy() if feature is not None else np.zeros(11, dtype=np.float64)
        )
        self.feature_history: deque[np.ndarray] = deque(maxlen=20)
        self.feature_history.append(self.feature)

        self.hits: int = 1
        self.lost_frames: int = 0
        self.time_since_update: int = 0
        self.confirmed: bool = False

        self.last_predicted_bbox: np.ndarray = np.asarray(bbox, dtype=np.float32).copy()
        self.crop_center: np.ndarray = np.array([cx, cy], dtype=np.float32)
        self.crop_size: float = float(max(w, h) * 1.8)

        self.motion_vector: np.ndarray = np.zeros(2, dtype=np.float32)
        self.last_pose_dense: np.ndarray | None = None

        self.user_label: str | None = None
        self.head_kpt_idx: int | None = None
        self.head_xy: tuple[float, float] | None = None

    def predict(self) -> np.ndarray:
        bbox = self.kf.predict()
        self.last_predicted_bbox = bbox
        cx = 0.5 * (float(bbox[0]) + float(bbox[2]))
        cy = 0.5 * (float(bbox[1]) + float(bbox[3]))
        self.crop_center = 0.9 * self.crop_center + 0.1 * np.array([cx, cy], dtype=np.float32)
        return bbox

    def update(self, bbox: np.ndarray | list[float], feature: np.ndarray | None = None, conf: float | None = None) -> None:
        cx, cy, w, h = self.kf.update(bbox)
        self.history.append((cx, cy, w, h))
        target_center = np.array([cx, cy], dtype=np.float32)
        target_size = float(max(w, h) * 1.8)
        self.crop_center = 0.8 * self.crop_center + 0.2 * target_center
        self.crop_size = 0.85 * self.crop_size + 0.15 * target_size
        self.time_since_update = 0
        self.lost_frames = 0
        self.hits += 1
        if conf is not None:
            self.conf = float(conf)
        if feature is not None:
            self.feature = np.asarray(feature, dtype=np.float64).copy()
            self.feature_history.append(self.feature)

    def feed_pose(self, dense: np.ndarray | None) -> None:
        """由姿态层回填中心线，以更新平滑运动方向向量 motion_vector。"""
        if dense is None or len(dense) == 0:
            return
        arr = np.asarray(dense, dtype=np.float32)
        if self.last_pose_dense is not None and self.last_pose_dense.shape == arr.shape:
            delta = arr - self.last_pose_dense
            self.motion_vector = 0.7 * self.motion_vector + 0.3 * np.mean(delta, axis=0)
        elif len(self.history) >= 2:
            p1 = np.array(self.history[-2][:2], dtype=np.float32)
            p2 = np.array(self.history[-1][:2], dtype=np.float32)
            self.motion_vector = 0.7 * self.motion_vector + 0.3 * (p2 - p1)
        self.last_pose_dense = arr.copy()

    def get_motion_vector(self) -> np.ndarray:
        if float(np.linalg.norm(self.motion_vector)) > 1e-6:
            return self.motion_vector
        if len(self.history) >= 2:
            p1 = np.array(self.history[-2][:2], dtype=np.float32)
            p2 = np.array(self.history[-1][:2], dtype=np.float32)
            return p2 - p1
        vel = np.asarray(self.kf.get_velocity(), dtype=np.float32)
        if vel.shape[0] >= 2:
            return vel[:2]
        return np.zeros(2, dtype=np.float32)

    def get_state(self) -> np.ndarray:
        return self.kf.get_state()

    def get_velocity(self) -> np.ndarray:
        return self.kf.get_velocity()

    def get_predicted_center(self) -> tuple[float, float]:
        return self.kf.get_predicted_center()

    def get_feature(self) -> np.ndarray:
        if len(self.feature_history) > 0:
            return np.mean(self.feature_history, axis=0)
        return self.feature

    def get_stable_crop_window(self, frame_w: int, frame_h: int, scale: float = 1.35, min_size: int = 72) -> tuple[int, int, int, int]:
        size = int(max(min_size, round(self.crop_size * scale)))
        size = min(size, int(frame_w), int(frame_h))
        half = size // 2

        cx = int(round(float(self.crop_center[0])))
        cy = int(round(float(self.crop_center[1])))

        x1 = cx - half
        y1 = cy - half
        x2 = x1 + size
        y2 = y1 + size

        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > frame_w:
            shift = x2 - frame_w
            x1 = max(0, x1 - shift)
            x2 = frame_w
        if y2 > frame_h:
            shift = y2 - frame_h
            y1 = max(0, y1 - shift)
            y2 = frame_h

        return int(x1), int(y1), int(x2), int(y2)
