"""04 §3.1: 8 维匀速边界框卡尔曼滤波器 (KalmanBoxFilter)。

状态向量: [cx, cy, w, h, vx, vy, vw, vh]^T，与旧 KalmanBoxTracker 数值保持 1e-5 内完全一致。
"""
from __future__ import annotations

import cv2
import numpy as np


class KalmanBoxFilter:
    """使用 cv2.KalmanFilter(8, 4) 实现边界框位置与尺寸平滑预测。"""

    def __init__(self, bbox: np.ndarray | list[float], dt: float = 1.0) -> None:
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        d = float(dt)
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, d, 0, 0, 0],
                [0, 1, 0, 0, 0, d, 0, 0],
                [0, 0, 1, 0, 0, 0, d, 0],
                [0, 0, 0, 1, 0, 0, 0, d],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.05
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1

        b = np.asarray(bbox, dtype=np.float64)
        cx = (b[0] + b[2]) / 2.0
        cy = (b[1] + b[3]) / 2.0
        w = b[2] - b[0]
        h = b[3] - b[1]
        self.kf.statePost = np.array([[cx], [cy], [w], [h], [0.0], [0.0], [0.0], [0.0]], dtype=np.float32)
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 100.0
        self.last_predicted_bbox: np.ndarray = np.asarray(bbox, dtype=np.float32).copy()

    def predict(self) -> np.ndarray:
        pred = self.kf.predict()
        cx = float(pred[0, 0].item())
        cy = float(pred[1, 0].item())
        w = float(pred[2, 0].item())
        h = float(pred[3, 0].item())
        bbox = np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)
        self.last_predicted_bbox = bbox
        return bbox

    def update(self, bbox: np.ndarray | list[float]) -> tuple[float, float, float, float]:
        b = np.asarray(bbox, dtype=np.float64)
        cx = float((b[0] + b[2]) / 2.0)
        cy = float((b[1] + b[3]) / 2.0)
        w = float(b[2] - b[0])
        h = float(b[3] - b[1])
        measurement = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
        self.kf.correct(measurement)
        return cx, cy, w, h

    def get_state(self) -> np.ndarray:
        state = self.kf.statePost
        cx = float(state[0, 0].item())
        cy = float(state[1, 0].item())
        w = float(state[2, 0].item())
        h = float(state[3, 0].item())
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float64)

    def get_velocity(self) -> np.ndarray:
        vx = float(self.kf.statePost[4, 0].item())
        vy = float(self.kf.statePost[5, 0].item())
        return np.array([vx, vy], dtype=np.float64)

    def get_predicted_center(self) -> tuple[float, float]:
        state = self.kf.statePost
        return float(state[0, 0].item()), float(state[1, 0].item())
