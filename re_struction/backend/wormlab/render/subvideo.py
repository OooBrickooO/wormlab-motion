"""05 §3: 单虫子视频稳定裁剪窗口状态机与写入器 (subvideo.py)。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .video_io import create_compatible_video_writer, faststart_mp4, safe_rename


class StableCrop:
    """复刻旧 KalmanBoxTracker 的平滑裁剪中心与尺寸状态机。"""

    def __init__(self, initial_bbox: np.ndarray | list[float]) -> None:
        b = np.asarray(initial_bbox, dtype=np.float32)[:4]
        cx = 0.5 * float(b[0] + b[2])
        cy = 0.5 * float(b[1] + b[3])
        w = float(b[2] - b[0])
        h = float(b[3] - b[1])
        self.center = np.array([cx, cy], dtype=np.float32)
        self.size = float(max(w, h) * 1.8)

    def on_detect(self, bbox: np.ndarray | list[float]) -> None:
        b = np.asarray(bbox, dtype=np.float32)[:4]
        cx = 0.5 * float(b[0] + b[2])
        cy = 0.5 * float(b[1] + b[3])
        w = float(b[2] - b[0])
        h = float(b[3] - b[1])
        target_c = np.array([cx, cy], dtype=np.float32)
        target_s = float(max(w, h) * 1.8)
        self.center = 0.8 * self.center + 0.2 * target_c
        self.size = 0.85 * self.size + 0.15 * target_s

    def on_predict(self, bbox: np.ndarray | list[float]) -> None:
        b = np.asarray(bbox, dtype=np.float32)[:4]
        cx = 0.5 * float(b[0] + b[2])
        cy = 0.5 * float(b[1] + b[3])
        self.center = 0.9 * self.center + 0.1 * np.array([cx, cy], dtype=np.float32)

    def window(
        self,
        frame_w: int,
        frame_h: int,
        scale: float = 1.35,
        min_size: int = 72,
    ) -> tuple[int, int, int, int]:
        sz = int(max(min_size, round(self.size * float(scale))))
        sz = min(sz, int(frame_w), int(frame_h))
        half = sz // 2

        cx = int(round(float(self.center[0])))
        cy = int(round(float(self.center[1])))

        x1 = cx - half
        y1 = cy - half
        x2 = x1 + sz
        y2 = y1 + sz

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


def crop_roi(
    frame: np.ndarray,
    window: tuple[int, int, int, int],
    out_size: tuple[int, int] = (200, 200),
) -> np.ndarray | None:
    x1, y1, x2, y2 = window
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return cv2.resize(roi, out_size, interpolation=cv2.INTER_LINEAR)


class SubVideoWriter:
    """单条轨迹子视频写入器，支持结束时重命名为 worm_{id}_cycles_{n}.mp4。"""

    def __init__(
        self,
        sub_dir: str | Path,
        track_id: int,
        fps: float,
        initial_bbox: np.ndarray | list[float],
        out_size: tuple[int, int] = (200, 200),
    ) -> None:
        self.sub_dir = Path(sub_dir)
        self.sub_dir.mkdir(parents=True, exist_ok=True)
        self.track_id = int(track_id)
        self.out_size = out_size
        self.temp_path = self.sub_dir / f"worm_{self.track_id}.mp4"
        self.crop = StableCrop(initial_bbox)
        self.writer: cv2.VideoWriter | None = create_compatible_video_writer(
            self.temp_path, fps=fps, frame_size=out_size
        )
        self.frames_written = 0

    def write_frame(self, frame: np.ndarray, bbox: np.ndarray | list[float], is_ghost: bool = False) -> None:
        if self.writer is None:
            return
        if is_ghost:
            self.crop.on_predict(bbox)
        else:
            self.crop.on_detect(bbox)
        h, w = frame.shape[:2]
        win = self.crop.window(w, h)
        patch = crop_roi(frame, win, out_size=self.out_size)
        if patch is not None:
            self.writer.write(patch)
            self.frames_written += 1

    def close(self, total_cycles: int | None = None) -> Path | None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if not self.temp_path.exists():
            return None
        if total_cycles is not None:
            dst = self.sub_dir / f"worm_{self.track_id}_cycles_{int(total_cycles)}.mp4"
            final_str = safe_rename(self.temp_path, dst)
            faststart_mp4(final_str)
            return Path(final_str)
        faststart_mp4(self.temp_path)
        return self.temp_path
