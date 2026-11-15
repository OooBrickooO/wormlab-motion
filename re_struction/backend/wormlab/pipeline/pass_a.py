"""05 §2: Pass A (GPU 检测 + 跟踪 + 姿态提取 + 实时流式运动学 + 缓存落盘)。"""
from __future__ import annotations

import argparse
import base64
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from wormlab.compat.legacy_payload import to_legacy_track
from wormlab.config import Config, load_config
from wormlab.detection.detector import Detector
from wormlab.kinematics.counter import KinematicsEngine
from wormlab.pose.extractor import PoseExtractor
from wormlab.render.overlay import DrawItem, draw_frame
from wormlab.schemas import Detection, Pose, TrackOut
from wormlab.tracking.recovery import compute_static_median_gray, recover_missing_confirmed_tracks
from wormlab.tracking.tracker import Tracker

from .cache import CacheWriter
from .late_entry import is_counted

logger = logging.getLogger(__name__)

IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class JobSpec:
    task_id: str
    video_path: str
    profile: str = "crawl_agar"
    c_unit: str = "cycle"
    worm_count: int | None = None
    head_config: list[dict[str, Any]] = field(default_factory=list)
    model_path: str = ""
    result_dir: str = ""
    cache_dir: str = ""
    max_frames: int | None = None


class Control:
    """线程安全的任务运行控制对象（暂停/恢复/取消/动态修改虫数上限）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: str = "running"
        self._stop_evt = threading.Event()
        self._worm_count_q: queue.Queue[int | None] = queue.Queue()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def cancelled(self) -> bool:
        return self._stop_evt.is_set()

    def pause(self) -> None:
        with self._lock:
            if self._state == "running":
                self._state = "paused"

    def resume(self) -> None:
        with self._lock:
            if self._state == "paused":
                self._state = "running"

    def cancel(self) -> None:
        with self._lock:
            self._state = "cancelled"
        self._stop_evt.set()

    def set_worm_count(self, n: int | None) -> None:
        val = int(n) if (n is not None and int(n) > 0) else None
        self._worm_count_q.put(val)

    def drain_worm_count(self) -> tuple[bool, int | None]:
        changed = False
        latest: int | None = None
        while True:
            try:
                latest = self._worm_count_q.get_nowait()
                changed = True
            except queue.Empty:
                break
        return changed, latest

    def apply(self, op: str, value: Any = None) -> None:
        if op == "pause":
            self.pause()
        elif op == "resume":
            self.resume()
        elif op == "cancel":
            self.cancel()
        elif op == "set_worm_count":
            self.set_worm_count(value)

    def wait_if_paused(self) -> bool:
        """若处于 paused 则循环等待；返回 True 表示已收到 cancel 信号。"""
        while not self._stop_evt.is_set():
            if self.state != "paused":
                return False
            time.sleep(0.05)
        return True


class NoopSink:
    """无服务器模式（CLI/测试）下的默认事件接收器。"""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.state: dict[str, Any] = {}

    def emit(self, event: str, data: dict[str, Any], room: str | None = None) -> None:
        self.events.append((event, data))

    def update_state(self, **kwargs: Any) -> None:
        self.state.update(kwargs)


class VideoSource:
    def __init__(self, path: str | Path, max_frames: int | None = None) -> None:
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {self.path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if self.fps <= 0.0 or not np.isfinite(self.fps):
            self.fps = 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if max_frames is not None and max_frames > 0:
            self.total_frames = min(raw_total, int(max_frames)) if raw_total > 0 else int(max_frames)
        else:
            self.total_frames = raw_total
        self._read_count = 0
        self._max_frames = max_frames

    def read_batch(self, batch_size: int) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        for _ in range(int(batch_size)):
            if self._max_frames is not None and self._read_count >= self._max_frames:
                break
            ok, frame = self.cap.read()
            if not ok or frame is None:
                break
            frames.append(frame)
            self._read_count += 1
        return frames

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


class ImageSource:
    """将单张静态图片封装为 36 帧短视频流（沿用旧行为）。"""

    def __init__(self, path: str | Path, repeat: int = 36, fps: float = 12.0) -> None:
        self.path = str(path)
        img = cv2.imread(self.path)
        if img is None:
            raise RuntimeError(f"无法读取图像文件: {self.path}")
        self._img = img
        self.height, self.width = img.shape[:2]
        self.fps = float(fps)
        self.total_frames = int(repeat)
        self._read_count = 0

    def read_batch(self, batch_size: int) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        for _ in range(int(batch_size)):
            if self._read_count >= self.total_frames:
                break
            frames.append(self._img.copy())
            self._read_count += 1
        return frames

    def close(self) -> None:
        pass


def open_source(path: str | Path, max_frames: int | None = None) -> VideoSource | ImageSource:
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTS:
        return ImageSource(path)
    return VideoSource(path, max_frames=max_frames)


def _put_with_cancel(q: queue.Queue[Any], item: Any, ctrl: Control) -> bool:
    while not ctrl.cancelled:
        try:
            q.put(item, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


class PassA:
    """四线程流式 Pass A 管线：reader -> main(GPU infer) -> tracker+pose -> emitter。"""

    def __init__(
        self,
        cfg: Config,
        job: JobSpec,
        detector: Detector,
        sink: Any | None = None,
        ctrl: Control | None = None,
    ) -> None:
        self.cfg = cfg
        self.job = job
        self.detector = detector
        self.sink = sink if sink is not None else NoopSink()
        self.ctrl = ctrl if ctrl is not None else Control()

    def run(self) -> Path:
        cache_dir = (
            Path(self.job.cache_dir)
            if self.job.cache_dir
            else Path(self.cfg.paths.data_dir) / "cache" / self.job.task_id
        )
        if self.ctrl.cancelled:
            return cache_dir

        source = open_source(self.job.video_path, max_frames=self.job.max_frames)
        fps = source.fps
        width, height = source.width, source.height
        total_frames = source.total_frames

        writer = CacheWriter(cache_dir)

        tracker = Tracker(
            cfg=self.cfg.tracking,
            fps=fps,
            frame_wh=(width, height),
            max_worm_id=self.job.worm_count,
            head_config=self.job.head_config,
        )

        pose_extractors: dict[int, PoseExtractor] = {}
        kin_engines: dict[int, KinematicsEngine] = {}
        ref_area_hist: dict[int, list[float]] = {}

        rec_cfg = self.cfg.tracking.recovery
        bg_gray: np.ndarray | None = None
        if rec_cfg.enabled and rec_cfg.bg_sub_enabled and not isinstance(source, ImageSource):
            bg_gray = compute_static_median_gray(
                self.job.video_path,
                total_frames=total_frames,
                n_samples=int(rec_cfg.bg_sample_frames),
                cancel_check=lambda: self.ctrl.cancelled,
            )

        batch_size = max(1, int(self.cfg.detection.batch_size))
        q_size = max(2, int(self.cfg.pipeline.frame_queue))
        frame_q: queue.Queue[tuple[int, list[np.ndarray]] | None] = queue.Queue(maxsize=q_size)
        infer_q: queue.Queue[tuple[int, list[np.ndarray], list[list[Detection]]] | None] = queue.Queue(
            maxsize=max(2, int(self.cfg.pipeline.infer_queue))
        )
        emit_q: queue.Queue[tuple[np.ndarray, list[DrawItem], int, int, int, list[dict[str, Any]]] | None] = (
            queue.Queue(maxsize=max(2, int(self.cfg.pipeline.frame_queue)))
        )

        pose_pool = ThreadPoolExecutor(max_workers=max(1, int(self.cfg.pipeline.pose_workers)))
        worker_err: list[Exception] = []

        self.sink.update_state(
            status="processing",
            phase="detect",
            total_frames=total_frames,
            current_frame=0,
        )
        self.sink.emit(
            "phase",
            {"task_id": self.job.task_id, "phase": "detect", "progress": 0.0},
            room=self.job.task_id,
        )

        def reader_loop() -> None:
            idx = 0
            try:
                while not self.ctrl.cancelled:
                    if self.ctrl.wait_if_paused():
                        break
                    batch = source.read_batch(batch_size)
                    if not batch:
                        break
                    if not _put_with_cancel(frame_q, (idx, batch), self.ctrl):
                        break
                    idx += len(batch)
            except Exception as exc:  # noqa: BLE001
                worker_err.append(exc)
                self.ctrl.cancel()
            finally:
                try:
                    frame_q.put(None, timeout=1.0)
                except queue.Full:
                    pass

        def tracker_loop() -> None:
            emit_every = max(1, int(self.cfg.pipeline.emit_every))
            try:
                while not self.ctrl.cancelled:
                    try:
                        item = infer_q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if item is None:
                        break
                    start_idx, frames, dets_per_frame = item
                    for offset, (frm, dets) in enumerate(zip(frames, dets_per_frame)):
                        if self.ctrl.cancelled:
                            break
                        cur_idx = start_idx + offset
                        changed, new_max = self.ctrl.drain_worm_count()
                        if changed:
                            tracker.set_max_worm_id(new_max)

                        outs = tracker.update(dets, frame_idx=cur_idx)
                        if rec_cfg.enabled:
                            ref_med = {uid: float(np.median(vals)) for uid, vals in ref_area_hist.items() if vals}
                            outs, dets = recover_missing_confirmed_tracks(
                                frame_bgr=frm,
                                bg_gray=bg_gray,
                                tracker=tracker,
                                detector=self.detector,
                                outs=outs,
                                dets=dets,
                                ref_areas=ref_med,
                                rec_cfg=rec_cfg,
                                frame_idx=cur_idx,
                            )

                        # 记录高质量主检测面积作为背景差分面积参考（滞后召回 conf=0.20 不污染基准）
                        for to in outs:
                            if not to.is_ghost and to.confirmed and to.det_idx is not None and to.det_idx < len(dets):
                                d_ref = dets[to.det_idx]
                                if d_ref.mask is not None and float(d_ref.conf) >= float(self.cfg.detection.min_confidence):
                                    hist = ref_area_hist.setdefault(to.uid, [])
                                    hist.append(float(d_ref.mask.sum()))
                                    if len(hist) > 25:
                                        del hist[0]

                        # 并行提取活跃目标的中心线姿态
                        def _extract_one(to: TrackOut) -> Pose | None:
                            if to.is_ghost or to.det_idx is None or to.det_idx >= len(dets):
                                return None
                            det = dets[to.det_idx]
                            if det.mask is None:
                                return None
                            ext = pose_extractors.get(to.uid)
                            if ext is None:
                                ext = PoseExtractor(self.cfg.pose, fps=fps, user_head_xy=to.head_xy)
                                pose_extractors[to.uid] = ext
                            elif to.head_xy is not None and ext.user_head_xy is None and ext.prev_dense is None:
                                ext.set_user_head_xy(to.head_xy)
                            return ext.process(det, frame_idx=cur_idx)

                        poses = list(pose_pool.map(_extract_one, outs))

                        draw_items: list[DrawItem] = []
                        live_tracks: list[dict[str, Any]] = []

                        for to, pose in zip(outs, poses):
                            eng = kin_engines.get(to.uid)
                            if eng is None:
                                eng = KinematicsEngine(
                                    fps=fps,
                                    cfg=self.cfg.kinematics,
                                    c_unit=self.job.c_unit,
                                    frame_height=height,
                                )
                                kin_engines[to.uid] = eng

                            if pose is not None and pose.dense is not None:
                                tracker.feed_pose(to.track_id, pose.dense)
                            eng.push(cur_idx, pose if not to.is_ghost else None)
                            writer.append(cur_idx, to, pose)

                            counted_flag = is_counted(to.first_frame, fps=fps, cfg_or_grace=self.cfg)
                            snap = eng.snapshot()
                            obs_sec = max(0.0, float(cur_idx - to.first_frame + 1) / float(fps))
                            live_dict = to_legacy_track(
                                tid=to.track_id,
                                label=to.user_label,
                                behavior_state=snap.behavior,
                                report_like=snap,
                                pose=pose,
                                frame_h=height,
                                is_ghost=to.is_ghost,
                                lost_frames=to.lost_frames,
                                head_kpt_idx=to.head_kpt_idx,
                                counted=counted_flag,
                                c_unit=self.job.c_unit,
                                observed_sec=obs_sec,
                            )
                            live_tracks.append(live_dict)

                            head_pt = (
                                (float(pose.dense[0, 0]), float(pose.dense[0, 1]))
                                if (pose is not None and pose.dense is not None and len(pose.dense) > 0)
                                else to.head_xy
                            )
                            trail_pts = list(tracker.track_history.get(to.track_id, []))[-30:]
                            draw_items.append(
                                DrawItem(
                                    track_id=to.track_id,
                                    bbox=to.bbox,
                                    is_ghost=to.is_ghost,
                                    lost_frames=to.lost_frames,
                                    user_label=to.user_label,
                                    contour=pose.contour if pose is not None else None,
                                    head_xy=head_pt,
                                    trail=trail_pts,
                                    counted=counted_flag,
                                    counts={
                                        "c_count": live_dict["c_count"],
                                        "s_count": live_dict["s_count"],
                                        "is_bidirectional": False,
                                        "avg_freq": live_dict["avg_freq"],
                                        "bends_per_min": live_dict["bends_per_min"],
                                        "motion_type_legacy": live_dict["motion_type"],
                                        "activity_score": live_dict["activity_score"],
                                    },
                                )
                            )

                        active_cnt = sum(1 for o in outs if not o.is_ghost)
                        ghost_cnt = sum(1 for o in outs if o.is_ghost)
                        self.sink.update_state(
                            current_frame=cur_idx + 1,
                            total_frames=total_frames,
                            active_tracks=active_cnt,
                            tracks=live_tracks,
                        )
                        if cur_idx % emit_every == 0:
                            try:
                                emit_q.put_nowait((frm.copy(), draw_items, cur_idx, active_cnt, ghost_cnt, live_tracks))
                            except queue.Full:
                                pass
            except Exception as exc:  # noqa: BLE001
                worker_err.append(exc)
                self.ctrl.cancel()
            finally:
                try:
                    emit_q.put(None, timeout=1.0)
                except queue.Full:
                    pass

        def emitter_loop() -> None:
            jpeg_quality = int(self.cfg.pipeline.jpeg_quality)
            try:
                while not self.ctrl.cancelled:
                    try:
                        item = emit_q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if item is None:
                        break
                    frm, draw_items, cur_idx, active_cnt, ghost_cnt, live_tracks = item
                    rendered = draw_frame(frm, draw_items, cur_idx, active_cnt, ghost_cnt)
                    ok, buf = cv2.imencode(".jpg", rendered, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                    if not ok:
                        continue
                    b64_str = base64.b64encode(buf.tobytes()).decode("ascii")
                    self.sink.update_state(latest_frame=b64_str)
                    self.sink.emit(
                        "frame",
                        {
                            "task_id": self.job.task_id,
                            "frame": cur_idx + 1,
                            "total_frames": total_frames,
                            "image": b64_str,
                            "stats": {
                                "frame": cur_idx + 1,
                                "active_tracks": active_cnt,
                                "tracks": live_tracks,
                            },
                        },
                        room=self.job.task_id,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Emitter 线程异常: %s", exc)

        t_reader = threading.Thread(target=reader_loop, name="PassA-Reader", daemon=True)
        t_tracker = threading.Thread(target=tracker_loop, name="PassA-Tracker", daemon=True)
        t_emitter = threading.Thread(target=emitter_loop, name="PassA-Emitter", daemon=True)
        t_reader.start()
        t_tracker.start()
        t_emitter.start()

        try:
            while not self.ctrl.cancelled:
                if self.ctrl.wait_if_paused():
                    break
                try:
                    batch_item = frame_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if batch_item is None:
                    break
                start_idx, batch_frames = batch_item
                low_conf = self.cfg.detection.recovery_confidence
                if low_conf is not None and isinstance(self.detector, Detector):
                    dets_batch = self.detector.infer(batch_frames, min_confidence=float(low_conf))
                else:
                    dets_batch = self.detector.infer(batch_frames)
                if not _put_with_cancel(infer_q, (start_idx, batch_frames, dets_batch), self.ctrl):
                    break
        finally:
            try:
                if not self.ctrl.cancelled:
                    infer_q.put(None, timeout=1.0)
                else:
                    infer_q.put_nowait(None)
            except queue.Full:
                pass
            t_reader.join(timeout=5.0)
            t_tracker.join(timeout=10.0)
            t_emitter.join(timeout=5.0)
            pose_pool.shutdown(wait=True)
            source.close()

        if worker_err:
            raise worker_err[0]

        meta = {
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": max(total_frames, source._read_count),
            "video_path": str(self.job.video_path),
            "model": str(self.job.model_path),
            "profile": str(self.job.profile),
            "c_unit": str(self.job.c_unit),
            "cfg_hash": self.cfg.hash(),
            "worm_count": self.job.worm_count,
            "head_config": self.job.head_config,
        }
        out_dir = writer.close(meta)
        if not self.ctrl.cancelled:
            self.sink.emit(
                "phase",
                {"task_id": self.job.task_id, "phase": "analyze", "progress": 0.0},
                room=self.job.task_id,
            )
        return out_dir


def run_pass_a(
    video_path: str | Path,
    cache_dir: str | Path,
    cfg: Config,
    model_path: str | Path | None = None,
    max_worm_id: int | None = None,
    head_config: list[dict[str, Any]] | None = None,
    max_frames: int | None = None,
    detector: Detector | None = None,
) -> Path:
    """便捷无服务器调用入口（供 CLI、评测脚本及集成测试使用）。"""
    re_root = Path(__file__).resolve().parents[3]
    m_path = Path(model_path) if model_path else (re_root / "backend" / "models" / cfg.detection.default_model)
    det = detector if detector is not None else Detector(m_path, cfg)
    job = JobSpec(
        task_id=Path(cache_dir).name,
        video_path=str(video_path),
        profile="crawl_agar",
        c_unit=cfg.counting.c_unit,
        worm_count=max_worm_id,
        head_config=head_config or [],
        model_path=str(m_path),
        cache_dir=str(cache_dir),
        max_frames=max_frames,
    )
    return PassA(cfg=cfg, job=job, detector=det).run()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="运行 Pass A 并在 data/cache/ 下生成 tracks.npz + meta.json")
    parser.add_argument("--video", type=str, required=True, help="输入视频路径或文件名")
    parser.add_argument("--max-frames", type=int, default=None, help="最大处理帧数")
    parser.add_argument("--profile", type=str, default="crawl_agar")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args(argv)

    re_root = Path(__file__).resolve().parents[3]
    repo_root = re_root.parent
    v_cand = Path(args.video)
    if not v_cand.exists():
        for folder in (re_root / "referAnswer", repo_root / "backend" / "uploads", repo_root / "videos"):
            if (folder / args.video).exists():
                v_cand = folder / args.video
                break
    if not v_cand.exists():
        raise FileNotFoundError(f"找不到视频文件: {args.video}")

    cfg = load_config(args.profile)
    out_dir = Path(args.out_dir) if args.out_dir else (re_root / "data" / "cache" / v_cand.stem)
    res_dir = run_pass_a(
        video_path=v_cand,
        cache_dir=out_dir,
        cfg=cfg,
        max_frames=args.max_frames,
    )
    logger.info("Pass A 完成，缓存目录: %s", res_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
