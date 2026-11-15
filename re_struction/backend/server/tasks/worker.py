"""05 §8: 常驻 GPU Worker 进程与测试用假 Worker (worker.py)。"""
from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from wormlab.config import load_config
from wormlab.detection.detector import Detector
from wormlab.pipeline.pass_a import Control, JobSpec, PassA
from wormlab.pipeline.pass_b import PassB

logger = logging.getLogger(__name__)


class QueueSink:
    """将 Worker 内的状态更新与 Socket.IO 事件写入跨进程 evt_q。"""

    def __init__(self, evt_q: Any, task_id: str) -> None:
        self.evt_q = evt_q
        self.task_id = str(task_id)

    def emit(self, event: str, data: dict[str, Any], room: str | None = None) -> None:
        try:
            self.evt_q.put(
                {
                    "type": "emit",
                    "task_id": self.task_id,
                    "event": event,
                    "data": data,
                    "room": room or self.task_id,
                }
            )
        except Exception:  # noqa: BLE001
            pass

    def update_state(self, **kwargs: Any) -> None:
        try:
            self.evt_q.put(
                {
                    "type": "update_state",
                    "task_id": self.task_id,
                    "kwargs": kwargs,
                }
            )
        except Exception:  # noqa: BLE001
            pass


def worker_main(cmd_q: Any, evt_q: Any, data_dir_str: str) -> None:
    """常驻子进程主循环：串行执行排队任务并复用 GPU YOLO 模型实例。"""
    detector_cache: dict[str, Detector] = {}

    while True:
        try:
            msg = cmd_q.get()
        except (EOFError, KeyboardInterrupt):
            break
        if not isinstance(msg, dict):
            continue
        mtype = msg.get("type")
        if mtype == "shutdown":
            break
        if mtype != "start":
            continue

        job = JobSpec(**msg["job"])
        ctrl = Control()
        initial_ctrl = msg.get("initial_control", "running")
        if initial_ctrl == "paused":
            ctrl.pause()
        elif initial_ctrl == "cancelled":
            ctrl.cancel()

        sink = QueueSink(evt_q, job.task_id)
        stop_ctrl_listener = threading.Event()

        def _ctrl_listener() -> None:
            while not stop_ctrl_listener.is_set():
                try:
                    c_msg = cmd_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                except Exception:  # noqa: BLE001
                    break
                if not isinstance(c_msg, dict):
                    continue
                if c_msg.get("type") == "shutdown":
                    ctrl.cancel()
                    cmd_q.put({"type": "shutdown"})
                    break
                if c_msg.get("type") == "control" and c_msg.get("task_id") == job.task_id:
                    ctrl.apply(str(c_msg.get("op", "")), c_msg.get("value"))

        listener_thread = threading.Thread(target=_ctrl_listener, daemon=True)
        listener_thread.start()

        try:
            if ctrl.cancelled:
                sink.update_state(status="cancelled", control="cancelled", queued=False)
                sink.emit("cancelled", {"task_id": job.task_id, "message": "Task cancelled by user"})
                continue

            cfg = load_config(job.profile, overrides={"paths": {"data_dir": data_dir_str}})
            m_key = str(job.model_path)
            if m_key not in detector_cache:
                detector_cache.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                detector_cache[m_key] = Detector(Path(m_key), cfg)
            det = detector_cache[m_key]

            if ctrl.cancelled:
                sink.update_state(status="cancelled", control="cancelled", queued=False)
                sink.emit("cancelled", {"task_id": job.task_id, "message": "Task cancelled by user"})
            else:
                sink.update_state(queued=False, status="paused" if ctrl.state == "paused" else "processing")
                cache_dir = PassA(cfg=cfg, job=job, detector=det, sink=sink, ctrl=ctrl).run()
                if ctrl.cancelled:
                    sink.update_state(status="cancelled", control="cancelled")
                    sink.emit("cancelled", {"task_id": job.task_id, "message": "Task cancelled by user"})
                else:
                    PassB(cfg=cfg).run(cache_dir=cache_dir, job=job, sink=sink, ctrl=ctrl)
                    if ctrl.cancelled:
                        sink.update_state(status="cancelled", control="cancelled")
                        sink.emit("cancelled", {"task_id": job.task_id, "message": "Task cancelled by user"})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Worker 执行任务 %s 失败: %s", job.task_id, exc)
            sink.update_state(status="failed", control="failed", error_message=str(exc))
            sink.emit("error", {"task_id": job.task_id, "message": str(exc)})
        finally:
            stop_ctrl_listener.set()
            listener_thread.join(timeout=1.0)
            evt_q.put({"type": "done", "task_id": job.task_id})


class WorkerProcess:
    """封装 spawn 模式常驻 Worker 子进程。"""

    def __init__(self, data_dir: str | Path) -> None:
        ctx = mp.get_context("spawn")
        self.cmd_q: Any = ctx.Queue()
        self.evt_q: Any = ctx.Queue()
        self.data_dir = str(data_dir)
        self.proc = ctx.Process(
            target=worker_main,
            args=(self.cmd_q, self.evt_q, self.data_dir),
            daemon=True,
        )

    def start(self) -> None:
        if not self.proc.is_alive():
            self.proc.start()

    def send_start(self, job: JobSpec, initial_control: str = "running") -> None:
        self.cmd_q.put({"type": "start", "job": asdict(job), "initial_control": initial_control})

    def send_control(self, task_id: str, op: str, value: Any = None) -> None:
        self.cmd_q.put({"type": "control", "task_id": str(task_id), "op": str(op), "value": value})

    def shutdown(self) -> None:
        try:
            self.cmd_q.put({"type": "shutdown"})
            self.proc.join(timeout=3.0)
            if self.proc.is_alive():
                self.proc.terminate()
        except Exception:  # noqa: BLE001
            pass


class FakeWorker:
    """契约测试专用内存假 Worker（不启动子进程，不加载 YOLO，可控推送 frame/complete）。"""

    def __init__(self, data_dir: str | Path, auto_complete: bool = False) -> None:
        self.cmd_q: queue.Queue[dict[str, Any]] = queue.Queue()
        self.evt_q: queue.Queue[dict[str, Any]] = queue.Queue()
        self.data_dir = str(data_dir)
        self.auto_complete = bool(auto_complete)
        self.started_jobs: list[JobSpec] = []
        self.controls: list[tuple[str, str, Any]] = []

    def start(self) -> None:
        pass

    def send_start(self, job: JobSpec, initial_control: str = "running") -> None:
        self.started_jobs.append(job)
        sink = QueueSink(self.evt_q, job.task_id)
        sink.update_state(queued=False, status="paused" if initial_control == "paused" else "processing")
        if self.auto_complete:
            def _sim() -> None:
                time.sleep(0.02)
                sink.update_state(current_frame=10, total_frames=10, active_tracks=1)
                sink.emit(
                    "frame",
                    {
                        "task_id": job.task_id,
                        "frame": 10,
                        "total_frames": 10,
                        "image": "",
                        "stats": {"frame": 10, "active_tracks": 1, "tracks": []},
                    },
                )
                sink.update_state(
                    status="completed",
                    phase="done",
                    video_url=f"/media/res_{job.task_id}.mp4",
                    summary_url=f"/download/res_{job.task_id}_id_pose_summary.json",
                    csv_url=f"/download/res_{job.task_id}_swing_summary.csv",
                    log_url=f"/download/res_{job.task_id}_log.json",
                    sub_videos_url=f"/download/{job.task_id}_sub_videos.zip",
                    id_pose_summary=[],
                )
                sink.emit(
                    "complete",
                    {
                        "task_id": job.task_id,
                        "video_url": f"/media/res_{job.task_id}.mp4",
                        "summary_url": f"/download/res_{job.task_id}_id_pose_summary.json",
                        "csv_url": f"/download/res_{job.task_id}_swing_summary.csv",
                        "log_url": f"/download/res_{job.task_id}_log.json",
                        "sub_videos_url": f"/download/{job.task_id}_sub_videos.zip",
                        "id_pose_summary": [],
                    },
                )
                self.evt_q.put({"type": "done", "task_id": job.task_id})

            threading.Thread(target=_sim, daemon=True).start()

    def send_control(self, task_id: str, op: str, value: Any = None) -> None:
        self.controls.append((str(task_id), str(op), value))

    def shutdown(self) -> None:
        pass
