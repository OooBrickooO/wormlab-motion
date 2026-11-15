"""05 §8: 任务排队与运行态调度管理器 (manager.py)。"""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from typing import Any

from wormlab.pipeline.pass_a import JobSpec

from .store import TaskStore
from .worker import FakeWorker, WorkerProcess


def generate_task_id() -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


class TaskManager:
    """串行任务队列管理器：维护排队中的 JobSpec 与当前运行任务，转发暂停/恢复/取消/动态虫数指令。"""

    def __init__(self, store: TaskStore, worker: WorkerProcess | FakeWorker) -> None:
        self.store = store
        self.worker = worker
        self._lock = threading.Lock()
        self.queue: deque[JobSpec] = deque()
        self.current_task_id: str | None = None

    def submit(self, job: JobSpec) -> None:
        with self._lock:
            if self.current_task_id is None:
                self.current_task_id = job.task_id
                self.store.update(job.task_id, queued=False, status="processing", control="running")
                self.worker.send_start(job, initial_control="running")
            else:
                self.queue.append(job)
                self.store.update(job.task_id, queued=True, status="processing", control="running")

    def control(self, task_id: str, op: str, value: Any = None) -> dict[str, Any] | None:
        tid = str(task_id)
        st = self.store.get(tid)
        if st is None:
            return None

        with self._lock:
            is_running = self.current_task_id == tid
            queued_job = next((j for j in self.queue if j.task_id == tid), None)

            if op == "pause":
                updated = self.store.update(tid, status="paused", control="paused")
                if is_running:
                    self.worker.send_control(tid, "pause")
                return updated

            if op == "resume":
                updated = self.store.update(tid, status="processing", control="running")
                if is_running:
                    self.worker.send_control(tid, "resume")
                return updated

            if op == "cancel":
                if queued_job is not None:
                    self.queue.remove(queued_job)
                updated = self.store.update(tid, status="cancelled", control="cancelled", queued=False)
                if is_running:
                    self.worker.send_control(tid, "cancel")
                return updated

            if op == "set_worm_count":
                new_val = int(value) if (value is not None and int(value) > 0) else None
                if queued_job is not None:
                    queued_job.worm_count = new_val
                updated = self.store.update(tid, worm_count=new_val)
                if is_running:
                    self.worker.send_control(tid, "set_worm_count", new_val)
                return updated

        return st

    def on_done(self, task_id: str) -> None:
        tid = str(task_id)
        self.store.persist_if_terminal(tid)
        with self._lock:
            if self.current_task_id == tid:
                self.current_task_id = None
            while self.queue:
                next_job = self.queue.popleft()
                st = self.store.get(next_job.task_id)
                if st is None or st.get("status") == "cancelled":
                    continue
                self.current_task_id = next_job.task_id
                init_ctrl = "paused" if st.get("control") == "paused" else "running"
                self.store.update(
                    next_job.task_id,
                    queued=False,
                    status="paused" if init_ctrl == "paused" else "processing",
                )
                self.worker.send_start(next_job, initial_control=init_ctrl)
                break
