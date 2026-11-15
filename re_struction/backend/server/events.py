"""05 §8: 跨进程事件泵守护线程 (events.py)。"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import Any

import socketio

from .tasks.manager import TaskManager
from .tasks.store import TaskStore

logger = logging.getLogger(__name__)


class EventPump:
    """在主服务进程中后台消费 evt_q，更新 TaskStore 并向 Socket.IO 房间广播事件。"""

    def __init__(
        self,
        evt_q: Any,
        store: TaskStore,
        manager: TaskManager,
        sio: socketio.AsyncServer,
    ) -> None:
        self.evt_q = evt_q
        self.store = store
        self.manager = manager
        self.sio = sio
        self.loop: asyncio.AbstractEventLoop | None = None
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._run, name="EventPump", daemon=True)

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self.loop = loop
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def drain_once(self) -> int:
        """同步消费当前队列中的所有待处理事件（方便单元测试立即断言）。"""
        processed = 0
        while True:
            try:
                msg = self.evt_q.get_nowait()
            except queue.Empty:
                break
            except Exception:  # noqa: BLE001
                break
            self._handle_msg(msg)
            processed += 1
        return processed

    def _handle_msg(self, msg: Any) -> None:
        if not isinstance(msg, dict):
            return
        mtype = msg.get("type")
        task_id = str(msg.get("task_id", ""))
        if mtype == "update_state":
            kwargs = msg.get("kwargs") or {}
            self.store.update(task_id, **kwargs)
        elif mtype == "emit":
            event = str(msg.get("event", ""))
            data = msg.get("data") or {}
            room = msg.get("room") or task_id
            if self.loop is not None and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self.sio.emit(event, data, room=room),
                    self.loop,
                )
        elif mtype == "done":
            self.manager.on_done(task_id)

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                msg = self.evt_q.get(timeout=0.1)
            except queue.Empty:
                continue
            except Exception:  # noqa: BLE001
                continue
            try:
                self._handle_msg(msg)
            except Exception as exc:  # noqa: BLE001
                logger.warning("EventPump 处理消息异常: %s", exc)
