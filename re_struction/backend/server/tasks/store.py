"""05 §9 & API_CONTRACT §3: 运行时内存任务状态存储与终态原子落盘/恢复 (store.py)。"""
from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from typing import Any

TERMINAL_STATUSES: set[str] = {"completed", "failed", "cancelled"}


class TaskStore:
    """线程安全的任务状态存储：运行期驻留内存，进入终态时原子写盘（剥离 latest_frame），启动时自动恢复历史。"""

    def __init__(self, tasks_dir: str | Path) -> None:
        self.tasks_dir = Path(tasks_dir)
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._tasks: dict[str, dict[str, Any]] = {}

    def load_history(self) -> int:
        count = 0
        if not self.tasks_dir.exists():
            return 0
        with self._lock:
            for fp in sorted(self.tasks_dir.glob("*.json")):
                try:
                    data = json.loads(fp.read_text(encoding="utf-8"))
                    tid = fp.stem
                    data.setdefault("latest_frame", "")
                    if data.get("status") not in TERMINAL_STATUSES:
                        data["status"] = "cancelled"
                    data["control"] = data["status"]
                    data["queued"] = False
                    self._tasks[tid] = data
                    count += 1
                except Exception:  # noqa: BLE001
                    continue
        return count

    def create(
        self,
        task_id: str,
        filename: str,
        video_path: str,
        worm_count: int | None,
        profile: str = "crawl_agar",
        c_unit: str = "cycle",
        queued: bool = True,
    ) -> dict[str, Any]:
        state: dict[str, Any] = {
            "status": "processing",
            "video_path": str(video_path),
            "filename": str(filename),
            "current_frame": 0,
            "total_frames": 0,
            "active_tracks": 0,
            "latest_frame": "",
            "tracks": [],
            "video_url": "",
            "log_url": "",
            "sub_videos_url": "",
            "summary_url": "",
            "csv_url": "",
            "id_pose_summary": [],
            "error_message": "",
            "control": "running",
            "worm_count": worm_count,
            "profile": str(profile),
            "c_unit": str(c_unit),
            "phase": "detect",
            "queued": bool(queued),
        }
        with self._lock:
            self._tasks[str(task_id)] = state
            return copy.deepcopy(state)

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            st = self._tasks.get(str(task_id))
            return copy.deepcopy(st) if st is not None else None

    def update(self, task_id: str, **kwargs: Any) -> dict[str, Any] | None:
        tid = str(task_id)
        persist_snapshot: dict[str, Any] | None = None
        with self._lock:
            st = self._tasks.get(tid)
            if st is None:
                return None
            cur_status = st.get("status")
            incoming_status = kwargs.get("status")
            # 终态任务禁止被异步事件泵中的过期进度/状态消息回退覆盖（尤其已取消的任务）
            if cur_status == "cancelled":
                return copy.deepcopy(st)
            if cur_status in TERMINAL_STATUSES and incoming_status is not None and incoming_status != cur_status:
                return copy.deepcopy(st)
            st.update(kwargs)
            if st.get("status") in TERMINAL_STATUSES:
                st["queued"] = False
                st["control"] = st["status"]
                persist_snapshot = copy.deepcopy(st)
            res = copy.deepcopy(st)

        if persist_snapshot is not None:
            self._persist_terminal(tid, persist_snapshot)
        return res

    def cancel_active_on_shutdown(self) -> int:
        """服务关闭时将所有仍未进入终态的任务收敛为 cancelled 并落盘，避免重启后前端轮询/取消 404。"""
        to_persist: list[tuple[str, dict[str, Any]]] = []
        with self._lock:
            for tid, st in self._tasks.items():
                if st.get("status") not in TERMINAL_STATUSES:
                    st["status"] = "cancelled"
                    st["control"] = "cancelled"
                    st["queued"] = False
                    if not st.get("error_message"):
                        st["error_message"] = "服务关闭，任务已终止"
                    to_persist.append((tid, copy.deepcopy(st)))
        for tid, snap in to_persist:
            self._persist_terminal(tid, snap)
        return len(to_persist)

    def persist_if_terminal(self, task_id: str) -> None:
        tid = str(task_id)
        with self._lock:
            st = self._tasks.get(tid)
            if st is None or st.get("status") not in TERMINAL_STATUSES:
                return
            snapshot = copy.deepcopy(st)
        self._persist_terminal(tid, snapshot)

    def _persist_terminal(self, task_id: str, snapshot: dict[str, Any]) -> None:
        to_save = dict(snapshot)
        to_save.pop("latest_frame", None)
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        final_path = self.tasks_dir / f"{task_id}.json"
        tmp_path = self.tasks_dir / f".{task_id}.json.tmp"
        tmp_path.write_text(json.dumps(to_save, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(str(tmp_path), str(final_path))
