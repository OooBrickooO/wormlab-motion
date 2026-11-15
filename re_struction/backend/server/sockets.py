"""05 §7 & API_CONTRACT §2: Socket.IO 异步服务端与房间订阅 (sockets.py)。"""
from __future__ import annotations

from typing import Any

import socketio


def create_sio() -> socketio.AsyncServer:
    sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")

    @sio.on("join")
    async def handle_join(sid: str, data: Any) -> dict[str, Any]:
        task_id = data.get("task_id") if isinstance(data, dict) else str(data or "")
        if task_id:
            await sio.enter_room(sid, str(task_id))
        return {"ok": True, "task_id": task_id}

    return sio
