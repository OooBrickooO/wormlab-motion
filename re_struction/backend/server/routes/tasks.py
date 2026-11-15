"""05 §7 & API_CONTRACT §1: 任务状态查询与暂停/恢复/取消/动态虫数/子视频列表路由 (tasks.py)。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/task/{task_id}/status")
async def get_task_status(task_id: str, request: Request) -> JSONResponse:
    pump = getattr(request.app.state, "pump", None)
    if pump is not None:
        pump.drain_once()
    store = request.app.state.store
    state = store.get(task_id)
    if state is None:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return JSONResponse(state)


@router.post("/task/{task_id}/pause")
async def pause_task(task_id: str, request: Request) -> JSONResponse:
    manager = request.app.state.manager
    updated = manager.control(task_id, "pause")
    if updated is None:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return JSONResponse({"ok": True, "task_id": task_id, "status": "paused"})


@router.post("/task/{task_id}/resume")
async def resume_task(task_id: str, request: Request) -> JSONResponse:
    manager = request.app.state.manager
    updated = manager.control(task_id, "resume")
    if updated is None:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return JSONResponse({"ok": True, "task_id": task_id, "status": "processing"})


@router.post("/task/{task_id}/cancel")
async def cancel_task(task_id: str, request: Request) -> JSONResponse:
    manager = request.app.state.manager
    updated = manager.control(task_id, "cancel")
    if updated is None:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return JSONResponse({"ok": True, "task_id": task_id, "status": "cancelled"})


@router.post("/task/{task_id}/set_worm_count")
async def set_worm_count(task_id: str, request: Request) -> JSONResponse:
    store = request.app.state.store
    if store.get(task_id) is None:
        return JSONResponse({"error": "Task not found"}, status_code=404)

    raw_val: Any = None
    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                raw_val = body.get("worm_count")
        except Exception:  # noqa: BLE001
            raw_val = None
    else:
        form = await request.form()
        raw_val = form.get("worm_count")

    if raw_val is None:
        return JSONResponse({"error": "worm_count 必须为整数"}, status_code=400)
    try:
        n_int = int(str(raw_val).strip())
    except ValueError:
        return JSONResponse({"error": "worm_count 必须为整数"}, status_code=400)

    final_n = n_int if n_int > 0 else None
    manager = request.app.state.manager
    manager.control(task_id, "set_worm_count", final_n)
    msg = f"虫数上限已设为 {final_n}" if final_n is not None else "已取消虫数上限"
    return JSONResponse({"ok": True, "task_id": task_id, "worm_count": final_n, "message": msg})


@router.get("/task/{task_id}/sub_videos")
async def list_sub_videos(task_id: str, request: Request) -> JSONResponse:
    results_dir: Path = request.app.state.results_dir
    sub_dir = results_dir / "sub_videos" / task_id
    videos: list[dict[str, str]] = []
    if sub_dir.exists():
        for fp in sorted(sub_dir.glob("*.mp4"), key=lambda p: p.name):
            videos.append(
                {
                    "filename": fp.name,
                    "url": f"/sub_video/{task_id}/{fp.name}",
                }
            )
    return JSONResponse({"videos": videos})
