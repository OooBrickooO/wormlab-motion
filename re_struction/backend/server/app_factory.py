"""05 §7: FastAPI + python-socketio ASGI 应用工厂 (app_factory.py)。"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.model_registry import ModelRegistry
from wormlab.config import Config, load_config

from .events import EventPump
from .routes import egg, media, models, preview, tasks, upload
from .sockets import create_sio
from .tasks import FakeWorker, TaskManager, TaskStore, WorkerProcess


def create_app(
    cfg: Config | None = None,
    data_dir: str | Path | None = None,
    models_dir: str | Path | None = None,
    use_fake_worker: bool = False,
    fake_auto_complete: bool = False,
) -> tuple[FastAPI, socketio.AsyncServer, socketio.ASGIApp]:
    app_cfg = cfg if cfg is not None else load_config("crawl_agar")
    re_backend = Path(__file__).resolve().parents[1]

    resolved_data = (
        Path(data_dir)
        if data_dir is not None
        else (re_backend / app_cfg.paths.data_dir).resolve()
    )
    uploads_dir = resolved_data / "uploads"
    results_dir = resolved_data / "results"
    cache_dir = resolved_data / "cache"
    tasks_dir = results_dir / "tasks"
    for d in (uploads_dir, results_dir, cache_dir, tasks_dir):
        d.mkdir(parents=True, exist_ok=True)

    store = TaskStore(tasks_dir)
    store.load_history()

    worker: WorkerProcess | FakeWorker
    if use_fake_worker:
        worker = FakeWorker(resolved_data, auto_complete=fake_auto_complete)
    else:
        worker = WorkerProcess(resolved_data)

    manager = TaskManager(store=store, worker=worker)
    sio = create_sio()
    pump = EventPump(evt_q=worker.evt_q, store=store, manager=manager, sio=sio)
    registry = ModelRegistry(app_cfg, models_dir=models_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        loop = asyncio.get_running_loop()
        worker.start()
        pump.start(loop=loop)
        try:
            yield
        finally:
            pump.stop()
            worker.shutdown()
            store.cancel_active_on_shutdown()

    fastapi_app = FastAPI(title="WormLab Motion API", version="2.0.0", lifespan=lifespan)
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=list(app_cfg.server.cors_origins) + ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    fastapi_app.state.cfg = app_cfg
    fastapi_app.state.data_dir = resolved_data
    fastapi_app.state.uploads_dir = uploads_dir
    fastapi_app.state.results_dir = results_dir
    fastapi_app.state.cache_dir = cache_dir
    fastapi_app.state.store = store
    fastapi_app.state.worker = worker
    fastapi_app.state.manager = manager
    fastapi_app.state.pump = pump
    fastapi_app.state.registry = registry
    fastapi_app.state.sio = sio

    fastapi_app.include_router(upload.router)
    fastapi_app.include_router(preview.router)
    fastapi_app.include_router(tasks.router)
    fastapi_app.include_router(media.router)
    fastapi_app.include_router(models.router)
    fastapi_app.include_router(egg.router)

    asgi_app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)
    return fastapi_app, sio, asgi_app
