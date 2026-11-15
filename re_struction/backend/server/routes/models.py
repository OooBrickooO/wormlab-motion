"""05 §7 & API_CONTRACT §1: 模型枚举/切换、Profile 枚举与健康检查路由 (models.py)。"""
from __future__ import annotations

import torch
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()

PROFILES_LIST = [
    {"id": "crawl_agar", "label": "琼脂爬行 (S形行波)"},
    {"id": "swim_liquid", "label": "液体游泳 (C形摆动)"},
]


@router.get("/models")
async def get_models(request: Request) -> JSONResponse:
    registry = request.app.state.registry
    return JSONResponse(
        {
            "models": registry.list_models(),
            "current_model": registry.current_name,
        }
    )


@router.post("/select_model")
async def select_model(request: Request) -> JSONResponse:
    model_name: str | None = None
    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                model_name = body.get("model_name")
        except Exception:  # noqa: BLE001
            model_name = None
    else:
        form = await request.form()
        val = form.get("model_name")
        model_name = str(val) if val is not None else None

    if not model_name:
        return JSONResponse({"error": "未提供 model_name"}, status_code=400)

    registry = request.app.state.registry
    try:
        registry.set_current(str(model_name))
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)

    return JSONResponse(
        {
            "ok": True,
            "current_model": registry.current_name,
            "message": f"模型已成功切换为 {registry.current_name}",
        }
    )


@router.get("/profiles")
async def get_profiles() -> JSONResponse:
    return JSONResponse(
        {
            "profiles": PROFILES_LIST,
            "default": "crawl_agar",
        }
    )


@router.get("/health")
async def get_health(request: Request) -> JSONResponse:
    registry = request.app.state.registry
    return JSONResponse(
        {
            "ok": True,
            "gpu": bool(torch.cuda.is_available()),
            "models": registry.list_models(),
        }
    )
