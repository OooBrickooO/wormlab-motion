"""05 §10 & API_CONTRACT §1: POST /egg_count 虫卵计数路由 (egg.py)。"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from services.egg_count import run_egg_count

router = APIRouter()


@router.post("/egg_count")
async def egg_count_route(request: Request) -> JSONResponse:
    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        return JSONResponse({"error": "未提供文件（file 字段缺失）"}, status_code=400)

    filename = getattr(upload, "filename", "upload.jpg") or "upload.jpg"
    file_bytes = await upload.read()
    if not file_bytes:
        return JSONResponse({"error": "文件内容为空"}, status_code=400)

    registry = request.app.state.registry
    results_dir = request.app.state.results_dir
    try:
        res = run_egg_count(
            file_bytes=file_bytes,
            filename=str(filename),
            registry=registry,
            result_dir=results_dir,
        )
        return JSONResponse(res)
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)
