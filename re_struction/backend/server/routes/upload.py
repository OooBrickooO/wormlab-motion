"""05 §7 & API_CONTRACT §1: POST /upload 上传与任务创建路由 (upload.py)。"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from server.tasks import generate_task_id
from wormlab.pipeline.pass_a import JobSpec

router = APIRouter()

VALID_PROFILES: set[str] = {"crawl_agar", "swim_liquid"}
VALID_C_UNITS: set[str] = {"cycle", "bend"}


def _secure_filename(name: str) -> str:
    clean = Path(name or "upload.mp4").name
    clean = re.sub(r"[^A-Za-z0-9_.\-\u4e00-\u9fa5×（）()]", "_", clean)
    return clean or "upload.mp4"


@router.post("/upload")
async def upload_video(request: Request) -> JSONResponse:
    form = await request.form()
    video_file = form.get("video")
    if video_file is None or not hasattr(video_file, "read"):
        return JSONResponse({"error": "No video file provided"}, status_code=400)

    raw_filename = getattr(video_file, "filename", "") or ""
    if not raw_filename:
        return JSONResponse({"error": "No selected file"}, status_code=400)

    cfg = request.app.state.cfg
    ext = Path(raw_filename).suffix.lstrip(".").lower()
    allowed = {e.lower() for e in (cfg.server.allowed_ext + cfg.server.image_ext)}
    if ext not in allowed:
        return JSONResponse(
            {"error": f"不支持的文件格式 .{ext}，允许的格式: {sorted(allowed)}"},
            status_code=400,
        )

    worm_count_raw = form.get("worm_count")
    if worm_count_raw is None or str(worm_count_raw).strip() == "":
        return JSONResponse({"error": "请填写初始虫子数量（worm_count 为必填正整数）"}, status_code=400)
    try:
        worm_count = int(str(worm_count_raw).strip())
        if worm_count <= 0:
            raise ValueError("must be > 0")
    except ValueError:
        return JSONResponse({"error": "worm_count 必须为正整数"}, status_code=400)

    profile = str(form.get("profile") or "crawl_agar").strip()
    if profile not in VALID_PROFILES:
        return JSONResponse({"error": f"无效的 profile: {profile}"}, status_code=400)

    c_unit = str(form.get("c_unit") or cfg.counting.c_unit).strip()
    if c_unit not in VALID_C_UNITS:
        return JSONResponse({"error": f"无效的 c_unit: {c_unit}"}, status_code=400)

    head_config: list[dict[str, Any]] = []
    hc_raw = form.get("head_config")
    if hc_raw:
        try:
            parsed = json.loads(str(hc_raw))
            if isinstance(parsed, list):
                head_config = parsed
            elif isinstance(parsed, dict):
                head_config = list(parsed.values())
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": f"head_config JSON 解析失败: {exc}"}, status_code=400)

    model_name_raw = form.get("model_name")
    registry = request.app.state.registry
    model_path = registry.model_path(str(model_name_raw) if model_name_raw else None)

    content = await video_file.read()
    if not content:
        return JSONResponse({"error": "上传文件内容为空"}, status_code=400)

    task_id = generate_task_id()
    safe_name = _secure_filename(raw_filename)
    uploads_dir: Path = request.app.state.uploads_dir
    uploads_dir.mkdir(parents=True, exist_ok=True)
    saved_path = uploads_dir / f"{task_id}_{safe_name}"
    saved_path.write_bytes(content)

    store = request.app.state.store
    manager = request.app.state.manager
    results_dir: Path = request.app.state.results_dir
    cache_root: Path = request.app.state.cache_dir

    store.create(
        task_id=task_id,
        filename=safe_name,
        video_path=str(saved_path),
        worm_count=worm_count,
        profile=profile,
        c_unit=c_unit,
        queued=True,
    )
    job = JobSpec(
        task_id=task_id,
        video_path=str(saved_path),
        profile=profile,
        c_unit=c_unit,
        worm_count=worm_count,
        head_config=head_config,
        model_path=str(model_path),
        result_dir=str(results_dir),
        cache_dir=str(cache_root / task_id),
    )
    manager.submit(job)
    return JSONResponse({"task_id": task_id}, status_code=202)
