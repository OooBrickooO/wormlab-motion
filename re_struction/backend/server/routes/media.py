"""05 §7 & API_CONTRACT §1: 文件下载、在线流媒体 (支持 HTTP Range) 与子视频访问路由 (media.py)。"""
from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse

router = APIRouter()


def _safe_resolve(base_dir: Path, filename: str) -> Path | None:
    clean_name = Path(filename).name
    if not clean_name or clean_name in {".", ".."}:
        return None
    target = (base_dir / clean_name).resolve()
    if not str(target).startswith(str(base_dir.resolve())):
        return None
    if not target.exists() or not target.is_file():
        return None
    return target


def _serve_with_range(file_path: Path, request: Request, media_type: str = "video/mp4") -> Response:
    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")
    if not range_header or not range_header.startswith("bytes="):
        return FileResponse(
            file_path,
            media_type=media_type,
            headers={"Accept-Ranges": "bytes"},
        )

    try:
        byte_range = range_header.replace("bytes=", "").split(",")[0].strip()
        start_str, end_str = byte_range.split("-")
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
    except ValueError:
        return FileResponse(file_path, media_type=media_type)

    start = max(0, min(start, file_size - 1))
    end = max(start, min(end, file_size - 1))
    chunk_len = end - start + 1

    def _iter_file(path: Path, offset: int, remaining: int, buf_size: int = 262144):
        with open(path, "rb") as f:
            f.seek(offset)
            while remaining > 0:
                data = f.read(min(buf_size, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(chunk_len),
    }
    return StreamingResponse(
        _iter_file(file_path, start, chunk_len),
        status_code=206,
        media_type=media_type,
        headers=headers,
    )


@router.get("/download/{filename}")
async def download_file(filename: str, request: Request) -> Response:
    results_dir: Path = request.app.state.results_dir
    target = _safe_resolve(results_dir, filename)
    if target is None:
        return PlainTextResponse("File not found", status_code=404)
    return FileResponse(target, filename=target.name)


@router.get("/media/{filename}")
async def serve_media(filename: str, request: Request) -> Response:
    results_dir: Path = request.app.state.results_dir
    target = _safe_resolve(results_dir, filename)
    if target is None:
        return PlainTextResponse("File not found", status_code=404)
    mime, _ = mimetypes.guess_type(target.name)
    return _serve_with_range(target, request, media_type=mime or "video/mp4")


@router.get("/sub_video/{task_id}/{filename}")
async def serve_sub_video(task_id: str, filename: str, request: Request) -> Response:
    results_dir: Path = request.app.state.results_dir
    clean_tid = Path(task_id).name
    sub_dir = results_dir / "sub_videos" / clean_tid
    target = _safe_resolve(sub_dir, filename)
    if target is None:
        return PlainTextResponse("File not found", status_code=404)
    return _serve_with_range(target, request, media_type="video/mp4")
