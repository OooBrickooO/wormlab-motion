"""05 §10: 虫卵计数推理与 Eggcount/<stem>.json 导出服务 (egg_count.py)。"""
from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .model_registry import ModelRegistry

VIDEO_EXTS: set[str] = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def extract_first_frame_from_bytes(file_bytes: bytes, filename: str) -> np.ndarray:
    ext = Path(filename or "upload.jpg").suffix.lower()
    if ext in VIDEO_EXTS:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
            tf.write(file_bytes)
            tmp_path = tf.name
        try:
            cap = cv2.VideoCapture(tmp_path)
            ok, frame = cap.read()
            cap.release()
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        if not ok or frame is None:
            raise ValueError("无法从视频中提取第一帧")
        return frame

    nparr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("无法解码图片，请确认文件格式")
    return frame


def run_egg_count(
    file_bytes: bytes,
    filename: str,
    registry: ModelRegistry,
    result_dir: str | Path,
) -> dict[str, Any]:
    frame = extract_first_frame_from_bytes(file_bytes, filename)
    height, width = frame.shape[:2]

    egg_model, egg_device = registry.get_egg_model()
    model_names: dict[int, str] = getattr(egg_model, "names", None) or {0: "egg", 1: "worm"}
    egg_class_ids = [cid for cid, name in model_names.items() if "egg" in str(name).lower()]
    egg_class_name = model_names[egg_class_ids[0]] if egg_class_ids else "egg"

    with registry.lock:
        results = egg_model(frame, verbose=False, device=egg_device)[0]

    boxes = getattr(results, "boxes", None)
    eggs: list[dict[str, Any]] = []

    if boxes is not None and len(boxes.xyxy) > 0:
        xyxy_np = boxes.xyxy.detach().cpu().numpy() if hasattr(boxes.xyxy, "detach") else np.asarray(boxes.xyxy)
        conf_np = boxes.conf.detach().cpu().numpy() if hasattr(boxes.conf, "detach") else np.asarray(boxes.conf)
        cls_np = (
            boxes.cls.detach().cpu().numpy().astype(int)
            if hasattr(boxes.cls, "detach")
            else np.asarray(boxes.cls, dtype=int)
        )
        for i in range(len(cls_np)):
            cid = int(cls_np[i])
            conf = float(conf_np[i])
            bbox = [round(float(v), 1) for v in xyxy_np[i][:4]]
            entry = {"bbox": bbox, "confidence": round(conf, 4), "class_id": cid}
            if not egg_class_ids or cid in egg_class_ids:
                eggs.append(entry)

    egg_count_90 = sum(1 for det in eggs if det["confidence"] >= 0.90)
    egg_count_95 = sum(1 for det in eggs if det["confidence"] >= 0.95)

    eggcount_dir = Path(result_dir) / "Eggcount"
    eggcount_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filename or "upload").stem or "upload"
    json_path = eggcount_dir / f"{stem}.json"

    egg_details = {str(idx): f"{det['confidence'] * 100:.1f}%" for idx, det in enumerate(eggs, start=1)}
    json_path.write_text(json.dumps(egg_details, ensure_ascii=False, indent=2), encoding="utf-8")

    _, buf = cv2.imencode(".png", frame)
    raw_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    return {
        "egg_count": len(eggs),
        "egg_count_90": egg_count_90,
        "egg_count_95": egg_count_95,
        "egg_class_name": egg_class_name,
        "raw_image": raw_b64,
        "eggs": eggs,
        "frame_width": width,
        "frame_height": height,
    }
