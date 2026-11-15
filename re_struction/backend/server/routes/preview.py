"""05 §10 & API_CONTRACT §1: 首帧数量预览、首帧关键点分析与手动补框提取路由 (preview.py)。"""
from __future__ import annotations

import base64
import json
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from wormlab.pose.centerline import extract_pose

router = APIRouter()


async def _read_frame_from_form(request: Request) -> tuple[np.ndarray | None, dict[str, Any], JSONResponse | None]:
    form = await request.form()
    frame_file = form.get("frame")
    if frame_file is None or not hasattr(frame_file, "read"):
        return None, dict(form), JSONResponse({"error": "未提供帧图像（frame 字段缺失）"}, status_code=400)
    img_bytes = await frame_file.read()
    if not img_bytes:
        return None, dict(form), JSONResponse({"error": "帧图像为空"}, status_code=400)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, dict(form), JSONResponse({"error": "无法解码帧图像"}, status_code=400)
    return frame, dict(form), None


@router.post("/preview_count")
async def preview_count(request: Request) -> JSONResponse:
    frame, _, err_resp = await _read_frame_from_form(request)
    if err_resp is not None or frame is None:
        return err_resp  # type: ignore[return-value]

    try:
        registry = request.app.state.registry
        det = registry.get_preview_detector()
        with registry.lock:
            dets = det.infer([frame])[0]

        count = len(dets)
        annotated = frame.copy()
        for i, d in enumerate(dets):
            bx1, by1, bx2, by2 = [int(v) for v in d.bbox[:4]]
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 200, 80), 2)
            cv2.putText(
                annotated,
                f"#{i + 1} {d.conf:.2f}",
                (bx1, max(14, by1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 200, 80),
                1,
                lineType=cv2.LINE_AA,
            )
        cv2.putText(
            annotated,
            f"detected: {count}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 200, 80),
            2,
            lineType=cv2.LINE_AA,
        )
        _, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return JSONResponse({"count": count, "raw_count": count, "annotated": b64})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/analyze_first_frame")
async def analyze_first_frame(request: Request) -> JSONResponse:
    frame, _, err_resp = await _read_frame_from_form(request)
    if err_resp is not None or frame is None:
        return err_resp  # type: ignore[return-value]

    try:
        height, width = frame.shape[:2]
        cfg = request.app.state.cfg
        registry = request.app.state.registry
        det = registry.get_preview_detector()
        with registry.lock:
            dets = det.infer([frame])[0]

        worms: list[dict[str, Any]] = []
        annotated = frame.copy()

        for i, d in enumerate(dets):
            bbox = [float(v) for v in d.bbox[:4]]
            x1, y1, x2, y2 = bbox
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)

            keypoints: list[list[float]] | None = None
            if d.mask is not None and d.mask.size > 0:
                pose = extract_pose(d.mask, d.mask_origin, cfg.pose)
                pts11 = pose.pts11
                if pts11 is not None and len(pts11) == 11:
                    keypoints = [[round(float(pt[0]), 1), round(float(pt[1]), 1)] for pt in pts11]
                if pose.contour is not None and len(pose.contour) >= 3:
                    cnt = np.round(pose.contour).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated, [cnt], True, (255, 180, 60), 2, lineType=cv2.LINE_AA)

            worms.append(
                {
                    "det_idx": i,
                    "confidence": round(float(d.conf), 4),
                    "bbox": [round(v, 1) for v in bbox],
                    "centroid": [round(cx, 1), round(cy, 1)],
                    "keypoints": keypoints,
                }
            )

            bx1, by1, bx2, by2 = map(int, bbox)
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (22, 119, 255), 2)
            cv2.putText(
                annotated,
                f"#{i} {d.conf:.2f}",
                (bx1, max(14, by1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (22, 119, 255),
                1,
            )
            if keypoints:
                for ki, (kx, ky) in enumerate(keypoints):
                    color = (0, 220, 80) if ki in (0, 10) else (200, 200, 200)
                    radius = 5 if ki in (0, 10) else 3
                    cv2.circle(annotated, (int(kx), int(ky)), radius, color, -1)
                    cv2.putText(
                        annotated,
                        str(ki),
                        (int(kx) + 6, int(ky) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        (255, 255, 0),
                        1,
                    )

        cv2.putText(annotated, f"detected: {len(worms)}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 80), 2)
        _, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return JSONResponse(
            {
                "frame_width": width,
                "frame_height": height,
                "annotated": b64,
                "worms": worms,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/extract_manual_worm")
async def extract_manual_worm(request: Request) -> JSONResponse:
    frame, form_dict, err_resp = await _read_frame_from_form(request)
    if err_resp is not None or frame is None:
        return err_resp  # type: ignore[return-value]

    bbox_str = form_dict.get("bbox")
    if not bbox_str:
        return JSONResponse({"error": "未提供 bbox"}, status_code=400)
    try:
        bbox_raw = json.loads(str(bbox_str))
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_raw[:4]]
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"bbox 格式错误: {exc}"}, status_code=400)

    h, w = frame.shape[:2]
    rx1, rx2 = max(0, min(x1, x2)), min(w, max(x1, x2))
    ry1, ry2 = max(0, min(y1, y2)), min(h, max(y1, y2))
    if rx2 - rx1 < 6 or ry2 - ry1 < 6:
        return JSONResponse({"error": "选区过小"}, status_code=400)

    roi = frame[ry1:ry2, rx1:rx2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edge_mean = (
        float(np.mean(blurred[0, :]))
        + float(np.mean(blurred[-1, :]))
        + float(np.mean(blurred[:, 0]))
        + float(np.mean(blurred[:, -1]))
    ) / 4.0
    center_mean = float(np.mean(blurred))
    if center_mean < edge_mean:
        _, mask_u8 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, mask_u8 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    keypoints: list[list[float]] | None = None
    try:
        cfg = request.app.state.cfg
        pose = extract_pose(mask_u8 > 0, (rx1, ry1), cfg.pose)
        pts11 = pose.pts11
        if pts11 is not None and len(pts11) == 11:
            keypoints = [[round(float(pt[0]), 1), round(float(pt[1]), 1)] for pt in pts11]
    except Exception:  # noqa: BLE001
        keypoints = None

    if keypoints is None:
        keypoints = [
            [round(float(rx1 + (rx2 - rx1) * i / 10.0), 1), round(float(ry1 + (ry2 - ry1) * i / 10.0), 1)]
            for i in range(11)
        ]

    cx = 0.5 * (rx1 + rx2)
    cy = 0.5 * (ry1 + ry2)
    return JSONResponse(
        {
            "bbox": [rx1, ry1, rx2, ry2],
            "centroid": [round(cx, 1), round(cy, 1)],
            "keypoints": keypoints,
        }
    )
