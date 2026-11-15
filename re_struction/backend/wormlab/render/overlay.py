"""05 §3: 主画面纯白极简叠加渲染 (overlay.py) —— 1:1 复刻旧实现并支持后进入虫灰框标记。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .fonts import get_fonts


@dataclass
class DrawItem:
    track_id: int
    bbox: np.ndarray | list[float]
    is_ghost: bool = False
    lost_frames: int = 0
    user_label: str | None = None
    contour: np.ndarray | None = None  # (K, 2) float32 整帧外轮廓折线
    head_xy: tuple[float, float] | None = None
    trail: list[tuple[float, float]] = field(default_factory=list)
    counted: bool = True
    counts: dict[str, Any] = field(default_factory=dict)


def _draw_dashed_rect(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash: int = 10,
    gap: int = 6,
) -> None:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    step = max(1, dash + gap)
    for x in range(x1, x2, step):
        cv2.line(frame, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(frame, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, step):
        cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def draw_frame(
    frame_bgr: np.ndarray,
    items: list[DrawItem],
    frame_idx: int,
    active_count: int | None = None,
    ghost_count: int | None = None,
) -> np.ndarray:
    """在 BGR 帧副本上绘制尾迹、轮廓、边框、PIL 药丸徽章、头点与左上角统计信息。"""
    canvas = frame_bgr.copy()

    if active_count is None:
        active_count = sum(1 for it in items if not it.is_ghost)
    if ghost_count is None:
        ghost_count = sum(1 for it in items if it.is_ghost)

    if len(items) > 0:
        # 1. 绘制活跃目标最近 30 帧渐变抗锯齿绿线尾迹
        for it in items:
            if it.is_ghost or not it.trail:
                continue
            hist = list(it.trail)[-30:]
            if len(hist) > 1:
                pts = [[int(p[0]), int(p[1])] for p in hist if len(p) >= 2 and np.isfinite(p[0]) and np.isfinite(p[1])]
                if len(pts) > 1:
                    p_arr = np.array(pts, dtype=np.int32)
                    n_pts = len(p_arr)
                    for j in range(1, n_pts):
                        color_intensity = int(255 * (j / n_pts))
                        cv2.line(
                            canvas,
                            tuple(p_arr[j - 1]),
                            tuple(p_arr[j]),
                            (0, color_intensity, 0),
                            2,
                            lineType=cv2.LINE_AA,
                        )

        # 2. 绘制纯白外轮廓折线
        for it in items:
            if it.is_ghost or it.contour is None or len(it.contour) < 3:
                continue
            cnt = np.round(np.asarray(it.contour, dtype=np.float32)).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [cnt], isClosed=True, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # 3. 绘制边界框（普通白色 / 后进入灰色 / 幽灵浅绿虚线）
        for it in items:
            x1, y1, x2, y2 = [int(v) for v in it.bbox[:4]]
            if it.is_ghost:
                _draw_dashed_rect(canvas, (x1, y1, x2, y2), (160, 220, 160), 1)
            elif not it.counted:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (150, 150, 150), 2)
            else:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # 4. 绘制药丸徽章（PIL 矢量抗锯齿渲染）
        font_top, font_bot = get_fonts()
        img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        frame_h = int(canvas.shape[0])

        for it in items:
            tid_int = int(it.track_id)
            x1, y1, x2, y2 = [int(v) for v in it.bbox[:4]]
            user_lbl = f"[{it.user_label}] " if it.user_label else ""

            if it.is_ghost:
                ghost_text = f"ID {tid_int} GHOST lost={int(it.lost_frames)}f"
                tb = draw.textbbox((0, 0), ghost_text, font=font_top)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
                bx1, by1 = x1, max(0, y1 - th - 8)
                bx2, by2 = bx1 + tw + 10, by1 + th + 8
                draw.rectangle([bx1, by1, bx2, by2], fill=(240, 255, 240), outline=(160, 220, 160), width=1)
                draw.text((bx1 + 5, by1 + 2), ghost_text, fill=(30, 120, 30), font=font_top)
            elif not it.counted:
                late_text = f"ID {tid_int} {user_lbl}后进入·不计数"
                tb = draw.textbbox((0, 0), late_text, font=font_top)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
                bx1, by1 = x1, max(0, y1 - th - 8)
                bx2, by2 = bx1 + tw + 10, by1 + th + 8
                draw.rectangle([bx1, by1, bx2, by2], fill=(235, 235, 235), outline=(150, 150, 150), width=1)
                draw.text((bx1 + 5, by1 + 2), late_text, fill=(60, 60, 60), font=font_top)
            else:
                info = it.counts or {}
                c_cnt = int(info.get("c_count", 0))
                s_cnt = int(info.get("s_count", 0))
                bi_flag = " (双向)" if info.get("is_bidirectional") else ""
                freq = float(info.get("avg_freq", 0.0))
                bpm = float(info.get("bends_per_min", 0.0))
                motion_type = str(info.get("motion_type_legacy") or info.get("motion_type") or "static")
                activity = float(info.get("activity_score", 0.0))

                top_text = f"ID {tid_int} {user_lbl}C:{c_cnt} S:{s_cnt}{bi_flag} {freq:.2f}Hz"
                tb = draw.textbbox((0, 0), top_text, font=font_top)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
                bx1, by1 = x1, max(0, y1 - th - 8)
                bx2, by2 = bx1 + tw + 10, by1 + th + 8
                draw.rectangle([bx1, by1, bx2, by2], fill=(255, 255, 255), outline=(255, 255, 255), width=1)
                draw.text((bx1 + 5, by1 + 2), top_text, fill=(0, 0, 0), font=font_top)

                bot_text = f"#C:{c_cnt} S:{s_cnt} {freq:.2f}Hz {bpm:.0f}/min {motion_type} {activity:.0%}"
                bb = draw.textbbox((0, 0), bot_text, font=font_bot)
                bw, bh = bb[2] - bb[0], bb[3] - bb[1]
                bot_x1, bot_y1 = x1, min(frame_h - bh - 6, y2)
                bot_x2, bot_y2 = bot_x1 + bw + 8, bot_y1 + bh + 6
                draw.rectangle([bot_x1, bot_y1, bot_x2, bot_y2], fill=(255, 255, 255), outline=(255, 255, 255), width=1)
                draw.text((bot_x1 + 4, bot_y1 + 2), bot_text, fill=(0, 0, 0), font=font_bot)

        canvas = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # 5. 绘制头部圆点（半径 3px）
        for it in items:
            if not it.is_ghost and it.head_xy is not None:
                hx_f, hy_f = float(it.head_xy[0]), float(it.head_xy[1])
                if np.isfinite(hx_f) and np.isfinite(hy_f):
                    cv2.circle(canvas, (int(hx_f), int(hy_f)), 3, (0, 255, 100), -1, lineType=cv2.LINE_AA)

    # 6. 左上角全局帧统计
    cv2.putText(
        canvas,
        f"Frame: {int(frame_idx)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Active IDs: {int(active_count)} (Ghost: {int(ghost_count)})",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    return canvas
