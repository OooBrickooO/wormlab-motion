"""05 §3: PIL 矢量字体缓存加载器 (fonts.py)。"""
from __future__ import annotations

import os
from functools import lru_cache

from PIL import ImageFont

FONT_CANDIDATES: tuple[str, ...] = (
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/arial.ttf",
)


@lru_cache(maxsize=8)
def get_fonts(
    top_size: int = 15,
    bot_size: int = 13,
    candidates: tuple[str, ...] = FONT_CANDIDATES,
) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, ImageFont.FreeTypeFont | ImageFont.ImageFont]:
    """返回缓存的 (顶部药丸字体, 底部药丸字体)。"""
    font_path: str | None = None
    for p in candidates:
        if os.path.exists(p):
            font_path = p
            break
    if font_path is not None:
        return ImageFont.truetype(font_path, int(top_size)), ImageFont.truetype(font_path, int(bot_size))
    default_f = ImageFont.load_default()
    return default_f, default_f
