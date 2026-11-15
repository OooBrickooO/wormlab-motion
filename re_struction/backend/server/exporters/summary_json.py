"""05 §6: 姿态运动学汇总 JSON 导出器 (summary_json.py)。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write(path: str | Path, id_pose_summary: list[dict[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(id_pose_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
