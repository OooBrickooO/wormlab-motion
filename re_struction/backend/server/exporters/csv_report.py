"""05 §6: 兼容旧格式 + 扩展双口径列的 UTF-8-SIG CSV 汇总报表导出器 (csv_report.py)。"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from wormlab.render.video_io import ensure_unique_path

CSV_HEADER: list[str] = [
    "track_id",
    "用户标签",
    "运动状态",
    "C形计数",
    "S形计数",
    "是否双向",
    "平均摆动频率(Hz)",
    "实时摆动频率(Hz)",
    "完整周期数",
    "单向摆动数",
    "次/分钟",
    "活性分(0-1)",
    "总运动路径(px)",
    "净位移(px)",
    "C形周期数",
    "C形单次弯曲数",
]


def write(result_dir: str | Path, stem: str, id_pose_summary: list[dict[str, Any]]) -> str:
    """写出 <stem>_swing_summary.csv 并返回生成的文件名（含扩展名）。"""
    out_dir = Path(result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{stem}_swing_summary.csv"
    final_path = ensure_unique_path(raw_path)

    with open(final_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for item in id_pose_summary:
            sp = item.get("spatial_stats") or {}
            c_cnt = int(item.get("c_count", 0))
            s_cnt = int(item.get("s_count", 0))
            c_cyc = int(item.get("c_cycles", c_cnt))
            c_bnd = int(item.get("c_bends", c_cnt * 2))
            writer.writerow(
                [
                    item.get("track_id", ""),
                    item.get("user_label") or "",
                    item.get("motion_type", "static"),
                    c_cnt,
                    s_cnt,
                    "是" if item.get("is_bidirectional") else "否",
                    f"{float(item.get('avg_freq', 0.0)):.4f}",
                    f"{float(item.get('live_freq', 0.0)):.4f}",
                    int(item.get("cycles", c_cnt + s_cnt)),
                    int(item.get("swings", c_cnt + s_cnt)),
                    f"{float(item.get('bends_per_min', 0.0)):.2f}",
                    f"{float(item.get('activity_score', 0.0)):.4f}",
                    f"{float(sp.get('total_path_length_px', 0.0)):.2f}",
                    f"{float(sp.get('net_displacement_px', 0.0)):.2f}",
                    c_cyc,
                    c_bnd,
                ]
            )

    return os.path.basename(final_path)
