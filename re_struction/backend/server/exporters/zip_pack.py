"""05 §6: 子视频目录打包为 ZIP 归档导出器 (zip_pack.py)。"""
from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def write(zip_path: str | Path, sub_dir: str | Path) -> Path:
    zp = Path(zip_path)
    zp.parent.mkdir(parents=True, exist_ok=True)
    sd = Path(sub_dir)

    with ZipFile(zp, "w", compression=ZIP_DEFLATED) as zf:
        if sd.exists():
            for fp in sorted(sd.glob("*.mp4")):
                zf.write(fp, arcname=fp.name)
    return zp
