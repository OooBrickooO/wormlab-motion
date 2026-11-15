"""05 §3: 跨平台视频写入器与安全文件重命名辅助 (video_io.py)。"""
from __future__ import annotations

import logging
import os
import struct
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def _register_openh264_dll() -> None:
    """在 Windows 下将 backend/、re_struction/ 或仓库根目录下的 openh264-1.8.0-win64.dll 所在目录加入搜索路径。"""
    backend_dir = Path(__file__).resolve().parents[2]
    root_re = Path(__file__).resolve().parents[3]
    repo_root = root_re.parent
    for cand_dir in (backend_dir, root_re, repo_root):
        dll_file = cand_dir / "openh264-1.8.0-win64.dll"
        if dll_file.exists():
            dir_str = str(cand_dir)
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(dir_str)
                except OSError:
                    pass
            cur_path = os.environ.get("PATH", "")
            if dir_str not in cur_path.split(os.pathsep):
                os.environ["PATH"] = dir_str + os.pathsep + cur_path
            break


_register_openh264_dll()


def ensure_unique_path(path: str | Path) -> str:
    p_str = str(path)
    if not os.path.exists(p_str):
        return p_str
    root, ext = os.path.splitext(p_str)
    idx = 1
    while True:
        candidate = f"{root}_{idx}{ext}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def safe_rename(src: str | Path, dst: str | Path) -> str:
    unique_dst = ensure_unique_path(dst)
    os.replace(str(src), unique_dst)
    return unique_dst


def create_compatible_video_writer(
    path: str | Path,
    fps: float,
    frame_size: tuple[int, int],
    codecs: tuple[str, ...] = ("avc1", "H264", "X264", "mp4v"),
) -> cv2.VideoWriter:
    p_str = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(p_str)), exist_ok=True)
    last_error: Exception | None = None
    w, h = int(frame_size[0]), int(frame_size[1])
    fps_val = max(1.0, float(fps))
    for codec in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(p_str, fourcc, fps_val, (w, h))
            if writer is not None and writer.isOpened():
                logger.debug("VideoWriter 启用编码器 %s -> %s", codec, p_str)
                return writer
            if writer is not None:
                writer.release()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"无法创建视频写入器 ({p_str}): {last_error}")


_MP4_CONTAINER_ATOMS = frozenset({b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"mvex"})


def _read_top_level_atoms(fp, file_size: int) -> list[tuple[bytes, int, int]]:
    """读取 MP4 文件的顶层 atom 列表: (type_bytes, start, size)。"""
    atoms: list[tuple[bytes, int, int]] = []
    pos = 0
    while pos + 8 <= file_size:
        fp.seek(pos)
        hdr = fp.read(8)
        if len(hdr) < 8:
            break
        size32, atom_type = struct.unpack(">I4s", hdr)
        if size32 == 1:
            ext = fp.read(8)
            if len(ext) < 8:
                return []
            size = struct.unpack(">Q", ext)[0]
            hdr_len = 16
        elif size32 == 0:
            size = file_size - pos
            hdr_len = 8
        else:
            size = size32
            hdr_len = 8
        if size < hdr_len or pos + size > file_size:
            return []
        atoms.append((atom_type, pos, size))
        pos += size
    return atoms if pos == file_size else []


def _patch_moov_offsets(buf: bytearray, start: int, end: int, insert_pos: int, moov_start: int, delta: int) -> bool:
    """递归修正 moov 内 stco/co64 的偏移量（加上 delta）。"""
    pos = start
    while pos + 8 <= end:
        size32, atom_type = struct.unpack_from(">I4s", buf, pos)
        if size32 == 1:
            if pos + 16 > end:
                return False
            size = struct.unpack_from(">Q", buf, pos + 8)[0]
            hdr_len = 16
        elif size32 == 0:
            size = end - pos
            hdr_len = 8
        else:
            size = size32
            hdr_len = 8
        if size < hdr_len or pos + size > end:
            return False

        body_start = pos + hdr_len
        body_end = pos + size
        if atom_type in _MP4_CONTAINER_ATOMS:
            if not _patch_moov_offsets(buf, body_start, body_end, insert_pos, moov_start, delta):
                return False
        elif atom_type == b"stco":
            if body_start + 8 > body_end:
                return False
            count = struct.unpack_from(">I", buf, body_start + 4)[0]
            tbl_start = body_start + 8
            if tbl_start + 4 * count > body_end:
                return False
            for idx in range(count):
                off_pos = tbl_start + 4 * idx
                val = struct.unpack_from(">I", buf, off_pos)[0]
                if insert_pos <= val < moov_start:
                    new_val = val + delta
                    if new_val > 0xFFFFFFFF:
                        return False
                    struct.pack_into(">I", buf, off_pos, new_val)
        elif atom_type == b"co64":
            if body_start + 8 > body_end:
                return False
            count = struct.unpack_from(">I", buf, body_start + 4)[0]
            tbl_start = body_start + 8
            if tbl_start + 8 * count > body_end:
                return False
            for idx in range(count):
                off_pos = tbl_start + 8 * idx
                val = struct.unpack_from(">Q", buf, off_pos)[0]
                if insert_pos <= val < moov_start:
                    struct.pack_into(">Q", buf, off_pos, val + delta)
        pos += size
    return True


def faststart_mp4(path: str | Path) -> bool:
    """将 MP4 的 moov atom 移到 mdat 之前并修正 stco/co64 偏移，使浏览器可立即流式播放。"""
    target = Path(path)
    if not target.exists() or not target.is_file():
        return False
    file_size = target.stat().st_size
    if file_size < 16:
        return False

    with open(target, "rb") as fp:
        atoms = _read_top_level_atoms(fp, file_size)
        if not atoms:
            return False
        types = [a[0] for a in atoms]
        if b"moov" not in types or b"mdat" not in types:
            return False
        moov_idx = types.index(b"moov")
        mdat_idx = types.index(b"mdat")
        if moov_idx < mdat_idx:
            return False

        _, moov_start, moov_size = atoms[moov_idx]
        insert_idx = (types.index(b"ftyp") + 1) if b"ftyp" in types else 0
        if insert_idx > mdat_idx:
            insert_idx = 0
        insert_pos = atoms[insert_idx][1] if insert_idx < len(atoms) else 0

        fp.seek(moov_start)
        moov_buf = bytearray(fp.read(moov_size))
        if len(moov_buf) != moov_size:
            return False
        if not _patch_moov_offsets(moov_buf, 0, moov_size, insert_pos, moov_start, moov_size):
            logger.warning("faststart_mp4 跳过 %s: 修正 stco/co64 失败", target)
            return False

        tmp_path = target.with_suffix(".faststart.tmp")
        try:
            with open(tmp_path, "wb") as out:
                for idx, (atype, start, size) in enumerate(atoms):
                    if idx == insert_idx:
                        out.write(moov_buf)
                    if idx == moov_idx:
                        continue
                    fp.seek(start)
                    rem = size
                    while rem > 0:
                        chunk = fp.read(min(1048576, rem))
                        if not chunk:
                            break
                        out.write(chunk)
                        rem -= len(chunk)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

    os.replace(str(tmp_path), str(target))
    return True
