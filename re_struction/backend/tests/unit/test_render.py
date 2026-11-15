"""T23 (05 §3): 渲染模块单元测试与新旧画面对拍图导出。"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import supervision as sv

from wormlab.render import DrawItem, StableCrop, SubVideoWriter, draw_frame

ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "backend"))

import worm_processing as legacy_wp  # noqa: E402


def test_draw_frame_and_compare_with_legacy() -> None:
    h, w = 480, 640
    base_frame = np.full((h, w, 3), 45, dtype=np.uint8)

    trail = [(100.0 + 3.0 * i, 200.0 + 1.5 * i) for i in range(15)]
    contour = np.array([[135, 215], [160, 212], [185, 222], [160, 230]], dtype=np.float32)

    items = [
        DrawItem(
            track_id=1,
            bbox=np.array([130.0, 205.0, 190.0, 238.0], dtype=np.float32),
            is_ghost=False,
            lost_frames=0,
            user_label="N2",
            contour=contour,
            head_xy=(185.0, 222.0),
            trail=trail,
            counted=True,
            counts={
                "c_count": 0,
                "s_count": 3,
                "is_bidirectional": False,
                "avg_freq": 0.45,
                "bends_per_min": 27.0,
                "motion_type_legacy": "sinusoidal",
                "activity_score": 0.82,
            },
        ),
        DrawItem(
            track_id=2,
            bbox=np.array([350.0, 150.0, 405.0, 195.0], dtype=np.float32),
            is_ghost=True,
            lost_frames=4,
        ),
        DrawItem(
            track_id=3,
            bbox=np.array([480.0, 300.0, 535.0, 345.0], dtype=np.float32),
            is_ghost=False,
            counted=False,
        ),
    ]

    rendered_new = draw_frame(base_frame, items, frame_idx=42)
    assert rendered_new.shape == base_frame.shape
    assert not np.array_equal(rendered_new, base_frame)

    # 同时用旧渲染函数生成对拍图保存至 data/diag/
    diag_dir = ROOT / "data" / "diag"
    diag_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(diag_dir / "render_new.jpg"), rendered_new)

    mask_1 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_1, [contour.astype(np.int32)], 1)
    mask_2 = np.zeros((h, w), dtype=np.uint8)
    tracked_dets = sv.Detections(
        xyxy=np.array([[130.0, 205.0, 190.0, 238.0], [350.0, 150.0, 405.0, 195.0]], dtype=np.float32),
        mask=np.stack([mask_1, mask_2], axis=0).astype(bool),
        tracker_id=np.array([1, 2], dtype=int),
    )
    fake_tracker = SimpleNamespace(
        ghost_ids={2},
        track_history={1: deque(trail, maxlen=3600)},
        tracks=[
            SimpleNamespace(track_id=1, user_label="N2", head_kpt_idx=None, last_head_pos=(185.0, 222.0), last_keypoints=None, last_observed_keypoints=None, is_static=False, hits=10, lost_frames=0),
            SimpleNamespace(track_id=2, user_label=None, head_kpt_idx=None, last_head_pos=None, last_keypoints=None, last_observed_keypoints=None, is_static=False, hits=5, lost_frames=4),
        ],
        cycle_counts={
            1: {
                "c_count": 0,
                "s_count": 3,
                "is_bidirectional": False,
                "avg_freq": 0.45,
                "bends_per_min": 27.0,
                "motion_type": "sinusoidal",
                "activity_score": 0.82,
                "lost_frames": 0,
            },
            2: {"lost_frames": 4, "is_ghost": True},
        },
        update_movement_status=lambda tid: "sinusoidal",
    )
    rendered_old, _ = legacy_wp.annotate_and_collect_data(base_frame, tracked_dets, fake_tracker, frame_idx=42)
    cv2.imwrite(str(diag_dir / "render_old.jpg"), rendered_old)
    assert (diag_dir / "render_new.jpg").exists()
    assert (diag_dir / "render_old.jpg").exists()


def test_stable_crop_and_subvideo_writer(tmp_path: Path) -> None:
    crop = StableCrop([100.0, 100.0, 150.0, 140.0])
    w0 = crop.window(640, 480)
    assert w0[2] - w0[0] == w0[3] - w0[1] >= 72

    sw = SubVideoWriter(tmp_path, track_id=1, fps=15.0, initial_bbox=[100.0, 100.0, 150.0, 140.0])
    frm = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(5):
        sw.write_frame(frm, [100.0 + i, 100.0, 150.0 + i, 140.0], is_ghost=(i == 4))
    out_path = sw.close(total_cycles=3)
    assert out_path is not None and out_path.exists()
    assert out_path.name == "worm_1_cycles_3.mp4"


def test_register_openh264_dll_and_faststart_mp4(tmp_path: Path) -> None:
    import struct

    from wormlab.render.video_io import _read_top_level_atoms, _register_openh264_dll, faststart_mp4

    _register_openh264_dll()
    dll_path = ROOT / "backend" / "openh264-1.8.0-win64.dll"
    assert dll_path.exists()

    def _box(tag: bytes, payload: bytes) -> bytes:
        return struct.pack(">I4s", 8 + len(payload), tag) + payload

    ftyp = _box(b"ftyp", b"isom\x00\x00\x02\x00")
    mdat = _box(b"mdat", b"\x11\x22\x33\x44" * 8)
    chunk_offset_before = len(ftyp) + 8
    stco = _box(b"stco", struct.pack(">II", 0, 1) + struct.pack(">I", chunk_offset_before))
    moov = _box(b"moov", stco)

    mp4_file = tmp_path / "tail_moov.mp4"
    mp4_file.write_bytes(ftyp + mdat + moov)

    assert faststart_mp4(mp4_file) is True
    raw = mp4_file.read_bytes()
    with mp4_file.open("rb") as fp:
        atoms = _read_top_level_atoms(fp, len(raw))
    assert [t for t, _, _ in atoms] == [b"ftyp", b"moov", b"mdat"]
    stco_pos = raw.find(b"stco")
    patched_off = struct.unpack_from(">I", raw, stco_pos + 12)[0]
    assert patched_off == chunk_offset_before + len(moov)
