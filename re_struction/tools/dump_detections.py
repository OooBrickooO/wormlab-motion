"""T21 (04 §7): 对基准视频前 300 帧运行 Detector，将检测结果保存为 data/cache/dets_<id>.npz。"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "backend"))

from wormlab.config import load_config  # noqa: E402
from wormlab.detection.detector import Detector  # noqa: E402
from wormlab.schemas import Detection  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dump_detections")

DEFAULT_VIDEOS: dict[str, Path] = {
    "10x": ROOT / "referAnswer" / "10×.mp4",
    "1.6x-1": ROOT / "referAnswer" / "1.6X（1）.mp4",
    "Move-3": REPO_ROOT / "backend" / "uploads" / "Move-3.mp4",
}


def pack_detections(frames_dets: list[list[Detection]], fps: float, width: int, height: int) -> dict[str, np.ndarray]:
    """将多帧 Detection 列表打包为无 pickle 扁平 npz 字典。"""
    n_frames = len(frames_dets)
    frame_offsets = np.zeros(n_frames + 1, dtype=np.int32)
    bboxes_list: list[np.ndarray] = []
    confs_list: list[float] = []
    origins_list: list[tuple[int, int]] = []
    mask_shapes_list: list[tuple[int, int]] = []
    mask_Offsets: list[int] = [0]
    mask_bytes_chunks: list[np.ndarray] = []

    idx = 0
    for f_i, dets in enumerate(frames_dets):
        frame_offsets[f_i] = idx
        for d in dets:
            bboxes_list.append(np.asarray(d.bbox, dtype=np.float32)[:4])
            confs_list.append(float(d.conf))
            origins_list.append((int(d.mask_origin[0]), int(d.mask_origin[1])))
            if d.mask is not None and d.mask.size > 0:
                m_bool = np.asarray(d.mask, dtype=bool)
                h, w = m_bool.shape[:2]
                packed = np.packbits(m_bool.reshape(-1))
                mask_shapes_list.append((int(h), int(w)))
                mask_bytes_chunks.append(packed)
                mask_Offsets.append(mask_Offsets[-1] + int(packed.size))
            else:
                mask_shapes_list.append((0, 0))
                mask_Offsets.append(mask_Offsets[-1])
            idx += 1
    frame_offsets[n_frames] = idx

    bboxes = np.vstack(bboxes_list).astype(np.float32) if bboxes_list else np.zeros((0, 4), dtype=np.float32)
    confs = np.array(confs_list, dtype=np.float32)
    origins = np.array(origins_list, dtype=np.int32) if origins_list else np.zeros((0, 2), dtype=np.int32)
    mask_shapes = np.array(mask_shapes_list, dtype=np.int32) if mask_shapes_list else np.zeros((0, 2), dtype=np.int32)
    mask_offsets_arr = np.array(mask_Offsets, dtype=np.int64)
    mask_bits = np.concatenate(mask_bytes_chunks) if mask_bytes_chunks else np.zeros(0, dtype=np.uint8)

    return {
        "meta": np.array([float(fps), float(width), float(height), float(n_frames)], dtype=np.float64),
        "frame_offsets": frame_offsets,
        "bboxes": bboxes,
        "confs": confs,
        "origins": origins,
        "mask_shapes": mask_shapes,
        "mask_offsets": mask_offsets_arr,
        "mask_bits": mask_bits,
    }


def unpack_detections(npz_path: Path) -> tuple[list[list[Detection]], float, int, int]:
    """从 dets_<id>.npz 还原多帧 Detection 列表及 (fps, width, height)。"""
    with np.load(npz_path, allow_pickle=False) as data:
        meta = data["meta"]
        fps, width, height, n_frames = float(meta[0]), int(meta[1]), int(meta[2]), int(meta[3])
        frame_offsets = data["frame_offsets"]
        bboxes = data["bboxes"]
        confs = data["confs"]
        origins = data["origins"]
        mask_shapes = data["mask_shapes"]
        mask_offsets = data["mask_offsets"]
        mask_bits = data["mask_bits"]

    frames_dets: list[list[Detection]] = []
    for f_i in range(n_frames):
        start, end = int(frame_offsets[f_i]), int(frame_offsets[f_i + 1])
        dets: list[Detection] = []
        for k in range(start, end):
            h, w = int(mask_shapes[k, 0]), int(mask_shapes[k, 1])
            if h > 0 and w > 0:
                m_start, m_end = int(mask_offsets[k]), int(mask_offsets[k + 1])
                unpacked = np.unpackbits(mask_bits[m_start:m_end])[: h * w]
                mask = unpacked.reshape((h, w)).astype(bool)
            else:
                mask = None
            dets.append(
                Detection(
                    bbox=bboxes[k].copy(),
                    conf=float(confs[k]),
                    mask=mask,
                    mask_origin=(int(origins[k, 0]), int(origins[k, 1])),
                    det_idx=k - start,
                )
            )
        frames_dets.append(dets)
    return frames_dets, fps, width, height


def dump_video(video_id: str, video_path: Path, det: Detector, out_dir: Path, max_frames: int = 300) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_dets: list[list[Detection]] = []
    batch: list[np.ndarray] = []
    batch_size = int(det.cfg.detection.batch_size)
    read_count = 0

    while read_count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        batch.append(frame)
        read_count += 1
        if len(batch) >= batch_size:
            frames_dets.extend(det.infer(batch))
            batch.clear()
    if batch:
        frames_dets.extend(det.infer(batch))
    cap.release()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dets_{video_id}.npz"
    packed = pack_detections(frames_dets, fps, width, height)
    np.savez_compressed(out_path, **packed)
    logger.info("已保存 %s: %d 帧, 总检测数=%d -> %s", video_id, len(frames_dets), int(packed["bboxes"].shape[0]), out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="导出基准视频前 N 帧检测缓存供 T21 回放对拍使用")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "data" / "cache"))
    args = parser.parse_args()

    cfg = load_config("crawl_agar")
    model_path = ROOT / "backend" / "models" / cfg.detection.default_model
    det = Detector(model_path, cfg)
    out_dir = Path(args.out_dir)

    for vid, vpath in DEFAULT_VIDEOS.items():
        if not vpath.exists():
            logger.warning("跳过不存在的视频: %s (%s)", vid, vpath)
            continue
        dump_video(vid, vpath, det, out_dir, max_frames=args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
