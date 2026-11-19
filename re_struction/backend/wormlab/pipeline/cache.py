"""05 §1: Pass A -> Pass B 无 pickle 长格式轨迹姿态缓存 (cache.py)。"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from wormlab.schemas import (
    N_DENSE,
    FrameRecord,
    Pose,
    Quality,
    TrackOut,
    decode_flags,
    encode_flags,
)


class CacheWriter:
    """将逐帧 TrackOut 与 Pose 在内存中累积并在结束时一次性写入 tracks.npz + meta.json。"""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self._frame_idx: list[int] = []
        self._track_id: list[int] = []
        self._track_uid: list[int] = []
        self._first_frame: list[int] = []
        self._bbox: list[np.ndarray] = []
        self._conf: list[float] = []
        self._is_ghost: list[bool] = []
        self._quality: list[int] = []
        self._dense: list[np.ndarray] = []
        self._length: list[float] = []
        self._chord: list[float] = []
        self._width: list[float] = []
        self._area: list[float] = []
        self._flags: list[int] = []
        self._contours: list[np.ndarray] = []
        self._contour_offsets: list[int] = [0]
        self._ell_aspect: list[float] = []
        self._ell_ecc: list[float] = []
        self._ell_angle: list[float] = []
        self._ell_skew_minor: list[float] = []
        self._roi_motion: list[float] = []
        self._user_labels: dict[str, str] = {}
        self._head_kpt_idx: dict[str, int] = {}

    @staticmethod
    def _shape_field(pose: Pose | None, name: str) -> float:
        """读取免骨架形状描述子字段；pose 缺失或字段非有限值时统一写 NaN。"""
        if pose is None:
            return float("nan")
        try:
            v = float(getattr(pose, name))
        except Exception:  # noqa: BLE001
            return float("nan")
        return v if np.isfinite(v) else float("nan")

    def append(self, frame_idx: int, track_out: TrackOut, pose: Pose | None) -> None:
        self._frame_idx.append(int(frame_idx))
        tid = int(track_out.track_id)
        uid = int(track_out.uid) if int(track_out.uid) > 0 else tid
        self._track_id.append(tid)
        self._track_uid.append(uid)
        self._first_frame.append(int(track_out.first_frame))
        self._bbox.append(np.asarray(track_out.bbox, dtype=np.float32)[:4].copy())
        self._conf.append(float(track_out.conf))
        self._is_ghost.append(bool(track_out.is_ghost))

        if track_out.user_label is not None:
            self._user_labels[str(uid)] = str(track_out.user_label)
        if track_out.head_kpt_idx is not None:
            self._head_kpt_idx[str(uid)] = int(track_out.head_kpt_idx)

        if pose is None or pose.dense is None:
            q_val = int(pose.quality) if pose is not None else int(Quality.MISSING)
            self._quality.append(q_val)
            self._dense.append(np.full((N_DENSE, 2), np.nan, dtype=np.float32))
            self._length.append(float(pose.length) if pose is not None else 0.0)
            self._chord.append(float(pose.chord) if pose is not None else 0.0)
            self._width.append(float(pose.width) if pose is not None else 0.0)
            self._area.append(float(pose.area) if pose is not None else 0.0)
            self._flags.append(encode_flags(pose.flags) if pose is not None else 0)
        else:
            self._quality.append(int(pose.quality))
            self._dense.append(np.asarray(pose.dense, dtype=np.float32).reshape((N_DENSE, 2)).copy())
            self._length.append(float(pose.length))
            self._chord.append(float(pose.chord))
            self._width.append(float(pose.width))
            self._area.append(float(pose.area))
            self._flags.append(encode_flags(pose.flags))

        if pose is not None and pose.contour is not None and len(pose.contour) > 0:
            c_arr = np.asarray(pose.contour, dtype=np.float32).reshape((-1, 2))
            self._contours.append(c_arr)
            self._contour_offsets.append(self._contour_offsets[-1] + int(c_arr.shape[0]))
        else:
            self._contour_offsets.append(self._contour_offsets[-1])

        self._ell_aspect.append(self._shape_field(pose, "ell_aspect"))
        self._ell_ecc.append(self._shape_field(pose, "ell_ecc"))
        self._ell_angle.append(self._shape_field(pose, "ell_angle"))
        self._ell_skew_minor.append(self._shape_field(pose, "ell_skew_minor"))
        self._roi_motion.append(self._shape_field(pose, "roi_motion"))

    def close(self, meta: dict[str, Any]) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        m = len(self._frame_idx)

        if m > 0:
            bbox_arr = np.vstack(self._bbox).astype(np.float32)
            dense_arr = np.stack(self._dense, axis=0).astype(np.float32)
        else:
            bbox_arr = np.zeros((0, 4), dtype=np.float32)
            dense_arr = np.zeros((0, N_DENSE, 2), dtype=np.float32)

        contour_xy = (
            np.vstack(self._contours).astype(np.float32)
            if self._contours
            else np.zeros((0, 2), dtype=np.float32)
        )

        npz_path = self.cache_dir / "tracks.npz"
        np.savez_compressed(
            npz_path,
            frame_idx=np.array(self._frame_idx, dtype=np.int32),
            track_id=np.array(self._track_id, dtype=np.int32),
            track_uid=np.array(self._track_uid, dtype=np.int32),
            first_frame=np.array(self._first_frame, dtype=np.int32),
            bbox=bbox_arr,
            conf=np.array(self._conf, dtype=np.float32),
            is_ghost=np.array(self._is_ghost, dtype=bool),
            quality=np.array(self._quality, dtype=np.int8),
            dense=dense_arr,
            length=np.array(self._length, dtype=np.float32),
            chord=np.array(self._chord, dtype=np.float32),
            width=np.array(self._width, dtype=np.float32),
            area=np.array(self._area, dtype=np.float32),
            flags=np.array(self._flags, dtype=np.uint16),
            contour_xy=contour_xy,
            contour_off=np.array(self._contour_offsets, dtype=np.int64),
            ell_aspect=np.array(self._ell_aspect, dtype=np.float32),
            ell_ecc=np.array(self._ell_ecc, dtype=np.float32),
            ell_angle=np.array(self._ell_angle, dtype=np.float32),
            ell_skew_minor=np.array(self._ell_skew_minor, dtype=np.float32),
            roi_motion=np.array(self._roi_motion, dtype=np.float32),
        )

        full_meta = {
            "schema_version": 2,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "user_labels": {**self._user_labels, **(meta.get("user_labels") or {})},
            "head_kpt_idx": {**self._head_kpt_idx, **(meta.get("head_kpt_idx") or {})},
            **meta,
        }
        meta_path = self.cache_dir / "meta.json"
        meta_path.write_text(json.dumps(full_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.cache_dir


class CacheReader:
    """从 tracks.npz + meta.json 读取长格式轨迹记录并构建帧索引与 uid 索引。"""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / "meta.json"
        npz_path = self.cache_dir / "tracks.npz"
        self.meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

        self.records: list[FrameRecord] = []
        self._by_frame: dict[int, list[FrameRecord]] = {}
        self._by_uid: dict[int, list[FrameRecord]] = {}

        if not npz_path.exists():
            return

        user_labels = self.meta.get("user_labels") or {}
        head_kpts = self.meta.get("head_kpt_idx") or {}

        with np.load(npz_path, allow_pickle=False) as data:
            frame_idx = data["frame_idx"]
            track_id = data["track_id"]
            track_uid = data["track_uid"] if "track_uid" in data else track_id
            first_frame = data["first_frame"] if "first_frame" in data else np.zeros_like(frame_idx)
            bbox = data["bbox"]
            conf = data["conf"]
            is_ghost = data["is_ghost"]
            quality = data["quality"]
            dense = data["dense"]
            length = data["length"]
            chord = data["chord"]
            width = data["width"]
            area = data["area"]
            flags = data["flags"]
            contour_xy = data["contour_xy"]
            contour_off = data["contour_off"]
            # 旧版 tracks.npz 可能不含免骨架形状描述子列：统一回退为 NaN，保证向后兼容
            m_len = len(frame_idx)
            ell_aspect = (
                data["ell_aspect"]
                if "ell_aspect" in data
                else np.full(m_len, np.nan, dtype=np.float32)
            )
            ell_ecc = data["ell_ecc"] if "ell_ecc" in data else np.full(m_len, np.nan, dtype=np.float32)
            ell_angle = data["ell_angle"] if "ell_angle" in data else np.full(m_len, np.nan, dtype=np.float32)
            ell_skew_minor = (
                data["ell_skew_minor"]
                if "ell_skew_minor" in data
                else np.full(m_len, np.nan, dtype=np.float32)
            )
            roi_motion = data["roi_motion"] if "roi_motion" in data else np.full(m_len, np.nan, dtype=np.float32)

        m = len(frame_idx)
        for i in range(m):
            f_i = int(frame_idx[i])
            tid = int(track_id[i])
            uid = int(track_uid[i])
            f0 = int(first_frame[i])
            b = bbox[i].astype(np.float32).copy()
            c_val = float(conf[i])
            ghost = bool(is_ghost[i])
            q_enum = Quality(int(quality[i]))
            d_arr = dense[i]
            has_dense = bool(np.isfinite(d_arr).all())

            c_start, c_end = int(contour_off[i]), int(contour_off[i + 1])
            cnt = contour_xy[c_start:c_end].astype(np.float32).copy() if c_end > c_start else None

            if ghost and not has_dense and q_enum == Quality.MISSING and cnt is None and int(flags[i]) == 0:
                pose_obj: Pose | None = None
            else:
                pose_obj = Pose(
                    dense=d_arr.astype(np.float32).copy() if has_dense else None,
                    length=float(length[i]),
                    chord=float(chord[i]),
                    width=float(width[i]),
                    area=float(area[i]),
                    quality=q_enum,
                    flags=decode_flags(int(flags[i])),
                    contour=cnt,
                    ell_aspect=float(ell_aspect[i]),
                    ell_ecc=float(ell_ecc[i]),
                    ell_angle=float(ell_angle[i]),
                    ell_skew_minor=float(ell_skew_minor[i]),
                    roi_motion=float(roi_motion[i]),
                )

            if pose_obj is not None and pose_obj.dense is not None:
                c_pt = (float(np.mean(pose_obj.dense[:, 0])), float(np.mean(pose_obj.dense[:, 1])))
            else:
                c_pt = (0.5 * float(b[0] + b[2]), 0.5 * float(b[1] + b[3]))

            u_lbl = user_labels.get(str(uid)) or user_labels.get(str(tid))
            h_idx_raw = head_kpts.get(str(uid)) if str(uid) in head_kpts else head_kpts.get(str(tid))
            h_idx = int(h_idx_raw) if h_idx_raw is not None else None

            rec = FrameRecord(
                frame_idx=f_i,
                track_id=tid,
                bbox=b,
                conf=c_val,
                is_ghost=ghost,
                pose=pose_obj,
                centroid=c_pt,
                uid=uid,
                first_frame=f0,
                user_label=u_lbl,
                head_kpt_idx=h_idx,
            )
            self.records.append(rec)
            self._by_frame.setdefault(f_i, []).append(rec)
            self._by_uid.setdefault(uid, []).append(rec)

    def frame_rows(self, frame_idx: int) -> list[FrameRecord]:
        return list(self._by_frame.get(int(frame_idx), []))

    def tracks_by_uid(self) -> dict[int, list[FrameRecord]]:
        return {uid: list(rows) for uid, rows in self._by_uid.items()}


def load_cache(cache_dir: str | Path) -> tuple[dict[str, Any], list[FrameRecord]]:
    """便捷加载函数：返回 (meta, records)。"""
    reader = CacheReader(cache_dir)
    return reader.meta, reader.records
