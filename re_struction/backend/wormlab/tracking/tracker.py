"""04 §6: 纯跟踪编排器 (Tracker) —— 不含绘制、子视频或运动学计数职责。"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np

from wormlab.config import TrackingCfg
from wormlab.geometry import iou_xyxy
from wormlab.schemas import Detection, TrackOut

from .associator import associate
from .features import extract_features
from .head_config import apply_head_config
from .id_pool import IdPool
from .track import Track


def _nms_preserve_orig_idx(dets: list[Detection], overlap_thr: float) -> tuple[list[Detection], list[int]]:
    """按旧 EnhancedWormTracker.apply_nms 逻辑过滤重叠框，同时返回保留项在原列表中的下标映射。"""
    n = len(dets)
    if n <= 1:
        return list(dets), list(range(n))

    boxes = np.vstack([np.asarray(d.bbox, dtype=np.float32)[:4] for d in dets])
    scores = np.array([float(d.conf) for d in dets], dtype=np.float64)
    order = np.argsort(-scores)
    keep: list[int] = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        ious = np.divide(inter, union, out=np.zeros_like(union), where=union > 0)
        order = rest[ious <= float(overlap_thr)]

    return [dets[i] for i in keep], keep


class Tracker:
    """多目标卡尔曼-匈牙利跟踪器（与旧 EnhancedWormTracker 保持 1:1 跟踪行为）。"""

    def __init__(
        self,
        cfg: TrackingCfg,
        fps: float,
        frame_wh: tuple[int, int],
        max_worm_id: int | None = None,
        head_config: list[dict[str, Any]] | dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.fps = float(fps)
        self.width, self.height = int(frame_wh[0]), int(frame_wh[1])

        self.persistence_frames = int(round(float(cfg.persistence_sec) * self.fps))
        self.ghost_overlay_frames = int(round(float(cfg.ghost_overlay_sec) * self.fps))
        self.max_lost_frames = max(int(self.fps * float(cfg.max_lost_sec_legacy)), self.persistence_frames)
        self.confirm_hits = int(cfg.confirm_hits)

        self.tracks: list[Track] = []
        self.id_pool = IdPool(max_id=max_worm_id)
        self.head_config = head_config or []
        self._head_config_applied: bool = False

        self._next_uid: int = 1
        self.ghost_ids: set[int] = set()
        self.track_history: dict[int, deque[tuple[float, float]]] = defaultdict(lambda: deque(maxlen=3600))
        self.id_last_seen: dict[int, int] = {}

    def alloc_uid(self) -> int:
        u = self._next_uid
        self._next_uid += 1
        return u

    def set_max_worm_id(self, max_id: int | None) -> None:
        active = {t.track_id for t in self.tracks}
        self.id_pool.set_max(max_id, active_ids=active)

    def feed_pose(self, track_id: int, dense: np.ndarray | None) -> None:
        for t in self.tracks:
            if t.track_id == int(track_id):
                t.feed_pose(dense)
                break

    def get_track(self, track_id: int) -> Track | None:
        for t in self.tracks:
            if t.track_id == int(track_id):
                return t
        return None

    @property
    def active_ids(self) -> set[int]:
        return {t.track_id for t in self.tracks if t.track_id not in self.ghost_ids}

    @staticmethod
    def _scale_kalman_velocity(track: Track, factor: float) -> None:
        inner = getattr(getattr(track, "kf", None), "kf", None)
        if inner is not None and getattr(inner, "statePost", None) is not None and inner.statePost.shape[0] >= 8:
            inner.statePost[4:, 0] *= float(factor)

    def _handle_lost_tracks(self, track_indices: list[int]) -> None:
        for t_idx in track_indices:
            track = self.tracks[t_idx]
            track.lost_frames += 1
            if self.cfg.damp_lost_velocity and track.lost_frames >= 2:
                # 衰减丢失期间卡尔曼速度状态，防止遮挡/卷曲期间预测框漂离真实位置
                self._scale_kalman_velocity(track, 0.5)
            tid = track.track_id
            if track.lost_frames <= self.ghost_overlay_frames:
                self.ghost_ids.add(tid)
                pred_bbox = track.last_predicted_bbox
                if pred_bbox is not None:
                    cx = 0.5 * (float(pred_bbox[0]) + float(pred_bbox[2]))
                    cy = 0.5 * (float(pred_bbox[1]) + float(pred_bbox[3]))
                    self.track_history[tid].append((cx, cy))
            else:
                self.ghost_ids.discard(tid)

    def _prune_lost_tracks(self) -> None:
        remaining: list[Track] = []
        unconfirmed_limit = min(3, self.confirm_hits)
        confirmed_boxes = (
            [ct.get_state() for ct in self.tracks if ct.confirmed]
            if self.cfg.suppress_parasite_unconfirmed
            else []
        )
        for t in self.tracks:
            max_lost = self.max_lost_frames if t.confirmed else unconfirmed_limit
            is_parasite = False
            if self.cfg.suppress_parasite_unconfirmed and not t.confirmed and confirmed_boxes:
                tb = t.get_state()
                is_parasite = any(iou_xyxy(tb, cb) > 0.20 for cb in confirmed_boxes)
            if t.lost_frames > max_lost or is_parasite:
                self.ghost_ids.discard(t.track_id)
                self.id_pool.release(t.track_id)
            else:
                remaining.append(t)
        self.tracks = remaining

    def _make_track_out(self, track: Track, bbox: np.ndarray, det_idx: int | None, is_ghost: bool, conf: float) -> TrackOut:
        return TrackOut(
            track_id=int(track.track_id),
            bbox=np.asarray(bbox, dtype=np.float32)[:4],
            det_idx=det_idx,
            is_ghost=bool(is_ghost),
            lost_frames=int(track.lost_frames),
            confirmed=bool(track.confirmed),
            user_label=track.user_label,
            head_kpt_idx=track.head_kpt_idx,
            head_xy=track.head_xy,
            conf=float(conf),
            first_frame=int(track.first_frame),
            uid=int(track.uid),
        )

    def update(self, dets: list[Detection], frame_idx: int) -> list[TrackOut]:
        """更新一帧检测并返回活跃及视觉残留 (ghost) 轨迹输出列表。"""
        fidx = int(frame_idx)
        if len(dets) == 0:
            for track in self.tracks:
                track.predict()
            self._handle_lost_tracks(list(range(len(self.tracks))))
            self._prune_lost_tracks()
            ghost_outs: list[TrackOut] = []
            for t in self.tracks:
                if t.track_id in self.ghost_ids and t.track_id > 0:
                    b = t.last_predicted_bbox if t.last_predicted_bbox is not None else t.get_state()
                    ghost_outs.append(
                        self._make_track_out(
                            t,
                            bbox=b,
                            det_idx=None,
                            is_ghost=True,
                            conf=float(self.cfg.prediction_confidence),
                        )
                    )
            return ghost_outs

        nms_dets, orig_indices = _nms_preserve_orig_idx(dets, self.cfg.overlap_threshold)
        matches, unmatched_dets, unmatched_tracks, det_features = associate(
            self.tracks, nms_dets, self.cfg, (self.width, self.height)
        )

        # 1. 更新已匹配轨迹
        for t_idx, d_idx in matches:
            track = self.tracks[t_idx]
            det = nms_dets[d_idx]
            feat = det_features[d_idx]
            if self.cfg.damp_lost_velocity and track.lost_frames >= 2:
                self._scale_kalman_velocity(track, 0.0)
            track.update(det.bbox, feature=feat, conf=det.conf)
            if not track.confirmed and track.hits >= self.confirm_hits:
                track.confirmed = True
            cx = 0.5 * (float(det.bbox[0]) + float(det.bbox[2]))
            cy = 0.5 * (float(det.bbox[1]) + float(det.bbox[3]))
            self.track_history[track.track_id].append((cx, cy))
            self.id_last_seen[track.track_id] = fidx
            self.ghost_ids.discard(track.track_id)

        # 2. 按置信度降序为未匹配检测分配新轨迹 ID
        new_tracks: list[Track] = []
        new_tracks_by_d: dict[int, Track] = {}
        init_conf = self.cfg.init_confidence
        confirmed_boxes_existing = (
            [ct.get_state() for ct in self.tracks if ct.confirmed]
            if self.cfg.suppress_parasite_unconfirmed
            else []
        )
        sorted_unmatched_dets = sorted(
            unmatched_dets,
            key=lambda d_i: float(nms_dets[d_i].conf),
            reverse=True,
        )

        for d_idx in sorted_unmatched_dets:
            det = nms_dets[d_idx]
            if init_conf is not None and float(det.conf) < float(init_conf):
                continue
            if confirmed_boxes_existing and any(iou_xyxy(det.bbox, cb) > 0.20 for cb in confirmed_boxes_existing):
                continue
            allocated_id = self.id_pool.alloc()
            if allocated_id is None:
                continue
            feat = det_features[d_idx] if d_idx < len(det_features) else None
            if feat is None and det.mask is not None:
                ox, oy = det.mask_origin
                b_in = det.bbox - np.array([ox, oy, ox, oy], dtype=np.float32)
                feat = extract_features(det.mask, b_in)

            new_t = Track(
                track_id=allocated_id,
                bbox=det.bbox,
                uid=self.alloc_uid(),
                first_frame=fidx,
                feature=feat,
                conf=det.conf,
            )
            new_t.confirmed = False
            new_tracks.append(new_t)
            new_tracks_by_d[d_idx] = new_t

            cx = 0.5 * (float(det.bbox[0]) + float(det.bbox[2]))
            cy = 0.5 * (float(det.bbox[1]) + float(det.bbox[3]))
            self.track_history[allocated_id].append((cx, cy))
            self.id_last_seen[allocated_id] = fidx

        count_before_hc = len(self.tracks) + len(new_tracks)
        self.tracks.extend(new_tracks)
        if new_tracks and not self._head_config_applied and self.head_config:
            apply_head_config(self, self.head_config)
            self._head_config_applied = True
        manual_hc_tracks = self.tracks[count_before_hc:]

        # 3. 处理未匹配轨迹
        self._handle_lost_tracks(unmatched_tracks)

        # 为 nms_dets 分配 tracker_id（与旧实现顺序完全一致）
        tracker_ids = [-1] * len(nms_dets)
        for t_idx, d_idx in matches:
            tracker_ids[d_idx] = self.tracks[t_idx].track_id
        for d_idx, new_t in new_tracks_by_d.items():
            tracker_ids[d_idx] = new_t.track_id

        # 收集仍在残留绘制窗口内的未匹配轨迹（顺序与旧实现一致）
        predicted_tracks: list[Track] = []
        for t_idx in unmatched_tracks:
            track = self.tracks[t_idx]
            if track.lost_frames <= self.ghost_overlay_frames:
                predicted_tracks.append(track)

        # 4. 回收过期轨迹
        self._prune_lost_tracks()

        # 5. 组装输出 TrackOut（匹配/新建轨迹使用 Kalman 平滑框，ghost 使用 last_predicted_bbox）
        tracks_by_id = {t.track_id: t for t in self.tracks}
        out_list: list[TrackOut] = []

        for d_idx, tid in enumerate(tracker_ids):
            if tid is not None and tid > 0:
                track = tracks_by_id.get(tid)
                if track is not None:
                    out_list.append(
                        self._make_track_out(
                            track,
                            bbox=track.get_state(),
                            det_idx=orig_indices[d_idx],
                            is_ghost=False,
                            conf=float(nms_dets[d_idx].conf),
                        )
                    )

        for mt in manual_hc_tracks:
            if mt.track_id > 0 and mt.track_id in tracks_by_id:
                out_list.append(
                    self._make_track_out(
                        mt,
                        bbox=mt.get_state(),
                        det_idx=None,
                        is_ghost=False,
                        conf=float(mt.conf),
                    )
                )

        for track in predicted_tracks:
            tid = track.track_id
            if tid is not None and tid > 0 and tid in tracks_by_id:
                b = track.last_predicted_bbox if track.last_predicted_bbox is not None else track.get_state()
                out_list.append(
                    self._make_track_out(
                        track,
                        bbox=b,
                        det_idx=None,
                        is_ghost=True,
                        conf=float(self.cfg.prediction_confidence),
                    )
                )

        return out_list

    def apply_recovered_detection(
        self,
        track_id: int,
        det: Detection,
        det_idx: int,
        frame_idx: int,
    ) -> TrackOut | None:
        """用滞后召回（ROI YOLO / 背景差分）找回的结果直接更新指定已确认丢失轨迹。"""
        track = self.get_track(track_id)
        if track is None:
            return None
        if self.cfg.damp_lost_velocity and track.lost_frames >= 2:
            self._scale_kalman_velocity(track, 0.0)
        feat = None
        if det.mask is not None:
            ox, oy = det.mask_origin
            b_in = det.bbox - np.array([ox, oy, ox, oy], dtype=np.float32)
            feat = extract_features(det.mask, b_in)
        track.update(det.bbox, feature=feat, conf=det.conf)
        cx = 0.5 * (float(det.bbox[0]) + float(det.bbox[2]))
        cy = 0.5 * (float(det.bbox[1]) + float(det.bbox[3]))
        self.track_history[track.track_id].append((cx, cy))
        self.id_last_seen[track.track_id] = int(frame_idx)
        self.ghost_ids.discard(track.track_id)
        return self._make_track_out(
            track,
            bbox=track.get_state(),
            det_idx=int(det_idx),
            is_ghost=False,
            conf=float(det.conf),
        )
