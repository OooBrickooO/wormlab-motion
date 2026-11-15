"""05 §4: Pass B (CPU 终稿批量运动学 + 主视频/子视频重渲染 + JSON/CSV/ZIP 导出)。"""
from __future__ import annotations

import json
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wormlab.compat.legacy_payload import to_legacy_summary, to_legacy_track
from wormlab.config import Config
from wormlab.kinematics.counter import KinematicsEngine, PerFrame, TrackReport
from wormlab.pose.orientation import reorient_track_poses, reorient_track_poses_coil_aware
from wormlab.render.overlay import DrawItem, draw_frame
from wormlab.render.subvideo import SubVideoWriter
from wormlab.render.video_io import create_compatible_video_writer, ensure_unique_path, faststart_mp4

from .cache import CacheReader
from .late_entry import partition_tracks_by_uid, select_primary_uid_per_track_id
from .pass_a import Control, JobSpec, NoopSink, open_source


@dataclass
class PassBResult:
    video_url: str
    log_url: str
    sub_videos_url: str
    summary_url: str
    csv_url: str
    id_pose_summary: list[dict[str, Any]] = field(default_factory=list)
    dropped_uids: list[int] = field(default_factory=list)
    video_path: str = ""
    summary_path: str = ""
    csv_path: str = ""
    log_path: str = ""
    zip_path: str = ""


class PassB:
    """从 Pass A 缓存计算终稿运动学并渲染主视频、子视频及导出全套报表。"""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def run(
        self,
        cache_dir: str | Path,
        job: JobSpec,
        sink: Any | None = None,
        ctrl: Control | None = None,
        render_videos: bool = True,
    ) -> PassBResult:
        ev_sink = sink if sink is not None else NoopSink()
        control = ctrl if ctrl is not None else Control()

        reader = CacheReader(cache_dir)
        meta = reader.meta
        fps = float(meta.get("fps", 30.0))
        width = int(meta.get("width", 640))
        height = int(meta.get("height", 480))
        total_frames = int(meta.get("total_frames", 0))
        video_path = str(job.video_path or meta.get("video_path", ""))
        c_unit = str(job.c_unit or meta.get("c_unit") or self.cfg.counting.c_unit)

        result_dir = (
            Path(job.result_dir)
            if job.result_dir
            else Path(self.cfg.paths.data_dir) / "results"
        )
        result_dir.mkdir(parents=True, exist_ok=True)

        uid_groups = partition_tracks_by_uid(
            reader.records,
            fps=fps,
            grace_s=self.cfg.counting.late_entry_grace_s,
        )
        primary_by_tid, dropped_uids = select_primary_uid_per_track_id(uid_groups)

        # 1. 终稿批量运动学计算
        ev_sink.update_state(phase="analyze")
        reports_by_uid: dict[int, TrackReport] = {}
        perframes_by_uid: dict[int, list[PerFrame]] = {}
        reoriented_poses_by_uid: dict[int, list[Any]] = {}
        n_groups = max(1, len(uid_groups))
        coil_aware = bool(getattr(self.cfg.pose, "coil_aware_reorient", False) and self.cfg.kinematics.s_run.enabled)

        for k, (uid, grp) in enumerate(uid_groups.items(), start=1):
            if control.cancelled:
                break
            raw_poses = grp.as_pose_list(max(total_frames, 1))
            if coil_aware:
                poses_seq = reorient_track_poses_coil_aware(
                    raw_poses,
                    fps=fps,
                    cfg=self.cfg.kinematics,
                    preserve_first_head=bool(grp.head_kpt_idx is not None),
                )
            else:
                poses_seq = reorient_track_poses(raw_poses)
            reoriented_poses_by_uid[uid] = poses_seq
            rep, pfs = KinematicsEngine.run_postprocessed(
                poses_seq,
                fps=fps,
                cfg=self.cfg.kinematics,
                c_unit=c_unit,
                frame_height=height,
            )
            reports_by_uid[uid] = rep
            perframes_by_uid[uid] = pfs
            ev_sink.emit(
                "phase",
                {"task_id": job.task_id, "phase": "analyze", "progress": round(k / n_groups, 3)},
                room=job.task_id,
            )

        # 组装汇总列表（按 track_id 升序，仅取每个 track_id 的主 uid）
        reports_by_tid: dict[int, TrackReport] = {}
        labels_by_tid: dict[int, str | None] = {}
        counted_by_tid: dict[int, bool] = {}
        primary_uids: set[int] = set()

        for tid, grp in primary_by_tid.items():
            primary_uids.add(grp.uid)
            reports_by_tid[tid] = reports_by_uid[grp.uid]
            labels_by_tid[tid] = grp.user_label
            counted_by_tid[tid] = grp.counted

        id_pose_summary = to_legacy_summary(
            reports_by_tid=reports_by_tid,
            labels_by_tid=labels_by_tid,
            counted_by_tid=counted_by_tid,
            c_unit=c_unit,
            include_late=self.cfg.counting.include_late_in_outputs,
        )

        stem = Path(video_path).stem if video_path else "video"
        base_prefix = f"{stem}_result_{job.task_id}"

        video_out_path = Path(ensure_unique_path(result_dir / f"{base_prefix}.mp4"))
        log_out_path = Path(ensure_unique_path(result_dir / f"{base_prefix}_log.json"))
        summary_out_path = Path(ensure_unique_path(result_dir / f"{base_prefix}_id_pose_summary.json"))
        sub_dir = result_dir / "sub_videos" / job.task_id
        sub_dir.mkdir(parents=True, exist_ok=True)
        zip_out_path = Path(ensure_unique_path(result_dir / f"{job.task_id}_sub_videos.zip"))

        # 2. 重渲染主视频 + 子视频 + 生成逐帧日志
        ev_sink.update_state(phase="render")
        ev_sink.emit(
            "phase",
            {"task_id": job.task_id, "phase": "render", "progress": 0.0},
            room=job.task_id,
        )

        log_entries: list[dict[str, Any]] = []
        sub_writers: dict[int, SubVideoWriter] = {}
        trails: dict[int, deque[tuple[float, float]]] = defaultdict(lambda: deque(maxlen=30))
        sub_size = int(self.cfg.pipeline.sub_video_size)

        was_cancelled = bool(control.cancelled)
        rendered_main = False
        if not was_cancelled and render_videos and video_path and os.path.exists(video_path) and total_frames > 0:
            src = open_source(video_path, max_frames=total_frames)
            main_writer = create_compatible_video_writer(video_out_path, fps=fps, frame_size=(width, height))
            try:
                f_idx = 0
                while f_idx < total_frames:
                    if control.cancelled:
                        was_cancelled = True
                        break
                    batch = src.read_batch(16)
                    if not batch:
                        break
                    for frame in batch:
                        if control.cancelled:
                            was_cancelled = True
                            break
                        rows = reader.frame_rows(f_idx)
                        draw_items: list[DrawItem] = []
                        frame_tracks: list[dict[str, Any]] = []

                        for rec in rows:
                            grp = uid_groups.get(rec.uid)
                            if grp is None:
                                continue
                            pfs = perframes_by_uid.get(rec.uid, [])
                            pf = pfs[f_idx] if f_idx < len(pfs) else None
                            obs_sec = max(0.0, float(f_idx - grp.first_frame + 1) / float(fps))
                            beh_val = pf.behavior if pf is not None else "static"

                            t_live = to_legacy_track(
                                tid=rec.track_id,
                                label=rec.user_label,
                                behavior_state=beh_val,
                                report_like=pf,
                                pose=rec.pose,
                                frame_h=height,
                                is_ghost=rec.is_ghost,
                                lost_frames=0,
                                head_kpt_idx=rec.head_kpt_idx,
                                counted=grp.counted,
                                c_unit=c_unit,
                                observed_sec=obs_sec,
                            )
                            frame_tracks.append(t_live)

                            cx = 0.5 * float(rec.bbox[0] + rec.bbox[2])
                            cy = 0.5 * float(rec.bbox[1] + rec.bbox[3])
                            trails[rec.uid].append((cx, cy))

                            re_list = reoriented_poses_by_uid.get(rec.uid)
                            pose_draw = (
                                re_list[f_idx]
                                if (re_list is not None and f_idx < len(re_list) and re_list[f_idx] is not None)
                                else rec.pose
                            )
                            head_pt = (
                                (float(pose_draw.dense[0, 0]), float(pose_draw.dense[0, 1]))
                                if (pose_draw is not None and pose_draw.dense is not None and len(pose_draw.dense) > 0)
                                else None
                            )
                            draw_items.append(
                                DrawItem(
                                    track_id=rec.track_id,
                                    bbox=rec.bbox,
                                    is_ghost=rec.is_ghost,
                                    lost_frames=0,
                                    user_label=rec.user_label,
                                    contour=rec.pose.contour if rec.pose is not None else None,
                                    head_xy=head_pt,
                                    trail=list(trails[rec.uid]),
                                    counted=grp.counted,
                                    counts={
                                        "c_count": t_live["c_count"],
                                        "s_count": t_live["s_count"],
                                        "is_bidirectional": False,
                                        "avg_freq": t_live["avg_freq"],
                                        "bends_per_min": t_live["bends_per_min"],
                                        "motion_type_legacy": t_live["motion_type"],
                                        "activity_score": t_live["activity_score"],
                                    },
                                )
                            )

                            # 仅对 counted=True 的主轨迹写出子视频（后进入的虫不生成子视频）
                            if grp.counted and rec.uid in primary_uids:
                                sw = sub_writers.get(rec.track_id)
                                if sw is None:
                                    sw = SubVideoWriter(
                                        sub_dir=sub_dir,
                                        track_id=rec.track_id,
                                        fps=fps,
                                        initial_bbox=rec.bbox,
                                        out_size=(sub_size, sub_size),
                                    )
                                    sub_writers[rec.track_id] = sw
                                sw.write_frame(frame, rec.bbox, is_ghost=rec.is_ghost)

                        active_cnt = sum(1 for r in rows if not r.is_ghost)
                        ghost_cnt = sum(1 for r in rows if r.is_ghost)
                        rendered = draw_frame(frame, draw_items, f_idx, active_cnt, ghost_cnt)
                        main_writer.write(rendered)

                        log_entries.append(
                            {
                                "frame": f_idx + 1,
                                "active_tracks": active_cnt,
                                "tracks": frame_tracks,
                            }
                        )
                        f_idx += 1
                        if f_idx % 15 == 0 or f_idx == total_frames:
                            ev_sink.emit(
                                "phase",
                                {
                                    "task_id": job.task_id,
                                    "phase": "render",
                                    "progress": round(f_idx / max(1, total_frames), 3),
                                },
                                room=job.task_id,
                            )
                rendered_main = not was_cancelled and not control.cancelled
            finally:
                main_writer.release()
                src.close()

        if was_cancelled or control.cancelled:
            for sw in sub_writers.values():
                sw.close(total_cycles=None)
            return PassBResult(
                video_url="",
                log_url="",
                sub_videos_url="",
                summary_url="",
                csv_url="",
                id_pose_summary=[],
                dropped_uids=dropped_uids,
            )

        if rendered_main:
            faststart_mp4(video_out_path)

        # 关闭子视频并重命名为 worm_{id}_cycles_{c+s}.mp4
        summary_by_tid = {int(item["track_id"]): item for item in id_pose_summary}
        for tid, sw in sub_writers.items():
            s_item = summary_by_tid.get(tid, {})
            tot_cyc = int(s_item.get("c_count", 0)) + int(s_item.get("s_count", 0))
            sw.close(total_cycles=tot_cyc)

        # 3. 导出 JSON / CSV / ZIP
        from server.exporters import csv_report, summary_json, zip_pack  # 延迟导入，避免与 server 包循环依赖
        summary_json.write(summary_out_path, id_pose_summary)
        csv_filename = csv_report.write(result_dir, base_prefix, id_pose_summary)
        log_out_path.write_text(json.dumps(log_entries, ensure_ascii=False), encoding="utf-8")
        zip_pack.write(zip_out_path, sub_dir)

        video_url = f"/media/{video_out_path.name}"
        log_url = f"/download/{log_out_path.name}"
        sub_videos_url = f"/download/{zip_out_path.name}"
        summary_url = f"/download/{summary_out_path.name}"
        csv_url = f"/download/{csv_filename}"

        ev_sink.update_state(
            status="completed",
            phase="done",
            video_url=video_url,
            log_url=log_url,
            sub_videos_url=sub_videos_url,
            summary_url=summary_url,
            csv_url=csv_url,
            id_pose_summary=id_pose_summary,
        )
        ev_sink.emit(
            "complete",
            {
                "task_id": job.task_id,
                "video_url": video_url,
                "log_url": log_url,
                "sub_videos_url": sub_videos_url,
                "summary_url": summary_url,
                "csv_url": csv_url,
                "id_pose_summary": id_pose_summary,
            },
            room=job.task_id,
        )

        return PassBResult(
            video_url=video_url,
            log_url=log_url,
            sub_videos_url=sub_videos_url,
            summary_url=summary_url,
            csv_url=csv_url,
            id_pose_summary=id_pose_summary,
            dropped_uids=dropped_uids,
            video_path=str(video_out_path),
            summary_path=str(summary_out_path),
            csv_path=str(result_dir / csv_filename),
            log_path=str(log_out_path),
            zip_path=str(zip_out_path),
        )
