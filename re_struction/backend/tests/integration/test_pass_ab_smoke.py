"""T33: Pass A -> Pass B 稳定性冒烟、中断控制与重启恢复集成测试。"""
from __future__ import annotations

import json
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from server.tasks import FakeWorker, TaskManager, TaskStore
from wormlab.config import load_config
from wormlab.pipeline.cache import CacheReader
from wormlab.pipeline.late_entry import partition_tracks_by_uid
from wormlab.pipeline.pass_a import Control, JobSpec, PassA
from wormlab.pipeline.pass_b import PassB
from wormlab.schemas import Detection


class _RecordingSink:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.state: dict[str, Any] = {}

    def emit(self, event: str, data: dict[str, Any], room: str | None = None) -> None:
        with self._lock:
            self.events.append((event, data))

    def update_state(self, **kwargs: Any) -> None:
        with self._lock:
            self.state.update(kwargs)


class _ContourWormDetector:
    """从合成灰度视频帧中提取深色条状虫体并返回掩码与 Detection（不依赖 GPU/YOLO）。"""

    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = float(delay_s)

    def infer(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        if self.delay_s > 0.0:
            time.sleep(self.delay_s)
        out: list[list[Detection]] = []
        for frm in frames:
            gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            fg = (gray < 140).astype(np.uint8)
            n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
            dets: list[Detection] = []
            h_img, w_img = gray.shape
            for lbl_id in range(1, n_lbl):
                area = int(stats[lbl_id, cv2.CC_STAT_AREA])
                if area < 40:
                    continue
                x = int(stats[lbl_id, cv2.CC_STAT_LEFT])
                y = int(stats[lbl_id, cv2.CC_STAT_TOP])
                w = int(stats[lbl_id, cv2.CC_STAT_WIDTH])
                h = int(stats[lbl_id, cv2.CC_STAT_HEIGHT])
                x1, y1 = max(0, x - 3), max(0, y - 3)
                x2, y2 = min(w_img, x + w + 3), min(h_img, y + h + 3)
                sub_mask = labels[y1:y2, x1:x2] == lbl_id
                dets.append(
                    Detection(
                        bbox=np.array([float(x1), float(y1), float(x2), float(y2)], dtype=np.float32),
                        conf=0.92,
                        mask=sub_mask,
                        mask_origin=(x1, y1),
                        det_idx=len(dets),
                    )
                )
            out.append(dets)
        return out


def _draw_worm(img: np.ndarray, cx: float, cy: float, length: float, phase: float) -> None:
    xs = np.linspace(-length * 0.5, length * 0.5, 25)
    pts = np.stack(
        [
            cx + xs,
            cy + 9.0 * np.sin(phase - 2.0 * np.pi * (xs / max(1.0, length))),
        ],
        axis=1,
    ).astype(np.int32)
    cv2.polylines(img, [pts], False, (35, 35, 35), thickness=7, lineType=cv2.LINE_AA)


def _write_synthetic_video(
    out_mp4: Path,
    n_frames: int,
    fps: float = 30.0,
    width: int = 360,
    height: int = 240,
    include_late_worm: bool = False,
) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened(), f"无法创建测试视频: {out_mp4}"
    try:
        for i in range(n_frames):
            t = float(i) / float(fps)
            frm = np.full((height, width, 3), 225, dtype=np.uint8)
            # 虫 1：自首帧起向前爬行
            _draw_worm(frm, 70.0 + 18.0 * t, 75.0, 90.0, phase=2.0 * np.pi * 0.5 * t)
            # 虫 2：自首帧起轻微原地停留
            _draw_worm(frm, 180.0, 170.0, 85.0, phase=0.2 * np.sin(2.0 * np.pi * t))
            # 虫 3（可选）：2.0s 后才进入画面 -> 必须判为 late_entry 并清零
            if include_late_worm and t >= 2.0:
                _draw_worm(frm, 100.0 + 20.0 * (t - 2.0), 125.0, 85.0, phase=2.0 * np.pi * 0.6 * t)
            writer.write(frm)
    finally:
        writer.release()


def test_full_pass_ab_smoke_with_exports(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    video_path = tmp_path / "smoke.mp4"
    _write_synthetic_video(video_path, n_frames=120, fps=30.0, include_late_worm=True)

    cfg = load_config("crawl_agar", overrides={"paths": {"data_dir": str(data_dir)}})
    job = JobSpec(
        task_id="smoke_task_01",
        video_path=str(video_path),
        profile="crawl_agar",
        c_unit="cycle",
        worm_count=None,
        model_path="stub.pt",
        result_dir=str(data_dir / "results"),
        cache_dir=str(data_dir / "cache" / "smoke_task_01"),
    )
    sink = _RecordingSink()
    ctrl = Control()

    cache_dir = PassA(cfg=cfg, job=job, detector=_ContourWormDetector(), sink=sink, ctrl=ctrl).run()
    assert (cache_dir / "tracks.npz").exists()

    res = PassB(cfg=cfg).run(cache_dir=cache_dir, job=job, sink=sink, ctrl=ctrl, render_videos=True)
    assert res.video_path and Path(res.video_path).exists()
    assert res.summary_path and Path(res.summary_path).exists()
    assert res.csv_path and Path(res.csv_path).exists()
    assert res.zip_path and Path(res.zip_path).exists()

    summary_doc = json.loads(Path(res.summary_path).read_text(encoding="utf-8"))
    assert isinstance(summary_doc, list) and len(summary_doc) == len(res.id_pose_summary)
    # 缓存中应记录后进入虫，但默认 include_late_in_outputs=False 不输出至终稿汇总
    reader = CacheReader(cache_dir)
    groups = partition_tracks_by_uid(reader.records, fps=float(reader.meta["fps"]), grace_s=cfg.counting.late_entry_grace_s)
    assert any(g.late_entry for g in groups.values())
    assert all(not row.get("late_entry") for row in summary_doc)

    # 开启 include_late_in_outputs=True 重跑 Pass B：后进入虫保留在汇总中但所有计数强制清零
    cfg_inc = load_config("crawl_agar", overrides={"paths": {"data_dir": str(data_dir)}, "counting": {"include_late_in_outputs": True}})
    res_inc = PassB(cfg=cfg_inc).run(cache_dir=cache_dir, job=job, sink=sink, ctrl=ctrl, render_videos=False)
    late_entries = [row for row in res_inc.id_pose_summary if row.get("late_entry")]
    assert len(late_entries) >= 1
    for le in late_entries:
        assert le["s_count"] == 0
        assert le["c_cycles"] == 0
        assert le["c_bends"] == 0

    with zipfile.ZipFile(res.zip_path, "r") as zf:
        names = zf.namelist()
        assert len(names) >= 2
        assert all(n.endswith(".mp4") for n in names)


def test_pass_a_pause_resume_cancel_and_set_worm_count(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    video_path = tmp_path / "ctrl.mp4"
    _write_synthetic_video(video_path, n_frames=150, fps=30.0, include_late_worm=False)

    cfg = load_config("crawl_agar", overrides={"paths": {"data_dir": str(data_dir)}})
    job = JobSpec(
        task_id="ctrl_task_01",
        video_path=str(video_path),
        profile="crawl_agar",
        c_unit="cycle",
        worm_count=5,
        model_path="stub.pt",
        result_dir=str(data_dir / "results"),
        cache_dir=str(data_dir / "cache" / "ctrl_task_01"),
    )
    sink = _RecordingSink()
    ctrl = Control()
    det = _ContourWormDetector(delay_s=0.05)

    t = threading.Thread(target=lambda: PassA(cfg=cfg, job=job, detector=det, sink=sink, ctrl=ctrl).run(), daemon=True)
    t.start()

    # 等 Pass A 推进到至少第 8 帧后暂停
    deadline = time.time() + 5.0
    while sink.state.get("current_frame", 0) < 8 and time.time() < deadline:
        time.sleep(0.05)
    assert sink.state.get("current_frame", 0) >= 8

    ctrl.pause()
    time.sleep(0.25)
    paused_at = sink.state.get("current_frame", 0)
    time.sleep(0.35)
    assert abs(sink.state.get("current_frame", 0) - paused_at) <= cfg.detection.batch_size

    # 暂停期间下发动态虫数上限与恢复指令
    ctrl.set_worm_count(1)
    ctrl.resume()
    deadline2 = time.time() + 5.0
    while sink.state.get("current_frame", 0) <= paused_at + 4 and time.time() < deadline2:
        time.sleep(0.05)
    assert sink.state.get("current_frame", 0) > paused_at

    # 触发取消 -> 快速退出且不抛异常
    ctrl.cancel()
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert ctrl.cancelled is True


def test_pass_ab_long_clip_stability(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    video_path = tmp_path / "long_clip.mp4"
    _write_synthetic_video(video_path, n_frames=360, fps=30.0, include_late_worm=False)

    cfg = load_config("swim_liquid", overrides={"paths": {"data_dir": str(data_dir)}})
    job = JobSpec(
        task_id="long_task_01",
        video_path=str(video_path),
        profile="swim_liquid",
        c_unit="bend",
        worm_count=2,
        model_path="stub.pt",
        result_dir=str(data_dir / "results"),
        cache_dir=str(data_dir / "cache" / "long_task_01"),
    )
    sink = _RecordingSink()
    ctrl = Control()

    cache_dir = PassA(cfg=cfg, job=job, detector=_ContourWormDetector(), sink=sink, ctrl=ctrl).run()
    reader = CacheReader(cache_dir)
    assert int(reader.meta["total_frames"]) == 360
    res = PassB(cfg=cfg).run(cache_dir=cache_dir, job=job, sink=sink, ctrl=ctrl, render_videos=False)
    assert len(res.id_pose_summary) >= 1


def test_task_manager_worker_restart_recovery(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "data" / "results" / "tasks"
    store = TaskStore(tasks_dir)
    worker = FakeWorker(tmp_path / "data")
    mgr = TaskManager(store, worker)

    j1 = JobSpec(task_id="job_1", video_path="v1.mp4", worm_count=3)
    j2 = JobSpec(task_id="job_2", video_path="v2.mp4", worm_count=4)
    store.create(j1.task_id, "v1.mp4", "v1.mp4", worm_count=3, queued=True)
    store.create(j2.task_id, "v2.mp4", "v2.mp4", worm_count=4, queued=True)
    mgr.submit(j1)
    mgr.submit(j2)

    assert mgr.current_task_id == "job_1"
    assert store.get("job_2")["queued"] is True

    # job_1 终态完成并自动出队启动 job_2
    store.update("job_1", status="completed", video_url="/download/out.mp4")
    mgr.on_done("job_1")
    assert mgr.current_task_id == "job_2"
    assert (tasks_dir / "job_1.json").exists()

    # 模拟 job_2 进行中服务崩溃重启：仅终态任务落盘可恢复
    store.update("job_2", status="processing", current_frame=55)
    store._persist_terminal("job_2", store.get("job_2"))

    restarted_store = TaskStore(tasks_dir)
    loaded = restarted_store.load_history()
    assert loaded == 2
    assert restarted_store.get("job_1")["status"] == "completed"
    # 重启时未达终态的残存任务应被安全收敛为 cancelled
    assert restarted_store.get("job_2")["status"] == "cancelled"


def test_pass_b_cancel_aborts_cleanly_without_completed_urls(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    video_path = tmp_path / "cancel_b.mp4"
    _write_synthetic_video(video_path, n_frames=90, fps=30.0, include_late_worm=False)

    cfg = load_config("crawl_agar", overrides={"paths": {"data_dir": str(data_dir)}})
    job = JobSpec(
        task_id="cancel_b_01",
        video_path=str(video_path),
        profile="crawl_agar",
        c_unit="cycle",
        worm_count=None,
        model_path="stub.pt",
        result_dir=str(data_dir / "results"),
        cache_dir=str(data_dir / "cache" / "cancel_b_01"),
    )
    sink = _RecordingSink()
    ctrl_a = Control()
    cache_dir = PassA(cfg=cfg, job=job, detector=_ContourWormDetector(), sink=sink, ctrl=ctrl_a).run()

    class _CancelOnRenderSink(_RecordingSink):
        def __init__(self, ctrl: Control) -> None:
            super().__init__()
            self._ctrl = ctrl

        def update_state(self, **kwargs: Any) -> None:
            super().update_state(**kwargs)
            if kwargs.get("phase") == "render":
                self._ctrl.cancel()

    ctrl_b = Control()
    sink_b = _CancelOnRenderSink(ctrl_b)
    res = PassB(cfg=cfg).run(cache_dir=cache_dir, job=job, sink=sink_b, ctrl=ctrl_b, render_videos=True)
    assert res.video_url == ""
    assert res.sub_videos_url == ""
    assert sink_b.state.get("status") != "completed"


def test_store_cancelled_not_overwritten_and_shutdown_persists(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "data" / "results" / "tasks"
    store = TaskStore(tasks_dir)
    store.create("t_cancel", "a.mp4", "a.mp4", worm_count=2, queued=False)
    store.create("t_active", "b.mp4", "b.mp4", worm_count=3, queued=False)
    store.update("t_active", status="processing", current_frame=12)

    # 取消后禁止被过期 update_state 覆盖回 processing 或 completed
    store.update("t_cancel", status="cancelled", control="cancelled", queued=False)
    store.update("t_cancel", status="processing", current_frame=99)
    store.update("t_cancel", status="completed", video_url="/media/fake.mp4")
    assert store.get("t_cancel")["status"] == "cancelled"

    # 服务关闭时自动收敛残存运行中任务为 cancelled 并落盘
    n_conv = store.cancel_active_on_shutdown()
    assert n_conv == 1
    assert store.get("t_active")["status"] == "cancelled"

    reloaded = TaskStore(tasks_dir)
    assert reloaded.load_history() == 2
    assert reloaded.get("t_active")["status"] == "cancelled"
    assert reloaded.get("t_cancel")["status"] == "cancelled"
