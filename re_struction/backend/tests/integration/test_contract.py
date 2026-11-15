"""T26 & T27 (05 §7–§10 & API_CONTRACT): FastAPI 全路由契约测试与落盘恢复集成测试。"""
from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi.testclient import TestClient

from server.app_factory import create_app
from server.tasks import TaskStore
from wormlab.schemas import Detection


def _make_jpeg_bytes(w: int = 320, h: int = 240) -> bytes:
    img = np.full((h, w, 3), 220, dtype=np.uint8)
    # 画一条深色线虫供手动框选阈值分割测试
    cv2.line(img, (80, 100), (180, 120), (30, 30, 30), 7)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


class _StubPreviewDetector:
    def infer(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        mask = np.zeros((40, 100), dtype=bool)
        mask[16:24, 5:95] = True
        det = Detection(
            bbox=np.array([80.0, 90.0, 180.0, 130.0], dtype=np.float32),
            conf=0.93,
            mask=mask,
            mask_origin=(80, 90),
            det_idx=0,
        )
        return [[det] for _ in frames]


class _StubEggModel:
    names = {0: "egg", 1: "worm"}

    def __call__(self, frame: np.ndarray, **kwargs: object) -> list[SimpleNamespace]:
        boxes = SimpleNamespace(
            xyxy=np.array([[10.0, 10.0, 25.0, 25.0], [40.0, 40.0, 55.0, 55.0]], dtype=np.float32),
            conf=np.array([0.96, 0.91], dtype=np.float32),
            cls=np.array([0, 0], dtype=np.int32),
        )
        return [SimpleNamespace(boxes=boxes)]


def test_models_profiles_health_and_upload_lifecycle(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "origin.pt").write_bytes(b"fake_pt")
    (models_dir / "CJLUdata_Old.pt").write_bytes(b"fake_pt2")
    (models_dir / "EggCount.pt").write_bytes(b"fake_egg")

    data_dir = tmp_path / "data"
    fastapi_app, _, _ = create_app(data_dir=data_dir, models_dir=models_dir, use_fake_worker=True)

    with TestClient(fastapi_app) as client:
        # 1. /health, /profiles, /models, /select_model
        r_health = client.get("/health")
        assert r_health.status_code == 200
        assert r_health.json()["ok"] is True

        r_prof = client.get("/profiles")
        assert r_prof.status_code == 200
        assert r_prof.json()["default"] == "crawl_agar"

        r_models = client.get("/models")
        assert r_models.status_code == 200
        m_data = r_models.json()
        assert "origin.pt" in m_data["models"]
        assert "EggCount.pt" not in m_data["models"]

        assert client.post("/select_model", json={}).status_code == 400
        assert client.post("/select_model", json={"model_name": "nonexistent.pt"}).status_code == 404
        r_sel = client.post("/select_model", json={"model_name": "CJLUdata_Old.pt"})
        assert r_sel.status_code == 200 and r_sel.json()["current_model"] == "CJLUdata_Old.pt"

        # 2. /upload 参数校验与 202 创建
        assert client.post("/upload", data={"worm_count": "5"}).status_code == 400
        assert (
            client.post(
                "/upload",
                files={"video": ("bad.txt", b"hello", "text/plain")},
                data={"worm_count": "5"},
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/upload",
                files={"video": ("a.mp4", b"fake_mp4", "video/mp4")},
                data={"worm_count": "0"},
            ).status_code
            == 400
        )

        r_up = client.post(
            "/upload",
            files={"video": ("test_worm.mp4", b"fake_mp4_content", "video/mp4")},
            data={
                "worm_count": "8",
                "profile": "swim_liquid",
                "c_unit": "bend",
                "head_config": json.dumps([{"det_idx": 0, "head_kpt_idx": 0, "user_label": "W1", "bbox": [10, 10, 50, 50]}]),
            },
        )
        assert r_up.status_code == 202
        task_id = r_up.json()["task_id"]

        # 3. /task/<id>/status & 控制接口
        r_st = client.get(f"/task/{task_id}/status")
        assert r_st.status_code == 200
        st_json = r_st.json()
        assert st_json["worm_count"] == 8
        assert st_json["profile"] == "swim_liquid"

        assert client.post(f"/task/{task_id}/pause").json()["status"] == "paused"
        assert client.get(f"/task/{task_id}/status").json()["status"] == "paused"

        assert client.post(f"/task/{task_id}/resume").json()["status"] == "processing"

        r_swc = client.post(f"/task/{task_id}/set_worm_count", json={"worm_count": 12})
        assert r_swc.status_code == 200 and r_swc.json()["worm_count"] == 12
        r_swc_none = client.post(f"/task/{task_id}/set_worm_count", json={"worm_count": 0})
        assert r_swc_none.status_code == 200 and r_swc_none.json()["worm_count"] is None

        assert client.post(f"/task/{task_id}/cancel").json()["status"] == "cancelled"
        assert client.get("/task/not_a_real_task/status").status_code == 404

        # 4. 验证终态落盘（不含 latest_frame）与重启恢复
        tasks_dir = data_dir / "results" / "tasks"
        persisted_file = tasks_dir / f"{task_id}.json"
        assert persisted_file.exists()
        saved_doc = json.loads(persisted_file.read_text(encoding="utf-8"))
        assert "latest_frame" not in saved_doc
        assert saved_doc["status"] == "cancelled"

        recovered_store = TaskStore(tasks_dir)
        assert recovered_store.load_history() >= 1
        rec_state = recovered_store.get(task_id)
        assert rec_state is not None and rec_state["status"] == "cancelled"

        # 5. /download, /media (含 Range), /sub_video, /task/<id>/sub_videos
        results_dir = data_dir / "results"
        (results_dir / "sample_out.mp4").write_bytes(b"0123456789abcdef")
        sub_dir = results_dir / "sub_videos" / task_id
        sub_dir.mkdir(parents=True, exist_ok=True)
        (sub_dir / "worm_1_cycles_3.mp4").write_bytes(b"subvideo_bytes")

        assert client.get("/download/sample_out.mp4").status_code == 200
        assert client.get("/download/missing.mp4").status_code == 404

        r_range = client.get("/media/sample_out.mp4", headers={"Range": "bytes=0-5"})
        assert r_range.status_code == 206
        assert r_range.content == b"012345"

        r_subs = client.get(f"/task/{task_id}/sub_videos")
        assert r_subs.status_code == 200
        assert len(r_subs.json()["videos"]) == 1

        r_sub_file = client.get(f"/sub_video/{task_id}/worm_1_cycles_3.mp4")
        assert r_sub_file.status_code == 200


def test_preview_and_egg_count_routes(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "origin.pt").write_bytes(b"fake_pt")
    data_dir = tmp_path / "data"

    fastapi_app, _, _ = create_app(data_dir=data_dir, models_dir=models_dir, use_fake_worker=True)
    # 注入轻量级桩模型以加速契约测试
    fastapi_app.state.registry.get_preview_detector = lambda: _StubPreviewDetector()
    fastapi_app.state.registry.get_egg_model = lambda: (_StubEggModel(), "cpu")

    jpg_bytes = _make_jpeg_bytes()

    with TestClient(fastapi_app) as client:
        # /preview_count
        r_prev = client.post("/preview_count", files={"frame": ("f0.jpg", io.BytesIO(jpg_bytes), "image/jpeg")})
        assert r_prev.status_code == 200
        prev_json = r_prev.json()
        assert prev_json["count"] == 1 and prev_json["raw_count"] == 1 and prev_json["annotated"]

        # /analyze_first_frame
        r_aff = client.post("/analyze_first_frame", files={"frame": ("f0.jpg", io.BytesIO(jpg_bytes), "image/jpeg")})
        assert r_aff.status_code == 200
        aff_json = r_aff.json()
        assert aff_json["frame_width"] == 320 and aff_json["frame_height"] == 240
        assert len(aff_json["worms"]) == 1
        assert len(aff_json["worms"][0]["keypoints"]) == 11

        # /extract_manual_worm
        r_emw = client.post(
            "/extract_manual_worm",
            files={"frame": ("f0.jpg", io.BytesIO(jpg_bytes), "image/jpeg")},
            data={"bbox": json.dumps([70, 85, 190, 135])},
        )
        assert r_emw.status_code == 200
        emw_json = r_emw.json()
        assert len(emw_json["keypoints"]) == 11

        # 选区过小 -> 400
        r_small = client.post(
            "/extract_manual_worm",
            files={"frame": ("f0.jpg", io.BytesIO(jpg_bytes), "image/jpeg")},
            data={"bbox": json.dumps([10, 10, 12, 12])},
        )
        assert r_small.status_code == 400

        # /egg_count
        r_egg = client.post("/egg_count", files={"file": ("plate1.jpg", io.BytesIO(jpg_bytes), "image/jpeg")})
        assert r_egg.status_code == 200
        egg_json = r_egg.json()
        assert egg_json["egg_count"] == 2
        assert egg_json["egg_count_90"] == 2
        assert egg_json["egg_count_95"] == 1
        assert (data_dir / "results" / "Eggcount" / "plate1.json").exists()
