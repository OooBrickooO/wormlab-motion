"""05 §10: 线程安全模型注册表与懒加载预览/虫卵检测器 (model_registry.py)。"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from wormlab.config import Config
from wormlab.detection.detector import Detector


class ModelRegistry:
    """管理 models/ 目录下的权重枚举、当前选中模型、预览 Detector 与 EggCount 模型实例。"""

    def __init__(self, cfg: Config, models_dir: str | Path | None = None) -> None:
        self.cfg = cfg
        re_backend = Path(__file__).resolve().parents[1]
        self.models_dir = Path(models_dir) if models_dir else (re_backend / cfg.paths.models_dir)
        self.lock = threading.Lock()
        self._current_name: str = self._pick_initial_model()
        self._preview_detector: Detector | None = None
        self._egg_model: Any | None = None
        self._egg_device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def _pick_initial_model(self) -> str:
        avail = self.list_models()
        for pref in self.cfg.detection.model_preference:
            if pref in avail:
                return pref
        if self.cfg.detection.default_model in avail:
            return self.cfg.detection.default_model
        return avail[0] if avail else self.cfg.detection.default_model

    def list_models(self) -> list[str]:
        if not self.models_dir.exists():
            return []
        egg_name = self.cfg.detection.egg_model.lower()
        pts = sorted(
            p.name
            for p in self.models_dir.glob("*.pt")
            if p.name.lower() != egg_name
        )
        return pts

    @property
    def current_name(self) -> str:
        with self.lock:
            return self._current_name

    def set_current(self, name: str) -> Path:
        clean = Path(name).name
        target = self.models_dir / clean
        if not target.exists():
            raise FileNotFoundError(f"模型文件 {clean} 不存在")
        with self.lock:
            self._current_name = clean
            self._preview_detector = None
        return target

    def model_path(self, name: str | None = None) -> Path:
        if name:
            clean = Path(name).name
            cand = self.models_dir / clean
            if cand.exists():
                return cand
        return self.models_dir / self.current_name

    def get_preview_detector(self) -> Detector:
        with self.lock:
            if self._preview_detector is None:
                m_path = self.models_dir / self._current_name
                if not m_path.exists():
                    raise FileNotFoundError(f"预览模型文件不存在: {m_path}")
                self._preview_detector = Detector(m_path, self.cfg)
            return self._preview_detector

    def get_egg_model(self) -> tuple[Any, str]:
        with self.lock:
            if self._egg_model is None:
                egg_path = self.models_dir / self.cfg.detection.egg_model
                if not egg_path.exists():
                    raise FileNotFoundError(f"虫卵检测模型文件不存在: {egg_path}")
                model = YOLO(str(egg_path))
                model.to(self._egg_device)
                self._egg_model = model
            return self._egg_model, self._egg_device
