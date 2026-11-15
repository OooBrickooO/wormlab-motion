"""T05 单测：配置加载、profile 覆盖、未知键报错与秒→帧换算。"""
from __future__ import annotations

import pytest

from wormlab.config import load_config


def test_load_crawl_agar_default() -> None:
    cfg = load_config("crawl_agar")
    assert cfg.profile == "crawl_agar"
    assert cfg.detection.imgsz == 416
    assert cfg.kinematics.s_run.enabled is True
    assert cfg.kinematics.swing.enabled is False
    assert cfg.kinematics.turn.enabled is True
    assert cfg.counting.late_entry_grace_s == pytest.approx(1.0)
    assert cfg.counting.c_unit == "cycle"


def test_load_swim_liquid_profile() -> None:
    cfg = load_config("swim_liquid")
    assert cfg.profile == "swim_liquid"
    assert cfg.kinematics.s_run.enabled is False
    assert cfg.kinematics.turn.enabled is False
    assert cfg.kinematics.swing.enabled is True
    assert cfg.kinematics.smooth_s == pytest.approx(0.07)
    assert cfg.kinematics.wave.win_s == pytest.approx(1.0)


def test_overrides_and_unknown_key_error() -> None:
    cfg = load_config("crawl_agar", overrides={"counting": {"c_unit": "bend"}})
    assert cfg.counting.c_unit == "bend"

    with pytest.raises(ValueError, match="未知键"):
        load_config("crawl_agar", overrides={"kinematics": {"non_existent_param": 123}})


def test_runtime_cfg_frames_and_sec() -> None:
    cfg = load_config("crawl_agar")
    rt30 = cfg.with_fps(30.0)
    rt10 = cfg.with_fps(10.0)

    assert rt30.frames(3.0) == 90
    assert rt10.frames(3.0) == 30
    assert rt30.frames(0.001) == 1  # 至少 1 帧
    assert rt30.sec(45) == pytest.approx(1.5)
    assert rt30.detection.imgsz == 416
