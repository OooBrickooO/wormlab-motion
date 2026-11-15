"""03 §7: 逐帧行为分类 (classify_behavior) 与活性分计算。"""
from __future__ import annotations

from wormlab.config import KinCfg
from wormlab.schemas import Behavior

from .features import Feat
from .wave import GateResult


def classify_behavior(
    frame_idx: int,
    feat: Feat | None,
    gate: GateResult,
    s_active: bool,
    in_curl: bool,
    swing_last_high: int,
    swing_last_flip: int,
    speed_bls: float,
    cfg: KinCfg,
    fps: float,
) -> Behavior:
    """按优先级判定单帧行为状态。"""
    if feat is None:
        return Behavior.INVALID

    win_1s = max(1, int(round(1.0 * float(fps))))
    if cfg.swing.enabled and (
        (frame_idx - swing_last_high) <= win_1s or (frame_idx - swing_last_flip) <= win_1s
    ):
        return Behavior.THRASH

    if cfg.turn.enabled and in_curl:
        return Behavior.TURN

    if gate.ok and s_active:
        return Behavior.CRAWL

    max_amp = max(float(gate.amp_ant), float(gate.amp_post))
    if float(speed_bls) < float(cfg.static_speed_bls) and max_amp < float(cfg.static_amp_rad):
        return Behavior.STATIC

    return Behavior.MOVING


def behavior_to_legacy_motion_type(beh: Behavior) -> str:
    """映射至旧前端/导出契约的 motion_type 字符串。"""
    mapping = {
        Behavior.STATIC: "static",
        Behavior.CRAWL: "sinusoidal",
        Behavior.THRASH: "turning",
        Behavior.TURN: "turning",
        Behavior.MOVING: "moving",
        Behavior.INVALID: "unknown",
    }
    return mapping.get(beh, "unknown")
