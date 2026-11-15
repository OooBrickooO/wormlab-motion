"""03: 运动学特征、行波门限、S/C 计数与延迟因果引擎。"""
from __future__ import annotations

from .behavior import behavior_to_legacy_motion_type, classify_behavior
from .counter import KinematicsEngine
from .features import Feat, compute_features
from .metrics import LiveCounts, PerFrame, TrackReport, compute_spatial_stats
from .modes import compute_total_turn_c, eigenworm_basis, modal_phase, project_trig_modes
from .s_runs import SEvent, SRunCounter
from .swing import SwingCounter
from .turns import TurnCounter
from .wave import GateResult, wave_gate

__all__ = [
    "Feat",
    "GateResult",
    "KinematicsEngine",
    "LiveCounts",
    "PerFrame",
    "SEvent",
    "SRunCounter",
    "SwingCounter",
    "TrackReport",
    "TurnCounter",
    "behavior_to_legacy_motion_type",
    "classify_behavior",
    "compute_features",
    "compute_spatial_stats",
    "compute_total_turn_c",
    "eigenworm_basis",
    "modal_phase",
    "project_trig_modes",
    "wave_gate",
]
