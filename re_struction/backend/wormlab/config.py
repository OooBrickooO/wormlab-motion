"""01 §3: 配置加载、深合并、严格未知键校验与秒→帧运行时换算。"""
from __future__ import annotations

import copy
import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsCfg:
    models_dir: str = "models"
    data_dir: str = "../data"


@dataclass(frozen=True)
class ServerCfg:
    host: str = "0.0.0.0"
    port: int = 5001
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:3001", "http://localhost:3000"])
    max_upload_mb: int = 2048
    allowed_ext: list[str] = field(default_factory=lambda: ["mp4", "avi", "mov"])
    image_ext: list[str] = field(default_factory=lambda: ["jpg", "jpeg", "png", "bmp"])


@dataclass(frozen=True)
class TileCfg:
    enabled: bool = False
    auto: bool = False
    size: int = 1088
    overlap: float = 0.2
    infer_imgsz: int | None = None
    seam_ios_thresh: float = 0.55


@dataclass(frozen=True)
class DetectionCfg:
    default_model: str = "origin.pt"
    model_preference: list[str] = field(
        default_factory=lambda: ["origin.pt", "OnlineVision.pt", "CJLUdata_Old.pt", "CJLUdata.pt", "best.pt"]
    )
    egg_model: str = "EggCount.pt"
    imgsz: int | str = 416
    oom_fallback: bool = True
    tile: TileCfg = field(default_factory=TileCfg)
    min_confidence: float = 0.30
    recovery_confidence: float | None = None
    min_area_ratio: float = 0.000015
    max_area_ratio: float | None = 0.10
    nms_iou: float = 0.3
    batch_size: int = 8
    mask_margin_ratio: float = 0.25


@dataclass(frozen=True)
class CountingCfg:
    late_entry_grace_s: float = 1.0
    include_late_in_outputs: bool = False
    c_unit: str = "cycle"


@dataclass(frozen=True)
class ImageModeCfg:
    fps: float = 12.0
    repeat_frames: int = 36


@dataclass(frozen=True)
class PipelineCfg:
    emit_every: int = 2
    jpeg_quality: int = 80
    frame_queue: int = 4
    infer_queue: int = 6
    pose_workers: int = 4
    sub_video_size: int = 200
    image_mode: ImageModeCfg = field(default_factory=ImageModeCfg)
    pose_every_n_frames: int = 1


@dataclass(frozen=True)
class TrackingWeightsCfg:
    iou: float = 0.25
    motion: float = 0.45
    direction: float = 0.15
    appearance: float = 0.15


@dataclass(frozen=True)
class StitchCfg:
    enabled: bool = False


@dataclass(frozen=True)
class RecoveryCfg:
    enabled: bool = False
    roi_size: int = 416
    roi_confidence: float = 0.12
    bg_sub_enabled: bool = True
    bg_sample_frames: int = 20
    bg_thresholds: list[int] = field(default_factory=lambda: [12, 9, 7])


@dataclass(frozen=True)
class TrackingCfg:
    confirm_hits: int = 5
    iou_match_thresh: float = 0.5
    iou_secondary: float = 0.3
    cost_threshold: float = 2.0
    max_speed: float = 50.0
    weights: TrackingWeightsCfg = field(default_factory=TrackingWeightsCfg)
    overlap_threshold: float = 0.3
    persistence_sec: float = 60.0
    ghost_overlay_sec: float = 3.0
    ghost_reid_max_lost: int = 90
    max_lost_sec_legacy: float = 15.0
    prediction_confidence: float = 0.18
    head_match_radius_px: float = 120.0
    stitch: StitchCfg = field(default_factory=StitchCfg)
    init_confidence: float | None = None
    suppress_parasite_unconfirmed: bool = False
    damp_lost_velocity: bool = False
    recovery: RecoveryCfg = field(default_factory=RecoveryCfg)


@dataclass(frozen=True)
class MaskRefineCfg:
    enabled: bool = False


@dataclass(frozen=True)
class PoseCfg:
    n_dense: int = 33
    min_mask_px: int = 30
    fragment_ratio: float = 0.25
    upscale_below_px: int = 120
    upscale_target_px: int = 240
    max_upscale: float = 4.0
    loop_hole_frac: float = 0.005
    center_penalty: float = 2.0
    branch_flag_ratio: float = 0.35
    smooth_sigma_frac: float = 0.03
    endcap: bool = True
    endcap_gain: float = 0.9
    ref_window_s: float = 2.0
    min_ref_samples: int = 10
    bad_area_hi: float = 1.35
    bad_area_lo: float = 0.55
    branch_bad_ratio: float = 0.5
    suspect_len_lo: float = 0.75
    suspect_len_hi: float = 1.25
    min_len_over_width: float = 2.2  # 原 quality.py 等效 3.0，放宽以兼容 C 形弯曲与低倍镜小虫
    reorient_gap_s: float = 1.0
    coil_aware_reorient: bool = False
    mask_refine: MaskRefineCfg = field(default_factory=MaskRefineCfg)


@dataclass(frozen=True)
class WaveCfg:
    win_s: float = 3.0
    min_valid_frac: float = 0.6
    r_min: float = 0.25
    eps_min: float = 0.40
    post_ratio_min: float = 0.35
    amp_post_min: float = 0.10


@dataclass(frozen=True)
class SRunCfg:
    enabled: bool = True
    k_body: float = 1.0
    dir_lock_rad: float = 0.8
    reverse_phase_eps_rad: float = 0.3
    reverse_disp_ratio: float = 0.03
    min_disp_ratio: float = 0.10
    gap_hold_s: float = 0.30
    slowdown_hold_s: float = 2.5
    bridge_gap_s: float = 0.0
    adaptive_k_body: bool = False
    eof_tail_frac: float | None = None


@dataclass(frozen=True)
class SwingCfg:
    enabled: bool = False
    bend_high: float = 0.9
    min_flip_interval_s: float = 0.08


@dataclass(frozen=True)
class TurnCfg:
    enabled: bool = True
    R_enter: float = 0.55
    R_exit: float = 0.72
    min_dwell_s: float = 0.2
    cooldown_s: float = 2.5
    max_curl_s: float = 2.0  # CURL 状态最大驻留秒数，超时自动释放回 EXT，防止永久锁死 S 计数
    min_peak_c_rad: float = 0.85  # CURL 期间需至少达到的 |c| 弯曲度，区分真实 C 弯与掩码残缺缩短


@dataclass(frozen=True)
class KinCfg:
    smooth_s: float = 0.17
    gap_max_s: float = 0.30
    ref_window_s: float = 2.0
    k_body: float = 1.0
    dc_remove_window_s: float = 5.0  # S 波模态慢时变 DC 扣除窗口 (s)；<= 0 关闭
    wave: WaveCfg = field(default_factory=WaveCfg)
    s_run: SRunCfg = field(default_factory=SRunCfg)
    swing: SwingCfg = field(default_factory=SwingCfg)
    turn: TurnCfg = field(default_factory=TurnCfg)
    static_speed_bls: float = 0.03
    static_amp_rad: float = 0.08


@dataclass(frozen=True)
class RuntimeCfg:
    """绑定具体视频帧率 (fps) 的只读配置包装器，提供秒↔帧换算。"""

    cfg: Config
    fps: float

    def frames(self, sec: float) -> int:
        """将秒数换算为帧数（至少 1 帧）。"""
        return max(1, int(round(float(sec) * float(self.fps))))

    def sec(self, n_frames: int) -> float:
        """将帧数换算为秒数。"""
        return float(n_frames) / max(1e-6, float(self.fps))

    def __getattr__(self, item: str) -> Any:
        return getattr(self.cfg, item)


@dataclass(frozen=True)
class Config:
    paths: PathsCfg = field(default_factory=PathsCfg)
    server: ServerCfg = field(default_factory=ServerCfg)
    detection: DetectionCfg = field(default_factory=DetectionCfg)
    counting: CountingCfg = field(default_factory=CountingCfg)
    pipeline: PipelineCfg = field(default_factory=PipelineCfg)
    tracking: TrackingCfg = field(default_factory=TrackingCfg)
    pose: PoseCfg = field(default_factory=PoseCfg)
    kinematics: KinCfg = field(default_factory=KinCfg)
    profile: str = "crawl_agar"

    def with_fps(self, fps: float) -> RuntimeCfg:
        """返回绑定 fps 的 RuntimeCfg。"""
        if fps <= 0:
            raise ValueError(f"fps 必须 > 0，收到 {fps}")
        return RuntimeCfg(cfg=self, fps=float(fps))

    def hash(self) -> str:
        """返回配置内容的稳定短哈希字符串。"""
        import hashlib
        import json

        raw = json.dumps(dataclasses.asdict(self), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _dict_to_dataclass(cls: type[Any], data: dict[str, Any], prefix: str = "") -> Any:
    if not dataclasses.is_dataclass(cls):
        return data
    valid_fields = {f.name: f for f in dataclasses.fields(cls)}
    for k in data:
        if k not in valid_fields:
            full_key = f"{prefix}.{k}" if prefix else k
            raise ValueError(f"配置中存在未知键: '{full_key}'")

    kwargs: dict[str, Any] = {}
    # 解析类型注解映射
    type_map = {
        "PathsCfg": PathsCfg,
        "ServerCfg": ServerCfg,
        "TileCfg": TileCfg,
        "DetectionCfg": DetectionCfg,
        "CountingCfg": CountingCfg,
        "ImageModeCfg": ImageModeCfg,
        "PipelineCfg": PipelineCfg,
        "TrackingWeightsCfg": TrackingWeightsCfg,
        "StitchCfg": StitchCfg,
        "RecoveryCfg": RecoveryCfg,
        "TrackingCfg": TrackingCfg,
        "MaskRefineCfg": MaskRefineCfg,
        "PoseCfg": PoseCfg,
        "WaveCfg": WaveCfg,
        "SRunCfg": SRunCfg,
        "SwingCfg": SwingCfg,
        "TurnCfg": TurnCfg,
        "KinCfg": KinCfg,
    }
    for name, f_obj in valid_fields.items():
        if name not in data:
            continue
        val = data[name]
        f_type_str = str(f_obj.type)
        sub_cls = type_map.get(f_type_str)
        if sub_cls is not None and isinstance(val, dict):
            sub_prefix = f"{prefix}.{name}" if prefix else name
            kwargs[name] = _dict_to_dataclass(sub_cls, val, prefix=sub_prefix)
        else:
            kwargs[name] = val
    return cls(**kwargs)


def load_config(
    profile: str = "crawl_agar",
    overrides: dict[str, Any] | None = None,
    config_dir: Path | str | None = None,
) -> Config:
    """加载 default.yaml → 合并 profiles/<profile>.yaml → 合并 overrides → 环境变量覆盖。"""
    if config_dir is None:
        base_dir = Path(__file__).resolve().parents[1] / "config"
    else:
        base_dir = Path(config_dir)

    default_path = base_dir / "default.yaml"
    if not default_path.exists():
        raise FileNotFoundError(f"未找到默认配置文件: {default_path}")

    merged: dict[str, Any] = yaml.safe_load(default_path.read_text(encoding="utf-8")) or {}

    if profile:
        prof_path = base_dir / "profiles" / f"{profile}.yaml"
        if not prof_path.exists():
            raise ValueError(f"未知的 profile '{profile}'，文件不存在: {prof_path}")
        prof_data = yaml.safe_load(prof_path.read_text(encoding="utf-8")) or {}
        merged = _deep_merge(merged, prof_data)

    if overrides:
        merged = _deep_merge(merged, overrides)

    # 环境变量覆盖
    if os.environ.get("WORM_MODELS_DIR"):
        merged.setdefault("paths", {})["models_dir"] = os.environ["WORM_MODELS_DIR"]
    if os.environ.get("WORM_DATA_DIR"):
        merged.setdefault("paths", {})["data_dir"] = os.environ["WORM_DATA_DIR"]
    if os.environ.get("WORM_PORT"):
        merged.setdefault("server", {})["port"] = int(os.environ["WORM_PORT"])

    merged["profile"] = profile
    return _dict_to_dataclass(Config, merged)
