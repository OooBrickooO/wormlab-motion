# 01 核心：数据结构 · 配置 · 工具函数 · 检测层

对应任务：T05、T06。

## 1. 常量与枚举（`wormlab/schemas.py`）
```python
N_DENSE = 33          # 稠密中心线点数
N_UI    = 11          # 兼容 UI 的关键点数

class Quality(IntEnum):
    OK = 0; SUSPECT = 1; BAD = 2; MISSING = 3     # MISSING = 本帧没有姿态

class Behavior(str, Enum):
    STATIC="static"; CRAWL="crawl"; THRASH="thrash"; TURN="turn"; MOVING="moving"; INVALID="invalid"
```

## 2. 数据类（全部 `@dataclass`，数组 `float32`）
```python
@dataclass
class Detection:
    bbox: np.ndarray            # (4,) xyxy，帧坐标
    conf: float
    mask: np.ndarray | None     # 裁剪后的 bool 掩码，形状 (h, w)，范围 = mask_origin 起、bbox 外扩 margin
    mask_origin: tuple[int,int] # (x0, y0) 掩码左上角在整帧中的位置
    det_idx: int                # 本帧内序号

@dataclass
class Pose:
    dense: np.ndarray | None    # (33,2) 帧坐标，head=0；None 表示提取失败
    length: float               # 弧长(px, 已含端帽修正)
    chord: float                # 首尾直线距离(px)
    width: float                # 平均体宽(px)
    area: float                 # 掩码面积(px², 原分辨率)
    quality: Quality
    flags: frozenset[str]       # {"loop","branch","fragment","fallback","empty","merge","clipped","upscaled"}
    contour: np.ndarray | None  # 平滑外轮廓折线 (K,2) float32，供渲染；K<=80
    @property
    def pts11(self) -> np.ndarray | None: ...   # 按弧长在 dense 上等距重采样 11 点

@dataclass
class TrackOut:                 # tracker.update() 每帧每轨迹的输出
    track_id: int; bbox: np.ndarray; det_idx: int | None   # None=ghost(无检测)
    is_ghost: bool; lost_frames: int; confirmed: bool
    user_label: str | None; head_kpt_idx: int | None; head_xy: tuple[float,float] | None
    conf: float
    first_frame: int            # 该轨迹首次出现的帧号（用于"后进入"判定）
    uid: int                    # 轨迹实例唯一号（单调递增，永不复用）；track_id 是显示 ID，可能被 ID 池回收复用

@dataclass
class FrameRecord:              # Pass A 缓存里的一行（长格式）
    frame_idx: int; track_id: int; bbox: np.ndarray; conf: float; is_ghost: bool
    pose: Pose | None; centroid: tuple[float,float]
```
`Pose.pts11` 计算：`s=cumulative arc length of dense / total; np.interp` 到 `linspace(0,1,11)`，x、y 分别插值。

## 3. 配置（`config.py` + `config/*.yaml`）
- `load_config(profile: str = "crawl_agar", overrides: dict | None = None) -> Config`：读 `default.yaml` → 深合并 `profiles/<profile>.yaml` → 深合并 `overrides`。
- `Config` 是嵌套 dataclass（`paths, server, detection, pipeline, tracking, pose, kinematics`），每个叶子字段有类型；未知键 → `ValueError`。
- `cfg.with_fps(fps) -> RuntimeCfg`：返回带 `fps` 的只读对象，提供 `frames(sec)`、`sec(frames)`。
- 环境变量覆盖：`WORM_MODELS_DIR`、`WORM_DATA_DIR`、`WORM_PORT`。

### `config/default.yaml`
```yaml
paths:
  models_dir: "../backend/models"     # 相对 re_struction/backend/ ；切换时再迁移
  data_dir: "data"                    # uploads/ results/ cache/ 都在其下
server: {host: "0.0.0.0", port: 5001, cors_origins: ["http://localhost:3001", "http://localhost:3000"],
         max_upload_mb: 2048, allowed_ext: ["mp4","avi","mov"], image_ext: ["jpg","jpeg","png","bmp"]}
detection:
  default_model: "origin.pt"
  model_preference: ["origin.pt","OnlineVision.pt","CJLUdata_Old.pt","CJLUdata.pt","best.pt"]
  egg_model: "EggCount.pt"
  imgsz: 416                 # 用户决定：保持 416（匹配 origin.pt 训练尺寸）。也可写整数或 native 覆盖
  oom_fallback: true         # 显存不足时自动减半 batch；batch=1 仍失败才降 imgsz（并 log.warning）
  tile: {enabled: false, size: 1088, overlap: 0.2}
  min_confidence: 0.30
  min_area_ratio: 0.000015
  nms_iou: 0.3               # 旧 apply_nms 的 overlap_threshold（worm_processing.py:1349/1675），保持不变
  batch_size: 8
  mask_margin_ratio: 0.25    # 掩码裁剪外扩比例
counting:
  late_entry_grace_s: 1.0    # 视频开头这段时间内首次出现的轨迹视为"初始虫"（容忍首帧漏检）；之后才首次出现 = 后进入，一律不计数
  include_late_in_outputs: false   # 后进入的虫是否出现在汇总 JSON/CSV（默认不出现；实时画面仍以灰框标注 late）
  c_unit: cycle              # 默认 C 统计口径：cycle(左+右完整来回) | bend(单次弯曲)；上传时可覆盖
pipeline:
  emit_every: 2              # 每 N 帧推一次实时画面
  jpeg_quality: 80
  frame_queue: 4
  infer_queue: 6
  pose_workers: 4
  sub_video_size: 200
  image_mode: {fps: 12.0, repeat_frames: 36}   # 单张图片按 36 帧短视频处理（沿用旧行为）
  pose_every_n_frames: 1     # D10：swim 必须 1
tracking: {}                 # 见 04 §9，数值取自旧实现
pose: {}                     # 见 02 §9
kinematics: {}               # 见 03 §10
```
profile 文件只覆盖 `kinematics`（及 `pose.upscale_*` 如需要），见 03 §10。

## 4. 检测层（`wormlab/detection/`）

### `detector.py`
```python
class Detector:
    def __init__(self, model_path: str, cfg: Config, device: str | None = None):
        # 1. device = "cuda" if torch.cuda.is_available() else "cpu"；cpu 时 log.warning
        # 2. self.model = YOLO(model_path); self.model.to(device)
        # 3. 首次推理时确定 imgsz：cfg.imgsz=='native' → 取该视频首批帧 max(H,W) 上取整到 32 的倍数；之后固定（同一任务内不变）
        #    warm-up：用 1 张黑图按该 imgsz 跑一次
    def infer(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """批推理。每帧返回过滤后的 Detection 列表。"""
        # results = self.model(frames, verbose=False, device=self.device,
        #                      retina_masks=True, imgsz=self.imgsz, conf=cfg.min_confidence)
        # OOM 处理：捕获 torch.cuda.OutOfMemoryError → torch.cuda.empty_cache() → batch 对半拆分重试；batch=1 仍失败才降 imgsz 并告警
        # 若 cfg.tile.enabled：走 _infer_tiled（见下）
        # for each result r (对应 frame f, H,W = f.shape[:2]):
        #   boxes = r.boxes.xyxy.cpu().numpy(); confs = r.boxes.conf.cpu().numpy()
        #   masks_t = r.masks.data   # (n,H,W) GPU tensor（retina_masks=True 时为原分辨率）
        #   for i in range(n):
        #       bbox 外扩 margin = max(8, mask_margin_ratio*max(w,h)) 并 clip 到帧内 → (x0,y0,x1,y1)
        #       mask_crop = (masks_t[i, y0:y1, x0:x1] > 0.5).cpu().numpy()    # ★在 GPU 上裁剪后再 .cpu()，避免整帧搬运
        #   -> Detection(...)
        # 过滤: filters.filter_detections(dets, frame_wh, cfg)
    def _infer_tiled(self, frame): ...   # 见下
```
**切片推理（`tile.enabled`）**：把帧按 `size`×`size`、重叠 `overlap` 切块，逐块推理，坐标/掩码平移回整帧；跨块重复检测用 mask-IoU > 0.5 的 NMS 合并（保留置信度高者）。**仅在 T03 实验证明有益后才启用**，先实现接口与单测（对合成图）。

### `filters.py`
```python
def filter_detections(dets, frame_wh, cfg) -> list[Detection]:
    # 保留: bbox 面积 >= W*H*min_area_ratio 且 conf >= min_confidence
    # 注意: 旧实现用 bbox 面积而非掩码面积；保持不变
def apply_nms(dets, iou_thr) -> list[Detection]:
    # 与旧 apply_nms（worm_processing.py:1653-1677）逻辑一致；按置信度降序，bbox IoU > thr 则丢弃低者
```
单测：假 `Detection` 列表覆盖面积/置信度/NMS 边界。

## 5. 通用几何工具（`wormlab/geometry.py`，T05 一并实现）
```python
def arc_length(pts) -> np.ndarray            # (n,) 累积弧长，首项 0
def resample_polyline(pts, n) -> np.ndarray  # 按弧长等距重采样到 n 点（线性插值）
def tangent_angles(pts) -> np.ndarray        # (n-1,) atan2(dy,dx)，沿 s 做 np.unwrap
def wrap_pi(x) -> x                          # 折到 (-π, π]
def iou_xyxy(a, b) -> float
def polyline_length(pts) -> float
```
单测：直线、半圆（长度 πr 误差<1e-3）、已知切线角。
