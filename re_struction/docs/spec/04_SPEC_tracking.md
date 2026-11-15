# 04 跟踪层（迁移 + 拆模块，不改算法）

对应任务：T17–T22。**原则：行为 1:1 迁移**——旧实现已调好，本阶段只做拆分、去耦（去掉绘制/子视频/姿态/计数职责）、加单测。旧代码只读参照（`backend/worm_processing.py`，行号如下）。

## 1. 模块与旧代码对照
| 新模块 | 内容 | 旧代码位置 |
| :-- | :-- | :-- |
| `tracking/kalman.py` | `KalmanBoxFilter`（8 维状态、`predict/update/get_state/get_velocity/get_predicted_center`） | `KalmanBoxTracker` 1096-1285 的滤波与 bbox 部分 |
| `tracking/track.py` | `Track` 数据类：Kalman + 元数据（见 §3.2） | 同上的字段部分 |
| `tracking/features.py` | `extract_features(mask_crop, bbox_in_crop)` 11 维外观特征 | 1499-1535（改为接收裁剪掩码，见 §3.3） |
| `tracking/cost.py` | IoU/中心距离/方向/运动一致性/外观距离 | 1537-1596、1678-1724 |
| `tracking/associator.py` | 三阶段匹配 | `associate_detections_to_tracks` 1734-1911，`_match_gate` 1726-1732 |
| `tracking/id_pool.py` | ID 池 | `_alloc_id/_release_id/set_max_worm_id` 1355-1380 |
| `tracking/head_config.py` | 首帧用户头标记匹配与人工框植入 | `_match_head_config_to_tracks` 1389-1463 |
| `tracking/tracker.py` | 编排 `update()`、丢失/残留/回收 | `update_tracking` 2069-2247，`_handle_lost_tracks` 2018-2051，`_prune_lost_tracks` 2053-2067 |
| `detection/filters.py::apply_nms` | NMS | 1653-1676（阈值 `overlap_threshold=0.3`——**以旧值 0.3 为准**，覆盖 01 §3 里的占位 0.5） |

**旧代码里不迁移的**：绘制、子视频、`TrackHistory`/计数（已被 kinematics 取代）、`orient_pose_pair`（被 02 §5 取代）、`build_prediction_detections`（改为 `TrackOut(is_ghost=True)`）、`compute_keypoint_cost`（旧代码里并未参与代价矩阵，丢弃）。

## 2. ID 池（`id_pool.py`）——行为必须锁定
```python
class IdPool:
    def __init__(self, max_id: int | None): ...
    def alloc(self) -> int | None
    def release(self, tid: int) -> None
    def set_max(self, max_id: int | None, active_ids: set[int]) -> None
```
**单测用例表（全部要过）**
| 用例 | 期望 |
| :-- | :-- |
| `max_id=3`：连续 alloc 4 次 | 返回 1,2,3,`None` |
| `release(2)` 后 alloc | 返回 2（池按升序） |
| `release(7)`（不在 1..max）| 忽略，池不变 |
| 重复 `release(2)` | 池中只有一个 2 |
| `max_id=None`：alloc ×3 | 1,2,3 单调递增（`next_id` 起点 1） |
| `set_max(5, active={1,2})` | 池=[3,4,5]；`next_id=max(next_id,6)` |
| `set_max(None, …)` | 池清空，转无限制模式；已存在 ID 不回收 |
| 受限模式下 `next_id` | 起点 `max_id+1`，但受限模式永不使用（alloc 在池空时返回 None）|

## 3. Kalman / 特征 / 代价
### 3.1 `KalmanBoxFilter`
状态 `[cx,cy,w,h,vx,vy,vw,vh]`，观测 `[cx,cy,w,h]`，`dt=1`；`processNoiseCov=I8*0.05`，`measurementNoiseCov=I4*0.1`，`errorCovPost=I8*100`；初始速度 0。**可继续用 `cv2.KalmanFilter`** 以保证数值一致（单测对拍 1e-5）。`predict()` 返回 xyxy 并保存为 `last_predicted_bbox`。
### 3.2 `Track`
```python
@dataclass
class Track:
    track_id:int; kf:KalmanBoxFilter
    history: deque[(cx,cy,w,h)] (maxlen=30)
    feature: np.ndarray(11); feature_history: deque(maxlen=20)   # get_feature() = mean(feature_history)
    hits:int=1; lost_frames:int=0; time_since_update:int=0; confirmed:bool=False
    last_predicted_bbox: np.ndarray
    motion_vector: np.ndarray(2)          # 见下
    last_pose_dense: np.ndarray|None      # 由 Pass A 回填（姿态层产出），仅用于 motion_vector 与 head 对齐
    user_label:str|None=None; head_kpt_idx:int|None=None; head_xy:tuple|None=None
    uid:int                               # Tracker 内部单调计数，创建时分配，永不复用（ID 池回收复用 track_id 时用来区分实例）
    first_frame:int                       # 创建该轨迹时的 frame_idx
```
`motion_vector` 更新：`tracker.feed_pose(track_id, dense)`：若有上一帧 `last_pose_dense` → `mv = 0.7*mv + 0.3*mean(dense - last_pose_dense, axis=0)`；否则用 `history` 末两帧中心差（与旧 `update_keypoints` 1222-1238 等价，11 点换成 33 点，均值不变）。`get_motion_vector()` 同旧 1240-1250。
### 3.3 外观特征
旧实现接收整帧掩码；新实现接收 `Detection.mask`（裁剪）和 **相对裁剪原点的 bbox**：`bbox_in_crop = bbox - [x0,y0,x0,y0]`，其余公式逐行照旧（log 面积/周长、圆度、`w/h`、7 个 Hu 矩 `sign*log1p|hu|`；面积<50 返回零向量）。
### 3.4 代价
`cost.py` 提供纯函数：`iou(a,b)`、`center_dist(a,b)`、`direction_cost(track, det_center)`、`motion_cost(track, det_bbox, det_size, max_speed)`、`appearance_dist(f1,f2)`。公式照旧。

## 4. 三阶段关联（`associator.py`）
```python
def associate(tracks, dets, cfg, frame_wh) -> (matches, unmatched_dets, unmatched_tracks, det_features):
    # 完全照搬旧 1734-1911，包括：
    #  - 每个 track.predict() 恰好调用一次（在关联里；无检测时在 tracker.update 里调用一次）
    #  - gate: min(max_speed*(1+min(lost,90)/20), max(0.18*max(W,H), max_speed*1.5))
    #  - 阶段1 权重: iou .25 / motion .45 / dir .15 / app .15 + conf_penalty (1-conf)*0.1；
    #            lost_frames==0 且 iou<0.5 额外 +0.5；匈牙利，cost<2.0 接受；超 gate 置 1e6
    #  - 阶段2 (iou>=0.3): 0.3*(1-iou)+0.4*motion+0.3*app < 1.2
    #  - 阶段3 re-ID (lost>=3，按 lost 降序): 尺寸比<=2.5, app<=8, cost=.55*min(d/gate,1)+.25*min(|ratio-1|,1)+.20*min(app/5,1) < .85
```
所有数值（`max_speed=50, confirm_hits=5, iou_match_thresh=0.5, iou_secondary=0.3, cost_threshold=2.0, GHOST_REID_MAX_LOST=90, overlap_threshold=0.3`）进配置（§9）。`reid.py` 放阶段 3 的独立函数，便于单测。

## 5. 首帧头标记（`head_config.py`）
```python
def apply_head_config(tracker, head_config) -> None:
    # 旧逻辑 1389-1463：
    # (1) 对每个 cfg，用 centroid（cfg['centroid'] 或 centroid_x/y；前端 /upload 传的是 bbox+keypoints，centroid 缺省时取 bbox 中心）
    #     找最近且尚无 user_label 的 track，距离 < 120px 则设置 user_label/confirmed=True/head_kpt_idx
    # (2) 未匹配且有 bbox 的 cfg（手动框选）：alloc ID（池空则抢占一个"未确认且无标签"的 track 的 ID），植入 Kalman 轨迹，confirmed=True
    # ★ 与旧版的唯一差别：不再靠"head_kpt_idx==10 就翻转"，而是记录 head_xy = cfg.keypoints[head_kpt_idx]（若有），
    #   由 PoseExtractor 首帧用 orient(user_head_xy=head_xy) 对齐（02 §5），对任意 head_kpt_idx 都成立。
    # 仅在首次出现新 track 且 head_config 非空时执行一次（_head_config_applied）
```
单测：3 个 cfg（2 个对应检测、1 个手动框）→ 2 个标签匹配、1 个植入；池满时抢占规则；距离>120 不匹配。

## 6. 编排（`tracker.py`）
```python
class Tracker:
    def __init__(self, cfg, fps, frame_wh, max_worm_id=None, head_config=None): ...
    def update(self, dets: list[Detection], frame_idx: int) -> list[TrackOut]:
        # 对应旧 update_tracking 2069-2247：
        # 1) 无检测：所有 track.predict() 一次；_handle_lost；_prune；返回仍在残留窗口内的 ghost TrackOut
        # 2) NMS(overlap 0.3) → associate → 对 matches 调 track.update(bbox, feat)，hits>=confirm_hits 则 confirmed；清除 ghost
        # 3) 未匹配检测按 conf 降序 alloc ID（池空 → 跳过，严格受限）→ 新 Track(confirmed=False)
        # 4) 新轨迹出现且 head_config 未应用 → apply_head_config
        # 5) 未匹配轨迹 _handle_lost（lost_frames+=1；<=ghost_overlay_frames 视为 ghost 并继续输出预测框）
        # 6) _prune: confirmed 轨迹 lost>max_lost_frames 才回收；未确认 lost>min(3,confirm_hits) 即回收并 release(ID)
        # 7) 返回 TrackOut：匹配/新建的用 Kalman 平滑后的 bbox，ghost 用 last_predicted_bbox；过滤 track_id<=0
    def feed_pose(self, track_id, dense): ...      # §3.2
    def set_max_worm_id(self, n|None): ...         # 委托 IdPool.set_max
    @property active_ids / ghost_ids
```
- `persistence_frames=round(60*fps)`、`ghost_overlay_frames=round(3*fps)`、`max_lost_frames=max(int(fps*15), persistence_frames)`（旧 1318-1320）。
- `TrackOut` 带上 `uid` 与 `first_frame`；人工植入的轨迹（head_config 手动框）`first_frame=0`。
- **Tracker 不做姿态、不写视频、不画图。** 姿态由 Pass A 在 `update()` 之后按 `TrackOut.det_idx` 取掩码调用 `PoseExtractor`。

## 7. 回放对拍（T21）——保证"没改坏"
1. `tools/dump_detections.py`：对基准视频（`10×`、`1.6X1`、`Move-3`，各前 300 帧）跑一次 YOLO，保存每帧检测（bbox、conf、**掩码裁剪+origin**）到 `data/cache/dets_<video>.npz`。
2. `tests/integration/test_tracker_replay.py`：同一份检测流分别喂旧 `EnhancedWormTracker`（只读导入，需把裁剪掩码还原成整帧掩码喂旧接口）与新 `Tracker`，逐帧比较 `bbox → track_id` 分配。**一致率 ≥ 99%**；不一致处允许记录但需解释（浮点/NMS 排序）。数据缓存不存在时测试 `skip`（标记 `slow`）。

## 8. 离线 tracklet 缝合（T22，可选，默认关）
```python
def stitch(rows_by_track, cfg) -> dict[int,int]:     # 返回 id 重映射 {旧id: 新id}
    # 仅处理"ID 交换"：找出两条轨迹在某帧 bbox 距离 < 0.5*max(L)（交叉/接触）的时间点 t_c，
    # 比较 [t_c+Δ, t_c+2Δ] 窗口内 "保持" 与 "交换" 两种连接的代价（位置外推残差 + 体长差 + 外观特征差），
    # 若 cost_swap < 0.6 * cost_keep → 在 t_c 之后交换两个 ID 的后续片段。
    # 必须遵守：ID 池上限、user_label/头标记所在轨迹的身份不可被替换（带标签者为锚）。
```
`cfg.tracking.stitch.enabled=false`。启用与否以 `tools/eval_benchmark.py` 中"ID 切换数/计数误差"是否改善为准（07 §6）。

## 9. 配置块（`tracking:`）
```yaml
tracking:
  confirm_hits: 5
  iou_match_thresh: 0.5
  iou_secondary: 0.3
  cost_threshold: 2.0
  max_speed: 50
  weights: {iou: 0.25, motion: 0.45, direction: 0.15, appearance: 0.15}
  overlap_threshold: 0.3
  persistence_sec: 60.0
  ghost_overlay_sec: 3.0
  ghost_reid_max_lost: 90
  max_lost_sec_legacy: 15
  prediction_confidence: 0.18
  head_match_radius_px: 120
  stitch: {enabled: false}
```
