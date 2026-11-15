# 07 测试 · 合成数据 · 基准评测

对应任务：T02、T03、T04、T09、T16、T32。

## 1. 合成数据生成器（`tools/make_synthetic.py`，T09）
目的：**没有人工标注也能验证数学**；每个场景自带真值。同时提供给 pose、kinematics 的单测。
```python
@dataclass
class Scn:
    kind: str; duration_s: float = 20; fps: float = 30; length_px: float = 120; width_px: float = 7
    A: float = 0.8            # 行波切线角幅度(rad)
    f: float = 0.5            # 波频率(Hz)
    k: float = 1.0            # 每体长波数
    disp_per_cycle: float = 0.4   # 每个波周期前进的体长数（0 = 原地）
    jitter_px: float = 0.0    # 对中心线加的高斯抖动
    heading: float = 0.0      # 运动方向(rad)
    ...
def make_series(scn: Scn) -> tuple[np.ndarray, dict]:
    """返回 centerlines (T,33,2)（head=0，帧坐标）和 expected dict（s_count, direction, swing_count, turn_count ...）"""
def render_mask(dense33: np.ndarray, width_px: float, pad: int = 12) -> tuple[np.ndarray, tuple[int,int]]:
    """cv2.polylines 粗线 + 两端圆帽，返回 (bool 掩码, origin)。可选 mask_noise: 轮廓随机 1px 腐蚀/膨胀。"""
```
**构造方法**：给定切线角场 `θ(s,t)`（`s∈[0,1]` 自头向尾，**沿 head→tail 积分**）：`P(s)=P_head + L∫₀ˢ[cosθ, sinθ]ds'`；整体平移使质心沿"前进方向"匀速移动。前进方向 = 平均切线角 + π（因为切线是头→尾，前进是朝头）。
- 前进爬行：`θ = θ0 + A sin(2πk s − 2πf t)`，质心沿前进方向速度 `v = disp_per_cycle·L·f`（px/s）。
- 后退爬行：`θ = θ0 + A sin(2πk s + 2πf t)`，质心沿**后退**方向（−前进）运动。
- `θ0 = heading + π`（使头在前进方向一侧）。

**场景表（`tests/synthetic/` 与 03 §4 一一对应）**
| kind | 构造 | expected |
| :-- | :-- | :-- |
| `crawl_fwd` | 如上，f=0.5，20 s，disp 0.4 | s=10(±1)，dir=+1 |
| `crawl_bwd` | 波反向 + 后退 | s=10(±1)，dir=−1 |
| `treadmill` | disp=0 | s=0 |
| `wrong_dir` | 波向前、质心后退 | s=0 |
| `head_wiggle` | 仅 `s<0.25` 有 `0.6·sin(2π·1.5t)·sin(πs/0.25)`，身体前进 | s=0 |
| `wobble` | 相位 `0.4·2π·sin(2π·0.2t)` 来回 | s=0 |
| `fwd_then_back` | 前进波 3.5 周期（前进）→ 反向波 3.5 周期（后退） | s=6（前 3 + 后 3） |
| `standing` | `θ=A sin(2πs)·cos(2πft)` | s=0 |
| `static_noise` | 直线 + jitter 0.5px | s=0, c=0 |
| `swim_thrash` | `θ=θ0 + κ(t)(s−0.5)`，`κ=Ac·sin(2π f t)`，Ac 使 `|c|` 峰值≈1.5 rad，f=2 Hz，10 s | swing_full=20(±1)，swing_half=40(±1) |
| `swim_small_amp` | 同上 `|c|` 峰值 0.5 | swing=0 |
| `omega` | 曲率因子使 `R: 0.9→0.4→0.9`，周期 4 s，12 s | turn=3 |
| `merge` | 两条合成虫的掩码在若干帧相交成一团 | pose 质量在这些帧为 BAD |
所有序列另提供 `fps=10` 与 `fps=30` 两个版本（检验秒→帧换算）。

## 2. 单测目录与最低覆盖
```
tests/unit/{test_config.py, test_geometry.py}
tests/unit/pose/{test_centerline.py, test_orientation.py, test_quality.py}
tests/unit/kinematics/{test_features.py, test_wave.py, test_s_runs.py, test_swing.py, test_turns.py, test_engine_parity.py}
tests/unit/tracking/{test_id_pool.py, test_kalman.py, test_cost.py, test_associator.py, test_head_config.py}
tests/unit/{test_cache_roundtrip.py, test_compat_payload.py, test_exporters.py}
tests/synthetic/{test_pose_synth.py, test_kin_synth.py}
tests/integration/{test_tracker_replay.py (slow), test_contract.py, test_pass_ab_smoke.py (slow)}
```
`pytest -q -m "not slow"` 必须快（< 60 s）；`slow` 标记的需要视频/GPU。

## 3. 编号 → 虫的映射（`tools/map_labels.py`，T02）
```
输入：benchmarks/ground_truth.yaml 中某视频条目
1) 读编号图；去除红色数字：mask = (R>180 & G<90 & B<90) 膨胀 3px → cv2.inpaint(TELEA)
2) 对 视频第 0~30 帧（每 5 帧）与"去字后编号图"做 ORB(2000) + BFMatcher + RANSAC(cv2.estimateAffinePartial2D)，取内点最多的帧与变换 M
   （编号图可能是截图/缩放/裁剪；若内点 < 30 则报错提示人工指定）
3) labels_xy 经 M 映射到视频坐标；用 Detector 在该帧检测；每个标号找最近的检测（点到掩码最小距离，≤ 120px），匈牙利保证一对一
4) 输出：benchmarks/labels/<id>.json = {frame_idx, M, assignments:[{label, gt, det_bbox, centroid}], unmatched_labels:[...]}
         benchmarks/labels/<id>.overlay.jpg（在帧上画 bbox + 标号 + gt，供人工目检）
5) 若存在 benchmarks/labels/<id>.manual.json（{label: [x1,y1,x2,y2]}），以其为准覆盖自动匹配
```
Swim 视频无编号图：输出第 0 帧、第 N 帧的检测编号图（`det_idx` 标在图上），请用户填 `manual.json`。
**评测时的"标号↔轨迹"**：取 Pass A 缓存中第 `frame_idx` 帧 bbox 与 `det_bbox` IoU 最大的 track_id。

## 4. D12 推理分辨率实验（`tools/bench_infer.py`，T03）
```
对象：10×、1.6X1、Swim-3、Swim-4（各取帧 0、150、300）
模型：origin.pt、CJLUdata_Old.pt
配置：imgsz ∈ {416(默认), 640, 1280, native(=1920)} × tile ∈ {off, on(size=1088, overlap=0.2)}。**默认已定为 416（用户最终决定：保持 416 匹配 `origin.pt` 训练尺寸）**，本实验用于验证与留档。
指标（每组合）：
  - 已标注虫的召回（用 labels/*.json 的 det_bbox 做 IoU>0.3 匹配；Swim 用人工 manual 或"检测到的最大/最小虫"替代并注明）
  - 平均置信度、平均掩码面积、与 imgsz=1920 掩码的 IoU（作稳定性参照）
  - （若 wormlab.pose 已存在）姿态有效率 & 连续 30 帧体长的变异系数
  - 每帧推理耗时（GPU，batch=8）
输出：benchmarks/report_d12.md（表格）+ benchmarks/d12/*.jpg（并排裁剪图）
记录：native 的 GPU 显存峰值与每帧耗时（batch=8/4/2/1 哪个能跑）、OOM 回退是否触发；若 native + batch=8 不可行，确定可行的最大 batch 写入 `pipeline`/`detection.batch_size`。把结果写入 DECISIONS.md。
```

## 5. 基线与评测
### 5.1 旧系统基线（`tools/dump_legacy_baseline.py`，T04）
只读 `sys.path.insert(0, "backend")` 导入 `worm_processing`。对 ground_truth 里每个视频：
`init_model(origin.pt)` → `EnhancedWormTracker(fps, (w,h), sub_video_dir=None, max_worm_id=<第0帧检测数>, head_config=[])` → 按 app.py 的批处理方式（batch=8，`process_track_single`）跑完 → 读 `tracker.cycle_counts`。用 `tracker.track_history[tid][0]`（首个质心）与 labels 的 centroid 最近邻得到"标号→旧 tid"，输出 `benchmarks/baseline_legacy.json`：`{video_id: {label: {c_count, s_count, motion_type}}}`。**不要改旧代码；若旧代码需 CUDA 才够快，用新 venv 运行即可。**

### 5.2 新系统评测（`tools/eval_benchmark.py`，T16；真实视频部分需 T24 的缓存，之前可先对合成缓存自测）
```
python tools/eval_benchmark.py [--videos 10x,1.6x-1,...] [--profile auto] [--cache-dir data/cache] [--rerun-pass-a]
流程：对每个视频：缓存存在则读，否则调用 PassA → 对每条标注虫的 track 用 KinematicsEngine.run → 对比 gt
输出：benchmarks/report.md，含
  - 每视频每虫：gt / 新 / 旧(baseline) / 误差 / 是否通过
  - 汇总：通过率、MAE、零真值虫的最大/平均输出、旧系统同指标；Swim 同时给 `c_cycles`/`c_bends`；列出被判 late 的轨迹数
  - 标号↔轨迹稳定性（该 track_id 覆盖其出现帧的比例）
  - 质量统计：BAD/SUSPECT 帧占比
  - 拒绝周期统计 rejected.no_disp / wrong_dir（用于诊断）
```

## 6. 验收线（P2/P6 使用；数值可在你确认后调整）
**单虫通过判据**
- 爬行 S：`gt ≤ 5 → |pred−gt| ≤ 1`；`gt > 5 → |pred−gt| ≤ max(2, 0.2·gt)`。
- 游泳 C：`|pred−gt| ≤ 0.15·gt`。**`c_cycles` 与 `c_bends` 两种口径都要在报告里列出**，通过判据按"与真值更接近的那一种"算，并注明是哪一种（由甲方最终选定口径）。
- 只评测**初始虫**（`counted=True`）；后进入的虫不参与，且其 `late_entry` 轨迹必须输出 0。
**视频/总体**
1. 爬行视频标注虫：**通过率 ≥ 80%**；**所有 gt=0 的虫输出 ≤ 1，且其均值 ≤ 0.5**（1.6x-1 共 6 条）。
2. 游泳视频：标注虫通过率 ≥ 75%。
3. 与旧系统基线比较：新系统总 MAE 严格小于旧系统，且 gt=0 虫的误报总数不高于旧系统。
4. 姿态：爬行标注虫 `BAD` 帧 < 5%；游泳 < 15%。
5. 跟踪：标注虫同一 track_id 覆盖其可见帧 ≥ 90%。
6. 吞吐（不阻塞，仅报告）：Pass A 在选定 imgsz 下的 fps。
7. 工程：`pytest -q -m "not slow"` 全绿；契约测试全绿；旧前端指向新后端可完成一次完整流程。

## 7. 调参工具（`tools/tune.py`，T16 之后）
在**已缓存姿态**上网格搜索：爬行 `r_min∈{0.15,0.2,0.25,0.3,0.4}`、`eps_min∈{0.3,0.4,0.5}`、`min_disp_ratio∈{0.05,0.08,0.1,0.15}`、`post_ratio_min∈{0.25,0.35,0.5}`；游泳 `bend_high∈{0.5,0.7,0.9,1.1}`、`min_flip_interval_s∈{0.05,0.08,0.12}`。
- 目标：最小化 `Σ|pred−gt|/max(1,gt)`，约束：gt=0 的虫不得 >1。
- **留一视频交叉验证**并报告（避免过拟合 6 个视频）；只输出到 `benchmarks/tuned_<profile>.yaml`，**不自动改 default.yaml**，由人确认。
