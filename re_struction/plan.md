# 08_SPEC_accuracy_overhaul.md — C/S 运动精准计数与掩码姿态系统重构规范

> **目标读者**：Harness 端编码代理（AI Coding Agent）  
> **作用范围**：仅修改 `re_struction/` 目录下的代码、配置与测试；**不要修改**旧主分支 `backend/worm_processing.py`（仅作为对照参考）。  
> **核心目标**：
> 1. **攻克 C 形运动（`swim_liquid` 游泳摆动 & `crawl_agar` 深弯/C 弯）**：突破 YOLOv8-seg `104×104` 掩码分辨率天花板，引入原图 ROI 锐利前景与免骨架椭圆/不变矩双轨计数，修复 `SwingCounter` 旋转反噬与 `TurnCounter` 几何阈值乌龙。
> 2. **校准 S 形运动（`crawl_agar` 正弦爬行）**：消除当前重构版“偏保守/漏计数”的 7 个叠加漏斗损耗，同时保持对驻波、头部试探、倒车、原地打滑的严格拦截（绝不退回旧版过激进状态）。
> 3. **打通评测与调参闭环**：修复 `tools/tune.py` 与 `KinematicsEngine` 逻辑脱节问题，为 `swim_liquid` 补齐离线调参能力。

---

## 0. 硬性不变量与执行纪律（Agent Guardrails）

在改动任何文件前，必须严格遵守以下架构不变量：

1. **配置严格校验 (`wormlab/config.py`)**
   - `_dict_to_dataclass` 开启了**严格未知键检查**。任何新增到 `default.yaml`、`profiles/crawl_agar.yaml`、`profiles/swim_liquid.yaml` 的配置键，**必须**同步在 `wormlab/config.py` 对应的 `@dataclass(frozen=True)` 中声明字段与默认值；若新增嵌套 dataclass，必须同时注册到 `_dict_to_dataclass` 的 `type_map` 中。
   - `default.yaml` 必须保持向后兼容，不得破坏已有单测的默认假设。
2. **缓存向后兼容 (`wormlab/pipeline/cache.py` & `wormlab/schemas.py`)**
   - `tracks.npz` 严格 `allow_pickle=False`。
   - 给 `Pose` 新增形状描述子字段时**必须提供默认值**（如 `0.0` / `float("nan")`），确保所有既有 `Pose(dense=..., length=..., ...)` 构造调用不报错。
   - `CacheReader` 读取新增数组时**必须做键存在性判断**（例如 `data["ell_aspect"] if "ell_aspect" in data else ...`），保证能够读取旧版本已生成的 `data/cache/*/tracks.npz`。
3. **流式/批量一致性不变量 (`tests/unit/kinematics/test_engine_parity.py`)**
   - `KinematicsEngine.push()` + `finalize()` 必须与 `KinematicsEngine.run()` 保持**逐位完全一致**。
   - `KinematicsEngine.snapshot()` 返回的计数（`s_count`、`c_count`、`c_cycles`、`c_bends`、`c_turn_count`）必须**单调不减**。
   - 任何需要整条轨迹全局信息的离线增强（如全轨迹自适应 $k^*$、SVD Eigenworms、全局头尾定向、免骨架离线频谱/峰谷复核），必须放在 `KinematicsEngine.run_postprocessed()` 或 `PassB` 中，不得破坏 `push()` 的因果延迟语义。
4. **外部契约不破坏 (`wormlab/compat/legacy_payload.py` & `tests/integration/test_contract.py`)**
   - 不得删除或重命名已有的 JSON / CSV / Socket.IO 导出字段（`s_count`, `c_count`, `c_cycles`, `c_bends`, `c_turn_count`, `c_unit`, `motion_type` 等）。
5. **既有函数签名兼容**
   - `wave_gate(window_feats, cfg, expected_len=None)`、`SRunCounter.step(i, gate_ok, phi, center, u, L_ref)`、`TurnCounter.step(i, R, valid, c_val=None)`、`SwingCounter.step(i, c, valid, u_curr=None)` 新增参数**全部使用带默认值的可选参数**，确保现有单测（`test_wave.py`, `test_s_runs.py`, `test_swing.py`, `test_turns.py`）零破坏。
   - 注意 `test_wave.py` 显式断言 `with pytest.raises(NotImplementedError): eigenworm_basis()` —— 若实现轨迹内 PCA/SVD 特征虫基，请命名为新函数 `fit_track_eigenworms(...)`（或明确同步更新 `test_wave.py`）。

---

## 1. 根因定位总表（问题 → 精确文件与符号）

| 编号 | 问题现象 | 精确位置（文件 & 符号） | 数学 / 物理 / 代码根因 |
| :--- | :--- | :--- | :--- |
| **M1** | 小虫/游泳 C 形掩码糊成椭圆，`Swim-4` 全 0 | `detection/detector.py`, `pose/mask_refine.py`, `pipeline/pass_a.py` | `origin.pt` 在 `imgsz=416` 下 `mask_ratio=4` → 真实掩码原型仅 **`104×104`**（`640` 切片下 1 格 = `6.15 px`）。`retina_masks=True` 仅双线性放大，小虫弯成 C 形时内凹被填平变实心椭圆，触发 `quality.py` `length < min_len_over_width * width` 判 `BAD`。同时 `mask_refine.enabled=false` 且现实现用 `np.logical_and(m, local_bin)` 无法还原被糊平的边界。 |
| **C1** | 琼脂 C 弯几乎不触发 `TurnCounter` | `kinematics/turns.py`, `profiles/crawl_agar.yaml` | 均匀圆弧弦长比 $R = \frac{\sin(\Delta\theta/2)}{\Delta\theta/2}$：**完美 $180^\circ$ 半圆 C 形 $R = 2/\pi \approx 0.637$**，但 `R_enter=0.58/0.55` 要求弯曲 $\ge 196^\circ\text{–}204^\circ$（已是 $\Omega$ 环）。 |
| **C2** | 游泳虫旋转或深 C 弯时 `c(t)` 符号被反锁 | `kinematics/swing.py:_fix_head_tail_flip` | 用缓慢 EMA `u_ref = 0.85*u_ref + 0.15*u_unit` 与当前首尾弦 `u_curr` 点积 `< -0.15` 判翻转。深 C 时弦长坍缩噪声大；真实旋转 $>98^\circ$ 时 EMA 滞后导致误触发并反向锁死 `u_ref`。 |
| **C3** | C 形波峰被坐标平滑/插值拉直 | `kinematics/counter.py:_interp_dense_at, _smoothed_dense` | 在笛卡尔 `(x, y)` 空间对缺失帧线性插值、对高速游泳帧做坐标均值，几何上直接将高曲率弯弧拉直，削平 `|c(t)|` 波峰；且 `SwingCounter` 错用 `cfg.s_run.gap_hold_s`。 |
| **C4** | 低像素小虫强抽骨架必然崩溃 | `pose/centerline.py`, `kinematics/swing.py` | 虫长 `< 55 px` 或低倍液滴中骨架拓扑高度不稳定；文献（wrMTrck / WF-NTP / Imaginera）证明此时必须切**免骨架椭圆长短轴比 + 3 阶偏度/图像不变矩**。 |
| **S1** | 琼脂爬行空间波长错配 | `profiles/crawl_agar.yaml`, `kinematics/counter.py:run_postprocessed` | *C. elegans* 琼脂爬行真实波长 $\lambda \approx 0.65\,L$（$k_{\text{body}} \approx 1.45\text{–}1.60$），当前配置为 `1.0 / 1.15`；且 `adaptive_k_body` 被 `base_score < 0.30` 锁死几乎不触发，导致相平面压扁成椭圆、圆度 $\varepsilon$ 偏低。 |
| **S2** | 固定长窗 PCA 圆度 `eps` 掐头去尾且误杀慢爬 | `kinematics/wave.py:wave_gate` | 理论推导：窗内扫过 $60^\circ$ 弧 $\varepsilon=0.136$、$90^\circ$ 弧 $\varepsilon=0.206$、$180^\circ$ 弧 $\varepsilon=0.435$。`win_s=1.8s` 对慢爬（0.15–0.22 Hz）及每段起止 0.5–0.8s 边缘都会因圆度不足关门，每段白丢 0.3–0.7 周期。 |
| **S3** | `freeze()` 恢复时静默丢弃相位进度 | `kinematics/s_runs.py:114-120` | 门限或 `BAD` 帧哪怕短暂闪烁 1～5 帧触发 `freeze()`，恢复时直接 `self.phi_prev = float(phi)` 抛弃冻结期间真实推进的相位，多次闪烁累计吞掉 0.3～0.5 周期。 |
| **S4** | 弧线爬行时直线弦位移误杀 | `kinematics/s_runs.py:139-160` | `dot(c_arr - anchor_center, axis_cur)` 用首尾直线弦投影；虫体沿弧线转弯爬行时直线弦远小于真实推进弧长，误落入 `no_disp` / `wrong_dir`。 |
| **S5** | 深弯打断前未结算近满周期 | `kinematics/counter.py:244` | 爬行 0.85 周期后转入深弯直接 `force_abort("curl_or_bad")`，未先调用 `settle_partial`。 |
| **S6** | 卷曲段头尾纠偏逻辑过粗 | `pose/orientation.py:reorient_track_poses_coil_aware` | 仅在 `R < 0.45` 切段（漏掉 $R \in [0.45, 0.65]$ 浅卷翻头）；非首段无条件 `do_flip = score < 0.0` 会把**真实后退爬行**强翻成假前进。 |
| **S7** | 调参工具与真实引擎脱节 + 质量参考漂移 | `tools/tune.py:_eval_fast`, `pose/quality.py:judge` | `tune.py` 未复现 `s_run.k_body`、`freeze`、`slowdown_hold_s`、`settle_partial`、`deep_coil_R`；`quality.py` 仅在 `OK` 帧更新 `area_hist`，一旦焦点/亮度轻微变化导致连续非 `OK`，参考基准永久锁死。 |

---

## 2. Workstream A：高清原图 ROI 前景精修 & 形状描述子缓存

### 2.1 目标
1. 复用 `PassA` 里已经算好的静止中值背景 `bg_gray`（`tracking/recovery.py:compute_static_median_gray`），在 **1080p 原图 ROI** 内重建像素级锐利虫体前景掩码，绕过 `104×104` YOLO 糊掩码。
2. 在 `extract_pose()` 内零额外开销计算**免骨架形状描述子**（等效椭圆长短轴比、偏心率、主轴角、沿短轴 3 阶偏度、ROI 运动能量），写入 `Pose` 与 `tracks.npz`，供实时 Pass A 与离线 Pass B 的游泳 C 计数使用。
3. 修复 `PoseQualityJudge` 的参考面积锁死问题。

### 2.2 涉及文件与数据结构改动

#### (1) `wormlab/config.py` — 扩展 `MaskRefineCfg`
```python
@dataclass(frozen=True)
class MaskRefineCfg:
    enabled: bool = False
    use_bg_sub: bool = True          # 优先使用原图静态中值背景差分
    prior_dilate_px: int = 8         # YOLO 掩码向外膨胀半径（作为空间先验，不再做严格交集）
    bg_mad_k: float = 3.5            # 背景差分自适应阈值系数: median + k * MAD
    bg_min_diff: int = 8             # 最小灰度差下限，抑制平坦背景噪声
    min_iou_fallback: float = 0.25   # 允许比旧版 0.70 更宽松（因旧糊掩码本身偏胖/缺凹槽）
    area_ratio_lo: float = 0.35      # 精修后允许变瘦（去掉糊膨胀外圈），但不允许空洞碎裂
    area_ratio_hi: float = 1.60
```

#### (2) `wormlab/schemas.py` — 扩展 `Pose`（全部带默认值，保持兼容）
```python
@dataclass
class Pose:
    # ... 保留所有既有字段不变 ...
    ell_aspect: float = float("nan")      # 等效椭圆短长轴比 b / a ∈ (0, 1]，直虫小、C弯大
    ell_ecc: float = float("nan")         # 偏心率 sqrt(1 - (b/a)^2) ∈ [0, 1)
    ell_angle: float = float("nan")       # 主轴方向角 (rad) ∈ [-π/2, π/2)
    ell_skew_minor: float = float("nan")  # 沿椭圆短轴的归一化 3 阶中心矩偏度（无需头尾即可指示 C 弯凸侧！）
    roi_motion: float = float("nan")      # ROI 帧间像素活跃度（区分“真 C 摆动”与“瘫痪僵卷”）
```

#### (3) `wormlab/pipeline/cache.py` — `CacheWriter` & `CacheReader`
- `CacheWriter` 新增 5 个 `float32` 列表并落盘到 `tracks.npz`：`ell_aspect`, `ell_ecc`, `ell_angle`, `ell_skew_minor`, `roi_motion`。
- `CacheReader` 读取时若旧版 `.npz` 不含上述键，自动填充 `np.full(m, np.nan, dtype=np.float32)`。

#### (4) `wormlab/pipeline/pass_a.py` & `wormlab/pose/extractor.py`
- `PassA.run()`：只要 `(rec_cfg.enabled and rec_cfg.bg_sub_enabled) or (self.cfg.pose.mask_refine.enabled and self.cfg.pose.mask_refine.use_bg_sub)` 为真，且非 `ImageSource`，即计算 `bg_gray`。
- 在 `tracker_loop()` 内保留 `prev_frame_gray`（上一帧灰度图），当 `self.cfg.pose.mask_refine.enabled` 或 `self.cfg.kinematics.swing.enabled` 时计算当前帧 `frame_gray`。
- `PoseExtractor.process(det, frame_gray=None, bg_gray=None, prev_frame_gray=None, frame_idx=None)`：
  - 将 `gray_crop`、`bg_crop`、`prev_gray_crop` 裁切（按 `det.mask_origin` 向外扩边 `prior_dilate_px` 并同步调整 `mask_origin`！）传入 `refine_mask(...)`。
  - 计算 `roi_motion = mean(|gray_crop - prev_gray_crop|[dilated_prior])` 并附加到返回的 `Pose` 上。

### 2.3 核心伪代码

#### `pose/mask_refine.py`（替换旧版 `np.logical_and(m, local_bin)`）
```python
def refine_mask(
    gray_crop: np.ndarray | None,
    mask_crop: np.ndarray,
    cfg: PoseCfg,
    bg_crop: np.ndarray | None = None,
) -> np.ndarray:
    m = np.asarray(mask_crop, dtype=bool)
    if not cfg.mask_refine.enabled or gray_crop is None:
        return m
    orig_area = int(m.sum())
    if orig_area < cfg.min_mask_px:
        return m

    r = int(cfg.mask_refine.prior_dilate_px)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    prior = cv2.dilate(m.astype(np.uint8) * 255, kernel, iterations=1) > 0

    cand = None
    # 路径 1（首选）: 静态中值背景差分 —— 对细小 C 形虫最锐利，且不依赖 YOLO 糊轮廓边界
    if cfg.mask_refine.use_bg_sub and bg_crop is not None and bg_crop.shape == gray_crop.shape:
        diff = cv2.absdiff(gray_crop, bg_crop).astype(np.float32)
        ring = prior & (~m)
        bg_vals = diff[ring] if np.count_nonzero(ring) >= 16 else diff[prior]
        med = float(np.median(bg_vals))
        mad = float(np.median(np.abs(bg_vals - med))) * 1.4826
        thr = max(float(cfg.mask_refine.bg_min_diff), med + float(cfg.mask_refine.bg_mad_k) * mad)
        fg = (diff >= thr) & prior
        cand = keep_largest_cc_overlapping(fg, seed_mask=m)

    # 路径 2（回退）: 无背景图时，用局部对比度 + prior 约束（不再与糊掩码 m 强求交！）
    if cand is None or int(cand.sum()) < cfg.min_mask_px:
        # 使用局部背景 ring 估计虫体极性（暗虫/亮虫）与自适应阈值
        ring = prior & (~m)
        if np.count_nonzero(ring) >= 16:
            bg_med = float(np.median(gray_crop[ring]))
            bg_mad = float(np.median(np.abs(gray_crop[ring].astype(np.float32) - bg_med))) * 1.4826
            in_med = float(np.median(gray_crop[m]))
            delta = max(float(cfg.mask_refine.bg_min_diff), 3.0 * bg_mad)
            local_bin = (gray_crop <= bg_med - delta) if in_med < bg_med else (gray_crop >= bg_med + delta)
            cand = keep_largest_cc_overlapping(local_bin & prior, seed_mask=m)

    if cand is None or int(cand.sum()) < cfg.min_mask_px:
        return m

    # 护栏：防止把邻近另一条虫合并进来或缩成噪点
    area_ratio = float(cand.sum()) / float(orig_area)
    inter = int(np.logical_and(cand, m).sum())
    union = int(np.logical_or(cand, m).sum())
    iou = float(inter) / float(max(1, union))
    if not (cfg.mask_refine.area_ratio_lo <= area_ratio <= cfg.mask_refine.area_ratio_hi):
        return m
    if iou < cfg.mask_refine.min_iou_fallback:
        return m
    return cand
```

#### `pose/centerline.py` — 补充 `_compute_shape_moments(mask)`
```python
def _compute_shape_moments(mask: np.ndarray) -> tuple[float, float, float, float]:
    """从二值前景计算等效椭圆长短轴比 b/a、偏心率、主轴角、沿短轴 3 阶标准化偏度。
    特点：完全免骨架，对 10~60px 极小虫与 C 形弯曲高度稳定。
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 6:
        return float("nan"), float("nan"), float("nan"), float("nan")
    dx = xs.astype(np.float64) - float(xs.mean())
    dy = ys.astype(np.float64) - float(ys.mean())
    cxx = float(np.mean(dx * dx))
    cyy = float(np.mean(dy * dy))
    cxy = float(np.mean(dx * dy))

    tr = cxx + cyy
    det_term = math.sqrt(max(0.0, (cxx - cyy) ** 2 + 4.0 * cxy * cxy))
    lam1 = max(1e-6, 0.5 * (tr + det_term))   # 长轴方差
    lam2 = max(1e-6, 0.5 * (tr - det_term))   # 短轴方差
    aspect = float(math.sqrt(lam2 / lam1))    # b / a ∈ (0, 1]：直虫≈0.15~0.30，C弯跃升至 0.45~0.85
    ecc = float(math.sqrt(max(0.0, 1.0 - lam2 / lam1)))
    angle = float(0.5 * math.atan2(2.0 * cxy, cxx - cyy))  # 主轴角 ∈ [-π/2, π/2]

    # 沿短轴方向 v_minor = (-sin(angle), cos(angle)) 投影计算 3 阶标准化偏度
    # 物理意义：C 形弧的质心偏向凹侧，导致短轴投影呈显著非对称偏态；
    # 符号受主轴 ±π 不定性影响，后续在时间轴上对 angle 做 unwrap(2*angle)/2 即可得到连续带符号弯向！
    v_min_x, v_min_y = -math.sin(angle), math.cos(angle)
    proj_minor = dx * v_min_x + dy * v_min_y
    std_minor = math.sqrt(lam2)
    skew_minor = float(np.mean(proj_minor ** 3) / (std_minor ** 3 + 1e-9))
    return aspect, ecc, angle, skew_minor
```
> **调用要求**：在 `extract_pose()` 中，对 `m_main` 计算 `_compute_shape_moments(m_main)` 并写入返回的 `Pose`；**即使骨架抽取出错走 `_fallback_pca_axis`，也必须填充这 4 个字段！**

#### `pose/quality.py` — 防止 `area_hist` 永久锁死
- 新增成员 `self._drift_areas: deque[float]`（长度 `int(1.5 * fps)`）。
- 当某帧仅因 `clipped` 或 `merge` 判 `BAD`，但**同时满足** `length >= min_len_over_width * width` 且不含 `branch`/`fragment` 时，将该帧 `area` 记入 `_drift_areas`；若连续 `>= 1.5s` 所有帧面积高度一致（`std / median < 0.15`）且未出现 `OK` 帧，说明发生了持续光照/焦距漂移（而非单帧双虫粘连），此时平滑重置 `area_hist` 为 `_drift_areas` 并清空。
- **验证要求**：`tests/unit/pose/test_quality.py` 的单帧 `merge`（面积 × 1.6）测试必须继续判 `BAD`。

---

## 3. Workstream B：C 形运动双轨精准计数（`swim_liquid` & `crawl_agar`）

### 3.1 修复 `SwingCounter._fix_head_tail_flip`（解决 C2）
- 文件：`wormlab/kinematics/swing.py`
- **问题**：原代码用 `0.85 * u_ref + 0.15 * u_unit` 追踪 `u_ref`，游泳虫真实旋转时 EMA 滞后超过 $98^\circ$ 触发 `< -0.15`，反向锁死 `c_val`。
- **修复**：改为 **1 帧对齐参考 `u_prev_aligned`** + **深弯弦长保护 `R_curr`** + **单帧物理极限阈值 `< -0.50`**（1 帧内真实旋转绝不可能超过 $120^\circ$）：

```python
class SwingCounter:
    def __init__(self, cfg: SwingCfg, fps: float, gap_hold_s: float = 0.30) -> None:
        # ... 保留原有字段 ...
        self.u_prev_aligned: np.ndarray | None = None

    def _fix_head_tail_flip(
        self,
        c_val: float,
        u_curr: np.ndarray | None,
        R_curr: float | None = None,
    ) -> float:
        if u_curr is None:
            return c_val
        # 深 C/Ω 弯时首尾距离极小，弦向量不可靠，跳过弦翻转检测
        if R_curr is not None and np.isfinite(R_curr) and float(R_curr) < 0.35:
            return c_val
        u_arr = np.asarray(u_curr, dtype=np.float64).reshape(-1)
        if u_arr.shape[0] != 2 or not np.all(np.isfinite(u_arr)):
            return c_val
        u_norm = float(np.linalg.norm(u_arr))
        if u_norm <= 1e-6:
            return c_val
        u_unit = u_arr / u_norm
        if self.u_prev_aligned is None:
            self.u_prev_aligned = u_unit.copy()
            return c_val
        # 仅当单帧突跳 > 120° (dot < -0.50) 时才判定为头尾 180° 翻转；
        # 正常游泳旋转（如 15°~30°/帧）时 u_prev_aligned 同步旋转，绝不误翻转！
        if float(np.dot(u_unit, self.u_prev_aligned)) < -0.50:
            c_val = -c_val
            u_unit = -u_unit
        self.u_prev_aligned = u_unit.copy()
        return c_val
```
- **兼容性检查**：`tests/unit/kinematics/test_swing.py` 中 `test_swing_flip_immunity_and_biased_prominence` 在帧 4 和 8 注入单帧 `u_rev = [-1, 0]`（`dot = -1.0 < -0.50`）仍将精确翻转纠正并通过测试；同时新增一个**连续 360° 匀速旋转游泳单测**验证新实现不再产生假翻转！

### 3.2 修复 `KinematicsEngine` 在游泳模式下的坐标拉直 & 错用参数（解决 C3）
- 文件：`wormlab/kinematics/counter.py`, `wormlab/config.py`
- 改动点：
  1. `SwingCfg` 新增独立字段：
     ```python
     @dataclass(frozen=True)
     class SwingCfg:
         enabled: bool = False
         bend_high: float = 0.75
         min_flip_interval_s: float = 0.08
         gap_hold_s: float = 0.35               # 独立于 s_run.gap_hold_s！
         c_smooth_s: float = 0.07               # 在标量 c(t) 域平滑，而非在 (x, y) 坐标域平滑
         shape_fallback_enabled: bool = True    # 开启免骨架椭圆/偏度双轨计数
         shape_min_len_px: float = 55.0         # 中位体长低于此值自动启用免骨架主轨
         shape_bad_ratio_thresh: float = 0.15   # 骨架 BAD 率高于此值自动启用免骨架主轨
         shape_aspect_prom: float = 0.06        # b/a 带通波峰最小突出度
         shape_motion_min: float = 1.5          # 最小 ROI 帧间灰度变化，过滤静态卷虫
     ```
  2. `KinematicsEngine.__init__` 中改为 `self.swing = SwingCounter(cfg.swing, fps=self.fps, gap_hold_s=cfg.swing.gap_hold_s)`。
  3. 当 `cfg.swing.enabled and not cfg.s_run.enabled` 时：
     - `dense` **不做** `(x, y)` 线性均值平滑（令 `h_smooth = 0`），改为对标量序列 `c(t)` 做轻度 3 点中值/Savitzky-Golay 平滑，彻底避免高速游泳弯曲峰被拉直。
     - `self.swing.step(i, c_smooth, valid=(feat is not None), u_curr=..., R_curr=feat.R)` 透传 `R_curr`。

### 3.3 新增免骨架游泳计数模块 `wormlab/kinematics/swim_shape.py`（解决 C4 & `Swim-3/4`）
- **理论依据**：wrMTrck (Nussbaum-Krammer 2015) + WF-NTP (Koopman et al., Nature Protocols 2020) + Imaginera (Ji et al., G3 2024)。
- **核心思路**：
  - 每次游泳从“直 → 左 C → 直 → 右 C”时，等效椭圆短长轴比 `ar(t) = ell_aspect = b / a` 产生 **2 个清晰波峰**（每次弯曲一个峰，即直接对应 `c_bends`！）；且完全不需要骨架、完全免疫头尾翻转、完全免疫虫体 360° 整体旋转。
  - 叠加两个辅助物理量防误报：
    1. **动态活跃度门限**（`roi_motion >= shape_motion_min` 或局部滑窗 `std(ar) >= 0.03`）：避免静止死卷的高 `ar` 虫被误计。
    2. **连续带符号弯向 `signed_bend(t)`**：对 `ell_angle(t)` 做双倍角展开 `phi_u = 0.5 * unwrap(2 * ell_angle)`，令相邻帧主轴向量连续（若 `cos(phi_u[t] - angle_raw[t]) < 0` 则翻转 `ell_skew_minor[t]` 符号），得到无需头尾定义的带符号弯曲信号 `s_bend(t) = ar_hp(t) * sign(skew_aligned(t))`，可辅助交叉验证 `c_cycles`。
- **伪代码**：
```python
# wormlab/kinematics/swim_shape.py
def count_swim_from_shape(
    poses: Sequence[Pose | None],
    fps: float,
    cfg: SwingCfg,
) -> tuple[int, int, np.ndarray]:
    """基于 Pose.ell_aspect / ell_angle / ell_skew_minor / roi_motion 的免骨架游泳计数。
    返回 (c_cycles, c_bends, peak_frame_indices)。
    """
    n = len(poses)
    ar = np.full(n, np.nan, dtype=np.float64)
    mot = np.full(n, np.nan, dtype=np.float64)
    for i, p in enumerate(poses):
        if p is not None and np.isfinite(p.ell_aspect):
            ar[i] = float(p.ell_aspect)
            mot[i] = float(p.roi_motion) if np.isfinite(p.roi_motion) else float("inf")

    valid_idx = np.nonzero(np.isfinite(ar))[0]
    if len(valid_idx) < max(8, int(0.5 * fps)):
        return 0, 0, np.zeros(0, dtype=np.int32)

    # 1) 仅对 <= 0.25s 的极短缺口做 1D 标量插值（绝不做长缺口插值）
    ar_interp = interp_short_gaps_1d(ar, max_gap_frames=max(1, int(round(0.25 * fps))))

    # 2) 按连续有效段分别做带通滤波 (0.4 Hz ~ min(6.5 Hz, 0.42 * fps))
    #    - 0.4 Hz 高通：滤除虫体静态偏置与慢速体态漂移
    #    - 6.5 Hz 低通：滤除单帧像素抖动（野生型最高 ~4.5 bends/s）
    min_dist = max(2, int(round(float(cfg.min_flip_interval_s) * fps)))
    peaks_all: list[int] = []
    for seg_s, seg_e in contiguous_finite_segments(ar_interp, min_len=max(6, int(0.4 * fps))):
        sig = ar_interp[seg_s:seg_e]
        sig_bp = zero_phase_bandpass(sig, lo_hz=0.4, hi_hz=min(6.5, 0.42 * fps), fps=fps)
        # scipy.signal.find_peaks 突出度检测：每个波峰 = 1 次单侧 C 弯曲
        pks, props = scipy.signal.find_peaks(
            sig_bp,
            prominence=float(cfg.shape_aspect_prom),
            distance=min_dist,
        )
        for pk in pks:
            g_idx = seg_s + int(pk)
            # 活跃度校验：排除完全静止的死卷虫
            win_lo, win_hi = max(seg_s, g_idx - int(0.5 * fps)), min(seg_e, g_idx + int(0.5 * fps) + 1)
            local_mot = np.nanmedian(mot[win_lo:win_hi])
            if np.isfinite(local_mot) and local_mot < float(cfg.shape_motion_min):
                continue
            peaks_all.append(g_idx)

    c_bends = len(peaks_all)
    c_cycles = c_bends // 2
    return int(c_cycles), int(c_bends), np.asarray(peaks_all, dtype=np.int32)
```
- **触发融合策略（在 `KinematicsEngine.run_postprocessed` 内）**：
  - 当 `cfg.swing.enabled and cfg.swing.shape_fallback_enabled`：
    - 计算轨迹中位体长 `med_len` 与骨架无效占比 `bad_ratio = (n_bad + n_missing_with_shape) / max(1, n_with_shape)`。
    - 若 `med_len < cfg.swing.shape_min_len_px` **或** `bad_ratio > cfg.swing.shape_bad_ratio_thresh` **或**（骨架输出 `c_bends == 0` 但免骨架检测到显著周期运动）：采用 `count_swim_from_shape` 的 `(c_cycles, c_bends)` 回填终稿 `TrackReport` 与逐帧 `PerFrame`（同步更新 `c_count` 和 `behavior=Behavior.THRASH`）。

### 3.4 校准 `TurnCounter` 几何阈值（解决 C1，同时保住 `test_turns.py`）
- 文件：`wormlab/config.py` (`TurnCfg`), `wormlab/kinematics/turns.py`
- `TurnCfg` 新增两个可选字段：
  ```python
  @dataclass(frozen=True)
  class TurnCfg:
      # ... 保留原有字段 R_enter, R_exit, min_dwell_s, cooldown_s, max_curl_s, min_peak_c_rad ...
      R_c_enter: float | None = None    # C 形弯触发舒展比上限（如 0.76，对应 ~145°~180° 标准 C 形）
      c_enter_rad: float | None = None  # 与 R_c_enter 联合使用（如 1.65 rad），需同时满足才触发 C 弯
  ```
- `TurnCounter.step` 的 `EXT -> CURL` 触发条件改为：
  ```python
  deep_trigger = r_val <= r_enter
  c_bend_trigger = (
      self.cfg.R_c_enter is not None
      and self.cfg.c_enter_rad is not None
      and c_val is not None
      and np.isfinite(c_val)
      and r_val <= float(self.cfg.R_c_enter)
      and abs(float(c_val)) >= float(self.cfg.c_enter_rad)
  )
  if deep_trigger or c_bend_trigger:
      self.state = "CURL"
      ...
  ```
- **退出条件**：`r_exit_eff = max(r_exit, (R_c_enter + 0.08) if R_c_enter else r_exit)`（如 `R_c_enter=0.75` 时 `R_exit=0.82`）。
- **兼容性**：`test_turns.py` 中 `tc2` 调用 `tc2.step(i, 0.65, valid=True)` 未传 `c_val`，`c_bend_trigger` 为 `False`，原断言 `tc2.count == 0` 保持 100% 通过！而在真实 `crawl_agar` 视频中传入 `c_val` 后，标准 $150^\circ\text{–}180^\circ$ C 弯（$R \approx 0.64\text{–}0.74, |c| \ge 1.65$）能够准确触发。

---

## 4. Workstream C：S 形运动精准计数改造（消除 7 大漏斗损耗）

### 4.1 `Feat` 新增中段切向量 `tangent_mid`（支撑 S4）
- 文件：`wormlab/kinematics/features.py`
- 符号约定确认：`dense[0]` = 头，`dense[-1]` = 尾，`pts[0] - pts[-1]` 指向**头部（前进方向）**。
- 在 `Feat` 中新增字段 `tangent_mid: np.ndarray | None = None`（默认值 `None` 保兼容），在 `compute_features()` 内计算：
  ```python
  # 取中段 25%~75% 指向头部的单位切向量，避开头端觅食摆动与尾端卷曲噪声
  n_pts = len(pts)
  i_head = max(0, n_pts // 4)
  i_tail = min(n_pts - 1, (3 * n_pts) // 4)
  mid_vec = pts[i_head] - pts[i_tail]
  mid_norm = float(np.hypot(mid_vec[0], mid_vec[1]))
  tangent_mid = (mid_vec / mid_norm).astype(np.float32) if mid_norm > 1e-6 else u.astype(np.float32)
  ```

### 4.2 用“瞬时角动量 $L_z$ + 方向一致性”升级 `wave_gate`（解决 S2）
- 文件：`wormlab/config.py` (`WaveCfg`), `wormlab/kinematics/wave.py`
- **物理推导**：
  - 已知前向行波（头→尾）对应 $a_1 = A\cos(\omega t), a_2 = A\sin(\omega t)$，$\dot{\Phi} = +\omega > 0$。
  - 二维相平面瞬时面积速度（角动量）：
    $$L_z(t) = a_1(t)\dot{a}_2(t) - a_2(t)\dot{a}_1(t) = r^2(t)\,\dot{\Phi}(t)$$
  - **驻波 / 原地线性抖动**：$a_1, a_2$ 共线，$L_z(t) \equiv 0$。
  - **行波（无论慢速 0.15 Hz 还是刚起步 0.5s 短爆发）**：$L_z(t)$ 瞬时恒同号，短窗口（`win_s = 0.8~1.2s`）内方向一致性 $C_{\text{wave}} = \frac{|\sum L_z|}{\sum |L_z|} \approx 0.8\text{–}1.0$！
- **配置扩展 (`WaveCfg`)**：
  ```python
  @dataclass(frozen=True)
  class WaveCfg:
      win_s: float = 3.0
      min_valid_frac: float = 0.6
      r_min: float = 0.25
      eps_min: float = 0.40
      post_ratio_min: float = 0.35
      amp_post_min: float = 0.10
      # 新增角动量门限参数（gate_mode 默认 "eps" 保持 default.yaml 兼容；crawl_agar.yaml 切换为 "angular_momentum"）
      gate_mode: str = "eps"             # "eps" | "angular_momentum" | "hybrid"
      coherence_min: float = 0.55        # 行波方向一致性阈值 |ΣLz| / Σ|Lz|
      omega_min_rad_s: float = 0.35      # 最低加权角速度 (rad/s) ≈ 0.055 Hz，放行慢速爬行同时挡住静止漂移
  ```
- **`GateResult` 与 `wave_gate` 伪代码**：
  ```python
  @dataclass(frozen=True)
  class GateResult:
      ok: bool
      eps: float
      r_med: float
      amp_ant: float
      amp_post: float
      coherence: float = 0.0
      mean_omega: float = 0.0

  def wave_gate(
      window_feats: list[Feat | None],
      cfg: WaveCfg,
      expected_len: int | None = None,
      fps: float = 30.0,
  ) -> GateResult:
      # ... 保留有效帧数校验、r_med、eps（仍照常计算，保证 test_wave.py 断言与诊断字段完整）、amp_ant、amp_post ...

      # 按原始时间索引计算角动量 Lz = a1 * da2/dt - a2 * da1/dt（尊重中间缺失间隔 dt_k）
      valid_pairs = [(idx, f) for idx, f in enumerate(window_feats) if f is not None]
      lz_list: list[float] = []
      r2_list: list[float] = []
      for (i0, f0), (i1, f1) in zip(valid_pairs[:-1], valid_pairs[1:]):
          dt = float(i1 - i0) / max(1e-6, float(fps))
          if dt <= 0.0 or dt > 0.35:
              continue
          a1_m = 0.5 * (f0.a1 + f1.a1)
          a2_m = 0.5 * (f0.a2 + f1.a2)
          da1 = (f1.a1 - f0.a1) / dt
          da2 = (f1.a2 - f0.a2) / dt
          lz_list.append(a1_m * da2 - a2_m * da1)
          r2_list.append(a1_m * a1_m + a2_m * a2_m)

      if lz_list:
          lz_arr = np.asarray(lz_list, dtype=np.float64)
          r2_arr = np.asarray(r2_list, dtype=np.float64)
          coherence = float(abs(np.sum(lz_arr)) / (np.sum(np.abs(lz_arr)) + 1e-9))
          mean_omega = float(np.sum(lz_arr) / (np.sum(r2_arr) + 1e-9))
      else:
          coherence, mean_omega = 0.0, 0.0

      base_amp_ok = (
          r_med >= float(cfg.r_min)
          and amp_post >= float(cfg.post_ratio_min) * amp_ant
          and amp_post >= float(cfg.amp_post_min)
      )
      mode = getattr(cfg, "gate_mode", "eps")
      if mode == "angular_momentum":
          travel_ok = (coherence >= float(cfg.coherence_min)) and (abs(mean_omega) >= float(cfg.omega_min_rad_s))
      elif mode == "hybrid":
          travel_ok = (eps >= float(cfg.eps_min)) or (
              coherence >= float(cfg.coherence_min) and abs(mean_omega) >= float(cfg.omega_min_rad_s)
          )
      else:  # "eps"
          travel_ok = eps >= float(cfg.eps_min)

      return GateResult(
          ok=bool(base_amp_ok and travel_ok),
          eps=eps, r_med=r_med, amp_ant=amp_ant, amp_post=amp_post,
          coherence=coherence, mean_omega=mean_omega,
      )
  ```
- **注意**：`KinematicsEngine._step_frame` 调用 `wave_gate` 时显式传入 `fps=self.fps`。

### 4.3 修复 `SRunCounter` 的 `freeze()` 吞相位 & 切向路径积分（解决 S3 & S4）
- 文件：`wormlab/config.py` (`SRunCfg`), `wormlab/kinematics/s_runs.py`
- `SRunCfg` 新增字段：
  ```python
  @dataclass(frozen=True)
  class SRunCfg:
      # ... 保留原有字段 ...
      phase_bridge_s: float = 0.45       # freeze() 短暂掉出 <= 此秒数恢复时，若顺向且跳变 < 0.65π 则接续相位而非清空
      use_path_disp: bool = True         # True: 沿中段切线逐帧积分路径前进位移；False: 旧版首尾直线弦投影
  ```
- `SRunCounter` 改动伪代码：
  ```python
  # __init__ / _start 中新增：
  self.path_disp: float = 0.0            # 当前周期内沿体轴切线累计的有符号推进弧长 (px)
  self.prev_center: np.ndarray | None = None
  self.phase_bridge_frames = max(1, int(round(float(cfg.phase_bridge_s) * float(fps))))

  def step(
      self,
      i: int,
      gate_ok: bool,
      phi: float | None,
      center: np.ndarray | None,
      u: np.ndarray | None,
      L_ref: float,
      tangent_mid: np.ndarray | None = None,  # 新增可选参数，缺省回退 u
  ) -> None:
      if not gate_ok or phi is None or center is None or u is None:
          if self.active:
              self.gap += 1
              if self.gap > self.gap_hold_frames:
                  self._abort("gate_lost")
          return

      c_arr = np.asarray(center, dtype=np.float64)
      t_vec = np.asarray(tangent_mid if tangent_mid is not None else u, dtype=np.float64)
      t_unit = self._unit(t_vec)

      if not self.active or self.phi_prev is None:
          self._start(float(phi), c_arr)
          return

      dphi = float(wrap_pi(float(phi) - self.phi_prev))

      if self.gap > 0:
          # 关键修复 S3: 短间隙 (<= phase_bridge_frames) 恢复时，若方向一致且相位跳变合理，接续这段相位与位移！
          can_bridge = (
              self.gap <= self.phase_bridge_frames
              and abs(dphi) < 0.65 * np.pi
              and (self.direction == 0 or (self.direction * dphi) > 0.0)
          )
          self.gap = 0
          self.phi_prev = float(phi)
          if not can_bridge:
              # 超长间隙或发生大相位跳变，只重锚不计入，保持 prev_center 更新
              self.prev_center = c_arr.copy()
              self.last_center = c_arr.copy()
              return
          # 若 can_bridge=True，继续落入下方正常累加 dphi 与 path_disp！
      else:
          self.phi_prev = float(phi)

      # 关键修复 S4: 逐帧切向位移积分（弧线拐弯不再被直线弦压扁；原地抖动正负抵消依然≈0）
      if self.prev_center is not None:
          step_vec = c_arr - self.prev_center
          # 限制单帧异常跳变（如追踪闪跳 > 0.4 * L_ref），防脏帧注入假位移
          if float(np.linalg.norm(step_vec)) <= 0.45 * max(20.0, float(L_ref)):
              self.path_disp += float(np.dot(step_vec, t_unit))
      self.prev_center = c_arr.copy()
      self.last_center = c_arr.copy()
      self.axis_sum = self.axis_sum + np.asarray(u, dtype=np.float64)

      # ... direction == 0 锁定逻辑不变 ...

      self.prog += float(self.direction) * dphi
      cyc = self.prog - self.anchor_prog

      # 当前轮位移度量选择：path_disp 或 chord_disp
      chord_disp = float(np.dot(c_arr - self.anchor_center, self._unit(self.axis_sum)))
      p_eff = self.path_disp if self.cfg.use_path_disp else chord_disp

      # 反向清零规则保持不变（使用 p_eff）
      if (
          cyc < -float(self.cfg.reverse_phase_eps_rad)
          and int(np.sign(p_eff)) == -self.direction
          and abs(p_eff) >= float(self.cfg.reverse_disp_ratio) * float(L_ref)
      ):
          self._abort("reversal")
          self._start(float(phi), c_arr)
          return

      # 2π 结算周期：使用 p_eff 校验 ok_dir / ok_disp；每次结算后重置 path_disp = 0.0！
      # settle_partial() 同步使用 p_eff（同样在结算后重置 path_disp = 0.0）
  ```
- **防踩坑提示**：在 `counter.py` 中，每次调用 `self.s.freeze()` 时把 `self.s.gap` 递增 `self.s.gap += 1`（原代码 `if self.gap < 1: self.gap = 1` 无法知道实际冻结了多少帧，必须改为递增计数，这样 `self.gap <= self.phase_bridge_frames` 才能准确度量冻结时长！）。

### 4.4 解除 `adaptive_k_body` 死锁 + 深弯前先结算（解决 S1 & S5）
- 文件：`wormlab/kinematics/counter.py`, `wormlab/kinematics/modes.py`
- 改动点 1（`counter.py:run_postprocessed`）：
  - 删除 `if base_score < 0.30` 的死锁门槛！
  - 扩大候选网格 `k_grid = (1.00, 1.15, 1.30, 1.45, 1.60, 1.75)`。
  - 评分指标从单纯 `median(r)` 升级为 **能量 × 频带圆度/一致性**：
    ```python
    # 在候选 k 中选择使行波能量（活跃帧 75 分位 r）最大者；
    # 要求优于 base_k 至少 8%（1.08x）才切换，防止无意义微调抖动
    score_k = float(np.percentile(r_vals, 75))
    ```
  - 同时在 `modes.py` 提供可选函数 `fit_track_eigenworms(th_matrix)`（保留 `eigenworm_basis` 抛出 `NotImplementedError` 以兼容 `test_wave.py`），供后续 A/B 实验：对有效帧 $\tilde{\Theta} \in \mathbb{R}^{T \times 32}$ 做 SVD 取前 2 阶主模态 $(e_1, e_2)$，用 $k=1.45$ 的三角基对齐符号和手性（确保头→尾行波对应 $d\Phi > 0$）。
- 改动点 2（`counter.py:_step_frame` line 244）：
  - 在 `self.turn.in_curl and deep_coil` 且 `dwell >= min_dwell_frames` 执行 `self.s.force_abort("curl_or_bad")` **之前**：
    ```python
    tail_frac = getattr(self.cfg.s_run, "eof_tail_frac", None)
    if tail_frac is not None and self.s.active:
        self.s.settle_partial(i, L_ref, tail_frac=float(tail_frac))
    self.s.force_abort("curl_or_bad")
    ```
- 改动点 3：`crawl_agar.yaml` 默认 `s_run.k_body` 从 `1.15` 提升至 `1.45`（对齐 *C. elegans* 琼脂爬行物理波长 $\lambda \approx 0.65\,L$）。

### 4.5 升级卷曲感知头尾定向 `reorient_track_poses_coil_aware`（解决 S6）
- 文件：`wormlab/pose/orientation.py`
- 现有漏洞：
  1. 非首段（`si >= 1`）遇到**真实后退爬行**时，`score < 0.0` 会将其强行翻转头尾，变成假前进！
  2. 只在 `R < 0.45` 切段，浅卷（$R \in [0.45, 0.60]$）处翻头不被切段。
- 改进策略（吸收 Tierpsy 多线索思路，同时 100% 兼容 `test_orientation.py`）：
  1. **保留原 `coil_ratio` 切段**，并新增**单帧方向突跳切点**（若连续两有效帧间 `dot(u_t, u_{t-1}) < -0.5` 且期间无平滑过渡，视为潜在翻转点切开）。
  2. **区分“短时真实倒车”与“整段头尾装反”**：
     - 计算当前段长度 `seg_dur_s` 与全轨迹主推进速度特征。
     - 当 `preserve_first_head=True` 且 `si >= 1`（跨过卷曲后恢复）时：
       - 先比较跨卷曲前后两端姿态的**几何/运动学连续性**（若卷曲时间 $< 1.5\text{s}$，且该段是短暂倒车逃逸 `< 3.5s` 后立刻正向继续前进，则参考后继长段方向而不翻转短倒车段）；
       - 仅当该段持续时间 $\ge 3.0\text{s}$ 且 `score < -0.25`（线虫罕见持续倒车超过 3s 以上）或全轨迹绝大多数帧都一致反向时，才判定为跨卷曲翻头并翻转。
     - 当 `preserve_first_head=False`（未人工标头）时：
       - 结合**头部高频觅食微颤先验**（对前后各 15% 躯干切线角高频差分方差 `var(diff(theta[:, :5]))` vs `var(diff(theta[:, -5:]))`）与全轨迹主导运动方向联合投票，确定全局正方向，不再对每个短段孤立按 `score < 0` 乱翻。

---

## 5. Workstream D：`tools/tune.py` 对齐与评测基准闭环（解决 S7）

### 5.1 修复 `tools/tune.py`
当前 `tools/tune.py` 的 `_eval_fast` 与 `KinematicsEngine` 严重脱节（缺 `s_run.k_body`、`freeze`、`slowdown_hold_s`、`settle_partial`、`deep_coil_R`、角动量门限）。

**改造要求**：
1. 在 `tune.py` 中，对每组候选超参**直接调用 `KinematicsEngine.run_postprocessed(...)`**（或复用同一套 `wave_gate` + `SRunCounter` 步进函数）——10～20 条 20s 缓存轨迹跑一次 `run_postprocessed` 仅需 **~15 ms**，100 组网格搜索总耗时不到 2 秒，完全没有必要维护一套失真的 `_eval_fast` 副本！
2. 扩展 `tune.py` 支持 `--profile swim_liquid`：
   - 加载 `data/cache/{Swim-3, Swim-4}` 与 `benchmarks/labels/{Swim-3, Swim-4}.json`；
   - 网格搜索 `swing.bend_high`, `swing.shape_aspect_prom`, `swing.shape_motion_min`, `swing.min_flip_interval_s` 并输出 `benchmarks/tuned_swim_liquid.yaml`。

### 5.2 新增单测清单（必须补充）
在保留并跑通所有现有单测的前提下，新增以下测试：
1. `tests/unit/kinematics/test_wave.py`：
   - 新增 `test_wave_gate_angular_momentum_slow_crawl()`：验证 `f = 0.18 Hz` 慢爬在 `win_s = 1.2s` 下 `gate_mode="angular_momentum"` 返回 `ok=True`（原 `eps` 模式会失败），且驻波 `standing`、头部摆动 `head_wiggle`、静态噪声 `static_noise` 全部严格返回 `ok=False`！
2. `tests/unit/kinematics/test_s_runs.py`：
   - 新增 `test_s_run_freeze_bridges_short_gap_without_dropping_phase()`：验证 2 周期爬行中间插入 3 次 `0.15s` `freeze()` 闪烁后仍准确输出 `s_count == 2`（修复前只输出 `1`）。
   - 新增 `test_s_run_curved_path_displacement()`：验证沿半圆弧爬行 2 周期（直线弦缩水至 0.3 倍弧长）时 `use_path_disp=True` 准确计数 `s_count == 2` 且 `rejected["no_disp"] == 0`。
3. `tests/unit/kinematics/test_swing.py`：
   - 新增 `test_swing_continuous_rotation_no_false_flip()`：验证虫体在 `c = 1.4 * sin(2π * 1.8t)` 游泳同时以 $20^\circ/\text{frame}$ 匀速旋转 360° 时，计数仍精准为预期值（旧 `_fix_head_tail_flip` 会误触发大量假翻转）。
   - 新增 `test_swim_shape_counter_ellipse_aspect()`：构造合成二值 C 形摆动掩码序列（虫长仅 25 px），验证 `count_swim_from_shape` 误差 $\le 1$，且对完全静止的 C 形死卷掩码输出 `0`。
4. `tests/unit/kinematics/test_turns.py`：
   - 新增 `test_turn_counter_semicircle_c_bend()`：验证 $180^\circ$ 半圆 C 形（`R = 0.64, c = 2.5 rad`）在配置 `R_c_enter=0.75, c_enter_rad=1.65` 时准确计数，而直虫掩码缺角（`R = 0.64, c = 0.3 rad`）仍输出 `0`。
5. `tests/unit/pose/test_centerline.py`：
   - 验证 `extract_pose` 填充有效的 `ell_aspect, ell_ecc, ell_angle, ell_skew_minor`，且对细直虫 `ell_aspect < 0.35`、对半圆 C 虫 `ell_aspect > 0.50`。
6. `tests/unit/test_cache_roundtrip.py`：
   - 验证新增的 5 个 `Pose` 形状字段在 `CacheWriter -> CacheReader` 往返无损，且缺少这些键的旧版 `.npz` 能无缝读取不报错。

---

## 6. 推荐配置文件初始值草案（供调参前基线验证）

> **注意**：先在 `crawl_agar.yaml` 和 `swim_liquid.yaml` 中生效并跑通基准测试；`default.yaml` 仅追加新字段的保守默认值以保单测兼容。

### `re_struction/backend/config/profiles/crawl_agar.yaml`（增量改动建议）
```yaml
pose:
  coil_aware_reorient: true
  min_len_over_width: 2.2
  mask_refine:
    enabled: true
    use_bg_sub: true
    prior_dilate_px: 8
    bg_mad_k: 3.5
    bg_min_diff: 8

kinematics:
  k_body: 1.0
  smooth_s: 0.15
  gap_max_s: 0.30
  dc_remove_window_s: 0.0
  wave:
    win_s: 1.0                  # 角动量门限只需 1.0s 即可稳定判别，大幅减少边缘相位损失
    min_valid_frac: 0.55
    r_min: 0.14
    eps_min: 0.15
    gate_mode: "angular_momentum"
    coherence_min: 0.55
    omega_min_rad_s: 0.35
    post_ratio_min: 0.30
    amp_post_min: 0.06
  s_run:
    enabled: true
    k_body: 1.45                # 对齐 C. elegans 琼脂爬行真实波长 λ ≈ 0.65 L
    dir_lock_rad: 0.55
    reverse_phase_eps_rad: 0.45
    gap_hold_s: 0.90
    phase_bridge_s: 0.45        # 修复短闪烁吞相位
    use_path_disp: true         # 弯道切向路径积分
    slowdown_hold_s: 3.0
    min_disp_ratio: 0.08
    bridge_gap_s: 0.50
    adaptive_k_body: true
    eof_tail_frac: 0.65
  turn:
    enabled: true
    R_enter: 0.58               # 深 Ω 弯阈值（保持兼容 test_turns.py）
    R_c_enter: 0.75             # 新增：标准 145°~180° C 形弯联合阈值
    c_enter_rad: 1.65
    R_exit: 0.82
    min_dwell_s: 0.15
    cooldown_s: 0.45
    max_curl_s: 2.0
    min_peak_c_rad: 0.85
```

### `re_struction/backend/config/profiles/swim_liquid.yaml`（增量改动建议）
```yaml
pose:
  min_mask_px: 12
  min_len_over_width: 1.6
  upscale_below_px: 120
  mask_refine:
    enabled: true
    use_bg_sub: true
    prior_dilate_px: 8
    bg_mad_k: 3.0
    bg_min_diff: 6

kinematics:
  smooth_s: 0.0                 # 游泳模式禁止 (x, y) 坐标域均值平滑（防止拉直 C 峰）
  wave:
    win_s: 1.0
  s_run:
    enabled: false
  turn:
    enabled: false
  swing:
    enabled: true
    bend_high: 0.75
    min_flip_interval_s: 0.08
    gap_hold_s: 0.35
    c_smooth_s: 0.07            # 在标量 c(t) 域轻度平滑
    shape_fallback_enabled: true
    shape_min_len_px: 55.0
    shape_bad_ratio_thresh: 0.15
    shape_aspect_prom: 0.06
    shape_motion_min: 1.5
```

---

## 7. 分阶段实施顺序与验收命令

请按以下顺序分 PR / Commit 推进，每完成一阶段立刻跑验收命令并记录对比指标：

### Phase 1：基础设施（Workstream A）
- 改动：`config.py`, `schemas.py`, `pipeline/cache.py`, `pose/mask_refine.py`, `pose/centerline.py`, `pose/extractor.py`, `pose/quality.py`, `pipeline/pass_a.py`
- 验收：
  ```bash
  pytest re_struction/backend/tests/unit/test_config.py \
         re_struction/backend/tests/unit/test_cache_roundtrip.py \
         re_struction/backend/tests/unit/pose/
  ```

### Phase 2：S 形计数核心升级（Workstream C）
- 改动：`kinematics/features.py`, `kinematics/modes.py`, `kinematics/wave.py`, `kinematics/s_runs.py`, `kinematics/counter.py`, `pose/orientation.py`, `profiles/crawl_agar.yaml`
- 验收：
  ```bash
  pytest re_struction/backend/tests/unit/kinematics/ \
         re_struction/backend/tests/synthetic/test_kin_synth.py
  ```
  **必达指标**：
  - `test_engine_parity.py` 100% 通过（流式与批量逐位一致）；
  - `test_s_runs.py` 全部 12 类合成场景在 30fps & 10fps 通过，且 `crawl_bwd / treadmill / wrong_dir / head_wiggle / standing / static_noise / wobble` 严格输出 `0`。

### Phase 3：C 形计数双轨升级（Workstream B）
- 改动：`kinematics/swing.py`, `kinematics/swim_shape.py`（新增）, `kinematics/turns.py`, `kinematics/counter.py`, `profiles/swim_liquid.yaml`
- 验收：
  ```bash
  pytest re_struction/backend/tests/unit/kinematics/test_swing.py \
         re_struction/backend/tests/unit/kinematics/test_turns.py \
         re_struction/backend/tests/integration/
  ```

### Phase 4：真视频基准回归 & 调参对齐（Workstream D）
- 改动：`tools/tune.py`
- 验收命令（需重跑一次 Pass A 以生成含原图精修掩码与椭圆特征的新缓存）：
  ```bash
  python re_struction/tools/eval_benchmark.py --rerun-pass-a
  python re_struction/tools/tune.py --profile crawl_agar --dataset real
  python re_struction/tools/tune.py --profile swim_liquid --dataset real
  ```
- **对比报告输出要求**：输出改造前后在 `10x`, `1.6x-1`, `1.6x-3`, `Swim-3`, `Swim-4` 上每条虫的 `pred vs gt`、`MAE`、`gt==0 误报率`、`BAD 帧占比`、`rejected` 分项明细。

---

## 8. Workstream E（后续模型升级指南 — 仅在 Phase 1–4 完成后触发）

> 若 Phase 1–4 跑完后，评测诊断显示 **`Swim-4`（15px 极小点）或 `1.6x-3 #3` 仍然因检测框完全漏检（recall=0）导致计数为 0**，或密集虫体交叉盘绕成为主要残余误差，则说明已撞到现有 `origin.pt`（`imgsz=416` 训练）的模型上限，按以下两个路线二选一执行训练脚本开发：

### 路线 E1（低成本、快速解决极小目标漏检）：SAHI 多尺度切片重训 YOLO-Seg
1. 将高分辨率原始训练集与 `Swim-3/4`、`1.6x` 难例抽帧按 `640×640`（overlap `0.25`）切块，确保极小虫在新图块中占比放大 `3×`。
2. 放弃 `imgsz=416`，改用 `yolov8s-seg` / `yolo11s-seg` 在 **`imgsz=640` 或 `800`** 下微调训练（开启多尺度增强 `scale=0.5` 与低置信度小目标保留），训练完成后同步更新 `crawl_agar.yaml` / `swim_liquid.yaml` 的 `infer_imgsz`。

### 路线 E2（长期治本架构，推荐）：YOLO-Pose 11 关键点直出（Deep-Pose-Tracker 范式）
1. **伪标签自动生成脚本**（可放在 `re_struction/tools/export_pose_pseudolabels.py`）：
   - 读取 `PassB` 中 `Quality.OK` 且经 `reorient_track_poses_coil_aware` 校验过头尾一致的帧；
   - 直接导出 YOLO-Pose 标注格式：`class cx cy w h x0 y0 v0 ... x10 y10 v10`（其中 `kpt-0` = 头部 `pts11[0]`，`kpt-10` = 尾部 `pts11[10]`）。
2. 人工仅需对 200～300 张困难帧（深 C、$\Omega$ 环、极小游泳虫、轻度粘连）修正 11 点标注。
3. 微调 `yolo11s-pose`（`kpt_shape: [11, 3]`，`fliplr` 禁用头尾互换索引 `flip_idx`）。
4. 推理时直接把输出的有序 11 点样条插值到 `N_DENSE = 33` 注入 `Pose.dense`——**彻底绕开低清掩码、骨架化、毛刺剪枝、闭环解环、与头尾 180° 翻转整条脆弱几何链**。