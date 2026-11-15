# 02 姿态层（掩码 → 稠密中心线 → 朝向 → 质量）

对应任务：T07、T08、T10。目标：**C 形/深弯/小虫体也能稳定给出中心线**，不再因"分支多"或"体长变化"整帧丢弃。

## 1. 入口接口
```python
# wormlab/pose/centerline.py
def extract_pose(det_mask: np.ndarray, origin: tuple[int,int], cfg: PoseCfg) -> Pose:
    """det_mask: 裁剪后的 bool 掩码；origin=(x0,y0)。永不抛异常；失败返回 Pose(dense=None, quality=MISSING, flags={...})。"""

# wormlab/pose/extractor.py  （T08 一并实现，Pass A 使用）
class PoseExtractor:
    """每个轨迹一份状态：上一帧姿态（用于朝向）、质量判定历史。线程安全：不同轨迹的实例可并行。"""
    def __init__(self, cfg: PoseCfg, fps: float, user_head_xy=None): ...
    def process(self, det: Detection, frame_gray: np.ndarray | None = None) -> Pose:
        pose = extract_pose(det.mask, det.mask_origin, cfg)
        if pose.dense is not None:
            pose.dense = orient(pose.dense, self.prev_dense, self.user_head_xy)   # §5
            self.user_head_xy = None                                              # 只在首次生效
        pose = self.judge.judge(pose)                                             # §6
        if pose.dense is not None and pose.quality != Quality.BAD:
            self.prev_dense = pose.dense
        return pose
```

## 2. 预处理（ROI、放大、填洞）
```
def extract_pose(mask, origin, cfg):
    m = mask.astype(bool)
    area0 = int(m.sum())
    if area0 < cfg.min_mask_px (=30): return _missing({"empty"})

    # 2.1 最大连通域
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    keep = argmax(stats[1:, CC_STAT_AREA]) + 1
    flags = set()
    if n > 2 and second_largest_area > cfg.fragment_ratio(=0.25) * largest: flags.add("fragment")
    m = (lab == keep)

    # 2.2 放大（小虫体）：long_side = max(包围盒宽, 高)
    if long_side < cfg.upscale_below_px (=120):
        scale = min(cfg.max_upscale (=4.0), cfg.upscale_target_px (=240) / long_side)
        f = cv2.GaussianBlur(m.astype(np.float32), (0,0), 0.8)
        f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        m = f > 0.5 ; flags.add("upscaled")
    else: scale = 1.0

    # 2.3 填洞（Ω 弯自接触会产生空洞）
    filled = scipy.ndimage.binary_fill_holes(m)
    if (filled.sum() - m.sum()) > cfg.loop_hole_frac (=0.005) * filled.sum(): flags.add("loop")
    m = filled
```
> 放大的目的：Swim 视频里虫体仅 40–60 px 长，1 px 的骨架量化误差就是 2% 体长；放大 3–4 倍后骨架抖动变成亚像素。

## 3. 骨架与测地路径（取代旧"BFS 最长路 + 分支比熔断"）
```
    dt = scipy.ndimage.distance_transform_edt(m)            # 距离变换
    sk = skimage.morphology.skeletonize(m)
    ys, xs = np.nonzero(sk); n = len(xs)
    if n < 8: return _fallback_axis(...)                      # §3.3

    # 3.1 建图：8 邻域，只枚举 4 个"半邻域"避免重复边
    idx = -np.ones(m.shape, int); idx[ys, xs] = arange(n)
    for (dy,dx) in [(0,1),(1,-1),(1,0),(1,1)]:
        a = idx[ys, xs]; b = idx[ys+dy, xs+dx] (越界/-1 过滤)
        geo = hypot(dy,dx)
        cost = geo * (1 + cfg.center_penalty (=2.0) * (1 - 0.5*(dt[a]+dt[b])/dt.max()))   # 越靠中轴越便宜
    G_geo  = csr_matrix(权重=geo)       # 仅用于找端点
    G_cost = csr_matrix(权重=cost)      # 用于取路径

    # 3.2 直径端点（双向扫描）+ 路径
    d0 = dijkstra(G_geo, indices=0, directed=False);          a = argmax(where(isfinite(d0), d0, -1))
    da = dijkstra(G_geo, indices=a, directed=False);          b = argmax(where(isfinite(da), da, -1))
    _, pred = dijkstra(G_cost, indices=a, return_predecessors=True, directed=False)
    path = backtrack(pred, b)   # 节点序列 a→b；坐标 P = stack([xs[path], ys[path]], 1).astype(float)
    # 骨架不连通（多个连通块）→ 只取含 path 的那块；其余像素计入 branch

    # 3.3 _fallback_axis（骨架点<8 或失败）：对掩码像素做 PCA，主轴上投影，取 2%~98% 分位端点，
    #     沿主轴线性生成 N_DENSE 点；flags.add("fallback")；quality 稍后在 §6 至少为 SUSPECT

    # 3.4 分支度量（只用于质量判定，不丢帧）
    on = zeros_like(sk); on[ys[path], xs[path]] = True
    near = binary_dilation(on, iterations=2)
    branch_px = int((sk & ~near).sum()); branch_ratio = branch_px / len(path)
    if branch_ratio > cfg.branch_flag_ratio (=0.35): flags.add("branch")
```

## 4. 平滑、端帽、重采样、度量
```
    # 4.1 沿路径高斯平滑（端点固定，避免收缩）
    sigma = max(1.5, cfg.smooth_sigma_frac (=0.03) * len(P))
    Ps = gaussian_filter1d(P, sigma, axis=0, mode="nearest"); Ps[0] = P[0]; Ps[-1] = P[-1]

    # 4.2 端帽修正：骨架比真实虫体每端短约一个半宽
    if cfg.endcap:
        for end in (head_end, tail_end):
            k = max(3, int(0.05*len(Ps)));  t = normalize(Ps[end] - Ps[end±k])     # 向外的切向
            ext = dt[round(Ps[end].y), round(Ps[end].x)] * cfg.endcap_gain (=0.9)
            # 沿 t 以 0.5px 步进延伸，直到走出掩码 m 或达到 ext，取最后一个仍在 m 内的点
            Ps = 在对应端追加该点
    # 4.3 弧长等距重采样到 N_DENSE 点
    dense = resample_polyline(Ps, N_DENSE)

    # 4.4 度量（单位：原始像素）
    length = polyline_length(Ps) / scale
    chord  = norm(dense[0]-dense[-1]) / scale
    width  = 2 * mean(dt[path]) / scale
    area   = area0                                  # 注意：用填洞前、原分辨率面积
    dense  = dense / scale + origin                 # 回到整帧坐标

    # 4.5 轮廓（供渲染，Pass B 无掩码可用）
    bin_f = GaussianBlur(mask_original.astype(float32), (5,5), 1.0) >= 0.5   # 同旧 _draw_contour_overlay
    cs = cv2.findContours(bin_f.astype(uint8), RETR_EXTERNAL, CHAIN_APPROX_NONE)
    取面积>50 的最大轮廓 → cv2.approxPolyDP(eps=0.7) → 若点数>80 则等间隔抽样到 80 → +origin
```
**验收（T07）**：用 `tools/make_synthetic.py` 的渲染掩码（07 §1）：直线/弧/S/C/Ω，长度误差 <5%（C、Ω <8%），端点位置误差 <2 px（原分辨率），`loop` 标记在 Ω 自接触用例触发，**任何用例都不得返回 None 的 dense（除非掩码<30px）**。

## 5. 朝向（`pose/orientation.py`）
```python
def orient(dense, prev_dense=None, user_head_xy=None, margin=0.9) -> np.ndarray:
    # 1) 首帧有用户头点 → 使 dense[0] 更接近 user_head_xy：
    #      if |dense[-1]-head| < |dense[0]-head|: dense = dense[::-1]
    # 2) 有上一帧 prev_dense (33,2)：
    #      d_f = mean(norm(dense - prev, axis=1));  d_r = mean(norm(dense[::-1] - prev, axis=1))
    #      if d_r < margin * d_f: dense = dense[::-1]           # margin=0.9 迟滞，近似对称时保持不翻
    # 3) 都没有 → 原样返回
    return dense
```
间隔很久（丢失重现）时同样使用最近一次 `prev_dense`；若丢失 > `reorient_gap_s` 秒，改用"端点相对 Kalman 预测速度方向"不可靠，故直接沿用 prev 比较并记 `flags.add("reoriented_after_gap")`。
单测：构造序列 + 在若干帧人为翻转，输出应恢复连续；对称直线静止不翻。

## 6. 质量判定（`pose/quality.py`）——把"无效"和"异常"解耦
```python
class PoseQualityJudge:
    def __init__(self, cfg, fps): self.area_hist = deque(maxlen=frames(cfg.ref_window_s)); self.len_hist = deque(...)
    def judge(self, pose) -> Pose:
        if pose.dense is None: pose.quality = MISSING; return pose
        ref_n = len(self.area_hist)
        area_ref = median(area_hist) if ref_n >= cfg.min_ref_samples (=10) else None
        bad = False; suspect = False
        if area_ref:
            r = pose.area / area_ref
            if r > cfg.bad_area_hi (=1.35) or r < cfg.bad_area_lo (=0.55): bad = True; pose.flags |= {"merge" if r>1 else "clipped"}
            lr = pose.length / median(len_hist)
            if not (cfg.suspect_len_lo (=0.75) <= lr <= cfg.suspect_len_hi (=1.25)): suspect = True
        if "branch" in flags and branch_ratio > cfg.branch_bad_ratio (=0.5): bad = True
        if pose.length < cfg.min_len_over_width (=3.0) * pose.width: bad = True     # 圆球/虫卵/碎片
        if flags & {"loop","fragment","fallback"}: suspect = True
        pose.quality = BAD if bad else (SUSPECT if suspect else OK)
        if pose.quality == OK: area_hist.append(pose.area); len_hist.append(pose.length)
        return pose
```
**使用语义（写进注释）**：`OK/SUSPECT` 都参与计数；`BAD` 视为缺失，由运动学层按 `gap_max_s` 插值，超时才断开 run；**任何质量等级都不得复位计数状态**。
单测：稳定序列后插入面积×1.6 的帧 → BAD；弯曲导致长度 −30% 但面积不变 → SUSPECT 且仍有 dense。

## 7. 掩码精修（可选，默认关；`cfg.mask_refine.enabled`）
```python
def refine_mask(gray_crop, mask_crop, cfg) -> np.ndarray:
    # 在 mask 膨胀 2px 的区域内做局部 Otsu（虫体亮/暗按区域均值判定，同旧 extract_manual_worm 的 edge_mean/center_mean 逻辑）
    # 与原掩码求交；若 IoU(refined, mask) < 0.7 或面积变化 > ±20% → 回退原掩码
```
T07 只需把接口留好并写单测；是否启用由 T10 的诊断图决定。

## 8. 诊断工具（`tools/diagnose_track.py`，T10）
```
python tools/diagnose_track.py --video PATH --track-id N [--start-frame a --end-frame b] [--out data/diag/]
输出 PNG：
 (1) 6 帧拼图：原图 ROI + 掩码轮廓 + 33 点中心线（头=绿点）+ 质量标签
 (2) 时间曲线：length/area/chord、R、质量等级色带
 (3) 若已有运动学结果（T15 后）：a1-a2 相平面圆度、Φ(t)、c(t)、门限与 run/周期事件标注
```

## 9. 配置块（并入 `default.yaml` 的 `pose:`）
```yaml
pose:
  n_dense: 33
  min_mask_px: 30
  fragment_ratio: 0.25
  upscale_below_px: 120
  upscale_target_px: 240
  max_upscale: 4.0
  loop_hole_frac: 0.005
  center_penalty: 2.0
  branch_flag_ratio: 0.35
  smooth_sigma_frac: 0.03
  endcap: true
  endcap_gain: 0.9
  ref_window_s: 2.0
  min_ref_samples: 10
  bad_area_hi: 1.35
  bad_area_lo: 0.55
  branch_bad_ratio: 0.5
  suspect_len_lo: 0.75
  suspect_len_hi: 1.25
  min_len_over_width: 3.0
  reorient_gap_s: 1.0
  mask_refine: {enabled: false}
```
