# 线虫运动识别与姿态运动学核心算法架构文档 (Core Algorithm Reference)

> **文档定位**：本文档详细整理了本项目中关于**线虫图像识别、分割骨架提取、多目标跟踪（MOT）及运动姿态计数（C型/S型运动状态机）**的核心实现源码与架构设计。旨在提供给更高能力的 AI 模型或算法工程师进行深度审查、算法优化与代码重构。

---

## 一、 系统全景架构与数据流 (System Architecture & Pipeline)

整个处理管线采用**多进程/多线程解耦架构**，将“视频读取预取”、“GPU 深度学习推理”、“轻量目标跟踪与姿态分析”以及“前端 WebSocket 遥测”分离：

```mermaid
graph TD
    A[输入视频 / 实时视频流] --> B[帧捕获队列 FrameQueue]
    B --> C[GPU Batch 推理: YOLOv8-Seg with retina_masks]
    C --> D[检测预过滤: MIN_CONFIDENCE & MIN_AREA]
    D --> E[增强追踪器: EnhancedWormTracker]
    E -->|8状态卡尔曼滤波| F[运动位置预测与匈牙利多特征匹配]
    E -->|局部稳定 ROI 裁剪| G[自适应二值化 / skimage 骨架化 BFS]
    G --> H[等距重采样: 11 节点中心线 & 头尾朝向对齐]
    H --> I[运动学分析引擎: TrackHistory]
    I -->|迂曲度 & 距离CV| J[C型弯曲 / 回折计数状态机]
    I -->|带符号曲率 k(t) & 过零回直检测| K[S型正弦波 2π 全周期双向闭合状态机]
    J -->|C型触发强力熔断| K
    I --> L[实时生理指标: 摆动频率 Hz / BPM / 活跃度]
    E --> M[平滑抗锯齿轮廓绘制 & 遥测上报 WebSocket]
```

---

## 二、 目标检测与骨架提取算法 (Detection & Centerline Extraction)

### 1. YOLOv8 分割掩码与过滤机制
* 模型采用基于 Ultralytics YOLOv8-Seg 的实例分割模型。
* 推理开启 `retina_masks=True`，保证输出原生高分辨率二值掩码，杜绝低分辨率上采样引入的边缘锯齿。
* 过滤规则：
  $$\text{Keep} = (\text{Area} \ge W \times H \times \text{MIN\_AREA\_RATIO}) \land (\text{Confidence} \ge \text{MIN\_CONFIDENCE})$$
  其中默认 `MIN_AREA_RATIO = 0.0003`，`MIN_CONFIDENCE = 0.35`。

### 2. 局部稳定窗口中心线骨架提取 (`extract_centerline_from_mask`)
为了避免在全图 ($1920 \times 1080$) 上进行昂贵的形态学细化，跟踪器维护动态平滑裁剪窗口 `crop_center` 与 `crop_size`，仅在局部 ROI 内提取骨架：

```python
def extract_centerline_from_mask(mask, n_keypoints=11):
    """
    输入: 局部二值掩码 (uint8 或 bool)
    输出: 
        centerline_resampled: (n_keypoints, 2) 沿体轴等距排列的中心线坐标
        keypoints: (n_keypoints, 2) 关键点（等同于中心线采样点）
    """
    if mask is None or np.sum(mask > 0) < 10:
        return None, None
    
    # 1. 骨架化：提取单像素中轴线 (Lee's Algorithm)
    skeleton = skeletonize(mask > 0)
    pts = np.argwhere(skeleton)
    if len(pts) < n_keypoints:
        return None, None
    pts = pts[:, ::-1]  # (y, x) -> (x, y)

    # 2. 邻接图构建与端点发现 (8邻域统计)
    # 统计每个骨架点的邻居数，邻居数为 1 的点即为候选端点（头部/尾部）
    neighbors = defaultdict(list)
    pts_set = set(map(tuple, pts))
    for x, y in pts:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (x + dx, y + dy)
                if neighbor in pts_set:
                    neighbors[(x, y)].append(neighbor)
    
    endpoints = [pt for pt, neigh in neighbors.items() if len(neigh) == 1]
    
    # 3. 测地最长路径搜索 (BFS / Dijkstra)
    if len(endpoints) >= 2:
        # 在候选端点对中寻找最远路径作为主干
        best_path = []
        for start in endpoints:
            visited = {start}
            queue = deque([(start, [start])])
            while queue:
                curr, path = queue.popleft()
                if len(path) > len(best_path):
                    best_path = path
                for neighbor in neighbors[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
        ordered_pts = np.array(best_path, dtype=np.float32)
    else:
        # 端点缺失（例如自闭环）：退化采用主成分分析 (PCA) 轴线排序投影
        pca = PCA(n_components=1)
        proj = pca.fit_transform(pts)
        ordered_pts = pts[np.argsort(proj.flatten())].astype(np.float32)

    if len(ordered_pts) < 2:
        return None, None

    # 基于主干像素占比的粘连/交叉熔断：计算未落在最长主干上的骨架分支像素数
    # 若存在长于体长 18% 的真实交叉分支（双虫重叠粘连），直接熔断丢弃
    branch_pixels = len(pts) - len(ordered_pts)
    if branch_pixels > max(8, 0.18 * len(ordered_pts)):
        return None, None

    # 4. 样条曲线沿弧长等距重采样为标准 11 节点
    diffs = np.diff(ordered_pts, axis=0)
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    total_len = cum_dists[-1]
    if total_len < 1e-3:
        return None, None

    # 两端各内缩 5% 避开掩码末端分叉与毛刺
    even_dists = np.linspace(0.05 * total_len, 0.95 * total_len, n_keypoints)
    sampled_x = np.interp(even_dists, cum_dists, ordered_pts[:, 0])
    sampled_y = np.interp(even_dists, cum_dists, ordered_pts[:, 1])
    centerline = np.column_stack([sampled_x, sampled_y])
    
    return centerline, centerline
```

### 3. 自适应多模式头尾朝向一致性对齐 (`orient_pose_pair`)
骨架细化算法本身没有头尾概念，首尾序号可能在相邻帧发生 $180^\circ$ 翻转。系统自适应支持无论调用方传入的是 `(11, 2)` 的完整上一帧关键点序列，还是 `(2,)` 的上一帧头部单点坐标 `last_head_pos`，都能正确对齐头尾，严禁出现跳过对齐：

```python
def orient_pose_pair(centerline, keypoints, last_pose_or_head=None, ref_head=None, last_keypoints=None):
    """
    自适应保持 head=index 0 的时间连续性：
    - 若传入 (11, 2) 序列：比较正序与反序相对于上一帧全序列的平均位移 (推荐主模式)
    - 若传入 (2,) 单点：比较 head 与 tail 距离锚点的欧氏距离 (回退模式)
    """
    if keypoints is None or len(keypoints) < 2:
        return centerline, keypoints
    kp = np.asarray(keypoints, dtype=np.float32)
    
    target_pose = last_pose_or_head if last_pose_or_head is not None else last_keypoints
    if target_pose is not None:
        arr = np.asarray(target_pose, dtype=np.float32)
        if arr.ndim == 2 and len(arr) == len(kp):
            d_forward = float(np.mean(np.linalg.norm(kp - arr, axis=1)))
            d_reverse = float(np.mean(np.linalg.norm(kp[::-1] - arr, axis=1)))
            if d_reverse < d_forward:
                kp = kp[::-1].copy()
                if centerline is not None:
                    centerline = np.asarray(centerline, dtype=np.float32)[::-1].copy()
            return centerline, kp
        elif arr.ndim == 1 and arr.size == 2:
            ref_head = arr

    if ref_head is not None:
        head_d = float(np.hypot(kp[0, 0] - ref_head[0], kp[0, 1] - ref_head[1]))
        tail_d = float(np.hypot(kp[-1, 0] - ref_head[0], kp[-1, 1] - ref_head[1]))
        if tail_d < head_d:
            kp = kp[::-1].copy()
            if centerline is not None:
                centerline = np.asarray(centerline, dtype=np.float32)[::-1].copy()
    return centerline, kp
```

---

## 三、 多目标跟踪与数据关联引擎 (Multi-Object Tracking & Association)

### 1. 卡尔曼滤波器状态空间 (`KalmanBoxTracker`)
* **状态向量** (8维): $x = [c_x, c_y, w, h, v_x, v_y, v_w, v_h]^T$
* **观测向量** (4维): $z = [c_x, c_y, w, h]^T$
* **状态转移协方差**：
  $$Q = I_8 \times 0.05, \quad R = I_4 \times 0.1$$

### 2. 多特征加权关联代价矩阵 (Weighted Cost Matrix)
关联阶段结合了空间重叠、动力学一致性、航向夹角及多阶不变矩外观特征：

$$\text{Cost}(t, d) = w_{\text{iou}} C_{\text{iou}} + w_{\text{motion}} C_{\text{motion}} + w_{\text{dir}} C_{\text{dir}} + w_{\text{app}} C_{\text{app}}$$

* 当前设定权重：
  * $w_{\text{motion}} = 0.45$（主导项：基于速度平滑差与虫体尺寸稳定性）
  * $w_{\text{iou}} = 0.25$（空间 IoU 距离）
  * $w_{\text{dir}} = 0.15$（瞬时运动矢量与质心夹角余弦相似度）
  * $w_{\text{app}} = 0.15$（11维轮廓特征：对数面积、周长、圆度、长宽比及 Hu 不变矩）

### 3. 三阶段数据匹配机制
1. **第一阶段（主匹配）**：基于上述加权矩阵执行匈牙利匹配，截断阈值 $\text{Cost} < 0.9$。
2. **第二阶段（低置信度次匹配）**：对于未匹配的轨迹，与低置信度检测框进行松弛阈值匹配。
3. **第三阶段（视觉残留找回 Re-ID）**：
   * 针对丢失帧数在 `GHOST_OVERLAY_SEC` 范围内的目标，采用自适应搜索门限：
     $$\text{Gate} = \min\left(v_{\max} \times \left(1.0 + \frac{\text{lost\_frames}}{20.0}\right), 0.18 \times \max(W, H)\right)$$
   * 跨越遮挡或短暂漏检重新检出时，强制复用原有 ID，坚决避免无谓新建 ID。

### 4. 零冲突严格受限 ID 分配池 (`id_pool`)
* 初始化时根据用户设定总数限制：`id_pool = list(range(1, max_worm_id + 1))`。
* **置信度优先分配机制**：每帧对未匹配的候选检测框按置信度降序排序，有限的 ID 优先分配给高置信度单位；当 ID 池用尽时直接截断，拒绝为多余噪点分配 ID。
* **严格受限闭环**：当用户设定虫数上限时，`_alloc_id()` 严禁溢出分配（池空返回 `None`），保证活跃轨迹与 ID 总数绝对不超过用户设定的上限；未设上限（无限制模式）时才允许单调递增。
* **瞬态误检快速回收**：未确认轨迹（`hits < confirm_hits`）若短时丢失（$\ge 3$ 帧）立即回收并归还 ID 给 `id_pool`，防止单帧噪点长期锁定池内名额。

---

## 四、 运动姿态分类与周期计数核心 (Posture Analysis & Cycle State Machines)

这是本项目最具原创性的核心算法部分。我们彻底摒弃了传统方案中粗暴的“波峰简单累加”与“端点抖动求和”，构建了**互斥熔断与严谨 $2\pi$ 正弦波双向闭合状态机**。

### 1. 物理量计算：体长归一化躯干曲率 (`compute_normalized_body_curvatures`)

> **诊断背景 (Argon)**：旧版算法在头部前 3 个节点计算曲率，存在两大致命缺陷：
> 1. **生物学觅食微动干扰**：线虫在原地缓慢爬行或静止时，头部（前 15% 体长）会频繁发生非推进性的觅食摆头（Foraging），波形并未向后传导，却导致曲率高频过零并触发计数。
> 2. **掩码边缘毛刺放大**：骨架端点受二值化毛刺影响剧烈，短间距二阶导数极易出现离群尖峰。

**改进方案**：弃用头部端点，改在 11 节点中取**躯干前中段 (索引 2, 3, 4)** 与 **躯干中后段 (索引 6, 7, 8)** 计算无量纲体长归一化曲率 $\tilde{k} = k \times L$：

$$k_{\text{seg}} = \frac{2 \cdot (v_1 \times v_2)}{\|v_1\| \|v_2\| \|p_{\text{next}} - p_{\text{prev}}\| + \epsilon} \times L_{\text{body}}$$

```python
def compute_normalized_body_curvatures(centerline):
    """
    计算体长归一化身体曲率 (Argon 躯干中段双段相位方案)
    取躯干前中段(节点2,3,4)与躯干中后段(节点6,7,8)计算无量纲归一化曲率
    返回: (k_anterior, k_posterior, body_len)
    """
    if centerline is None or len(centerline) < 11:
        return None, None, 0.0

    centerline = np.asarray(centerline, dtype=float)
    diffs = np.diff(centerline, axis=0)
    body_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
    if body_len < 10.0:
        return None, None, 0.0

    def _segment_curv(p_prev, p_curr, p_next):
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        d1 = np.linalg.norm(v1)
        d2 = np.linalg.norm(v2)
        d3 = np.linalg.norm(p_next - p_prev)
        curvature = 2.0 * cross / (d1 * d2 * d3 + 1e-6)
        return float(curvature * body_len)

    k_anterior = np.mean([_segment_curv(centerline[i-1], centerline[i], centerline[i+1]) for i in (2, 3, 4)])
    k_posterior = np.mean([_segment_curv(centerline[i-1], centerline[i], centerline[i+1]) for i in (6, 7, 8)])
    return float(k_anterior), float(k_posterior), float(body_len)
```

---

### 2. C 型回折运动识别 (`_detect_c_motion`)

#### 判定原理：
当线虫发生剧烈深度弯曲（Omega 弯、C-Turn）时，头尾端点距离瞬间大幅缩短，质心路径严重盘旋。

```python
def _detect_c_motion(self):
    win = self.C_WIN # 20 采样点 (~2.0s)
    if len(self.centers) < win or len(self.end_dists) < win:
        return

    centers_w = list(self.centers)[-win:]
    dists_w   = np.array(list(self.end_dists)[-win:])

    # 1. 端点距离变异系数 CV = std / mean
    mean_d = float(np.mean(dists_w))
    cv = float(np.std(dists_w)) / (mean_d + 1e-9)

    # 2. 轨迹迂曲比 = 累积路径长度 / 净位移
    path_len   = sum(np.linalg.norm(centers_w[i] - centers_w[i-1]) for i in range(1, win))
    net_disp   = np.linalg.norm(centers_w[-1] - centers_w[0])
    tortuosity = path_len / (net_disp + 1e-9)

    # 3. C 态综合判定
    is_c = (cv > self.C_CV_THRESH) and (tortuosity > self.C_TORTUOSITY) and (path_len > self.C_MIN_MOVE)
    
    if is_c:
        # ★ 关键互斥保护：发生 C 型转弯瞬间彻底打断正弦波传递，立即弃用挂起半周期！
        self._reset_s_pending()

    if not is_c:
        return

    # 4. 原点带动态质心与进出穿越计数
    cx_arr = np.array([c[0] for c in centers_w])
    cy_arr = np.array([c[1] for c in centers_w])
    origin_new = np.array([np.median(cx_arr), np.median(cy_arr)])
    if self._origin_center is None:
        self._origin_center = origin_new
    else:
        self._origin_center = 0.85 * self._origin_center + 0.15 * origin_new

    spread = np.std(np.linalg.norm(np.column_stack([cx_arr, cy_arr]) - self._origin_center, axis=1))
    self._origin_radius = max(3.0, spread * self.C_ORIGIN_MARGIN)

    curr_center = list(self.centers)[-1]
    dist_to_origin = np.linalg.norm(curr_center - self._origin_center)
    is_outside = dist_to_origin > self._origin_radius

    if self._was_outside is not None and is_outside != self._was_outside:
        cool = self.frame_counter - self._last_cross_idx > self.C_CROSS_COOLDOWN
        if cool:
            self._c_cross_count += 1
            self._last_cross_idx = self.frame_counter
            if self._c_cross_count >= self.C_CROSSES_PER_COUNT: # 2次穿越 = 1次完整动作
                self.c_count += 1
                self._c_cross_count = 0
                self._reset_s_pending()

    self._was_outside = is_outside
```

---

### 3. S 型正弦蠕动前后躯干伴随传导迟滞状态机 (`_detect_s_motion`)

> **诊断背景与核心漏洞**：
> 1. **半身摆动误判**：若仅检测前段曲率 $k_{\text{ant}}$，后半身定死在原地、仅前 $1/3$ 身体左右摆动的线虫也会被误计为 S 运动。因此必须同时校验波是否传导至后段 $k_{\text{post}}$。
> 2. **推进比除数漏洞**：原先过滤掉 0 后的 `len(non_zero_signs)` 作为分母，导致 25 帧内即便 24 帧静止、仅 1 帧因摆头引起微移，除出来的也是 $1.0 \ge 0.25$。修复为严格除以完整窗口长度 `S_WIN`，并新增质心净位移下限门槛。

**重构方案 (前后躯干伴随传导验证 + 质心净位移门限 + 施密特双门限迟滞状态机)**：
1. **重叠/粘连形态熔断**：通过 `_is_overlapping_or_abnormal` 检测体长相对于最近 10 帧中位数的突变，超出 $20\%$ 判定发生交叠遮挡，立即重置并冻结计数。
2. **前后段零滞后平滑滤波**：$\alpha = 0.35$ 的指数移动平均分别更新 $\tilde{k}_{\text{ant\_ema}}$ 与 $\tilde{k}_{\text{post\_ema}}$。
3. **后半身传导锁定**：挂起半周期期间记录后躯干是否发生正负屈曲（$|\tilde{k}_{\text{post\_ema}}| \ge K_{\text{POST\_MIN}} = 0.85$）。只有后半身在全周期内发生了正负双向屈曲传导，才确认记为 1 次全身 S 周期。
4. **推进运动严格门槛**：
   * 窗口内非静止帧占比需 $\ge 35\%$ (`S_MIN_ACTIVE_RATIO`)。
   * 窗口内有效推进方向点积平均需 $\ge 0.25$ (以完整 `S_WIN` 帧数为分母)。
   * 窗口首尾质心净位移需 $\ge 8\%$ 体长 (`S_MIN_NET_DISP_RATIO`)，彻底拦截原地摆动。

```python
def _is_overlapping_or_abnormal(self, body_len):
    if body_len <= 10.0:
        return True
    if len(self.body_lengths) >= 10:
        median_len = float(np.median(self.body_lengths))
        if abs(body_len - median_len) / (median_len + 1e-6) > 0.20:
            return True
    self.body_lengths.append(body_len)
    return False

def _detect_s_motion(self, k_ant, k_post, body_len):
    if k_ant is None or k_post is None or body_len <= 10.0:
        return

    if self._is_overlapping_or_abnormal(body_len):
        self._reset_s_pending()
        return

    # 1. 前后段曲率 EMA 平滑
    alpha = 0.35
    self._k_ant_ema = k_ant if self._k_ant_ema is None else (alpha * k_ant + (1.0 - alpha) * self._k_ant_ema)
    self._k_post_ema = k_post if self._k_post_ema is None else (alpha * k_post + (1.0 - alpha) * self._k_post_ema)

    # 记录后段躯干是否发生双向屈曲传导（排除后半身定死在原地的半截摆头）
    if self._s_pending_phase != 0:
        if self._k_post_ema >= self.K_POST_MIN:
            self._post_seen_pos = True
        elif self._k_post_ema <= -self.K_POST_MIN:
            self._post_seen_neg = True

    # 2. 超时熔断
    if self._s_pending_phase != 0 and (self.frame_counter - self._s_pending_phase_idx > self.S_MAX_CYCLE_GAP):
        self._reset_s_pending()

    # 3. 连续静止熔断（近 S_STILL_TIMEOUT 帧内 >= 80% 为静止即重置，防止偶发1帧摆头绕过静止检测）
    recent_signs = list(self.motion_signs)[-self.S_STILL_TIMEOUT:]
    if len(recent_signs) == self.S_STILL_TIMEOUT:
        zero_ratio = sum(1 for s in recent_signs if s == 0) / float(self.S_STILL_TIMEOUT)
        if zero_ratio >= 0.80:
            self._reset_s_pending()
            return

    if len(self.end_dists) < self.S_WIN or len(self.centers) < self.S_WIN:
        return

    # 4. 舒展度检查（排除 C 型卷曲）
    dists_w = np.array(list(self.end_dists)[-self.S_WIN:])
    if float(np.std(dists_w)) / (float(np.mean(dists_w)) + 1e-9) > self.S_CV_THRESH:
        self._reset_s_pending()
        return

    # 5. 真实推进位移与有效运动帧占比检查（修复除数漏洞）
    window_signs = list(self.motion_signs)[-self.S_WIN:]
    non_zero_signs = [s for s in window_signs if s != 0]
    if len(non_zero_signs) < self.S_WIN * self.S_MIN_ACTIVE_RATIO:
        return
    # 分母使用完整窗口长度 self.S_WIN
    if abs(sum(non_zero_signs) / float(self.S_WIN)) < self.S_DOT_THRESH:
        return

    # 质心净位移门限：窗口首尾质心净移动必须 >= 8% 体长，彻底拦截原地搖头引起的质心微晃
    centers_w = list(self.centers)[-self.S_WIN:]
    net_disp = float(np.linalg.norm(centers_w[-1] - centers_w[0]))
    if net_disp < body_len * self.S_MIN_NET_DISP_RATIO:
        return

    # 6. 双门限迟滞状态机
    if self._s_pending_phase == 1 and self._k_ant_ema < -self.K_ZERO:
        self._s_has_zero_crossed = True
    elif self._s_pending_phase == -1 and self._k_ant_ema > self.K_ZERO:
        self._s_has_zero_crossed = True

    if self._k_ant_ema >= self.K_HIGH:
        if self._s_pending_phase == -1 and self._s_has_zero_crossed:
            if self.frame_counter - self._s_pending_phase_idx >= self.S_PEAK_COOLDOWN:
                self._advance_half_cycle(polarity=1)
        elif self._s_pending_phase == 0:
            self._s_pending_phase = 1
            self._s_pending_phase_idx = self.frame_counter
            self._s_has_zero_crossed = False
            self._s_half_count = 1
            self._post_seen_pos = (self._k_post_ema >= self.K_POST_MIN)
            self._post_seen_neg = (self._k_post_ema <= -self.K_POST_MIN)

    elif self._k_ant_ema <= -self.K_HIGH:
        if self._s_pending_phase == 1 and self._s_has_zero_crossed:
            if self.frame_counter - self._s_pending_phase_idx >= self.S_PEAK_COOLDOWN:
                self._advance_half_cycle(polarity=-1)
        elif self._s_pending_phase == 0:
            self._s_pending_phase = -1
            self._s_pending_phase_idx = self.frame_counter
            self._s_has_zero_crossed = False
            self._s_half_count = 1
            self._post_seen_pos = (self._k_post_ema >= self.K_POST_MIN)
            self._post_seen_neg = (self._k_post_ema <= -self.K_POST_MIN)

def _advance_half_cycle(self, polarity):
    self._s_half_count += 1
    self._s_pending_phase = polarity
    self._s_pending_phase_idx = self.frame_counter
    self._s_has_zero_crossed = False

    if self._s_half_count >= 2:
        # 只有后半段躯干在周期内也发生了正负双向屈曲（证明S波真正向尾部传导），才确认为1次全身S周期
        if self._post_seen_pos and self._post_seen_neg:
            self.s_count += 1
            valid_signs = [s for s in list(self.motion_signs)[-self.S_WIN:] if s != 0]
            if valid_signs:
                dir_sign = 1 if (sum(valid_signs) / len(valid_signs)) > 0 else -1
                self._s_dirs_seen.add(dir_sign)
                if len(self._s_dirs_seen) > 1:
                    self.is_bidirectional = True
            self._s_half_count = 0
            self._post_seen_pos = (self._k_post_ema >= self.K_POST_MIN)
            self._post_seen_neg = (self._k_post_ema <= -self.K_POST_MIN)
        else:
            # 若后半段躯干未发生反转（半身原地摆动），保留当前最新半波相位，不增加 s_count
            self._s_half_count = 1
```

---

## 五、 核心参数与调优索引字典 (Hyperparameter Tuning Reference)

所有参数均定义于 `TrackHistory` 类顶部，可直接调整：

| 模块 | 参数变量名 | 默认值 | 物理/生物学意义与调参指南 |
| :--- | :--- | :---: | :--- |
| **基础运动** | `MOVE_THRESH` | `2.0 px` | **有效位移门限**：质心移动小于此值直接判定为静止，杜绝端点微抖动误判为前进航向。 |
| | `DIR_THRESH` | `0.15` | **航向投影死区**：瞬时航向与主航向夹角 $|\cos\theta| < 0.15$ 判定为横向漂移，不计入有效推进。 |
| **C型动作** | `C_WIN` | `20` 采样点 | 滑动评估窗口大小（约 2.0 秒）。 |
| | `C_CV_THRESH` | `0.12` | 端点距离变异系数阈值。身体大幅卷折时头尾距离剧烈收缩，调大可提高 C 型判定的严格程度。 |
| | `C_TORTUOSITY`| `1.4` | 轨迹迂曲比（$\text{Path} / \text{Displacement}$）。大于 1.4 说明盘旋回折，小于此值偏向直线运动。 |
| | `C_MIN_MOVE` | `3.0 px` | 窗口内位移下限，防止死虫或发呆样本因光斑噪点误报。 |
| | `C_CROSSES_PER_COUNT`| `2` | 进出原点带分界线次数。2 次（一出一进）记为 1 次动作。 |
| **S型动作 (传导双门限版)** | `K_HIGH` | `2.1` | **前段迟滞高门限**：体长归一化躯干曲率绝对值阈值，必须超过此值才算有效波峰。 |
| | `K_ZERO` | `0.55` | **前段中立过零门限**：两半波转换间必须跨入中立反向过零区，确保身体前段回直。 |
| | `K_POST_MIN` | `0.85` | **后段伴随传导门限**：后躯干归一化曲率必须达到此值，证明波动向后传播而非前身孤立摆头。 |
| | `S_PEAK_COOLDOWN`| `12` 帧 | 两次半周期翻转的最小帧间隔，杜绝高频假波动。 |
| | `S_MIN_ACTIVE_RATIO`| `0.35` | 窗口内非静止采样帧的最低比例（35%），防止偶尔1帧摆动借除数漏洞蒙混过关。 |
| | `S_MIN_NET_DISP_RATIO`| `0.08` | 窗口首尾质心净位移占体长比例下限（8%），彻底过滤原地摆头。 |
| | `S_CV_THRESH` | `0.14` | **舒展度硬门槛**：CV 必须 $< 0.14$，确保身体充分展开。若超出立即熔断丢弃半周期。 |
| | `S_DOT_THRESH` | `0.25` | **前进推进门槛**：窗口内平均方向点积下限（以整窗为分母），排除原地静止微颤。 |
| | `S_MAX_CYCLE_GAP`| `50` (~5.0s)| **超时熔断**：单侧半周期挂起超过 5 秒未闭合，判定波形断开，果断丢弃！严禁跨时段拼凑。 |
| | `S_STILL_TIMEOUT`| `15` (~1.5s)| **停顿熔断**：近 15 帧内静止帧比例 $\ge 80\%$ 立即重置挂起半波。 |
| | `体长突变阈值` | `20%` | **重叠/粘连熔断**：当前体长相对于近 10 帧中位数变化超过 20% 判定为交叉或粘连，冻结计数。 |
| | `主干分支比例` | `> max(8, 0.18*L)`| **骨架粘连熔断**：非最长主干骨架像素超出主干长度 18% 时判定发生虫体交叉，丢弃该中心线。 |

---

## 六、 总结与重构成果

1. **骨架粘连精准熔断**：用最长路径外分支像素比例（$>18\%$）取代原端点数粗暴过滤，既杜绝了边缘毛刺造成的断帧漏检，又彻底锁死了真交叠虫体的虚假中心线。
2. **多模式自适应朝向对齐**：统一 `orient_pose_pair` 接口，无缝兼容 `(11, 2)` 全序列比对与 `(2,)` 头部锚点，保证追踪器时序头尾绝对一致。
3. **躯干波形传导验证与除数修复**：同时检测前中段与中后段曲率，要求必须发生波后传导才承认 S 周期；以完整窗口长度归一推进比并增设质心净位移门限，彻底消灭停滞期原地摆动造成的过计数。
4. **骨架分叉检测与体长变异熔断**实现了轻量级重叠/粘连静默锁，杜绝两条线虫接触时曲率跳变产生的误计数。
