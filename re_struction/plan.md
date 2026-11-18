# 08 — S形与C形运动精确计数修复规范 (Harness AI 实施指南)

> **目标读者**：接手 `re_struction/` 代码库的 AI 编码代理（Harness Agent）。
> **背景说明**：当前 `re_struction` 分支的整体架构（Pass A / Pass B 分离、33点等弧长中心线、三角基投影 $a_1, a_2$）已成型，勿推倒重来。但当前存在 **C形运动（游泳摆动 + 琼脂深弯）几乎失效**、**S形正弦运动过于保守（大量漏计）** 以及 **液体游泳小虫漏检** 三大瓶颈。
> **任务要求**：请严格按照本文档列出的 **7 个原子任务（Task 1 ~ Task 7）** 依次修改指定文件，不得破坏现有 API 契约（`API_CONTRACT.md`），修改后必须通过单元测试与 `tools/eval_benchmark.py` 验证。

---

## 一、 核心根因清单（供 AI 理解上下文，勿跳过）

1. **配置缺失与质量误杀（`swim_liquid.yaml` & `quality.py` & `pass_a.py`）**：
   - `crawl_agar.yaml` 配置了 `detection.tile`（640切片推理）和 `tracking.recovery`（含静态背景差分 `bg_sub_enabled`），但最需要它们的 `swim_liquid.yaml` 完全漏配，导致 `Swim-4.mp4` 等 1080p 视频中的小虫在全图缩至 `416` 时直接漏检（输出 0）。
   - `pass_a.py` 第 266 行调用 `ext.process(det, frame_idx=cur_idx)` 时漏传了 `frame_gray`，导致 `mask_refine` 永远无法生效。
   - `quality.py` 中硬编码 `pose.length < 3.0 * pose.width` 判为 `Quality.BAD`。当小虫或 C 形深弯线虫在 `imgsz=416` 下掩码偏胖时，长宽比常落在 `2.0 ~ 2.9` 之间，导致弯曲顶点帧被直接当成 `BAD` 丢弃。
2. **琼脂 C/Ω 弯计数器（`turns.py`）的几何数学错误与无超时死锁 Bug**：
   - **数学错误**：均匀圆弧的弦长比 $R = \frac{\text{chord}}{L} = \frac{2\sin(\Delta\theta/2)}{\Delta\theta}$。$90^\circ$ C 弯的 $R \approx 0.90$，$180^\circ$ 标准半圆 C 弯的 $R = 2/\pi \approx 0.637$。而 `crawl_agar.yaml` 将 `R_enter` 设为 `0.50`（要求弯曲超过 $217^\circ$ 才能触发），导致所有标准 C 形弯曲 100% 无法触发 `TurnCounter`。
   - **永久死锁 S 计数的 Bug**：`turns.py` 中进入 `state == "CURL"` 后没有任何超时释放机制！一旦某帧因掩码缺角或弯曲使 $R \le R_{\text{enter}}$，而后续线虫以较大波幅爬行（常态 $R \in [0.52, 0.67] < R_{\text{exit}}(0.68)$），`self.turn.in_curl` 将在整段视频剩余时间内**永久保持 `True`**，`counter.py:204` 每帧调用 `self.s.force_abort("curl_or_bad")`，彻底锁死该虫后续所有 S 计数！
3. **液体游泳 C 摆动（`modes.py` & `swing.py`）的头尾翻转假计数与偏置漏计**：
   - 原 `compute_total_turn_c` 仅取首尾 20% 切线角差 `mean(theta[-6:]) - mean(theta[:6])`，丢弃了躯干中段 60% 的信息，且极易受首尾圆帽噪声干扰。
   - **头尾翻转 1 帧即触发假计数**：当头尾点序反转（`dense[::-1]`）时，$c(t)$ 的正负号瞬间反转（如 $+1.4 \to -1.4$）。由于对称 C 形去质心后正反序列极度接近，`orient()` 一旦在某帧翻转头尾，`SwingCounter` 会立即误判为一次反向摆动。
   - **固定双侧硬阈值漏计偏置摆动**：原 `SwingCounter` 要求必须交替跨过 `+bend_high` 和 `-bend_high`。若虫体存在姿态偏置（如左弯 $+1.3$、右回弹仅到 $-0.5$），将永远无法触发另一侧阈值，导致计数归零。
4. **S 形正弦运动（`wave.py` & `s_runs.py` & `counter.py`）过保守的 4 个卡点**：
   - **滑窗中位数陷阱**：`wave_gate` 在 `2.5s~3.0s` 窗口内对振幅取中位数 `np.median(r)`。对于只爬行 1~2 个周期（约 1.2s）就停下的线虫，窗口内静止帧超过 50% 时中位数直接归零，掐断有效相位。
   - **中途停顿抹零**：`counter.py:397` 的尾数补全 `eof_tail_frac = 0.68` **仅在整段视频最后一帧（EOF）检查 `eng.s.active`**。若线虫在视频中途爬了 1.8 个周期后停下休息，`slowdown_hold_s` 到期会调用 `force_abort("gate_lost")` 将未满 $2\pi$ 的 0.8 周期直接清零（1.8 周期变成 1 周期，0.8 周期变成 0）。
   - **单周期位移门槛过高**：`crawl_agar.yaml` 中 `min_disp_ratio: 0.18`（要求每周期质心必须推进 18% 体长），导致湿润琼脂表面打滑或慢速爬行（每周期推进 7%~14% 体长）的真实 S 波被 `rejected["no_disp"]` 拒掉。
   - **头尾 50% 盲猜导致 `rejected["backward"]` 误杀**：无人工逐虫标注头部时，正向爬行但初始头尾标反的线虫会出现 $\Delta\Phi < 0$ 且位移投影 $p < 0$（两者同号，符合正向行波物理规律，但符号为负）。原逻辑在 `preserve_first_head=True`（前端默认传 `head_kpt_idx=0` 时）或 `eval_benchmark.py`（未调用 `run_postprocessed`）中将其全部当成 `backward` 拒掉。

---

## 二、 逐文件修改规范（Task 1 ~ Task 7）

### Task 1：修复姿态质量长宽比硬编码与 `PassA` 灰度图透传

**目标文件 1**：`re_struction/backend/wormlab/config.py`
- 在 `PoseConfig` dataclass 中新增字段：
  ```python
  min_len_over_width: float = 2.2  # 原 quality.py 硬编码为 3.0，放宽以兼容 C 形弯曲与低倍镜小虫
  ```

**目标文件 2**：`re_struction/backend/wormlab/pose/quality.py`
- 在 `PoseQualityJudge.__init__` 中读取 `self.min_len_over_width = float(getattr(cfg, "min_len_over_width", 2.2))`。
- 将第 49 行的硬编码 `3.0` 替换为配置值：
  ```python
  if pose.length < self.min_len_over_width * max(pose.width, 1e-6):
      bad = True
  ```

**目标文件 3**：`re_struction/backend/wormlab/pipeline/pass_a.py`
- 在 `PassA._extract_poses` 及 `_extract_one`（约第 259~278 行）中，接收当前帧灰度图 `frame_gray: np.ndarray | None` 并传给 `ext.process(det, frame_gray=frame_gray, frame_idx=cur_idx)`。
- 在 `PassA.run` 主循环中，若 `self.cfg.pose.mask_refine.enabled` 或 `bg_sub_enabled` 开启，将 `gray` 传入 `self._extract_poses(dets, cur_idx, frame_gray=gray)`。

---

### Task 2：补齐 `swim_liquid.yaml` 并调整 `crawl_agar.yaml` 过严参数

**目标文件 1**：`re_struction/backend/config/profiles/swim_liquid.yaml`
- 替换为以下完整配置（补齐切片推理、背景差分找回、小虫掩码下限）：
  ```yaml
  # 液体游泳 profile（C 运动计数主场景）
  detection:
    confidence: 0.20
    recovery_confidence: 0.10
    tile:
      auto: true
      size: 640
      overlap: 0.25
      infer_imgsz: 416

  tracking:
    damp_lost_velocity: true
    recovery:
      enabled: true
      roi_size: 416
      roi_confidence: 0.10
      max_Window_frames: 90
      bg_sub_enabled: true
      bg_sample_frames: 25
      bg_thresholds: [12, 8, 5]

  pose:
    min_mask_px: 15
    min_len_over_width: 1.8
    upscale_below_px: 120
    upscale_factor: 2

  kinematics:
    s_run:
      enabled: false
    turn:
      enabled: false
    swing:
      enabled: true
      bend_high: 0.75
      min_flip_interval_s: 0.08
      count_unit: half_cycle
  ```

**目标文件 2**：`re_struction/backend/config/profiles/crawl_agar.yaml`
- 修改 `pose` 与 `kinematics` 部分参数，解除过严门限：
  ```yaml
  pose:
    coil_aware_reorient: true
    min_len_over_width: 2.2

  kinematics:
    k_body: 1.15
    dc_remove_window_s: 0.0   # 关闭动态跳变的 DC 扣除，避免相平面突变
    eof_tail_frac: 0.65
    wave:
      win_s: 1.8              # 从 2.5s 缩短为 1.8s，防止短促单周期被静止帧稀释
      r_min: 0.12
      eps_min: 0.15
      post_ratio_min: 0.30
      slowdown_hold_s: 3.0
    s_run:
      enabled: true
      dir_lock_rad: 0.55      # 从 0.80 降至 0.55
      min_disp_ratio: 0.08    # ★ 从 0.18 降至 0.08，修复打滑/慢爬 S 周期被 no_disp 误杀
      rev_tol_rad: 0.45
    turn:
      enabled: true
      R_enter: 0.65           # ★ 从 0.50 提至 0.65（对应 ~165° C 弯，原 0.50 要求 >217° 死弯）
      R_exit: 0.78            # 从 0.68 提至 0.78
      min_dwell_s: 0.15
      cooldown_s: 0.45
    swing:
      enabled: false
  ```

---

### Task 3：修复 `TurnCounter`（`turns.py` & `counter.py`）的永久死锁与 C 弯判定

**目标文件 1**：`re_struction/backend/wormlab/config.py`
- 在 `TurnCfg` 中增加超时保护与最小弯曲模态门槛：
  ```python
  max_curl_s: float = 2.0      # CURL 状态最大驻留秒数，超时自动释放回 EXT，防止永久锁死 S 计数
  min_peak_c_rad: float = 0.85 # CURL 期间需至少达到的 |c| 弯曲度，区分真实 C 弯与掩码残缺缩短
  ```

**目标文件 2**：`re_struction/backend/wormlab/kinematics/turns.py`
- 在 `TurnCounter.__init__` 中增加：
  ```python
  self.max_curl_frames = max(10, int(round(float(getattr(cfg, "max_curl_s", 2.0)) * fps)))
  self.min_peak_c_rad = float(getattr(cfg, "min_peak_c_rad", 0.85))
  self.peak_c_in_curl = 0.0
  ```
- 修改 `TurnCounter.step(self, i: int, r_val: float, valid: bool, c_val: float = 0.0)`（为兼容旧测试调用，`c_val` 默认设为 `0.0`，若未传则不卡 `min_peak_c_rad`）：
  ```python
  def step(self, i: int, r_val: float, valid: bool, c_val: float | None = None) -> None:
      if not self.cfg.enabled:
          return
      if not valid or not np.isfinite(r_val):
          self.invalid_run += 1
          if self.invalid_run > self.gap_hold_frames:
              self.state = "EXT"
              self.peak_c_in_curl = 0.0
          return
      self.invalid_run = 0
      r_enter = float(self.cfg.R_enter)
      r_exit = float(self.cfg.R_exit)
      c_abs = abs(float(c_val)) if (c_val is not None and np.isfinite(c_val)) else 999.0

      if self.state == "EXT":
          if r_val <= r_enter:
              self.state = "CURL"
              self.curl_start = int(i)
              self.peak_c_in_curl = c_abs
      elif self.state == "CURL":
          self.peak_c_in_curl = max(self.peak_c_in_curl, c_abs)
          dwell = int(i) - self.curl_start
          # ★ 修复 1：若在 CURL 中驻留超过 max_curl_frames 仍未恢复，强制释放回 EXT，绝不永久锁死 S 运动
          if dwell > self.max_curl_frames:
              if dwell >= self.min_dwell_frames and self.peak_c_in_curl >= self.min_peak_c_rad:
                  if (int(i) - self.last_turn) >= self.cooldown_frames:
                      self.count += 1
                      self.last_turn = int(i)
                      self.events.append(TurnEvent(start_frame=self.curl_start, end_frame=int(i)))
              self.state = "EXT"
              self.peak_c_in_curl = 0.0
              return

          if r_val >= r_exit:
              if dwell >= self.min_dwell_frames and self.peak_c_in_curl >= self.min_peak_c_rad:
                  if (int(i) - self.last_turn) >= self.cooldown_frames:
                      self.count += 1
                      self.last_turn = int(i)
                      self.events.append(TurnEvent(start_frame=self.curl_start, end_frame=int(i)))
              self.state = "EXT"
              self.peak_c_in_curl = 0.0
  ```

**目标文件 3**：`re_struction/backend/wormlab/kinematics/counter.py`
- 第 198 行调用 `self.turn.step(i, feat.R, True, c_val=feat.c)`。
- 第 202~205 行修改：当 `self.turn.in_curl` 为真时，仅当 `feat.R <= 0.52`（真正深度盘绕自交）时才执行 `self.s.force_abort("curl_or_bad")`，否则仅执行 `self.s.freeze()`！这样普通弧线爬行或中等弯曲不会把正在累积的 S 相位清零。

---

### Task 4：升级 C 模态计算（`modes.py`）与抗翻转峰谷突出度游泳计数器（`swing.py`）

**目标文件 1**：`re_struction/backend/wormlab/kinematics/modes.py`
- 修改 `compute_total_turn_c(theta: np.ndarray) -> float`：
  将原来的首尾 20% 端点差改为 **50% 全躯干一阶对称余弦模态 $\cos(\pi s)$ + 50% 内缩躯干段 (`[10%..30%]` vs `[70%..90%]`) 混合积分**，彻底避开最外侧 10% (`[:3]` 和 `[-3:]`) 的掩码圆帽噪声，同时保持与原测试量纲一致：
  ```python
  def compute_total_turn_c(theta: np.ndarray) -> float:
      """
      计算鲁棒 C 弯曲模态 (rad)：
      结合全躯干 cos(πs) 对称模态（等价于对曲率按 sin(πs) 加权积分，中心权重最大、两端为 0）
      与去尖端首尾段差，彻底消除掩码两端圆帽毛刺。
      """
      n = len(theta)
      if n < 4:
          return 0.0
      th_zero = theta - float(np.mean(theta))
      s_mid = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
      # 半圆弧 Δθ = θ_tail - θ_head 时，π * mean(th_zero * cos(πs)) 与首尾差同号同量纲
      c_cos = float(-np.pi * np.mean(th_zero * np.cos(np.pi * s_mid)))

      # 内缩首尾段（避开首尾各 6% 的端帽噪声点）
      pad = max(1, int(round(0.06 * n)))
      m = max(1, int(round(0.20 * n)))
      head_seg = theta[pad : pad + m]
      tail_seg = theta[max(pad, n - pad - m) : n - pad]
      c_end = float(np.mean(tail_seg) - np.mean(head_seg))
      return 0.65 * c_cos + 0.35 * c_end
  ```

**目标文件 2**：`re_struction/backend/wormlab/kinematics/swing.py`
- 升级 `SwingCounter`，支持：
  1. **方向向量点积符号锁定（`u_curr`）**：若传入了当前帧首尾单位向量 `u_curr`，当 `np.dot(u_curr, self.u_ref) < 0`（说明 `orient()` 在该帧发生了头尾 $180^\circ$ 翻转）时，自动将 `c` 取反（`c = -c`），彻底杜绝单帧头尾翻转引发的假计数！
  2. **双模式触发（绝对阈值迟滞 OR 峰谷相对突出度 Prominence）**：既保留原有的 `±bend_high` 跨零翻转（100% 兼容现有单元测试 `test_swing.py`），又增加峰谷突出度检测——当虫体存在姿态偏置（如在一侧摆动到 $+1.2\text{ rad}$，回摆到 $-0.3\text{ rad}$ 未达 $-0.75$，但回摆幅度 $\Delta c = 1.5 \ge 1.6 \times \text{bend\_high}$ 且跨过近期均值基线）时，同样计入一次有效摆动！
  ```python
  class SwingCounter:
      """Mode C1：液体游泳 C 摆动计数器（含头尾翻转免疫 + 峰谷突出度自适应）。"""

      def __init__(self, cfg: SwingCfg, fps: float, gap_hold_s: float = 0.30) -> None:
          self.cfg = cfg
          self.fps = float(fps)
          self.min_flip_frames = max(1, int(round(cfg.min_flip_interval_s * fps)))
          self.gap_hold_frames = max(1, int(round(gap_hold_s * fps)))
          self.side: int = 0
          self.entries: int = 0
          self.last_flip: int = -(10**9)
          self.invalid_run: int = 0
          self.extreme: float = 0.0
          self.u_ref: np.ndarray | None = None
          self._recent_c: list[float] = []

      def step(self, i: int, c: float, valid: bool, u_curr: np.ndarray | None = None) -> None:
          if not self.cfg.enabled:
              return
          if not valid or not np.isfinite(c):
              self.invalid_run += 1
              if self.invalid_run > self.gap_hold_frames:
                  self.side = 0
                  self.u_ref = None
              return
          self.invalid_run = 0

          c_val = float(c)
          # ★ 修复 1：头尾 180° 翻转符号保护
          if u_curr is not None and len(u_curr) == 2 and np.all(np.isfinite(u_curr)):
              u_norm = float(np.linalg.norm(u_curr))
              if u_norm > 1e-6:
                  u_unit = u_curr / u_norm
                  if self.u_ref is not None:
                      if float(np.dot(u_unit, self.u_ref)) < -0.15:
                          c_val = -c_val
                          u_unit = -u_unit
                      self.u_ref = 0.85 * self.u_ref + 0.15 * u_unit
                      self.u_ref /= max(float(np.linalg.norm(self.u_ref)), 1e-9)
                  else:
                      self.u_ref = u_unit.copy()

          self._recent_c.append(c_val)
          if len(self._recent_c) > int(round(2.0 * self.fps)):
              self._recent_c.pop(0)
          c_bias = float(np.median(self._recent_c)) if len(self._recent_c) >= 10 else 0.0

          thr = float(self.cfg.bend_high)
          prom_thr = 1.55 * thr  # 峰谷突出度阈值

          # 标准对称迟滞判定
          new_side = 1 if c_val >= thr else (-1 if c_val <= -thr else 0)

          # 追踪当前侧极值，支持偏置姿态下的峰谷突出度翻转
          if self.side == 1:
              self.extreme = max(self.extreme, c_val)
              if new_side == 0 and (self.extreme - c_val) >= prom_thr and (c_val - c_bias) <= -0.35 * thr:
                  new_side = -1
          elif self.side == -1:
              self.extreme = min(self.extreme, c_val)
              if new_side == 0 and (c_val - self.extreme) >= prom_thr and (c_val - c_bias) >= 0.35 * thr:
                  new_side = 1

          if new_side == 0 or new_side == self.side:
              return
          if (int(i) - self.last_flip) < self.min_flip_frames:
              return
          self.side = new_side
          self.extreme = c_val
          self.entries += 1
          self.last_flip = int(i)
  ```
- 在 `counter.py` 第 199 行调用时传入 `feat.u`：`self.swing.step(i, feat.c, True, u_curr=feat.u)`。

---

### Task 5：修复 `wave_gate`（`wave.py`）短促爬行被静止帧中位数稀释的问题

**目标文件**：`re_struction/backend/wormlab/kinematics/wave.py`
- 在 `wave_gate`（第 40~42 行）中，当窗口内存在部分静止帧时，`np.median(r)` 会因为静止帧刚巧过半而直接跌破 `r_min`。
- 将 `r_med` 的计算改为 `max(float(np.median(r)), float(np.percentile(r, 65)) * 0.85)`（即兼顾中位数与窗口内前 35% 活跃帧的波幅），使持续 0.8s~1.2s 的单次 S 波在进入和离开滑窗时不会被前后静止帧过早掐断。

---

### Task 6：修复 `SRunCounter`（`s_runs.py` & `counter.py`）中途停顿抹零问题

**目标文件 1**：`re_struction/backend/wormlab/kinematics/s_runs.py`
- 在 `SRunCounter` 中新增方法 `settle_partial(self, i: int, L_ref: float, tail_frac: float = 0.65) -> bool`：
  ```python
  def settle_partial(self, i: int, L_ref: float, tail_frac: float = 0.65) -> bool:
      """
      当一次正向爬行因停顿 (gate_lost) 或视频结束 (EOF) 终止时，
      若当前未完成周期已累积 >= tail_frac * 2π (如 65% 周期) 且顺向推进位移按比例达标，
      则结算为 +1 次有效 S 周期，避免 1.7~1.9 周期被硬截断为 1。
      """
      if not self.active or self.direction <= 0 or self.last_center is None:
          return False
      rem_frac = (self.prog - self.anchor_prog) / (2.0 * np.pi)
      if rem_frac < float(tail_frac):
          return False
      axis_cur = self._unit(self.axis_sum)
      p_cur = float(np.dot(self.last_center - self.anchor_center, axis_cur))
      req_disp = float(self.cfg.min_disp_ratio) * float(tail_frac) * float(L_ref)
      if p_cur >= req_disp:
          self.s_count += 1
          self.events.append(SEvent(frame_idx=int(i), direction=+1, disp_px=p_cur, reason="tail_settled"))
          self.anchor_prog += 2.0 * np.pi
          return True
      return False
  ```

**目标文件 2**：`re_struction/backend/wormlab/kinematics/counter.py`
- 在 `_step_frame` 的第 214~216 行（当 `self._pause_run > self._slowdown_hold_frames` 触发 `self.s.force_abort("gate_lost")` **之前**），先调用尾数结算：
  ```python
  if self._pause_run > self._slowdown_hold_frames:
      tail_frac = getattr(self.cfg, "eof_tail_frac", None)
      if tail_frac is not None:
          self.s.settle_partial(i, self.L_ref, tail_frac=float(tail_frac))
      self.s.force_abort("gate_lost")
  ```
- 在 `finalize()` / `run_postprocessed()` 末尾也直接复用 `eng.s.settle_partial(...)`。

---

### Task 7：修复头尾 50% 盲猜导致的 `backward` 误杀（`pass_b.py` & `eval_benchmark.py`）

**目标文件 1**：`re_struction/backend/wormlab/pose/orientation.py`
- 在 `reorient_track_poses_coil_aware`（第 127~143 行）中，当 `preserve_first_head=True` 时，原代码无条件跳过 `seg_idx == 0` 的翻转。
- 但如果用户在前端并未手动修正头部，或者初始检测框匹配存在偏差，第一段可能会出现 **所有周期全部落入 `rejected["backward"] >= 1` 且 `s_count == 0` 且净位移与相位同为负（即典型的“正向爬行但头尾标反”）**。
- 修改逻辑：在 `reorient_track_poses_coil_aware` 中，即便 `preserve_first_head=True`，若第 0 段出现 `eng_s.s.s_count == 0 and eng_s.s.rejected.get("backward", 0) >= 1 and score < -0.25`，也允许自动翻转纠正头尾！

**目标文件 2**：`re_struction/tools/eval_benchmark.py`
- 检查第 177~179 行（`evaluate_video` 函数内）：
  原代码写的是：
  ```python
  poses_seq = reorient_track_poses(poses_seq)
  res = KinematicsEngine.run(poses_seq, cfg=cfg.kinematics, fps=fps)
  ```
  这与 `PassB.run()` 中的真实生产管线不一致（漏掉了 `reorient_track_poses_coil_aware` 和 `run_postprocessed`）！
- 请将其对齐为与 `pass_b.py` 第 104~126 行完全一致的调用：
  ```python
  coil_aware = bool(getattr(cfg.pose, "coil_aware_reorient", False) and cfg.kinematics.s_run.enabled)
  if coil_aware:
      poses_seq = reorient_track_poses_coil_aware(
          poses_seq,
          cfg=cfg.kinematics,
          fps=fps,
          preserve_first_head=has_manual_head,
      )
      res = KinematicsEngine.run_postprocessed(poses_seq, cfg=cfg.kinematics, fps=fps)
  else:
      poses_seq = reorient_track_poses(poses_seq)
      res = KinematicsEngine.run(poses_seq, cfg=cfg.kinematics, fps=fps)
  ```

---

## 三、 验证与回归测试指令（AI 执行完毕后必须跑通）

1. **运行全量单元测试与合成基准测试**：
   ```bash
   cd re_struction/backend
   pytest -q
   ```
   *要求*：现有的 25 个测试文件全部 `PASSED`（若因参数微调导致个别合成阈值断言变化，需检查是否合理）。
2. **运行真实视频基准评测（秒级缓存评测）**：
   ```bash
   cd re_struction
   python tools/eval_benchmark.py
   ```
   *观察指标*：对比修改前后的 `overall_mae`、`within_pm1` 以及 `Swim-1/3/4`、`10x`、`1.6x-1`、`1.6x-3` 各虫的预测值与真值差异。