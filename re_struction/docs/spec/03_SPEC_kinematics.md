# 03 运动学层（姿态序列 → S / C 计数）

对应任务：T11–T15。这是本项目**最关键的算法**，请严格按规格实现；参数全部走配置（§10）。

## 0. 业务定义（验收口径，务必先读；均为甲方确认）
- **一次 S** = 全身完成**一个完整正弦波（上-下-上）**，波传播方向一致，**并朝该方向发生净位移**。只有头部轻微上下（试探/寻路）→ **不计**。
- **只统计向前 S**。向后爬行产生的完整周期**完全不统计**（只在诊断字段 `rejected.backward` 里记数，不进 `s_count`、不进频率）。
- **反向运动的处理**：当 **本轮(当前周期)相位进度回退到为负，且质心沿体轴实际后退** 两个条件**同时**成立 → 视为反向运动 → **作废当前未完成的半截周期**，从该点重新开始；**已完成并已计入的 S 保留**。
- **一次 C**：同时输出**两种口径**，由甲方/用户选择：
  - `c_cycles`（周期）：左弯 + 右弯完成一个完整来回 = 1 次；
  - `c_bends`（单次弯曲）：左弯、右弯**各算 1 次**（每次进入一侧高位弯曲计 1 次）。
  对外的 `c_count` = 所选口径（`c_unit=cycle|bend`，默认 `cycle`）+ 深弯折返次数；两种口径的数字**始终都输出**。
- **后进入画面的虫一律不计数**（判定见 05 §4：首次出现帧 > `counting.late_entry_grace_s`）。
- 爬行视频验收看 S；游泳视频验收看 C（两种口径都报告）。

## 1. 总体结构与"延迟因果"
```python
# wormlab/kinematics/counter.py
class KinematicsEngine:
    def __init__(self, cfg: KinCfg, fps: float): ...
    def push(self, frame_idx: int, pose: Pose | None) -> None:
        """每个存活帧调用一次（轨迹丢失/幽灵帧传 None）。frame_idx 必须递增 1。"""
    def snapshot(self) -> LiveCounts:
        """已处理帧范围内的累计计数（比 push 滞后 D 帧）。供实时画面用。"""
    def finalize(self) -> tuple[TrackReport, PerFrame]:
        """冲刷尾部帧（窗口截断）后返回终稿。"""
    @staticmethod
    def run(poses: list[Pose|None], fps, cfg) -> tuple[TrackReport, PerFrame]:
        e = KinematicsEngine(cfg, fps); [e.push(i,p) for i,p in enumerate(poses)]; return e.finalize()
```
- **同一套代码**服务实时与终稿：引擎保存全部历史（每轨迹至多几千帧，内存可忽略），`push` 之后每当 `n_pushed - 1 >= i + D` 就处理第 `i` 帧；`finalize` 处理剩余帧（窗口取可用样本）。
- 延迟 `D = ceil((wave.win_s/2 + smooth_s/2) * fps)` 帧。这样第 `i` 帧的平滑与行波门限用的是**以 i 为中心**的窗口（"零相位"），实时与终稿结果**逐位一致**（测试要求）。
- `PerFrame`：每帧 `behavior, s_count, c_count, wave_ok, phi, c, R, quality`，长度 = push 次数；Pass B 渲染用。

## 2. 逐帧特征（`features.py`）
输入：平滑后的 `dense (33,2)`（或 None=无效）。
```python
def compute_features(dense, cfg) -> Feat | None:
    d = np.diff(dense, axis=0); seg = np.linalg.norm(d, axis=1)
    L = seg.sum(); chord = norm(dense[0]-dense[-1]); R = clip(chord / L, 0, 1)
    theta = np.unwrap(np.arctan2(d[:,1], d[:,0]))            # (32,) 沿 s 展开
    th = theta - theta.mean()                                  # θ̃，去整体朝向
    s_mid = (np.arange(32) + 0.5) / 32
    a1 = 2*np.mean(th*np.cos(2*np.pi*cfg.k_body*s_mid)); a2 = 2*np.mean(th*np.sin(2*np.pi*cfg.k_body*s_mid))
    m = round(0.2*32)                                          # =6
    c = theta[-m:].mean() - theta[:m].mean()                   # 总转角(rad)，有符号；C 形大，一波长 S 形≈0
    u = unit(dense[0] - dense[-1])                             # 指向头
    center = dense.mean(axis=0)
    return Feat(L, chord, R, theta, th, a1, a2, c, u, center)    # theta=未去均值的切线角，th=θ̃
```
> **`c` 的性质（已数值验证）**：`c` 是首尾 20% 切线角之差，C 形时 `|c|` 很大（半圆弧 ≈ 2.55 rad）；但一个波长的 S 波（A=0.8）也会给出 `|c|≈0.84` 并随波以同频率振荡。所以 `c` **不区分** C 与 S，只度量"首尾朝向差"。这对游泳是对的（游泳时每个来回对应 `c` 一个完整振荡），**爬行 profile 不使用 `c` 计数**（swing 关闭）。
**符号约定验证（单测必须覆盖）**：`θ(s,t)=A·sin(2π s − ωt)`（s 头→尾，波头→尾传播=向前）→ `(a1,a2)=A·(−sin ωt, cos ωt)`，`Φ=atan2(a2,a1)` **随 t 增大**。图像 y 向下的镜像不改变 Φ 的增减方向（`(a1,a2)→(−a1,−a2)` 是旋转）。头尾互换：Φ 增减方向与位移方向同时反号。
单测：解析波形 `A=0.8, k=1` 时 `|a|≈0.8`；直线 `a≈0, c≈0`；半圆弧 `|c|≈π`，`R≈2/π`。

## 3. 行波门限与模态（`wave.py`, `modes.py`）
在以 `i` 为中心、宽 `W=wave.win_s` 的窗口内（需 ≥ `min_valid_frac`=0.6 的有效帧，否则门限=False）：
```python
def wave_gate(window_feats, cfg) -> GateResult:
    A = np.array([[f.a1, f.a2] for f in window_feats])          # (n,2)
    r_med = np.median(np.hypot(A[:,0], A[:,1]))
    ev = np.linalg.eigvalsh(np.cov(A.T)); eps = sqrt(max(ev[0],0) / max(ev[1],1e-12))   # 圆度 ε∈[0,1]
    # ★ 注意：这里不能用 f.th（已减 s 均值）。头部局部摆动会通过"减均值"污染后半身，
    #   已数值验证：用均值时头摆的 amp_post/amp_ant≈0.39（会误过门限），用中位数则≈0。
    TH = np.stack([f.theta - np.median(f.theta) for f in window_feats])   # (n,32) 每帧减沿 s 的中位数
    TH = TH - TH.mean(axis=0)                                    # 再减每段的窗口时间均值
    amp_ant  = sqrt(mean(var(TH[:, :16], axis=0)))               # 各段时间标准差的均方根
    amp_post = sqrt(mean(var(TH[:, 16:], axis=0)))
    ok = (r_med >= cfg.r_min and eps >= cfg.eps_min
          and amp_post >= cfg.post_ratio_min * amp_ant and amp_post >= cfg.amp_post_min)
    return GateResult(ok, eps, r_med, amp_ant, amp_post)
```
- `amp_post` 条件就是"**排除只有头部试探**"：头部摆动时 `amp_ant` 大、`amp_post` 几乎为 0。
- `modes.py` 只放 a1/a2/c 的公式与 `phase = atan2(a2,a1)`；保留 `eigenworm_basis` 的接口（`NotImplementedError`），D4' 后续再用。
单测：①圆形轨迹 → ε≈1；②线性往复 → ε≈0；③头部摆动序列 → ok=False；④全身行波 → ok=True；⑤噪声幅度 < r_min → False。

## 4. S 计数：单调 run + 位移校验 + 反向清零（`s_runs.py`）
```python
class SRunCounter:
    # 状态
    active=False; direction=0                     # 0 未锁定 / +1 / -1
    phi_prev=None; prog=0.0                       # prog: 沿 direction 的累计相位（未锁定时为有符号累计）
    anchor_prog=0.0; anchor_center=None; axis_sum=np.zeros(2); gap=0
    s_count=0; rejected={"no_disp":0,"wrong_dir":0,"backward":0}; aborted=0; events=[]

    def step(i, gate_ok, phi, center, u, L_ref):
        if not gate_ok or phi is None:
            if self.active:
                self.gap += 1
                if self.gap > frames(cfg.gap_hold_s): self._abort("gate_lost")
            return
        if self.active and self.gap > 0:               # 门限短暂掉出后恢复：冻结期间的相位不计，重新对齐
            self.phi_prev = phi; self.gap = 0
        if not self.active:
            self._start(i, phi, center); return
        dphi = wrap_pi(phi - self.phi_prev); self.phi_prev = phi
        self.axis_sum += u
        if self.direction == 0:                         # 方向锁定阶段
            self.prog += dphi
            if abs(self.prog) >= cfg.dir_lock_rad:      # 默认 0.8 rad
                self.direction = int(np.sign(self.prog)); self.prog = abs(self.prog); self.anchor_prog = 0.0
            return
        self.prog += self.direction * dphi
        cyc = self.prog - self.anchor_prog               # 本轮(当前周期)相位进度
        axis = unit(self.axis_sum)
        p = float(np.dot(center - self.anchor_center, axis))   # 本轮质心沿体轴位移（u 指向头：向前为正）
        # ★ 反向运动（甲方定义）：本轮相位进度回退为负 且 质心沿体轴实际后退 → 作废半截周期；已完成的 s_count 保留
        if cyc < -cfg.reverse_phase_eps_rad and np.sign(p) == -self.direction and abs(p) >= cfg.reverse_disp_ratio * L_ref:
            self._abort("reversal"); self._start(i, phi, center); return
        while self.prog - self.anchor_prog >= 2*np.pi:   # 完成一个完整周期
            p = float(np.dot(center - self.anchor_center, unit(self.axis_sum)))
            ok_dir  = (np.sign(p) == self.direction)
            ok_disp = abs(p) >= cfg.min_disp_ratio * L_ref
            if ok_dir and ok_disp and self.direction > 0:
                self.s_count += 1; self.events.append(SEvent(i, +1, p))
            elif ok_dir and ok_disp:                      # 向后 S：不统计，只记诊断
                self.rejected["backward"] += 1
            else:
                self.rejected["wrong_dir" if not ok_dir else "no_disp"] += 1
            self.anchor_prog += 2*np.pi; self.anchor_center = center.copy(); self.axis_sum[:] = 0
    def _start(i, phi, center): active=True; direction=0; prog=0; phi_prev=phi; anchor_prog=0; anchor_center=center.copy(); axis_sum=0; gap=0
    def _abort(reason): 若 active: aborted += 1; active=False   # 不保留任何残余相位；s_count 不动
    def force_abort(reason): ...   # 供深弯/BAD 过久调用
```
说明：
- `direction>0` ⇔ 相位增 ⇔ 波头→尾 ⇔ 应向**头方向**位移（`u` 指向头，故期望 `p>0`）。`direction<0`（向后爬）的完整周期一律进 `rejected.backward`。这个"符号一致"判据对头尾翻转不变。
- 相位回退的判据看的是**本轮相对本轮起点**的进度是否为负（不是相对历史极值），再加质心后退才触发——两个条件缺一不可，所以单纯的相位抖动或单纯的质心噪声不会误清零。
- `L_ref` = 该轨迹最近 `ref_window_s` 秒体长中位数。
- 引擎在每帧：若 `pose.quality==BAD` 且插值失败的连续时长 > `gap_hold_s`，或 `TurnCounter` 处于 CURL 状态 → `force_abort`。
> **参考原型 `docs/spec/reference/proto_kinematics.py` 已按本节逻辑实现并跑通下表所有合成场景**；先让自己的 `s_runs.py` 与原型在同样合成数据上结果一致，再接入引擎。20 s×0.5 Hz 理论恰好 10 周期，但采样窗口少 1 帧，总相位 9.98 周期，故期望写作 "10±1"（原型输出 9）。

**单测（T13，合成序列，07 §1 生成器；30 fps 与 10 fps 各跑一遍，A=0.8 rad，f=0.5 Hz，默认 20 s）**
| 用例 | 期望 |
| :-- | :-- |
| 前进 10 周期，每周期位移 0.4L | `s_count=10±1` |
| 后退 10 周期（波反向+身体后退） | `s_count=0`，`rejected.backward>=8` |
| 行波但原地（跑步机）| 0，`rejected.no_disp+wrong_dir>=8` |
| 波向前但身体向后漂（符号不一致）| 0，`rejected.wrong_dir>=8` |
| 只有头部 s<0.25 摆动，身体前进 | 0（门限 False） |
| 相位来回晃 ±0.4 周期，永不满周期 | 0 |
| 前进 3.5 周期后反向 3.5 周期（身体同步反向；总长 14 s） | `s_count=3`，`rejected.backward>=1`，`aborted>=1` |
| 前进 2.5 周期后反向 2.5 周期（总长 10 s） | `s_count=2`（半截作废，已完成保留） |
| 驻波（c 往复，无行波）| 0 |
| 纯噪声（幅度 < r_min），含 jitter 1 px | 0 |
| 行波中途 BAD 帧缺失 0.2 s（≤gap_hold）| 计数不变（仍 10±1） |
| 行波中途缺失 1.0 s | run 被 abort，总数 ≤ 10 且 `aborted>=1` |

## 5. 游泳摆动（`swing.py`）——同时输出"周期"与"单次弯曲"
```python
class SwingCounter:
    side=0; entries=0; flips=0; last_flip=-10**9; invalid_run=0
    def step(i, c, valid):
        if not valid:
            invalid_run += 1
            if invalid_run > frames(gap_hold_s): side=0        # 长缺失后重新开始，不保留半截
            return
        invalid_run = 0
        new = +1 if c >= cfg.bend_high else (-1 if c <= -cfg.bend_high else 0)
        if new == 0 or new == self.side: return
        if self.side != 0:                                       # 从一侧高位翻到另一侧高位（必经过零带）
            if i - self.last_flip < frames(cfg.min_flip_interval_s): return   # 太快视为噪声，忽略
            self.flips += 1; self.last_flip = i
        self.entries += 1                                        # 每次进入一侧高位 = 1 次单次弯曲（含首次）
        self.side = new
    @property
    def bends(self):  return self.entries            # c_bends：左弯、右弯各算 1 次
    @property
    def cycles(self): return self.entries // 2       # c_cycles：左+右 = 1 个完整来回
```
- 用的是 §2 的 `c`（总转角）；符号随头尾翻转而变，计数对称，无影响。
- 单测：`c=1.5·sin(2π·2t)`，10 s → `bends=40(±1)`，`cycles=20(±1)`；幅度 0.5 (<bend_high) → 均为 0；幅度 1.5 叠加 ±0.3 高频噪声 → 仍 40±1 / 20±1；序列 `+,−,+`（3 次进入）→ `bends=3, cycles=1`。

## 6. 深弯 / Ω 折返（`turns.py`，爬行 profile）
```python
class TurnCounter:
    state="INIT"; curl_start=0; last_turn=-10**9; count=0
    def step(i, R, valid):
        if not valid: return
        if state=="INIT":  if R >= R_exit: state="EXT"; return
        if state=="EXT" and R <= R_enter: state="CURL"; curl_start=i
        elif state=="CURL" and R >= R_exit:
            if i-curl_start >= frames(min_dwell_s) and i-last_turn >= frames(cooldown_s): count+=1; last_turn=i
            state="EXT"
    @property in_curl -> state=="CURL"
```
单测：R 序列 0.9→0.4(保持0.5s)→0.9，间隔 ≥3 s 重复 3 次 → 3；只降到 0.65 → 0；INIT 阶段从 0.4 开始 → 回升不计。

## 7. 行为状态（`behavior.py`）
逐帧（优先级自上而下）：
```
invalid  : 无有效特征
thrash   : swing.enabled 且 近 1.0 s 内 |c|>=bend_high 出现过 或 近 1.0 s 有翻边
turn     : turn.enabled 且 TurnCounter.in_curl
crawl    : wave_gate_ok 且 SRunCounter.active
static   : 体中心速度 < static_speed_bls (体长/秒) 且 max(amp_ant, amp_post) < static_amp_rad
moving   : 其他
```
体中心速度：以 `i` 为中心 ±0.5 s 的平滑中心位移 / 1 s / `L_ref`。
`activity_score` = 最近 1.5 s 内 behavior ∈ {crawl,thrash,turn,moving} 的帧比例。
旧字段映射（给 compat 用）：`static→"static"`, `crawl→"sinusoidal"`, `thrash/turn→"turning"`, `moving→"moving"`, `invalid→"unknown"`。

## 8. 汇总指标（`metrics.py`）
```python
@dataclass
class TrackReport:
    s_count:int; s_cycles_float:float          # 含当前未完成周期的小数部分（诊断用）
    c_cycles:int; c_bends:int; c_turn_count:int          # 游泳：周期/单次弯曲；爬行深弯折返
    c_count:int                                          # = (c_cycles 或 c_bends，按 cfg.c_unit) + c_turn_count
    c_unit:str                                           # 'cycle' | 'bend'
    rejected:dict; aborted_runs:int                      # rejected: no_disp / wrong_dir / backward
    # 说明：只统计向前 S，故无 is_bidirectional；compat 层对旧字段 is_bidirectional 恒输出 false
    mode_time:dict[str,float]                  # 秒
    wave_eps_median:float; observed_seconds:float          # 有效帧数/fps
    freq_s_hz:float; freq_c_hz:float; avg_freq_legacy:float   # (c_count+s_count)/observed_seconds，保持旧语义
    activity_score:float; motion_type_legacy:str
    n_ok:int; n_suspect:int; n_bad:int; n_missing:int
    spatial: dict   # total_path_length_px, net_displacement_px, start/end/bbox（沿用旧 build_id_pose_summary 的算法，以 dense.mean 为质心）
```

## 9. 引擎主循环（T15）
```python
def _process(self, i):                       # i 的窗口数据已齐
    dense_s = self._smoothed(i)              # ±h 帧均值，间隙 ≤gap_max_s 线性补，更长则 None
    feat = compute_features(dense_s) if dense_s is not None else None
    self.feats[i] = feat
    gate = wave_gate(self._window_feats(i, cfg.wave.win_s)) if (feat and cfg.s_run.enabled) else GateResult.off()
    L_ref = self._L_ref(i)
    if cfg.s_run.enabled:
        if self.turn.in_curl or self._bad_too_long(i): self.s.force_abort("curl/bad")
        self.s.step(i, gate.ok, atan2(feat.a2, feat.a1) if feat else None, feat.center if feat else None, feat.u if feat else None, L_ref)
    if cfg.swing.enabled: self.swing.step(i, feat.c if feat else 0, feat is not None)   # 同时累计 bends / cycles
    if cfg.turn.enabled:  self.turn.step(i, feat.R if feat else 0, feat is not None)
    beh = classify_behavior(...)
    self.per_frame.append(PerFrame(i, beh, self.s.s_count, c_cycles=self.swing.cycles, c_bends=self.swing.bends, c_turn=self.turn.count, wave_ok=gate.ok, ...))   # c_count 由 c_unit 选择
```
- 质量语义：`BAD/MISSING` → 原始位置当 None；`SUSPECT` 正常使用。
- **parity 测试**：同一序列分别 (a) `run` (b) 逐帧 `push` 并在途中调用 `snapshot()`，最终 `finalize()` 的 `TrackReport` 与 `PerFrame` 必须完全相等；且任一时刻 `snapshot()` 的计数 ≤ 最终计数（单调不减）。

## 10. 配置块（profile）
`config/default.yaml` 中 `kinematics:` 给出爬行默认；`profiles/swim_liquid.yaml` 覆盖差异。
```yaml
kinematics:                      # ← crawl_agar（也是默认）
  smooth_s: 0.17                 # 中心平滑窗（秒）
  gap_max_s: 0.30                # 允许线性插值的最长缺失
  ref_window_s: 2.0
  k_body: 1.0                    # 每体长波数
  wave: {win_s: 3.0, min_valid_frac: 0.6, r_min: 0.25, eps_min: 0.40, post_ratio_min: 0.35, amp_post_min: 0.10}
  s_run: {enabled: true, dir_lock_rad: 0.8, reverse_phase_eps_rad: 0.3, reverse_disp_ratio: 0.03, min_disp_ratio: 0.10, gap_hold_s: 0.30}   # 向后 S 一律不统计（无开关）
  swing: {enabled: false, bend_high: 0.9, min_flip_interval_s: 0.08}      # cycles 与 bends 始终都计算
  turn:  {enabled: true, R_enter: 0.55, R_exit: 0.72, min_dwell_s: 0.2, cooldown_s: 2.5}
  static_speed_bls: 0.03
  static_amp_rad: 0.08
```
```yaml
# profiles/swim_liquid.yaml
kinematics:
  smooth_s: 0.07                 # 游泳摆动可达 ~3 Hz，平滑必须很轻
  s_run: {enabled: false}
  turn:  {enabled: false}
  swing: {enabled: true, bend_high: 0.9, min_flip_interval_s: 0.08}
  wave:  {win_s: 1.0}
```
> 这些数值是**起点**。T16 之后用基准真值调参（`tools/tune.py` 网格搜索 `r_min, eps_min, min_disp_ratio, bend_high`，目标见 07 §6）。**不要为了过某一个视频而改规格里的结构**，只调这些数值。

## 11. 已知风险（实现时留意）
1. 真值显示 Swim-3 一只虫 22 s 数到 73 次（≈3.3 次/秒）。因为现在**同时输出**周期(`c_cycles`)与单次弯曲(`c_bends`)，评测报告里两种口径都要列出，由甲方判断哪种与其人工口径一致。
2. 爬行周期约 2 s（1.6X3 中第 3 只 16 次/31 s），`wave.win_s=3.0` 刚好 ≥1.5 周期；周期更慢的视频可能需要更长窗口。
3. 极小的游泳虫（40–60 px）：`bend_high` 过大会漏计，过小会把像素抖动当摆动；以基准为准。
4. 只统计向前 S 后，**方向判断依赖头尾朝向正确**（02 §5）。头尾标反会把向前误判为向后而漏计；用户首帧头标记能降低该风险，评测时对 `rejected.backward` 偏高的虫要人工看一眼朝向。
