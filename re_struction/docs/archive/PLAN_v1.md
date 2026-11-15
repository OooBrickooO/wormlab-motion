# 线虫运动分析算法底层重构计划 (Reconstruction Plan)

> **文档定位**：新会话交接指导文件。在新会话中，只需将本文档路径或内容提供给 AI，即可直接进入重构实施阶段。

---

## 一、 重构背景与核心目标

### 1. 现状痛点诊断
目前 [`backend/worm_processing.py`](file:///d:/Project/Python/Project/motion/backend/worm_processing.py)（2550+ 行）因经历多次缝合，存在难以通过简单调参解决的底层病灶：
1. **C 形无法识别的结构性死结**：
   - 骨架分支熔断（`branch_pixels > max(8, 0.18*len)`）在深弯与闭环时极易误杀；
   - B 样条拟合的弦切拉直效应导致 C 形体长缩短 25%~35%；
   - `_check_and_update_body_length` 判定体长变化 >25% 即视为伪影，连续 4 帧硬重置 C 状态机，形成了“越弯曲越被当作异常丢弃”的悖论。
2. **S 运动偏大且不稳定的数学原因**：
   - 用局部三点离散曲率乘以体长，像素抖动被放大百倍；
   - 依赖 20 多个一维阈值和布尔标志位模拟周期，缺乏真正行波的物理约束；
   - 关键点更新间隔（每 3 帧）与窗口点数硬编码，未与物理时间（FPS）自适应联动。
3. **架构臃肿**：单文件混杂了 YOLO 跟踪、骨架化、多代不同计数逻辑、渲染绘图与视频写入。

### 2. 重构范围界定
* **严格保留（不动）**：
  - 前端工程：[`frontend/src/App.js`](file:///d:/Project/Python/Project/motion/frontend/src/App.js)（首帧标记/跳过、虫卵计数面板、实时画面看板等 UI 与接口交互）。
  - 后端流水线：[`backend/app.py`](file:///d:/Project/Python/Project/motion/backend/app.py)（多进程 GPU 批处理任务调度、WebSocket 推流、任务控制队列）。
* **彻底推倒重构（核心层）**：
  - 将原 `backend/worm_processing.py` 拆解并重写于 `re_struction/` 目录下，完成后无缝替换原模块。

---

## 二、 核心数学模型与算法设计

### 1. S 运动：二维相空间全波展开法 (Phase Unwrapping)
彻底废弃“阈值过零状态机”，采用生物物理学权威方法（类似 Eigenworms / Tierpsy）：
- **正交正弦基投影**：
  将等弧长 11 点中心线投影到前两阶正交空间模态：
  $$a_1(t) = \int_0^1 \theta(s, t) \cos(2\pi s) \, ds, \quad a_2(t) = \int_0^1 \theta(s, t) \sin(2\pi s) \, ds$$
- **复平面相位展开**：
  $$\Phi(t) = \text{unwrap}\big(\text{atan2}(a_2(t), a_1(t))\big)$$
  $$S\_count = \left\lfloor \frac{|\Phi(t) - \Phi(0)|}{2\pi} \right\rfloor$$
- **优势**：
  只有当正弦波真正自头向尾贯通传导时，相平面 $(a_1, a_2)$ 才会绕原点旋转一整圈（$\Delta \Phi = \pm 2\pi$）；原地晃头或边缘抖动在相平面仅是局部摆动，**净积分天然为 0**，根治偏大与抖动。

### 2. C 运动：一阶对称屈曲模态 + 弦长比迟滞环
- **对称弯曲投影**：
  $$c(t) = \int_0^1 \theta(s, t) \cos(\pi s) \, ds$$
- **弦长舒展比**：
  $$R(t) = \frac{\|P_{\text{head}} - P_{\text{tail}}\|}{L_{\text{body}}}$$
- **判定逻辑**：
  - **模式 C1（液体快速对称游泳 Thrashing）**：$c(t)$ 的对称极值双向交替过零，配合 $R(t) < 0.82$。
  - **模式 C2（深弯/Omega 转向）**：$R(t)$ 由舒展态（$>0.72$）下探至深弯阈值（$<0.55$）并成功舒展复原，独立计 1 次。
  - **解耦防误杀**：放宽或重构体长保护，弯曲时体长允许平滑伸缩，绝不对弯曲帧进行清空熔断。

---

## 三、 重构模块化架构设计 (`re_struction/`)

建议将原本 2550 行的单体文件拆解为高内聚、低耦合的 5 个独立模块：

```text
re_struction/
├── RECONSTRUCTION_PLAN.md      # [本文件] 重构规划与交接说明
├── skeleton.py                 # 模块 1：骨架与中心线提取（鲁棒去分支、等弧长采样、体长自适应）
├── kinematics.py               # 模块 2：相空间 S/C 运动学判别器与频率/活性分析
├── tracker.py                  # 模块 3：卡尔曼多目标跟踪器与严格 ID 池管理器
├── visualizer.py               # 模块 4：帧绘制、药丸徽章与子视频裁剪导出
├── diagnostic.py               # 模块 5：离线可视化诊断台（提取真实视频波形曲线校验）
└── worm_processing_new.py      # 对外统一入口（完全兼容原有 app.py 的 API 签名）
```

### 接口兼容性保证
`worm_processing_new.py` 必须对外暴露与原 `worm_processing.py` 完全一致的入口函数与类：
* `class EnhancedWormTracker` (包含 `update_tracking`, `set_max_worm_id` 等)
* `def init_model(model_path)`
* `def process_frame(...)` / `def process_frame_batch(...)`
* `def annotate_and_collect_data(...)`

---

## 四、 实施步骤路线图

### 第一阶段：离线算法原型与诊断台 (Phase 1)
1. **编写 `skeleton.py`**：实现抗 C 形自交剪枝的中心线提取与 11 节点等弧长重采样；
2. **编写 `kinematics.py`**：实现基于 Phase Unwrapping 的 S 计数器与一阶弯曲模态 C 计数器；
3. **编写 `diagnostic.py`**：选择 1~2 段具有代表性的真实线虫视频（一段典型 S 爬行，一段含 C 转向/游泳），运行提取并绘制：
   - 相空间轨迹曲线 $(a_1, a_2)$
   - 相位展开曲线 $\Phi(t)$
   - 弯曲模态与弦长比曲线 $c(t), R(t)$
   - 直接对比人工目测真值，调校好基准参数。

### 第二阶段：工程模块构建 (Phase 2)
1. **编写 `tracker.py`**：移植并精简现有稳定的卡尔曼跟踪、置信度降序分配、严格受限 ID 池（禁止溢出、快速回收瞬态误检）；
2. **编写 `visualizer.py`**：承接原有的抗锯齿轨迹、多边形平滑轮廓、药丸徽章与子视频保存；
3. **组装 `worm_processing_new.py`**：将各模块打通，严格保持接口与原系统一致。

### 第三阶段：全系统联调与上线 (Phase 3)
1. 在 `scratch/` 或测试脚本中运行完整集成测试；
2. 验证与 [`backend/app.py`](file:///d:/Project/Python/Project/motion/backend/app.py) 的多进程、WebSocket 推流及状态控制无缝联动；
3. 将测试成熟的代码正式替换覆盖 `backend/worm_processing.py`；
4. 运行前端，端到端测试真实视频流。

---

## 五、 新会话交接 Prompt 模板

新开对话时，可直接向 AI 发送以下指令启动重构：

```text
我们准备正式开始线虫运动分析算法的底层重构。
重构目标与技术方案已详述在 re_struction/RECONSTRUCTION_PLAN.md 中：
1. 保持前端 (App.js) 与后端调度 (app.py) 外框不变；
2. 在 re_struction/ 目录下重构中心线提取 (skeleton.py)、相空间运动学模型 (kinematics.py) 及跟踪器 (tracker.py)；
3. 请先阅读 re_struction/RECONSTRUCTION_PLAN.md，然后从第一阶段开始推进实施。
```
