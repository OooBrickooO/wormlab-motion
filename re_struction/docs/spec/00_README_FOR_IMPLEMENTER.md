# 实现者须知（给执行编码的模型）

你要把 `re_struction/` 下的规格落成代码。**规格已经把算法、接口、参数、验收都写死了**；遇到规格没写清的地方，先按"最保守、最简单"的做法实现，并在 `docs/spec/OPEN_QUESTIONS.md` 追加一条，不要自行发明新算法。

## 0. 硬性规则
1. **只在 `re_struction/` 内新增/修改文件。** 禁止修改 `backend/`、`frontend/`、`core.md` 等原项目任何文件。需要读旧代码时只读（`tools/` 里用 `sys.path.insert(0, "../backend")` 导入，**不要复制旧文件**）。
2. 语言/版本：Python 3.12；类型注解 + `dataclass`；文档字符串与注释用中文；日志用 `logging`（不要 `print`）。
3. 无模块级可变全局状态（模型、配置、线程池都通过对象传递）。`wormlab/` 包**禁止 import flask/fastapi/socketio**。
4. 坐标：图像坐标，x 向右、y 向下，像素，`float32`。**只有** `compat/legacy_payload.py` 允许做 `y_bl = H - y` 的左下角换算。
5. 时间：配置里全是**秒**，函数内部用 `cfg.frames(sec)`（= `max(1, round(sec*fps))`）换算。禁止出现魔法"帧数"常量。
6. 点序：中心线 `dense[0]` = 头，`dense[-1]` = 尾；`N_DENSE = 33`；`pts11` 由 `dense` 按弧长重采样得到。
7. 每个任务完成的定义（DoD）：代码 + 对应单测通过（`pytest -q <path>`）+ 在任务末尾给出一行汇报：`Txx 完成 | 测试: n passed | 偏离规格: 无/说明`。**测试不过不得进入下一个任务。**
8. 不要"顺手优化"规格之外的东西；不要引入新依赖（依赖清单见 §2）。

## 1. 阅读顺序
`RECONSTRUCTION_PLAN.md`（背景、业务定义 §2.1、真值 §2.2）→ `01_SPEC_core.md` → `02_SPEC_pose.md` → `03_SPEC_kinematics.md` → `04_SPEC_tracking.md` → `05_SPEC_pipeline_server.md` → `API_CONTRACT.md` → `06_SPEC_frontend.md` → `07_SPEC_tests_benchmarks.md`。

## 2. 依赖（`backend/requirements.txt`，需锁版本）
`numpy scipy scikit-image opencv-python ultralytics supervision pyyaml pillow fastapi uvicorn[standard] python-socketio python-multipart pydantic pytest`；PyTorch 必须是 **CUDA 12.8+ 构建**（RTX 5070 Ti / Blackwell），安装命令见 T01。前端：`vite react react-dom typescript axios socket.io-client vitest @testing-library/react`。

## 3. 目录约定
以 `RECONSTRUCTION_PLAN.md §3.1` 的目录树为准。测试与源码镜像：`tests/unit/<package>/test_<module>.py`。运行产物一律写 `data/`（已 gitignore）。

## 4. 任务清单（按顺序做；括号内为依赖）

### P0 冻结与基线
| ID | 任务 | 规格 | 验收 |
| :-- | :--- | :--- | :--- |
| T01 | 新 venv + `requirements.txt` + `tools/check_env.py`（断言 `torch.cuda.is_available()`，打印 GPU 名/算力/torch/cuda 版本，加载 `origin.pt` 在 GPU 上跑一帧） | 本文 §2 | `python tools/check_env.py` 退出码 0 |
| T02 | `tools/map_labels.py`：把 `referAnswer/numbering_*.jpg` 上的编号映射到视频第 0 帧的虫（见 07 §3） | 07 §3 | 生成 `benchmarks/labels/*.overlay.jpg` 供人工目检；输出 `labels/*.json` |
| T03 | `tools/bench_infer.py`：推理分辨率**验证**实验（默认已定为 416，本实验对比 416/640/1280/native 并记录，见 07 §4） | 07 §4 | 产出 `benchmarks/report_d12.md` 与对比图 |
| T04 | `tools/dump_legacy_baseline.py`：只读导入旧 `worm_processing`，在基准视频上跑，导出每虫 S/C 计数 JSON | 07 §5 | `benchmarks/baseline_legacy.json` |

### P1 姿态层（`wormlab/pose`）
| ID | 任务 | 规格 | 验收 |
| :-- | :--- | :--- | :--- |
| T05 | `schemas.py` `config.py` + `config/default.yaml` `profiles/*.yaml` | 01 | `tests/unit/test_config.py`：两 profile 都能加载、秒→帧换算正确 |
| T06 | `detection/detector.py` `filters.py`（YOLO 批推理、GPU 上裁剪掩码、过滤、NMS、可选切片） | 01 §4 | 单测用假 results 对象；集成：对 `Swim-4` 第 0 帧检出 1 只 |
| T07 | `pose/skeleton.py` `pose/centerline.py` `extract_pose()` | 02 §2–§4 | 合成直线/弧/S/C/Ω 形：长度误差 <5%（C/Ω <8%），端点误差 <2px |
| T08 | `pose/orientation.py` `pose/quality.py` | 02 §5–§6 | 翻转用例、质量判定用例 |
| T09 | `tools/make_synthetic.py`（合成中心线 + 渲染掩码，07 §1）与 `tests/synthetic/test_pose_synth.py` | 07 §1 | 全部通过 |
| T10 | `tools/diagnose_track.py`（输出姿态/曲线诊断图） | 02 §8 | 在 `Move-3`、`Swim-4` 上出图 |

### P2 运动学层（`wormlab/kinematics`）
| ID | 任务 | 规格 | 验收 |
| :-- | :--- | :--- | :--- |
| T11 | `features.py` | 03 §2 | 单测：切线角、R、c(t)、a1/a2 对解析波形的值 |
| T12 | `wave.py`（门限）+ `modes.py` | 03 §3 | 行波/驻波/头部摆动三类信号的门限结果正确 |
| T13 | `s_runs.py`（S 单调 run 计数器；**先对照 `reference/proto_kinematics.py`**） | 03 §4 | 合成：行波 N 周期→`s_count≈N`；向后/回退半截/无位移/头摆/驻波→不计（见 03 §4 用例表） |
| T14 | `swing.py` `turns.py` | 03 §5–§6 | 合成：C 来回 N 次→N；Ω 折返 |
| T15 | `behavior.py` `metrics.py` `counter.py`（引擎，延迟因果） | 03 §7–§9 | 合成全套场景；`push` 逐帧与 `run` 一次性结果完全一致 |
| T16 | `tools/eval_benchmark.py` + `tools/tune.py`（读 Pass A 缓存→运动学→对比真值，输出表格）。**真实视频评测需要 T24 的缓存**：此时先用合成缓存自测，T24 完成后再跑真实基准 | 07 §5–§7 | 能产出 `benchmarks/report.md` |

### P3 跟踪层（`wormlab/tracking`）
| ID | 任务 | 规格 | 验收 |
| :-- | :--- | :--- | :--- |
| T17 | `id_pool.py` + 单测（锁定旧行为） | 04 §2 | 用例表全过 |
| T18 | `kalman.py` `cost.py` | 04 §3 | 与旧实现数值对拍（容差 1e-5） |
| T19 | `associator.py` `reid.py` | 04 §4 | 回放测试（T21）前置 |
| T20 | `head_config.py` `tracker.py` | 04 §5–§6 | 首帧头标记用例 |
| T21 | 回放对拍：`tools/dump_detections.py` + `tests/integration/test_tracker_replay.py` | 04 §7 | 新旧 ID 序列一致率 ≥99% |
| T22 | （可选，默认关）`tracking/stitch.py` | 04 §8 | 合成交叉用例；开关默认 off |

### P4 流水线与服务
| ID | 任务 | 规格 | 验收 |
| :-- | :--- | :--- | :--- |
| T23 | `render/*`（overlay、fonts、subvideo、video_io） | 05 §3 | 渲染一帧与旧版目视一致（保存对比图） |
| T24 | `pipeline/cache.py` `pass_a.py` | 05 §1–§2 | 无服务器跑 `Move-3` 前 300 帧得到 `tracks.npz` |
| T25 | `pipeline/pass_b.py` + `compat/legacy_payload.py` + `server/exporters/*` | 05 §4–§6 | 产出 mp4/json/csv/zip，字段符合 `API_CONTRACT.md` |
| T26 | `server/*`（FastAPI、socket.io、worker、任务管理、落盘/恢复） | 05 §7–§9 | `tests/integration/test_contract.py` 全绿 |
| T27 | `services/egg_count.py` `model_registry.py` + 预览类路由 | 05 §10 | 契约测试覆盖 `/preview_count` `/analyze_first_frame` `/extract_manual_worm` `/egg_count` |

### P5 前端
| ID | 任务 | 规格 | 验收 |
| :-- | :--- | :--- | :--- |
| T28 | Vite+TS 脚手架、`api/types.ts` `client.ts` `socket.ts` | 06 §1–§2 | `npm run build` 通过 |
| T29 | 组件迁移：HeadMarkModal、GlassVideoPlayer、EggCountResultPanel | 06 §3 | 渲染测试 |
| T30 | hooks 与 App 拆分、TrackTable/SummaryTable、profile 选择器 | 06 §4–§5 | 与旧 UI 功能清单逐项对照（06 §6） |
| T31 | 前端测试 + 与新后端端到端冒烟 | 06 §7 | 上传 `Swim-4` 走完全流程 |

### P6 联调
| ID | 任务 | 验收 |
| :-- | :--- | :--- |
| T32 | `tools/compare_legacy.py` 新旧 A/B + `benchmarks/report.md` | 达到 07 §6 的验收线 |
| T33 | 暂停/取消/改虫数/长视频/重启恢复的稳定性冒烟 | 全过 |
| T34 | `docs/CUTOVER.md`（切换清单，仅文档，不执行） | 文档 |

> 说明：T01–T34 为全量清单。每个任务的代码量都应控制在几百行内；若某个文件超过 400 行，请拆分。

## 5. 汇报模板
```
T07 完成 | 文件: wormlab/pose/skeleton.py, centerline.py, tests/unit/pose/test_centerline.py
测试: 14 passed | 偏离规格: 无
遗留/疑问: （无则写无）
```
