# `re_struction` 重构工程文件清单 (`projectFileList.md`)

本文档记录 `re_struction/` 重构工程中的所有文件分类：
- **直接复用**：原封不动从旧工程复制到 `re_struction/` 中使用的文件（零改动）。
- **重写**：无论是全新创建、架构重构拆分，还是基于原文件微调/修改的文件，一律归入“重写”名单。

---

## 一、直接复用（原封不动复制）

| 序号 | 新路径 (`re_struction/` 内) | 原路径 | 标记 | 用途说明 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `backend/openh264-1.8.0-win64.dll` | `openh264-1.8.0-win64.dll` | **直接复用** | OpenCV H.264 视频编码动态链接库 |
| 2 | `backend/models/origin.pt` | `backend/models/origin.pt` | **直接复用** | 默认线虫分割 YOLO 权重 (`imgsz=416`) |
| 3 | `backend/models/CJLUdata_Old.pt` | `backend/models/CJLUdata_Old.pt` | **直接复用** | 备选线虫分割 YOLO 权重 |
| 4 | `backend/models/EggCount.pt` | `backend/models/EggCount.pt` | **直接复用** | 虫卵与线虫计数检测 YOLO 权重 |
| 5 | `frontend/src/styles/App.css` | `frontend/src/App.css` | **直接复用** | 前端完整样式表与玻璃拟态主题变量 |
| 6 | `frontend/src/styles/index.css` | `frontend/src/index.css` | **直接复用** | 前端全局基础重置样式 |
| 7 | `frontend/public/favicon.ico` | `frontend/public/favicon.ico` | **直接复用** | 站点图标 |
| 8 | `frontend/public/logo192.png` | `frontend/public/logo192.png` | **直接复用** | 站点 Logo (192x192) |
| 9 | `frontend/public/logo512.png` | `frontend/public/logo512.png` | **直接复用** | 站点 Logo (512x512) |
| 10 | `frontend/public/manifest.json` | `frontend/public/manifest.json` | **直接复用** | PWA Web App Manifest |
| 11 | `frontend/public/robots.txt` | `frontend/public/robots.txt` | **直接复用** | 爬虫规则文件 |

---

## 二、重写（含全新架构、重构拆分与微调）

### 2.1 架构设计、契约、基准与切换文档
| 路径 (`re_struction/` 内) | 对应原文件 / 来源 | 标记 | 说明 |
| :--- | :--- | :--- | :--- |
| `RECONSTRUCTION_PLAN.md` | `re_struction/plan.md` | **重写** | 重构总方案 v2 |
| `projectFileList.md` | 新增 | **重写** | 重构工程文件分类清单（直接复用 vs 重写） |
| `benchmarks/ground_truth.yaml` | `re_struction/referAnswer/*` | **重写** | 结构化人工真值配置 |
| `benchmarks/baseline_legacy.json` | 旧系统基线评测 | **重写** | 旧版 `worm_processing.py` 基准计数基线 |
| `benchmarks/labels/*.json`, `*.overlay.jpg` | `referAnswer/numbering_*` | **重写** | 标号→检测框映射及首帧 `head_xy` 标注 |
| `benchmarks/report.md`, `benchmarks/report_d12.md`, `benchmarks/d12/*` | 新增 | **重写** | D12 分辨率结论与新旧 A/B 评测报告 |
| `benchmarks/tuned_crawl_agar.yaml` | 新增 | **重写** | LOVO 交叉验证超参数搜索结果 |
| `docs/API_CONTRACT.md` | `backend/app.py`, `frontend/src/App.js` | **重写** | 前后端 REST 与 Socket.IO 冻结契约 |
| `docs/CUTOVER.md` | 新增 | **重写** | 并行试运行 (`5001`/`3001`) 与生产切换回滚手册 (T34) |
| `docs/archive/PLAN_v1.md` | `re_struction/plan.md` | **重写** | v1 草案归档 |
| `docs/spec/00_README_FOR_IMPLEMENTER.md` | 新增 | **重写** | 编码规范与 T01–T34 任务清单 |
| `docs/spec/01_SPEC_core.md` | `backend/worm_processing.py` | **重写** | 核心数据结构、配置与检测层规格 |
| `docs/spec/02_SPEC_pose.md` | `backend/worm_processing.py` | **重写** | 骨架测地最短路与 33 点中心线姿态规格 |
| `docs/spec/03_SPEC_kinematics.md` | `backend/worm_processing.py` | **重写** | 行波门限、S 单调相位计数器与 C 摆动规格 |
| `docs/spec/04_SPEC_tracking.md` | `backend/worm_processing.py` | **重写** | 卡尔曼、ID 池、三阶段匈牙利关联跟踪规格 |
| `docs/spec/05_SPEC_pipeline_server.md` | `backend/app.py`, `backend/worm_processing.py` | **重写** | Pass A/B 双遍流水线与 FastAPI 服务层规格 |
| `docs/spec/06_SPEC_frontend.md` | `frontend/src/App.js` | **重写** | Vite + TypeScript 前端组件与 Hooks 拆分规格 |
| `docs/spec/07_SPEC_tests_benchmarks.md` | 新增 | **重写** | 合成数据、单测与基准评测规格 |
| `docs/spec/DECISIONS.md` | 新增 | **重写** | 架构决策记录 (ADR D1–D16) |
| `docs/spec/OPEN_QUESTIONS.md` | 新增 | **重写** | 开放问题与确认记录 |
| `docs/spec/SMOKE_RESULT.md` | 新增 | **重写** | `Swim-4.mp4` 端到端冒烟验证记录 |
| `docs/spec/reference/proto_kinematics.py` | 新增 | **重写** | S/C 运动学算法可运行参考原型 |

### 2.2 后端与算法代码 (`backend/`)
| 路径 (`re_struction/` 内) | 对应原文件 / 来源 | 标记 | 任务编号与说明 |
| :--- | :--- | :--- | :--- |
| `backend/pytest.ini`, `backend/requirements.txt` | `backend/requirements.txt` | **重写** | 测试与依赖声明 (`pyproject`/pip 环境隔离) |
| `backend/run.py` | `backend/app.py` | **重写** | T27: FastAPI + Socket.IO ASGI 单进程入口 (`:5001`) |
| `backend/config/default.yaml`, `backend/config/profiles/{crawl_agar,swim_liquid}.yaml` | 硬编码超参数 | **重写** | T02/T32: 分层 YAML 配置、强校验与环境调优参数 |
| `backend/wormlab/__init__.py`, `schemas.py`, `config.py`, `geometry.py` | `worm_processing.py` | **重写** | T01–T03: 数据契约、配置解析与几何基础工具 |
| `backend/wormlab/detection/{__init__.py,detector.py,filters.py}` | `worm_processing.py` | **重写** | T04/T32: YOLO 分割批推理 (`imgsz=416`)、OOM 退避及面积上下限过滤 |
| `backend/wormlab/pose/{__init__.py,skeleton.py,centerline.py,orientation.py,quality.py,mask_refine.py,extractor.py}` | `worm_processing.py` | **重写** | T05–T09: ROI 放大、去环骨架直径最短路、33 点重心法对称采样、方向追踪与质量分级 |
| `backend/wormlab/kinematics/{__init__.py,modes.py,features.py,wave.py,s_runs.py,swing.py,turns.py,behavior.py,metrics.py,counter.py}` | `worm_processing.py` | **重写** | T10–T15: `wave_gate` 门限、单调展开相位 S 计数、C 双单位摆动/转弯及因果 `KinematicsEngine` |
| `backend/wormlab/tracking/{__init__.py,kalman.py,id_pool.py,features.py,cost.py,associator.py,head_config.py,track.py,tracker.py,reid.py,stitch.py}` | `worm_processing.py` | **重写** | T17–T21: 三阶段匈牙利匹配、ID 池复用、首帧头标对齐、ReID 与断轨拼接预留 |
| `backend/wormlab/compat/{__init__.py,legacy_payload.py}` | `worm_processing.py` | **重写** | T22: 左下角坐标变换 (`y_bl=H-y`) 与旧字段 1:1 映射层 |
| `backend/wormlab/pipeline/{__init__.py,cache.py,late_entry.py,pass_a.py,pass_b.py}` | `worm_processing.py` | **重写** | T22–T24: `tracks.npz` 磁盘缓存、后进入虫禁计过滤与 Pass A/B 双遍解耦管线 |
| `backend/services/{__init__.py,model_registry.py,egg_count.py}` | `backend/app.py` | **重写** | T25: 运行期按需加载模型与虫卵分级计数服务 |
| `backend/server/{__init__.py,app_factory.py,events.py,sockets.py}` | `backend/app.py` | **重写** | T26–T27: FastAPI 工厂、Socket.IO 房间推送与异常收敛 |
| `backend/server/routes/{__init__.py,models.py,preview.py,egg.py,upload.py,tasks.py,media.py}` | `backend/app.py` | **重写** | T26–T27: 模块化 REST 路由（含 HTTP Range 流媒体播放） |
| `backend/server/tasks/{__init__.py,store.py,manager.py,worker.py}` | `backend/app.py` | **重写** | T25–T26: 内存热存储 + 终态 JSON 落盘重启恢复 + 常驻 GPU Worker 进程 |
| `backend/server/exporters/{__init__.py,csv_report.py,summary_json.py,zip_pack.py}` | `backend/app.py` | **重写** | T24: 终稿 CSV、JSON 汇总与子视频 ZIP 归档导出 |

### 2.3 前端 Vite + TypeScript 代码 (`frontend/`)
| 路径 (`re_struction/` 内) | 对应原文件 / 来源 | 标记 | 任务编号与说明 |
| :--- | :--- | :--- | :--- |
| `frontend/index.html`, `package.json`, `package-lock.json`, `tsconfig.json`, `vite.config.ts`, `.env.development`, `src/vite-env.d.ts`, `src/main.tsx` | `frontend/package.json`, `index.html`, `index.js` | **重写** | T28: CRA → Vite + TypeScript 工程迁移（监听 `:3001`） |
| `frontend/src/api/{types.ts,urls.ts,client.ts,socket.ts}`, `frontend/src/utils/{firstFrame.ts,format.ts}` | `frontend/src/App.js` | **重写** | T28: API 契约类型、Axios/Fetch 封装、Socket 房间重连与浏览器首帧抽帧 |
| `frontend/src/components/{HeadMarkModal/HeadMarkModal.tsx,HeadMarkModal/drawCanvas.ts,GlassVideoPlayer.tsx,EggCountResultPanel.tsx}` | `frontend/src/components/*` | **重写** | T29: TS 重写首帧头标模态框（新增端点 0/10 快捷按钮）、玻璃播放器与虫卵面板 |
| `frontend/src/hooks/{useModels.ts,useHistory.ts,useTaskSocket.ts,useTaskPolling.ts,usePreviewCount.ts,useHeadMark.ts,useTaskControl.ts,useEggCount.ts}` | `frontend/src/App.js` | **重写** | T30: 原 828 行单组件状态机拆分为 8 个职责单一 Hooks |
| `frontend/src/components/{InputPanel.tsx,ProgressStats.tsx,CUnitToggle.tsx,LiveTrackTable.tsx,SummaryTable.tsx,SubVideoList.tsx,HistoryPanel.tsx}`, `frontend/src/App.tsx` | `frontend/src/App.js` | **重写** | T30: 声明式 UI 拆分子组件（新增 C 单位切换与 `late_entry` 标识展示） |

### 2.4 评测工具与自动化测试 (`tools/` & `tests/`)
| 路径 (`re_struction/` 内) | 对应原文件 / 来源 | 标记 | 任务编号与说明 |
| :--- | :--- | :--- | :--- |
| `tools/{check_env.py,make_synthetic.py,bench_infer.py,dump_detections.py,dump_legacy_baseline.py,map_labels.py,diagnose_track.py,eval_benchmark.py,tune.py,compare_legacy.py,smoke_e2e.py}` | 新增 | **重写** | T01/T16/T24/T31/T32: 环境校验、合成对抗生成器、基准评测、LOVO 调参、A/B 对比与 E2E 冒烟脚本 |
| `backend/tests/unit/**` (19 个单测文件) | 新增 | **重写** | T02–T26: 检测/姿态/运动学/跟踪/兼容/缓存/导出器全模块单元测试 |
| `backend/tests/synthetic/{test_pose_synth.py,test_kin_synth.py}` | 新增 | **重写** | T09/T16: 合成姿态几何误差与运动学 9 组对抗场景验证 |
| `backend/tests/integration/{test_tracker_replay.py,test_contract.py,test_pass_ab_smoke.py}` | 新增 | **重写** | T21/T27/T33: 跟踪回放、REST+落盘契约测试、Pass A/B 稳定性与暂停/取消/重启动恢复冒烟 |
| `frontend/src/__tests__/{components.test.tsx,hooks.test.tsx}` | 新增 | **重写** | T29/T31: Vitest + Testing Library 前端组件与 Hooks 单测 |

> 注：`re_struction/referAnswer/` 为甲方提供的原始测试视频、参考图片与计数真值原始素材，仅作只读评测输入。

