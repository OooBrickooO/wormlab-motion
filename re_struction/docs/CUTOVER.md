# 生产环境切换与回滚操作手册 (`CUTOVER.md`)

> 本文档为重构系统 (`re_struction/`) 从并行试运行端口（后端 `5001` / 前端 `3001`）切换至生产默认环境的核对清单。**当前阶段仅交付文档，严禁直接覆盖或修改原仓根目录 `backend/` 与 `frontend/`。**

---

## 1. 切换前就绪检查 (Pre-Cutover Gate)

执行切换前，在 `re_struction/` 目录下逐项确认全部验证已通过：

1. **后端非慢速全量测试通过**：
   ```powershell
   $env:PYTHONPATH = "d:\Project\Python\Project\motion\re_struction\backend"
   d:\Project\Python\Project\motion\re_struction\backend\.venv\Scripts\python.exe -m pytest d:\Project\Python\Project\motion\re_struction\backend -q -m "not slow"
   ```
   期望：零失败（包含单元、9 组合成运动学/姿态场景、FastAPI 路由契约、Pass A/B 集成冒烟测试）。
2. **前端类型检查、单元测试与生产构建通过**：
   ```powershell
   cd d:\Project\Python\Project\motion\re_struction\frontend
   npm run build
   npx vitest run
   ```
   期望：`tsc --noEmit && vite build` 零错误、`vitest` 全部通过。
3. **新旧 A/B 基准报告已复核**：
   查看 [benchmarks/report.md](../benchmarks/report.md)，确认爬行标注通过率 `80.0% (16/20)`、`gt=0` 总误报 `0`（旧系统 `10`）、总体 MAE `3.17`（旧系统 `6.48`）、平均轨迹覆盖率 `> 98%`、汇总 `BAD` 帧占比（爬行 `2.5%` / 游泳 `3.0%`）均在预期边界内。
4. **环境与权重依赖齐全**：
   - `re_struction/backend/models/` 下存在 `origin.pt`、`CJLUdata_Old.pt`、`EggCount.pt`
   - `re_struction/backend/openh264-1.8.0-win64.dll` 位于后端工作目录（保证 H.264 兼容视频编码）

---

## 2. 并行端口预演 (`5001` + `3001`)

在不停止旧系统服务的前提下，独立启停重构版本进行人工验收：

1. **启动新后端 (FastAPI + Socket.IO ASGI，端口 `5001`)**：
   ```powershell
   cd d:\Project\Python\Project\motion\re_struction\backend
   d:\Project\Python\Project\motion\re_struction\backend\.venv\Scripts\python.exe run.py
   ```
   - 监听地址配置于 `backend/config/default.yaml` 的 `server.host: "0.0.0.0"`、`server.port: 5001`。
   - 健康检查：`GET http://localhost:5001/health` 返回 `{"ok": true}`。
2. **启动新前端 (Vite + React + TypeScript，端口 `3001`)**：
   ```powershell
   cd d:\Project\Python\Project\motion\re_struction\frontend
   npm run dev
   ```
   - `frontend/vite.config.ts` 锁定端口 `3001`，`frontend/.env.development` 配置 `VITE_API_BASE=http://localhost:5001`。
3. **关键人工验收路径**：
   - 视频上传与分析：验证 `/preview_count`、`/analyze_first_frame` + `HeadMarkModal` 首帧标记（含"端点 0/10"快捷切换及手动画框 `/extract_manual_worm`）。
   - 任务运行态：验证暂停/恢复/取消、中途动态调整 `worm_count` (`set_worm_count`)、Socket.IO `frame`/`phase`/`complete` 断线重连后自动恢复轮询或 `join_task`。
   - 双单位切换：在 `swim_liquid` / `crawl_agar` 下切换 `CUnitToggle`（往返周期 `cycle` vs 单次弯曲 `bend`），核对 `LiveTrackTable`、`SummaryTable` 与子视频下载。
   - 虫卵计数模式：`EggCountResultPanel`（总计 / 90% / 95% 置信度分级及叠加图）。
   - 重启恢复：终止 `run.py` 后重启，检查已进入终态的任务仍可通过 `/task/<id>/status` 与历史面板查回，未完成任务收敛为 `cancelled`。

---

## 3. 正式生产切换步骤 (Production Cutover)

确认并行预演无误后，按以下顺序执行正式切换：

1. **停止旧服务进程**（旧 Flask `app.py` 与旧 CRA 前端进程）。
2. **保留原仓备份**：将原根目录旧 `backend/` 与 `frontend/` 整体备份或打 Git Tag（如 `git tag pre-cutover-legacy`）。
3. **迁移运行数据与模型（如有新增现场任务输出）**：
   - 如需保留旧历史文件，仅拷贝下载成品；旧版任务 JSON 与新版 `results/tasks/<task_id>.json` 结构兼容核心字段（见 [docs/API_CONTRACT.md](API_CONTRACT.md)）。
4. **切换生产端口配置（如生产环境仍要求原端口 `5000` / `3000`）**：
   - 后端修改 `backend/config/default.yaml` 中 `server.port` 与 `server.cors_origins`（如 `5000` 与 `["http://localhost:3000"]`）。
   - 前端构建静态产物并更新 `.env.production` 中的 `VITE_API_BASE`，执行 `npm run build` 生成 `frontend/dist/` 交由 Nginx / 静态托管或生产预览服务分发。
5. **切换后冒烟**：
   ```powershell
   $env:PYTHONPATH = "d:\Project\Python\Project\motion\re_struction\backend"
   d:\Project\Python\Project\motion\re_struction\backend\.venv\Scripts\python.exe d:\Project\Python\Project\motion\re_struction\tools\smoke_e2e.py
   ```
   确认 `completed` 终态及 5 类产物下载链接均为 HTTP 200。

---

## 4. 应急回滚方案 (Rollback)

若切换后出现任何阻断性问题：

1. 立即停止 `re_struction/backend/run.py` 及新前端服务。
2. 直接恢复 `pre-cutover-legacy` 标签（或启动原仓保留的旧版后端 `app.py` 与旧版前端）。
3. 因 `re_struction/` 全程保持目录隔离（数据目录 `re_struction/data/` 与原仓完全独立），旧系统可在 **1 分钟内零损伤恢复运行**，随后将 `re_struction/data/results/tasks/<id>.json` 与视频输入留存供离线复现排查。
