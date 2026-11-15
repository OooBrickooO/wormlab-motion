# 06 前端（Vite + TypeScript）

对应任务：T28–T31。**目标：功能与外观与旧前端一致，只做工程化重构 + 类型 + 一个新增控件（分析环境 profile）。** 旧前端只读参照：`frontend/src/App.js`（1677 行）、`App.css`（直接复制，视觉不改）。

## 1. 脚手架（T28）
```
re_struction/frontend/
├── package.json            # vite react react-dom typescript axios socket.io-client vitest @testing-library/react jsdom
├── vite.config.ts          # server.port=3001；test: {environment:'jsdom'}
├── tsconfig.json           # strict: true
├── .env.development        # VITE_API_BASE=http://localhost:5001
├── index.html
└── src/
    ├── main.tsx            # createRoot；import './styles/index.css'
    ├── App.tsx             # 只做布局与组合（< 250 行）
    ├── api/{types.ts, client.ts, socket.ts, urls.ts}
    ├── hooks/{useModels.ts, useTaskSocket.ts, useTaskPolling.ts, useHistory.ts, usePreviewCount.ts, useHeadMark.ts, useEggCount.ts, useTaskControl.ts}
    ├── components/
    │   ├── HeadMarkModal/{HeadMarkModal.tsx, drawCanvas.ts}
    │   ├── GlassVideoPlayer.tsx
    │   ├── EggCountResultPanel.tsx
    │   ├── InputPanel.tsx          # 模式页签、文件选择、虫数输入、模型、profile、开始/暂停/继续/取消
    │   ├── ProgressStats.tsx
    │   ├── LiveTrackTable.tsx      # "当前帧 ID 与姿态计数"
    │   ├── SummaryTable.tsx        # "最终输出 ID 与姿态计数"
    │   ├── SubVideoList.tsx
    │   └── HistoryPanel.tsx
    ├── utils/{format.ts, firstFrame.ts}
    └── styles/{index.css, App.css}   # 复制自旧 index.css / App.css，**不改**
```
`npm run build` 与 `npx tsc --noEmit` 必须 0 错误。

## 2. API 层（T28）
- `urls.ts`：`API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:5001'`；`toAbsoluteUrl(u)`（同旧 App.js 14-19：相对路径补前缀）。
- `client.ts`：`axios.create({baseURL: API_BASE})`；函数一一对应契约（`uploadVideo(form)`, `previewCount(blob)`, `analyzeFirstFrame(blob)`, `extractManualWorm(blob,bbox)`, `getStatus(id)`, `taskAction(id,'pause'|'resume'|'cancel')`, `setWormCount(id,n)`, `listSubVideos(id)`, `getModels()`, `selectModel(name)`, `getProfiles()`, `eggCount(file)`），返回类型来自 `types.ts`（直接复制 `API_CONTRACT.md §6`）。
- `socket.ts`：`connectSocket(handlers)` 封装 `io(API_BASE)`、`connect→join(task_id)`、`frame/complete/cancelled/error` 回调；断线重连后对"进行中的任务"重新 `join`（同旧 App.js 863-880）。

## 3. 组件迁移（T29）——逐段对照旧代码，**逻辑不改**
| 新文件 | 旧 App.js 行 | 说明 |
| :-- | :-- | :-- |
| `HeadMarkModal/drawCanvas.ts` | 32-144 `drawCanvas` | 纯函数，签名加类型 |
| `HeadMarkModal/HeadMarkModal.tsx` | 145-470 | 首帧标记/跳过/手动框选/头点选择/虫体启用与标签 |
| `GlassVideoPlayer.tsx` | 471-534 | 播放/倒带/截图/全屏/下载 |
| `EggCountResultPanel.tsx` | 535-723 | 置信度过滤、缩放、拖拽预览 |
| `utils/firstFrame.ts` | 1070-1087 `extractFirstFrame` | 视频取首帧→Blob |
| `utils/format.ts` | 20-31 `formatPercent` 等 | |
测试：每个组件至少一个渲染冒烟测试（vitest + RTL），`HeadMarkModal` 额外测"未给所有已启用虫选头时确认按钮不可用"（旧 337-343 逻辑）。

## 4. Hooks 拆分（T30）——把 `App()` 的约 40 个 `useState` 归并
| hook | 承载的旧逻辑（行） |
| :-- | :-- |
| `useModels` | 777-830（模型列表、切换、`EggCount.pt` 模式联动） |
| `useTaskSocket` | 863-910（frame/complete/cancelled/error → 更新 `currentFrame/currentStats/status`） |
| `useTaskPolling` | 912-1015（`/task/<id>/status` 轮询、完成后拉 sub_videos、恢复刷新前的任务） |
| `useHistory` | 1016-1068 + 849-862（localStorage `motion_history_records`、`motion_active_task_id`、`motion_active_file_name`，**键名保持不变**，使旧历史可读） |
| `usePreviewCount` | 1088-1169（选文件→取首帧→并发 `preview_count` + `analyze_first_frame`→弹出头标记窗口） |
| `useHeadMark` | 1227-1264（确认/跳过、构造 `head_config`：`det_idx, head_kpt_idx, user_label(默认"虫子{idx+1}"), bbox, centroid?, keypoints`） |
| `useTaskControl` | 1265-1320（暂停/继续/取消、历史状态同步、`set_worm_count`） |
| `useEggCount` | 831-848 |
提交任务 `submitTask`（旧 1170-1216）放在 `App.tsx` 或 `useSubmit.ts`：表单字段 `video, worm_count, head_config(JSON), model_name`，**新增 `profile`**。

## 5. 新增 UI（仅两处）
**(a) C 形统计口径**：`InputPanel` 增加下拉"C形统计口径"：`周期（左+右一个来回）`(`cycle`，默认) / `单次弯曲`(`bend`)，作为上传表单字段 `c_unit`（决定服务端 `c_count` 与视频叠加文字）。**(b) 结果表内的即时切换**：`LiveTrackTable`/`SummaryTable` 的"C形"表头加一个小切换（周期 | 单次弯曲），直接读 payload 的 `c_cycles / c_bends`（缺失则回退 `c_count`），**不需要重新处理**。`late_entry===true` 的轨迹在实时表里显示徽章"后进入·不计数"且计数列显示"—"，汇总表默认不含它们。

**(c) 分析环境**：`InputPanel` 的"选择视频"下方增加下拉 **分析环境**：选项来自 `GET /profiles`（`crawl_agar`=琼脂爬行 / `swim_liquid`=液体游泳），默认 `crawl_agar`；失败时回退为这两个硬编码选项。样式复用已有 select 的 className。`TrackTable` 在"状态"列旁无需新增列；若 `behavior` 存在，鼠标悬停徽章显示 `behavior`。

## 6. 功能清单（T30 验收：逐项对照旧前端，全部勾选才算完成）
- [ ] 模式页签：线虫跟踪 / 虫卵计数
- [ ] 多文件选择（跟踪模式）/ 视频或图片（虫卵模式）
- [ ] 选文件后：取首帧 → 预估虫数提示（可点采纳）→ 首帧标记弹窗（选头、启用/禁用、改标签、手动框选新增、跳过）
- [ ] 虫数必填且为正整数，校验提示同旧
- [ ] 开始处理；实时画面、进度条、活跃目标/当前帧/总帧数
- [ ] 暂停/继续/取消；运行中修改虫数上限
- [ ] "当前帧 ID 与姿态计数"表、"最终输出 ID 与姿态计数"表（列同旧）
- [ ] 主视频播放器；下载：主视频/日志/子视频 ZIP/汇总 JSON/摆动 CSV
- [ ] 子视频列表与选择播放
- [ ] 历史记录：持久化、查看/日志/子视频/汇总、点击恢复、刷新页面后自动恢复进行中任务
- [ ] 虫卵计数：上传 → 结果面板（置信度过滤 80–100、缩放、拖拽）
- [ ] 模型下拉与切换
- [ ] socket 断线重连后继续接收
- [ ] 错误提示（后端 `error` 事件与 HTTP 错误）

## 7. 测试与联调（T31）
- `vitest`：hooks 用 mock 的 `client/socket` 测状态流转（`processing → completed`、`error`、重连 `join`）。
- 联调冒烟（手动 + 脚本记录）：启动新后端(5001) + 新前端(3001)，上传 `Swim-4.mp4`，profile=`swim_liquid`，worm_count=1，走完整流程；在 `docs/spec/SMOKE_RESULT.md` 记录截图/日志/耗时。
- 兼容性：**旧前端**（把其 `BACKEND_BASE_URL` 临时指向 5001，仅本地验证，不提交）应可完成一次上传与结果展示——用来证明契约冻结有效。
