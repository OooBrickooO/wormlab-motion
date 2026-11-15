# API 契约（冻结，旧前端与新前端都必须能用）

来源：旧 `backend/app.py` 与 `frontend/src/App.js` 的实际用法。**新增字段一律为可选**，旧前端忽略即可。
新后端默认端口 **5001**（旧 5000），CORS 允许 `http://localhost:3001`、`http://localhost:3000`。

## 1. REST

| 方法 路径 | 请求 | 成功响应 | 错误 |
| :-- | :-- | :-- | :-- |
| `POST /upload` | multipart：`video`(file, 必填, mp4/avi/mov)、`worm_count`(正整数, **必填**)、`head_config`(JSON 字符串, 可选)、`model_name`(可选)、**`profile`(新, 可选: `crawl_agar`\|`swim_liquid`, 默认 `crawl_agar`)**、**`c_unit`(新, 可选: `cycle`\|`bend`, 默认 `cycle`)** | `202 {"task_id": str}` | `400 {"error": str}`（缺文件/类型不允许/worm_count 缺失或非正整数） |
| `POST /preview_count` | multipart：`frame`(jpg) | `{"count":int,"raw_count":int,"annotated":b64jpg}` | 400/500 `{"error"}` |
| `POST /analyze_first_frame` | multipart：`frame` | `{"frame_width","frame_height","annotated":b64jpg,"worms":[{"det_idx","confidence","bbox":[x1,y1,x2,y2],"centroid":[cx,cy],"keypoints":[[x,y]×11]\|null}]}` | 400/500 |
| `POST /extract_manual_worm` | multipart：`frame`、`bbox`(JSON `[x1,y1,x2,y2]`) | `{"bbox":[...int],"centroid":[cx,cy],"keypoints":[[x,y]×11]}`（骨架失败时沿 bbox 对角线均匀 11 点兜底） | 400 `选区过小` 等 |
| `POST /egg_count` | multipart：`file`(视频取首帧/图片) | `{"egg_count","egg_count_90","egg_count_95","egg_class_name","raw_image":b64png,"eggs":[{"bbox","confidence","class_id"}],"frame_width","frame_height"}`；并写 `results/Eggcount/<stem>.json`（`{"1":"93.2%",…}`） | 503 模型缺失 / 500 |
| `GET /models` | — | `{"models":[str],"current_model":str}` | — |
| `POST /select_model` | JSON 或 form：`model_name` | `{"ok":true,"current_model","message"}` | 400/404 |
| `GET /profiles`(新) | — | `{"profiles":[{"id","label"}],"default":"crawl_agar"}` | — |
| `GET /task/<id>/status` | — | 任务状态字典（§3） | 404 `{"error":"Task not found"}` |
| `POST /task/<id>/pause` `resume` `cancel` | — | `{"ok":true,"task_id","status"}`（`paused`/`processing`/`cancelled`） | 404 |
| `POST /task/<id>/set_worm_count` | JSON 或 form：`worm_count`(int；≤0=取消上限) | `{"ok":true,"task_id","worm_count":int\|null,"message"}` | 400 `worm_count 必须为整数` / 404 |
| `GET /task/<id>/sub_videos` | — | `{"videos":[{"filename","url":"/sub_video/<id>/<filename>"}]}`（按文件名排序） | — |
| `GET /download/<filename>` | — | 文件下载（`results/` 平铺目录） | 404 `File not found` |
| `GET /media/<filename>` | — | 在线播放（mp4→`video/mp4`，支持 Range） | 404 |
| `GET /sub_video/<task_id>/<filename>` | — | 子视频文件流 | 404 |
| `GET /health`(新) | — | `{"ok":true,"gpu":bool,"models":[...]}` | — |

`head_config` 元素：`{"det_idx":int,"head_kpt_idx":int,"user_label":str,"bbox":[x1,y1,x2,y2],"centroid":[cx,cy]?,"keypoints":[[x,y]×11]?}`。

## 2. Socket.IO（协议兼容 `socket.io-client` 4.x；服务端 python-socketio 5.x）
- 客户端连接后 `emit("join", {"task_id": str})` → 服务端把该 socket 加入房间 `task_id`。
- 服务端事件（均带 `task_id`，发到房间）：
  - `frame`：`{"task_id","frame":int,"total_frames":int,"image":b64jpg,"stats":{"frame":int,"active_tracks":int,"tracks":[TrackLive]}}`
  - `complete`：`{"task_id","video_url","log_url","sub_videos_url","summary_url","csv_url","id_pose_summary":[TrackSummary]}`
  - `cancelled`：`{"task_id","message"}`；`error`：`{"task_id","message"}`
  - **新（可选）** `phase`：`{"task_id","phase":"detect"|"analyze"|"render","progress":0..1}`
- `frame` 的进度含义：Pass A（检测/跟踪）期间推送；`frame/total_frames` 在 Pass A 结束时到 100%，之后还有 Pass B（运动学+渲染），期间发 `phase` 事件，最后 `complete`。旧前端只认 `frame/complete`，不受影响。

## 3. 任务状态字典（`/task/<id>/status`，运行时在内存，结束写盘）
```jsonc
{ "status": "processing|paused|completed|failed|cancelled",     // 排队中也用 "processing"（附 "queued": true）
  "video_path":"", "filename":"", "current_frame":0, "total_frames":0, "active_tracks":0,
  "latest_frame":"<b64jpg>", "tracks":[TrackLive],
  "video_url":"", "log_url":"", "sub_videos_url":"", "summary_url":"", "csv_url":"",
  "id_pose_summary":[TrackSummary], "error_message":"", "control":"running|paused|cancelled",
  "worm_count": 12,                          // int|null
  // 新增（可选）
  "profile":"crawl_agar", "phase":"detect|analyze|render|done", "queued": false }
```
结束写盘位置：`data/results/tasks/<task_id>.json`（**不含** `latest_frame`）；服务启动时扫描该目录恢复历史（只读）。

## 4. 数据对象

### TrackLive（`frame.stats.tracks[i]`）——旧字段全部保留
```jsonc
{ "track_id":1, "user_label":null, "c_count":0, "s_count":0, "is_bidirectional":false,
  "motion_type":"static|sinusoidal|turning|moving|unknown", "activity_score":0.0, "is_static":false,
  "is_ghost":false, "lost_frames":0, "live_freq":0.0, "avg_freq":0.0, "cycles":0, "swings":0, "bends_per_min":0.0,
  "centroid_bl":{"x":0,"y":0}|null, "head_bl":{"x":0,"y":0}|null, "head_kpt_idx":null,
  "keypoints_bl":[{"x":0,"y":0}×11]|null,
  // 新增（可选）
  "behavior":"crawl", "s_cycles":0.0, "freq_s_hz":0.0, "freq_c_hz":0.0, "pose_quality":"OK",
  "c_cycles":0, "c_bends":0, "c_unit":"cycle",   // c_count = 所选口径(+深弯折返)；两种口径始终都给
  "counted":true, "late_entry":false }          // late_entry=true：后进入，计数全 0
```
`*_bl` = 左下角原点坐标（`y_bl = H - y`），仅 `compat/legacy_payload.py` 生成。

### TrackSummary（`id_pose_summary[i]`、`*_id_pose_summary.json`）
`track_id,user_label,c_count,s_count,is_bidirectional,motion_type,activity_score,is_static,live_freq,avg_freq,cycles,swings,bends_per_min` + `spatial_stats{total_path_length_px,net_displacement_px,start_centroid_bl:[x,y],end_centroid_bl:[x,y],bbox_coverage_bl:{x_min,y_min,x_max,y_max}}`（沿用旧 `build_id_pose_summary` 的算法）。新增可选：`s_cycles,c_cycles,c_bends,c_unit,c_turn_count,freq_s_hz,freq_c_hz,mode_time,rejected`。**后进入的虫默认不出现在汇总里**；`is_bidirectional` 恒为 false（只统计向前 S，字段仅为兼容保留）。

### 字段兼容映射（`compat/legacy_payload.py`）
| 旧字段 | 新来源 |
| :-- | :-- |
| `c_count` / `s_count` | `TrackReport.c_count`（所选 c_unit）/ `.s_count`（仅向前 S） |
| `is_bidirectional` | 恒 `false` |
| `c_cycles` / `c_bends` | `TrackReport.c_cycles` / `.c_bends`（新） |
| `cycles` `swings` | `c_count + s_count` |
| `avg_freq` `live_freq` | `(c_count+s_count)/observed_seconds`（旧语义，保留） |
| `bends_per_min` | `avg_freq*60` |
| `motion_type` | 行为→旧字符串映射（03 §7） |
| `is_static` | `behavior=="static"` |
| `activity_score` | `TrackReport.activity_score` |
| `centroid_bl` | `dense.mean(0)` 换算 |
| `head_bl` | `dense[0]` 换算 |
| `keypoints_bl` | `pts11` 换算 |

## 5. 结果文件（沿用旧命名，保证 URL 兼容）
`data/results/<stem>_result_<task_id>.mp4`、`<…>_log.json`（`[{"frame","active_tracks","tracks":[TrackLive]}]`）、`<…>_id_pose_summary.json`、`<stem>_result_<task_id>_swing_summary.csv`（列同旧，utf-8-sig）、`<task_id>_sub_videos.zip`、`sub_videos/<task_id>/worm_<id>_cycles_<n>.mp4`。重名用 `ensure_unique_path` 追加后缀。

## 6. TypeScript 类型（前端 `src/api/types.ts` 直接照抄）
```ts
export type TaskStatus = 'processing'|'paused'|'completed'|'failed'|'cancelled';
export interface Pt { x:number; y:number }
export interface TrackLive { track_id:number; user_label:string|null; c_count:number; s_count:number; is_bidirectional:boolean;
  motion_type:string; activity_score:number; is_static:boolean; is_ghost:boolean; lost_frames:number;
  live_freq:number; avg_freq:number; cycles:number; swings:number; bends_per_min:number;
  centroid_bl:Pt|null; head_bl:Pt|null; head_kpt_idx:number|null; keypoints_bl:Pt[]|null;
  behavior?:string; s_cycles?:number; freq_s_hz?:number; freq_c_hz?:number; pose_quality?:string;
  c_cycles?:number; c_bends?:number; c_unit?:'cycle'|'bend'; counted?:boolean; late_entry?:boolean }
export interface SpatialStats { total_path_length_px:number; net_displacement_px:number;
  start_centroid_bl:[number,number]; end_centroid_bl:[number,number];
  bbox_coverage_bl:{x_min:number;y_min:number;x_max:number;y_max:number} }
export interface TrackSummary extends Omit<TrackLive,'is_ghost'|'lost_frames'|'centroid_bl'|'head_bl'|'head_kpt_idx'|'keypoints_bl'|'behavior'|'pose_quality'> { spatial_stats:SpatialStats }
export interface FrameStats { frame:number; active_tracks:number; tracks:TrackLive[] }
export interface FrameEvent { task_id:string; frame:number; total_frames:number; image:string; stats:FrameStats }
export interface CompleteEvent { task_id:string; video_url:string; log_url:string; sub_videos_url:string; summary_url:string; csv_url:string; id_pose_summary:TrackSummary[] }
export interface TaskState { status:TaskStatus; filename:string; current_frame:number; total_frames:number; active_tracks:number;
  latest_frame:string; tracks:TrackLive[]; video_url:string; log_url:string; sub_videos_url:string; summary_url:string; csv_url:string;
  id_pose_summary:TrackSummary[]; error_message:string; control:string; worm_count:number|null; profile?:string; phase?:string; queued?:boolean }
export interface FirstFrameWorm { det_idx:number; confidence:number|null; bbox:[number,number,number,number]; centroid:[number,number]; keypoints:[number,number][]|null }
export interface FirstFrameData { frame_width:number; frame_height:number; annotated:string; worms:FirstFrameWorm[] }
export interface HeadConfigItem { det_idx:number; head_kpt_idx:number; user_label:string; bbox:number[]; centroid?:[number,number]; keypoints?:[number,number][] }
```
