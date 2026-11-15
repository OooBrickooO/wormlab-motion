# 05 流水线 · 渲染 · 导出 · 服务

对应任务：T23–T27。关键决策已定：**两遍式（D5）**、**常驻 GPU worker + 任务队列（D2）**、**FastAPI + python-socketio（D1）**、**运行时内存、结束落盘（D7）**。接口契约见 `docs/API_CONTRACT.md`。

```
Pass A (GPU, 一次)：读帧 → YOLO 批推理 → 跟踪 → 姿态 → 流式运动学(实时计数) → 实时画面推送 → 写缓存 tracks.npz
Pass B (CPU, 可重跑)：读缓存 → [可选缝合] → 批量运动学(终稿) → 重渲染主视频/子视频 → 导出 json/csv/zip
```

## 1. 缓存格式（`pipeline/cache.py`）
`data/cache/<task_id>/tracks.npz` + `meta.json`。长格式：每行 = 一条轨迹在一帧的记录（含幽灵帧）。
```
frame_idx  int32 (M,)      track_id  int32 (M,)      track_uid int32 (M,)   # uid=实例唯一号，Pass B 按 uid 分组      bbox  float32 (M,4)      conf float32 (M,)
is_ghost   bool  (M,)      quality   int8 (M,)  (Quality 枚举；无姿态=3)
dense      float32 (M,33,2)   无姿态为 NaN
length chord width area  float32 (M,)
flags      uint16 (M,)     位序见 schemas.FLAG_BITS = ["loop","branch","fragment","fallback","empty","merge","clipped","upscaled","reoriented_after_gap"]
contour_xy float32 (K,2)   所有行轮廓顺序拼接；contour_off int64 (M+1,) 为偏移（无轮廓的行长度 0）
```
`meta.json`：`{schema_version:1, fps, width, height, total_frames, video_path, model, profile, cfg_hash, worm_count, head_config, user_labels:{id:label}, head_kpt_idx:{id:int}, created_at}`。
```python
class CacheWriter:  append(frame_idx, track_out, pose|None); close(meta) -> Path   # 内存列表 → 末尾 np.savez_compressed
class CacheReader:  meta; tracks() -> dict[int, TrackRows]; frame_rows(frame_idx) -> list[Row]   # 预建 frame→行索引
```
单测：写 → 读往返逐字段相等（含 NaN、空轮廓）。

## 2. Pass A（`pipeline/pass_a.py`）
```python
class PassA:
    def __init__(self, cfg, job: JobSpec, detector: Detector, sink: EventSink, ctrl: Control): ...
    def run(self) -> Path:   # 返回缓存目录
```
`JobSpec`：`task_id, video_path, profile, c_unit('cycle'|'bend'), worm_count|None, head_config:list, model_path, result_dir`。
`Control`：线程安全对象，`state ∈ {running,paused,cancelled}`，`set_worm_count(n)` 队列；方法 `wait_if_paused()`、`cancelled`。
`EventSink`：`emit(event, data, room=None)`、`update_state(**kw)`（worker 里实现为写 `evt_q`；测试里是内存假对象）。

**帧源抽象**：`VideoSource(path)`（`fps, w, h, total`，`read_batch(B)`）与 `ImageSource(path, repeat=36, fps=12)`（把单张图当 36 帧短视频，沿用旧行为）。

**线程模型（沿用旧结构，4 线程）**
```
reader  : source.read_batch(B) → frame_q(maxsize=cfg.pipeline.frame_queue)；pause 时 sleep(0.05)
main    : 取批 → detector.infer(frames) → infer_q(maxsize=…)；处理 control（pause/cancel/set_worm_count）
tracker : 取 (frames, dets_per_frame, start_idx) → 逐帧：
            outs = tracker.update(dets, idx)
            poses = pose_pool.map(extract per track)       # ThreadPoolExecutor(cfg.pipeline.pose_workers)
            for each out: engine[tid].push(idx, pose or None); tracker.feed_pose(tid, pose.dense)
            cache.append(...)
            if idx % emit_every == 0: emit_q.put_nowait((frame_copy, items, idx))   # 队列满则丢帧，不阻塞
            sink.update_state(current_frame=..., active_tracks=...)
emitter : 取 → render.overlay.draw_frame(...) → JPEG(质量80) → base64 → sink.emit('frame', ...)
```
- 每个 track 一份 `PoseExtractor` 与 `KinematicsEngine`（`dict[tid]`）；`Track` 被回收后保留其 engine 到结束（供 Pass B 前不丢数据）——**Pass B 会用缓存重算，所以 Pass A 里的 engine 只服务实时计数**。
- 实时画面计数用 `engine.snapshot()`（滞后约 1.5 s，可接受）；`frame.stats.tracks` 由 `compat.to_legacy_track(...)` 生成。
- `set_worm_count(n)` → `tracker.set_max_worm_id(n or None)`。
- 取消：`stop_event`；各线程用 `put(timeout=1.0)` 循环检查，避免死锁（同旧）。
- 结束：关闭线程池，`cache.close(meta)`；`sink.emit('phase', {phase:'analyze', progress:0})`。
- 异常：捕获 → `update_state(status='failed', error_message=...)` + `emit('error')`；`finally` 释放 capture/线程池。
验收（T24）：无服务器，`python -m wormlab.pipeline.pass_a --video Move-3.mp4 --max-frames 300 --profile crawl_agar` 产出 `tracks.npz`，且行数 > 0、`frame_idx` 单调。

## 3. 渲染（`wormlab/render/`）
```python
@dataclass
class DrawItem: track_id; bbox; is_ghost; lost_frames; user_label; contour(K,2)|None; head_xy|None; trail:list[(x,y)]  # 最近30个 bbox 中心
                counted: bool                      # False=后进入的虫：画灰框+"ID n 后进入·不计数"，不显示计数
                counts: dict(c_count,s_count,is_bidirectional(恒False),avg_freq,bends_per_min,motion_type_legacy,activity_score)
def draw_frame(frame_bgr, items, frame_idx, active_count, ghost_count) -> np.ndarray   # overlay.py
```
**视觉 1:1 移植旧实现**（`worm_processing.py` 2263-2423、2425-2435）：
1. 轨迹尾迹：最近 30 点，`(0, 255*j/n, 0)` 渐变抗锯齿线，宽 2；跳过 ghost。
2. 轮廓：白色 `(255,255,255)` 宽 2 `LINE_AA`（**数据来自缓存 `contour_xy`**，不再依赖整帧掩码；旧代码的"高斯模糊后取轮廓"已在姿态层 02 §4.5 完成）。
3. bbox：普通白色宽 2；ghost 用虚线框 `(160,220,160)` 宽 1（`_draw_dashed_rect`，dash=10, gap=6）。
4. 药丸徽章（PIL，字体 15/13，候选 `C:/Windows/Fonts/msyhbd.ttc, msyh.ttc, simsun.ttc, arial.ttf`，`fonts.py` 用 `functools.lru_cache`）：顶部 `ID {id} [{label}] C:{c} S:{s}{ (双向)} {freq:.2f}Hz`；底部 `#C:{c} S:{s} {freq:.2f}Hz {bpm:.0f}/min {motion_type} {activity:.0%}`；ghost 用浅绿底 `ID {id} GHOST lost={n}f`。
5. 头点：有 `head_xy` 画半径 3 的 `(0,255,100)` 实心圆。
6. 左上角 `Frame: N`、`Active IDs: a (Ghost: g)`，黄 `(0,255,255)` 字号 0.7 宽 2。
验收（T23）：用固定输入渲染一帧，保存到 `data/diag/render_new.jpg`；同一输入用旧函数（只读导入）渲染 `render_old.jpg`，肉眼一致（文字/布局/颜色）。

`render/subvideo.py`：
```python
class StableCrop:   # 复刻 KalmanBoxTracker 的 crop 状态机（1137、1169、1183-1184、1192-1220）
    center, size; def on_detect(bbox): center=0.8c+0.2*b; size=0.85s+0.15*max(w,h)*1.8
                    def on_predict(bbox): center=0.9c+0.1*b          # ghost 帧
                    def window(W,H,scale=1.35,min_size=72) -> (x1,y1,x2,y2)
def crop_roi(frame, window, out=(200,200)) -> np.ndarray | None   # cv2.resize
class SubVideoWriter: 每 track 一个 mp4（200×200，fps=视频 fps），close()
```
`render/video_io.py`：`create_compatible_video_writer(path, fps, size)`（codec 顺序 `avc1,H264,X264,mp4v`，同 app.py 329-342；Windows 需项目根的 `openh264-1.8.0-win64.dll` 在 PATH——**在 `video_io` 里启动时把 `re_struction/../openh264-1.8.0-win64.dll` 所在目录加入 `os.add_dll_directory`/PATH，仅当文件存在**）、`ensure_unique_path`、`safe_rename`（同 app.py 311-326）。

## 3.5 "后进入的虫一律不计数"（A 与 B 共用 `pipeline/late_entry.py`）
```python
def is_counted(first_frame: int, fps: float, cfg) -> bool:
    return first_frame <= round(cfg.counting.late_entry_grace_s * fps)      # 默认 1.0 s，容忍首帧漏检
```
- 以**轨迹实例 uid** 的首次出现帧为准（不是 track_id，ID 池会回收复用）。同一 uid 丢失后找回仍算初始虫；人工植入轨迹 `first_frame=0`。
- **Pass A 实时**：新建轨迹时即可判定（`frame_idx > grace` 的新轨迹立即标记 late）。late 轨迹仍做姿态/运动学（几乎无开销，便于调试），但：`counted=False`，对外计数一律输出 0。
- **Pass B 终稿**：`counted=False` 的轨迹 → 汇总 JSON/CSV **默认不出现**（`counting.include_late_in_outputs=false`；为 true 时出现但计数全 0 且带 `late_entry:true`）；主视频里画灰色 `(150,150,150)` 框 + 标签 `ID n 后进入·不计数`；**不生成子视频**。
- 同一 `track_id` 对应多个 uid（ID 被回收复用）：汇总只保留 `counted` 且持续帧数最长的那个 uid，其余记入日志 json 的 `dropped_uids`。

## 4. Pass B（`pipeline/pass_b.py`）
```python
class PassB:
    def run(self, cache_dir, job: JobSpec, sink, ctrl) -> Result:
        reader = CacheReader(cache_dir)
        rows = reader.tracks()
        remap = stitch(rows, cfg) if cfg.tracking.stitch.enabled else {}        # 默认不做
        # 1) 终稿运动学
        reports, perframes = {}, {}
        for uid, tr in rows.items():                              # rows 按 track_uid 分组
            tid = tr.track_id; counted = is_counted(tr.first_frame, fps, cfg)
            poses = tr.as_pose_list(fill_missing_frames=True)      # 首帧~末帧，缺失帧为 None；幽灵帧也是 None
            reports[uid], perframes[uid] = KinematicsEngine.run(poses, fps, cfg_kin)   # c_unit 取自 job.c_unit
            sink.emit('phase', {phase:'analyze', progress: k/n})
        # 2) 重渲染主视频 + 子视频（再次解码原视频）
        for frame_idx, frame in decode(video):
            items = [DrawItem(... counts 来自 perframes[tid][idx - first_idx] ...) for row in reader.frame_rows(idx)]
            main_writer.write(draw_frame(frame, items, idx, ...))
            for each non-ghost-or-ghost row: sub_writers[tid].write(crop_roi(frame, StableCrop(tid).window(...)))
            sink.emit('phase', {phase:'render', progress: idx/total})
        # 3) 关闭 writer；子视频重命名 worm_{id}_cycles_{c+s}.mp4；打包 zip
        # 4) 日志 json、summary json、csv；返回 Result(url 字典, id_pose_summary)
```
- 渲染里每帧的计数 = 该轨迹 `perframes[tid]` 在该帧的 `s_count/c_count/behavior`（终稿，**非因果平滑后的累计值**）；显示频率 = `(c+s)/该帧已观察秒数`（旧语义）。
- 日志 json（`_log.json`）每帧：`{"frame", "active_tracks", "tracks":[TrackLive...]}`，TrackLive 由 compat 用 perframes 生成。
- 取消：每 30 帧检查一次 `ctrl.cancelled`。
- 渲染里 C 显示：`C:{c_count}`（所选口径）。
验收（T25）：对 T24 的缓存运行，得到 mp4（可播放）、sub_videos（每个 track 一个）、json/csv/zip，字段与 `API_CONTRACT.md` 逐项对照（写一个 schema 校验函数 `validate_summary(obj)` 并在测试里调用）。

## 5. 兼容层（`wormlab/compat/legacy_payload.py`）
```python
def to_legacy_track(tid, label, behavior_state, report_like, pose, frame_h, is_ghost, lost_frames, head_kpt_idx) -> dict   # TrackLive
def to_legacy_summary(reports, spatial, labels, frame_h) -> list[dict]                                                    # TrackSummary，按 track_id 升序
def motion_type_legacy(behavior) -> str     # 03 §7 映射
```
`report_like`：实时用 `engine.snapshot()`，终稿用 `PerFrame`。映射表见 `API_CONTRACT.md §4`。所有 `y_bl = frame_h - y`。
单测：构造一条轨迹，断言输出键集合**包含**旧字段全集（硬编码一份旧键清单对比）。

## 6. 导出（`server/exporters/`）
- `summary_json.write(path, id_pose_summary)`：`json.dump(ensure_ascii=False)`。
- `csv_report.write(result_dir, stem, id_pose_summary) -> filename`：编码 `utf-8-sig`，**表头与旧版完全一致**：`track_id,用户标签,运动状态,C形计数,S形计数,是否双向,平均摆动频率(Hz),实时摆动频率(Hz),完整周期数,单向摆动数,次/分钟,活性分(0-1),总运动路径(px),净位移(px)`（旧 `write_swing_summary_csv`，app.py 274-304），**之后追加两列**：`C形周期数,C形单次弯曲数`（旧列顺序/含义不变，`C形计数`=所选口径；`是否双向`恒为"否"）。
- `zip_pack.write(zip_path, sub_dir)`：`ZipFile(..., 'w')`，`arcname=file`。

## 7. 服务（FastAPI）`server/`
```
app_factory.create_app(cfg) -> (fastapi_app, sio, asgi_app)
  - CORS(cfg.server.cors_origins)；挂载路由；sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
  - asgi_app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)      # uvicorn 运行 asgi_app
  - lifespan：启动 EventPump 线程、WorkerProcess、TaskStore.load_history()；关闭时优雅停止
run.py: uvicorn.run(asgi_app, host=cfg.server.host, port=cfg.server.port)    # 单进程；不要 reload
```
路由文件对应：`upload.py`(/upload)、`preview.py`(/preview_count /analyze_first_frame /extract_manual_worm)、`task.py`(status/pause/resume/cancel/set_worm_count/sub_videos)、`media.py`(/download /media /sub_video)、`models.py`(/models /select_model /profiles /health)、`egg.py`(/egg_count)。语义、字段、错误码**逐条以 `API_CONTRACT.md` 和旧 `app.py` 对应段落为准**（旧行号：upload 875-964、analyze_first_frame 966-1092、extract_manual_worm 1095-1164、preview_count 1167-1262、download/media 1265-1290、task 1292-1403、models 1406-1444、egg 1447-1581）。

`/upload` 流程：校验（文件、类型、worm_count 正整数、profile ∈ 已有 profile、`c_unit ∈ {cycle,bend}` 可选默认取配置）→ `task_id=generate_task_id()`（`YYYYmmdd_HHMMSS_<6hex>`）→ 保存到 `data/uploads/<task_id>_<secure_filename>`（**防重名覆盖**）→ `TaskStore.create(...)` → `TaskManager.submit(job)` → `202 {task_id}`。`model_name` 解析：存在于 `models_dir` 则用，否则用 registry 当前模型。

## 8. 任务管理与 worker（`server/tasks/`）
**为什么是常驻 worker**：模型只加载一次（5070 Ti 16GB），任务串行排队，避免旧版"每任务一个进程、各载一份模型"。
```python
class WorkerProcess:       # multiprocessing.get_context("spawn").Process(target=worker_main)
    cmd_q: mp.Queue         # server → worker
    evt_q: mp.Queue         # worker → server
def worker_main(cmd_q, evt_q, cfg_dict):
    registry = {}            # model_path -> Detector（只缓存最近一个，换模型时释放 GPU 缓存 torch.cuda.empty_cache()）
    while True:
        msg = cmd_q.get()
        if msg["type"] == "shutdown": break
        if msg["type"] == "start":
            job = JobSpec(**msg["job"]); ctrl = Control(); sink = QueueSink(evt_q, job.task_id)
            # 控制消息监听线程：持续读 cmd_q，type=="control" 且 task_id 匹配 → ctrl.apply(op, value)
            try: cache = PassA(...).run(); PassB(...).run(...)
            except Exception: sink.update_state(status="failed", error_message=str(e)); sink.emit("error", ...)
            finally: evt_q.put({"type":"done","task_id":job.task_id})
class TaskManager:          # 在 server 进程
    queue: deque[JobSpec]; current: str|None
    submit(job): 入队；若 worker 空闲立即 start；排队中的任务状态 status="processing", queued=True
    control(task_id, op, value): 
        - 任务在队列中：cancel → 直接从队列移除并置 cancelled；pause/resume 仅改状态；set_worm_count 更新 job.worm_count
        - 任务运行中：转发 {"type":"control", ...} 到 cmd_q
    on_done(task_id): current=None；取下一个
```
`evt_q` 消息格式（沿用旧 progress_q）：`{"type":"emit","task_id","event","data","room"}`、`{"type":"update_state","task_id","kwargs"}`、`{"type":"done","task_id"}`。
`EventPump`（server 进程里的守护线程）：循环 `evt_q.get()` → `emit` 用 `asyncio.run_coroutine_threadsafe(sio.emit(event, data, room=room), loop)`；`update_state` 调 `TaskStore.update`；`done` 调 `TaskManager.on_done` 并触发落盘。

## 9. 状态、落盘与恢复（`TaskStore`）
- 运行时：`dict[str, dict]` + `threading.Lock`（结构同 `API_CONTRACT.md §3`）。
- 任务进入终态（completed/failed/cancelled）时写 `data/results/tasks/<task_id>.json`（去掉 `latest_frame`），写临时文件再 `os.replace`。
- 启动时扫描该目录加载为只读历史（`control` 置为终态）。
- `GET /task/<id>/status` 对内存/历史一视同仁；`404` 仅当两处都没有。
- **验收（T26）** `tests/integration/test_contract.py`：用 `fastapi.testclient` 起应用，worker 换成**假 worker**（不加载 YOLO，按脚本推 `frame`/`complete`），逐路由断言状态码与字段集合；再加一个落盘/重启恢复用例（写任务→新建 `TaskStore`→读到相同状态）。

## 10. 模型与预览（`services/`）
```python
class ModelRegistry:      # server 进程内，线程安全
    list_models() -> list[str]；current_name；set_current(name)（切换时清空预览模型缓存）
    get_preview_detector() -> Detector（懒加载，带 lock）；get_egg_model() -> YOLO（EggCount.pt，缺失 → FileNotFoundError → 503）
    model_path(name|None) -> str                  # 偏好顺序见 detection.model_preference
```
- 预览/首帧分析/虫卵计数的推理放在 **server 进程**（独立于 worker 的任务模型；显存多占一份小模型可接受）。推理处加 `registry.lock`，防并发。
- `/analyze_first_frame`：Detector.infer([frame]) → 对每个检测调 `extract_pose` → `pts11`；标注图沿用旧画法（supervision 的 Mask/Polygon annotator + 蓝框 `(22,119,255)` + 节点序号，头尾点绿色）。`det_idx` 为过滤后序号。
- `/extract_manual_worm`：ROI 局部阈值（旧 1127-1140 逻辑逐行照搬）→ `extract_pose(mask, origin=(rx1,ry1))`；失败时沿 bbox 对角线兜底 11 点。
- `/egg_count`：照搬旧 1447-1581（含 `results/Eggcount/<stem>.json`）。
