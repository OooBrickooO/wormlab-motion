import os
import base64
import json
import csv
import threading
from multiprocessing import Process, Queue, Manager
import queue
import zipfile
import io
import time
import uuid
import cv2
from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO, emit, join_room
from flask_cors import CORS
from werkzeug.utils import secure_filename
from worm_processing import (
    init_model, process_frame, process_frame_batch, process_track_single,
    EnhancedWormTracker, MIN_AREA_RATIO, MIN_CONFIDENCE,
    extract_centerline_from_mask,
)
import numpy as np
import threading
import queue

# ==================== 配置 ====================
app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['SUB_VIDEO_FOLDER'] = os.path.join(app.config['RESULT_FOLDER'], 'sub_videos')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs(app.config['SUB_VIDEO_FOLDER'], exist_ok=True)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 全局模型（在启动时加载一次）
model, device = None, None

def get_models_dir():
    """获取 models 目录路径"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    if os.path.exists(models_dir):
        return models_dir
    rel_models_dir = os.path.join(os.getcwd(), 'models')
    if os.path.exists(rel_models_dir):
        return rel_models_dir
    return models_dir

def list_available_models():
    """扫描 models 目录下所有的 .pt 模型文件"""
    mdir = get_models_dir()
    if not os.path.exists(mdir):
        return []
    files = [f for f in os.listdir(mdir) if f.endswith('.pt') and os.path.isfile(os.path.join(mdir, f))]
    files.sort()
    return files

def get_current_model_name():
    """获取当前选择的模型文件名"""
    curr = globals().get('global_model_path', '')
    if curr and os.path.exists(curr):
        return os.path.basename(curr)
    models = list_available_models()
    return models[0] if models else 'best.pt'

def get_default_model_path():
    mdir = get_models_dir()
    models = list_available_models()
    if models:
        preferred = ["origin.pt", "OnlineVision.pt", "CJLUdata_Old.pt", "CJLUdata.pt", "best.pt"]
        for p in preferred:
            if p in models:
                return os.path.join(mdir, p)
        return os.path.join(mdir, models[0])
    return os.path.join(mdir, 'best.pt')

global_model_path = get_default_model_path()

# ── 预览用模型缓存（惰性加载，首次调用 /preview_count 时初始化）──────────────
_preview_model = None
_preview_device = None
_preview_model_lock = threading.Lock()

def get_preview_model():
    """线程安全地获取预览推理模型（全局只加载一次）。"""
    global _preview_model, _preview_device
    with _preview_model_lock:
        if _preview_model is None:
            model_path = globals().get('global_model_path') or get_default_model_path()
            print(f"[preview] 加载预览模型: {model_path}")
            _preview_model, _preview_device = init_model(model_path)
            print("[preview] 预览模型加载完成")
    return _preview_model, _preview_device

# ── 虫卵计数专用模型缓存（EggCount.pt，task=detect，classes: 0=egg,1=worm）────
EGG_COUNT_MODEL_NAME = 'EggCount.pt'
_egg_model = None
_egg_device = None
_egg_model_lock = threading.Lock()

def get_egg_count_model():
    """线程安全地懒加载 EggCount.pt 模型。"""
    global _egg_model, _egg_device
    with _egg_model_lock:
        if _egg_model is None:
            mdir = get_models_dir()
            egg_path = os.path.join(mdir, EGG_COUNT_MODEL_NAME)
            if not os.path.exists(egg_path):
                raise FileNotFoundError(f"未找到虫卵计数模型: {egg_path}，请将 {EGG_COUNT_MODEL_NAME} 放入 models/ 目录")
            print(f"[egg_count] 加载虫卵计数模型: {egg_path}")
            _egg_model, _egg_device = init_model(egg_path)
            print(f"[egg_count] 模型加载完成，类别: {_egg_model.names}")
    return _egg_model, _egg_device

# 存储每个任务的状态（可选）
processing_tasks = {}
task_lock = threading.Lock()

progress_queue = Queue()
# Dictionary to hold control queues for each task (task_id -> Queue)
# We need this to send pause/resume/cancel to processes
control_queues = {}

def queue_listener():
    while True:
        try:
            msg = progress_queue.get()
            if msg is None:
                break
            
            task_id = msg.get('task_id')
            msg_type = msg.get('type')
            
            if msg_type == 'update_state':
                update_task_state(task_id, **msg.get('kwargs', {}))
            elif msg_type == 'emit':
                socketio.emit(msg.get('event'), msg.get('data'), room=msg.get('room'))
        except Exception as e:
            print(f"Queue listener error: {e}")

listener_thread = threading.Thread(target=queue_listener, daemon=True)
listener_thread.start()



def update_task_state(task_id, **kwargs):
    with task_lock:
        if task_id not in processing_tasks:
            return
        processing_tasks[task_id].update(kwargs)


def get_task_state(task_id):
    with task_lock:
        state = processing_tasks.get(task_id)
        if state is None:
            return None
        return dict(state)


def get_task_control(task_id):
    with task_lock:
        state = processing_tasks.get(task_id)
        if state is None:
            return None
        return state.get('control', 'running')



def wait_or_cancel_process(task_id, control_q):
    """进程内部使用的等待/取消逻辑"""
    try:
        # non-blocking get
        control = control_q.get_nowait()
        if control == 'cancelled':
            return False, 'cancelled'
        if control == 'paused':
            return False, 'paused'
        if control == 'running':
            return True, 'running'
    except queue.Empty:
        return True, 'running'
    return True, 'running'

def wait_or_cancel(task_id):
    """返回 False 表示任务被取消，True 表示可以继续处理。"""
    while True:
        control = get_task_control(task_id)
        if control is None:
            return False
        if control == 'cancelled':
            update_task_state(task_id, status='cancelled', error_message='任务已取消')
            socketio.emit('cancelled', {'task_id': task_id, 'message': '任务已取消'}, room=task_id)
            return False
        if control == 'paused':
            update_task_state(task_id, status='paused')
            time.sleep(0.15)
            continue
        if control == 'running':
            state = get_task_state(task_id)
            if state and state.get('status') == 'paused':
                update_task_state(task_id, status='processing')
            return True
        return True


def build_id_pose_summary(tracker):
    summary = []
    for track_id, cycle_info in sorted(tracker.cycle_counts.items(), key=lambda x: int(x[0])):
        # 空间统计：优先采用经过姿态/跳变校验的 TrackHistory.centers，回退到 track_history
        th_obj = getattr(tracker, 'track_histories', {}).get(track_id)
        if th_obj is not None and len(getattr(th_obj, 'centers', [])) > 1:
            history = list(th_obj.centers)
        else:
            history = list(tracker.track_history.get(track_id, []))
        spatial_stats = {}
        if len(history) > 1:
            pts_arr = np.array(history, dtype=float)
            diffs = np.diff(pts_arr, axis=0)
            seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
            total_path = float(np.sum(seg_lens))
            net_disp = float(np.linalg.norm(pts_arr[-1] - pts_arr[0]))
            h = tracker.height
            start_bl = [float(pts_arr[0][0]), float(h - pts_arr[0][1])]
            end_bl = [float(pts_arr[-1][0]), float(h - pts_arr[-1][1])]
            x_vals = pts_arr[:, 0]
            y_vals = pts_arr[:, 1]
            spatial_stats = {
                'total_path_length_px': round(total_path, 2),
                'net_displacement_px': round(net_disp, 2),
                'start_centroid_bl': [round(v, 2) for v in start_bl],
                'end_centroid_bl': [round(v, 2) for v in end_bl],
                'bbox_coverage_bl': {
                    'x_min': round(float(x_vals.min()), 2),
                    'y_min': round(float(h - y_vals.max()), 2),
                    'x_max': round(float(x_vals.max()), 2),
                    'y_max': round(float(h - y_vals.min()), 2),
                },
            }

        # 当前轨迹的用户标签
        user_label = None
        for t in tracker.tracks:
            if t.track_id == track_id:
                user_label = t.user_label
                break

        summary.append({
            'track_id': int(track_id),
            'user_label': user_label,
            'c_count': int(cycle_info.get('c_count', 0)),
            's_count': int(cycle_info.get('s_count', 0)),
            'is_bidirectional': bool(cycle_info.get('is_bidirectional', False)),
            'motion_type': str(cycle_info.get('motion_type', 'unknown')),
            'activity_score': float(cycle_info.get('activity_score', 0.0)),
            'is_static': bool(cycle_info.get('is_static', False)),
            'live_freq': float(cycle_info.get('live_freq', 0.0)),
            'avg_freq': float(cycle_info.get('avg_freq', 0.0)),
            'cycles': int(cycle_info.get('cycles', 0)),
            'swings': int(cycle_info.get('swings', 0)),
            'bends_per_min': float(cycle_info.get('bends_per_min', 0.0)),
            'spatial_stats': spatial_stats,
        })
    return summary


def write_swing_summary_csv(id_pose_summary, result_folder, stem):
    """把每只虫的摆动与 C/S 汇总写成 CSV（Excel 可直接打开），返回文件名。"""
    filename = f"{stem}_swing_summary.csv"
    path = ensure_unique_path(os.path.join(result_folder, filename))
    filename = os.path.basename(path)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            'track_id', '用户标签', '运动状态', 'C形计数', 'S形计数', '是否双向',
            '平均摆动频率(Hz)', '实时摆动频率(Hz)', '完整周期数', '单向摆动数',
            '次/分钟', '活性分(0-1)', '总运动路径(px)', '净位移(px)'
        ])
        for item in id_pose_summary:
            sp = item.get('spatial_stats', {})
            writer.writerow([
                item.get('track_id', ''),
                item.get('user_label', '') or '',
                item.get('motion_type', 'unknown'),
                item.get('c_count', 0),
                item.get('s_count', 0),
                '是' if item.get('is_bidirectional') else '否',
                round(item.get('avg_freq', 0.0), 2),
                round(item.get('live_freq', 0.0), 2),
                item.get('cycles', 0),
                item.get('swings', 0),
                round(item.get('bends_per_min', 0.0), 1),
                round(item.get('activity_score', 0.0), 2),
                sp.get('total_path_length_px', 0.0),
                sp.get('net_displacement_px', 0.0),
            ])
    return filename


def generate_task_id():
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def ensure_unique_path(path):
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    idx = 1
    while True:
        candidate = f"{root}_{idx}{ext}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def safe_rename(src, dst):
    unique_dst = ensure_unique_path(dst)
    os.rename(src, unique_dst)
    return unique_dst


def create_compatible_video_writer(path, fps, frame_size):
    codecs = ['avc1', 'H264', 'X264', 'mp4v']
    last_error = None
    for codec in codecs:
        try:
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, frame_size)
            if writer is not None and writer.isOpened():
                print(f"🎬 VideoWriter codec: {codec}")
                return writer
            if writer is not None:
                writer.release()
        except Exception as e:
            last_error = e
    raise RuntimeError(f"无法创建视频写入器: {last_error}")


# ==================== 辅助函数 ====================
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def is_image_file(filename):
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    return ext in {'jpg', 'jpeg', 'png', 'bmp'}

def process_video_task(task_id, video_path, progress_q, control_q, app_config):
    """后台进程处理视频，通过 Queue 发送结果到主进程"""
    def emit(event, data, room=None):
        progress_q.put({'type': 'emit', 'task_id': task_id, 'event': event, 'data': data, 'room': room})

    def update_state(**kwargs):
        progress_q.put({'type': 'update_state', 'task_id': task_id, 'kwargs': kwargs})

    print(f"开始处理任务 {task_id}, 视频路径: {video_path}")
    update_state(status='processing', current_frame=0, total_frames=0, active_tracks=0)
    
    cap = None
    out_writer = None
    tracker = None
    model = None
    device = None
    
    try:
        # 在子进程中加载模型
        model_path = app_config.get('MODEL_PATH', 'models/best.pt')
        print(f"正在进程中加载模型: {model_path}")
        model, device = init_model(model_path)
        
        if is_image_file(video_path):
            frame = cv2.imread(video_path)
            if frame is None:
                update_state(status='failed', error_message='无法读取图片文件')
                emit('error', {'task_id': task_id, 'message': '无法读取图片文件'}, room=task_id)
                return

            height, width = frame.shape[:2]
            fps = 12.0
            total_frames = 36

            task_sub_dir = os.path.join(app_config['SUB_VIDEO_FOLDER'], task_id)
            os.makedirs(task_sub_dir, exist_ok=True)
            tracker = EnhancedWormTracker(
                fps=fps,
                frame_size=(width, height),
                sub_video_dir=task_sub_dir,
                max_worm_id=app_config.get('WORM_COUNT') or None,
                head_config=app_config.get('HEAD_CONFIG') or [],
            )

            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_filename = f"{base_name}_result_{task_id}.mp4"
            output_path = ensure_unique_path(os.path.join(app_config['RESULT_FOLDER'], output_filename))
            output_filename = os.path.basename(output_path)
            out_writer = create_compatible_video_writer(output_path, fps, (width, height))

            log_entries = []
            
            # 单张图片也用 batch
            # Image fallback
            annotated_frames, track_data_list = process_frame_batch([frame]*total_frames, model, device, tracker, 1)
            for i, annotated_frame in enumerate(annotated_frames):
                frame_idx = i + 1
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                track_data = track_data_list[i]
                frame_info = {
                    'frame': frame_idx,
                    'active_tracks': len(tracker.tracks),
                    'tracks': track_data
                }
                log_entries.append(frame_info)

                emit('frame', {
                    'task_id': task_id,
                    'frame': frame_idx,
                    'total_frames': total_frames,
                    'image': frame_b64,
                    'stats': frame_info
                }, room=task_id)

                update_state(
                    status='processing',
                    current_frame=frame_idx,
                    total_frames=total_frames,
                    active_tracks=len(tracker.tracks),
                    latest_frame=frame_b64,
                    tracks=track_data,
                )

                out_writer.write(annotated_frame)
                time.sleep(0.05) # simulate processing time for image

            out_writer.release()
            out_writer = None

            for track in tracker.tracks:
                if track.sub_video_writer is not None:
                    track.sub_video_writer.release()
                    track.sub_video_writer = None

            log_path = os.path.join(app_config['RESULT_FOLDER'], f"{os.path.splitext(output_filename)[0]}_log.json")
            with open(log_path, 'w') as f:
                json.dump(log_entries, f)

            for track_id, cycle_info in tracker.cycle_counts.items():
                c_count = cycle_info.get('c_count', 0)
                s_count = cycle_info.get('s_count', 0)
                old_path = os.path.join(task_sub_dir, f"worm_{track_id}.mp4")
                if os.path.exists(old_path):
                    new_name = f"worm_{track_id}_C_{c_count}_S_{s_count}.mp4"
                    new_path = os.path.join(task_sub_dir, new_name)
                    try:
                        safe_rename(old_path, new_path)
                    except Exception as rename_err:
                        print(f"重命名子视频 {old_path} 失败: {rename_err}")

            zip_path = ensure_unique_path(os.path.join(app_config['RESULT_FOLDER'], f"{task_id}_sub_videos.zip"))
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for root, _, files in os.walk(task_sub_dir):
                    for file in files:
                        full_path = os.path.join(root, file)
                        zf.write(full_path, arcname=file)

            id_pose_summary = build_id_pose_summary(tracker)
            summary_filename = f"{os.path.splitext(output_filename)[0]}_id_pose_summary.json"
            summary_path = ensure_unique_path(os.path.join(app_config['RESULT_FOLDER'], summary_filename))
            summary_filename = os.path.basename(summary_path)
            with open(summary_path, 'w') as f:
                json.dump(id_pose_summary, f)

            csv_filename = write_swing_summary_csv(
                id_pose_summary, app_config['RESULT_FOLDER'], os.path.splitext(output_filename)[0]
            )

            emit('complete', {
                'task_id': task_id,
                'video_url': f"/media/{output_filename}",
                'log_url': f"/download/{os.path.basename(log_path)}",
                'sub_videos_url': f"/download/{os.path.basename(zip_path)}",
                'summary_url': f"/download/{summary_filename}",
                'csv_url': f"/download/{csv_filename}",
                'id_pose_summary': id_pose_summary,
            }, room=task_id)

            update_state(
                status='completed',
                current_frame=total_frames,
                total_frames=total_frames,
                video_url=f"/media/{output_filename}",
                log_url=f"/download/{os.path.basename(log_path)}",
                sub_videos_url=f"/download/{os.path.basename(zip_path)}",
                summary_url=f"/download/{summary_filename}",
                csv_url=f"/download/{csv_filename}",
                id_pose_summary=id_pose_summary,
            )
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 {video_path}")
            update_state(status='failed', error_message='无法打开视频文件')
            emit('error', {'task_id': task_id, 'message': '无法打开视频文件'}, room=task_id)
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"视频信息: {width}x{height}, {fps:.2f} FPS, 总帧数: {total_frames}")
        update_state(total_frames=total_frames)

        task_sub_dir = os.path.join(app_config['SUB_VIDEO_FOLDER'], task_id)
        os.makedirs(task_sub_dir, exist_ok=True)
        tracker = EnhancedWormTracker(
            fps=fps,
            frame_size=(width, height),
            sub_video_dir=task_sub_dir,
            max_worm_id=app_config.get('WORM_COUNT') or None,
            head_config=app_config.get('HEAD_CONFIG') or [],
        )

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{base_name}_result_{task_id}.mp4"
        output_path = ensure_unique_path(os.path.join(app_config['RESULT_FOLDER'], output_filename))
        output_filename = os.path.basename(output_path)
        out_writer = create_compatible_video_writer(output_path, fps, (width, height))

        frame_idx = 0
        log_entries = []
        # ── 流水线参数 ──
        # 50系 GPU 吞吐充足，8帧/批最大化 GPU 利用率
        BATCH_SIZE  = 8
        EMIT_EVERY  = 2       # 每隔N帧向前端推送一次画面
        JPEG_QUALITY = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

        is_paused   = False
        stop_event  = threading.Event()   # 统一停止信号（取消/异常时置位）
        error_event = threading.Event()

        # ── 队列：各阶段解耦（扩容以匹配 50 系 GPU 的高吞吐）──
        frame_queue = queue.Queue(maxsize=4)    # 预读缓冲（最多预读4批≈32帧）
        yolo_queue  = queue.Queue(maxsize=6)    # GPU推理结果缓冲
        write_queue = queue.Queue(maxsize=32)   # 磁盘写入队列
        emit_queue  = queue.Queue(maxsize=8)    # JPEG压缩+SocketIO发送队列

        # ── 线程1：帧读取（专职 cap.read，与GPU推理并行）──
        def read_worker():
            while not stop_event.is_set():
                if is_paused:
                    time.sleep(0.05)
                    continue
                batch = []
                for _ in range(BATCH_SIZE):
                    ret, frm = cap.read()
                    if not ret:
                        break
                    batch.append(frm)
                if not batch:
                    while not stop_event.is_set():
                        try:
                            frame_queue.put(None, timeout=1.0)
                            break
                        except queue.Full:
                            continue
                    return
                
                while not stop_event.is_set():
                    try:
                        frame_queue.put(batch, timeout=1.0)
                        break
                    except queue.Full:
                        continue

        read_thread = threading.Thread(target=read_worker, daemon=True)
        read_thread.start()

        # ── 线程2：磁盘写入（IO独立，不阻塞跟踪线程）──
        def io_writer():
            while True:
                item = write_queue.get()
                if item is None:
                    break
                try:
                    out_writer.write(item)
                except Exception as e:
                    print(f"io_writer error: {e}")

        io_thread = threading.Thread(target=io_writer, daemon=True)
        io_thread.start()

        # ── 线程3：JPEG压缩 + SocketIO发送（从tracking_worker解耦，不阻塞跟踪）──
        def emit_worker():
            counter = 0
            while True:
                item = emit_queue.get()
                if item is None:
                    break
                try:
                    ann_frame, track_data, cur_idx, frame_info = item
                    counter += 1
                    if counter % EMIT_EVERY == 0:
                        _, buf = cv2.imencode('.jpg', ann_frame, JPEG_QUALITY)
                        frame_b64 = base64.b64encode(buf).decode('utf-8')
                        emit('frame', {
                            'task_id': task_id,
                            'frame': cur_idx,
                            'total_frames': total_frames,
                            'image': frame_b64,
                            'stats': frame_info
                        }, room=task_id)
                except Exception as e:
                    print(f"emit_worker error: {e}")

        emit_thread = threading.Thread(target=emit_worker, daemon=True)
        emit_thread.start()

        # ── 线程4：跟踪（仅负责 process_track_single，不再做JPEG压缩）──
        def tracking_worker():
            try:
                nonlocal log_entries
                while not error_event.is_set():
                    item = yolo_queue.get()
                    if item is None:
                        write_queue.put(None)   # 通知 io_writer 退出
                        emit_queue.put(None)    # 通知 emit_worker 退出
                        break
                    batch_frames, batch_results, start_idx = item

                    for i, results in enumerate(batch_results):
                        frm = batch_frames[i]
                        cur_idx = start_idx + i

                        annotated_frame, track_data = process_track_single(frm, results, tracker, cur_idx)

                        # 磁盘写入（异步，增加超时和事件检测避免死锁）
                        while not error_event.is_set():
                            try:
                                write_queue.put(annotated_frame, timeout=1.0)
                                break
                            except queue.Full:
                                pass

                        frame_info = {
                            'frame': cur_idx,
                            'active_tracks': len(tracker.tracks),
                            'tracks': track_data
                        }
                        log_entries.append(frame_info)

                        # 非阻塞放入emit队列：队列满时跳过此帧推送，不卡跟踪
                        try:
                            emit_queue.put_nowait((annotated_frame, track_data, cur_idx, frame_info))
                        except queue.Full:
                            pass

                    update_state(
                        status='processing',
                        current_frame=start_idx + len(batch_results) - 1,
                        total_frames=total_frames,
                        active_tracks=len(tracker.tracks),
                    )

                    if start_idx % 50 < BATCH_SIZE:
                        print(f"任务 {task_id}: 已处理 {start_idx}/{total_frames} 帧, 活跃目标: {len(tracker.tracks)}")

            except Exception as e:
                print(f"Tracking worker error: {e}")
                import traceback; traceback.print_exc()
                error_event.set()
                write_queue.put(None)
                emit_queue.put(None)

        tracker_thread = threading.Thread(target=tracking_worker, daemon=True)
        tracker_thread.start()

        # ── 主循环：专职 GPU 推理（从 frame_queue 取帧 → 推理 → put yolo_queue）──
        try:
            while True:
                if error_event.is_set():
                    break

                # 检查控制状态（暂停 / 取消 / 设置虫子数量）
                try:
                    while not control_q.empty():
                        control = control_q.get_nowait()
                        # 支持元组格式： ('set_worm_count', n)
                        if isinstance(control, tuple) and control[0] == 'set_worm_count':
                            new_worm_count = control[1]
                            if tracker is not None:
                                try:
                                    if new_worm_count is None or new_worm_count <= 0:
                                        # 取消限制：切回无限制模式
                                        tracker.set_max_worm_id(None)
                                        print(f"任务 {task_id}: 已取消虫子数量上限（无限制模式）")
                                    else:
                                        tracker.set_max_worm_id(new_worm_count)
                                        print(f"任务 {task_id}: 虫子数量上限已更新为 {new_worm_count}")
                                except Exception as e:
                                    print(f"设置虫子数量失败: {e}")
                            continue
                        # 字符串格式
                        if control == 'cancelled':
                            emit('cancelled', {'task_id': task_id, 'message': '任务已取消'}, room=task_id)
                            update_state(status='cancelled', error_message='任务已取消')
                            stop_event.set()
                            yolo_queue.put(None)
                            return
                        elif control == 'paused':
                            is_paused = True
                            update_state(status='paused')
                        elif control == 'running':
                            if is_paused:
                                update_state(status='processing')
                            is_paused = False
                except Exception:
                    pass

                if is_paused:
                    time.sleep(0.05)
                    continue

                # 从读帧线程取帧批（带超时，避免暂停或结束时永久阻塞）
                try:
                    frames_batch = frame_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                if frames_batch is None:        # 视频读取完毕
                    yolo_queue.put(None)
                    break

                # GPU 推理（主循环专职此步，read_thread 在此期间预读下一批，retina_masks=True 生成原生高精度轮廓）
                results_list = model(frames_batch, verbose=False, device=device, retina_masks=True)
                while not stop_event.is_set() and not error_event.is_set():
                    try:
                        yolo_queue.put((frames_batch, results_list, frame_idx + 1), timeout=1.0)
                        break
                    except queue.Full:
                        pass
                frame_idx += len(frames_batch)

            # 等待所有线程有序完成
            stop_event.set()
            read_thread.join(timeout=5)
            tracker_thread.join()
            emit_thread.join()
            io_thread.join()      # 确保所有帧写入磁盘后再释放 writer

            if error_event.is_set():
                raise RuntimeError("跟踪工作线程发生异常退出，视频处理未正常完成")
        except Exception as e:
            stop_event.set()
            yolo_queue.put(None)
            raise e
        cap.release()
        cap = None
        out_writer.release()
        out_writer = None
        
        # 修正因 OpenCV 帧数计算不准或提早结束导致的进度条未满问题
        if frame_idx > 0 and frame_idx != total_frames:
            total_frames = frame_idx
            update_state(total_frames=total_frames)

        print(f"任务 {task_id} 视频读取完成，实际共处理 {frame_idx} 帧")

        print("正在关闭子视频写入器...")
        for track in tracker.tracks:
            if track.sub_video_writer is not None:
                track.sub_video_writer.release()
                track.sub_video_writer = None
                print(f"已关闭 track {track.track_id} 的子视频写入器")

        log_path = os.path.join(app_config['RESULT_FOLDER'], f"{os.path.splitext(output_filename)[0]}_log.json")
        with open(log_path, 'w') as f:
            json.dump(log_entries, f)

        for track_id, cycle_info in tracker.cycle_counts.items():
            cycles = cycle_info.get('cycles', 0)
            old_path = os.path.join(task_sub_dir, f"worm_{track_id}.mp4")
            if os.path.exists(old_path):
                # Ensure the sub video writer for this track is closed before renaming
                for track in tracker.tracks:
                    if track.track_id == track_id and track.sub_video_writer is not None:
                        track.sub_video_writer.release()
                        track.sub_video_writer = None
                new_name = f"worm_{track_id}_cycles_{cycles}.mp4"
                new_path = os.path.join(task_sub_dir, new_name)
                try:
                    final_path = safe_rename(old_path, new_path)
                    print(f"子视频重命名: {os.path.basename(final_path)}")
                except Exception as rename_err:
                    print(f"重命名子视频 {old_path} 失败: {rename_err}")

        zip_path = ensure_unique_path(os.path.join(app_config['RESULT_FOLDER'], f"{task_id}_sub_videos.zip"))
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for root, _, files in os.walk(task_sub_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    zf.write(full_path, arcname=file)
        print(f"子视频已打包: {zip_path}")

        id_pose_summary = build_id_pose_summary(tracker)
        summary_filename = f"{os.path.splitext(output_filename)[0]}_id_pose_summary.json"
        summary_path = ensure_unique_path(os.path.join(app_config['RESULT_FOLDER'], summary_filename))
        summary_filename = os.path.basename(summary_path)
        with open(summary_path, 'w') as f:
            json.dump(id_pose_summary, f)

        csv_filename = write_swing_summary_csv(
            id_pose_summary, app_config['RESULT_FOLDER'], os.path.splitext(output_filename)[0]
        )

        emit('complete', {
            'task_id': task_id,
            'video_url': f"/media/{output_filename}",
            'log_url': f"/download/{os.path.basename(log_path)}",
            'sub_videos_url': f"/download/{os.path.basename(zip_path)}",
            'summary_url': f"/download/{summary_filename}",
            'csv_url': f"/download/{csv_filename}",
            'id_pose_summary': id_pose_summary,
        }, room=task_id)
        update_state(
            status='completed',
            current_frame=frame_idx,
            total_frames=total_frames,
            video_url=f"/media/{output_filename}",
            log_url=f"/download/{os.path.basename(log_path)}",
            sub_videos_url=f"/download/{os.path.basename(zip_path)}",
            summary_url=f"/download/{summary_filename}",
            csv_url=f"/download/{csv_filename}",
            id_pose_summary=id_pose_summary,
        )
        print(f"任务 {task_id} 处理完成，已通知前端")

    except Exception as e:
        print(f"任务 {task_id} 发生未捕获异常: {e}")
        import traceback
        traceback.print_exc()
        update_state(status='failed', error_message=str(e))
        emit('error', {'task_id': task_id, 'message': str(e)}, room=task_id)

    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if out_writer is not None:
            try:
                out_writer.release()
            except Exception:
                pass
        if tracker is not None and hasattr(tracker, 'tracks'):
            for track in tracker.tracks:
                if track.sub_video_writer is not None:
                    try:
                        track.sub_video_writer.release()
                    except Exception:
                        pass
            # 释放持久骨架提取线程池
            try:
                tracker.shutdown()
            except Exception:
                pass
# ==================== HTTP 路由 ====================
@app.route('/upload', methods=['POST'])
def upload_video():
    """接收上传的视频，启动处理任务"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    # 必填：用户手动确认的虫子数量（正整数），用于约束 ID 分配池上限
    worm_count_raw = request.form.get('worm_count', None)
    if worm_count_raw is None:
        return jsonify({'error': '必须提供初始虫子数量（请输入正整数）'}), 400

    try:
        worm_count = int(worm_count_raw)
        if worm_count <= 0:
            return jsonify({'error': '初始虫子数量必须是大于 0 的正整数'}), 400
    except (ValueError, TypeError):
        return jsonify({'error': '初始虫子数量必须是有效的整数'}), 400

    # 可选：用户在「头部标记弹窗」中设置的头部配置（JSON 字符串）
    head_config = []
    head_config_raw = request.form.get('head_config', None)
    if head_config_raw:
        try:
            parsed = json.loads(head_config_raw)
            if isinstance(parsed, list):
                head_config = parsed
        except Exception:
            pass  # 解析失败则忽略，按无头部标记处理

    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)
    print(f"📁 文件已保存: {video_path}")

    # 生成唯一任务 ID
    task_id = generate_task_id()
    with task_lock:
        processing_tasks[task_id] = {
            'status': 'processing',
            'video_path': video_path,
            'filename': filename,
            'current_frame': 0,
            'total_frames': 0,
            'active_tracks': 0,
            'latest_frame': '',
            'tracks': [],
            'video_url': '',
            'log_url': '',
            'sub_videos_url': '',
            'summary_url': '',
            'id_pose_summary': [],
            'error_message': '',
            'control': 'running',
            'worm_count': worm_count,  # 用户设置的虫子数量上限（None 表示无限制）
        }
    print(f"🆔 生成任务ID: {task_id}")

    # 启动后台线程处理视频
    
    control_queues[task_id] = Queue()

    model_name_param = request.form.get('model_name', None)
    if model_name_param:
        mdir = get_models_dir()
        candidate = os.path.join(mdir, os.path.basename(model_name_param))
        target_model_path = candidate if os.path.exists(candidate) else (globals().get('global_model_path') or get_default_model_path())
    else:
        target_model_path = globals().get('global_model_path') or get_default_model_path()

    app_config_dict = {
        'UPLOAD_FOLDER': app.config['UPLOAD_FOLDER'],
        'RESULT_FOLDER': app.config['RESULT_FOLDER'],
        'SUB_VIDEO_FOLDER': app.config['SUB_VIDEO_FOLDER'],
        'MODEL_PATH': target_model_path,
        'WORM_COUNT': worm_count,   # 传给子进程的虫子数量上限
        'HEAD_CONFIG': head_config, # 用户头部标记配置
    }
    # 启动后台进程处理视频
    process = Process(target=process_video_task, args=(task_id, video_path, progress_queue, control_queues[task_id], app_config_dict))
    process.daemon = True
    process.start()

    print(f"已启动后台线程处理任务 {task_id}")

    return jsonify({'task_id': task_id}), 202

@app.route('/analyze_first_frame', methods=['POST'])
def analyze_first_frame():
    """
    分析第一帧：运行 YOLO 推理并提取每条虫子的11点骨架关键点。
    前端在头部标记弹窗中使用该数据让用户指定头部。

    Request Body (multipart/form-data):
        frame (file): 前端提取的第一帧 JPEG 图像。

    Response:
        {
          "frame_width": 1920,
          "frame_height": 1080,
          "annotated": "<base64>",
          "worms": [
            {
              "det_idx": 0,
              "confidence": 0.92,
              "bbox": [x1, y1, x2, y2],
              "centroid": [cx, cy],
              "keypoints": [[x0,y0], ..., [x10,y10]]  // 11 points, null if extraction fails
            }
          ]
        }
    """
    if 'frame' not in request.files:
        return jsonify({'error': '未提供帧图像（frame 字段缺失）'}), 400

    file = request.files['frame']
    img_bytes = file.read()
    if not img_bytes:
        return jsonify({'error': '帧图像为空'}), 400

    try:
        import supervision as sv

        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': '无法解码帧图像'}), 400

        height, width = frame.shape[:2]
        prev_model, prev_device = get_preview_model()

        # YOLO 推理（加锁防止并发调用同一模型实例引发 PyTorch / CUDA 竞争）
        with _preview_model_lock:
            results = prev_model(frame, verbose=False, device=prev_device, retina_masks=True)[0]
        detections = sv.Detections.from_ultralytics(results)

        # 面积/置信度过滤
        if len(detections) > 0:
            min_area = (width * height) * MIN_AREA_RATIO
            x1a, y1a, x2a, y2a = (
                detections.xyxy[:, 0], detections.xyxy[:, 1],
                detections.xyxy[:, 2], detections.xyxy[:, 3],
            )
            areas = (x2a - x1a) * (y2a - y1a)
            conf = detections.confidence if detections.confidence is not None \
                   else np.ones(len(detections), dtype=np.float32)
            keep = (areas >= min_area) & (conf >= MIN_CONFIDENCE)
            detections = detections[keep]

        worms = []
        annotated = frame.copy()
        if len(detections) > 0 and hasattr(detections, 'mask') and detections.mask is not None and len(detections.mask) > 0:
            mask_ann = sv.MaskAnnotator(opacity=0.35)
            poly_ann = sv.PolygonAnnotator(thickness=2)
            annotated = mask_ann.annotate(scene=annotated, detections=detections)
            annotated = poly_ann.annotate(scene=annotated, detections=detections)
        masks = detections.mask if hasattr(detections, 'mask') and detections.mask is not None else None

        for i in range(len(detections)):
            bbox = detections.xyxy[i].tolist()
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            conf_val = float(detections.confidence[i]) if detections.confidence is not None else None

            # 提取11点骨架关键点
            keypoints = None
            if masks is not None and i < len(masks):
                mask = masks[i]
                if mask is not None:
                    mask_uint8 = (mask * 255).astype(np.uint8) if mask.dtype != np.uint8 else mask
                    _, kpts = extract_centerline_from_mask(mask_uint8, n_keypoints=11)
                    if kpts is not None:
                        keypoints = [[float(pt[0]), float(pt[1])] for pt in kpts]

            worms.append({
                'det_idx': i,
                'confidence': conf_val,
                'bbox': [round(v, 1) for v in bbox],
                'centroid': [round(cx, 1), round(cy, 1)],
                'keypoints': keypoints,
            })

            # 标注帧：画框 + 节点序号
            bx1, by1, bx2, by2 = map(int, bbox)
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (22, 119, 255), 2)
            label = f"#{i} {conf_val:.2f}" if conf_val is not None else f"#{i}"
            cv2.putText(annotated, label, (bx1, max(14, by1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (22, 119, 255), 1)
            if keypoints:
                for ki, (kx, ky) in enumerate(keypoints):
                    color = (0, 220, 80) if ki in (0, 10) else (200, 200, 200)
                    radius = 5 if ki in (0, 10) else 3
                    cv2.circle(annotated, (int(kx), int(ky)), radius, color, -1)
                    cv2.putText(annotated, str(ki), (int(kx) + 6, int(ky) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1)

        cv2.putText(annotated, f"detected: {len(worms)}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 80), 2)

        _, buf = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        annotated_b64 = base64.b64encode(buf).decode('utf-8')

        return jsonify({
            'frame_width': width,
            'frame_height': height,
            'annotated': annotated_b64,
            'worms': worms,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/extract_manual_worm', methods=['POST'])
def extract_manual_worm():
    """
    接收用户在前端手动框选的 ROI 区域，使用局部自适应二值化与骨架化提取该虫子的 11 节点中心线。
    """
    if 'frame' not in request.files:
        return jsonify({'error': '未提供帧图像'}), 400
    file = request.files['frame']
    bbox_str = request.form.get('bbox')
    if not bbox_str:
        return jsonify({'error': '未提供 bbox'}), 400
    try:
        bbox = json.loads(bbox_str)
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    except Exception as e:
        return jsonify({'error': f'bbox 格式错误: {e}'}), 400

    img_bytes = file.read()
    if not img_bytes:
        return jsonify({'error': '帧图像为空'}), 400

    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': '无法解码图像'}), 400

    h, w = frame.shape[:2]
    rx1, rx2 = max(0, min(x1, x2)), min(w, max(x1, x2))
    ry1, ry2 = max(0, min(y1, y2)), min(h, max(y1, y2))
    if rx2 - rx1 < 6 or ry2 - ry1 < 6:
        return jsonify({'error': '选区过小'}), 400

    roi = frame[ry1:ry2, rx1:rx2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edge_mean = (np.mean(blurred[0, :]) + np.mean(blurred[-1, :]) + np.mean(blurred[:, 0]) + np.mean(blurred[:, -1])) / 4.0
    center_mean = np.mean(blurred)
    if center_mean < edge_mean:
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    keypoints = None
    try:
        _, kpts = extract_centerline_from_mask(mask, n_keypoints=11)
        if kpts is not None and len(kpts) == 11:
            keypoints = [[float(pt[0] + rx1), float(pt[1] + ry1)] for pt in kpts]
    except Exception as e:
        print(f"局部骨架提取异常: {e}")

    # 若骨架提取失败（例如选区对比度低），沿中轴生成 11 个均匀采样点作为兜底
    if keypoints is None:
        keypoints = [
            [float(rx1 + (rx2 - rx1) * i / 10.0), float(ry1 + (ry2 - ry1) * i / 10.0)]
            for i in range(11)
        ]

    cx = (rx1 + rx2) / 2.0
    cy = (ry1 + ry2) / 2.0

    return jsonify({
        'bbox': [rx1, ry1, rx2, ry2],
        'centroid': [round(cx, 1), round(cy, 1)],
        'keypoints': keypoints,
    })


@app.route('/preview_count', methods=['POST'])
def preview_count():
    """
    前端接口：接收视频第一帧（JPEG 图像），运行 YOLO 推理，
    返回预估的虫子检测数量，供用户参考填写初始虫子数量。

    Request Body (multipart/form-data):
        frame (file): 前端用 Canvas 提取的第一帧 JPEG 图像。

    Response:
        {
          "count": 12,             # 检测到的虫子数量（过滤后）
          "raw_count": 15,         # 未过滤时的原始检测数量
          "annotated": "base64..." # 可选：带标注框的帧预览（JPEG base64）
        }

    注意：
      - 此接口使用全局缓存模型（首次调用时加载，后续复用）。
      - 仅做单帧推理，速度快，不影响任务进程的模型使用。
    """
    if 'frame' not in request.files:
        return jsonify({'error': '未提供帧图像（frame 字段缺失）'}), 400

    file = request.files['frame']
    img_bytes = file.read()
    if not img_bytes:
        return jsonify({'error': '帧图像为空'}), 400

    try:
        # 解码图像
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': '无法解码帧图像，请确认发送的是有效的 JPEG/PNG 文件'}), 400

        height, width = frame.shape[:2]

        # 获取（或惰性加载）预览模型
        prev_model, prev_device = get_preview_model()

        # YOLO 推理（加锁防止并发调用同一模型实例引发 PyTorch / CUDA 竞争）
        with _preview_model_lock:
            results = prev_model(frame, verbose=False, device=prev_device, retina_masks=True)[0]

        import supervision as sv
        detections = sv.Detections.from_ultralytics(results)
        raw_count = len(detections)

        # 过滤：面积过小 / 置信度过低
        if len(detections) > 0:
            min_area = (width * height) * MIN_AREA_RATIO
            x1, y1, x2, y2 = (
                detections.xyxy[:, 0], detections.xyxy[:, 1],
                detections.xyxy[:, 2], detections.xyxy[:, 3],
            )
            areas = (x2 - x1) * (y2 - y1)
            conf = detections.confidence if detections.confidence is not None \
                   else np.ones(len(detections), dtype=np.float32)
            keep = (areas >= min_area) & (conf >= MIN_CONFIDENCE)
            detections = detections[keep]

        count = len(detections)

        # 生成标注预览帧（画出检测框）
        annotated = frame.copy()
        if len(detections) > 0:
            if hasattr(detections, 'mask') and detections.mask is not None and len(detections.mask) > 0:
                mask_ann = sv.MaskAnnotator(opacity=0.35)
                poly_ann = sv.PolygonAnnotator(thickness=2)
                annotated = mask_ann.annotate(scene=annotated, detections=detections)
                annotated = poly_ann.annotate(scene=annotated, detections=detections)
            box_ann = sv.BoxAnnotator(thickness=2)
            lbl_ann = sv.LabelAnnotator(text_scale=0.48, text_thickness=1, text_padding=4, border_radius=4)
            labels = [
                f"#{i+1}" if c is None else f"#{i+1} {c:.2f}"
                for i, c in enumerate(detections.confidence if detections.confidence is not None else [None] * len(detections))
            ]
            annotated = box_ann.annotate(scene=annotated, detections=detections)
            annotated = lbl_ann.annotate(scene=annotated, detections=detections, labels=labels)

        cv2.putText(annotated, f"detected: {count}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 80), 2, lineType=cv2.LINE_AA)

        _, buf = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        annotated_b64 = base64.b64encode(buf).decode('utf-8')

        return jsonify({
            'count': count,
            'raw_count': raw_count,
            'annotated': annotated_b64,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download_file(filename):
    """提供结果文件下载（完整视频、日志、子视频 ZIP）"""
    file_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_file(file_path, as_attachment=True)


@app.route('/media/<filename>')
def media_file(filename):
    """提供可在线播放媒体文件（主要用于前端 video 标签）。"""
    file_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    if not os.path.exists(file_path):
        return "File not found", 404

    ext = os.path.splitext(filename)[1].lower()
    mimetype = 'application/octet-stream'
    if ext == '.mp4':
        mimetype = 'video/mp4'
    elif ext == '.json':
        mimetype = 'application/json'
    elif ext == '.zip':
        mimetype = 'application/zip'

    return send_file(file_path, mimetype=mimetype, as_attachment=False, conditional=True)

@app.route('/task/<task_id>/sub_videos')
def list_sub_videos(task_id):
    """列出任务的所有子视频"""
    safe_id = secure_filename(task_id)
    task_sub_dir = os.path.join(app.config['SUB_VIDEO_FOLDER'], safe_id)
    if not os.path.exists(task_sub_dir):
        return jsonify({'videos': []})
    videos = []
    for f in sorted(os.listdir(task_sub_dir)):
        if f.endswith('.mp4'):
            videos.append({
                'filename': f,
                'url': f'/sub_video/{safe_id}/{f}'
            })
    return jsonify({'videos': videos})


@app.route('/task/<task_id>/status')
def task_status(task_id):
    """返回任务状态，用于前端刷新后恢复处理进度"""
    state = get_task_state(str(task_id))
    if state is None:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(state)


@app.route('/task/<task_id>/pause', methods=['POST'])
def pause_task(task_id):
    task_id = str(task_id)
    state = get_task_state(task_id)
    if state is None:
        return jsonify({'error': 'Task not found'}), 404
    update_task_state(task_id, control='paused', status='paused')
    if task_id in control_queues:
        control_queues[task_id].put('paused')
    return jsonify({'ok': True, 'task_id': task_id, 'status': 'paused'})


@app.route('/task/<task_id>/resume', methods=['POST'])
def resume_task(task_id):
    task_id = str(task_id)
    state = get_task_state(task_id)
    if state is None:
        return jsonify({'error': 'Task not found'}), 404
    update_task_state(task_id, control='running', status='processing')
    if task_id in control_queues:
        control_queues[task_id].put('running')
    return jsonify({'ok': True, 'task_id': task_id, 'status': 'processing'})


@app.route('/task/<task_id>/cancel', methods=['POST'])
def cancel_task(task_id):
    task_id = str(task_id)
    state = get_task_state(task_id)
    if state is None:
        return jsonify({'error': 'Task not found'}), 404
    update_task_state(task_id, control='cancelled', status='cancelled', error_message='任务已取消')
    if task_id in control_queues:
        control_queues[task_id].put('cancelled')
    return jsonify({'ok': True, 'task_id': task_id, 'status': 'cancelled'})


@app.route('/task/<task_id>/set_worm_count', methods=['POST'])
def set_worm_count(task_id):
    """
    前端接口：在任务运行中（或启动前）设置/更新虫子数量上限。

    Request Body (JSON 或 form-data):
        worm_count (int): 用户手动数得的虫子总数，必须为正整数。
                          传 0 或省略表示取消限制（切换回无限制模式，
                          但此操作不会回收已存在轨迹的 ID）。

    Response:
        {
          "ok": true,
          "task_id": "...",
          "worm_count": 12,
          "message": "虫子数量上限已更新为 12"
        }

    注意：
      - 子进程中运行的 Tracker 通过 control_queue 接收通知。
      - 子进程读取 'set_worm_count' 控制消息后调用 tracker.set_max_worm_id()。
      - 任务状态中同步更新 worm_count 字段，供前端轮询状态时获取。
    """
    task_id = str(task_id)
    state = get_task_state(task_id)
    if state is None:
        return jsonify({'error': 'Task not found'}), 404

    # 解析 worm_count（支持 JSON body 和 form-data）
    data = request.get_json(silent=True) or {}
    raw = data.get('worm_count') if data else None
    if raw is None:
        raw = request.form.get('worm_count')
    try:
        worm_count = int(raw)
    except (TypeError, ValueError):
        return jsonify({'error': 'worm_count 必须为整数'}), 400

    if worm_count <= 0:
        worm_count = None  # 0 或负数 → 取消上限

    # 更新主进程任务状态
    update_task_state(task_id, worm_count=worm_count)

    # 向子进程发送控制消息
    if task_id in control_queues:
        control_queues[task_id].put(('set_worm_count', worm_count))

    msg = f"虫子数量上限已更新为 {worm_count}" if worm_count else "已取消虫子数量上限（无限制模式）"
    return jsonify({'ok': True, 'task_id': task_id, 'worm_count': worm_count, 'message': msg})


@app.route('/models', methods=['GET'])
def get_models_route():
    """获取 models 目录下的可用模型文件列表及当前选中的模型"""
    models = list_available_models()
    current = get_current_model_name()
    return jsonify({
        'models': models,
        'current_model': current
    })


@app.route('/select_model', methods=['POST'])
def select_model_route():
    """选择/切换后端当前使用的模型"""
    global global_model_path, _preview_model, _preview_device
    data = request.get_json(silent=True) or {}
    model_name = data.get('model_name') or request.form.get('model_name')
    if not model_name:
        return jsonify({'error': '未提供 model_name'}), 400

    model_name = os.path.basename(model_name)
    mdir = get_models_dir()
    target_path = os.path.join(mdir, model_name)
    if not os.path.exists(target_path):
        return jsonify({'error': f'模型文件 {model_name} 不存在'}), 404

    global_model_path = target_path

    # 清空预览模型缓存，确保后续 preview 使用新选择的模型
    with _preview_model_lock:
        _preview_model = None
        _preview_device = None

    print(f"已成功切换当前模型为: {model_name} ({target_path})")
    return jsonify({
        'ok': True,
        'current_model': model_name,
        'message': f'模型已成功切换为 {model_name}'
    })


@app.route('/egg_count', methods=['POST'])
def egg_count():
    """
    虫卵计数接口：对上传的视频或图片第一帧运行 EggCount.pt 检测，
    返回虫卵数量、线虫数量及标注图。

    Request Body (multipart/form-data):
        file (file): 视频文件（mp4/avi/mov）或图片文件（jpg/png/bmp）。

    Response:
        {
          "worm_count": 12,
          "egg_count": 34,
          "total_count": 46,
          "worm_class_name": "worm",
          "egg_class_name": "egg",
          "annotated": "<base64 jpeg>",
          "worms": [{"bbox": [x1,y1,x2,y2], "confidence": 0.92}, ...],
          "eggs":  [{"bbox": [x1,y1,x2,y2], "confidence": 0.88}, ...]
        }
    """
    if 'file' not in request.files:
        return jsonify({'error': '未提供文件（file 字段缺失）'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename or 'upload')
    ext = os.path.splitext(filename)[1].lower()
    img_bytes = file.read()
    if not img_bytes:
        return jsonify({'error': '文件内容为空'}), 400

    try:
        # ── 1. 获取第一帧 ───────────────────────────────────────────
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
        frame = None

        if ext in video_exts:
            # 视频：写入临时文件，用 cv2.VideoCapture 读取第一帧
            tmp_video_path = os.path.join(app.config['UPLOAD_FOLDER'], f'_egg_tmp_{uuid.uuid4().hex[:8]}{ext}')
            with open(tmp_video_path, 'wb') as f:
                f.write(img_bytes)
            try:
                cap = cv2.VideoCapture(tmp_video_path)
                ret, frame = cap.read()
                cap.release()
            finally:
                try:
                    os.remove(tmp_video_path)
                except Exception:
                    pass
            if frame is None or not ret:
                return jsonify({'error': '无法从视频中提取第一帧'}), 400
        else:
            # 图片
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return jsonify({'error': '无法解码图片，请确认文件格式'}), 400

        height, width = frame.shape[:2]

        # ── 2. 推理 ────────────────────────────────────────────────
        egg_model, egg_device = get_egg_count_model()

        # 读取类别名称，自动区分 worm / egg
        model_names = egg_model.names   # e.g. {0: 'egg', 1: 'worm'}
        egg_class_ids  = [cid for cid, name in model_names.items() if 'egg'  in name.lower()]
        worm_class_ids = [cid for cid, name in model_names.items() if 'worm' in name.lower()]
        egg_class_name  = model_names[egg_class_ids[0]]  if egg_class_ids  else 'egg'
        worm_class_name = model_names[worm_class_ids[0]] if worm_class_ids else 'worm'

        with _egg_model_lock:
            results = egg_model(frame, verbose=False, device=egg_device)[0]

        # ── 3. 解析结果 ────────────────────────────────────────────
        boxes = results.boxes
        eggs, worms = [], []

        if boxes is not None and len(boxes) > 0:
            xyxy_np  = boxes.xyxy.cpu().numpy()
            conf_np  = boxes.conf.cpu().numpy()
            cls_np   = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(cls_np)):
                cid  = int(cls_np[i])
                conf = float(conf_np[i])
                bbox = [round(float(v), 1) for v in xyxy_np[i]]
                entry = {'bbox': bbox, 'confidence': round(conf, 4), 'class_id': cid}
                if cid in egg_class_ids:
                    eggs.append(entry)
                elif cid in worm_class_ids:
                    worms.append(entry)
                else:
                    # 未能匹配到 egg/worm 的检测，归入 worm 组
                    worms.append(entry)

        # ── 4. 返回原始图片及保存 JSON ──────────────────────────────────────────
        # 计算置信度统计
        egg_count_90 = sum(1 for det in eggs if det['confidence'] >= 0.90)
        egg_count_95 = sum(1 for det in eggs if det['confidence'] >= 0.95)

        # 保存结果到 json
        eggcount_dir = os.path.join(app.config['RESULT_FOLDER'], 'Eggcount')
        os.makedirs(eggcount_dir, exist_ok=True)
        base_name = os.path.splitext(filename)[0]
        json_path = os.path.join(eggcount_dir, f"{base_name}.json")
        
        egg_details = {}
        for idx, det in enumerate(eggs, 1):
            egg_details[str(idx)] = f"{det['confidence'] * 100:.1f}%"
            
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(egg_details, f, ensure_ascii=False, indent=2)

        # 前端将根据置信度动态绘制
        _, buf = cv2.imencode('.png', frame)
        raw_b64 = base64.b64encode(buf).decode('utf-8')

        return jsonify({
            'egg_count':       len(eggs),
            'egg_count_90':    egg_count_90,
            'egg_count_95':    egg_count_95,
            'egg_class_name':  egg_class_name,
            'raw_image':       raw_b64,
            'eggs':            eggs,
            'frame_width':     width,
            'frame_height':    height,
        })

    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/sub_video/<task_id>/<filename>')
def serve_sub_video(task_id, filename):
    """提供单个子视频文件"""
    safe_id = secure_filename(task_id)
    safe_name = secure_filename(filename)
    file_path = os.path.join(app.config['SUB_VIDEO_FOLDER'], safe_id, safe_name)
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_file(file_path, mimetype='video/mp4', as_attachment=False, conditional=True)

# ==================== WebSocket 事件 ====================
@socketio.on('connect')
def handle_connect():
    print('客户端已连接')

@socketio.on('join')
def handle_join(data):
    """客户端加入任务房间，用于接收该任务的帧数据"""
    task_id = data.get('task_id')
    if task_id is not None:
        task_id = str(task_id)
        join_room(task_id)
        print(f'🏠 客户端加入房间: {task_id}')

# ==================== 启动服务 ====================
if __name__ == '__main__':
    # 确认与选择默认模型
    global_model_path = get_default_model_path()
    if os.path.exists(global_model_path):
        print(f"默认模型路径已确认: {global_model_path} (可用模型: {list_available_models()})")
    else:
        print(f"警告：未找到模型文件 {global_model_path}，请放置模型于 models/ 目录！")

    # 启动 Flask-SocketIO 服务器
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)