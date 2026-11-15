# -*- coding: utf-8 -*-
"""
整合线虫跟踪与姿态分析模块
- 使用增强跟踪器（卡尔曼滤波+外观特征+运动方向+匈牙利匹配）
- 实时提取中心线（11个关键点）并计算正弦周期数
- 为每个虫子生成子视频（裁剪区域，以ID和周期数命名）
- 提供 process_frame 函数供 WebSocket 逐帧调用
"""

import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict, deque
import time
from scipy.optimize import linear_sum_assignment
from scipy import interpolate
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from skimage.morphology import skeletonize
from PIL import Image, ImageDraw, ImageFont

def to_bottom_left(x, y, frame_h):
    """将 OpenCV (左上角原点) 坐标转换为笛卡尔 (左下角原点) 坐标"""
    return float(x), float(frame_h - y)


# ==================== 配置参数（可根据需要调整） ====================
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.4
MAX_LOST_FRAMES = 15          # 兼容旧配置：丢失多少秒后回收（TRACK_PERSISTENCE_SEC 生效后仅作兜底）
MIN_CONFIDENCE = 0.3
MIN_AREA_RATIO = 0.000015     # 最小面积比例（相对于图像面积）
KEYPOINT_UPDATE_INTERVAL = 3   # 每隔多少帧重新提取一次关键点

# ==================== 摆动计数参数（可按需调整） ====================
SWING_COUNT_MODE = 'swing'     # swing = 头部每完成一次单方向摆动计 1 次（默认）；
                               # cycle = 头部完成一次左右往返（一个完整周期）计 1 次。
SWING_SIGNAL = 'bend'          # bend = 头部相对体轴弯曲角（抗整体转向漂移，默认）；
                               # absolute = 头部绝对方向角（旧行为）。
SWING_MIN_AMP_DEG = 8.0        # 计入一次摆动的最小偏转幅度（度）
SWING_MAX_AMP_DEG = 65.0       # 自适应幅度阈值上限（度），避免大幅度摆动被当成噪声
SWING_MIN_GAP_SEC = 0.10       # 两次摆动计数之间的最小时间间隔（秒）
SWING_BASELINE_SEC = 2.5       # 扣除整体转向漂移的滚动基线窗口（秒）
SWING_LIVE_WINDOW_SEC = 3.0    # 实时摆动频率统计窗口（秒）
SWING_LIVE_MIN_SEC = 1.0       # 实时频率至少累计多少秒数据才输出（避免开头数值离谱）
SWING_REF_MAX_HZ = 5.0         # 归一化摆动活性分的参考最大摆动频率（Hz）

# ==================== 视觉残留（识别中断时保留序号）参数 ====================
TRACK_PERSISTENCE_SEC = 60.0   # 目标丢失后 ID 保留时长（秒），期间不会被回收，也不会被新目标抢占
GHOST_OVERLAY_SEC = 3.0        # 丢失后仍绘制残留框 + ID 的时长（秒）
GHOST_REID_MAX_LOST = 90       # 丢失超过多少帧后，重新出现时放宽距离约束做找回匹配


def create_compatible_video_writer(path, fps, frame_size):
    codecs = ['avc1', 'H264', 'X264', 'mp4v']
    last_error = None
    for codec in codecs:
        try:
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, frame_size)
            if writer is not None and writer.isOpened():
                return writer
            if writer is not None:
                writer.release()
        except Exception as e:
            last_error = e
    raise RuntimeError(f"无法创建子视频写入器: {last_error}")

# ==================== 中心线提取与周期计数相关函数（从 yundong.py 移植） ====================

def keep_largest_component(mask_binary):
    """只保留最大的连通区域"""
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_binary
    largest = max(contours, key=cv2.contourArea)
    result = np.zeros_like(mask_binary)
    cv2.drawContours(result, [largest], -1, 1, thickness=-1)
    return result

def prune_skeleton(skel, min_branch_length=15):
    """骨架剪枝：去除短分支"""
    sk = skel.copy().astype(np.uint8)
    if np.sum(sk) == 0:
        return sk
    kernel = np.ones((3, 3), dtype=np.uint8)
    for _ in range(min_branch_length):
        neighbor_count = cv2.filter2D(sk, -1, kernel)
        endpoints = (sk == 1) & (neighbor_count == 2)
        if not np.any(endpoints):
            break
        sk[endpoints] = 0
    return sk

def skeleton_from_mask(mask):
    """从二值掩码生成骨架"""
    bw = (mask > 0)
    skel = skeletonize(bw)
    return skel.astype(np.uint8)

def find_any_skeleton_pixel(skel):
    """找到骨架上的任意一个像素点"""
    ys, xs = np.nonzero(skel)
    if len(xs) == 0:
        return None
    return (ys[0], xs[0])

def bfs_on_skeleton(skel, start_rc):
    """BFS在骨架上找到最远点"""
    h, w = skel.shape
    visited = np.zeros_like(skel, dtype=bool)
    prev = {}
    dist = {}
    q = deque()
    q.append(start_rc)
    visited[start_rc] = True
    dist[start_rc] = 0
    farthest = start_rc
    farthest_d = 0
    neigh = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1), (0, 1),
             (1, -1), (1, 0), (1, 1)]
    while q:
        r, c = q.popleft()
        d = dist[(r, c)]
        if d > farthest_d:
            farthest = (r, c)
            farthest_d = d
        for dr, dc in neigh:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and skel[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                prev[(nr, nc)] = (r, c)
                dist[(nr, nc)] = d + 1
                q.append((nr, nc))
    return farthest, farthest_d, prev, dist

def reconstruct_path(prev, start_rc, end_rc):
    """从prev字典重构路径，返回(x,y)格式的点列表"""
    path_rc = []
    cur = end_rc
    path_rc.append(cur)
    while cur != start_rc:
        cur = prev.get(cur)
        if cur is None:
            return None
        path_rc.append(cur)
    path_rc.reverse()
    path_xy = np.array([[c, r] for (r, c) in path_rc], dtype=float)
    return path_xy

def smooth_centerline(points, smoothness=3, num_points=200):
    """使用B样条平滑中心线"""
    if points is None or points.shape[0] == 0:
        return None
    if points.shape[0] < 3:
        xs = np.linspace(points[0, 0], points[-1, 0], num_points)
        ys = np.linspace(points[0, 1], points[-1, 1], num_points)
        return np.vstack([xs, ys]).T
    x = points[:, 0]
    y = points[:, 1]
    try:
        tck, u = interpolate.splprep([x, y], s=smoothness, k=3)
        u_fine = np.linspace(0, 1, num_points)
        x_f, y_f = interpolate.splev(u_fine, tck)
        return np.vstack([x_f, y_f]).T
    except Exception:
        t = np.linspace(0, 1, len(points))
        fx = interpolate.interp1d(t, x)
        fy = interpolate.interp1d(t, y)
        u = np.linspace(0, 1, num_points)
        return np.vstack([fx(u), fy(u)]).T

def sample_evenly_along_curve(points, n_samples):
    """沿曲线均匀采样n_samples个点"""
    if points is None or points.shape[0] == 0:
        return None
    diffs = np.diff(points, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cumlen = np.concatenate(([0.0], np.cumsum(seg_lens)))
    total = cumlen[-1]
    if total == 0:
        return np.tile(points[0], (n_samples, 1))
    # 两端各内缩 5% 避开掩码末端分叉与毛刺 (Argon 改进方案)
    target_lengths = np.linspace(0.05 * total, 0.95 * total, n_samples)
    xs = points[:, 0]
    ys = points[:, 1]
    x_sampled = np.interp(target_lengths, cumlen, xs)
    y_sampled = np.interp(target_lengths, cumlen, ys)
    return np.vstack([x_sampled, y_sampled]).T

def extract_centerline_from_mask(mask, n_keypoints=11):
    """
    从分割掩码中提取中心线和关键点
    返回: (centerline_smooth, keypoints) 或 (None, None)
    """
    try:
        mask_binary = (mask > 0).astype(np.uint8)
        if np.sum(mask_binary) < 10:
            return None, None

        # 只保留最大连通区域
        mask_binary = keep_largest_component(mask_binary)

        # 形态学开运算
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel_open, iterations=1)

        mask_binary = keep_largest_component(mask_binary)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        if np.sum(mask_binary) < 10:
            return None, None

        skel = skeleton_from_mask(mask_binary)
        if np.sum(skel) < 3:
            return None, None

        skel = prune_skeleton(skel, min_branch_length=10)
        if np.sum(skel) < 3:
            return None, None

        s_rc = find_any_skeleton_pixel(skel)
        if s_rc is None:
            return None, None

        a_rc, a_d, prev_s, dist_s = bfs_on_skeleton(skel, s_rc)
        b_rc, b_d, prev_a, dist_a = bfs_on_skeleton(skel, a_rc)

        path_ab_xy = reconstruct_path(prev_a, a_rc, b_rc)
        if path_ab_xy is None or len(path_ab_xy) < 3:
            return None, None

        # 基于主干像素占比的粘连/交叉熔断：计算未落在最长主干上的骨架分支像素数
        total_skel_pts = int(np.sum(skel > 0))
        branch_pixels = total_skel_pts - len(path_ab_xy)
        if branch_pixels > max(8, 0.18 * len(path_ab_xy)):
            return None, None

        center_smooth = smooth_centerline(path_ab_xy, smoothness=2, num_points=200)
        if center_smooth is None:
            return None, None

        keypoints = sample_evenly_along_curve(center_smooth, n_keypoints)
        return center_smooth, keypoints
    except Exception:
        return None, None

def compute_head_angle(keypoints):
    """计算头部方向角度（度）"""
    if keypoints is None or len(keypoints) < 3:
        return 0.0
    head_pts = keypoints[:3]
    dx = head_pts[-1][0] - head_pts[0][0]
    dy = head_pts[-1][1] - head_pts[0][1]
    angle = np.degrees(np.arctan2(dy, dx))
    return angle


def compute_head_bend_angle(keypoints):
    """头部相对体轴的弯曲角（度）。

    用 kp[0]->kp[2]（头颈方向）与 kp[0]->kp[-1]（体轴方向）的夹角表示头部弯曲，
    整体平移/转向会被自然抵消，因此比绝对头部方向角更适合作为摆动信号。
    """
    if keypoints is None or len(keypoints) < 4:
        return None
    kp = np.asarray(keypoints, dtype=np.float64)
    head_vec = kp[2] - kp[0]
    axis_vec = kp[-1] - kp[0]
    if np.hypot(head_vec[0], head_vec[1]) < 1e-3 or np.hypot(axis_vec[0], axis_vec[1]) < 1e-3:
        return None
    cross = float(head_vec[0] * axis_vec[1] - head_vec[1] * axis_vec[0])
    dot = float(head_vec[0] * axis_vec[0] + head_vec[1] * axis_vec[1])
    return float(np.degrees(np.arctan2(cross, dot)))


def swing_signal_from_keypoints(keypoints, mode=SWING_SIGNAL):
    """按配置选择摆动信号：默认相对体轴头部弯曲角，可选绝对头部方向角。"""
    if keypoints is None or len(keypoints) < 3:
        return None
    if str(mode).lower().startswith('abs'):
        return float(compute_head_angle(keypoints))
    bend = compute_head_bend_angle(keypoints)
    if bend is None:
        return float(compute_head_angle(keypoints))
    return bend


def orient_pose_pair(centerline, keypoints, last_pose_or_head=None, ref_head=None, last_keypoints=None):
    if keypoints is None or len(keypoints) < 2:
        return centerline, keypoints
    kp = np.asarray(keypoints, dtype=np.float32)
    
    target_pose = last_pose_or_head if last_pose_or_head is not None else last_keypoints
    if target_pose is not None:
        arr = np.asarray(target_pose, dtype=np.float32)
        if arr.ndim == 2 and len(arr) == len(kp):
            d_forward = float(np.mean(np.linalg.norm(kp - arr, axis=1)))
            d_reverse = float(np.mean(np.linalg.norm(kp[::-1] - arr, axis=1)))
            if d_reverse < d_forward:
                kp = kp[::-1].copy()
                if centerline is not None:
                    centerline = np.asarray(centerline, dtype=np.float32)[::-1].copy()
            return centerline, kp
        elif arr.ndim == 1 and arr.size == 2:
            ref_head = arr

    if ref_head is not None:
        head_d = float(np.hypot(kp[0, 0] - ref_head[0], kp[0, 1] - ref_head[1]))
        tail_d = float(np.hypot(kp[-1, 0] - ref_head[0], kp[-1, 1] - ref_head[1]))
        if tail_d < head_d:
            kp = kp[::-1].copy()
            if centerline is not None:
                centerline = np.asarray(centerline, dtype=np.float32)[::-1].copy()
    return centerline, kp


def compute_normalized_body_curvatures(centerline):
    """
    计算体长归一化身体曲率 (Argon 躯干中段双段相位方案)
    取躯干前中段(节点2,3,4)与躯干中后段(节点6,7,8)计算无量纲归一化曲率
    返回: (k_anterior, k_posterior, body_len)
    """
    if centerline is None or len(centerline) < 11:
        return None, None, 0.0

    centerline = np.asarray(centerline, dtype=float)
    diffs = np.diff(centerline, axis=0)
    body_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
    if body_len < 10.0:
        return None, None, 0.0

    def _segment_curv(p_prev, p_curr, p_next):
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        d1 = np.linalg.norm(v1)
        d2 = np.linalg.norm(v2)
        d3 = np.linalg.norm(p_next - p_prev)
        curvature = 2.0 * cross / (d1 * d2 * d3 + 1e-6)
        return float(curvature * body_len)

    k_anterior = np.mean([_segment_curv(centerline[i-1], centerline[i], centerline[i+1]) for i in (2, 3, 4)])
    k_posterior = np.mean([_segment_curv(centerline[i-1], centerline[i], centerline[i+1]) for i in (6, 7, 8)])
    return float(k_anterior), float(k_posterior), float(body_len)

def curvature_of_centerline(centerline):
    """计算单帧中心线的 signed curvature 平均值"""
    if centerline is None or len(centerline) < 3:
        return 0.0
    pts = np.array(centerline)
    x = pts[:, 0]
    y = pts[:, 1]
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx * dx + dy * dy) ** 1.5 + 1e-9
    k = (dx * ddy - dy * ddx) / denom
    if len(k) > 4:
        return float(np.nanmean(k[2:-2]))
    return float(np.nanmean(k))

def count_head_cycles(angle_seq, fps, smooth_sigma=2, min_prominence=5.0):
    """基于头部角度摆动统计周期数"""
    angle_seq = np.array(angle_seq)
    if len(angle_seq) < 6:
        return 0
    s = gaussian_filter1d(angle_seq, sigma=smooth_sigma)
    s = s - np.mean(s)
    if np.std(s) < 1e-6:
        return 0
    sign_changes = np.count_nonzero(np.diff(np.signbit(s)))
    return max(0, sign_changes // 2)


def compute_head_swing_frequency(angle_seq, fps, smooth_sigma=2, min_prominence=5.0):
    """根据头部角度序列计算摆动频率（Hz）。"""
    angle_seq = np.asarray(angle_seq, dtype=float)
    if len(angle_seq) < 6:
        return 0.0
    s = gaussian_filter1d(angle_seq, sigma=smooth_sigma)
    s = s - np.mean(s)
    if np.std(s) < 1e-6:
        return 0.0
    sign_changes = np.count_nonzero(np.diff(np.signbit(s)))
    duration = len(angle_seq) / max(float(fps), 1e-6)
    if duration <= 0:
        return 0.0
    return float(sign_changes / duration)

def count_body_cycles(center_seq, fps, smooth_sigma=3):
    """基于中心线序列的曲率变化统计全身正弦周期"""
    curv_list = []
    for cl in center_seq:
        curv_list.append(curvature_of_centerline(cl))
    if len(curv_list) < 6:
        return 0
    curv_list = np.array(curv_list)
    s = gaussian_filter1d(curv_list, sigma=smooth_sigma)
    signs = np.sign(s)
    zero_crossings = np.where(np.diff(signs) != 0)[0]
    cycles = len(zero_crossings) // 2
    return cycles

def count_final_cycles(history, fps):
    """综合头摆和身体曲率计算总周期数"""
    head_seq = history.get_head_angles()
    center_seq = history.get_centerlines()
    cycles_head = count_head_cycles(head_seq, fps)
    cycles_body = count_body_cycles(center_seq, fps)
    return max(cycles_head, cycles_body)

# ==================== 运动分类器 ====================

class MotionClassifier:
    """线虫运动分类器 - 识别具体运动类型（支持体长尺度无关归一化与正交曲率模态）"""
    
    MOTION_TYPES = {
        'static': 0,        # 静止
        'forward': 1,       # 前进
        'backward': 2,      # 后退
        'sinusoidal': 3,    # 正弦摆动
        'turning': 4,       # 转身
    }
    
    def __init__(self, speed_threshold=2.0, curvature_threshold=0.15):
        self.speed_threshold = speed_threshold
        self.curvature_threshold = curvature_threshold
    
    @staticmethod
    def classify_motion(keypoints, velocity, curvature, body_len=None, q_s=None, q_c=None,
                        dq_s_1s=None, dq_c_1s=None, v_trans_bl=None, is_moving=None,
                        in_deep_curl=False):
        """
        分类线虫运动类型（兼容传统调用与多模态视觉校准的体长归一化正交模态调用）
        
        参数:
            keypoints: (N, 2) 关键点数组
            velocity: (2,) 速度向量
            curvature: float 曲率
            body_len: float 归一化基准体长（可选）
            q_s, q_c: float 反对称S模态与对称C模态曲率（可选）
            dq_s_1s, dq_c_1s: float 1秒窗口内S/C模态曲率动态摆幅（可选）
            v_trans_bl: float 1秒窗口内质心平移速率（单位：体长/秒，可选）
            is_moving: bool 尺度无关真实活动标记（可选）
            in_deep_curl: bool 是否处于深度C/Omega卷曲折返状态
        
        返回:
            (motion_type: str, confidence: float)
        """
        if is_moving is not None and v_trans_bl is not None and dq_s_1s is not None and dq_c_1s is not None:
            if not is_moving:
                return 'static', 0.0
            qs_val = float(q_s) if q_s is not None else 0.0
            qc_val = float(q_c) if q_c is not None else 0.0
            if in_deep_curl or (dq_c_1s >= 1.50 and abs(qc_val) > abs(qs_val)):
                return 'turning', float(abs(qc_val))
            if dq_s_1s >= 1.00 or (v_trans_bl >= 0.025 and abs(qs_val) >= 1.25):
                return 'sinusoidal', float(abs(qs_val))
            if keypoints is not None and len(keypoints) >= 3:
                body_vec = np.asarray(keypoints[0], dtype=float) - np.asarray(keypoints[-1], dtype=float)
                b_norm = float(np.linalg.norm(body_vec))
                v_norm = float(np.linalg.norm(velocity))
                if b_norm > 1e-3 and v_norm > 1e-3:
                    cos_val = float(np.dot(velocity / v_norm, body_vec / b_norm))
                    if cos_val >= 0.30:
                        return 'forward', float(v_trans_bl)
                    elif cos_val <= -0.30:
                        return 'backward', float(v_trans_bl)
            return 'turning', float(v_trans_bl)

        speed = float(np.linalg.norm(velocity))
        eff_speed_thresh = max(0.8, 0.015 * float(body_len)) if (body_len is not None and body_len > 10.0) else 2.0
        curvature_threshold = 0.15
        
        # 判断静止
        if speed < eff_speed_thresh:
            return 'static', 0.0
        
        # 判断高曲率 -> 正弦摆动
        curvature_magnitude = abs(curvature)
        if curvature_magnitude > curvature_threshold:
            return 'sinusoidal', curvature_magnitude
        
        # 根据头部方向和速度方向判断前进/后退/转身
        if keypoints is not None and len(keypoints) >= 3:
            head_dir = keypoints[0] - keypoints[-1]
            head_angle = np.arctan2(head_dir[1], head_dir[0])
            velocity_angle = np.arctan2(velocity[1], velocity[0])
            angle_diff = abs(head_angle - velocity_angle)
            
            # 归一化角度差到 [0, pi]
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            angle_diff = abs(angle_diff)
            
            angle_threshold = np.pi / 6  # 30 度
            
            if angle_diff < angle_threshold:
                return 'forward', speed
            elif angle_diff > 5 * np.pi / 6:  # 150 度
                return 'backward', speed
            else:
                return 'turning', angle_diff
        else:
            return 'forward', speed
    
    @staticmethod
    def compute_activity_score(motion_history, fps, window_sec=1.5):
        """
        计算目标活性分数 (0-1)
        基于最近 N 秒的运动统计
        """
        motion_history = list(motion_history)
        window_frames = max(1, int(fps * window_sec))
        recent = motion_history[-window_frames:] if len(motion_history) >= window_frames else motion_history
        if not recent:
            return 0.0
        active_frames = sum(1 for motion_type, _ in recent if motion_type != 'static')
        return float(active_frames) / len(recent)

def compute_local_curvature(pts, head_idx=0):
    """计算指定端点的局部曲率"""
    pts = np.array(pts, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx * dx + dy * dy) ** 1.5 + 1e-9
    k = (dx * ddy - dy * ddx) / denom
    if head_idx == 0:
        return float(np.nan_to_num(np.nanmean(k[0:3])))
    else:
        return float(np.nan_to_num(np.nanmean(k[-3:])))


class TrackHistory:
    """存储每只线虫的历史数据，用于 C/S 动作判定和计数（正交曲率模态分解与多模态视觉真值校准版）

    核心物理机制：
      1. 正交曲率本征模态分解：
         - 反对称 S 模态：q_S = (k_ant - k_post) / 2，对首尾 180° 翻转严格保持数学不变 (q_S' = +q_S)；
         - 对称 C 模态：  q_C = (k_ant + k_post) / 2，表征全躯干同向 C 型屈曲（液体游泳摆动或调头）；
         - 能量正交恒等式：q_C^2 - q_S^2 == k_ant * k_post。
      2. 双模态 C 型动作检测：
         - 模式 C1（对称 C 模态快速周期性摆动，如液体环境中 Thrashing 游泳）：
           窗口内 C 模态能量占比 rho_CS = sum(q_C^2)/sum(q_C^2+q_S^2) >= 0.55 且 min(D/L) <= 0.85 时激活；
         - 模式 C2（深度卷曲/Omega 折返，如琼脂表面爬行大幅度调头）：
           首尾伸展比 D/L 由舒展态 (>= 0.72) 下探至 <= 0.58 并重新舒展至 >= 0.72 时计 1 次。
      3. 反对称 S 型全身推进波检测：
         - 双门限迟滞状态机监测 q_S 正负交替并过零，同时验证前段 (k_ant) 与后段 (k_post) 在周期内均完成正负双向屈曲传导，
           且窗口内具备真实平移推进（质心净位移 >= 7.5% 体长，排除静止微颤或原地摆头）。
    """

    # S 型正弦推进参数 (Antisymmetric S-mode q_S = (k_ant - k_post) / 2)
    S_WIN                = 25      # 2.5s 评估窗口（采样点数）
    K_S_HIGH             = 1.50    # S模态波峰触发阈值
    K_S_ZERO             = 0.25    # S模态过零回直阈值
    K_SEG_MIN            = 0.68    # 前/后躯干段各自必须达到的最小双向屈曲幅度
    S_PEAK_COOLDOWN      = 5       # 半周期最小间隔 (0.5s，支持最高 1.0Hz 爬行)
    S_MAX_CYCLE_GAP      = 48      # 离开波峰后的最大过渡超时 (4.8s)
    S_STILL_TIMEOUT      = 28      # 连续静止 2.8s 熔断
    S_MIN_NET_DISP_BL    = 0.075   # 窗口内质心净位移 >= 7.5% 体长

    # C 型运动参数 (Mode C1: 液体对称C摆动 q_C = (k_ant + k_post) / 2; Mode C2: 琼脂深弯Omega折返)
    K_C_HIGH             = 1.15    # C1模态全躯干同向快速摆动触发门限 (配合 rho_cs >= 0.55)
    K_C_ZERO             = 0.15    # C1模态回直过零门限
    C_PEAK_COOLDOWN      = 2       # C摆动半周期最小间隔 (0.2s，支持最高 2.5Hz 游泳摆动)
    C_MAX_CYCLE_GAP      = 35      # C半周期最大间隔 (3.5s)
    C_TURN_DL_ENTER      = 0.58    # 深弯折返进入门限 (D/L <= 0.58)
    C_TURN_DL_EXIT       = 0.72    # 深弯折返舒展退出门限 (D/L >= 0.72)
    C_TURN_COOLDOWN      = 25      # 两次深弯折返最小间隔 (2.5s)

    def __init__(self, max_len_frames=900, fps=10.0):
        self.fps            = max(float(fps), 1e-6)
        self.max_len_frames = max_len_frames

        # 原始与衍生序列队列（保持向后兼容字段名）
        self.centerlines    = deque(maxlen=max_len_frames)
        self.centers        = deque(maxlen=max_len_frames)
        self.end_dists      = deque(maxlen=max_len_frames)
        self.d_ratios       = deque(maxlen=max_len_frames)
        self.ep1_curvs      = deque(maxlen=max_len_frames)
        self.motion_signs   = deque(maxlen=max_len_frames)
        self.head_angles    = deque(maxlen=max_len_frames)
        self.velocities     = deque(maxlen=max_len_frames)
        self.curvatures     = deque(maxlen=max_len_frames)
        self.motion_types   = deque(maxlen=max_len_frames)
        self.swing_signals  = deque(maxlen=max_len_frames)

        # 正交模态与运动学序列
        self.k_ants         = deque(maxlen=max_len_frames)
        self.k_posts        = deque(maxlen=max_len_frames)
        self.q_s_seq        = deque(maxlen=max_len_frames)
        self.q_c_seq        = deque(maxlen=max_len_frames)
        self.qs_ema_seq     = deque(maxlen=max_len_frames)
        self.qc_ema_seq     = deque(maxlen=max_len_frames)
        self.is_moving_seq  = deque(maxlen=max_len_frames)
        self.body_lengths   = deque(maxlen=40)
        self._raw_lengths   = deque(maxlen=15)

        # 统计结果
        self.c_count          = 0
        self.s_count          = 0
        self.is_bidirectional = False
        self.frame_counter    = 0

        # 内部一致性与滤波状态
        self._abnormal_streak   = 0
        self._last_oriented_pts = None
        self._ka_ema            = None
        self._kp_ema            = None
        self._qs_ema            = None
        self._qc_ema            = None
        self._qc_fast           = None
        self._dl_ema            = None

        # S 型状态机
        self._s_phase         = 0
        self._s_phase_idx     = -999
        self._s_zero_crossed  = False
        self._s_half_count    = 0
        self._ant_pos         = False
        self._ant_neg         = False
        self._post_pos        = False
        self._post_neg        = False
        self._s_dirs_seen     = set()

        # C1 型 (对称C摆动/液体游泳) 状态机
        self._c_phase         = 0
        self._c_phase_idx     = -999
        self._c_zero_crossed  = False
        self._c_half_count    = 0

        # C2 型 (深弯/Omega折返) 状态机
        self._in_deep_curl        = False
        self._deep_curl_frames    = 0
        self._deep_curl_start_idx = -999
        self._last_deep_turn_idx  = -999

        # 最近一次计数的采样序号（用于动态主导运动类型判定）
        self._last_s_event_idx = -999
        self._last_c_event_idx = -999

        # 运动分类器
        self.motion_classifier = MotionClassifier()

    def _reset_s_pending(self):
        """重置未闭合的 S 型半周期状态"""
        self._s_phase        = 0
        self._s_phase_idx    = -999
        self._s_zero_crossed = False
        self._s_half_count   = 0
        self._ant_pos        = False
        self._ant_neg        = False
        self._post_pos       = False
        self._post_neg       = False

    def _reset_c_pending(self):
        """重置未闭合的 C 型状态"""
        self._c_phase           = 0
        self._c_phase_idx       = -999
        self._c_zero_crossed    = False
        self._c_half_count      = 0
        self._in_deep_curl      = False
        self._deep_curl_frames  = 0

    def _check_and_update_body_length(self, body_len):
        """自适应体长校验：过滤双虫粘连与残缺伪影，并允许边缘入画目标重锚定稳定体长平台"""
        if body_len <= 10.0:
            return False
        self._raw_lengths.append(body_len)
        if len(self._raw_lengths) >= 8:
            recent_raw = np.array(list(self._raw_lengths)[-8:], dtype=float)
            raw_mean = float(np.mean(recent_raw))
            raw_cv = float(np.std(recent_raw)) / (raw_mean + 1e-6)
            if raw_cv < 0.08 and len(self.body_lengths) >= 8:
                old_med = float(np.median(self.body_lengths))
                if abs(raw_mean - old_med) / (old_med + 1e-6) > 0.20:
                    self.body_lengths.clear()
                    for v in recent_raw:
                        self.body_lengths.append(float(v))
                    self._abnormal_streak = 0
                    return True

        if len(self.body_lengths) >= 10:
            med_len = float(np.median(self.body_lengths))
            rel_diff = (body_len - med_len) / (med_len + 1e-6)
            if abs(rel_diff) > 0.25:
                self._abnormal_streak += 1
                if self._abnormal_streak >= 4:
                    self._reset_s_pending()
                    self._reset_c_pending()
                return False
            if abs(rel_diff) <= 0.18:
                self.body_lengths.append(body_len)
            self._abnormal_streak = 0
            return True

        self._abnormal_streak = 0
        self.body_lengths.append(body_len)
        return True

    def append(self, centerline_11pts, head_angle_deg=None, velocity=None, curvature=None, keypoints=None):
        if centerline_11pts is None:
            return
        pts = np.asarray(centerline_11pts, dtype=float)
        if len(pts) < 11:
            return

        # 内部确保前后帧关键点序列方向连续一致
        if self._last_oriented_pts is not None:
            _, pts = orient_pose_pair(None, pts, last_keypoints=self._last_oriented_pts)
            pts = np.asarray(pts, dtype=float)

        ka, kp, body_len = compute_normalized_body_curvatures(pts)
        if ka is None or kp is None or not self._check_and_update_body_length(body_len):
            return

        med_bl = float(np.median(self.body_lengths))

        # 防止单步质心异常跳变（如跨目标误匹配或掩码飞溅）污染轨迹
        center = np.mean(pts, axis=0)
        if len(self.centers) > 0:
            step_jump = float(np.linalg.norm(center - self.centers[-1]))
            if step_jump > 0.45 * med_bl:
                return

        self._last_oriented_pts = pts.copy()
        self.centerlines.append(pts)
        self.centers.append(center)
        self.frame_counter += 1

        dist = float(np.linalg.norm(pts[0] - pts[-1]))
        d_ratio = dist / (body_len + 1e-6)
        self.end_dists.append(dist)
        self.d_ratios.append(d_ratio)

        curv = compute_local_curvature(pts, head_idx=0)
        self.ep1_curvs.append(curv)
        if head_angle_deg is None:
            head_angle_deg = compute_head_angle(pts)
        self.head_angles.append(head_angle_deg)

        if velocity is None:
            if len(self.centers) >= 2:
                velocity = self.centers[-1] - self.centers[-2]
            else:
                velocity = np.zeros(2, dtype=np.float32)
        self.velocities.append(velocity)

        if curvature is None:
            curvature = curv
        self.curvatures.append(curvature)

        # 正交曲率本征模态分解：反对称 S 模态 (qs) 与对称 C 模态 (qc)
        qs = 0.5 * (ka - kp)
        qc = 0.5 * (ka + kp)
        self.k_ants.append(ka)
        self.k_posts.append(kp)
        self.q_s_seq.append(qs)
        self.q_c_seq.append(qc)
        self.swing_signals.append(qs)

        # EMA 滤波
        alpha = 0.40
        alpha_c = 0.75
        self._ka_ema  = ka if self._ka_ema is None else (alpha * ka + (1.0 - alpha) * self._ka_ema)
        self._kp_ema  = kp if self._kp_ema is None else (alpha * kp + (1.0 - alpha) * self._kp_ema)
        self._qs_ema  = qs if self._qs_ema is None else (alpha * qs + (1.0 - alpha) * self._qs_ema)
        self._qc_ema  = qc if self._qc_ema is None else (alpha * qc + (1.0 - alpha) * self._qc_ema)
        self._qc_fast = qc if self._qc_fast is None else (alpha_c * qc + (1.0 - alpha_c) * self._qc_fast)
        self._dl_ema  = d_ratio if self._dl_ema is None else (alpha * d_ratio + (1.0 - alpha) * self._dl_ema)
        self.qs_ema_seq.append(self._qs_ema)
        self.qc_ema_seq.append(self._qc_ema)

        # 尺度无关运动学计算：使用 ~1.0 秒窗口评估平移速率 (BL/s) 与姿态屈曲动态幅度
        n_1s = max(1, int(round(self.fps * 1.0)))
        lb_1s = min(n_1s, len(self.centers) - 1)
        if lb_1s >= 1:
            disp_vec = center - self.centers[-(lb_1s + 1)]
            scale_t = float(self.fps) / float(lb_1s)
            v_trans_bl = (float(np.linalg.norm(disp_vec)) / (med_bl + 1e-6)) * scale_t
            recent_qs = list(self.qs_ema_seq)[-(lb_1s + 1):]
            recent_qc = list(self.qc_ema_seq)[-(lb_1s + 1):]
            dq_s_1s = float(max(recent_qs) - min(recent_qs))
            dq_c_1s = float(max(recent_qc) - min(recent_qc))
        else:
            disp_vec = np.zeros(2, dtype=float)
            v_trans_bl = 0.0
            dq_s_1s = 0.0
            dq_c_1s = 0.0

        n_15s = max(1, int(round(self.fps * 1.5)))
        lb_15 = min(n_15s, len(self.centers) - 1)
        disp_15_bl = (float(np.linalg.norm(center - self.centers[-(lb_15 + 1)])) / (med_bl + 1e-6)) if lb_15 >= 1 else 0.0

        # 真实活动判定：排除分割掩码边界微颤与冻结 S 姿态假阳性
        is_moving = (v_trans_bl >= 0.032 and disp_15_bl >= 0.045) or ((dq_s_1s + dq_c_1s) >= 1.55)
        self.is_moving_seq.append(is_moving)

        # 计算沿瞬时体轴方向的纵向运动符号 (+1 前进, -1 后退, 0 静止/侧移)
        body_vec = pts[0] - pts[-1]
        body_norm = float(np.linalg.norm(body_vec))
        disp_norm = float(np.linalg.norm(disp_vec))
        long_sign = 0
        if v_trans_bl >= 0.030 and body_norm > 1e-3 and disp_norm > 1e-3:
            cos_val = float(np.dot(disp_vec / disp_norm, body_vec / body_norm))
            if abs(cos_val) >= 0.30:
                long_sign = 1 if cos_val > 0 else -1
        self.motion_signs.append(long_sign)

        # 在线检测 C 与 S 动作
        self._detect_c_motion(med_bl, is_moving, dq_c_1s, ka, kp)
        self._detect_s_motion(med_bl, is_moving, v_trans_bl)

        # 瞬时运动类型分类
        m_type, conf = self._classify_instant(is_moving, v_trans_bl, dq_s_1s, dq_c_1s, long_sign)
        self.motion_types.append((m_type, conf))

    def _classify_instant(self, is_moving, v_trans_bl, dq_s_1s, dq_c_1s, long_sign):
        if not is_moving:
            return 'static', 0.0
        if self._in_deep_curl or (dq_c_1s >= 1.50 and abs(self._qc_ema) > abs(self._qs_ema)):
            return 'turning', float(abs(self._qc_ema))
        if dq_s_1s >= 1.00 or (v_trans_bl >= 0.025 and abs(self._qs_ema) >= 1.25):
            return 'sinusoidal', float(abs(self._qs_ema))
        if long_sign > 0:
            return 'forward', float(v_trans_bl)
        elif long_sign < 0:
            return 'backward', float(v_trans_bl)
        return 'turning', float(v_trans_bl)

    def _detect_c_motion(self, med_bl, is_moving, dq_c_1s, ka_raw, kp_raw):
        # ── 模式 C2: 深度卷曲/Omega折返 (如琼脂爬行中的大幅度调头) ──
        if self._dl_ema <= self.C_TURN_DL_ENTER and is_moving:
            self._deep_curl_frames += 1
            if not self._in_deep_curl and self._deep_curl_frames >= 2:
                if len(self.d_ratios) >= 8 and max(list(self.d_ratios)[-25:]) >= 0.72:
                    self._in_deep_curl = True
                    self._deep_curl_start_idx = self.frame_counter
                    self._reset_s_pending()
        else:
            self._deep_curl_frames = 0

        if self._in_deep_curl:
            if self.frame_counter - self._deep_curl_start_idx > 65:
                self._in_deep_curl = False
            elif self._dl_ema >= self.C_TURN_DL_EXIT:
                if self.frame_counter - self._last_deep_turn_idx >= self.C_TURN_COOLDOWN:
                    self.c_count += 1
                    self._last_deep_turn_idx = self.frame_counter
                    self._last_c_event_idx = self.frame_counter
                    self._reset_s_pending()
                    self._c_phase = 0
                    self._c_half_count = 0
                self._in_deep_curl = False

        # ── 模式 C1: 对称 C 模态周期性摆动 (如液体游泳 Thrashing) ──
        if self._c_phase != 0 and (self.frame_counter - self._c_phase_idx > self.C_MAX_CYCLE_GAP):
            self._c_phase = 0
            self._c_half_count = 0
            self._c_zero_crossed = False

        if not is_moving:
            return

        # 计算近期窗口内 C模态能量占比 rho_CS = sum(q_C^2) / sum(q_S^2 + q_C^2)
        win_c = min(len(self.q_c_seq), 18)
        if win_c < 6:
            return
        qc_w = np.array(list(self.q_c_seq)[-win_c:], dtype=float)
        qs_w = np.array(list(self.q_s_seq)[-win_c:], dtype=float)
        dl_w = np.array(list(self.d_ratios)[-win_c:], dtype=float)
        e_c = float(np.sum(qc_w ** 2))
        e_s = float(np.sum(qs_w ** 2))
        rho_cs = e_c / (e_c + e_s + 1e-6)

        # 只有当窗口内 C 模态显著占主导 (rho_cs >= 0.55) 且伴有身体弯曲收缩 (min(D/L) <= 0.85) 时才激活 C1 摆动检测
        if rho_cs < 0.55 or float(np.min(dl_w)) > 0.85:
            return

        qcf = self.q_c_seq[-1]
        qsf = self.q_s_seq[-1]

        if self._c_phase == 1 and (qcf < -self.K_C_ZERO or self.d_ratios[-1] >= 0.88):
            self._c_zero_crossed = True
        elif self._c_phase == -1 and (qcf > self.K_C_ZERO or self.d_ratios[-1] >= 0.88):
            self._c_zero_crossed = True

        # 全躯干同向 C 屈曲波峰：q_C 达标且前后段无反向对抗 (ka * kp >= -0.35)
        is_pos_c_peak = (
            qcf >= self.K_C_HIGH
            and (ka_raw * kp_raw >= -0.35)
            and abs(qcf) >= 0.80 * abs(qsf)
        )
        is_neg_c_peak = (
            qcf <= -self.K_C_HIGH
            and (ka_raw * kp_raw >= -0.35)
            and abs(qcf) >= 0.80 * abs(qsf)
        )

        if is_pos_c_peak:
            if self._c_phase == -1 and self._c_zero_crossed:
                if self.frame_counter - self._c_phase_idx >= self.C_PEAK_COOLDOWN:
                    self._advance_c_half_cycle(1)
            elif self._c_phase == 0:
                self._c_phase = 1
                self._c_phase_idx = self.frame_counter
                self._c_zero_crossed = False
                self._c_half_count = 1
            elif self._c_phase == 1 and not self._c_zero_crossed:
                self._c_phase_idx = self.frame_counter

        elif is_neg_c_peak:
            if self._c_phase == 1 and self._c_zero_crossed:
                if self.frame_counter - self._c_phase_idx >= self.C_PEAK_COOLDOWN:
                    self._advance_c_half_cycle(-1)
            elif self._c_phase == 0:
                self._c_phase = -1
                self._c_phase_idx = self.frame_counter
                self._c_zero_crossed = False
                self._c_half_count = 1
            elif self._c_phase == -1 and not self._c_zero_crossed:
                self._c_phase_idx = self.frame_counter

    def _advance_c_half_cycle(self, polarity):
        self._c_half_count += 1
        self._c_phase = polarity
        self._c_phase_idx = self.frame_counter
        self._c_zero_crossed = False
        if self._c_half_count >= 2:
            self.c_count += 1
            self._last_c_event_idx = self.frame_counter
            self._c_half_count = 0
            self._reset_s_pending()

    def _detect_s_motion(self, med_bl, is_moving, v_trans_bl):
        if self._in_deep_curl or self._dl_ema < 0.52:
            return

        # 若近期窗口内 C 模态显著主导 (rho_cs >= 0.60，如液体游泳摆动)，抑制 S 型计数
        win_c = min(len(self.q_c_seq), 18)
        if win_c >= 6:
            e_c = float(np.sum(np.array(list(self.q_c_seq)[-win_c:], dtype=float) ** 2))
            e_s = float(np.sum(np.array(list(self.q_s_seq)[-win_c:], dtype=float) ** 2))
            if e_c / (e_c + e_s + 1e-6) >= 0.60:
                return

        # 记录前后躯干在当前周期内的双向屈曲情况
        if self._s_phase != 0:
            if self._ka_ema >= self.K_SEG_MIN:
                self._ant_pos = True
            elif self._ka_ema <= -self.K_SEG_MIN:
                self._ant_neg = True
            if self._kp_ema >= self.K_SEG_MIN:
                self._post_pos = True
            elif self._kp_ema <= -self.K_SEG_MIN:
                self._post_neg = True

        # 超时熔断
        if self._s_phase != 0 and (self.frame_counter - self._s_phase_idx > self.S_MAX_CYCLE_GAP):
            self._reset_s_pending()

        # 连续静止熔断
        recent_mov = list(self.is_moving_seq)[-self.S_STILL_TIMEOUT:]
        if len(recent_mov) == self.S_STILL_TIMEOUT and sum(recent_mov) <= 2:
            self._reset_s_pending()
            return

        # 推进位移检查：近 S_WIN (或已有帧) 内质心净位移 >= 7.5% 体长（排除原地摇头与静止微动）
        win_len = min(len(self.centers), self.S_WIN)
        if win_len < 6:
            return
        centers_w = list(self.centers)[-win_len:]
        net_disp_w = float(np.linalg.norm(centers_w[-1] - centers_w[0]))
        centers_short = list(self.centers)[-min(len(self.centers), 12):]
        net_disp_short = float(np.linalg.norm(centers_short[-1] - centers_short[0]))
        if max(net_disp_w, net_disp_short) < med_bl * self.S_MIN_NET_DISP_BL:
            return

        # 反对称 S 模态 (q_S) 双门限迟滞状态机
        if self._s_phase == 1 and self._qs_ema < -self.K_S_ZERO:
            self._s_zero_crossed = True
        elif self._s_phase == -1 and self._qs_ema > self.K_S_ZERO:
            self._s_zero_crossed = True

        if self._qs_ema >= self.K_S_HIGH and abs(self._qs_ema) >= 0.70 * abs(self._qc_ema):
            if self._s_phase == -1 and self._s_zero_crossed:
                if self.frame_counter - self._s_phase_idx >= self.S_PEAK_COOLDOWN:
                    self._advance_s_half_cycle(1)
            elif self._s_phase == 0:
                self._s_phase = 1
                self._s_phase_idx = self.frame_counter
                self._s_zero_crossed = False
                self._s_half_count = 1
                self._ant_pos = (self._ka_ema >= self.K_SEG_MIN)
                self._ant_neg = (self._ka_ema <= -self.K_SEG_MIN)
                self._post_pos = (self._kp_ema >= self.K_SEG_MIN)
                self._post_neg = (self._kp_ema <= -self.K_SEG_MIN)
            elif self._s_phase == 1 and not self._s_zero_crossed:
                self._s_phase_idx = self.frame_counter
                self._try_complete_s_cycle()

        elif self._qs_ema <= -self.K_S_HIGH and abs(self._qs_ema) >= 0.70 * abs(self._qc_ema):
            if self._s_phase == 1 and self._s_zero_crossed:
                if self.frame_counter - self._s_phase_idx >= self.S_PEAK_COOLDOWN:
                    self._advance_s_half_cycle(-1)
            elif self._s_phase == 0:
                self._s_phase = -1
                self._s_phase_idx = self.frame_counter
                self._s_zero_crossed = False
                self._s_half_count = 1
                self._ant_pos = (self._ka_ema >= self.K_SEG_MIN)
                self._ant_neg = (self._ka_ema <= -self.K_SEG_MIN)
                self._post_pos = (self._kp_ema >= self.K_SEG_MIN)
                self._post_neg = (self._kp_ema <= -self.K_SEG_MIN)
            elif self._s_phase == -1 and not self._s_zero_crossed:
                self._s_phase_idx = self.frame_counter
                self._try_complete_s_cycle()

    def _advance_s_half_cycle(self, polarity):
        self._s_half_count += 1
        self._s_phase = polarity
        self._s_phase_idx = self.frame_counter
        self._s_zero_crossed = False
        self._try_complete_s_cycle()

    def _try_complete_s_cycle(self):
        if self._s_half_count >= 2:
            if self._ant_pos and self._ant_neg and self._post_pos and self._post_neg:
                self.s_count += 1
                self._last_s_event_idx = self.frame_counter
                valid_signs = [s for s in list(self.motion_signs)[-self.S_WIN:] if s != 0]
                if valid_signs:
                    dir_sign = 1 if sum(valid_signs) >= 0 else -1
                    self._s_dirs_seen.add(dir_sign)
                    if len(self._s_dirs_seen) > 1:
                        self.is_bidirectional = True
                self._s_half_count = 0
                self._ant_pos = (self._ka_ema >= self.K_SEG_MIN)
                self._ant_neg = (self._ka_ema <= -self.K_SEG_MIN)
                self._post_pos = (self._kp_ema >= self.K_SEG_MIN)
                self._post_neg = (self._kp_ema <= -self.K_SEG_MIN)

    def observed_seconds(self):
        return self.frame_counter / max(float(self.fps), 1e-6)

    def get_average_frequency(self):
        secs = self.observed_seconds()
        tot = self.c_count + self.s_count
        return tot / secs if secs > 0 else 0.0

    def get_frequency(self):
        return self.get_average_frequency()

    def get_activity_score(self, window_sec=1.5):
        return self.motion_classifier.compute_activity_score(
            list(self.motion_types), self.fps, window_sec
        )

    def is_active(self, min_activity=0.25, window_sec=1.5):
        if self.get_activity_score(window_sec) >= min_activity:
            return True
        if (self.c_count + self.s_count) >= 1:
            return True
        return False

    def get_dominant_motion_type(self, window_frames=25):
        items = list(self.motion_types)
        recent = items[-window_frames:] if len(items) >= window_frames else items
        if not recent:
            return 'static'
        non_static = [m for m, _ in recent if m != 'static']
        if len(non_static) < max(2, int(0.25 * len(recent))):
            return 'static'
        if self.frame_counter - self._last_c_event_idx <= window_frames:
            return 'turning'
        if self.frame_counter - self._last_s_event_idx <= window_frames:
            return 'sinusoidal'
        type_counts = {}
        for m_type in non_static:
            type_counts[m_type] = type_counts.get(m_type, 0) + 1
        return max(type_counts.items(), key=lambda x: x[1])[0]

# ==================== 增强跟踪器（基于 genzong102.py） ====================

class KalmanBoxTracker:
    """卡尔曼滤波器，状态: [cx, cy, w, h, vx, vy, vw, vh]"""
    def __init__(self, bbox, feature=None, dt=1.0, track_id=None, sub_video_writer=None):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], np.float32)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], np.float32)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.05
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        self.kf.statePost = np.array([[cx], [cy], [w], [h], [0.0], [0.0], [0.0], [0.0]], np.float32)
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 100.0

        self.history = deque(maxlen=30)
        self.history.append((cx, cy, w, h))
        self.keypoints_history = deque(maxlen=10)
        self.centerline_history = deque(maxlen=10)
        self.motion_vector = np.zeros(2, dtype=np.float32)
        self.last_keypoints = None
        self.last_observed_keypoints = None   # 最近一次真实提取的关键点（摆动计数用）
        self.last_centerline = None
        self.last_keypoint_update_frame = -1
        self.last_predicted_bbox = np.array(bbox, dtype=np.float32)
        self.crop_center = np.array([cx, cy], dtype=np.float32)
        self.crop_size = float(max(w, h) * 1.8)
        self.time_since_update = 0
        self.hits = 1
        self.lost_frames = 0
        self.confirmed = False
        self.track_id = track_id

        self.feature = feature if feature is not None else np.zeros(11)
        self.feature_history = deque(maxlen=20)
        self.feature_history.append(self.feature)

        # 子视频写入器
        self.sub_video_writer = sub_video_writer

        # 头部方向一致性锚点（摆动计数帧间定向用）
        self.orient_head = None
        self.user_label = None
        self.head_kpt_idx = None


    def predict(self):
        pred = self.kf.predict()
        cx = pred[0, 0].item()
        cy = pred[1, 0].item()
        w = pred[2, 0].item()
        h = pred[3, 0].item()
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        self.last_predicted_bbox = bbox
        self.crop_center = 0.9 * self.crop_center + 0.1 * np.array([cx, cy], dtype=np.float32)
        return bbox

    def update(self, bbox, feature=None):
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        # 【修复】测量向量形状 (4,1)，与 Kalman 滤波器输入一致
        measurement = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
        self.kf.correct(measurement)
        self.history.append((cx, cy, w, h))
        target_center = np.array([cx, cy], dtype=np.float32)
        target_size = float(max(w, h) * 1.8)
        self.crop_center = 0.8 * self.crop_center + 0.2 * target_center
        self.crop_size = 0.85 * self.crop_size + 0.15 * target_size
        self.time_since_update = 0
        self.lost_frames = 0
        self.hits += 1
        if feature is not None:
            self.feature = feature.copy()
            self.feature_history.append(feature)

    def get_stable_crop_window(self, frame_w, frame_h, scale=1.35, min_size=72):
        size = int(max(min_size, round(self.crop_size * scale)))
        size = min(size, frame_w, frame_h)
        half = size // 2

        cx = int(round(float(self.crop_center[0])))
        cy = int(round(float(self.crop_center[1])))

        x1 = cx - half
        y1 = cy - half
        x2 = x1 + size
        y2 = y1 + size

        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > frame_w:
            shift = x2 - frame_w
            x1 = max(0, x1 - shift)
            x2 = frame_w
        if y2 > frame_h:
            shift = y2 - frame_h
            y1 = max(0, y1 - shift)
            y2 = frame_h

        return int(x1), int(y1), int(x2), int(y2)

    def update_keypoints(self, keypoints, centerline=None):
        if keypoints is None or len(keypoints) == 0:
            return
        keypoints = np.asarray(keypoints, dtype=np.float32)
        if self.last_keypoints is not None and len(self.last_keypoints) == len(keypoints):
            delta = keypoints - self.last_keypoints
            self.motion_vector = 0.7 * self.motion_vector + 0.3 * np.mean(delta, axis=0)
        elif len(self.history) >= 2:
            p1 = np.array(self.history[-2][:2], dtype=np.float32)
            p2 = np.array(self.history[-1][:2], dtype=np.float32)
            self.motion_vector = 0.7 * self.motion_vector + 0.3 * (p2 - p1)
        self.last_keypoints = keypoints
        self.orient_head = keypoints[0].copy()
        if centerline is not None:
            self.last_centerline = np.asarray(centerline, dtype=np.float32)
            self.centerline_history.append(self.last_centerline)
        self.keypoints_history.append(keypoints)

    def get_motion_vector(self):
        if np.linalg.norm(self.motion_vector) > 1e-6:
            return self.motion_vector
        if len(self.history) >= 2:
            p1 = np.array(self.history[-2][:2], dtype=np.float32)
            p2 = np.array(self.history[-1][:2], dtype=np.float32)
            return p2 - p1
        velocity = np.asarray(self.get_velocity(), dtype=np.float32)
        if velocity.shape[0] >= 2:
            return velocity[:2]
        return np.zeros(2, dtype=np.float32)

    def predict_keypoints(self):
        if self.last_keypoints is None:
            return None
        return self.last_keypoints + self.get_motion_vector()

    @property
    def last_head_pos(self):
        return self.orient_head

    def get_state(self):
        state = self.kf.statePost
        cx = state[0, 0].item()
        cy = state[1, 0].item()
        w = state[2, 0].item()
        h = state[3, 0].item()
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return np.array([x1, y1, x2, y2])

    def get_velocity(self):
        vx = self.kf.statePost[4, 0].item()
        vy = self.kf.statePost[5, 0].item()
        return np.array([vx, vy])

    def get_predicted_center(self):
        state = self.kf.statePost
        return state[0, 0].item(), state[1, 0].item()

    def get_feature(self):
        if len(self.feature_history) > 0:
            return np.mean(self.feature_history, axis=0)
        return self.feature


def _detections_mask_shape(detections):
    if detections is None or len(detections) == 0:
        return None
    mask = getattr(detections, 'mask', None)
    if mask is None:
        return None
    if isinstance(mask, np.ndarray) and mask.ndim >= 3:
        return mask.shape[1], mask.shape[2]
    return None


def _normalize_bbox(bbox):
    bbox = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if bbox.size < 4:
        return np.zeros(4, dtype=np.float32)
    return bbox[:4]

class EnhancedWormTracker:
    """
    增强线虫跟踪器：集成卡尔曼滤波、外观特征、运动方向、匈牙利匹配，
    并整合中心线提取与周期计数，支持子视频生成
    """
    def __init__(self, fps=25, frame_size=(640, 480), sub_video_dir=None, max_worm_id=None, head_config=None):
        self.fps = fps
        self.width, self.height = frame_size
        self.sub_video_dir = sub_video_dir   # 子视频保存目录

        # 跟踪参数
        # 视觉残留：目标丢失后 ID 保留时长（期间不会被回收，也不会被新目标抢占），
        # 以及仍绘制残留框 + 序号的时长；超过保留时长才真正回收轨迹。
        self.persistence_frames = int(round(TRACK_PERSISTENCE_SEC * fps))
        self.ghost_overlay_frames = int(round(GHOST_OVERLAY_SEC * fps))
        self.max_lost_frames = max(int(fps * MAX_LOST_FRAMES), self.persistence_frames)
        self.confirm_hits = 5
        self.iou_match_thresh = 0.5
        self.iou_secondary = 0.3
        self.appearance_weight = 0.3
        self.direction_weight = 0.2
        self.max_speed = 50
        self.cost_threshold = 2.0
        self.keypoint_weight = 0.9
        self.prediction_confidence = 0.18

        # 过滤参数
        self.MIN_CONFIDENCE = MIN_CONFIDENCE
        self.MIN_AREA = (self.width * self.height) * MIN_AREA_RATIO

        # 轨迹管理
        self.tracks = []          # 存储 KalmanBoxTracker 对象
        # ID 池管理
        self.max_worm_id = max_worm_id
        self.id_pool = list(range(1, max_worm_id + 1)) if max_worm_id else None
        self.head_config = head_config or []
        self._head_config_applied = False

        self.next_id = max(self.max_worm_id or 0, 0) + 1  # 溢出分配起始 ID，绝不与 1..max_worm_id 冲突
        self.ghost_ids = set()    # 当前处于视觉残留（丢失但保留 ID）状态的 track_id
        self.track_history = defaultdict(lambda: deque(maxlen=3600))
        self.id_last_seen = {}

        # 重叠处理参数
        self.overlap_threshold = 0.3

        # 姿态分析相关
        self.track_histories = {}      # track_id -> TrackHistory
        self.cycle_counts = {}         # track_id -> 周期信息

    def _alloc_id(self):
        if self.id_pool:
            return self.id_pool.pop(0)
        if self.max_worm_id is not None:
            # 严格限制模式：当设定了虫子数量上限且池中已无可用 ID 时，不得溢出分配，返回 None
            return None
        new_id = self.next_id
        self.next_id += 1
        return new_id

    def _release_id(self, tid):
        if self.max_worm_id and 1 <= tid <= self.max_worm_id:
            if self.id_pool is not None and tid not in self.id_pool:
                self.id_pool.append(tid)
                self.id_pool.sort()

    def set_max_worm_id(self, max_id):
        self.max_worm_id = max_id
        if max_id:
            active_ids = {t.track_id for t in self.tracks}
            self.id_pool = [i for i in range(1, max_id + 1) if i not in active_ids]
            self.id_pool.sort()
            self.next_id = max(self.next_id, max_id + 1)
        else:
            self.id_pool = None

    def shutdown(self):
        for track in self.tracks:
            if track.sub_video_writer is not None:
                try:
                    track.sub_video_writer.release()
                except Exception:
                    pass

    def _match_head_config_to_tracks(self, tracks):
        if not self.head_config or not tracks:
            return
        cfgs = self.head_config if isinstance(self.head_config, list) else list(self.head_config.values())
        for cfg in cfgs:
            u_lbl = cfg.get('user_label')
            h_idx = cfg.get('head_kpt_idx', 0)
            c_pos = cfg.get('centroid')
            if not c_pos and 'centroid_x' in cfg and 'centroid_y' in cfg:
                c_pos = (cfg['centroid_x'], cfg['centroid_y'])
            best_t = None
            best_dist = 1e6
            for t in tracks:
                if t.user_label is not None:
                    continue
                bbox = t.get_state()
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                if c_pos:
                    d = np.sqrt((cx - c_pos[0])**2 + (cy - c_pos[1])**2)
                else:
                    d = 0
                if d < best_dist:
                    best_dist = d
                    best_t = t
            if best_t is not None and best_dist < 120.0:
                best_t.user_label = u_lbl
                best_t.confirmed = True  # 用户标记目标直接确认为有效轨迹
                best_t.head_kpt_idx = h_idx
                if h_idx == 10 and best_t.last_keypoints is not None:
                    best_t.last_keypoints = best_t.last_keypoints[::-1].copy()
                    if best_t.last_centerline is not None:
                        best_t.last_centerline = best_t.last_centerline[::-1].copy()
                    best_t.orient_head = best_t.last_keypoints[0].copy()

        # 对首帧未被 YOLO 检测到的人工框选目标（manual box），主动植入 Kalman 初始轨迹
        for cfg in cfgs:
            u_lbl = cfg.get('user_label')
            already_matched = any(t.user_label == u_lbl for t in tracks)
            if not already_matched and cfg.get('bbox'):
                m_bbox = np.array(cfg['bbox'], dtype=np.float32)
                m_id = self._alloc_id()
                if m_id is None:
                    # 若受限模式下 ID 池已被未标记临时检测占满，抢占一个未确认且未标记的轨迹 ID 给人工标注目标
                    for cand in reversed(tracks):
                        if cand.user_label is None and not cand.confirmed:
                            tracks.remove(cand)
                            self._release_id(cand.track_id)
                            m_id = self._alloc_id()
                            break
                if m_id is None:
                    continue
                sub_writer = None
                if self.sub_video_dir is not None:
                    sub_path = os.path.join(self.sub_video_dir, f"worm_{m_id}.mp4")
                    try:
                        sub_writer = create_compatible_video_writer(sub_path, self.fps, (200, 200))
                    except Exception:
                        pass
                m_track = KalmanBoxTracker(m_bbox, track_id=m_id, sub_video_writer=sub_writer)
                m_track.user_label = u_lbl
                m_track.confirmed = True  # 用户人工标记直接确认
                m_track.head_kpt_idx = cfg.get('head_kpt_idx', 0)
                if cfg.get('keypoints'):
                    kpts = np.array(cfg['keypoints'], dtype=np.float32)
                    if m_track.head_kpt_idx == 10:
                        kpts = kpts[::-1].copy()
                    m_track.update_keypoints(kpts)
                    m_track.last_observed_keypoints = kpts
                    self.update_sine_history(m_id, kpts, track=m_track)
                tracks.append(m_track)
                centroid = ((m_bbox[0] + m_bbox[2]) / 2.0, (m_bbox[1] + m_bbox[3]) / 2.0)
                self.track_history[m_id].append(centroid)

        self._head_config_applied = True

    def extract_stable_roi(self, frame, track, output_size=(200, 200)):
        if frame is None:
            return None
        x1, y1, x2, y2 = track.get_stable_crop_window(self.width, self.height)
        if x2 <= x1 or y2 <= y1:
            return None
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        return cv2.resize(roi, output_size)

    def _extract_centerline_in_window(self, mask, x1, y1, x2, y2):
        """在指定窗口内提取中心线并返回整帧坐标（补回窗口偏移）。

        相比在全图 mask 上做 skeletonize/BFS，仅在小窗口内计算，
        可显著降低逐帧中心线提取的开销。
        """
        if mask is None or x2 <= x1 or y2 <= y1:
            return None, None
        if mask.ndim >= 3:
            roi = np.any(mask[y1:y2, x1:x2] > 0, axis=0)
        else:
            roi = mask[y1:y2, x1:x2]
        if roi.size == 0 or int(np.sum(roi > 0)) < 10:
            return None, None
        centerline, keypoints = extract_centerline_from_mask(roi, n_keypoints=11)
        if keypoints is None:
            return None, None
        offset = np.array([x1, y1], dtype=np.float32)
        if centerline is not None:
            centerline = np.asarray(centerline, dtype=np.float32) + offset
        keypoints = np.asarray(keypoints, dtype=np.float32) + offset
        return centerline, keypoints

    def extract_features(self, mask, bbox):
        """提取11维外观特征（仅在 bbox 邻域内找轮廓，避免全图计算）"""
        if mask is None or mask.size == 0:
            return np.zeros(11)
        if mask.ndim >= 3:
            mask = np.any(mask > 0, axis=0)
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        aspect_ratio = w / (h + 1e-6)
        roi_x1 = max(0, int(np.floor(x1)) - 2)
        roi_y1 = max(0, int(np.floor(y1)) - 2)
        roi_x2 = min(mask.shape[1], int(np.ceil(x2)) + 2)
        roi_y2 = min(mask.shape[0], int(np.ceil(y2)) + 2)
        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            return np.zeros(11)
        mask = mask[roi_y1:roi_y2, roi_x1:roi_x2]
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        area = np.sum(mask > 0)
        if area < 50:
            return np.zeros(11)
        log_area = np.log1p(area)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            log_perimeter = 0
            circularity = 0
            hu = np.zeros(7)
        else:
            cnt = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(cnt, True)
            log_perimeter = np.log1p(perimeter)
            circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
            moments = cv2.moments(cnt)
            hu = cv2.HuMoments(moments).flatten()
            hu = np.sign(hu) * np.log1p(np.abs(hu))
        return np.concatenate(([log_area, log_perimeter, circularity, aspect_ratio], hu))

    def compute_appearance_distance(self, feat1, feat2):
        return np.linalg.norm(feat1 - feat2)

    def compute_iou(self, bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def compute_center_distance(self, bbox1, bbox2):
        cx1 = (bbox1[0] + bbox1[2]) / 2.0
        cy1 = (bbox1[1] + bbox1[3]) / 2.0
        cx2 = (bbox2[0] + bbox2[2]) / 2.0
        cy2 = (bbox2[1] + bbox2[3]) / 2.0
        return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

    def compute_direction_cost(self, track, detection_center):
        v = track.get_motion_vector()
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            return 0.0
        v_dir = v / norm
        pred_center = np.array(track.get_predicted_center())
        delta = np.array(detection_center) - pred_center
        norm_delta = np.linalg.norm(delta)
        if norm_delta < 1e-6:
            return 0.0
        delta_dir = delta / norm_delta
        cos_sim = np.dot(v_dir, delta_dir)
        cost = (1 - cos_sim) / 2.0
        return cost

    def compute_keypoint_cost(self, track, detection_keypoints):
        if detection_keypoints is None or len(detection_keypoints) == 0:
            return 0.0
        predicted_keypoints = track.predict_keypoints()
        if predicted_keypoints is None or len(predicted_keypoints) == 0:
            if track.last_keypoints is None or len(track.last_keypoints) == 0:
                return 0.0
            predicted_keypoints = track.last_keypoints

        predicted_keypoints = np.asarray(predicted_keypoints, dtype=np.float32)
        detection_keypoints = np.asarray(detection_keypoints, dtype=np.float32)
        if predicted_keypoints.shape != detection_keypoints.shape:
            return 0.0

        scale = np.linalg.norm(predicted_keypoints[-1] - predicted_keypoints[0])
        if scale < 1e-6:
            scale = np.linalg.norm(track.get_velocity()) + 1.0

        direct_cost = np.mean(np.linalg.norm(predicted_keypoints - detection_keypoints, axis=1)) / scale
        reversed_cost = np.mean(np.linalg.norm(predicted_keypoints - detection_keypoints[::-1], axis=1)) / scale
        return min(direct_cost, reversed_cost)

    def build_prediction_detections(self, predicted_tracks, detections):
        if len(predicted_tracks) == 0:
            return detections

        pred_boxes = []
        pred_confidences = []
        pred_tracker_ids = []
        pred_class_ids = []
        pred_masks = []

        actual_mask = getattr(detections, 'mask', None)
        mask_shape = None
        if actual_mask is not None and isinstance(actual_mask, np.ndarray) and actual_mask.ndim >= 3:
            mask_shape = actual_mask.shape[1:]

        for track in predicted_tracks:
            bbox = track.last_predicted_bbox if track.last_predicted_bbox is not None else track.get_state()
            pred_boxes.append(_normalize_bbox(bbox))
            pred_confidences.append(self.prediction_confidence)
            pred_tracker_ids.append(int(track.track_id))
            pred_class_ids.append(0)
            if mask_shape is not None:
                pred_masks.append(np.zeros(mask_shape, dtype=bool))

        pred_boxes = np.vstack(pred_boxes).astype(np.float32) if len(pred_boxes) > 0 else np.zeros((0, 4), dtype=np.float32)
        pred_confidences = np.asarray(pred_confidences, dtype=np.float32)
        pred_tracker_ids = np.asarray(pred_tracker_ids, dtype=int)
        pred_class_ids = np.asarray(pred_class_ids, dtype=int)

        xyxy = np.asarray(detections.xyxy, dtype=np.float32).reshape(-1, 4) if len(detections) > 0 else np.zeros((0, 4), dtype=np.float32)
        confidence = detections.confidence if detections.confidence is not None else np.ones(len(detections), dtype=np.float32)
        tracker_id = detections.tracker_id if hasattr(detections, 'tracker_id') and detections.tracker_id is not None else np.full(len(detections), -1, dtype=int)
        class_id = detections.class_id if hasattr(detections, 'class_id') and detections.class_id is not None else np.zeros(len(detections), dtype=int)

        merged_xyxy = np.concatenate([xyxy, pred_boxes], axis=0) if len(xyxy) > 0 else pred_boxes
        merged_confidence = np.concatenate([confidence, pred_confidences], axis=0) if len(confidence) > 0 else pred_confidences
        merged_tracker_id = np.concatenate([tracker_id, pred_tracker_ids], axis=0) if len(tracker_id) > 0 else pred_tracker_ids
        merged_class_id = np.concatenate([class_id, pred_class_ids], axis=0) if len(class_id) > 0 else pred_class_ids

        merged_mask = None
        if actual_mask is not None and isinstance(actual_mask, np.ndarray) and actual_mask.ndim >= 3 and len(pred_masks) > 0:
            merged_mask = np.concatenate([actual_mask, np.asarray(pred_masks)], axis=0)
        elif actual_mask is not None:
            merged_mask = actual_mask
        elif len(pred_masks) > 0:
            merged_mask = np.asarray(pred_masks)

        return sv.Detections(
            xyxy=merged_xyxy,
            mask=merged_mask,
            confidence=merged_confidence,
            class_id=merged_class_id,
            tracker_id=merged_tracker_id,
        )

    def apply_nms(self, detections):
        n = len(detections)
        if n <= 1:
            return detections
        boxes = np.asarray(detections.xyxy, dtype=np.float32).reshape(-1, 4)
        scores = detections.confidence if detections.confidence is not None else np.ones(n)
        order = np.argsort(-scores)
        keep = []
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            union = areas[i] + areas[rest] - inter
            ious = np.divide(inter, union, out=np.zeros_like(union), where=union > 0)
            order = rest[ious <= self.overlap_threshold]
        return detections[keep]

    def compute_motion_consistency_cost(self, track, detection_bbox, detection_size=None):
        """
        计算运动一致性成本
        包括：速度一致性、尺寸稳定性
        
        参数:
            track: KalmanBoxTracker
            detection_bbox: (x1, y1, x2, y2)
            detection_size: float 检测目标大小，可选
        
        返回:
            float 成本值 (0-1)
        """
        if len(track.history) < 2:
            return 0.0
        
        # 历史位置
        cx_prev, cy_prev, w_prev, h_prev = track.history[-1]
        
        # 检测位置
        cx_det = (detection_bbox[0] + detection_bbox[2]) / 2.0
        cy_det = (detection_bbox[1] + detection_bbox[3]) / 2.0
        w_det = detection_bbox[2] - detection_bbox[0]
        h_det = detection_bbox[3] - detection_bbox[1]
        
        if detection_size is None:
            detection_size = w_det + h_det
        
        # 预测位置
        cx_pred, cy_pred = track.get_predicted_center()
        
        # 1. 速度一致性：检测位置应接近卡尔曼预测位置
        velocity_diff = np.sqrt((cx_det - cx_pred)**2 + (cy_det - cy_pred)**2)
        # 正规化到 [0, 1]
        velocity_cost = min(velocity_diff / (self.max_speed + 1e-6), 1.0)
        
        # 2. 尺寸稳定性：同一虫体尺寸变化应平缓
        size_prev = w_prev + h_prev
        size_curr = w_det + h_det
        size_ratio = max(size_curr, size_prev) / (min(size_curr, size_prev) + 1e-6)
        # 正规化：理想比值为1，差异越大成本越高
        size_cost = min(abs(size_ratio - 1.0) / 0.5, 1.0)
        
        # 综合成本：速度和尺寸各占50%
        motion_cost = 0.5 * velocity_cost + 0.5 * size_cost
        
        return motion_cost

    def _match_gate(self, track):
        """关联搜索半径：目标丢失越久，允许的位置偏差越大（配合残留 ID 找回）。"""
        # 丢失越久半径越大，但增长在 GHOST_REID_MAX_LOST 帧后封顶
        grown = min(float(track.lost_frames), float(GHOST_REID_MAX_LOST))
        base = self.max_speed * (1.0 + grown / 20.0)
        limit = max(0.18 * max(self.width, self.height), self.max_speed * 1.5)
        return float(min(base, limit))

    def associate_detections_to_tracks(self, detections, masks):
        """
        改进的关联算法：添加运动一致性权重
        
        返回:
            (matches, unmatched_dets, unmatched_tracks)
        """
        if len(self.tracks) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(self.tracks))), [None] * len(detections)

        pred_bboxes = [track.predict() for track in self.tracks]
        det_centers = []
        det_features = []
        det_sizes = []
        
        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            det_centers.append(((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2))
            det_sizes.append((bbox[2] - bbox[0]) + (bbox[3] - bbox[1]))
            mask = masks[i] if masks is not None and i < len(masks) else None
            feat = self.extract_features(mask, bbox)
            det_features.append(feat)

        num_tracks = len(self.tracks)
        num_dets = len(detections)
        confidences = detections.confidence if detections.confidence is not None else np.ones(num_dets)
        track_features = [track.get_feature() for track in self.tracks]

        cost_matrix = np.zeros((num_tracks, num_dets))
        
        # 新的权重配置（提高运动一致性权重）
        weight_iou = 0.25
        weight_motion = 0.45  # 运动一致性权重提高
        weight_direction = 0.15
        weight_appearance = 0.15
        
        for t in range(num_tracks):
            track = self.tracks[t]
            pred_bbox = pred_bboxes[t]
            
            for d in range(num_dets):
                det_bbox = detections.xyxy[d]
                
                # 距离检测：太远就不匹配（丢失越久搜索半径越大）
                center_dist = self.compute_center_distance(pred_bbox, det_bbox)
                if center_dist > self._match_gate(track):
                    cost_matrix[t, d] = 1e6
                    continue
                
                # 1. IoU 成本
                iou = self.compute_iou(pred_bbox, det_bbox)
                iou_cost = 1 - iou
                
                # 2. 运动一致性成本（新增）
                motion_cost = self.compute_motion_consistency_cost(
                    track, det_bbox, det_sizes[d]
                )
                
                # 3. 方向成本
                dir_cost = self.compute_direction_cost(track, det_centers[d])
                
                # 4. 外观成本
                app_dist = self.compute_appearance_distance(
                    track_features[t], det_features[d]
                )
                app_cost = min(app_dist / 5.0, 1.0)
                
                # 5. 置信度惩罚
                conf_penalty = (1 - confidences[d]) * 0.1
                
                # 综合成本
                combined_cost = (
                    weight_iou * iou_cost +
                    weight_motion * motion_cost +
                    weight_direction * dir_cost +
                    weight_appearance * app_cost +
                    conf_penalty
                )
                
                # 低 IoU 额外惩罚（残留找回时不加，否则长时间丢失后无法接回原 ID）
                if iou < self.iou_match_thresh and track.lost_frames == 0:
                    combined_cost += 0.5
                
                cost_matrix[t, d] = combined_cost

        # 匈牙利匹配
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        matches = []
        used_tracks = set()
        used_dets = set()
        
        for t, d in zip(row_idx, col_idx):
            if cost_matrix[t, d] < self.cost_threshold:
                matches.append((t, d))
                used_tracks.add(t)
                used_dets.add(d)

        unmatched_tracks = [t for t in range(num_tracks) if t not in used_tracks]
        unmatched_dets = [d for d in range(num_dets) if d not in used_dets]

        # 二次匹配（对于高 IoU 但未首次匹配的目标）
        for t in unmatched_tracks[:]:
            best_d = -1
            best_cost = 1e6
            
            for d in unmatched_dets:
                iou = self.compute_iou(pred_bboxes[t], detections.xyxy[d])
                if iou < self.iou_secondary:
                    continue
                
                center_dist = self.compute_center_distance(
                    pred_bboxes[t], detections.xyxy[d]
                )
                if center_dist > self._match_gate(self.tracks[t]):
                    continue
                
                motion_cost = self.compute_motion_consistency_cost(
                    self.tracks[t], detections.xyxy[d], det_sizes[d]
                )
                app_dist = self.compute_appearance_distance(
                    track_features[t], det_features[d]
                )
                app_cost = min(app_dist / 5.0, 1.0)
                
                # 二次匹配使用较宽松的成本
                cost = 0.3 * (1 - iou) + 0.4 * motion_cost + 0.3 * app_cost
                
                if cost < best_cost:
                    best_cost = cost
                    best_d = d
            
            if best_d != -1 and best_cost < 1.2:  # 二次匹配阈值稍高
                matches.append((t, best_d))
                used_tracks.add(t)
                used_dets.add(best_d)

        # ---- 第三阶段：视觉残留找回匹配 ----
        # 目标被遮挡/漏检后重新出现时，位置可能已偏离预测很远（IoU=0），
        # 用位置 + 尺寸 + 外观的宽松成本再匹配一次，尽量沿用原序号而不是新建 ID。
        unmatched_tracks = [t for t in range(num_tracks) if t not in used_tracks]
        unmatched_dets = [d for d in range(num_dets) if d not in used_dets]
        recovery_tracks = sorted(
            [t for t in unmatched_tracks if self.tracks[t].lost_frames >= 3],
            key=lambda t: -self.tracks[t].lost_frames
        )
        for t in recovery_tracks:
            if t not in unmatched_tracks:
                continue
            track = self.tracks[t]
            pred_bbox = pred_bboxes[t]
            gate = self._match_gate(track)
            size_prev = (pred_bbox[2] - pred_bbox[0]) + (pred_bbox[3] - pred_bbox[1])
            best_d, best_cost = -1, 1e6
            for d in list(unmatched_dets):
                det_bbox = detections.xyxy[d]
                dist = self.compute_center_distance(pred_bbox, det_bbox)
                if dist > gate:
                    continue
                size_det = det_sizes[d]
                ratio = max(size_prev, size_det) / (min(size_prev, size_det) + 1e-6)
                if ratio > 2.5:
                    continue
                app_dist = self.compute_appearance_distance(track_features[t], det_features[d])
                if app_dist > 8.0:
                    continue
                cost = (0.55 * min(dist / gate, 1.0) +
                        0.25 * min(abs(ratio - 1.0), 1.0) +
                        0.20 * min(app_dist / 5.0, 1.0))
                if cost < best_cost:
                    best_cost, best_d = cost, d
            if best_d >= 0 and best_cost < 0.85:
                matches.append((t, best_d))
                used_tracks.add(t)
                used_dets.add(best_d)
                unmatched_tracks.remove(t)
                unmatched_dets.remove(best_d)

        return matches, unmatched_dets, unmatched_tracks, det_features

    def update_sine_history(self, track_id, keypoints, velocity=None, curvature=None, track=None):
        """
        更新正弦运动历史与运动分类
        
        参数:
            track_id: 目标ID
            keypoints: (N, 2) 关键点数组
            velocity: (2,) 速度向量
            curvature: float 曲率
            track: KalmanBoxTracker 对象（用于提取速度信息）
        """
        if track_id not in self.track_histories:
            eff_fps = self.fps / max(float(KEYPOINT_UPDATE_INTERVAL), 1.0)
            self.track_histories[track_id] = TrackHistory(min(int(self.max_lost_frames), 900), fps=eff_fps)
            self.cycle_counts[track_id] = {
                'c_count': 0,
                's_count': 0,
                'is_bidirectional': False,
                'cycles': 0,
                'posture_count': 0,
                'swing_count': 0,
                'swings': 0,
                'observed_frames': 0,
                'observed_seconds': 0.0,
                'avg_freq': 0.0,
                'live_freq': 0.0,
                'bends_per_min': 0.0,
                'motion_type': 'static',
                'activity_score': 0.0,
                'is_ghost': False,
                'lost_frames': 0,
            }
        
        history = self.track_histories[track_id]
        
        # 计算速度（如果没有提供）
        if velocity is None and track is not None:
            velocity = track.get_motion_vector()
        if velocity is None:
            velocity = np.zeros(2, dtype=np.float32)
        
        # 计算曲率（如果没有提供）
        if curvature is None and keypoints is not None:
            curvature = compute_local_curvature(keypoints, head_idx=0)
        
        # 计算头部角度
        head_angle = compute_head_angle(keypoints) if keypoints is not None else 0.0
        
        # 添加到历史并执行 C/S 在线判定
        history.append(keypoints, head_angle_deg=head_angle, velocity=velocity, 
                      curvature=curvature, keypoints=keypoints)
        self._sync_cycle_counts(track_id, history, full=True)


    def feed_swing_sample(self, track_id, keypoints):
        """Backward compatible stub"""
        pass

    def _sync_cycle_counts(self, track_id, history, full=True):
        """把 C/S 动作计数、频率、活性写回 cycle_counts[track_id]"""
        c_count = getattr(history, 'c_count', 0)
        s_count = getattr(history, 's_count', 0)
        tot = c_count + s_count
        observed_sec = max(history.observed_seconds(), 0.001)
        avg_freq = tot / observed_sec if observed_sec > 0 else 0.0
        bpm = avg_freq * 60.0

        prev = self.cycle_counts.get(track_id, {})
        motion_type = history.get_dominant_motion_type(window_frames=25) if full else prev.get('motion_type', 'static')
        activity = history.get_activity_score(window_sec=1.5) if full else prev.get('activity_score', 0.0)

        info = {
            'c_count': c_count,
            's_count': s_count,
            'is_bidirectional': getattr(history, 'is_bidirectional', False),
            'cycles': tot,
            'posture_count': tot,
            'swing_count': tot,
            'swings': tot,
            'half_swing_count': tot * 2,
            'cycle_count': tot,
            'swing_cycle_count': tot,
            'observed_frames': history.frame_counter * KEYPOINT_UPDATE_INTERVAL,
            'observed_seconds': round(observed_sec, 3),
            'avg_freq': round(avg_freq, 3),
            'freq': round(avg_freq, 3),
            'live_freq': round(avg_freq, 3),
            'swing_freq_hz': round(avg_freq, 3),
            'live_swing_freq': round(avg_freq, 3),
            'cycle_freq': round(avg_cycle_freq if 'avg_cycle_freq' in locals() else avg_freq, 3),
            'inst_freq': round(avg_freq, 3),
            'bends_per_min': round(bpm, 1),
            'swing_activity': round(activity, 3),
            'average_frequency': round(avg_freq, 3),
            'realtime_freq': round(avg_freq, 3),
            'swing_speed': round(avg_freq, 3),
            'head_swing_frequency': round(avg_freq, 3),
            'motion_type': motion_type,
            'activity_score': round(activity, 3),
            'is_ghost': bool(prev.get('is_ghost', False)),
            'lost_frames': int(prev.get('lost_frames', 0)),
            'is_static': motion_type == 'static',
        }
        self.cycle_counts[track_id] = info

    def _handle_lost_tracks(self, track_indices, frame, frame_idx):
        """未匹配/丢失轨迹的统一处理：丢失计数、位置外推、视觉残留。

        视觉残留（GHOST_OVERLAY_SEC 内）：
          - 继续用卡尔曼预测外推位置与关键点，画面/子视频不出现空洞；
          - ID 保持显示为残留状态，重新检出后立即恢复正常状态。
        """
        for t_idx in track_indices:
            track = self.tracks[t_idx]
            track.lost_frames += 1
            tid = track.track_id
            info = self.cycle_counts.get(tid)
            if track.lost_frames <= self.ghost_overlay_frames:
                self.ghost_ids.add(tid)
                predicted_keypoints = track.predict_keypoints()
                if predicted_keypoints is not None:
                    track.update_keypoints(predicted_keypoints, track.last_centerline)
                if frame is not None and track.sub_video_writer is not None:
                    roi_resized = self.extract_stable_roi(frame, track, output_size=(200, 200))
                    if roi_resized is not None:
                        track.sub_video_writer.write(roi_resized)
                is_ghost = True
            else:
                # 超出残留绘制窗口：只保留 ID，不再外推/绘制，避免画面出现漂移的假目标
                self.ghost_ids.discard(tid)
                is_ghost = False
            if info is not None:
                info['is_ghost'] = is_ghost
                info['lost_frames'] = int(track.lost_frames)
            if is_ghost:
                pred_bbox = track.last_predicted_bbox
                if pred_bbox is not None:
                    pred_centroid = ((pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2)
                    self.track_history[tid].append(pred_centroid)

    def _prune_lost_tracks(self):
        """回收超过保留时长仍未找回的轨迹。
        已确认轨迹允许持久保留（如 60 秒）；未确认轨迹若短时间丢失（>= 3 帧）则迅速回收，释放 ID 资源。
        """
        remaining_tracks = []
        for t in self.tracks:
            max_lost = self.max_lost_frames if t.confirmed else min(3, self.confirm_hits)
            if t.lost_frames > max_lost:
                self.ghost_ids.discard(t.track_id)
                self._release_id(t.track_id)
                if t.sub_video_writer:
                    t.sub_video_writer.release()
            else:
                remaining_tracks.append(t)
        self.tracks = remaining_tracks

    def update_tracking(self, detections, masks, frame_idx, frame=None):
        """
        主更新函数，返回带有tracker_id的detections
        若提供frame参数，则会写入子视频
        """
        if len(detections) == 0:
            for track in self.tracks:
                track.predict()          # 本帧没有检测：先做一次卡尔曼预测（每帧只预测一次）
            self._handle_lost_tracks(list(range(len(self.tracks))), frame, frame_idx)
            self._prune_lost_tracks()
            ghost_tracks = [t for t in self.tracks if t.track_id in self.ghost_ids]
            empty_dets = sv.Detections.empty()
            empty_dets.tracker_id = np.array([], dtype=int)
            return self.build_prediction_detections(ghost_tracks, empty_dets)

        detections = self.apply_nms(detections)
        matches, unmatched_dets, unmatched_tracks, det_features = self.associate_detections_to_tracks(detections, masks)

        # 更新匹配的轨迹（复用关联阶段已提取的特征，避免重复计算）
        for t_idx, d_idx in matches:
            track = self.tracks[t_idx]
            bbox = detections.xyxy[d_idx]
            mask = masks[d_idx] if masks is not None else None
            feat = det_features[d_idx]
            if feat is None:
                feat = self.extract_features(mask, bbox)
            track.update(bbox, feat)
            if not track.confirmed and track.hits >= self.confirm_hits:
                track.confirmed = True
            centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            self.track_history[track.track_id].append(centroid)
            self.id_last_seen[track.track_id] = frame_idx
            # 目标已被找回：结束视觉残留状态
            self.ghost_ids.discard(track.track_id)
            _prev_info = self.cycle_counts.get(track.track_id)
            if _prev_info is not None:
                _prev_info['is_ghost'] = False
                _prev_info['lost_frames'] = 0

            # 姿态分析：提取中心线并更新历史（限定在稳定窗口内，避免全图骨架化）
            if mask is not None and (
                track.last_keypoint_update_frame < 0
                or frame_idx - track.last_keypoint_update_frame >= KEYPOINT_UPDATE_INTERVAL
            ):
                cw = track.get_stable_crop_window(self.width, self.height)
                centerline_raw, keypoints_raw = self._extract_centerline_in_window(mask, *cw)
                if keypoints_raw is not None:
                    # 保证该线虫头部方向与上一帧一致（骨架 BFS 端点可能翻转）
                    centerline_raw, keypoints_raw = orient_pose_pair(
                        centerline_raw, keypoints_raw, last_keypoints=track.last_keypoints, ref_head=track.orient_head
                    )
                    track.update_keypoints(keypoints_raw, centerline_raw)
                    track.last_observed_keypoints = keypoints_raw   # 摆动计数只用真实关键点
                    self.update_sine_history(track.track_id, keypoints_raw, track=track)
                    track.last_keypoint_update_frame = frame_idx
            elif track.last_keypoints is not None:
                track.update_keypoints(track.predict_keypoints(), track.last_centerline)

            if track.sub_video_writer is not None and frame is not None:
                roi_resized = self.extract_stable_roi(frame, track, output_size=(200, 200))
                if roi_resized is not None:
                    track.sub_video_writer.write(roi_resized)

        # 创建新轨迹
        new_tracks = []
        new_track_ids_map = {}

        # 保证 ID 优先分配给置信度高的单位：按检测置信度降序排序未匹配的检测框
        conf_arr = getattr(detections, 'confidence', None)
        sorted_unmatched_dets = sorted(
            unmatched_dets,
            key=lambda d: float(conf_arr[d]) if (conf_arr is not None and d < len(conf_arr)) else 0.0,
            reverse=True
        )

        for d_idx in sorted_unmatched_dets:
            bbox = detections.xyxy[d_idx]
            mask = masks[d_idx] if masks is not None and d_idx < len(masks) else None
            feat = det_features[d_idx] if d_idx < len(det_features) else None
            if feat is None:
                feat = self.extract_features(mask, bbox)

            allocated_id = self._alloc_id()
            if allocated_id is None:
                # 设定了虫数上限且池子已空，严格限制不再发放新 ID
                continue

            # 创建子视频写入器（如果指定了目录）
            sub_writer = None
            if self.sub_video_dir is not None:
                sub_path = os.path.join(self.sub_video_dir, f"worm_{allocated_id}.mp4")
                try:
                    sub_writer = create_compatible_video_writer(sub_path, self.fps, (200, 200))
                except Exception as e:
                    print(f"子视频写入器警告: {e}")
                    sub_writer = None

            new_track = KalmanBoxTracker(bbox, feat, track_id=allocated_id, sub_video_writer=sub_writer)
            new_track.track_id = allocated_id
            new_track.confirmed = False
            new_tracks.append(new_track)
            new_track_ids_map[d_idx] = allocated_id

            centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            self.track_history[new_track.track_id].append(centroid)
            self.id_last_seen[new_track.track_id] = frame_idx

            # 新轨迹也立即提取一次中心线（限定在轨迹窗口内，降低首帧开销）
            if mask is not None:
                cw = new_track.get_stable_crop_window(self.width, self.height)
                centerline_raw, keypoints_raw = self._extract_centerline_in_window(mask, *cw)
                if keypoints_raw is not None:
                    centerline_raw, keypoints_raw = orient_pose_pair(
                        centerline_raw, keypoints_raw, last_keypoints=new_track.last_keypoints, ref_head=new_track.orient_head
                    )
                    new_track.update_keypoints(keypoints_raw, centerline_raw)
                    new_track.last_observed_keypoints = keypoints_raw
                    self.update_sine_history(new_track.track_id, keypoints_raw, track=new_track)
                    new_track.last_keypoint_update_frame = frame_idx

        self.tracks.extend(new_tracks)
        if new_tracks and not self._head_config_applied and self.head_config:
            self._match_head_config_to_tracks(self.tracks)

        # 增加未匹配轨迹的丢失计数
        self._handle_lost_tracks(unmatched_tracks, frame, frame_idx)

        # 为detections分配tracker_id
        tracker_ids = [-1] * len(detections)
        for t_idx, d_idx in matches:
            tracker_ids[d_idx] = self.tracks[t_idx].track_id
        for d_idx, tid in new_track_ids_map.items():
            tracker_ids[d_idx] = tid

        predicted_tracks = []
        predicted_track_ids = []
        for t_idx in unmatched_tracks:
            track = self.tracks[t_idx]
            # 只在残留绘制窗口内把预测框（残留框）交给前端/画面；ID 仍在后台保留
            if track.lost_frames <= self.ghost_overlay_frames:
                predicted_tracks.append(track)
                predicted_track_ids.append(track.track_id)

        detections.tracker_id = np.array(tracker_ids, dtype=int)

        if len(predicted_tracks) > 0:
            detections = self.build_prediction_detections(predicted_tracks, detections)
            tracker_ids = list(tracker_ids) + predicted_track_ids
        else:
            tracker_ids = list(tracker_ids)

        detections.tracker_id = np.array(tracker_ids, dtype=int)

        self._prune_lost_tracks()

        # 平滑框（使用卡尔曼状态）
        tracks_by_id = {t.track_id: t for t in self.tracks}
        smoothed_bboxes = []
        for i, tid in enumerate(tracker_ids):
            if tid is not None and tid > 0:
                track = tracks_by_id.get(tid)
                if track is not None:
                    if i < len(detections.xyxy) - len(predicted_track_ids):
                        smoothed_bboxes.append(_normalize_bbox(track.get_state()))
                    else:
                        smoothed_bboxes.append(_normalize_bbox(track.last_predicted_bbox if track.last_predicted_bbox is not None else track.get_state()))
                else:
                    smoothed_bboxes.append(_normalize_bbox(detections.xyxy[i]))
            else:
                smoothed_bboxes.append(_normalize_bbox(detections.xyxy[i]))
        if len(smoothed_bboxes) > 0:
            detections.xyxy = np.vstack(smoothed_bboxes).astype(np.float32)

        # 过滤未分配到有效 ID 的冗余检测（确保画面与数据仅包含受控的有效目标）
        if len(detections) > 0:
            valid_mask = detections.tracker_id > 0
            detections = detections[valid_mask]

        return detections

# ==================== 初始化模型函数 ====================
def init_model(model_path):
    """加载YOLO模型，返回模型和设备"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    model.to(device)
    return model, device


# ==================== 纯白极简高级渲染与状态收集 ====================

_FONT_TOP = None
_FONT_BOT = None

def _get_fonts():
    global _FONT_TOP, _FONT_BOT
    if _FONT_TOP is None:
        font_candidates = [
            'C:/Windows/Fonts/msyhbd.ttc',
            'C:/Windows/Fonts/msyh.ttc',
            'C:/Windows/Fonts/simsun.ttc',
            'C:/Windows/Fonts/arial.ttf',
        ]
        font_p = None
        for p in font_candidates:
            if os.path.exists(p):
                font_p = p
                break
        if font_p:
            _FONT_TOP = ImageFont.truetype(font_p, 15)
            _FONT_BOT = ImageFont.truetype(font_p, 13)
        else:
            _FONT_TOP = ImageFont.load_default()
            _FONT_BOT = ImageFont.load_default()
    return _FONT_TOP, _FONT_BOT

def _draw_dashed_rect(frame, bbox, color, thickness=1, dash=10, gap=6):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    for x in range(x1, x2, dash + gap):
        cv2.line(frame, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(frame, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash + gap):
        cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), color, thickness)

def _draw_contour_overlay(frame, tracked_dets, tracker):
    if tracked_dets is None or len(tracked_dets) == 0:
        return frame

    tracker_ids = getattr(tracked_dets, 'tracker_id', None)
    masks = getattr(tracked_dets, 'mask', None)
    xyxy = getattr(tracked_dets, 'xyxy', None)

    if tracker_ids is None:
        return frame

    ghost_ids = getattr(tracker, 'ghost_ids', set())
    active_tids = {int(t) for t in tracker_ids if t is not None}

    # 1. 绘制活跃目标平滑渐变抗锯齿绿线轨迹（最近30帧尾迹）
    for tid, history in tracker.track_history.items():
        if tid in ghost_ids or tid not in active_tids:
            continue
        history_list = list(history)[-30:]
        if len(history_list) > 1:
            points = []
            for point in history_list:
                if isinstance(point, (tuple, list)) and len(point) == 2:
                    points.append([int(point[0]), int(point[1])])
            if len(points) > 1:
                p_arr = np.array(points, dtype=np.int32)
                for j in range(1, len(p_arr)):
                    color_intensity = int(255 * (j / len(p_arr)))
                    cv2.line(frame, tuple(p_arr[j - 1]), tuple(p_arr[j]), (0, color_intensity, 0), 2, lineType=cv2.LINE_AA)

    # 2. 绘制纯白多边形外部轮廓（按检测实例严格匹配）
    for idx, tid in enumerate(tracker_ids):
        if tid is None:
            continue
        tid_int = int(tid)
        if tid_int in ghost_ids:
            continue
        if masks is not None and len(masks) > idx:
            mask = masks[idx]
            if mask is not None:
                mask_array = np.asarray(mask)
                if mask_array.ndim == 3 and mask_array.shape[0] > 1:
                    binary = np.any(mask_array > 0, axis=0)
                else:
                    binary = (mask_array > 0).astype(np.uint8)
                bin_f = binary.astype(np.float32)
                blurred = cv2.GaussianBlur(bin_f, (5, 5), 1.0)
                bin_smooth = (blurred >= 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(bin_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for c in contours:
                    if cv2.contourArea(c) > 50:
                        cv2.drawContours(frame, [c], -1, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    # 3. 绘制纯白边界框或浅绿幽灵虚线框
    if xyxy is not None:
        for idx, tid in enumerate(tracker_ids):
            if tid is None or idx >= len(xyxy):
                continue
            tid_int = int(tid)
            bbox = xyxy[idx]
            x1, y1, x2, y2 = map(int, bbox)
            if tid_int in ghost_ids:
                _draw_dashed_rect(frame, (x1, y1, x2, y2), (160, 220, 160), 1)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # 4. 绘制纯白底色药丸徽章（PIL 矢量字体抗锯齿渲染）
    font_top, font_bot = _get_fonts()
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    frame_h = frame.shape[0]

    if xyxy is not None:
        for idx, tid in enumerate(tracker_ids):
            if tid is None or idx >= len(xyxy):
                continue
            tid_int = int(tid)
            bbox = xyxy[idx]
            x1, y1, x2, y2 = map(int, bbox)
            cycle_info = tracker.cycle_counts.get(tid_int, {})
            is_ghost = tid_int in ghost_ids

            tr = next((t for t in tracker.tracks if t.track_id == tid_int), None)
            user_lbl = f"[{tr.user_label}] " if (tr and tr.user_label) else ""

            if is_ghost:
                lost_f = int(cycle_info.get('lost_frames', 0))
                ghost_text = f"ID {tid_int} GHOST lost={lost_f}f"
                tb = draw.textbbox((0, 0), ghost_text, font=font_top)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
                bx1, by1 = x1, max(0, y1 - th - 8)
                bx2, by2 = bx1 + tw + 10, by1 + th + 8
                draw.rectangle([bx1, by1, bx2, by2], fill=(240, 255, 240), outline=(160, 220, 160), width=1)
                draw.text((bx1 + 5, by1 + 2), ghost_text, fill=(30, 120, 30), font=font_top)
            else:
                c_cnt = int(cycle_info.get('c_count', 0))
                s_cnt = int(cycle_info.get('s_count', 0))
                bi_flag = " (双向)" if cycle_info.get('is_bidirectional') else ""
                freq = float(cycle_info.get('avg_freq', 0.0))
                bpm = float(cycle_info.get('bends_per_min', 0.0))
                motion_type = str(cycle_info.get('motion_type', 'static'))
                activity = float(cycle_info.get('activity_score', 0.0))

                # 顶部药丸徽章：ID 4 [标签] C:0 S:1 0.50Hz
                top_text = f"ID {tid_int} {user_lbl}C:{c_cnt} S:{s_cnt}{bi_flag} {freq:.2f}Hz"
                tb = draw.textbbox((0, 0), top_text, font=font_top)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
                bx1, by1 = x1, max(0, y1 - th - 8)
                bx2, by2 = bx1 + tw + 10, by1 + th + 8
                draw.rectangle([bx1, by1, bx2, by2], fill=(255, 255, 255), outline=(255, 255, 255), width=1)
                draw.text((bx1 + 5, by1 + 2), top_text, fill=(0, 0, 0), font=font_top)

                # 底部药丸徽章：#C:0 S:1 0.50Hz 30/min sinusoidal 80%
                bot_text = f"#C:{c_cnt} S:{s_cnt} {freq:.2f}Hz {bpm:.0f}/min {motion_type} {activity:.0%}"
                bb = draw.textbbox((0, 0), bot_text, font=font_bot)
                bw, bh = bb[2] - bb[0], bb[3] - bb[1]
                bot_x1, bot_y1 = x1, min(frame_h - bh - 6, y2)
                bot_x2, bot_y2 = bot_x1 + bw + 8, bot_y1 + bh + 6
                draw.rectangle([bot_x1, bot_y1, bot_x2, bot_y2], fill=(255, 255, 255), outline=(255, 255, 255), width=1)
                draw.text((bot_x1 + 4, bot_y1 + 2), bot_text, fill=(0, 0, 0), font=font_bot)

    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 5. 若目标有头部锚点，绘制柔和的微型绿色头部点（半径3px）
    for track in tracker.tracks:
        if track.track_id not in ghost_ids and track.last_head_pos is not None:
            hx, hy = int(track.last_head_pos[0]), int(track.last_head_pos[1])
            cv2.circle(frame, (hx, hy), 3, (0, 255, 100), -1, lineType=cv2.LINE_AA)

    return frame

def annotate_and_collect_data(frame, tracked_dets, tracker, frame_idx):
    annotated_frame = frame.copy()
    annotated_frame = _draw_contour_overlay(annotated_frame, tracked_dets, tracker)

    # 全局帧统计信息
    ghost_ids = getattr(tracker, 'ghost_ids', set())
    active_count = len(tracker.tracks) - len(ghost_ids)
    cv2.putText(annotated_frame, f"Frame: {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(annotated_frame, f"Active IDs: {active_count} (Ghost: {len(ghost_ids)})", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, lineType=cv2.LINE_AA)

    # 收集跟踪数据
    track_data = []
    active_ids = []
    if hasattr(tracked_dets, 'tracker_id') and tracked_dets.tracker_id is not None:
        active_ids = sorted({int(tid) for tid in tracked_dets.tracker_id if tid is not None and int(tid) > 0})

    frame_h = frame.shape[0]
    for tid in active_ids:
        cycle_info = tracker.cycle_counts.get(tid, {})
        track = next((t for t in tracker.tracks if t.track_id == tid), None)

        centroid_bl = None
        head_bl = None
        keypoints_bl = None
        user_label = None
        head_kpt_idx = None

        if track is not None:
            user_label = track.user_label
            head_kpt_idx = track.head_kpt_idx
            history = tracker.track_history.get(tid)
            if history and len(history) > 0:
                lc = history[-1]
                cx_bl, cy_bl = to_bottom_left(lc[0], lc[1], frame_h)
                centroid_bl = {'x': cx_bl, 'y': cy_bl}
            if track.last_head_pos is not None:
                hx_bl, hy_bl = to_bottom_left(float(track.last_head_pos[0]), float(track.last_head_pos[1]), frame_h)
                head_bl = {'x': hx_bl, 'y': hy_bl}
            if track.last_keypoints is not None:
                keypoints_bl = [
                    {'x': kx_bl, 'y': ky_bl}
                    for kpt in track.last_keypoints
                    for kx_bl, ky_bl in [to_bottom_left(float(kpt[0]), float(kpt[1]), frame_h)]
                ]

        is_ghost = bool(cycle_info.get('is_ghost', tid in ghost_ids))
        is_static = bool(cycle_info.get('is_static', False) or (cycle_info.get('motion_type') == 'static'))

        track_data.append({
            'track_id': tid,
            'user_label': user_label,
            'c_count': int(cycle_info.get('c_count', 0)),
            's_count': int(cycle_info.get('s_count', 0)),
            'is_bidirectional': bool(cycle_info.get('is_bidirectional', False)),
            'motion_type': str(cycle_info.get('motion_type', 'unknown')),
            'activity_score': float(cycle_info.get('activity_score', 0.0)),
            'is_static': is_static,
            'is_ghost': is_ghost,
            'lost_frames': int(cycle_info.get('lost_frames', 0)),
            'live_freq': float(cycle_info.get('live_freq', 0.0)),
            'avg_freq': float(cycle_info.get('avg_freq', 0.0)),
            'cycles': int(cycle_info.get('cycles', 0)),
            'swings': int(cycle_info.get('swings', 0)),
            'bends_per_min': float(cycle_info.get('bends_per_min', 0.0)),
            'centroid_bl': centroid_bl,
            'head_bl': head_bl,
            'head_kpt_idx': head_kpt_idx,
            'keypoints_bl': keypoints_bl,
        })

    return annotated_frame, track_data

def process_frame(frame, model, device, tracker, frame_idx, min_activity=0.25):
    results = model(frame, verbose=False, device=device, retina_masks=True)[0]
    detections = sv.Detections.from_ultralytics(results)

    if len(detections) > 0:
        x1, y1, x2, y2 = detections.xyxy[:, 0], detections.xyxy[:, 1], detections.xyxy[:, 2], detections.xyxy[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        conf = detections.confidence if detections.confidence is not None else np.ones(len(detections))
        keep = (areas >= tracker.MIN_AREA) & (conf >= tracker.MIN_CONFIDENCE)
        detections = detections[keep]

    masks = detections.mask if hasattr(detections, 'mask') else None
    tracked_dets = tracker.update_tracking(detections, masks, frame_idx, frame=frame)
    return annotate_and_collect_data(frame, tracked_dets, tracker, frame_idx)

def process_frame_batch(frames, model, device, tracker, start_frame_idx, min_activity=0.25):
    results_list = model(frames, verbose=False, device=device, retina_masks=True)
    annotated_frames = []
    track_data_list = []

    for i, results in enumerate(results_list):
        frame = frames[i]
        frame_idx = start_frame_idx + i

        detections = sv.Detections.from_ultralytics(results)
        if len(detections) > 0:
            x1, y1, x2, y2 = detections.xyxy[:, 0], detections.xyxy[:, 1], detections.xyxy[:, 2], detections.xyxy[:, 3]
            areas = (x2 - x1) * (y2 - y1)
            conf = detections.confidence if detections.confidence is not None else np.ones(len(detections))
            keep = (areas >= tracker.MIN_AREA) & (conf >= tracker.MIN_CONFIDENCE)
            detections = detections[keep]

        masks = detections.mask if hasattr(detections, 'mask') else None
        tracked_dets = tracker.update_tracking(detections, masks, frame_idx, frame=frame)

        annotated_frame, track_data = annotate_and_collect_data(frame, tracked_dets, tracker, frame_idx)
        annotated_frames.append(annotated_frame)
        track_data_list.append(track_data)

    return annotated_frames, track_data_list

def process_track_single(frame, results, tracker, frame_idx):
    detections = sv.Detections.from_ultralytics(results)
    if len(detections) > 0:
        x1, y1, x2, y2 = detections.xyxy[:, 0], detections.xyxy[:, 1], detections.xyxy[:, 2], detections.xyxy[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        conf = detections.confidence if detections.confidence is not None else np.ones(len(detections))
        keep = (areas >= tracker.MIN_AREA) & (conf >= tracker.MIN_CONFIDENCE)
        detections = detections[keep]

    masks = detections.mask if hasattr(detections, 'mask') else None
    tracked_dets = tracker.update_tracking(detections, masks, frame_idx, frame=frame)
    return annotate_and_collect_data(frame, tracked_dets, tracker, frame_idx)
