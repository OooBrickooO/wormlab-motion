export type TaskStatus = 'processing' | 'paused' | 'completed' | 'failed' | 'cancelled';
export type CUnit = 'cycle' | 'bend';

export interface Pt {
  x: number;
  y: number;
}

export interface TrackLive {
  track_id: number;
  user_label: string | null;
  c_count: number;
  s_count: number;
  is_bidirectional: boolean;
  motion_type: string;
  activity_score: number;
  is_static: boolean;
  is_ghost: boolean;
  lost_frames: number;
  live_freq: number;
  avg_freq: number;
  cycles: number;
  swings: number;
  bends_per_min: number;
  centroid_bl: Pt | null;
  head_bl: Pt | null;
  head_kpt_idx: number | null;
  keypoints_bl: Pt[] | null;
  behavior?: string;
  s_cycles?: number;
  freq_s_hz?: number;
  freq_c_hz?: number;
  pose_quality?: string;
  c_cycles?: number;
  c_bends?: number;
  c_unit?: CUnit;
  counted?: boolean;
  late_entry?: boolean;
}

export interface SpatialStats {
  total_path_length_px: number;
  net_displacement_px: number;
  start_centroid_bl: [number, number];
  end_centroid_bl: [number, number];
  bbox_coverage_bl: {
    x_min: number;
    y_min: number;
    x_max: number;
    y_max: number;
  };
}

export interface TrackSummary
  extends Omit<
    TrackLive,
    'is_ghost' | 'lost_frames' | 'centroid_bl' | 'head_bl' | 'head_kpt_idx' | 'keypoints_bl' | 'behavior' | 'pose_quality'
  > {
  spatial_stats: SpatialStats;
  c_turn_count?: number;
  mode_time?: Record<string, number>;
  rejected?: Record<string, number>;
}

export interface FrameStats {
  frame: number;
  active_tracks: number;
  tracks: TrackLive[];
}

export interface FrameEvent {
  task_id: string;
  frame: number;
  total_frames: number;
  image: string;
  stats: FrameStats;
}

export interface CompleteEvent {
  task_id: string;
  video_url: string;
  log_url: string;
  sub_videos_url: string;
  summary_url: string;
  csv_url: string;
  id_pose_summary: TrackSummary[];
}

export interface PhaseEvent {
  task_id: string;
  phase: 'detect' | 'analyze' | 'render' | 'done';
  progress: number;
}

export interface TaskState {
  status: TaskStatus;
  filename: string;
  current_frame: number;
  total_frames: number;
  active_tracks: number;
  latest_frame: string;
  tracks: TrackLive[];
  video_url: string;
  log_url: string;
  sub_videos_url: string;
  summary_url: string;
  csv_url: string;
  id_pose_summary: TrackSummary[];
  error_message: string;
  control: string;
  worm_count: number | null;
  profile?: string;
  c_unit?: CUnit;
  phase?: string;
  queued?: boolean;
}

export interface FirstFrameWorm {
  det_idx: number;
  confidence: number | null;
  bbox: [number, number, number, number];
  centroid: [number, number];
  keypoints: [number, number][] | null;
  is_manual?: boolean;
}

export interface FirstFrameData {
  frame_width: number;
  frame_height: number;
  annotated: string;
  worms: FirstFrameWorm[];
}

export interface HeadConfigItem {
  det_idx: number;
  head_kpt_idx: number;
  user_label: string;
  bbox: number[];
  centroid?: [number, number];
  keypoints?: [number, number][];
}

export interface PreviewCountResponse {
  count: number;
  raw_count: number;
  annotated: string;
}

export interface ProfileItem {
  id: string;
  label: string;
}

export interface SubVideoItem {
  filename: string;
  url: string;
}

export interface EggDetectionItem {
  bbox: [number, number, number, number];
  confidence: number;
  class_id: number;
}

export interface EggCountResponse {
  egg_count: number;
  egg_count_90: number;
  egg_count_95: number;
  egg_class_name: string;
  raw_image: string;
  eggs: EggDetectionItem[];
  frame_width: number;
  frame_height: number;
}
