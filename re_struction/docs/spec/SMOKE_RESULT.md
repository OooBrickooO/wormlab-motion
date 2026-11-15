# T31 端到端冒烟结果

- 视频: Swim-4.mp4 | profile=swim_liquid | worm_count=1 | c_unit=cycle
- 状态: **completed**，耗时 34.3 s，socket 收到帧 366
- 事件: [('phase', 'detect'), ('phase', 'analyze'), ('phase', 'analyze'), ('phase', 'render'), ('phase', 'render'), ('phase', 'render'), ('phase', 'render'), ('phase', 'render'), ('phase', 'render'), ('phase', 'render')]
- 下载链接 HTTP 状态: {'video_url': 200, 'log_url': 200, 'sub_videos_url': 200, 'summary_url': 200, 'csv_url': 200}
- 子视频: {'videos': [{'filename': 'worm_1_cycles_0.mp4', 'url': '/sub_video/20261111_082910_ab43e5/worm_1_cycles_0.mp4'}]}

## id_pose_summary
```json
[
 {
  "track_id": 1,
  "user_label": null,
  "c_count": 0,
  "s_count": 0,
  "is_bidirectional": false,
  "motion_type": "static",
  "activity_score": 0.0,
  "is_static": true,
  "live_freq": 0.0,
  "avg_freq": 0.0,
  "cycles": 0,
  "swings": 0,
  "bends_per_min": 0.0,
  "spatial_stats": {
   "total_path_length_px": 0.0,
   "net_displacement_px": 0.0,
   "start_centroid_bl": [
    0.0,
    0.0
   ],
   "end_centroid_bl": [
    0.0,
    0.0
   ],
   "bbox_coverage_bl": {
    "x_min": 0.0,
    "y_min": 0.0,
    "x_max": 0.0,
    "y_max": 0.0
   }
  },
  "s_cycles": 0.0,
  "c_cycles": 0,
  "c_bends": 0,
  "c_unit": "cycle",
  "c_turn_count": 0,
  "freq_s_hz": 0.0,
  "freq_c_hz": 0.0,
  "mode_time": {
   "static": 24.401,
   "crawl": 0.0,
   "thrash": 0.0,
   "turn": 0.0,
   "moving": 0.0,
   "invalid": 0.0
  },
  "rejected": {
   "no_disp": 0,
   "wrong_dir": 0,
   "backward": 0
  },
  "counted": true,
  "late_entry": false
 }
]
```
