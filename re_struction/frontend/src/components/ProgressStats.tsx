import React from 'react';
import type { FrameStats } from '../api/types';
import { formatPercent } from '../utils/format';

export interface ProgressStatsProps {
  currentFrame: string | null;
  currentStats: FrameStats | null;
  totalFrames: number;
  phase?: string;
  hasVideoResult?: boolean;
}

export function ProgressStats({
  currentFrame,
  currentStats,
  totalFrames,
  phase,
  hasVideoResult,
}: ProgressStatsProps) {
  const progress = formatPercent(currentStats?.frame, totalFrames);
  const showStreamBox = Boolean(currentFrame) || !hasVideoResult;
  const placeholderText =
    phase === 'render'
      ? '检测已完成，正在渲染完整结果视频与局部子视频...'
      : '等待模型输出帧...';
  return (
    <>
      {showStreamBox && (
        <div className="stream-wrap">
          {currentFrame ? (
            <img src={currentFrame} alt="实时处理画面" className="stream-frame" />
          ) : (
            <div className="stream-placeholder">{placeholderText}</div>
          )}
        </div>
      )}

      <div className="progress-wrap">
        <div className="progress-text">
          进度 {currentStats?.frame || 0} / {totalFrames || 0}
        </div>
        <div className="progress-track">
          <div className="progress-value" style={{ width: `${progress}%` }} />
        </div>
      </div>

      <div className="stats-grid">
        <div className="stat-item">
          <span>活跃目标</span>
          <strong>{currentStats?.active_tracks ?? 0}</strong>
        </div>
        <div className="stat-item">
          <span>当前帧</span>
          <strong>{currentStats?.frame ?? 0}</strong>
        </div>
        <div className="stat-item">
          <span>总帧数</span>
          <strong>{totalFrames || 0}</strong>
        </div>
      </div>
    </>
  );
}
export default ProgressStats;
