import React, { useState } from 'react';
import type { CUnit, TrackLive } from '../api/types';
import { formatCoordPt } from '../utils/format';
import { CUnitToggle } from './CUnitToggle';

export function pickC(item: { c_count?: number; c_cycles?: number; c_bends?: number }, u: CUnit): number {
  const v = u === 'cycle' ? item.c_cycles : item.c_bends;
  return v ?? item.c_count ?? 0;
}

export interface LiveTrackTableProps {
  tracks: TrackLive[];
  defaultCUnit?: CUnit;
}

export function LiveTrackTable({ tracks, defaultCUnit = 'cycle' }: LiveTrackTableProps) {
  const [unit, setUnit] = useState<CUnit>(defaultCUnit);
  return (
    <section className="id-pose-section">
      <div className="sub-video-header">
        <h3>当前帧 ID 与姿态计数</h3>
      </div>
      {tracks.length === 0 ? (
        <div className="history-empty">当前帧暂无可显示的线虫计数</div>
      ) : (
        <div className="id-pose-table-wrap">
          <table className="id-pose-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>标签</th>
                <th>状态</th>
                <th>
                  C形 <CUnitToggle value={unit} onChange={setUnit} />
                </th>
                <th>S形</th>
                <th>双向</th>
                <th>频率(Hz)</th>
                <th>质心坐标</th>
                <th>头部坐标</th>
              </tr>
            </thead>
            <tbody>
              {tracks.map((item) => {
                const late = item.late_entry === true;
                let badge = (
                  <span className="status-badge active" title={item.behavior}>
                    {item.motion_type || '活跃'}
                  </span>
                );
                if (late) {
                  badge = <span className="status-badge ghost">后进入·不计数</span>;
                } else if (item.is_ghost) {
                  badge = <span className="status-badge ghost">残留(丢{item.lost_frames ?? 0}f)</span>;
                } else if (item.is_static || item.motion_type === 'static') {
                  badge = <span className="status-badge static">静止</span>;
                }
                const freq = Number(item.live_freq || item.avg_freq || 0).toFixed(2);
                return (
                  <tr key={`live-${item.track_id}`}>
                    <td>{item.track_id}</td>
                    <td>{item.user_label || '—'}</td>
                    <td>{badge}</td>
                    <td>{late ? '—' : pickC(item, unit)}</td>
                    <td>{late ? '—' : item.s_count ?? 0}</td>
                    <td>{item.is_bidirectional ? '是' : '否'}</td>
                    <td>{freq}</td>
                    <td>{formatCoordPt(item.centroid_bl)}</td>
                    <td>{formatCoordPt(item.head_bl)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
export default LiveTrackTable;
