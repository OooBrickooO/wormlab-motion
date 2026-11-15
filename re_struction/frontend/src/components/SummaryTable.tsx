import React, { useState } from 'react';
import type { CUnit, TrackSummary } from '../api/types';
import { CUnitToggle } from './CUnitToggle';
import { pickC } from './LiveTrackTable';

export interface SummaryTableProps {
  items: TrackSummary[];
  defaultCUnit?: CUnit;
}

export function SummaryTable({ items, defaultCUnit = 'cycle' }: SummaryTableProps) {
  const [unit, setUnit] = useState<CUnit>(defaultCUnit);
  // 后进入的虫默认不展示
  const rows = items.filter((i) => i.late_entry !== true && i.counted !== false);
  return (
    <section className="id-pose-section">
      <div className="sub-video-header">
        <h3>最终输出 ID 与姿态计数</h3>
      </div>
      {rows.length === 0 ? (
        <div className="history-empty">任务完成后会显示最终汇总</div>
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
                <th>平均频率(Hz)</th>
                <th>完整周期</th>
                <th>总路径(px)</th>
                <th>净位移(px)</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((item) => (
                <tr key={`summary-${item.track_id}`}>
                  <td>{item.track_id}</td>
                  <td>{item.user_label || '—'}</td>
                  <td>
                    {item.is_static || item.motion_type === 'static' ? (
                      <span className="status-badge static">静止</span>
                    ) : (
                      <span className="status-badge active">{item.motion_type || '活跃'}</span>
                    )}
                  </td>
                  <td>{pickC(item, unit)}</td>
                  <td>{item.s_count ?? 0}</td>
                  <td>{item.is_bidirectional ? '是' : '否'}</td>
                  <td>{Number(item.avg_freq ?? 0).toFixed(2)}</td>
                  <td>{item.cycles ?? 0}</td>
                  <td>{item.spatial_stats?.total_path_length_px?.toFixed(1) ?? '—'}</td>
                  <td>{item.spatial_stats?.net_displacement_px?.toFixed(1) ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
export default SummaryTable;
