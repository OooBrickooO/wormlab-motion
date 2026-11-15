import React from 'react';
import type { HistoryRecord, ResourceKey } from '../hooks/useHistory';

export interface HistoryPanelProps {
  items: HistoryRecord[];
  onMonitor: (item: HistoryRecord) => void;
  onOpen: (item: HistoryRecord, key: ResourceKey) => void;
}

export function HistoryPanel({ items, onMonitor, onOpen }: HistoryPanelProps) {
  return (
    <aside className="panel glass-card history-panel">
      <h2>历史</h2>
      {items.length === 0 && <div className="history-empty">暂无历史记录</div>}
      <div className="history-list">
        {items.map((item) => (
          <article key={item.id + item.time} className="history-item">
            <div className="history-title">{item.name}</div>
            <div className="history-time">{item.time}</div>
            <div className="history-time">状态: {item.status || 'completed'}</div>
            <div className="history-actions">
              <button type="button" className="glass-btn mini" onClick={() => onMonitor(item)}>
                监视
              </button>
              <button type="button" className="glass-btn mini" onClick={() => onOpen(item, 'video')}>
                查看
              </button>
              <button type="button" className="glass-btn mini" onClick={() => onOpen(item, 'log')}>
                日志
              </button>
              <button type="button" className="glass-btn mini" onClick={() => onOpen(item, 'zip')}>
                子视频
              </button>
              <button type="button" className="glass-btn mini" onClick={() => onOpen(item, 'summary')}>
                汇总
              </button>
            </div>
          </article>
        ))}
      </div>
    </aside>
  );
}
export default HistoryPanel;
