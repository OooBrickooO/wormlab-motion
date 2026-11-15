import React from 'react';
import type { CUnit, ProfileItem } from '../api/types';

export type AnalysisMode = 'tracking' | 'egg_count';

export interface InputPanelProps {
  analysisMode: AnalysisMode;
  onModeSwitch: (m: AnalysisMode) => void;
  status: string;
  taskId: string | null;
  selectedFiles: File[];
  selectedFileNames: string;
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  models: string[];
  selectedModel: string;
  modelsLoading: boolean;
  onModelChange: (name: string) => void;
  profiles: ProfileItem[];
  selectedProfile: string;
  onProfileChange: (id: string) => void;
  cUnit: CUnit;
  onCUnitChange: (u: CUnit) => void;
  wormCount: string;
  onWormCountChange: (v: string) => void;
  previewing: boolean;
  suggestedCount: number | null;
  previewAnnotated: string | null;
  onStart: () => void;
  onPause: () => void;
  onResume: () => void;
  onCancelTask: () => void;
  onCancelPreview: () => void;
  onEggCount: () => void;
  eggCounting: boolean;
  error: string;
  videoUrl: string;
  logUrl: string;
  zipUrl: string;
  summaryUrl: string;
  csvUrl: string;
  onOpenUrl: (url: string) => void;
}

const selectStyle: React.CSSProperties = {
  width: '100%',
  marginTop: '6px',
  padding: '9px 12px',
  borderRadius: '12px',
  border: '1px solid rgba(22, 119, 255, 0.18)',
  background: 'rgba(255, 255, 255, 0.75)',
  color: '#0f2f57',
  outline: 'none',
  fontWeight: 500,
};
const labelStyle: React.CSSProperties = {
  fontSize: '13px',
  color: 'var(--muted)',
  fontWeight: 500,
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
};

export function InputPanel(p: InputPanelProps) {
  const busy = p.status === 'uploading' || p.status === 'processing';
  const isEgg = p.analysisMode === 'egg_count';
  return (
    <aside className="panel glass-card left-panel">
      <h2>输入区</h2>

      <div className="mode-tab-row">
        <button
          type="button"
          id="mode-tab-tracking"
          className={`mode-tab ${!isEgg ? 'mode-tab-active' : ''}`}
          onClick={() => p.onModeSwitch('tracking')}
          disabled={busy}
        >
          运动追踪
        </button>
        <button
          type="button"
          id="mode-tab-egg"
          className={`mode-tab ${isEgg ? 'mode-tab-active mode-tab-egg' : ''}`}
          onClick={() => p.onModeSwitch('egg_count')}
          disabled={busy}
        >
          虫卵计数
        </button>
      </div>

      <label className="upload-box" htmlFor="file-input">
        <input
          id="file-input"
          type="file"
          accept="video/*,image/*"
          multiple={!isEgg}
          onChange={p.onFileChange}
          disabled={busy}
        />
        <span>{isEgg ? '选择视频或图片' : '选择视频'}</span>
      </label>

      <div className="file-meta">
        <div>文件: {p.selectedFileNames || '未选择'}</div>
        <div>文件数: {p.selectedFiles.length}</div>
        <div>状态: {p.status}</div>
        {!isEgg && <div>任务ID: {p.taskId || '无'}</div>}
      </div>

      {!isEgg && (
        <div className="model-select-wrap">
          <label htmlFor="profile-select" style={labelStyle}>
            <span>分析环境</span>
          </label>
          <select
            id="profile-select"
            className="glass-select model-select-input"
            value={p.selectedProfile}
            onChange={(e) => p.onProfileChange(e.target.value)}
            disabled={busy}
            style={selectStyle}
          >
            {p.profiles.map((it) => (
              <option key={it.id} value={it.id}>
                {it.label}
              </option>
            ))}
          </select>
          <label htmlFor="cunit-select" style={{ ...labelStyle, marginTop: 8 }}>
            <span>C形统计口径</span>
          </label>
          <select
            id="cunit-select"
            className="glass-select model-select-input"
            value={p.cUnit}
            onChange={(e) => p.onCUnitChange(e.target.value as CUnit)}
            disabled={busy}
            style={selectStyle}
          >
            <option value="cycle">周期（左+右一个来回）</option>
            <option value="bend">单次弯曲</option>
          </select>
        </div>
      )}

      <div className="model-select-wrap">
        <label htmlFor="model-select" style={labelStyle}>
          <span>选择识别模型</span>
          {p.modelsLoading && (
            <span style={{ fontSize: '12px', color: 'var(--primary)' }}>
              <span className="preview-spinner" /> 加载中...
            </span>
          )}
          {isEgg && (
            <span style={{ fontSize: '11px', color: '#26a65b', fontWeight: 600 }}>
              已锁定 EggCount.pt
            </span>
          )}
        </label>
        <select
          id="model-select"
          className="glass-select model-select-input"
          value={p.selectedModel}
          onChange={(e) => p.onModelChange(e.target.value)}
          disabled={isEgg || p.previewing || busy || p.models.length === 0}
          style={selectStyle}
        >
          {p.models.length === 0 ? (
            <option value="">{p.modelsLoading ? '加载模型列表中...' : '未找到可用模型'}</option>
          ) : (
            p.models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))
          )}
        </select>
      </div>

      {!isEgg && (
        <div className="worm-count-wrap">
          <label htmlFor="worm-count-input" style={labelStyle}>
            <span>
              初始虫子数量 <span style={{ color: '#d64040' }}>*</span>
            </span>
            {p.previewing && (
              <span style={{ fontSize: '12px', color: 'var(--primary)' }}>
                <span className="preview-spinner" /> 分析第一帧...
              </span>
            )}
            {!p.previewing && p.suggestedCount !== null && (
              <span style={{ fontSize: '12px', color: '#26a65b' }}>
                模型建议：{p.suggestedCount} 条
              </span>
            )}
          </label>
          <input
            id="worm-count-input"
            type="number"
            min="1"
            step="1"
            placeholder={
              p.previewing
                ? '正在分析...'
                : p.suggestedCount !== null
                ? `建议值: ${p.suggestedCount}`
                : '请输入正整数'
            }
            className="glass-input worm-count-input"
            value={p.wormCount}
            onChange={(e) => p.onWormCountChange(e.target.value)}
            disabled={p.previewing || busy}
            style={{
              width: '100%',
              marginTop: '6px',
              padding: '10px 14px',
              borderRadius: '12px',
              border: '1px solid rgba(22, 119, 255, 0.18)',
              background: 'rgba(255, 255, 255, 0.6)',
              color: '#0f2f57',
              outline: 'none',
            }}
          />
          {!p.previewing && p.previewAnnotated && (
            <div
              style={{
                marginTop: '8px',
                borderRadius: '10px',
                overflow: 'hidden',
                border: '1px solid rgba(22,119,255,0.18)',
              }}
            >
              <img src={p.previewAnnotated} alt="第一帧检测预览" style={{ width: '100%', display: 'block' }} />
            </div>
          )}
        </div>
      )}

      {isEgg && (
        <div className="egg-mode-hint">
          <div>
            <strong>虫卵计数模式</strong>
          </div>
          <div>仅分析第一帧图像，检测并统计线虫与虫卵数量，不进行后续分析。</div>
        </div>
      )}

      {!isEgg ? (
        <button
          type="button"
          className="glass-btn launch-btn"
          onClick={p.onStart}
          disabled={
            p.previewing ||
            p.selectedFiles.length === 0 ||
            p.status === 'uploading' ||
            !p.wormCount ||
            parseInt(p.wormCount, 10) <= 0
          }
        >
          {p.previewing
            ? '等待预览完成...'
            : p.status === 'uploading'
            ? '上传中...'
            : p.status === 'processing'
            ? '识别中...'
            : '启动识别'}
        </button>
      ) : (
        <button
          type="button"
          id="btn-egg-count"
          className="glass-btn launch-btn egg-launch-btn"
          onClick={p.onEggCount}
          disabled={p.eggCounting || p.selectedFiles.length === 0}
        >
          {p.eggCounting ? '计数中...' : '开始计数'}
        </button>
      )}

      {!isEgg && (
        <div className="task-control-row">
          <button
            type="button"
            className="glass-btn"
            onClick={p.onPause}
            disabled={p.previewing || !p.taskId || p.status !== 'processing'}
          >
            暂停处理
          </button>
          <button
            type="button"
            className="glass-btn"
            onClick={p.onResume}
            disabled={p.previewing || !p.taskId || p.status !== 'paused'}
          >
            继续处理
          </button>
          <button
            type="button"
            className="glass-btn"
            onClick={() => (p.previewing ? p.onCancelPreview() : p.onCancelTask())}
            disabled={!p.previewing && (!p.taskId || ['completed', 'cancelled', 'idle'].includes(p.status))}
          >
            {p.previewing ? '取消预览' : '取消任务'}
          </button>
        </div>
      )}

      {p.error && <div className="error-text">{p.error}</div>}

      {!isEgg && (
        <div className="download-row">
          <button type="button" className="glass-btn" onClick={() => p.onOpenUrl(p.videoUrl)} disabled={!p.videoUrl}>
            下载结果视频
          </button>
          <button type="button" className="glass-btn" onClick={() => p.onOpenUrl(p.logUrl)} disabled={!p.logUrl}>
            下载日志
          </button>
          <button type="button" className="glass-btn" onClick={() => p.onOpenUrl(p.zipUrl)} disabled={!p.zipUrl}>
            下载子视频ZIP
          </button>
          <button type="button" className="glass-btn" onClick={() => p.onOpenUrl(p.summaryUrl)} disabled={!p.summaryUrl}>
            下载ID姿态JSON
          </button>
          <button type="button" className="glass-btn" onClick={() => p.onOpenUrl(p.csvUrl)} disabled={!p.csvUrl}>
            下载摆动汇总CSV
          </button>
        </div>
      )}
    </aside>
  );
}
export default InputPanel;
