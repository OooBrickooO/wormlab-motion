import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ReactDOM from 'react-dom';
import io from 'socket.io-client';
import axios from 'axios';
import './App.css';

const ACTIVE_TASK_KEY = 'motion_active_task_id';
const ACTIVE_FILE_KEY = 'motion_active_file_name';
const HISTORY_KEY = 'motion_history_records';
const BACKEND_BASE_URL = 'http://localhost:5000';

axios.defaults.baseURL = BACKEND_BASE_URL;

function toAbsoluteUrl(url) {
  if (!url) return '';
  if (url.startsWith('http://') || url.startsWith('https://')) return url;
  return `${BACKEND_BASE_URL}${url}`;
}

function formatPercent(current, total) {
  if (!current || !total) return 0;
  return Math.min(100, Math.round((current / total) * 100));
}

// ===================== HeadMarkModal =====================
// 颜色池（每条虫子一种颜色）
const WORM_COLORS = [
  '#ff6b35', '#28c76f', '#7b68ee', '#ff9f43', '#00cec9',
  '#e84393', '#fdcb6e', '#0984e3', '#6c5ce7', '#a29bfe',
];

function drawCanvas(canvas, img, worms, wormSelections, hoveredDet, drawingRect = null) {
  if (!canvas || !img) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.drawImage(img, 0, 0, W, H);

  const scaleX = W / img.naturalWidth;
  const scaleY = H / img.naturalHeight;

  worms.forEach((worm, idx) => {
    const sel = wormSelections[worm.det_idx] || {};
    const enabled = sel.enabled !== false;
    const color = WORM_COLORS[idx % WORM_COLORS.length];
    const isHovered = hoveredDet === worm.det_idx;
    const alpha = enabled ? 1.0 : 0.28;

    ctx.globalAlpha = alpha;

    // Bounding box
    const [bx1, by1, bx2, by2] = worm.bbox;
    ctx.strokeStyle = isHovered ? '#fff' : color;
    ctx.lineWidth = isHovered ? 2.5 : 1.5;
    ctx.strokeRect(bx1 * scaleX, by1 * scaleY, (bx2 - bx1) * scaleX, (by2 - by1) * scaleY);

    // Label
    ctx.fillStyle = color;
    ctx.font = 'bold 13px sans-serif';
    ctx.fillText(sel.user_label || `#${worm.det_idx}`, bx1 * scaleX + 4, by1 * scaleY - 5);

    if (!worm.keypoints) { ctx.globalAlpha = 1; return; }

    const kpts = worm.keypoints;

    // Skeleton lines
    ctx.strokeStyle = `${color}88`;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    kpts.forEach(([kx, ky], ki) => {
      if (ki === 0) ctx.moveTo(kx * scaleX, ky * scaleY);
      else ctx.lineTo(kx * scaleX, ky * scaleY);
    });
    ctx.stroke();

    // Draw all keypoints
    kpts.forEach(([kx, ky], ki) => {
      const isEndpoint = ki === 0 || ki === 10;
      const isHead = sel.head_kpt_idx === ki;
      const cx2 = kx * scaleX;
      const cy2 = ky * scaleY;

      if (isHead) {
        // Glow ring for head
        ctx.shadowColor = '#26b450';
        ctx.shadowBlur = 12;
        ctx.beginPath();
        ctx.arc(cx2, cy2, 10, 0, Math.PI * 2);
        ctx.fillStyle = '#26b450';
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 9px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('H', cx2, cy2);
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
      } else if (isEndpoint && enabled) {
        // Endpoint: larger clickable circle
        ctx.beginPath();
        ctx.arc(cx2, cy2, 7, 0, Math.PI * 2);
        ctx.fillStyle = '#1677ff';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        // Index label
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 8px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(ki), cx2, cy2);
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
      } else {
        // Middle point: small gray dot
        ctx.beginPath();
        ctx.arc(cx2, cy2, 3, 0, Math.PI * 2);
        ctx.fillStyle = isEndpoint ? '#56a3ff' : 'rgba(200,200,200,0.7)';
        ctx.fill();
        // Index for non-endpoints
        ctx.fillStyle = 'rgba(255,255,200,0.85)';
        ctx.font = '9px sans-serif';
        ctx.fillText(String(ki), cx2 + 5, cy2 - 3);
      }
    });

    ctx.globalAlpha = 1;
  });

  // 绘制手动框选时的实时虚线预览框
  if (drawingRect && drawingRect.w > 0 && drawingRect.h > 0) {
    ctx.strokeStyle = '#28c76f';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(drawingRect.x, drawingRect.y, drawingRect.w, drawingRect.h);
    ctx.fillStyle = 'rgba(40, 199, 111, 0.16)';
    ctx.fillRect(drawingRect.x, drawingRect.y, drawingRect.w, drawingRect.h);
    ctx.setLineDash([]);
  }
}

function HeadMarkModal({ firstFrameData, firstFrameBlob, wormSelections, onSelectionsChange, onAddManualWorm, onConfirm, onSkip }) {
  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const wrapRef = useRef(null);
  const [imgLoaded, setImgLoaded] = useState(false);
  const [hoveredDet, setHoveredDet] = useState(null);
  const [canvasSize, setCanvasSize] = useState({ w: 800, h: 450 });

  // 手动框选状态
  const [isDrawMode, setIsDrawMode] = useState(false);
  const [dragStart, setDragStart] = useState(null);
  const [drawingRect, setDrawingRect] = useState(null);
  const [extracting, setExtracting] = useState(false);

  // Load image once
  useEffect(() => {
    if (!firstFrameData?.annotated) return;
    const img = new Image();
    img.onload = () => {
      imgRef.current = img;
      setImgLoaded(true);
    };
    img.src = `data:image/jpeg;base64,${firstFrameData.annotated}`;
  }, [firstFrameData]);

  // Size the canvas to fit container
  useEffect(() => {
    if (!imgLoaded || !wrapRef.current) return;
    const img = imgRef.current;
    const wrap = wrapRef.current;
    const maxW = wrap.clientWidth - 4;
    const maxH = wrap.clientHeight - 4;
    const ratio = img.naturalWidth / img.naturalHeight;
    let w = maxW;
    let h = w / ratio;
    if (h > maxH) { h = maxH; w = h * ratio; }
    setCanvasSize({ w: Math.floor(w), h: Math.floor(h) });
  }, [imgLoaded]);

  // Redraw whenever state changes
  useEffect(() => {
    if (!imgLoaded || !canvasRef.current) return;
    const canvas = canvasRef.current;
    canvas.width = canvasSize.w;
    canvas.height = canvasSize.h;
    drawCanvas(canvas, imgRef.current, firstFrameData.worms, wormSelections, hoveredDet, drawingRect);
  }, [imgLoaded, canvasSize, wormSelections, hoveredDet, firstFrameData, drawingRect]);

  const getClickedEndpoint = useCallback((clientX, clientY) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const mx = (clientX - rect.left) * (canvas.width / rect.width);
    const my = (clientY - rect.top) * (canvas.height / rect.height);
    const scaleX = canvas.width / imgRef.current.naturalWidth;
    const scaleY = canvas.height / imgRef.current.naturalHeight;
    const HIT_R = 18; // px hit radius in canvas coords

    let bestDist = HIT_R;
    let bestResult = null;

    firstFrameData.worms.forEach((worm) => {
      const sel = wormSelections[worm.det_idx];
      if (!sel?.enabled) return;
      if (!worm.keypoints) return;
      // Only allow endpoints: index 0 or 10
      [0, 10].forEach((ki) => {
        const [kx, ky] = worm.keypoints[ki];
        const dx = kx * scaleX - mx;
        const dy = ky * scaleY - my;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < bestDist) {
          bestDist = dist;
          bestResult = { det_idx: worm.det_idx, head_kpt_idx: ki };
        }
      });
    });
    return bestResult;
  }, [firstFrameData, wormSelections]);

  const handleMouseDown = (e) => {
    if (!isDrawMode || extracting) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    setDragStart({ x: mx, y: my });
    setDrawingRect(null);
  };

  const handleMouseMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    if (isDrawMode) {
      if (dragStart) {
        const rx = Math.min(dragStart.x, mx);
        const ry = Math.min(dragStart.y, my);
        const rw = Math.abs(mx - dragStart.x);
        const rh = Math.abs(my - dragStart.y);
        setDrawingRect({ x: rx, y: ry, w: rw, h: rh });
      }
      return;
    }

    const hit = getClickedEndpoint(e.clientX, e.clientY);
    setHoveredDet(hit ? hit.det_idx : null);
  };

  const handleMouseUp = async (e) => {
    if (!isDrawMode || !dragStart || !drawingRect) {
      setDragStart(null);
      setDrawingRect(null);
      return;
    }

    const rect = drawingRect;
    setDragStart(null);
    setDrawingRect(null);

    if (rect.w < 10 || rect.h < 10) return;

    const canvas = canvasRef.current;
    if (!canvas || !imgRef.current) return;
    const scaleX = imgRef.current.naturalWidth / canvas.width;
    const scaleY = imgRef.current.naturalHeight / canvas.height;
    const imgBbox = [
      Math.round(rect.x * scaleX),
      Math.round(rect.y * scaleY),
      Math.round((rect.x + rect.w) * scaleX),
      Math.round((rect.y + rect.h) * scaleY),
    ];

    try {
      setExtracting(true);
      const fd = new FormData();
      if (firstFrameBlob) {
        fd.append('frame', firstFrameBlob, 'frame.jpg');
      }
      fd.append('bbox', JSON.stringify(imgBbox));
      const res = await axios.post('/extract_manual_worm', fd);
      if (res.data) {
        const maxIdx = firstFrameData.worms.reduce((m, w) => Math.max(m, w.det_idx), -1);
        const newDetIdx = maxIdx + 1;
        const newWorm = {
          det_idx: newDetIdx,
          confidence: 1.0,
          bbox: res.data.bbox,
          centroid: res.data.centroid,
          keypoints: res.data.keypoints,
          is_manual: true,
        };
        if (onAddManualWorm) {
          onAddManualWorm(newWorm);
        }
        setIsDrawMode(false);
      }
    } catch (err) {
      console.error('手动提取虫体失败:', err);
    } finally {
      setExtracting(false);
    }
  };

  const handleCanvasClick = (e) => {
    if (isDrawMode) return;
    const hit = getClickedEndpoint(e.clientX, e.clientY);
    if (!hit) return;
    onSelectionsChange(prev => ({
      ...prev,
      [hit.det_idx]: { ...prev[hit.det_idx], head_kpt_idx: hit.head_kpt_idx },
    }));
  };

  const toggleEnabled = (det_idx) => {
    onSelectionsChange(prev => ({
      ...prev,
      [det_idx]: { ...prev[det_idx], enabled: !prev[det_idx]?.enabled },
    }));
  };

  const setLabel = (det_idx, label) => {
    onSelectionsChange(prev => ({
      ...prev,
      [det_idx]: { ...prev[det_idx], user_label: label },
    }));
  };

  const allEnabledHaveHead = firstFrameData.worms.every((w) => {
    const sel = wormSelections[w.det_idx];
    if (!sel?.enabled) return true; // disabled worms don't need head
    return sel.head_kpt_idx !== null && sel.head_kpt_idx !== undefined;
  });

  const enabledCount = firstFrameData.worms.filter(w => wormSelections[w.det_idx]?.enabled).length;

  return (
    <div className="head-mark-overlay">
      <div className="head-mark-dialog">
        <div className="head-mark-header">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px' }}>
            <h2>🐛 头部标记 — 第一帧分析</h2>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <button
                type="button"
                className={`glass-btn mini draw-mode-btn ${isDrawMode ? 'active' : ''}`}
                onClick={() => {
                  setIsDrawMode(!isDrawMode);
                  setDrawingRect(null);
                  setDragStart(null);
                }}
                disabled={extracting}
              >
                {extracting ? '提取骨架中...' : isDrawMode ? '✖ 退出框选' : '➕ 手动框选漏检虫子'}
              </button>
            </div>
          </div>
          <p>
            在右侧勾选要分析的虫子，然后<strong>点击每条虫子两端的绿色/蓝色节点（0 或 10）</strong>指定为头部。
            已选中 {enabledCount} 条 / 共 {firstFrameData.worms.length} 条虫子。
            {isDrawMode && <strong style={{ color: '#28c76f', marginLeft: 8 }}>[框选模式已开启：请在画面上按住鼠标左键拉框]</strong>}
          </p>
        </div>

        <div className="head-mark-body">
          <div className="head-mark-canvas-wrap" ref={wrapRef}>
            {imgLoaded ? (
              <>
                <canvas
                  ref={canvasRef}
                  width={canvasSize.w}
                  height={canvasSize.h}
                  style={{ cursor: isDrawMode ? 'crosshair' : 'default' }}
                  onMouseDown={handleMouseDown}
                  onMouseMove={handleMouseMove}
                  onMouseUp={handleMouseUp}
                  onClick={handleCanvasClick}
                  onMouseLeave={() => { setHoveredDet(null); setDragStart(null); setDrawingRect(null); }}
                />
                <div className="head-mark-canvas-hint">
                  {isDrawMode ? '按住鼠标左键拖拽拉框以添加漏检虫子' : '点击两端节点（0 或 10）设为头部 · 绿色=已选头部'}
                </div>
              </>
            ) : (
              <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: '14px' }}>加载中...</div>
            )}
          </div>

          <div className="worm-list-sidebar">
            <div className="worm-list-title">虫子列表</div>
            {firstFrameData.worms.map((worm, idx) => {
              const sel = wormSelections[worm.det_idx] || {};
              const enabled = sel.enabled !== false;
              const headSet = sel.head_kpt_idx !== null && sel.head_kpt_idx !== undefined;
              const color = WORM_COLORS[idx % WORM_COLORS.length];
              let itemClass = 'worm-list-item';
              if (!enabled) itemClass += ' worm-item-disabled';
              else if (headSet) itemClass += ' worm-item-highlighted';
              else itemClass += ' worm-item-active';

              return (
                <div key={worm.det_idx} className={itemClass}>
                  <div className="worm-item-top">
                    <input
                      type="checkbox"
                      checked={enabled}
                      onChange={() => toggleEnabled(worm.det_idx)}
                    />
                    <span style={{ width: 12, height: 12, borderRadius: 3, background: color, flexShrink: 0, display: 'inline-block' }} />
                    {worm.is_manual && (
                      <span style={{ fontSize: '10px', color: '#1a7a3a', background: 'rgba(40,199,111,0.2)', padding: '1px 5px', borderRadius: 4, fontWeight: 700 }}>手动</span>
                    )}
                    <input
                      type="text"
                      className="worm-item-label-input"
                      value={sel.user_label || ''}
                      disabled={!enabled}
                      placeholder={`虫子 #${worm.det_idx}`}
                      onChange={(e) => setLabel(worm.det_idx, e.target.value)}
                    />
                  </div>
                  {worm.confidence !== null && !worm.is_manual && (
                    <div className="worm-item-conf">置信度: {(worm.confidence * 100).toFixed(1)}%</div>
                  )}
                  {worm.keypoints ? (
                    <div className="worm-item-head-status">
                      头部节点:&nbsp;
                      <span className={`head-kpt-badge ${headSet ? 'badge-set' : 'badge-unset'}`}>
                        {headSet ? `pt ${sel.head_kpt_idx}` : '未选'}
                      </span>
                      {!headSet && enabled && (
                        <span style={{ color: '#d64040', fontSize: '11px' }}>← 请在图中点选</span>
                      )}
                    </div>
                  ) : (
                    <div className="worm-item-conf" style={{ color: '#e88' }}>骨架提取失败（仍可分析）</div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        <div className="head-mark-footer">
          <button type="button" className="glass-btn btn-skip" onClick={onSkip}>
            跳过（不标记头部）
          </button>
          <button
            type="button"
            className="glass-btn btn-confirm"
            disabled={!allEnabledHaveHead || enabledCount === 0}
            onClick={onConfirm}
          >
            确认开始分析 ({enabledCount} 条)
          </button>
        </div>
      </div>
    </div>
  );
}

// ===================== GlassVideoPlayer =====================
function GlassVideoPlayer({ title, src, downloadName }) {
  const videoRef = useRef(null);

  const togglePlay = () => {
    const v = videoRef.current;
    if (!v) return;
    v.paused ? v.play() : v.pause();
  };

  const rewind = () => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = Math.max(0, v.currentTime - 10);
  };

  const capture = () => {
    const v = videoRef.current;
    if (!v || !v.videoWidth) return;
    const canvas = document.createElement('canvas');
    canvas.width = v.videoWidth; canvas.height = v.videoHeight;
    canvas.getContext('2d').drawImage(v, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = `${downloadName || 'snapshot'}.png`; a.click();
      URL.revokeObjectURL(url);
    }, 'image/png');
  };

  const downloadVideo = async () => {
    if (!src) return;
    try {
      const response = await fetch(src);
      if (!response.ok) return;
      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = objectUrl; a.download = `${downloadName || 'result'}.mp4`; a.click();
      URL.revokeObjectURL(objectUrl);
    } catch (e) { console.error(e); }
  };

  const fullScreen = () => {
    const v = videoRef.current;
    if (v?.requestFullscreen) v.requestFullscreen();
  };

  return (
    <section className="video-block">
      <div className="video-title">{title}</div>
      <video ref={videoRef} className="video-surface" src={src} controls />
      <div className="control-row">
        <button type="button" className="glass-btn" onClick={togglePlay}>播放/暂停</button>
        <button type="button" className="glass-btn" onClick={capture}>截图</button>
        <button type="button" className="glass-btn" onClick={rewind}>回看10秒</button>
        <button type="button" className="glass-btn" onClick={downloadVideo}>下载</button>
        <button type="button" className="glass-btn" onClick={fullScreen}>全屏</button>
      </div>
    </section>
  );
}

// ===================== EggCountResultPanel =====================
function EggCountResultPanel({ result, onClear }) {
  const [isPreviewOpen, setIsPreviewOpen] = React.useState(false);
  const [zoom, setZoom] = React.useState(100);
  const [minConf, setMinConf] = React.useState(80);
  const [maxConf, setMaxConf] = React.useState(100);
  
  const [filteredImgSrc, setFilteredImgSrc] = React.useState('');

  // Drag-to-pan state
  const [position, setPosition] = React.useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = React.useState(false);
  const dragStartRef = React.useRef({ x: 0, y: 0 });
  const initialPosRef = React.useRef({ x: 0, y: 0 });

  React.useEffect(() => {
    if (!result || !result.raw_image) return;
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);

      const filteredEggs = (result.eggs || []).filter(e => {
        const c = e.confidence * 100;
        return c >= minConf && c <= maxConf;
      });

      filteredEggs.forEach(det => {
        const [x1, y1, x2, y2] = det.bbox;
        ctx.strokeStyle = '#ff8c00'; // 橙色
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        
        ctx.fillStyle = '#ff8c00';
        ctx.font = '14px sans-serif';
        ctx.fillText(`egg ${(det.confidence).toFixed(2)}`, x1, Math.max(14, y1 - 6));
      });

      // 左上角汇总文字
      ctx.font = 'bold 24px sans-serif';
      ctx.fillStyle = '#ff8c00';
      ctx.fillText(`Total Eggs: ${result.egg_count}`, 10, 40);
      ctx.fillStyle = '#26a65b';
      ctx.fillText(`Eggs >= 90%: ${result.egg_count_90}`, 10, 70);
      ctx.fillStyle = '#1677ff';
      ctx.fillText(`Eggs >= 95%: ${result.egg_count_95}`, 10, 100);
      ctx.fillStyle = '#e84393';
      ctx.fillText(`Custom (${minConf}-${maxConf}%): ${filteredEggs.length}`, 10, 130);

      setFilteredImgSrc(canvas.toDataURL('image/png'));
    };
    img.src = `data:image/png;base64,${result.raw_image}`;
  }, [result, minConf, maxConf]);

  // Event handlers for dragging
  const handlePointerDown = (e) => {
    e.preventDefault();
    setIsDragging(true);
    dragStartRef.current = { x: e.clientX, y: e.clientY };
    initialPosRef.current = { ...position };
  };
  const handlePointerMove = (e) => {
    if (!isDragging) return;
    e.preventDefault();
    setPosition({
      x: initialPosRef.current.x + (e.clientX - dragStartRef.current.x),
      y: initialPosRef.current.y + (e.clientY - dragStartRef.current.y)
    });
  };
  const handlePointerUp = () => setIsDragging(false);

  // Reset zoom and position when closing
  const closePreview = () => {
    setIsPreviewOpen(false);
    setZoom(100);
    setPosition({ x: 0, y: 0 });
  };

  if (!result) return null;
  const { egg_count, egg_count_90, egg_count_95, eggs = [] } = result;

  const customCount = eggs.filter(e => {
    const c = e.confidence * 100;
    return c >= minConf && c <= maxConf;
  }).length;

  return (
    <div className="egg-result-panel glass-card">
      <div className="egg-result-header">
        <span>虫卵计数结果</span>
        <button type="button" className="glass-btn mini" onClick={onClear}>清除</button>
      </div>

      <div className="egg-count-cards">
        <div className="egg-count-card card-egg">
          <div className="egg-count-num">{egg_count}</div>
          <div className="egg-count-label">总虫卵数</div>
        </div>
        <div className="egg-count-card card-egg-90" style={{ borderColor: 'rgba(38, 166, 91, 0.28)', background: 'rgba(38, 166, 91, 0.06)' }}>
          <div className="egg-count-num" style={{ background: 'linear-gradient(135deg, #26a65b, #1a7a41)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text' }}>{egg_count_90}</div>
          <div className="egg-count-label">置信度 ≥ 90%</div>
        </div>
        <div className="egg-count-card card-egg-95" style={{ borderColor: 'rgba(22, 119, 255, 0.28)', background: 'rgba(22, 119, 255, 0.06)' }}>
          <div className="egg-count-num" style={{ background: 'linear-gradient(135deg, #1677ff, #0f50cc)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text' }}>{egg_count_95}</div>
          <div className="egg-count-label">置信度 ≥ 95%</div>
        </div>
      </div>

      <div style={{ padding: '12px', background: 'rgba(255,255,255,0.6)', borderRadius: '12px', marginBottom: '16px', border: '1px solid rgba(22,119,255,0.15)' }}>
        <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text)', marginBottom: '8px' }}>自定义置信度区间查询：</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px' }}>
          <input type="number" value={minConf} onChange={e => setMinConf(Number(e.target.value))} style={{ width: '60px', padding: '4px 8px', borderRadius: '6px', border: '1px solid #ccc' }} /> %
          <span>至</span>
          <input type="number" value={maxConf} onChange={e => setMaxConf(Number(e.target.value))} style={{ width: '60px', padding: '4px 8px', borderRadius: '6px', border: '1px solid #ccc' }} /> %
          <span style={{ marginLeft: 'auto', fontWeight: 700, color: 'var(--primary)' }}>区间包含虫卵数: {customCount} 个</span>
        </div>
        <div style={{ fontSize: '11px', color: 'var(--muted)', marginTop: '6px' }}>* 修改区间后，下方的分析图会自动滤除区间外的虫卵识别框。</div>
      </div>

      {filteredImgSrc && (
        <>
          <div 
            className="egg-annotated-wrap" 
            onClick={() => setIsPreviewOpen(true)}
            style={{ cursor: 'zoom-in', position: 'relative' }}
          >
            <img
              src={filteredImgSrc}
              alt="虫卵检测标注图"
              className="egg-annotated-img"
            />
            <div style={{ textAlign: 'center', padding: '10px', fontSize: '13px', color: 'var(--primary)', fontWeight: 600, background: 'rgba(22, 119, 255, 0.05)' }}>
              🔍 点击放大查看无损原图 (支持鼠标拖动平移)
            </div>
          </div>

          {isPreviewOpen && ReactDOM.createPortal(
            <div 
              style={{
                position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
                backgroundColor: 'rgba(0, 0, 0, 0.9)', zIndex: 99999,
                display: 'flex', flexDirection: 'column'
              }}
            >
              <div style={{ padding: '16px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(10px)', zIndex: 2 }}>
                <div style={{ color: 'white', display: 'flex', alignItems: 'center', gap: '20px' }}>
                  <span style={{ fontSize: '14px', fontWeight: 600 }}>无损放大: {zoom}%</span>
                  <input type="range" min="10" max="500" value={zoom} onChange={(e) => setZoom(e.target.value)} style={{ width: '200px' }} />
                  <button onClick={() => { setZoom(100); setPosition({x:0, y:0}); }} className="glass-btn mini" style={{ color: 'white', borderColor: 'rgba(255,255,255,0.3)', background: 'transparent' }}>重置</button>
                </div>
                <button onClick={closePreview} className="glass-btn mini" style={{ color: 'white', borderColor: 'rgba(255,255,255,0.3)', background: 'transparent', fontSize: '14px' }}>✖ 关闭</button>
              </div>
              <div 
                style={{ 
                  flex: 1, overflow: 'hidden', display: 'flex', justifyContent: 'center', alignItems: 'center', 
                  cursor: isDragging ? 'grabbing' : 'grab' 
                }}
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
                onPointerCancel={handlePointerUp}
                onPointerLeave={handlePointerUp}
              >
                <img 
                  src={filteredImgSrc} 
                  alt="虫卵检测标注图放大" 
                  draggable={false}
                  style={{ 
                    transform: `translate(${position.x}px, ${position.y}px) scale(${zoom/100})`,
                    transformOrigin: 'center center',
                    transition: isDragging ? 'none' : 'transform 0.1s ease-out',
                    boxShadow: '0 12px 48px rgba(0,0,0,0.5)',
                    borderRadius: '4px',
                    maxWidth: '95vw', maxHeight: '95vh', objectFit: 'contain'
                  }} 
                />
              </div>
            </div>,
            document.body
          )}
        </>
      )}
    </div>
  );
}

// ===================== App =====================
function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [selectedFileNames, setSelectedFileNames] = useState('');
  const [taskId, setTaskId] = useState(null);
  const [status, setStatus] = useState('idle');
  const [socketStatus, setSocketStatus] = useState('disconnected');
  const [currentFrame, setCurrentFrame] = useState(null);
  const [currentStats, setCurrentStats] = useState(null);
  const [totalFrames, setTotalFrames] = useState(0);
  const [videoUrl, setVideoUrl] = useState('');
  const [logUrl, setLogUrl] = useState('');
  const [subVideosZipUrl, setSubVideosZipUrl] = useState('');
  const [summaryUrl, setSummaryUrl] = useState('');
  const [csvUrl, setCsvUrl] = useState('');
  const [subVideos, setSubVideos] = useState([]);
  const [selectedSubVideo, setSelectedSubVideo] = useState('');
  const [historyList, setHistoryList] = useState([]);
  const [idPoseSummary, setIdPoseSummary] = useState([]);
  const [error, setError] = useState('');
  const [wormCount, setWormCount] = useState('');

  // Analysis Mode: 'tracking' | 'egg_count'
  const [analysisMode, setAnalysisMode] = useState('tracking');
  // Egg Count Result
  const [eggCountResult, setEggCountResult] = useState(null);
  const [eggCounting, setEggCounting] = useState(false);

  // Model Selection
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [modelsLoading, setModelsLoading] = useState(false);

  const EGG_MODEL_NAME = 'EggCount.pt';

  // Preview
  const [previewing, setPreviewing] = useState(false);
  const [suggestedCount, setSuggestedCount] = useState(null);
  const [previewAnnotated, setPreviewAnnotated] = useState(null);
  const previewAbortRef = useRef(false);

  // Head marking
  const [firstFrameData, setFirstFrameData] = useState(null);   // { frame_width, frame_height, annotated, worms }
  const [firstFrameBlob, setFirstFrameBlob] = useState(null);
  const [showHeadMarkModal, setShowHeadMarkModal] = useState(false);
  const [wormSelections, setWormSelections] = useState({});      // { det_idx: { enabled, head_kpt_idx, user_label } }
  // Pending file for after head marking
  const pendingFileRef = useRef(null);

  const socketRef = useRef(null);
  const taskRef = useRef(null);

  useEffect(() => { taskRef.current = taskId; }, [taskId]);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        setModelsLoading(true);
        const res = await axios.get('/models');
        if (res.data && Array.isArray(res.data.models)) {
          setAvailableModels(res.data.models);
          if (res.data.current_model) {
            setSelectedModel(res.data.current_model);
          } else if (res.data.models.length > 0) {
            setSelectedModel(res.data.models[0]);
          }
        }
      } catch (e) {
        console.warn('获取模型列表失败:', e);
      } finally {
        setModelsLoading(false);
      }
    };
    fetchModels();
  }, []);

  const handleModelChange = async (e) => {
    const modelName = e.target.value;
    setSelectedModel(modelName);
    try {
      await axios.post('/select_model', { model_name: modelName });
      // 如果当前已有选择的文件，且未在处理中，重新触发第一帧预览分析
      if (selectedFiles.length > 0 && status === 'idle' && !previewing) {
        handleFileChange({ target: { files: selectedFiles } });
      }
    } catch (err) {
      console.error('切换模型失败:', err);
      setError('切换模型失败，请稍后重试');
    }
  };

  const handleAnalysisModeSwitch = async (mode) => {
    setAnalysisMode(mode);
    setError('');
    setEggCountResult(null);
    if (mode === 'egg_count') {
      // 自动切换到 EggCount.pt（若可用）
      if (availableModels.includes(EGG_MODEL_NAME) && selectedModel !== EGG_MODEL_NAME) {
        setSelectedModel(EGG_MODEL_NAME);
        try {
          await axios.post('/select_model', { model_name: EGG_MODEL_NAME });
        } catch (err) {
          console.warn('自动切换 EggCount.pt 失败:', err);
        }
      }
    }
  };

  const handleEggCount = async () => {
    if (selectedFiles.length === 0) { setError('请先选择视频或图片'); return; }
    setError('');
    setEggCounting(true);
    setEggCountResult(null);
    try {
      const formData = new FormData();
      formData.append('file', selectedFiles[0], selectedFiles[0].name);
      const res = await axios.post('/egg_count', formData);
      setEggCountResult(res.data);
    } catch (e) {
      const msg = e?.response?.data?.error || e.message || '虫卵计数失败';
      setError(msg);
    } finally {
      setEggCounting(false);
    }
  };

  useEffect(() => {
    const savedTaskId = sessionStorage.getItem(ACTIVE_TASK_KEY);
    const savedFileName = sessionStorage.getItem(ACTIVE_FILE_KEY);
    const savedHistory = localStorage.getItem(HISTORY_KEY);
    if (savedHistory) {
      try {
        const parsed = JSON.parse(savedHistory);
        if (Array.isArray(parsed)) setHistoryList(parsed);
      } catch (e) { console.error(e); }
    }
    if (savedFileName) setSelectedFileNames(savedFileName);
    if (savedTaskId) { setTaskId(savedTaskId); setStatus('processing'); }
  }, []);

  useEffect(() => {
    const socket = io('http://localhost:5000');
    socketRef.current = socket;

    socket.on('connect', () => {
      setSocketStatus('connected');
      if (taskRef.current) socket.emit('join', { task_id: String(taskRef.current) });
      const savedHistory = localStorage.getItem(HISTORY_KEY);
      if (savedHistory) {
        try {
          JSON.parse(savedHistory).forEach(item => {
            if (item.status === 'processing') socket.emit('join', { task_id: String(item.id) });
          });
        } catch (e) {}
      }
    });

    socket.on('disconnect', () => setSocketStatus('disconnected'));
    socket.on('connect_error', () => setSocketStatus('error'));

    socket.on('frame', (data) => {
      if (String(data.task_id) !== String(taskRef.current)) return;
      setCurrentFrame(`data:image/jpeg;base64,${data.image}`);
      setCurrentStats(data.stats);
      setTotalFrames(data.total_frames || 0);
    });

    socket.on('complete', async () => {});

    socket.on('cancelled', (payload) => {
      updateHistoryStatus(payload.task_id, 'cancelled');
      if (String(payload.task_id) !== String(taskRef.current)) return;
      setStatus('cancelled');
      setError(payload.message || '任务已取消');
      sessionStorage.removeItem(ACTIVE_TASK_KEY);
      sessionStorage.removeItem(ACTIVE_FILE_KEY);
    });

    socket.on('error', (payload) => {
      updateHistoryStatus(payload.task_id, 'failed');
      if (String(payload.task_id) !== String(taskRef.current)) return;
      setStatus('idle');
      setError(payload.message || '处理失败');
      sessionStorage.removeItem(ACTIVE_TASK_KEY);
    });

    return () => { socket.disconnect(); };
  }, []);

  useEffect(() => {
    if (taskId && socketRef.current?.connected) {
      socketRef.current.emit('join', { task_id: String(taskId) });
    }
  }, [taskId]);

  useEffect(() => {
    if (!taskId) return;
    let cancelled = false;
    let pollingTimer = null;

    const stopPolling = () => {
      cancelled = true;
      if (pollingTimer) { window.clearInterval(pollingTimer); pollingTimer = null; }
    };

    const fetchTaskStatus = async () => {
      try {
        const res = await axios.get(`/task/${taskId}/status`);
        if (cancelled) return;
        const data = res.data || {};

        if (data.status === 'processing') {
          setStatus('processing');
          upsertHistoryRecord({
            id: taskId,
            name: data.filename || `任务${taskId}`,
            status: 'processing',
            video: toAbsoluteUrl(data.video_url || ''),
            log: toAbsoluteUrl(data.log_url || ''),
            zip: toAbsoluteUrl(data.sub_videos_url || ''),
            summary: toAbsoluteUrl(data.summary_url || ''),
            csv: toAbsoluteUrl(data.csv_url || ''),
          });
        } else if (data.status === 'completed') {
          stopPolling();
          setStatus('completed');
          sessionStorage.removeItem(ACTIVE_TASK_KEY);
          sessionStorage.removeItem(ACTIVE_FILE_KEY);
          upsertHistoryRecord({
            id: taskId,
            name: data.filename || `任务${taskId}`,
            status: 'completed',
            video: toAbsoluteUrl(data.video_url || ''),
            log: toAbsoluteUrl(data.log_url || ''),
            zip: toAbsoluteUrl(data.sub_videos_url || ''),
            summary: toAbsoluteUrl(data.summary_url || ''),
            csv: toAbsoluteUrl(data.csv_url || ''),
          });
          try {
            const subRes = await axios.get(`/task/${taskId}/sub_videos`);
            const videos = (subRes.data.videos || []).map(item => ({ ...item, url: toAbsoluteUrl(item.url) }));
            setSubVideos(videos);
            if (!selectedSubVideo && videos.length > 0) setSelectedSubVideo(videos[0].url);
          } catch (e) { console.error(e); }
        } else if (data.status === 'paused') {
          setStatus('paused');
          updateHistoryStatus(taskId, 'paused');
        } else if (data.status === 'cancelled') {
          stopPolling();
          setStatus('cancelled');
          sessionStorage.removeItem(ACTIVE_TASK_KEY);
          sessionStorage.removeItem(ACTIVE_FILE_KEY);
          updateHistoryStatus(taskId, 'cancelled');
        } else if (data.status === 'failed') {
          stopPolling();
          setStatus('idle');
          if (data.error_message) setError(data.error_message);
          updateHistoryStatus(taskId, 'failed');
        }

        if (data.current_frame || data.total_frames) {
          setCurrentStats(prev => ({
            frame: Math.max(data.current_frame || 0, prev?.frame || 0),
            active_tracks: data.active_tracks || 0,
            tracks: data.tracks || prev?.tracks || [],
          }));
          setTotalFrames(tf => Math.max(data.total_frames || 0, tf));
        }

        const socketConnected = socketRef.current?.connected;
        if (!socketConnected && data.latest_frame) {
          setCurrentFrame(`data:image/jpeg;base64,${data.latest_frame}`);
        }

        if (data.video_url) setVideoUrl(toAbsoluteUrl(data.video_url));
        if (data.log_url) setLogUrl(toAbsoluteUrl(data.log_url));
        if (data.sub_videos_url) setSubVideosZipUrl(toAbsoluteUrl(data.sub_videos_url));
        if (data.summary_url) setSummaryUrl(toAbsoluteUrl(data.summary_url));
        if (data.csv_url) setCsvUrl(toAbsoluteUrl(data.csv_url));
        if (Array.isArray(data.id_pose_summary)) setIdPoseSummary(data.id_pose_summary);
        if (data.worm_count !== undefined && data.worm_count !== null) setWormCount(String(data.worm_count));
      } catch (e) {
        if (!cancelled) console.error(e);
      }
    };

    fetchTaskStatus();
    pollingTimer = window.setInterval(fetchTaskStatus, 1500);
    return () => { stopPolling(); };
  }, [taskId]);

  const progress = useMemo(() => formatPercent(currentStats?.frame, totalFrames), [currentStats, totalFrames]);

  const resetCurrentOutput = () => {
    setCurrentFrame(null); setCurrentStats(null); setTotalFrames(0);
    setVideoUrl(''); setLogUrl(''); setSubVideosZipUrl(''); setSummaryUrl(''); setCsvUrl('');
    setSubVideos([]); setSelectedSubVideo(''); setIdPoseSummary([]); setError('');
    setWormCount(''); setSuggestedCount(null); setPreviewAnnotated(null);
    setFirstFrameData(null); setShowHeadMarkModal(false); setWormSelections({});
  };

  const upsertHistoryRecord = (record) => {
    setHistoryList((prev) => {
      const id = String(record.id);
      const existing = prev.find(item => String(item.id) === id) || {};
      const payload = {
        id, name: record.name ?? existing.name ?? `任务${id}`,
        status: record.status ?? existing.status ?? 'processing',
        time: record.time ?? new Date().toLocaleString(),
        video: record.video ?? existing.video ?? '',
        log: record.log ?? existing.log ?? '',
        zip: record.zip ?? existing.zip ?? '',
        summary: record.summary ?? existing.summary ?? '',
      };
      const next = [payload, ...prev.filter(item => String(item.id) !== id)].slice(0, 50);
      localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
      return next;
    });
  };

  const openResultUrl = (url) => { if (url) window.open(toAbsoluteUrl(url), '_blank'); };

  const resolveHistoryResourceUrl = async (item, resourceKey) => {
    if (item?.[resourceKey]) return toAbsoluteUrl(item[resourceKey]);
    if (!item?.id) return '';
    try {
      const res = await axios.get(`/task/${item.id}/status`);
      const data = res.data || {};
      const recovered = {
        id: String(item.id), name: item.name, status: data.status || item.status,
        video: toAbsoluteUrl(data.video_url || item.video || ''),
        log: toAbsoluteUrl(data.log_url || item.log || ''),
        zip: toAbsoluteUrl(data.sub_videos_url || item.zip || ''),
        summary: toAbsoluteUrl(data.summary_url || item.summary || ''),
        csv: toAbsoluteUrl(data.csv_url || item.csv || ''),
      };
      upsertHistoryRecord(recovered);
      return recovered[resourceKey] || '';
    } catch (e) { console.error(e); return ''; }
  };

  const handleHistoryOpen = async (item, resourceKey) => {
    const url = await resolveHistoryResourceUrl(item, resourceKey);
    if (!url) { setError('该历史任务暂无可访问结果，请重新处理该视频后再试。'); return; }
    window.open(url, '_blank');
  };

  const extractFirstFrame = (videoFile) => new Promise((resolve, reject) => {
    const video = document.createElement('video');
    const url = URL.createObjectURL(videoFile);
    video.src = url; video.muted = true; video.preload = 'metadata';
    video.onloadedmetadata = () => { video.currentTime = 0.001; };
    video.onseeked = () => {
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => {
        URL.revokeObjectURL(url);
        blob ? resolve(blob) : reject(new Error('Canvas toBlob failed'));
      }, 'image/jpeg', 0.85);
    };
    video.onerror = (e) => { URL.revokeObjectURL(url); reject(e); };
  });

  const handleFileChange = async (event) => {
    const files = Array.from(event.target.files || []);
    if (files.length === 0) return;
    setSelectedFiles(files);
    setSelectedFileNames(files.map(f => f.name).join(', '));
    sessionStorage.setItem(ACTIVE_FILE_KEY, files[0].name);
    setStatus('idle');
    resetCurrentOutput();

    const file = files[0];
    if (!file.type.startsWith('video/') && !file.type.startsWith('image/')) return;
    
    previewAbortRef.current = false;
    setPreviewing(true);
    setSuggestedCount(null);
    setPreviewAnnotated(null);
    setFirstFrameData(null);
    setShowHeadMarkModal(false);

    try {
      let blob;
      if (file.type.startsWith('image/')) {
        blob = file;
      } else {
        blob = await extractFirstFrame(file);
      }
      if (previewAbortRef.current) return;

      // Call preview_count AND analyze_first_frame in parallel
      const fd1 = new FormData(); fd1.append('frame', blob, 'preview.jpg');
      const fd2 = new FormData(); fd2.append('frame', blob, 'preview.jpg');

      const [res1, res2] = await Promise.all([
        axios.post('/preview_count', fd1),
        axios.post('/analyze_first_frame', fd2).catch(e => { console.warn('analyze_first_frame failed:', e); return null; }),
      ]);
      if (previewAbortRef.current) return;

      const { count, annotated } = res1.data;
      setSuggestedCount(count);
      if (annotated) setPreviewAnnotated(`data:image/jpeg;base64,${annotated}`);
      setWormCount(prev => (prev === '' ? String(count) : prev));

      if (res2?.data?.worms) {
        const fd = res2.data;
        // Initialize selections
        const initSels = {};
        fd.worms.forEach((w, idx) => {
          initSels[w.det_idx] = {
            enabled: true,
            head_kpt_idx: null,
            user_label: `虫子${idx + 1}`,
          };
        });
        setFirstFrameData(fd);
        setFirstFrameBlob(blob);
        setWormSelections(initSels);
        pendingFileRef.current = file;
        
        // 仅在追踪模式下自动跳出弹窗
        if (analysisMode === 'tracking') {
          setShowHeadMarkModal(true);
        }
      }
    } catch (e) {
      console.warn('Preview failed:', e);
    } finally {
      if (!previewAbortRef.current) setPreviewing(false);
    }
  };

  const cancelPreview = () => {
    previewAbortRef.current = true;
    setPreviewing(false);
    setSuggestedCount(null);
    setPreviewAnnotated(null);
    setShowHeadMarkModal(false);
    setFirstFrameData(null);
    setFirstFrameBlob(null);
  };

  // Actually submit task — called after head marking confirmed or skipped
  const submitTask = async (files, headConfig, overrideCount = null) => {
    const rawCount = overrideCount !== null ? overrideCount : wormCount;
    const count = parseInt(rawCount, 10);
    if (isNaN(count) || count <= 0) { setError('请输入有效的初始虫子数量（正整数）'); return; }

    setError('');
    setStatus('uploading');
    let firstTaskId = null;

    for (const file of files) {
      const formData = new FormData();
      formData.append('video', file);
      formData.append('worm_count', count);
      if (selectedModel) {
        formData.append('model_name', selectedModel);
      }
      if (headConfig && headConfig.length > 0) {
        formData.append('head_config', JSON.stringify(headConfig));
      }
      try {
        const res = await axios.post('/upload', formData);
        const newTaskId = String(res.data.task_id);
        if (!firstTaskId) firstTaskId = newTaskId;
        upsertHistoryRecord({
          id: newTaskId,
          name: file.name || `任务${newTaskId}`,
          status: 'processing',
          time: new Date().toLocaleString(),
        });
        socketRef.current?.emit('join', { task_id: newTaskId });
      } catch (e) {
        console.error(e);
        setError('部分上传失败');
      }
    }

    if (firstTaskId) {
      setTaskId(firstTaskId);
      setStatus('processing');
      sessionStorage.setItem(ACTIVE_TASK_KEY, firstTaskId);
    } else {
      setStatus('idle');
    }
    setSelectedFiles([]);
    setSelectedFileNames('');
  };

  const startProcess = async () => {
    if (selectedFiles.length === 0) { setError('请先选择视频'); return; }
    // If head mark modal is available, open it; otherwise submit directly
    if (firstFrameData && firstFrameData.worms.length > 0) {
      setShowHeadMarkModal(true);
    } else {
      await submitTask(selectedFiles, []);
    }
  };

  const handleHeadMarkConfirm = async () => {
    setShowHeadMarkModal(false);
    // Build head_config from wormSelections
    const headConfig = (firstFrameData?.worms || [])
      .filter(w => {
        const sel = wormSelections[w.det_idx];
        return sel?.enabled && sel.head_kpt_idx !== null && sel.head_kpt_idx !== undefined;
      })
      .map(w => {
        const sel = wormSelections[w.det_idx];
        return {
          det_idx: w.det_idx,
          head_kpt_idx: sel.head_kpt_idx,
          user_label: sel.user_label || `虫子${w.det_idx + 1}`,
          centroid_x: w.centroid[0],
          centroid_y: w.centroid[1],
          bbox: w.bbox,
          keypoints: w.keypoints,
        };
      });
    const userCount = parseInt(wormCount, 10) || 0;
    const enabledCount = (firstFrameData?.worms || []).filter(w => wormSelections[w.det_idx]?.enabled).length;
    // 若用户输入了虫子数量，以用户输入为准（且不小于已勾选标记的数量）
    const finalCount = userCount > 0 ? Math.max(userCount, enabledCount) : enabledCount;
    if (finalCount > 0) setWormCount(String(finalCount));
    await submitTask(selectedFiles, headConfig, finalCount);
  };

  const handleHeadMarkSkip = async () => {
    setShowHeadMarkModal(false);
    const userCount = parseInt(wormCount, 10) || 0;
    const detectedCount = (firstFrameData?.worms || []).length;
    // 用户手动输入了虫子数量时，严格使用用户输入的值，绝不被 YOLO 检测到的数量覆盖
    const finalCount = userCount > 0 ? userCount : detectedCount;
    if (finalCount > 0) setWormCount(String(finalCount));
    await submitTask(selectedFiles, [], finalCount);
  };

  const updateHistoryStatus = (targetTaskId, nextStatus) => {
    upsertHistoryRecord({ id: targetTaskId, status: nextStatus, time: new Date().toLocaleString() });
  };

  const handleTaskControl = async (action) => {
    if (!taskId) return;
    try {
      const res = await axios.post(`/task/${taskId}/${action}`);
      const nextStatus = res.data?.status;
      if (nextStatus === 'paused') setStatus('paused');
      if (nextStatus === 'processing') setStatus('processing');
      if (nextStatus === 'cancelled') {
        setStatus('cancelled');
        sessionStorage.removeItem(ACTIVE_TASK_KEY);
        sessionStorage.removeItem(ACTIVE_FILE_KEY);
      }
      updateHistoryStatus(taskId, nextStatus || action);
    } catch (e) {
      console.error(e);
      setError('任务控制失败，请稍后重试');
    }
  };

  return (
    <div className="app-shell">
      <div className="bg-grid" />
      <div className="bg-orb orb-a" />
      <div className="bg-orb orb-b" />

      {/* Head Mark Modal */}
      {showHeadMarkModal && firstFrameData && (
        <HeadMarkModal
          firstFrameData={firstFrameData}
          firstFrameBlob={firstFrameBlob}
          wormSelections={wormSelections}
          onSelectionsChange={setWormSelections}
          onAddManualWorm={(newWorm) => {
            setFirstFrameData(prev => ({
              ...prev,
              worms: [...(prev?.worms || []), newWorm],
            }));
            setWormSelections(prev => ({
              ...prev,
              [newWorm.det_idx]: {
                enabled: true,
                head_kpt_idx: 0,
                user_label: `虫子${newWorm.det_idx + 1}(手动)`,
              },
            }));
          }}
          onConfirm={handleHeadMarkConfirm}
          onSkip={handleHeadMarkSkip}
        />
      )}

      <header className="topbar glass-card">
        <h1>线虫运动识别</h1>
        <p>上传视频，实时查看模型处理结果与线虫子视频。</p>
        <div className={`socket-pill socket-${socketStatus}`}>
          Socket: {socketStatus}
        </div>
      </header>

      <main className="layout">
        <aside className="panel glass-card left-panel">
          <h2>输入区</h2>

          {/* ── 模式切换 Tab ── */}
          <div className="mode-tab-row">
            <button
              type="button"
              id="mode-tab-tracking"
              className={`mode-tab ${analysisMode === 'tracking' ? 'mode-tab-active' : ''}`}
              onClick={() => handleAnalysisModeSwitch('tracking')}
              disabled={status === 'uploading' || status === 'processing'}
            >
              运动追踪
            </button>
            <button
              type="button"
              id="mode-tab-egg"
              className={`mode-tab ${analysisMode === 'egg_count' ? 'mode-tab-active mode-tab-egg' : ''}`}
              onClick={() => handleAnalysisModeSwitch('egg_count')}
              disabled={status === 'uploading' || status === 'processing'}
            >
              虫卵计数
            </button>
          </div>

          <label className="upload-box" htmlFor="file-input">
            <input
              id="file-input"
              type="file"
              accept="video/*,image/*"
              multiple={analysisMode === 'tracking'}
              onChange={handleFileChange}
              disabled={status === 'uploading' || status === 'processing'}
            />
            <span>{analysisMode === 'egg_count' ? '选择视频或图片' : '选择视频'}</span>
          </label>

          <div className="file-meta">
            <div>文件: {selectedFileNames || '未选择'}</div>
            <div>文件数: {selectedFiles.length}</div>
            <div>状态: {status}</div>
            {analysisMode === 'tracking' && <div>任务ID: {taskId || '无'}</div>}
          </div>

          <div className="model-select-wrap">
            <label
              htmlFor="model-select"
              style={{ fontSize: '13px', color: 'var(--muted)', fontWeight: 500, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
            >
              <span>选择识别模型</span>
              {modelsLoading && (
                <span style={{ fontSize: '12px', color: 'var(--primary)', display: 'flex', alignItems: 'center', gap: '5px' }}>
                  <span className="preview-spinner" /> 加载中...
                </span>
              )}
              {analysisMode === 'egg_count' && (
                <span style={{ fontSize: '11px', color: '#26a65b', fontWeight: 600 }}>已锁定 EggCount.pt</span>
              )}
            </label>
            <select
              id="model-select"
              className="glass-select model-select-input"
              value={selectedModel}
              onChange={handleModelChange}
              disabled={analysisMode === 'egg_count' || previewing || status === 'uploading' || status === 'processing' || availableModels.length === 0}
              style={{ width: '100%', marginTop: '6px', padding: '9px 12px', borderRadius: '12px', border: `1px solid ${analysisMode === 'egg_count' ? 'rgba(38,166,91,0.3)' : 'rgba(22, 119, 255, 0.18)'}`, background: 'rgba(255, 255, 255, 0.75)', color: '#0f2f57', outline: 'none', fontWeight: 500 }}
            >
              {availableModels.length === 0 ? (
                <option value="">{modelsLoading ? '加载模型列表中...' : '未找到可用模型'}</option>
              ) : (
                availableModels.map((modelName) => (
                  <option key={modelName} value={modelName}>
                    {modelName}
                  </option>
                ))
              )}
            </select>
          </div>

          {/* ── 追踪模式专属字段 ── */}
          {analysisMode === 'tracking' && (
            <div className="worm-count-wrap">
              <label
                htmlFor="worm-count-input"
                style={{ fontSize: '13px', color: 'var(--muted)', fontWeight: 500, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
              >
                <span>初始虫子数量 <span style={{ color: '#d64040' }}>*</span></span>
                {previewing && (
                  <span style={{ fontSize: '12px', color: 'var(--primary)', display: 'flex', alignItems: 'center', gap: '5px' }}>
                    <span className="preview-spinner" /> 分析第一帧...
                  </span>
                )}
                {!previewing && suggestedCount !== null && (
                  <span style={{ fontSize: '12px', color: '#26a65b' }}>模型建议：{suggestedCount} 条</span>
                )}
              </label>
              <input
                id="worm-count-input"
                type="number"
                min="1"
                step="1"
                placeholder={previewing ? '正在分析...' : suggestedCount !== null ? `建议值: ${suggestedCount}` : '请输入正整数'}
                className="glass-input worm-count-input"
                value={wormCount}
                onChange={(e) => setWormCount(e.target.value)}
                disabled={previewing || status === 'uploading' || status === 'processing'}
                style={{ width: '100%', marginTop: '6px', padding: '10px 14px', borderRadius: '12px', border: `1px solid ${previewing ? 'rgba(22, 119, 255, 0.38)' : 'rgba(22, 119, 255, 0.18)'}`, background: 'rgba(255, 255, 255, 0.6)', color: '#0f2f57', outline: 'none' }}
              />
              {!previewing && previewAnnotated && (
                <div style={{ marginTop: '8px', borderRadius: '10px', overflow: 'hidden', border: '1px solid rgba(22,119,255,0.18)' }}>
                  <img src={previewAnnotated} alt="第一帧检测预览" style={{ width: '100%', display: 'block' }} />
                </div>
              )}
            </div>
          )}

          {/* ── 虫卵计数模式说明 ── */}
          {analysisMode === 'egg_count' && (
            <div className="egg-mode-hint">
              <div><strong>虫卵计数模式</strong></div>
              <div>仅分析第一帧图像，检测并统计线虫与虫卵数量，不进行后续分析。</div>
            </div>
          )}

          {/* ── 启动按钮 ── */}
          {analysisMode === 'tracking' ? (
            <button
              type="button"
              className="glass-btn launch-btn"
              onClick={startProcess}
              disabled={previewing || selectedFiles.length === 0 || status === 'uploading' || !wormCount || parseInt(wormCount, 10) <= 0}
            >
              {previewing ? '等待预览完成...' : status === 'uploading' ? '上传中...' : status === 'processing' ? '识别中...' : '启动识别'}
            </button>
          ) : (
            <button
              type="button"
              id="btn-egg-count"
              className="glass-btn launch-btn egg-launch-btn"
              onClick={handleEggCount}
              disabled={eggCounting || selectedFiles.length === 0}
            >
              {eggCounting ? (
                <><span className="preview-spinner" style={{ marginRight: 6 }} />计数中...</>
              ) : '开始计数'}
            </button>
          )}

          {/* ── 追踪模式控制按钮 ── */}
          {analysisMode === 'tracking' && (
            <div className="task-control-row">
              <button type="button" className="glass-btn" onClick={() => handleTaskControl('pause')} disabled={previewing || !taskId || status !== 'processing'}>暂停处理</button>
              <button type="button" className="glass-btn" onClick={() => handleTaskControl('resume')} disabled={previewing || !taskId || status !== 'paused'}>继续处理</button>
              <button
                type="button" className="glass-btn"
                onClick={() => { if (previewing) { cancelPreview(); } else { handleTaskControl('cancel'); } }}
                disabled={!previewing && (!taskId || ['completed', 'cancelled', 'idle'].includes(status))}
              >{previewing ? '取消预览' : '取消任务'}</button>
            </div>
          )}

          {error && <div className="error-text">{error}</div>}

          {/* ── 追踪模式下载按钮 ── */}
          {analysisMode === 'tracking' && (
            <div className="download-row">
              <button type="button" className="glass-btn" onClick={() => openResultUrl(videoUrl)} disabled={!videoUrl}>下载结果视频</button>
              <button type="button" className="glass-btn" onClick={() => openResultUrl(logUrl)} disabled={!logUrl}>下载日志</button>
              <button type="button" className="glass-btn" onClick={() => openResultUrl(subVideosZipUrl)} disabled={!subVideosZipUrl}>下载子视频ZIP</button>
              <button type="button" className="glass-btn" onClick={() => openResultUrl(summaryUrl)} disabled={!summaryUrl}>下载ID姿态JSON</button>
              <button type="button" className="glass-btn" onClick={() => openResultUrl(csvUrl)} disabled={!csvUrl}>下载摆动汇总CSV</button>
            </div>
          )}
        </aside>

        <section className="panel glass-card center-panel">
          <h2>{analysisMode === 'egg_count' ? '虫卵计数结果' : '实时处理区'}</h2>

          {/* ── 虫卵计数模式：显示计数结果 ── */}
          {analysisMode === 'egg_count' && (
            <>
              {eggCounting && (
                <div className="egg-counting-placeholder">
                  <span className="preview-spinner" style={{ width: 22, height: 22, borderWidth: 3 }} />
                  <span style={{ marginTop: 10, color: 'var(--muted)' }}>正在分析第一帧，请稍候...</span>
                </div>
              )}
              {!eggCounting && !eggCountResult && (
                <div className="stream-wrap" style={{ minHeight: 200 }}>
                  <div className="stream-placeholder">选择文件后点击「开始计数」</div>
                </div>
              )}
              <EggCountResultPanel
                result={eggCountResult}
                onClear={() => setEggCountResult(null)}
              />
            </>
          )}

          {/* ── 追踪模式：实时帧流 ── */}
          {analysisMode === 'tracking' && (<>
          <div className="stream-wrap">
            {currentFrame ? (
              <img src={currentFrame} alt="实时处理画面" className="stream-frame" />
            ) : (
              <div className="stream-placeholder">等待模型输出帧...</div>
            )}
          </div>

          <div className="progress-wrap">
            <div className="progress-text">进度 {currentStats?.frame || 0} / {totalFrames || 0}</div>
            <div className="progress-track">
              <div className="progress-value" style={{ width: `${progress}%` }} />
            </div>
          </div>

          <div className="stats-grid">
            <div className="stat-item"><span>活跃目标</span><strong>{currentStats?.active_tracks ?? 0}</strong></div>
            <div className="stat-item"><span>当前帧</span><strong>{currentStats?.frame ?? 0}</strong></div>
            <div className="stat-item"><span>总帧数</span><strong>{totalFrames || 0}</strong></div>
          </div>

          <section className="id-pose-section">
            <div className="sub-video-header"><h3>当前帧 ID 与姿态计数</h3></div>
            {(currentStats?.tracks || []).length === 0 ? (
              <div className="history-empty">当前帧暂无可显示的线虫计数</div>
            ) : (
              <div className="id-pose-table-wrap">
                <table className="id-pose-table">
                  <thead>
                    <tr>
                      <th>ID</th><th>标签</th><th>状态</th><th>C形</th><th>S形</th><th>双向</th>
                      <th>频率(Hz)</th><th>质心坐标</th><th>头部坐标</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(currentStats?.tracks || []).map((item) => {
                      let statusBadge = <span className="status-badge active">{item.motion_type || '活跃'}</span>;
                      if (item.is_ghost) {
                        statusBadge = <span className="status-badge ghost">残留(丢{item.lost_frames ?? 0}f)</span>;
                      } else if (item.is_static || item.motion_type === 'static') {
                        statusBadge = <span className="status-badge static">静止</span>;
                      }
                      const freqVal = Number(item.live_freq || item.avg_freq || 0).toFixed(2);
                      const cStr = item.centroid_bl ? `(${item.centroid_bl.x.toFixed(1)}, ${item.centroid_bl.y.toFixed(1)})` : '—';
                      const hStr = item.head_bl ? `(${item.head_bl.x.toFixed(1)}, ${item.head_bl.y.toFixed(1)})` : '—';

                      return (
                        <tr key={`live-${item.track_id}`}>
                          <td>{item.track_id}</td>
                          <td>{item.user_label || '—'}</td>
                          <td>{statusBadge}</td>
                          <td>{item.c_count ?? 0}</td>
                          <td>{item.s_count ?? 0}</td>
                          <td>{item.is_bidirectional ? '是' : '否'}</td>
                          <td>{freqVal}</td>
                          <td>{cStr}</td>
                          <td>{hStr}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          <section className="id-pose-section">
            <div className="sub-video-header"><h3>最终输出 ID 与姿态计数</h3></div>
            {idPoseSummary.length === 0 ? (
              <div className="history-empty">任务完成后会显示最终汇总</div>
            ) : (
              <div className="id-pose-table-wrap">
                <table className="id-pose-table">
                  <thead>
                    <tr>
                      <th>ID</th><th>标签</th><th>状态</th><th>C形</th><th>S形</th><th>双向</th>
                      <th>平均频率(Hz)</th><th>完整周期</th><th>总路径(px)</th><th>净位移(px)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {idPoseSummary.map((item) => (
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
                        <td>{item.c_count ?? 0}</td>
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

          {videoUrl && <GlassVideoPlayer title="处理后主视频" src={videoUrl} downloadName="main_result" />}

          {subVideos.length > 0 && (
            <section className="sub-video-section">
              <div className="sub-video-header">
                <h3>线虫子视频</h3>
                <select value={selectedSubVideo} onChange={(e) => setSelectedSubVideo(e.target.value)} className="glass-select">
                  {subVideos.map(item => <option key={item.filename} value={item.url}>{item.filename}</option>)}
                </select>
              </div>
              {selectedSubVideo && <GlassVideoPlayer title="当前线虫局部子视频" src={selectedSubVideo} downloadName="worm_sub" />}
            </section>
          )}
          </>)} {/* end analysisMode === 'tracking' */}
        </section>

        <aside className="panel glass-card history-panel">
          <h2>历史</h2>
          {historyList.length === 0 && <div className="history-empty">暂无历史记录</div>}
          <div className="history-list">
            {historyList.map((item) => (
              <article key={item.id + item.time} className="history-item">
                <div className="history-title">{item.name}</div>
                <div className="history-time">{item.time}</div>
                <div className="history-time">状态: {item.status || 'completed'}</div>
                <div className="history-actions">
                  <button type="button" className="glass-btn mini" onClick={() => { setTaskId(item.id); setStatus(item.status); resetCurrentOutput(); }}>监视</button>
                  <button type="button" className="glass-btn mini" onClick={() => handleHistoryOpen(item, 'video')}>查看</button>
                  <button type="button" className="glass-btn mini" onClick={() => handleHistoryOpen(item, 'log')}>日志</button>
                  <button type="button" className="glass-btn mini" onClick={() => handleHistoryOpen(item, 'zip')}>子视频</button>
                  <button type="button" className="glass-btn mini" onClick={() => handleHistoryOpen(item, 'summary')}>汇总</button>
                </div>
              </article>
            ))}
          </div>
        </aside>
      </main>
    </div>
  );
}

export default App;