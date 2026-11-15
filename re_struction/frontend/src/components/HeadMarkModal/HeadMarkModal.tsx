import React, { useCallback, useEffect, useRef, useState } from 'react';
import { extractManualWorm } from '../../api/client';
import type { FirstFrameData, FirstFrameWorm } from '../../api/types';
import {
  drawCanvas,
  type DrawingRect,
  WORM_COLORS,
  type WormSelectionsMap,
} from './drawCanvas';

export interface HeadMarkModalProps {
  firstFrameData: FirstFrameData;
  firstFrameBlob: Blob | null;
  wormSelections: WormSelectionsMap;
  onSelectionsChange: React.Dispatch<React.SetStateAction<WormSelectionsMap>>;
  onAddManualWorm?: (worm: FirstFrameWorm) => void;
  onConfirm: () => void;
  onSkip: () => void;
}

export function HeadMarkModal({
  firstFrameData,
  firstFrameBlob,
  wormSelections,
  onSelectionsChange,
  onAddManualWorm,
  onConfirm,
  onSkip,
}: HeadMarkModalProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [imgLoaded, setImgLoaded] = useState(false);
  const [hoveredDet, setHoveredDet] = useState<number | null>(null);
  const [canvasSize, setCanvasSize] = useState({ w: 800, h: 450 });

  // 手动框选状态
  const [isDrawMode, setIsDrawMode] = useState(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [drawingRect, setDrawingRect] = useState<DrawingRect | null>(null);
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
    if (!imgLoaded || !wrapRef.current || !imgRef.current) return;
    const img = imgRef.current;
    const wrap = wrapRef.current;
    const maxW = Math.max(200, wrap.clientWidth - 4);
    const maxH = Math.max(150, wrap.clientHeight - 4);
    const ratio = Math.max(0.1, img.naturalWidth / Math.max(1, img.naturalHeight));
    let w = maxW;
    let h = w / ratio;
    if (h > maxH) {
      h = maxH;
      w = h * ratio;
    }
    setCanvasSize({ w: Math.floor(w), h: Math.floor(h) });
  }, [imgLoaded]);

  // Redraw whenever state changes
  useEffect(() => {
    if (!imgLoaded || !canvasRef.current) return;
    const canvas = canvasRef.current;
    canvas.width = canvasSize.w;
    canvas.height = canvasSize.h;
    drawCanvas(
      canvas,
      imgRef.current,
      firstFrameData.worms,
      wormSelections,
      hoveredDet,
      drawingRect,
    );
  }, [imgLoaded, canvasSize, wormSelections, hoveredDet, firstFrameData, drawingRect]);

  const getClickedEndpoint = useCallback(
    (clientX: number, clientY: number): { det_idx: number; head_kpt_idx: number } | null => {
      const canvas = canvasRef.current;
      if (!canvas || !imgRef.current) return null;
      const rect = canvas.getBoundingClientRect();
      const mx = (clientX - rect.left) * (canvas.width / Math.max(1, rect.width));
      const my = (clientY - rect.top) * (canvas.height / Math.max(1, rect.height));
      const scaleX = canvas.width / Math.max(1, imgRef.current.naturalWidth);
      const scaleY = canvas.height / Math.max(1, imgRef.current.naturalHeight);
      const HIT_R = 18; // px hit radius in canvas coords

      let bestDist = HIT_R;
      let bestResult: { det_idx: number; head_kpt_idx: number } | null = null;

      firstFrameData.worms.forEach((worm) => {
        const sel = wormSelections[worm.det_idx];
        if (!sel?.enabled) return;
        if (!worm.keypoints) return;
        // Only allow endpoints: index 0 or 10
        [0, 10].forEach((ki) => {
          const pt = worm.keypoints?.[ki];
          if (!pt) return;
          const [kx, ky] = pt;
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
    },
    [firstFrameData, wormSelections],
  );

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawMode || extracting) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    setDragStart({ x: mx, y: my });
    setDrawingRect(null);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
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

  const handleMouseUp = async () => {
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
    if (!canvas || !imgRef.current || !firstFrameBlob) return;
    const scaleX = imgRef.current.naturalWidth / Math.max(1, canvas.width);
    const scaleY = imgRef.current.naturalHeight / Math.max(1, canvas.height);
    const imgBbox: [number, number, number, number] = [
      Math.round(rect.x * scaleX),
      Math.round(rect.y * scaleY),
      Math.round((rect.x + rect.w) * scaleX),
      Math.round((rect.y + rect.h) * scaleY),
    ];

    try {
      setExtracting(true);
      const data = await extractManualWorm(firstFrameBlob, imgBbox);
      if (data) {
        const maxIdx = firstFrameData.worms.reduce((m, w) => Math.max(m, w.det_idx), -1);
        const newDetIdx = maxIdx + 1;
        const newWorm: FirstFrameWorm = {
          det_idx: newDetIdx,
          confidence: 1.0,
          bbox: data.bbox,
          centroid: data.centroid,
          keypoints: data.keypoints,
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

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDrawMode) return;
    const hit = getClickedEndpoint(e.clientX, e.clientY);
    if (!hit) return;
    onSelectionsChange((prev) => ({
      ...prev,
      [hit.det_idx]: { ...prev[hit.det_idx], head_kpt_idx: hit.head_kpt_idx },
    }));
  };

  const toggleEnabled = (det_idx: number) => {
    onSelectionsChange((prev) => ({
      ...prev,
      [det_idx]: { ...prev[det_idx], enabled: !prev[det_idx]?.enabled },
    }));
  };

  const setLabel = (det_idx: number, label: string) => {
    onSelectionsChange((prev) => ({
      ...prev,
      [det_idx]: { ...prev[det_idx], user_label: label },
    }));
  };

  const setHeadIdx = (det_idx: number, head_kpt_idx: number) => {
    onSelectionsChange((prev) => ({
      ...prev,
      [det_idx]: { ...prev[det_idx], head_kpt_idx },
    }));
  };

  const allEnabledHaveHead = firstFrameData.worms.every((w) => {
    const sel = wormSelections[w.det_idx];
    if (!sel?.enabled) return true; // disabled worms don't need head
    return sel.head_kpt_idx !== null && sel.head_kpt_idx !== undefined;
  });

  const enabledCount = firstFrameData.worms.filter(
    (w) => wormSelections[w.det_idx]?.enabled,
  ).length;

  return (
    <div className="head-mark-overlay">
      <div className="head-mark-dialog">
        <div className="head-mark-header">
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '12px',
            }}
          >
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
                {extracting
                  ? '提取骨架中...'
                  : isDrawMode
                  ? '✖ 退出框选'
                  : '➕ 手动框选漏检虫子'}
              </button>
            </div>
          </div>
          <p>
            在右侧勾选要分析的虫子，然后
            <strong>点击每条虫子两端的绿色/蓝色节点（0 或 10）</strong>指定为头部。 已选中{' '}
            {enabledCount} 条 / 共 {firstFrameData.worms.length} 条虫子。
            {isDrawMode && (
              <strong style={{ color: '#28c76f', marginLeft: 8 }}>
                [框选模式已开启：请在画面上按住鼠标左键拉框]
              </strong>
            )}
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
                  onMouseLeave={() => {
                    setHoveredDet(null);
                    setDragStart(null);
                    setDrawingRect(null);
                  }}
                />
                <div className="head-mark-canvas-hint">
                  {isDrawMode
                    ? '按住鼠标左键拖拽拉框以添加漏检虫子'
                    : '点击两端节点（0 或 10）设为头部 · 绿色=已选头部'}
                </div>
              </>
            ) : (
              <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: '14px' }}>加载中...</div>
            )}
          </div>

          <div className="worm-list-sidebar">
            <div className="worm-list-title">虫子列表</div>
            {firstFrameData.worms.map((worm, idx) => {
              const sel = wormSelections[worm.det_idx] || {
                enabled: true,
                head_kpt_idx: null,
                user_label: '',
              };
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
                    <span
                      style={{
                        width: 12,
                        height: 12,
                        borderRadius: 3,
                        background: color,
                        flexShrink: 0,
                        display: 'inline-block',
                      }}
                    />
                    {worm.is_manual && (
                      <span
                        style={{
                          fontSize: '10px',
                          color: '#1a7a3a',
                          background: 'rgba(40,199,111,0.2)',
                          padding: '1px 5px',
                          borderRadius: 4,
                          fontWeight: 700,
                        }}
                      >
                        手动
                      </span>
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
                    <div className="worm-item-conf">
                      置信度: {(worm.confidence * 100).toFixed(1)}%
                    </div>
                  )}
                  {worm.keypoints ? (
                    <div className="worm-item-head-status">
                      头部节点:&nbsp;
                      <span className={`head-kpt-badge ${headSet ? 'badge-set' : 'badge-unset'}`}>
                        {headSet ? `pt ${sel.head_kpt_idx}` : '未选'}
                      </span>
                      {enabled && (
                        <span style={{ display: 'inline-flex', gap: 4, marginLeft: 6 }}>
                          <button
                            type="button"
                            className="glass-btn mini"
                            style={{ padding: '1px 6px', fontSize: 11 }}
                            onClick={() => setHeadIdx(worm.det_idx, 0)}
                          >
                            端点0
                          </button>
                          <button
                            type="button"
                            className="glass-btn mini"
                            style={{ padding: '1px 6px', fontSize: 11 }}
                            onClick={() => setHeadIdx(worm.det_idx, 10)}
                          >
                            端点10
                          </button>
                        </span>
                      )}
                      {!headSet && enabled && (
                        <span style={{ color: '#d64040', fontSize: '11px' }}>← 请在图中点选</span>
                      )}
                    </div>
                  ) : (
                    <div className="worm-item-conf" style={{ color: '#e88' }}>
                      骨架提取失败（仍可分析）
                    </div>
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
export default HeadMarkModal;
