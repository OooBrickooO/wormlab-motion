import React, { useEffect, useRef, useState } from 'react';
import ReactDOM from 'react-dom';
import type { EggCountResponse } from '../api/types';

export interface EggCountResultPanelProps {
  result: EggCountResponse | null;
  onClear: () => void;
}

export function EggCountResultPanel({ result, onClear }: EggCountResultPanelProps) {
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  const [zoom, setZoom] = useState(100);
  const [minConf, setMinConf] = useState(80);
  const [maxConf, setMaxConf] = useState(100);

  const [filteredImgSrc, setFilteredImgSrc] = useState('');

  // Drag-to-pan state
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef({ x: 0, y: 0 });
  const initialPosRef = useRef({ x: 0, y: 0 });

  useEffect(() => {
    if (!result || !result.raw_image) return;
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.drawImage(img, 0, 0);

      const filteredEggs = (result.eggs || []).filter((e) => {
        const c = e.confidence * 100;
        return c >= minConf && c <= maxConf;
      });

      filteredEggs.forEach((det) => {
        const [x1, y1, x2, y2] = det.bbox;
        ctx.strokeStyle = '#ff8c00'; // 橙色
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        ctx.fillStyle = '#ff8c00';
        ctx.font = '14px sans-serif';
        ctx.fillText(`egg ${det.confidence.toFixed(2)}`, x1, Math.max(14, y1 - 6));
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
  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
    dragStartRef.current = { x: e.clientX, y: e.clientY };
    initialPosRef.current = { ...position };
  };
  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!isDragging) return;
    e.preventDefault();
    setPosition({
      x: initialPosRef.current.x + (e.clientX - dragStartRef.current.x),
      y: initialPosRef.current.y + (e.clientY - dragStartRef.current.y),
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

  const customCount = eggs.filter((e) => {
    const c = e.confidence * 100;
    return c >= minConf && c <= maxConf;
  }).length;

  return (
    <div className="egg-result-panel glass-card">
      <div className="egg-result-header">
        <span>虫卵计数结果</span>
        <button type="button" className="glass-btn mini" onClick={onClear}>
          清除
        </button>
      </div>

      <div className="egg-count-cards">
        <div className="egg-count-card card-egg">
          <div className="egg-count-num">{egg_count}</div>
          <div className="egg-count-label">总虫卵数</div>
        </div>
        <div
          className="egg-count-card card-egg-90"
          style={{
            borderColor: 'rgba(38, 166, 91, 0.28)',
            background: 'rgba(38, 166, 91, 0.06)',
          }}
        >
          <div
            className="egg-count-num"
            style={{
              background: 'linear-gradient(135deg, #26a65b, #1a7a41)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
            }}
          >
            {egg_count_90}
          </div>
          <div className="egg-count-label">置信度 ≥ 90%</div>
        </div>
        <div
          className="egg-count-card card-egg-95"
          style={{
            borderColor: 'rgba(22, 119, 255, 0.28)',
            background: 'rgba(22, 119, 255, 0.06)',
          }}
        >
          <div
            className="egg-count-num"
            style={{
              background: 'linear-gradient(135deg, #1677ff, #0f50cc)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
            }}
          >
            {egg_count_95}
          </div>
          <div className="egg-count-label">置信度 ≥ 95%</div>
        </div>
      </div>

      <div
        style={{
          padding: '12px',
          background: 'rgba(255,255,255,0.6)',
          borderRadius: '12px',
          marginBottom: '16px',
          border: '1px solid rgba(22,119,255,0.15)',
        }}
      >
        <div
          style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text)', marginBottom: '8px' }}
        >
          自定义置信度区间查询：
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px' }}>
          <input
            type="number"
            aria-label="min-confidence"
            value={minConf}
            onChange={(e) => setMinConf(Number(e.target.value))}
            style={{
              width: '60px',
              padding: '4px 8px',
              borderRadius: '6px',
              border: '1px solid #ccc',
            }}
          />{' '}
          %
          <span>至</span>
          <input
            type="number"
            aria-label="max-confidence"
            value={maxConf}
            onChange={(e) => setMaxConf(Number(e.target.value))}
            style={{
              width: '60px',
              padding: '4px 8px',
              borderRadius: '6px',
              border: '1px solid #ccc',
            }}
          />{' '}
          %
          <span style={{ marginLeft: 'auto', fontWeight: 700, color: 'var(--primary)' }}>
            区间包含虫卵数: {customCount} 个
          </span>
        </div>
        <div style={{ fontSize: '11px', color: 'var(--muted)', marginTop: '6px' }}>
          * 修改区间后，下方的分析图会自动滤除区间外的虫卵识别框。
        </div>
      </div>

      {filteredImgSrc && (
        <>
          <div
            className="egg-annotated-wrap"
            onClick={() => setIsPreviewOpen(true)}
            style={{ cursor: 'zoom-in', position: 'relative' }}
          >
            <img src={filteredImgSrc} alt="虫卵检测标注图" className="egg-annotated-img" />
            <div
              style={{
                textAlign: 'center',
                padding: '10px',
                fontSize: '13px',
                color: 'var(--primary)',
                fontWeight: 600,
                background: 'rgba(22, 119, 255, 0.05)',
              }}
            >
              🔍 点击放大查看无损原图 (支持鼠标拖动平移)
            </div>
          </div>

          {isPreviewOpen &&
            ReactDOM.createPortal(
              <div
                style={{
                  position: 'fixed',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  backgroundColor: 'rgba(0, 0, 0, 0.9)',
                  zIndex: 99999,
                  display: 'flex',
                  flexDirection: 'column',
                }}
              >
                <div
                  style={{
                    padding: '16px 24px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    background: 'rgba(0,0,0,0.6)',
                    backdropFilter: 'blur(10px)',
                    zIndex: 2,
                  }}
                >
                  <div
                    style={{ color: 'white', display: 'flex', alignItems: 'center', gap: '20px' }}
                  >
                    <span style={{ fontSize: '14px', fontWeight: 600 }}>无损放大: {zoom}%</span>
                    <input
                      type="range"
                      min="10"
                      max="500"
                      value={zoom}
                      onChange={(e) => setZoom(Number(e.target.value))}
                      style={{ width: '200px' }}
                    />
                    <button
                      type="button"
                      onClick={() => {
                        setZoom(100);
                        setPosition({ x: 0, y: 0 });
                      }}
                      className="glass-btn mini"
                      style={{
                        color: 'white',
                        borderColor: 'rgba(255,255,255,0.3)',
                        background: 'transparent',
                      }}
                    >
                      重置
                    </button>
                  </div>
                  <button
                    type="button"
                    onClick={closePreview}
                    className="glass-btn mini"
                    style={{
                      color: 'white',
                      borderColor: 'rgba(255,255,255,0.3)',
                      background: 'transparent',
                      fontSize: '14px',
                    }}
                  >
                    ✖ 关闭
                  </button>
                </div>
                <div
                  style={{
                    flex: 1,
                    overflow: 'hidden',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    cursor: isDragging ? 'grabbing' : 'grab',
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
                      transform: `translate(${position.x}px, ${position.y}px) scale(${zoom / 100})`,
                      transformOrigin: 'center center',
                      transition: isDragging ? 'none' : 'transform 0.1s ease-out',
                      boxShadow: '0 12px 48px rgba(0,0,0,0.5)',
                      borderRadius: '4px',
                      maxWidth: '95vw',
                      maxHeight: '95vh',
                      objectFit: 'contain',
                    }}
                  />
                </div>
              </div>,
              document.body,
            )}
        </>
      )}
    </div>
  );
}
export default EggCountResultPanel;
