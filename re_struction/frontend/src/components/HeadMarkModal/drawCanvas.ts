import type { FirstFrameWorm } from '../../api/types';

export const WORM_COLORS: string[] = [
  '#ff6b35',
  '#28c76f',
  '#7b68ee',
  '#ff9f43',
  '#00cec9',
  '#e84393',
  '#fdcb6e',
  '#0984e3',
  '#6c5ce7',
  '#a29bfe',
];

export interface WormSelection {
  enabled: boolean;
  head_kpt_idx: number | null;
  user_label: string;
}

export type WormSelectionsMap = Record<number, WormSelection>;

export interface DrawingRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export function drawCanvas(
  canvas: HTMLCanvasElement | null,
  img: HTMLImageElement | null,
  worms: FirstFrameWorm[],
  wormSelections: WormSelectionsMap,
  hoveredDet: number | null,
  drawingRect: DrawingRect | null = null,
): void {
  if (!canvas || !img) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.drawImage(img, 0, 0, W, H);

  const scaleX = W / Math.max(1, img.naturalWidth);
  const scaleY = H / Math.max(1, img.naturalHeight);

  worms.forEach((worm, idx) => {
    const sel: Partial<WormSelection> = wormSelections[worm.det_idx] || {};
    const enabled = sel.enabled !== false;
    const color = WORM_COLORS[idx % WORM_COLORS.length];
    const isHovered = hoveredDet === worm.det_idx;
    const alpha = enabled ? 1.0 : 0.28;

    ctx.globalAlpha = alpha;

    const [bx1, by1, bx2, by2] = worm.bbox;
    ctx.strokeStyle = isHovered ? '#fff' : color;
    ctx.lineWidth = isHovered ? 2.5 : 1.5;
    ctx.strokeRect(bx1 * scaleX, by1 * scaleY, (bx2 - bx1) * scaleX, (by2 - by1) * scaleY);

    ctx.fillStyle = color;
    ctx.font = 'bold 13px sans-serif';
    ctx.fillText(sel.user_label || `#${worm.det_idx}`, bx1 * scaleX + 4, by1 * scaleY - 5);

    if (!worm.keypoints) {
      ctx.globalAlpha = 1;
      return;
    }

    const kpts = worm.keypoints;

    ctx.strokeStyle = `${color}88`;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    kpts.forEach(([kx, ky], ki) => {
      if (ki === 0) ctx.moveTo(kx * scaleX, ky * scaleY);
      else ctx.lineTo(kx * scaleX, ky * scaleY);
    });
    ctx.stroke();

    kpts.forEach(([kx, ky], ki) => {
      const isEndpoint = ki === 0 || ki === 10;
      const isHead = sel.head_kpt_idx === ki;
      const cx2 = kx * scaleX;
      const cy2 = ky * scaleY;

      if (isHead) {
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
        ctx.beginPath();
        ctx.arc(cx2, cy2, 7, 0, Math.PI * 2);
        ctx.fillStyle = '#1677ff';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 8px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(ki), cx2, cy2);
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
      } else {
        ctx.beginPath();
        ctx.arc(cx2, cy2, 3, 0, Math.PI * 2);
        ctx.fillStyle = isEndpoint ? '#56a3ff' : 'rgba(200,200,200,0.7)';
        ctx.fill();
        ctx.fillStyle = 'rgba(255,255,200,0.85)';
        ctx.font = '9px sans-serif';
        ctx.fillText(String(ki), cx2 + 5, cy2 - 3);
      }
    });

    ctx.globalAlpha = 1;
  });

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
