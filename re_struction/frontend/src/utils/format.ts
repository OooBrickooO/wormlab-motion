export function formatPercent(current: number | undefined | null, total: number | undefined | null): number {
  if (!current || !total || total <= 0) return 0;
  return Math.min(100, Math.round((current / total) * 100));
}

export function formatCoordPt(pt: { x: number; y: number } | null | undefined): string {
  if (!pt) return '—';
  return `(${pt.x.toFixed(1)}, ${pt.y.toFixed(1)})`;
}
