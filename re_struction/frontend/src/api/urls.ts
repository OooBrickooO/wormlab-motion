const DEFAULT_API_BASE =
  typeof window !== 'undefined' && window.location?.hostname
    ? `${window.location.protocol === 'https:' ? 'https' : 'http'}://${window.location.hostname}:5001`
    : 'http://localhost:5001';

export const API_BASE: string =
  (import.meta.env && import.meta.env.VITE_API_BASE) || DEFAULT_API_BASE;

export function toAbsoluteUrl(url: string | undefined | null): string {
  if (!url) return '';
  if (url.startsWith('http://') || url.startsWith('https://')) return url;
  return `${API_BASE}${url.startsWith('/') ? '' : '/'}${url}`;
}
