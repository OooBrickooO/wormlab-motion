import { useCallback, useEffect, useState } from 'react';
import { getStatus } from '../api/client';
import { toAbsoluteUrl } from '../api/urls';

export const HISTORY_KEY = 'motion_history_records';
export const ACTIVE_TASK_KEY = 'motion_active_task_id';
export const ACTIVE_FILE_KEY = 'motion_active_file_name';

export interface HistoryRecord {
  id: string;
  name: string;
  status: string;
  time: string;
  video: string;
  log: string;
  zip: string;
  summary: string;
  csv?: string;
}

export type HistoryPatch = Partial<Omit<HistoryRecord, 'id'>> & { id: string | number };
export type ResourceKey = 'video' | 'log' | 'zip' | 'summary' | 'csv';

export function loadHistory(): HistoryRecord[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function useHistory() {
  const [historyList, setHistoryList] = useState<HistoryRecord[]>([]);

  useEffect(() => {
    setHistoryList(loadHistory());
  }, []);

  const upsertHistoryRecord = useCallback((record: HistoryPatch) => {
    setHistoryList((prev) => {
      const id = String(record.id);
      const existing: Partial<HistoryRecord> = prev.find((i) => String(i.id) === id) || {};
      const payload: HistoryRecord = {
        id,
        name: record.name ?? existing.name ?? `任务${id}`,
        status: record.status ?? existing.status ?? 'processing',
        time: record.time ?? new Date().toLocaleString(),
        video: record.video ?? existing.video ?? '',
        log: record.log ?? existing.log ?? '',
        zip: record.zip ?? existing.zip ?? '',
        summary: record.summary ?? existing.summary ?? '',
        csv: record.csv ?? existing.csv ?? '',
      };
      const next = [payload, ...prev.filter((i) => String(i.id) !== id)].slice(0, 50);
      localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  const updateHistoryStatus = useCallback(
    (taskId: string | number, status: string) => {
      upsertHistoryRecord({ id: taskId, status, time: new Date().toLocaleString() });
    },
    [upsertHistoryRecord],
  );

  const resolveResourceUrl = useCallback(
    async (item: HistoryRecord, key: ResourceKey): Promise<string> => {
      if (item[key]) return toAbsoluteUrl(item[key]);
      if (!item.id) return '';
      try {
        const data = await getStatus(item.id);
        const recovered = {
          video: toAbsoluteUrl(data.video_url || item.video || ''),
          log: toAbsoluteUrl(data.log_url || item.log || ''),
          zip: toAbsoluteUrl(data.sub_videos_url || item.zip || ''),
          summary: toAbsoluteUrl(data.summary_url || item.summary || ''),
          csv: toAbsoluteUrl(data.csv_url || item.csv || ''),
        };
        upsertHistoryRecord({
          id: item.id,
          name: item.name,
          status: data.status || item.status,
          ...recovered,
        });
        return recovered[key] || '';
      } catch (e) {
        console.error(e);
        return '';
      }
    },
    [upsertHistoryRecord],
  );

  /** 打开历史资源；无可用结果时通过 onError 提示 */
  const handleHistoryOpen = useCallback(
    async (item: HistoryRecord, key: ResourceKey, onError?: (msg: string) => void) => {
      const url = await resolveResourceUrl(item, key);
      if (!url) {
        onError?.('该历史任务暂无可访问结果，请重新处理该视频后再试。');
        return;
      }
      window.open(url, '_blank');
    },
    [resolveResourceUrl],
  );

  return { historyList, upsertHistoryRecord, updateHistoryStatus, handleHistoryOpen };
}

