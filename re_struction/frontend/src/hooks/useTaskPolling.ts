import { useCallback, useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { getStatus, listSubVideos } from '../api/client';
import { toAbsoluteUrl } from '../api/urls';
import type { FrameStats, SubVideoItem, TrackSummary } from '../api/types';
import { ACTIVE_FILE_KEY, ACTIVE_TASK_KEY, type HistoryPatch } from './useHistory';

export interface PollingOptions {
  taskId: string | null;
  refreshKey?: number;
  taskStatus?: string;
  socketConnected: boolean;
  setStatus: (s: string) => void;
  setError: (m: string) => void;
  setCurrentFrame: (f: string) => void;
  setCurrentStats: (fn: (prev: FrameStats | null) => FrameStats) => void;
  setTotalFrames: (fn: (prev: number) => number) => void;
  upsertHistoryRecord: (r: HistoryPatch) => void;
  updateHistoryStatus: (id: string, s: string) => void;
  onWormCount?: (n: string) => void;
}

export const POLL_INTERVAL_MS = 1500;

export function useTaskPolling(o: PollingOptions) {
  const [videoUrl, setVideoUrl] = useState('');
  const [logUrl, setLogUrl] = useState('');
  const [zipUrl, setZipUrl] = useState('');
  const [summaryUrl, setSummaryUrl] = useState('');
  const [csvUrl, setCsvUrl] = useState('');
  const [subVideos, setSubVideos] = useState<SubVideoItem[]>([]);
  const [selectedSubVideo, setSelectedSubVideo] = useState('');
  const [idPoseSummary, setIdPoseSummary] = useState<TrackSummary[]>([]);

  const ref = useRef(o);
  ref.current = o;

  const resetPolled = useCallback(() => {
    setVideoUrl('');
    setLogUrl('');
    setZipUrl('');
    setSummaryUrl('');
    setCsvUrl('');
    setSubVideos([]);
    setSelectedSubVideo('');
    setIdPoseSummary([]);
  }, []);

  useEffect(() => {
    const taskId = o.taskId;
    if (!taskId) return;
    let cancelled = false;
    let timer: number | null = null;
    const stop = () => {
      cancelled = true;
      if (timer !== null) {
        window.clearInterval(timer);
        timer = null;
      }
    };
    const clearActive = () => {
      sessionStorage.removeItem(ACTIVE_TASK_KEY);
      sessionStorage.removeItem(ACTIVE_FILE_KEY);
    };

    const tick = async () => {
      try {
        const data = await getStatus(taskId);
        if (cancelled) return;
        const c = ref.current;
        if (
          c.taskStatus === 'cancelled' &&
          data.status !== 'completed' &&
          data.status !== 'cancelled' &&
          data.status !== 'failed'
        ) {
          stop();
          return;
        }
        const urls = {
          video: toAbsoluteUrl(data.video_url || ''),
          log: toAbsoluteUrl(data.log_url || ''),
          zip: toAbsoluteUrl(data.sub_videos_url || ''),
          summary: toAbsoluteUrl(data.summary_url || ''),
          csv: toAbsoluteUrl(data.csv_url || ''),
        };
        const rec = { id: taskId, name: data.filename || `任务${taskId}`, ...urls };

        if (data.status === 'processing') {
          c.setStatus('processing');
          c.upsertHistoryRecord({ ...rec, status: 'processing' });
        } else if (data.status === 'completed') {
          stop();
          c.setStatus('completed');
          clearActive();
          c.upsertHistoryRecord({ ...rec, status: 'completed' });
          try {
            const sub = await listSubVideos(taskId);
            const videos = (sub.videos || []).map((v) => ({ ...v, url: toAbsoluteUrl(v.url) }));
            setSubVideos(videos);
            setSelectedSubVideo((prev) => prev || (videos[0]?.url ?? ''));
          } catch (e) {
            console.error(e);
          }
        } else if (data.status === 'paused') {
          c.setStatus('paused');
          c.updateHistoryStatus(taskId, 'paused');
        } else if (data.status === 'cancelled') {
          stop();
          c.setStatus('cancelled');
          clearActive();
          c.updateHistoryStatus(taskId, 'cancelled');
        } else if (data.status === 'failed') {
          stop();
          c.setStatus('idle');
          if (data.error_message) c.setError(data.error_message);
          c.updateHistoryStatus(taskId, 'failed');
        }

        if (data.current_frame || data.total_frames) {
          c.setCurrentStats((prev) => ({
            frame: Math.max(data.current_frame || 0, prev?.frame || 0),
            active_tracks: data.active_tracks || 0,
            tracks: data.tracks || prev?.tracks || [],
          }));
          c.setTotalFrames((tf) => Math.max(data.total_frames || 0, tf));
        }
        if (!c.socketConnected && data.latest_frame) {
          c.setCurrentFrame(`data:image/jpeg;base64,${data.latest_frame}`);
        }
        if (data.video_url) setVideoUrl(urls.video);
        if (data.log_url) setLogUrl(urls.log);
        if (data.sub_videos_url) setZipUrl(urls.zip);
        if (data.summary_url) setSummaryUrl(urls.summary);
        if (data.csv_url) setCsvUrl(urls.csv);
        if (Array.isArray(data.id_pose_summary)) setIdPoseSummary(data.id_pose_summary);
        if (data.worm_count !== undefined && data.worm_count !== null) {
          c.onWormCount?.(String(data.worm_count));
        }
      } catch (e) {
        if (cancelled) return;
        if (axios.isAxiosError(e) && e.response?.status === 404) {
          stop();
          clearActive();
          const c = ref.current;
          c.setStatus('idle');
          c.updateHistoryStatus(taskId, 'cancelled');
          c.setError('任务不存在或后端服务已重启，已退出当前状态');
          return;
        }
        console.error(e);
      }
    };

    void tick();
    timer = window.setInterval(tick, POLL_INTERVAL_MS);
    return stop;
  }, [o.taskId, o.refreshKey]);

  return {
    videoUrl,
    logUrl,
    zipUrl,
    summaryUrl,
    csvUrl,
    subVideos,
    selectedSubVideo,
    setSelectedSubVideo,
    idPoseSummary,
    resetPolled,
  };
}
