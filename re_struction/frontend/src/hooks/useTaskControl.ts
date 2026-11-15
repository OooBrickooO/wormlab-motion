import { useCallback } from 'react';
import axios from 'axios';
import { setWormCount as apiSetWormCount, taskAction } from '../api/client';
import { ACTIVE_FILE_KEY, ACTIVE_TASK_KEY } from './useHistory';

export interface TaskControlOptions {
  taskId: string | null;
  setStatus: (s: string) => void;
  setError: (m: string) => void;
  updateHistoryStatus: (id: string, s: string) => void;
}

export function useTaskControl({ taskId, setStatus, setError, updateHistoryStatus }: TaskControlOptions) {
  const control = useCallback(
    async (action: 'pause' | 'resume' | 'cancel') => {
      if (!taskId) return;
      try {
        const res = await taskAction(taskId, action);
        const next = res?.status;
        if (next === 'paused') setStatus('paused');
        if (next === 'processing') setStatus('processing');
        if (next === 'cancelled') {
          setStatus('cancelled');
          sessionStorage.removeItem(ACTIVE_TASK_KEY);
          sessionStorage.removeItem(ACTIVE_FILE_KEY);
        }
        updateHistoryStatus(taskId, next || action);
      } catch (e) {
        console.error(e);
        if (action === 'cancel' && axios.isAxiosError(e) && e.response?.status === 404) {
          setStatus('cancelled');
          sessionStorage.removeItem(ACTIVE_TASK_KEY);
          sessionStorage.removeItem(ACTIVE_FILE_KEY);
          updateHistoryStatus(taskId, 'cancelled');
          setError('任务已不在服务端，已停止本地监视');
          return;
        }
        setError('任务控制失败，请稍后重试');
      }
    },
    [taskId, setStatus, setError, updateHistoryStatus],
  );

  /** 运行中修改虫数上限（<=0 表示取消上限） */
  const changeWormCount = useCallback(
    async (n: number) => {
      if (!taskId) return;
      try {
        await apiSetWormCount(taskId, n);
      } catch (e) {
        console.error(e);
        setError('修改虫数失败，请稍后重试');
      }
    },
    [taskId, setError],
  );

  return { pause: () => control('pause'), resume: () => control('resume'), cancel: () => control('cancel'), changeWormCount };
}
