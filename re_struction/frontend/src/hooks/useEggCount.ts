import { useCallback, useState } from 'react';
import { eggCount } from '../api/client';
import type { EggCountResponse } from '../api/types';

export function useEggCount(setError: (m: string) => void) {
  const [eggCountResult, setEggCountResult] = useState<EggCountResponse | null>(null);
  const [eggCounting, setEggCounting] = useState(false);

  const runEggCount = useCallback(
    async (file: File | undefined) => {
      if (!file) {
        setError('请先选择视频或图片');
        return;
      }
      setError('');
      setEggCounting(true);
      setEggCountResult(null);
      try {
        setEggCountResult(await eggCount(file));
      } catch (e) {
        const err = e as { response?: { data?: { error?: string } }; message?: string };
        setError(err?.response?.data?.error || err?.message || '虫卵计数失败');
      } finally {
        setEggCounting(false);
      }
    },
    [setError],
  );

  const clearEggResult = useCallback(() => setEggCountResult(null), []);
  return { eggCountResult, eggCounting, runEggCount, clearEggResult };
}
