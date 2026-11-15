import { useCallback, useRef, useState } from 'react';
import { analyzeFirstFrame, previewCount } from '../api/client';
import type { FirstFrameData } from '../api/types';
import type { WormSelectionsMap } from '../components/HeadMarkModal/drawCanvas';
import { extractFirstFrame } from '../utils/firstFrame';

export interface PreviewOptions {
  analysisMode: 'tracking' | 'egg_count';
  setWormCount: (fn: (prev: string) => string) => void;
}

export function usePreviewCount({ analysisMode, setWormCount }: PreviewOptions) {
  const [previewing, setPreviewing] = useState(false);
  const [suggestedCount, setSuggestedCount] = useState<number | null>(null);
  const [previewAnnotated, setPreviewAnnotated] = useState<string | null>(null);
  const [firstFrameData, setFirstFrameData] = useState<FirstFrameData | null>(null);
  const [firstFrameBlob, setFirstFrameBlob] = useState<Blob | null>(null);
  const [showHeadMarkModal, setShowHeadMarkModal] = useState(false);
  const [wormSelections, setWormSelections] = useState<WormSelectionsMap>({});
  const abortRef = useRef(false);

  const resetPreview = useCallback(() => {
    setSuggestedCount(null);
    setPreviewAnnotated(null);
    setFirstFrameData(null);
    setShowHeadMarkModal(false);
    setWormSelections({});
  }, []);

  /** 选文件后：取首帧 → 并发 preview_count + analyze_first_frame → 初始化选择 */
  const runPreview = useCallback(
    async (file: File) => {
      if (!file.type.startsWith('video/') && !file.type.startsWith('image/')) return;
      abortRef.current = false;
      setPreviewing(true);
      resetPreview();
      try {
        const blob: Blob = file.type.startsWith('image/') ? file : await extractFirstFrame(file);
        if (abortRef.current) return;
        const [res1, res2] = await Promise.all([
          previewCount(blob),
          analyzeFirstFrame(blob).catch((e) => {
            console.warn('analyze_first_frame failed:', e);
            return null;
          }),
        ]);
        if (abortRef.current) return;
        setSuggestedCount(res1.count);
        if (res1.annotated) setPreviewAnnotated(`data:image/jpeg;base64,${res1.annotated}`);
        setWormCount((prev) => (prev === '' ? String(res1.count) : prev));
        if (res2?.worms) {
          const init: WormSelectionsMap = {};
          res2.worms.forEach((w, idx) => {
            init[w.det_idx] = { enabled: true, head_kpt_idx: null, user_label: `虫子${idx + 1}` };
          });
          setFirstFrameData(res2);
          setFirstFrameBlob(blob);
          setWormSelections(init);
          if (analysisMode === 'tracking') setShowHeadMarkModal(true);
        }
      } catch (e) {
        console.warn('Preview failed:', e);
      } finally {
        if (!abortRef.current) setPreviewing(false);
      }
    },
    [analysisMode, resetPreview, setWormCount],
  );

  const cancelPreview = useCallback(() => {
    abortRef.current = true;
    setPreviewing(false);
    resetPreview();
    setFirstFrameBlob(null);
  }, [resetPreview]);

  return {
    previewing,
    suggestedCount,
    previewAnnotated,
    firstFrameData,
    setFirstFrameData,
    firstFrameBlob,
    showHeadMarkModal,
    setShowHeadMarkModal,
    wormSelections,
    setWormSelections,
    runPreview,
    cancelPreview,
    resetPreview,
  };
}
