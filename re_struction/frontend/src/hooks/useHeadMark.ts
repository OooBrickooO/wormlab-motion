import { useCallback } from 'react';
import type { FirstFrameData, HeadConfigItem } from '../api/types';
import type { WormSelectionsMap } from '../components/HeadMarkModal/drawCanvas';

/** 由首帧选择构造 head_config（旧 App.js 1227-1246） */
export function buildHeadConfig(
  data: FirstFrameData | null,
  sels: WormSelectionsMap,
): HeadConfigItem[] {
  return (data?.worms || [])
    .filter((w) => {
      const s = sels[w.det_idx];
      return s?.enabled && s.head_kpt_idx !== null && s.head_kpt_idx !== undefined;
    })
    .map((w) => {
      const s = sels[w.det_idx];
      return {
        det_idx: w.det_idx,
        head_kpt_idx: s.head_kpt_idx as number,
        user_label: s.user_label || `虫子${w.det_idx + 1}`,
        centroid_x: w.centroid[0],
        centroid_y: w.centroid[1],
        bbox: w.bbox,
        keypoints: w.keypoints ?? undefined,
      } as HeadConfigItem;
    });
}

export interface HeadMarkOptions {
  firstFrameData: FirstFrameData | null;
  wormSelections: WormSelectionsMap;
  wormCount: string;
  setWormCount: (s: string) => void;
  setShowHeadMarkModal: (b: boolean) => void;
  submit: (headConfig: HeadConfigItem[], count: number) => Promise<void>;
}

export function useHeadMark(o: HeadMarkOptions) {
  const confirm = useCallback(async () => {
    o.setShowHeadMarkModal(false);
    const headConfig = buildHeadConfig(o.firstFrameData, o.wormSelections);
    const userCount = parseInt(o.wormCount, 10) || 0;
    const enabled = (o.firstFrameData?.worms || []).filter(
      (w) => o.wormSelections[w.det_idx]?.enabled,
    ).length;
    // 用户输入优先，且不小于已勾选数量
    const finalCount = userCount > 0 ? Math.max(userCount, enabled) : enabled;
    if (finalCount > 0) o.setWormCount(String(finalCount));
    await o.submit(headConfig, finalCount);
  }, [o]);

  const skip = useCallback(async () => {
    o.setShowHeadMarkModal(false);
    const userCount = parseInt(o.wormCount, 10) || 0;
    const detected = (o.firstFrameData?.worms || []).length;
    // 用户手动输入时严格使用，不被检测数覆盖
    const finalCount = userCount > 0 ? userCount : detected;
    if (finalCount > 0) o.setWormCount(String(finalCount));
    await o.submit([], finalCount);
  }, [o]);

  return { confirm, skip };
}
