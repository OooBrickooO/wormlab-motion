import { useCallback, useEffect, useState } from 'react';
import { getModels, getProfiles, selectModel } from '../api/client';
import type { ProfileItem } from '../api/types';

export const EGG_MODEL_NAME = 'EggCount.pt';

const FALLBACK_PROFILES: ProfileItem[] = [
  { id: 'crawl_agar', label: '琼脂爬行' },
  { id: 'swim_liquid', label: '液体游泳' },
];

export function useModels() {
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [modelsLoading, setModelsLoading] = useState(false);
  const [profiles, setProfiles] = useState<ProfileItem[]>(FALLBACK_PROFILES);
  const [selectedProfile, setSelectedProfile] = useState('crawl_agar');

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setModelsLoading(true);
        const res = await getModels();
        if (!alive) return;
        if (res && Array.isArray(res.models)) {
          setModels(res.models);
          if (res.current_model) setSelectedModel(res.current_model);
          else if (res.models.length > 0) setSelectedModel(res.models[0]);
        }
      } catch (e) {
        console.warn('获取模型列表失败:', e);
      } finally {
        if (alive) setModelsLoading(false);
      }
    })();
    (async () => {
      try {
        const res = await getProfiles();
        if (!alive) return;
        if (res && Array.isArray(res.profiles) && res.profiles.length > 0) {
          setProfiles(res.profiles);
          setSelectedProfile(res.default || res.profiles[0].id);
        }
      } catch {
        /* 保持回退选项 */
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  /** 切换模型；成功返回 true */
  const changeModel = useCallback(async (name: string): Promise<boolean> => {
    setSelectedModel(name);
    try {
      await selectModel(name);
      return true;
    } catch (err) {
      console.error('切换模型失败:', err);
      return false;
    }
  }, []);

  /** 虫卵模式：自动切到 EggCount.pt（若可用） */
  const ensureEggModel = useCallback(async () => {
    if (models.includes(EGG_MODEL_NAME) && selectedModel !== EGG_MODEL_NAME) {
      setSelectedModel(EGG_MODEL_NAME);
      try {
        await selectModel(EGG_MODEL_NAME);
      } catch (err) {
        console.warn('自动切换 EggCount.pt 失败:', err);
      }
    }
  }, [models, selectedModel]);

  return {
    models,
    selectedModel,
    modelsLoading,
    changeModel,
    ensureEggModel,
    profiles,
    selectedProfile,
    setSelectedProfile,
  };
}
