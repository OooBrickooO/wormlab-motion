import axios from 'axios';
import { API_BASE } from './urls';
import type {
  EggCountResponse,
  FirstFrameData,
  FirstFrameWorm,
  PreviewCountResponse,
  ProfileItem,
  SubVideoItem,
  TaskState,
} from './types';

export const http = axios.create({
  baseURL: API_BASE,
});

export async function uploadVideo(formData: FormData): Promise<{ task_id: string }> {
  const { data } = await http.post<{ task_id: string }>('/upload', formData);
  return data;
}

export async function previewCount(frameBlob: Blob): Promise<PreviewCountResponse> {
  const fd = new FormData();
  fd.append('frame', frameBlob, 'first_frame.jpg');
  const { data } = await http.post<PreviewCountResponse>('/preview_count', fd);
  return data;
}

export async function analyzeFirstFrame(frameBlob: Blob): Promise<FirstFrameData> {
  const fd = new FormData();
  fd.append('frame', frameBlob, 'first_frame.jpg');
  const { data } = await http.post<FirstFrameData>('/analyze_first_frame', fd);
  return data;
}

export async function extractManualWorm(
  frameBlob: Blob,
  bbox: [number, number, number, number],
): Promise<Pick<FirstFrameWorm, 'bbox' | 'centroid' | 'keypoints'>> {
  const fd = new FormData();
  fd.append('frame', frameBlob, 'first_frame.jpg');
  fd.append('bbox', JSON.stringify(bbox));
  const { data } = await http.post<Pick<FirstFrameWorm, 'bbox' | 'centroid' | 'keypoints'>>(
    '/extract_manual_worm',
    fd,
  );
  return data;
}

export async function getStatus(taskId: string): Promise<TaskState> {
  const { data } = await http.get<TaskState>(`/task/${taskId}/status`);
  return data;
}

export async function taskAction(
  taskId: string,
  action: 'pause' | 'resume' | 'cancel',
): Promise<{ ok: boolean; task_id: string; status: string }> {
  const { data } = await http.post<{ ok: boolean; task_id: string; status: string }>(
    `/task/${taskId}/${action}`,
  );
  return data;
}

export async function setWormCount(
  taskId: string,
  wormCount: number,
): Promise<{ ok: boolean; task_id: string; worm_count: number | null; message: string }> {
  const { data } = await http.post<{
    ok: boolean;
    task_id: string;
    worm_count: number | null;
    message: string;
  }>(`/task/${taskId}/set_worm_count`, { worm_count: wormCount });
  return data;
}

export async function listSubVideos(taskId: string): Promise<{ videos: SubVideoItem[] }> {
  const { data } = await http.get<{ videos: SubVideoItem[] }>(`/task/${taskId}/sub_videos`);
  return data;
}

export async function getModels(): Promise<{ models: string[]; current_model: string }> {
  const { data } = await http.get<{ models: string[]; current_model: string }>('/models');
  return data;
}

export async function selectModel(
  modelName: string,
): Promise<{ ok: boolean; current_model: string; message: string }> {
  const { data } = await http.post<{ ok: boolean; current_model: string; message: string }>(
    '/select_model',
    { model_name: modelName },
  );
  return data;
}

export async function getProfiles(): Promise<{ profiles: ProfileItem[]; default: string }> {
  const { data } = await http.get<{ profiles: ProfileItem[]; default: string }>('/profiles');
  return data;
}

export async function eggCount(file: File): Promise<EggCountResponse> {
  const fd = new FormData();
  fd.append('file', file);
  const { data } = await http.post<EggCountResponse>('/egg_count', fd);
  return data;
}
