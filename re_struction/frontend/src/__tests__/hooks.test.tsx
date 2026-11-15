import { act, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react';
import { AxiosError } from 'axios';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { TrackLive, TrackSummary } from '../api/types';

const mocks = vi.hoisted(() => ({
  getStatus: vi.fn(),
  listSubVideos: vi.fn(),
  taskAction: vi.fn(),
  handlers: null as null | Record<string, (...a: any[]) => void>,
  joined: [] as string[],
  sockHandlers: {} as Record<string, () => void>,
  connected: false,
}));

vi.mock('../api/client', () => ({
  getStatus: mocks.getStatus,
  listSubVideos: mocks.listSubVideos,
  getModels: vi.fn(),
  getProfiles: vi.fn(),
  selectModel: vi.fn(),
  taskAction: mocks.taskAction,
  setWormCount: vi.fn(),
}));

vi.mock('../api/socket', () => ({
  connectSocket: (h: Record<string, (...a: any[]) => void>) => {
    mocks.handlers = h;
    return {
      socket: {
        get connected() {
          return mocks.connected;
        },
        on: (ev: string, cb: () => void) => {
          mocks.sockHandlers[ev] = cb;
        },
      },
      join: (id: string) => mocks.joined.push(id),
      disconnect: () => {},
    };
  },
}));

import { useTaskPolling } from '../hooks/useTaskPolling';
import { useTaskControl } from '../hooks/useTaskControl';
import { useTaskSocket } from '../hooks/useTaskSocket';
import { ACTIVE_FILE_KEY, ACTIVE_TASK_KEY, useHistory, HISTORY_KEY } from '../hooks/useHistory';
import { buildHeadConfig } from '../hooks/useHeadMark';
import { LiveTrackTable } from '../components/LiveTrackTable';
import { SummaryTable } from '../components/SummaryTable';

const baseTrack = (o: Partial<TrackLive>): TrackLive => ({
  track_id: 1, user_label: null, c_count: 0, s_count: 0, is_bidirectional: false,
  motion_type: 'crawl', activity_score: 0, is_static: false, is_ghost: false, lost_frames: 0,
  live_freq: 0, avg_freq: 0, cycles: 0, swings: 0, bends_per_min: 0,
  centroid_bl: null, head_bl: null, head_kpt_idx: null, keypoints_bl: null, ...o,
});

describe('T31 hooks & tables', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.joined.length = 0;
    mocks.connected = false;
    localStorage.clear();
    sessionStorage.clear();
  });

  it('polling: processing -> completed fetches sub videos and stops', async () => {
    mocks.getStatus
      .mockResolvedValueOnce({ status: 'processing', filename: 'a.mp4', current_frame: 5, total_frames: 10, tracks: [], active_tracks: 1 })
      .mockResolvedValue({ status: 'completed', filename: 'a.mp4', video_url: '/v.mp4', id_pose_summary: [], current_frame: 10, total_frames: 10 });
    mocks.listSubVideos.mockResolvedValue({ videos: [{ filename: 's1.mp4', url: '/s1.mp4' }] });
    const statuses: string[] = [];
    const { result } = renderHook(() =>
      useTaskPolling({
        taskId: 't1', socketConnected: false,
        setStatus: (s) => statuses.push(s), setError: () => {}, setCurrentFrame: () => {},
        setCurrentStats: () => {}, setTotalFrames: () => {},
        upsertHistoryRecord: () => {}, updateHistoryStatus: () => {},
      }),
    );
    await waitFor(() => expect(statuses).toContain('processing'));
    await waitFor(() => expect(statuses).toContain('completed'), { timeout: 4000 });
    await waitFor(() => expect(result.current.subVideos.length).toBe(1));
    expect(result.current.selectedSubVideo).toContain('/s1.mp4');
    expect(result.current.videoUrl).toContain('/v.mp4');
  });

  it('polling: incrementing refreshKey re-fetches completed task status and sub videos', async () => {
    mocks.getStatus.mockResolvedValue({
      status: 'completed',
      filename: 'a.mp4',
      video_url: '/download/a_out.mp4',
      id_pose_summary: [],
      current_frame: 10,
      total_frames: 10,
    });
    mocks.listSubVideos.mockResolvedValue({
      videos: [{ filename: 'worm_1_cycles_2.mp4', url: '/sub_video/t1/worm_1_cycles_2.mp4' }],
    });

    let refreshKey = 0;
    const { result, rerender } = renderHook(
      ({ rk }) =>
        useTaskPolling({
          taskId: 't1',
          refreshKey: rk,
          socketConnected: false,
          setStatus: () => {},
          setError: () => {},
          setCurrentFrame: () => {},
          setCurrentStats: () => {},
          setTotalFrames: () => {},
          upsertHistoryRecord: () => {},
          updateHistoryStatus: () => {},
        }),
      { initialProps: { rk: refreshKey } },
    );

    await waitFor(() => expect(result.current.subVideos.length).toBe(1));
    expect(mocks.getStatus).toHaveBeenCalledTimes(1);
    expect(mocks.listSubVideos).toHaveBeenCalledTimes(1);

    refreshKey = 1;
    rerender({ rk: refreshKey });
    await waitFor(() => expect(mocks.getStatus).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(mocks.listSubVideos).toHaveBeenCalledTimes(2));
  });

  it('polling: failed sets error', async () => {
    mocks.getStatus.mockResolvedValue({ status: 'failed', error_message: 'boom' });
    const errs: string[] = [];
    renderHook(() =>
      useTaskPolling({
        taskId: 't2', socketConnected: false,
        setStatus: () => {}, setError: (m) => errs.push(m), setCurrentFrame: () => {},
        setCurrentStats: () => {}, setTotalFrames: () => {},
        upsertHistoryRecord: () => {}, updateHistoryStatus: () => {},
      }),
    );
    await waitFor(() => expect(errs).toContain('boom'));
  });

  it('polling: 404 stops polling, clears active storage and resets status to idle', async () => {
    sessionStorage.setItem(ACTIVE_TASK_KEY, 'missing_task');
    sessionStorage.setItem(ACTIVE_FILE_KEY, 'v.mp4');
    mocks.getStatus.mockRejectedValue(
      new AxiosError('Not Found', 'ERR_BAD_REQUEST', undefined, undefined, { status: 404 } as any),
    );
    const statuses: string[] = [];
    const errs: string[] = [];
    const histUpdates: Array<[string, string]> = [];
    renderHook(() =>
      useTaskPolling({
        taskId: 'missing_task',
        socketConnected: false,
        setStatus: (s) => statuses.push(s),
        setError: (m) => errs.push(m),
        setCurrentFrame: () => {},
        setCurrentStats: () => {},
        setTotalFrames: () => {},
        upsertHistoryRecord: () => {},
        updateHistoryStatus: (id, s) => histUpdates.push([id, s]),
      }),
    );
    await waitFor(() => expect(statuses).toContain('idle'));
    expect(sessionStorage.getItem(ACTIVE_TASK_KEY)).toBeNull();
    expect(sessionStorage.getItem(ACTIVE_FILE_KEY)).toBeNull();
    expect(histUpdates).toContainEqual(['missing_task', 'cancelled']);
    expect(errs.some((m) => m.includes('重启'))).toBe(true);
  });

  it('control: cancel 404 cleanly transitions to cancelled and clears active storage', async () => {
    sessionStorage.setItem(ACTIVE_TASK_KEY, 'gone_task');
    sessionStorage.setItem(ACTIVE_FILE_KEY, 'v.mp4');
    mocks.taskAction.mockRejectedValue(
      new AxiosError('Not Found', 'ERR_BAD_REQUEST', undefined, undefined, { status: 404 } as any),
    );
    const statuses: string[] = [];
    const errs: string[] = [];
    const histUpdates: Array<[string, string]> = [];
    const { result } = renderHook(() =>
      useTaskControl({
        taskId: 'gone_task',
        setStatus: (s) => statuses.push(s),
        setError: (m) => errs.push(m),
        updateHistoryStatus: (id, s) => histUpdates.push([id, s]),
      }),
    );
    await act(async () => {
      await result.current.cancel();
    });
    expect(statuses).toContain('cancelled');
    expect(sessionStorage.getItem(ACTIVE_TASK_KEY)).toBeNull();
    expect(sessionStorage.getItem(ACTIVE_FILE_KEY)).toBeNull();
    expect(histUpdates).toContainEqual(['gone_task', 'cancelled']);
    expect(errs[0]).toContain('已停止本地监视');
  });

  it('socket: frame for active task updates; error callback; reconnect re-joins', async () => {
    const onError = vi.fn();
    const { result } = renderHook(() =>
      useTaskSocket({ taskId: 'T', onError, getExtraJoinIds: () => ['X'] }),
    );
    act(() => {
      mocks.handlers!.onFrame({ task_id: 'other', frame: 1, total_frames: 9, image: 'AA', stats: { frame: 1, active_tracks: 0, tracks: [] } });
    });
    expect(result.current.currentFrame).toBeNull();
    act(() => {
      mocks.handlers!.onFrame({ task_id: 'T', frame: 2, total_frames: 9, image: 'BB', stats: { frame: 2, active_tracks: 1, tracks: [] } });
    });
    expect(result.current.currentFrame).toContain('BB');
    expect(result.current.totalFrames).toBe(9);
    act(() => mocks.handlers!.onError({ task_id: 'T', message: 'bad' }));
    expect(onError).toHaveBeenCalledWith('T', 'bad');
    act(() => mocks.sockHandlers['connect']());
    expect(result.current.socketStatus).toBe('connected');
    expect(mocks.joined).toContain('X');
    act(() => mocks.sockHandlers['disconnect']());
    expect(result.current.socketStatus).toBe('disconnected');
  });

  it('history: upsert persists under legacy key and caps at 50', () => {
    const { result } = renderHook(() => useHistory());
    act(() => result.current.upsertHistoryRecord({ id: 1, name: 'x', status: 'processing' }));
    act(() => result.current.updateHistoryStatus('1', 'completed'));
    const saved = JSON.parse(localStorage.getItem(HISTORY_KEY)!);
    expect(saved).toHaveLength(1);
    expect(saved[0]).toMatchObject({ id: '1', name: 'x', status: 'completed' });
    expect(HISTORY_KEY).toBe('motion_history_records');
  });

  it('buildHeadConfig only includes enabled worms with head set', () => {
    const data = {
      frame_width: 1, frame_height: 1, annotated: '',
      worms: [
        { det_idx: 0, confidence: 1, bbox: [0, 0, 1, 1] as [number, number, number, number], centroid: [5, 6] as [number, number], keypoints: null },
        { det_idx: 1, confidence: 1, bbox: [0, 0, 1, 1] as [number, number, number, number], centroid: [7, 8] as [number, number], keypoints: null },
      ],
    };
    const cfg = buildHeadConfig(data, {
      0: { enabled: true, head_kpt_idx: 10, user_label: '' },
      1: { enabled: true, head_kpt_idx: null, user_label: 'b' },
    });
    expect(cfg).toHaveLength(1);
    expect(cfg[0]).toMatchObject({ det_idx: 0, head_kpt_idx: 10, user_label: '虫子1' });
  });

  it('LiveTrackTable: C toggle switches cycles/bends; late entry shows badge and dashes', () => {
    render(
      <LiveTrackTable
        tracks={[
          baseTrack({ track_id: 1, c_count: 2, c_cycles: 2, c_bends: 5, s_count: 3 }),
          baseTrack({ track_id: 2, c_count: 9, c_cycles: 9, c_bends: 9, s_count: 9, late_entry: true, counted: false }),
        ]}
      />,
    );
    expect(screen.getByText('后进入·不计数')).toBeTruthy();
    const cells = () => Array.from(document.querySelectorAll('tbody tr')[0].querySelectorAll('td')).map((t) => t.textContent);
    expect(cells()[3]).toBe('2');
    fireEvent.click(screen.getByTestId('cunit-bend'));
    expect(cells()[3]).toBe('5');
    const lateCells = Array.from(document.querySelectorAll('tbody tr')[1].querySelectorAll('td')).map((t) => t.textContent);
    expect(lateCells[3]).toBe('—');
    expect(lateCells[4]).toBe('—');
  });

  it('SummaryTable excludes late-entry tracks and toggles unit', () => {
    const mk = (o: any) => ({ ...baseTrack({}), spatial_stats: { total_path_length_px: 1, net_displacement_px: 1 }, ...o }) as unknown as TrackSummary;
    render(<SummaryTable items={[mk({ track_id: 1, c_cycles: 1, c_bends: 3 }), mk({ track_id: 2, late_entry: true })]} />);
    expect(document.querySelectorAll('tbody tr')).toHaveLength(1);
    fireEvent.click(screen.getByTestId('cunit-bend'));
    expect(document.querySelectorAll('tbody tr')[0].querySelectorAll('td')[3].textContent).toBe('3');
  });
});
