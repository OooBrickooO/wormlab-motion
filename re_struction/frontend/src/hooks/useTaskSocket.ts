import { useCallback, useEffect, useRef, useState } from 'react';
import { connectSocket, type SocketController } from '../api/socket';
import type { FrameStats } from '../api/types';

export type SocketStatus = 'connected' | 'disconnected' | 'error';

export interface TaskSocketOptions {
  taskId: string | null;
  onCancelled?: (taskId: string, message: string) => void;
  onError?: (taskId: string, message: string) => void;
  /** 需要在重连后重新 join 的其它任务（如历史中仍在处理的） */
  getExtraJoinIds?: () => string[];
}

export function useTaskSocket(opts: TaskSocketOptions) {
  const [socketStatus, setSocketStatus] = useState<SocketStatus>('disconnected');
  const [currentFrame, setCurrentFrame] = useState<string | null>(null);
  const [currentStats, setCurrentStats] = useState<FrameStats | null>(null);
  const [totalFrames, setTotalFrames] = useState(0);
  const [phase, setPhase] = useState<string>('');
  const [socketConnected, setSocketConnected] = useState(false);

  const taskRef = useRef<string | null>(opts.taskId);
  const optsRef = useRef(opts);
  const ctlRef = useRef<SocketController | null>(null);
  optsRef.current = opts;
  taskRef.current = opts.taskId;

  useEffect(() => {
    const ctl = connectSocket({
      getActiveTaskId: () => taskRef.current,
      onFrame: (data) => {
        if (String(data.task_id) !== String(taskRef.current)) return;
        setCurrentFrame(`data:image/jpeg;base64,${data.image}`);
        setCurrentStats(data.stats);
        setTotalFrames(data.total_frames || 0);
      },
      onPhase: (evt) => {
        if (String(evt.task_id) !== String(taskRef.current)) return;
        setPhase(evt.phase);
      },
      onCancelled: (p) => optsRef.current.onCancelled?.(String(p.task_id), p.message),
      onError: (p) => optsRef.current.onError?.(String(p.task_id), p.message),
    });
    ctlRef.current = ctl;
    ctl.socket.on('connect', () => {
      setSocketStatus('connected');
      setSocketConnected(true);
      optsRef.current.getExtraJoinIds?.().forEach((id) => ctl.join(id));
    });
    ctl.socket.on('disconnect', () => {
      setSocketStatus('disconnected');
      setSocketConnected(false);
    });
    ctl.socket.on('connect_error', () => setSocketStatus('error'));
    return () => ctl.disconnect();
  }, []);

  useEffect(() => {
    if (opts.taskId && ctlRef.current?.socket.connected) ctlRef.current.join(String(opts.taskId));
  }, [opts.taskId]);

  const join = useCallback((id: string) => ctlRef.current?.join(id), []);
  const resetLive = useCallback(() => {
    setCurrentFrame(null);
    setCurrentStats(null);
    setTotalFrames(0);
    setPhase('');
  }, []);

  return {
    socketStatus,
    socketConnected,
    currentFrame,
    setCurrentFrame,
    currentStats,
    setCurrentStats,
    totalFrames,
    setTotalFrames,
    phase,
    join,
    resetLive,
  };
}
