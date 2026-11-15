import { io, Socket } from 'socket.io-client';
import { API_BASE } from './urls';
import type { CompleteEvent, FrameEvent, PhaseEvent } from './types';

export interface SocketHandlers {
  getActiveTaskId?: () => string | null;
  onFrame?: (evt: FrameEvent) => void;
  onComplete?: (evt: CompleteEvent) => void;
  onPhase?: (evt: PhaseEvent) => void;
  onCancelled?: (evt: { task_id: string; message: string }) => void;
  onError?: (evt: { task_id: string; message: string }) => void;
}

export interface SocketController {
  socket: Socket;
  join: (taskId: string) => void;
  disconnect: () => void;
}

export function connectSocket(handlers: SocketHandlers): SocketController {
  const socket = io(API_BASE, {
    transports: ['websocket', 'polling'],
  });

  socket.on('connect', () => {
    const activeId = handlers.getActiveTaskId?.();
    if (activeId) {
      socket.emit('join', { task_id: activeId });
    }
  });

  socket.on('frame', (data: FrameEvent) => {
    handlers.onFrame?.(data);
  });

  socket.on('complete', (data: CompleteEvent) => {
    handlers.onComplete?.(data);
  });

  socket.on('phase', (data: PhaseEvent) => {
    handlers.onPhase?.(data);
  });

  socket.on('cancelled', (data: { task_id: string; message: string }) => {
    handlers.onCancelled?.(data);
  });

  socket.on('error', (data: { task_id: string; message: string }) => {
    handlers.onError?.(data);
  });

  return {
    socket,
    join: (taskId: string) => {
      if (taskId) {
        socket.emit('join', { task_id: taskId });
      }
    },
    disconnect: () => {
      socket.disconnect();
    },
  };
}
