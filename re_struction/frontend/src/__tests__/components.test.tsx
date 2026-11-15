import React, { useState } from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { GlassVideoPlayer } from '../components/GlassVideoPlayer';
import { EggCountResultPanel } from '../components/EggCountResultPanel';
import { HeadMarkModal } from '../components/HeadMarkModal/HeadMarkModal';
import type { WormSelectionsMap } from '../components/HeadMarkModal/drawCanvas';
import type { EggCountResponse, FirstFrameData } from '../api/types';

describe('T29 Component Smoke & Behavior Tests', () => {
  it('renders GlassVideoPlayer and control buttons', () => {
    render(<GlassVideoPlayer title="测试主视频" src="http://localhost:5001/video/1" />);
    expect(screen.getByText('测试主视频')).toBeTruthy();
    expect(screen.getByText('播放/暂停')).toBeTruthy();
    expect(screen.getByText('回看10秒')).toBeTruthy();
  });

  it('renders EggCountResultPanel and filters egg count by custom confidence range', () => {
    const mockResult: EggCountResponse = {
      egg_count: 3,
      egg_count_90: 2,
      egg_count_95: 1,
      egg_class_name: 'egg',
      raw_image: '',
      frame_width: 640,
      frame_height: 480,
      eggs: [
        { bbox: [10, 10, 20, 20], confidence: 0.75, class_id: 0 },
        { bbox: [30, 30, 40, 40], confidence: 0.91, class_id: 0 },
        { bbox: [50, 50, 60, 60], confidence: 0.97, class_id: 0 },
      ],
    };
    const onClear = vi.fn();
    render(<EggCountResultPanel result={mockResult} onClear={onClear} />);
    expect(screen.getByText('虫卵计数结果')).toBeTruthy();
    // Default 80-100% includes 0.91 and 0.97 -> 2 eggs
    expect(screen.getByText(/区间包含虫卵数:\s*2\s*个/)).toBeTruthy();

    const minInput = screen.getByLabelText('min-confidence');
    fireEvent.change(minInput, { target: { value: '95' } });
    expect(screen.getByText(/区间包含虫卵数:\s*1\s*个/)).toBeTruthy();
  });

  it('disables HeadMarkModal confirm button until all enabled worms have head_kpt_idx set', () => {
    const firstFrameData: FirstFrameData = {
      frame_width: 640,
      frame_height: 480,
      annotated: '',
      worms: [
        {
          det_idx: 0,
          confidence: 0.95,
          bbox: [10, 10, 100, 100],
          centroid: [55, 55],
          keypoints: Array.from({ length: 11 }, (_, i) => [10 + i * 8, 50] as [number, number]),
        },
        {
          det_idx: 1,
          confidence: 0.92,
          bbox: [150, 150, 240, 240],
          centroid: [195, 195],
          keypoints: Array.from({ length: 11 }, (_, i) => [150 + i * 8, 195] as [number, number]),
        },
      ],
    };

    function Wrapper() {
      const [sels, setSels] = useState<WormSelectionsMap>({
        0: { enabled: true, head_kpt_idx: 0, user_label: '虫子1' },
        1: { enabled: true, head_kpt_idx: null, user_label: '虫子2' },
      });
      return (
        <HeadMarkModal
          firstFrameData={firstFrameData}
          firstFrameBlob={null}
          wormSelections={sels}
          onSelectionsChange={setSels}
          onConfirm={() => {}}
          onSkip={() => {}}
        />
      );
    }

    render(<Wrapper />);
    const confirmBtn = screen.getByRole('button', { name: /确认开始分析/ }) as HTMLButtonElement;
    // Worm 1 is enabled and has head_kpt_idx = null -> confirm button must be disabled
    expect(confirmBtn.disabled).toBe(true);

    // Click endpoint 0 button for worm 1
    const ep0Btns = screen.getAllByRole('button', { name: '端点0' });
    fireEvent.click(ep0Btns[1]);

    expect(confirmBtn.disabled).toBe(false);
  });
});
