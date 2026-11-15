import React from 'react';
import type { SubVideoItem } from '../api/types';
import { GlassVideoPlayer } from './GlassVideoPlayer';

export interface SubVideoListProps {
  videos: SubVideoItem[];
  selected: string;
  onSelect: (url: string) => void;
}

export function SubVideoList({ videos, selected, onSelect }: SubVideoListProps) {
  if (videos.length === 0) return null;
  return (
    <section className="sub-video-section">
      <div className="sub-video-header">
        <h3>线虫子视频</h3>
        <select value={selected} onChange={(e) => onSelect(e.target.value)} className="glass-select">
          {videos.map((v) => (
            <option key={v.filename} value={v.url}>
              {v.filename}
            </option>
          ))}
        </select>
      </div>
      {selected && <GlassVideoPlayer title="当前线虫局部子视频" src={selected} downloadName="worm_sub" />}
    </section>
  );
}
export default SubVideoList;
