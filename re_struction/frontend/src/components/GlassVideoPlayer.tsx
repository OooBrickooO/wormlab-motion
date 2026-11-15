import React, { useEffect, useRef, useState } from 'react';

export interface GlassVideoPlayerProps {
  title: string;
  src: string;
  downloadName?: string;
}

export function GlassVideoPlayer({ title, src, downloadName }: GlassVideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [mediaError, setMediaError] = useState<string>('');

  useEffect(() => {
    setMediaError('');
  }, [src]);

  const togglePlay = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) {
      void v.play();
    } else {
      v.pause();
    }
  };

  const rewind = () => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = Math.max(0, v.currentTime - 10);
  };

  const capture = () => {
    const v = videoRef.current;
    if (!v || !v.videoWidth) return;
    const canvas = document.createElement('canvas');
    canvas.width = v.videoWidth;
    canvas.height = v.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(v, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${downloadName || 'snapshot'}.png`;
      a.click();
      URL.revokeObjectURL(url);
    }, 'image/png');
  };

  const downloadVideo = async () => {
    if (!src) return;
    try {
      const response = await fetch(src);
      if (!response.ok) return;
      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = objectUrl;
      a.download = `${downloadName || 'result'}.mp4`;
      a.click();
      URL.revokeObjectURL(objectUrl);
    } catch (e) {
      console.error(e);
    }
  };

  const fullScreen = () => {
    const v = videoRef.current;
    if (v?.requestFullscreen) {
      void v.requestFullscreen();
    }
  };

  return (
    <section className="video-block">
      <div className="video-title">{title}</div>
      <video
        key={src}
        ref={videoRef}
        className="video-surface"
        src={src}
        controls
        preload="metadata"
        onLoadedMetadata={() => setMediaError('')}
        onError={() => setMediaError('视频解码失败或无法加载，可点击下方下载按钮保存本地播放。')}
      />
      {mediaError && <div className="status-err" role="alert">{mediaError}</div>}
      <div className="control-row">
        <button type="button" className="glass-btn" onClick={togglePlay}>
          播放/暂停
        </button>
        <button type="button" className="glass-btn" onClick={capture}>
          截图
        </button>
        <button type="button" className="glass-btn" onClick={rewind}>
          回看10秒
        </button>
        <button type="button" className="glass-btn" onClick={downloadVideo}>
          下载
        </button>
        <button type="button" className="glass-btn" onClick={fullScreen}>
          全屏
        </button>
      </div>
    </section>
  );
}
export default GlassVideoPlayer;
