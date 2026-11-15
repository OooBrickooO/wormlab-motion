import React, { useEffect, useRef, useState } from 'react';
import { uploadVideo } from './api/client';
import { toAbsoluteUrl } from './api/urls';
import type { CUnit, HeadConfigItem } from './api/types';
import { EggCountResultPanel } from './components/EggCountResultPanel';
import { GlassVideoPlayer } from './components/GlassVideoPlayer';
import { HeadMarkModal } from './components/HeadMarkModal/HeadMarkModal';
import { HistoryPanel } from './components/HistoryPanel';
import { InputPanel, type AnalysisMode } from './components/InputPanel';
import { LiveTrackTable } from './components/LiveTrackTable';
import { ProgressStats } from './components/ProgressStats';
import { SubVideoList } from './components/SubVideoList';
import { SummaryTable } from './components/SummaryTable';
import { useEggCount } from './hooks/useEggCount';
import { useHeadMark } from './hooks/useHeadMark';
import { ACTIVE_FILE_KEY, ACTIVE_TASK_KEY, useHistory } from './hooks/useHistory';
import { useModels } from './hooks/useModels';
import { usePreviewCount } from './hooks/usePreviewCount';
import { useTaskControl } from './hooks/useTaskControl';
import { useTaskPolling } from './hooks/useTaskPolling';
import { useTaskSocket } from './hooks/useTaskSocket';

export default function App() {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [selectedFileNames, setSelectedFileNames] = useState('');
  const [taskId, setTaskId] = useState<string | null>(null);
  const [monitorSeq, setMonitorSeq] = useState(0);
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState('');
  const [wormCount, setWormCount] = useState('');
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>('tracking');
  const [cUnit, setCUnit] = useState<CUnit>('cycle');

  const hist = useHistory();
  const models = useModels();
  const egg = useEggCount(setError);
  const preview = usePreviewCount({ analysisMode, setWormCount });

  const histRef = useRef(hist);
  histRef.current = hist;

  const sock = useTaskSocket({
    taskId,
    onCancelled: (id, msg) => {
      histRef.current.updateHistoryStatus(id, 'cancelled');
      if (id !== taskId) return;
      setStatus('cancelled');
      setError(msg || '任务已取消');
      sessionStorage.removeItem(ACTIVE_TASK_KEY);
      sessionStorage.removeItem(ACTIVE_FILE_KEY);
    },
    onError: (id, msg) => {
      histRef.current.updateHistoryStatus(id, 'failed');
      if (id !== taskId) return;
      setStatus('idle');
      setError(msg || '处理失败');
      sessionStorage.removeItem(ACTIVE_TASK_KEY);
    },
    getExtraJoinIds: () =>
      histRef.current.historyList.filter((i) => i.status === 'processing').map((i) => String(i.id)),
  });

  const poll = useTaskPolling({
    taskId,
    refreshKey: monitorSeq,
    taskStatus: status,
    socketConnected: sock.socketConnected,
    setStatus,
    setError,
    setCurrentFrame: sock.setCurrentFrame,
    setCurrentStats: (fn) => sock.setCurrentStats(fn),
    setTotalFrames: (fn) => sock.setTotalFrames(fn),
    upsertHistoryRecord: hist.upsertHistoryRecord,
    updateHistoryStatus: hist.updateHistoryStatus,
    onWormCount: setWormCount,
  });

  const control = useTaskControl({
    taskId,
    setStatus,
    setError,
    updateHistoryStatus: hist.updateHistoryStatus,
  });

  // 刷新页面后恢复进行中的任务
  useEffect(() => {
    const savedTaskId = sessionStorage.getItem(ACTIVE_TASK_KEY);
    const savedFileName = sessionStorage.getItem(ACTIVE_FILE_KEY);
    if (savedFileName) setSelectedFileNames(savedFileName);
    if (savedTaskId) {
      setTaskId(savedTaskId);
      setStatus('processing');
    }
  }, []);

  const resetOutput = () => {
    sock.resetLive();
    poll.resetPolled();
    preview.resetPreview();
    setError('');
    setWormCount('');
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;
    setSelectedFiles(files);
    setSelectedFileNames(files.map((f) => f.name).join(', '));
    sessionStorage.setItem(ACTIVE_FILE_KEY, files[0].name);
    setStatus('idle');
    resetOutput();
    await preview.runPreview(files[0]);
  };

  const handleModelChange = async (name: string) => {
    const ok = await models.changeModel(name);
    if (!ok) {
      setError('切换模型失败，请稍后重试');
      return;
    }
    if (selectedFiles.length > 0 && status === 'idle' && !preview.previewing) {
      await preview.runPreview(selectedFiles[0]);
    }
  };

  const handleModeSwitch = async (mode: AnalysisMode) => {
    setAnalysisMode(mode);
    setError('');
    egg.clearEggResult();
    if (mode === 'egg_count') await models.ensureEggModel();
  };

  const submitTask = async (headConfig: HeadConfigItem[], count: number) => {
    if (isNaN(count) || count <= 0) {
      setError('请输入有效的初始虫子数量（正整数）');
      return;
    }
    setError('');
    setStatus('uploading');
    let firstId: string | null = null;
    for (const file of selectedFiles) {
      const fd = new FormData();
      fd.append('video', file);
      fd.append('worm_count', String(count));
      if (models.selectedModel) fd.append('model_name', models.selectedModel);
      fd.append('profile', models.selectedProfile);
      fd.append('c_unit', cUnit);
      if (headConfig.length > 0) fd.append('head_config', JSON.stringify(headConfig));
      try {
        const res = await uploadVideo(fd);
        const id = String(res.task_id);
        if (!firstId) firstId = id;
        hist.upsertHistoryRecord({
          id,
          name: file.name || `任务${id}`,
          status: 'processing',
          time: new Date().toLocaleString(),
        });
        sock.join(id);
      } catch (err) {
        console.error(err);
        setError('部分上传失败');
      }
    }
    if (firstId) {
      setTaskId(firstId);
      setStatus('processing');
      sessionStorage.setItem(ACTIVE_TASK_KEY, firstId);
    } else {
      setStatus('idle');
    }
    setSelectedFiles([]);
    setSelectedFileNames('');
  };

  const headMark = useHeadMark({
    firstFrameData: preview.firstFrameData,
    wormSelections: preview.wormSelections,
    wormCount,
    setWormCount,
    setShowHeadMarkModal: preview.setShowHeadMarkModal,
    submit: (cfg, n) => submitTask(cfg, n),
  });

  const startProcess = async () => {
    if (selectedFiles.length === 0) {
      setError('请先选择视频');
      return;
    }
    if (preview.firstFrameData && preview.firstFrameData.worms.length > 0) {
      preview.setShowHeadMarkModal(true);
    } else {
      await submitTask([], parseInt(wormCount, 10));
    }
  };

  const isEgg = analysisMode === 'egg_count';

  return (
    <div className="app-shell">
      <div className="bg-grid" />
      <div className="bg-orb orb-a" />
      <div className="bg-orb orb-b" />

      {preview.showHeadMarkModal && preview.firstFrameData && (
        <HeadMarkModal
          firstFrameData={preview.firstFrameData}
          firstFrameBlob={preview.firstFrameBlob}
          wormSelections={preview.wormSelections}
          onSelectionsChange={preview.setWormSelections}
          onAddManualWorm={(w) => {
            preview.setFirstFrameData((prev) =>
              prev ? { ...prev, worms: [...prev.worms, w] } : prev,
            );
            preview.setWormSelections((prev) => ({
              ...prev,
              [w.det_idx]: { enabled: true, head_kpt_idx: 0, user_label: `虫子${w.det_idx + 1}(手动)` },
            }));
          }}
          onConfirm={headMark.confirm}
          onSkip={headMark.skip}
        />
      )}

      <header className="topbar glass-card">
        <h1>线虫运动识别</h1>
        <p>上传视频，实时查看模型处理结果与线虫子视频。</p>
        <div className={`socket-pill socket-${sock.socketStatus}`}>Socket: {sock.socketStatus}</div>
      </header>

      <main className="layout">
        <InputPanel
          analysisMode={analysisMode}
          onModeSwitch={handleModeSwitch}
          status={status}
          taskId={taskId}
          selectedFiles={selectedFiles}
          selectedFileNames={selectedFileNames}
          onFileChange={handleFileChange}
          models={models.models}
          selectedModel={models.selectedModel}
          modelsLoading={models.modelsLoading}
          onModelChange={handleModelChange}
          profiles={models.profiles}
          selectedProfile={models.selectedProfile}
          onProfileChange={models.setSelectedProfile}
          cUnit={cUnit}
          onCUnitChange={setCUnit}
          wormCount={wormCount}
          onWormCountChange={setWormCount}
          previewing={preview.previewing}
          suggestedCount={preview.suggestedCount}
          previewAnnotated={preview.previewAnnotated}
          onStart={startProcess}
          onPause={control.pause}
          onResume={control.resume}
          onCancelTask={control.cancel}
          onCancelPreview={preview.cancelPreview}
          onEggCount={() => egg.runEggCount(selectedFiles[0])}
          eggCounting={egg.eggCounting}
          error={error}
          videoUrl={poll.videoUrl}
          logUrl={poll.logUrl}
          zipUrl={poll.zipUrl}
          summaryUrl={poll.summaryUrl}
          csvUrl={poll.csvUrl}
          onOpenUrl={(u) => u && window.open(toAbsoluteUrl(u), '_blank')}
        />

        <section className="panel glass-card center-panel">
          <h2>{isEgg ? '虫卵计数结果' : '实时处理区'}</h2>
          {isEgg ? (
            <>
              {egg.eggCounting && (
                <div className="egg-counting-placeholder">
                  <span className="preview-spinner" style={{ width: 22, height: 22, borderWidth: 3 }} />
                  <span style={{ marginTop: 10, color: 'var(--muted)' }}>正在分析第一帧，请稍候...</span>
                </div>
              )}
              {!egg.eggCounting && !egg.eggCountResult && (
                <div className="stream-wrap" style={{ minHeight: 200 }}>
                  <div className="stream-placeholder">选择文件后点击「开始计数」</div>
                </div>
              )}
              <EggCountResultPanel result={egg.eggCountResult} onClear={egg.clearEggResult} />
            </>
          ) : (
            <>
              <ProgressStats
                currentFrame={sock.currentFrame}
                currentStats={sock.currentStats}
                totalFrames={sock.totalFrames}
                phase={sock.phase}
                hasVideoResult={Boolean(poll.videoUrl || poll.subVideos.length > 0)}
              />
              {poll.videoUrl && (
                <GlassVideoPlayer title="处理后主视频" src={poll.videoUrl} downloadName="main_result" />
              )}
              <SubVideoList
                videos={poll.subVideos}
                selected={poll.selectedSubVideo}
                onSelect={poll.setSelectedSubVideo}
              />
              <LiveTrackTable tracks={sock.currentStats?.tracks || []} defaultCUnit={cUnit} />
              <SummaryTable items={poll.idPoseSummary} defaultCUnit={cUnit} />
            </>
          )}
        </section>

        <HistoryPanel
          items={hist.historyList}
          onMonitor={(item) => {
            setTaskId(item.id);
            setStatus(item.status);
            resetOutput();
            setMonitorSeq((n) => n + 1);
          }}
          onOpen={(item, key) => hist.handleHistoryOpen(item, key, setError)}
        />
      </main>
    </div>
  );
}
