'use client';

import { useQuery } from '@tanstack/react-query';
import { useEffect, useId, useRef, useState } from 'react';
import { api, apiMediaUrl, type CameraPreview, type EpisodePreview, type FrameSample, type DatasetJob } from '@/lib/api';
import '@/app/dataset-explorer.css';

const pageSize = 6;

function ExplorerIcon({ kind }: { kind: 'camera' | 'play' | 'pause' | 'left' | 'right' }) {
  return <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {kind === 'camera' && <><rect x="3" y="6" width="13" height="12" rx="2" /><path d="m16 10 5-3v10l-5-3" /></>}
    {kind === 'play' && <path d="m8 4 12 8-12 8V4Z" />}
    {kind === 'pause' && <><path d="M8 5v14M16 5v14" /></>}
    {kind === 'left' && <path d="m15 5-7 7 7 7" />}
    {kind === 'right' && <path d="m9 5 7 7-7 7" />}
  </svg>;
}

function timeLabel(value: number) {
  const seconds = Math.max(0, Number.isFinite(value) ? value : 0);
  return `${Math.floor(seconds / 60)}:${(seconds % 60).toFixed(1).padStart(4, '0')}`;
}

function numberLabel(value: number | undefined) {
  if (value === undefined || !Number.isFinite(value)) return '—';
  if (value !== 0 && (Math.abs(value) < .001 || Math.abs(value) >= 1_000_000)) return value.toExponential(3);
  return value.toLocaleString('en-US', { maximumFractionDigits: 4 });
}

export function FeatureChips({ features }: { features: Record<string, unknown> }) {
  return <details className="dx-schema"><summary>Data structure</summary><div className="dx-features">{Object.entries(features).map(([key, value]) => {
    const feature = value && typeof value === 'object' ? value as { dtype?: unknown; shape?: unknown } : {};
    const shape = Array.isArray(feature.shape) ? feature.shape.filter(item => typeof item === 'number').join(' × ') : '';
    return <div className="dx-feature" key={key}><strong>{key}</strong><span>{shape ? `[${shape}]` : 'Shape not declared'}<span className="dx-feature-type">{typeof feature.dtype === 'string' ? feature.dtype : 'Unknown type'}</span></span></div>;
  })}</div></details>;
}

function PreviewWarnings({ warnings }: { warnings: string[] }) {
  return warnings.length ? <details className="dx-warnings"><summary>Preview notes <span>{warnings.length}</span></summary><ul>{warnings.map((warning, index) => <li key={`${index}-${warning}`}>{warning}</li>)}</ul></details> : null;
}

type CameraState = { status: 'loading' | 'ready' | 'error'; message?: string };

export function CameraPlayer({ preview, active, selection }: {
  preview: EpisodePreview;
  active: boolean;
  selection?: { keys: string[]; onToggle: (key: string) => void; disabled?: boolean };
}) {
  const sliderId = useId();
  const regionRef = useRef<HTMLDivElement>(null);
  const videos = useRef<Record<string, HTMLVideoElement | null>>({});
  const playbackGeneration = useRef(0);
  const playingRef = useRef(false);
  const stateRef = useRef<Record<string, CameraState>>({});
  const positionRef = useRef(0);
  const [position, setPosition] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [cameraStates, setCameraStates] = useState<Record<string, CameraState>>({});
  const [playbackMessage, setPlaybackMessage] = useState('');
  const cameras = preview.cameras;
  const duration = Math.max(0, preview.duration_seconds);
  const available = cameras.filter(camera => cameraStates[camera.key]?.status === 'ready');
  const loading = cameras.some(camera => !cameraStates[camera.key] || cameraStates[camera.key].status === 'loading');

  function markCamera(key: string, state: CameraState) {
    stateRef.current[key] = state;
    setCameraStates(previous => ({ ...previous, [key]: state }));
  }

  function stop() {
    playingRef.current = false;
    playbackGeneration.current += 1;
    Object.values(videos.current).forEach(video => video?.pause());
    setPlaying(false);
  }

  function cameraBoundary(camera: CameraPreview, video: HTMLVideoElement) {
    return Math.min(camera.end_seconds, camera.start_seconds + duration, Number.isFinite(video.duration) ? video.duration : Infinity);
  }

  function cameraEnd(camera: CameraPreview, video: HTMLVideoElement) {
    // The source interval is half-open. Its end may be the following episode's
    // first frame, so displayed media must stop one frame before that boundary.
    return Math.max(camera.start_seconds, cameraBoundary(camera, video) - 1 / camera.fps);
  }

  function finishEpisode(camera: CameraPreview, video: HTMLVideoElement) {
    stop();
    cameras.forEach(item => {
      const element = videos.current[item.key];
      if (element && element.readyState >= 1 && stateRef.current[item.key]?.status !== 'error') element.currentTime = cameraEnd(item, element);
    });
    const end = Math.max(0, cameraBoundary(camera, video) - camera.start_seconds);
    positionRef.current = end;
    setPosition(end);
  }

  function seek(value: number) {
    stop();
    const next = Math.max(0, Math.min(value, duration));
    positionRef.current = next;
    setPosition(next);
    cameras.forEach(camera => {
      const video = videos.current[camera.key];
      if (!video || video.readyState < 1 || cameraStates[camera.key]?.status === 'error') return;
      video.currentTime = Math.max(camera.start_seconds, Math.min(camera.start_seconds + next, cameraEnd(camera, video)));
    });
  }

  async function resumeReadyVideos() {
    if (!playingRef.current || cameras.some(camera => !stateRef.current[camera.key] || stateRef.current[camera.key].status === 'loading')) return;
    const generation = playbackGeneration.current;
    try {
      await Promise.all(cameras.filter(camera => stateRef.current[camera.key]?.status === 'ready').map(camera => videos.current[camera.key]?.play()));
      if (generation === playbackGeneration.current) setPlaybackMessage('');
    } catch {
      // A buffering camera pauses the group; its canplay event resumes the
      // user's existing playback request once every camera is ready again.
      if (generation !== playbackGeneration.current || cameras.some(camera => stateRef.current[camera.key]?.status === 'loading')) return;
      stop();
      setPlaybackMessage('Playback could not start. Wait for the video to load, then try Play again.');
    }
  }

  function play() {
    if (playingRef.current) { stop(); return; }
    const firstCamera = available[0];
    const firstVideo = firstCamera ? videos.current[firstCamera.key] : null;
    if (positionRef.current >= duration - .03 || (firstCamera && firstVideo && firstVideo.currentTime >= cameraEnd(firstCamera, firstVideo) - .03)) seek(0);
    playbackGeneration.current += 1;
    playingRef.current = true;
    setPlaying(true);
    setPlaybackMessage('');
    void resumeReadyVideos();
  }

  function ready(camera: CameraPreview) {
    if (stateRef.current[camera.key]?.status === 'error') return;
    markCamera(camera.key, { status: 'ready' });
    void resumeReadyVideos();
  }

  useEffect(() => { if (!active) stop(); }, [active]);

  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    let lastPaint = 0;
    const update = (now: number) => {
      if (cameras.some(camera => stateRef.current[camera.key]?.status === 'loading')) {
        frame = requestAnimationFrame(update);
        return;
      }
      const masterCamera = cameras.find(camera => cameraStates[camera.key]?.status === 'ready');
      const master = masterCamera ? videos.current[masterCamera.key] : null;
      if (master && masterCamera) {
        const next = Math.max(0, Math.min(master.currentTime - masterCamera.start_seconds, duration));
        positionRef.current = next;
        cameras.forEach(camera => {
          const video = videos.current[camera.key];
          if (!video || cameraStates[camera.key]?.status !== 'ready') return;
          const end = cameraEnd(camera, video);
          if (video.currentTime >= end) { video.pause(); video.currentTime = end; }
          else if (camera.key !== masterCamera.key && Math.abs(video.currentTime - camera.start_seconds - next) > .2) {
            video.currentTime = Math.min(camera.start_seconds + next, end);
          }
        });
        if (master.currentTime >= cameraEnd(masterCamera, master) || next >= duration - .015) {
          finishEpisode(masterCamera, master);
          return;
        }
        if (now - lastPaint > 100) { setPosition(next); lastPaint = now; }
      }
      frame = requestAnimationFrame(update);
    };
    frame = requestAnimationFrame(update);
    return () => cancelAnimationFrame(frame);
  }, [playing, cameras, cameraStates, duration]);

  useEffect(() => {
    const elements = Object.values(videos.current);
    const pause = () => {
      playingRef.current = false;
      playbackGeneration.current += 1;
      elements.forEach(video => video?.pause());
      setPlaying(false);
    };
    const onVisibility = () => { if (document.hidden) pause(); };
    document.addEventListener('visibilitychange', onVisibility);
    const observer = new IntersectionObserver(entries => { if (!entries[0]?.isIntersecting) pause(); });
    if (regionRef.current) observer.observe(regionRef.current);
    return () => {
      document.removeEventListener('visibilitychange', onVisibility);
      observer.disconnect();
      playingRef.current = false;
      playbackGeneration.current += 1;
      elements.forEach(video => video?.pause());
    };
  }, []);

  function retry(camera: CameraPreview) {
    stop();
    markCamera(camera.key, { status: 'loading' });
    videos.current[camera.key]?.load();
  }

  return <div className="dx-player" ref={regionRef}>
    <div className={`dx-camera-grid${cameras.length === 1 ? ' dx-camera-grid-single' : ''}`}>{cameras.map(camera => {
      const state = cameraStates[camera.key] ?? { status: 'loading' };
      return <figure className={`dx-camera${selection?.keys.includes(camera.key) ? ' dx-camera-selected' : ''}`} key={camera.key}>
        <div className="dx-video-wrap" style={{ aspectRatio: camera.width && camera.width > 0 && camera.height && camera.height > 0 ? `${camera.width} / ${camera.height}` : '4 / 3' }}>
          <video ref={element => { videos.current[camera.key] = element; }} src={apiMediaUrl(camera.url)} preload="metadata" muted playsInline controls={false} aria-label={`${camera.key}, episode ${preview.episode_index}. Use the shared playback controls below.`}
            onLoadedMetadata={event => {
              const video = event.currentTarget;
              if (Number.isFinite(video.duration) && video.duration <= camera.start_seconds) {
                markCamera(camera.key, { status: 'error', message: 'This video does not contain the requested episode segment.' });
                return;
              }
              video.currentTime = Math.min(camera.start_seconds + positionRef.current, cameraEnd(camera, video));
            }}
            onLoadedData={() => ready(camera)}
            onCanPlay={() => ready(camera)}
            onWaiting={() => {
              if (stateRef.current[camera.key]?.status === 'error') return;
              markCamera(camera.key, { status: 'loading' });
              if (playingRef.current) {
                Object.values(videos.current).forEach(video => video?.pause());
                setPlaybackMessage('Buffering camera views…');
              }
            }}
            onError={() => { stop(); markCamera(camera.key, { status: 'error', message: 'The source video could not be loaded or its format is not supported by this browser.' }); }}
            onEnded={event => finishEpisode(camera, event.currentTarget)}
          />
          {state.status === 'loading' && <div className="dx-video-status" role="status"><span className="dx-loading-dot" />Loading source video…</div>}
          {state.status === 'error' && <div className="dx-video-error" role="alert"><ExplorerIcon kind="camera" /><p>{state.message}</p><button type="button" onClick={() => retry(camera)}>Retry video</button></div>}
        </div>
        <figcaption>{selection ? <label className="dx-camera-choice"><input type="checkbox" aria-label={camera.key} checked={selection.keys.includes(camera.key)} disabled={selection.disabled} onChange={() => selection.onToggle(camera.key)} /><strong title={camera.key}>{camera.key.replace(/^observation\.images\./, '')}</strong></label> : <strong title={camera.key}>{camera.key.replace(/^observation\.images\./, '')}</strong>}<span>{camera.width && camera.height ? `${camera.width} × ${camera.height} · ` : ''}{camera.fps} FPS</span></figcaption>
      </figure>;
    })}</div>
    {cameras.length ? <>
      <div className="dx-playback-controls">
        <button className="dx-play-button" type="button" onClick={play} disabled={!active || (!playing && (!available.length || loading || !duration))} aria-label={playing ? 'Pause episode' : 'Play episode'}><ExplorerIcon kind={playing ? 'pause' : 'play'} /><span>{playing ? 'Pause' : 'Play'}</span></button>
        <div className="dx-seek"><label htmlFor={sliderId} className="visually-hidden">Episode playback position</label><input id={sliderId} type="range" min="0" max={duration} step="0.01" value={Math.min(position, duration)} onChange={event => seek(Number(event.target.value))} disabled={!available.length || !duration} aria-valuetext={`${position.toFixed(1)} of ${duration.toFixed(1)} seconds`} /></div>
        <span className="dx-playback-time">{timeLabel(position)} <span>/ {timeLabel(duration)}</span></span>
      </div>
      {playbackMessage && <p className="dx-player-note" role="status">{playbackMessage}</p>}
    </> : <div className="dx-no-media"><ExplorerIcon kind="camera" /><h4>No camera preview available</h4></div>}
  </div>;
}

function SampleTable({ samples, names, field }: { samples: FrameSample[]; names: string[]; field: 'action' | 'state' }) {
  const columns = Math.max(names.length, ...samples.map(sample => sample[field]?.length ?? 0));
  if (!columns || !samples.some(sample => sample[field]?.length)) return <p className="dx-empty-samples">No {field} samples were returned for this episode.</p>;
  return <div className="dx-table-scroll" tabIndex={0} role="region" aria-label={`${field === 'state' ? 'Joint and state' : 'Action'} sample values; scroll horizontally for more columns`}><table className="dx-sample-table">
    <caption className="visually-hidden">First returned {field} samples from the selected episode</caption>
    <thead><tr><th scope="col">Frame</th><th scope="col">Time (s)</th>{Array.from({ length: columns }, (_, index) => <th scope="col" key={index}>{names[index] || `Dimension ${index + 1}`}</th>)}</tr></thead>
    <tbody>{samples.map((sample, index) => <tr key={`${sample.frame_index}-${index}`}><th scope="row">{sample.frame_index}</th><td>{numberLabel(sample.timestamp)}</td>{Array.from({ length: columns }, (_, column) => <td key={column}>{numberLabel(sample[field]?.[column])}</td>)}</tr>)}</tbody>
  </table></div>;
}

function EpisodeDetail({ preview, active }: { preview: EpisodePreview; active: boolean }) {
  const [sampleView, setSampleView] = useState<'state' | 'action'>('state');
  const tableId = useId();
  return <div className="dx-episode-detail">
    <div className="dx-episode-heading"><div><h3>{preview.tasks.length ? preview.tasks.join(' · ') : `Episode ${preview.episode_index}`}</h3></div><div className="dx-episode-facts"><span>{preview.frame_count.toLocaleString()} frames</span><span>{timeLabel(preview.duration_seconds)}</span><span>{preview.cameras.length} {preview.cameras.length === 1 ? 'camera' : 'cameras'}</span></div></div>
    <CameraPlayer key={`${preview.revision}-${preview.episode_index}`} preview={preview} active={active} />
    <details className="dx-samples"><summary>Frame samples <span>{preview.samples.length} rows</span></summary><div className="dx-samples-heading"><div><h3>First frames</h3></div><div className="dx-sample-switch" role="group" aria-label="Sample values"><button type="button" aria-pressed={sampleView === 'state'} aria-controls={tableId} onClick={() => setSampleView('state')}>Joint / state</button><button type="button" aria-pressed={sampleView === 'action'} aria-controls={tableId} onClick={() => setSampleView('action')}>Action</button></div></div><div id={tableId}><SampleTable samples={preview.samples} names={sampleView === 'state' ? preview.state_names : preview.action_names} field={sampleView} /></div></details>
    <PreviewWarnings warnings={preview.warnings} />
  </div>;
}

function Explorer({ job, active }: { job: DatasetJob; active: boolean }) {
  const headingId = useId();
  const [offset, setOffset] = useState(0);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const profile = job.result;
  const canPreview = job.status === 'succeeded' && profile?.source === 'huggingface';
  const episodes = useQuery({ queryKey: ['dataset-episodes', job.id, offset], queryFn: () => api.episodes(job.id, offset, pageSize), enabled: active && canPreview, staleTime: Infinity, retry: false });
  const selected = selectedIndex ?? episodes.data?.episodes[0]?.episode_index;
  const preview = useQuery({ queryKey: ['dataset-episode', job.id, selected], queryFn: () => api.episode(job.id, selected!), enabled: active && canPreview && selected !== undefined, staleTime: Infinity, retry: false });

  if (!profile) return null;
  function changePage(nextOffset: number) { setSelectedIndex(null); setOffset(nextOffset); }

  return <section className="dataset-explorer" aria-labelledby={headingId}>
    <div className="dx-heading"><h2 id={headingId}>Cameras</h2></div>
    {!canPreview && <p className="dx-request-status">Camera previews are unavailable for local datasets.</p>}
    {canPreview && <>
      {episodes.isPending && <p className="dx-request-status" role="status">Loading episode index…</p>}
      {episodes.error && <div className="dx-request-error" role="alert"><p>{episodes.error.message}</p><button type="button" onClick={() => void episodes.refetch()} disabled={episodes.isFetching}>{episodes.isFetching ? 'Retrying…' : 'Retry episode index'}</button></div>}
      {episodes.data && <>
        <div className="dx-browser"><aside className="dx-episode-list" aria-label="Dataset episodes"><div className="dx-list-heading"><h3>Episodes</h3><span>{episodes.data.total_episodes.toLocaleString()}</span></div>
          <ul>{episodes.data.episodes.map(episode => <li key={episode.episode_index}><button type="button" className={selected === episode.episode_index ? 'selected' : ''} aria-current={selected === episode.episode_index ? 'true' : undefined} onClick={() => setSelectedIndex(episode.episode_index)}><span className="dx-episode-list-copy"><strong>Episode {episode.episode_index}</strong></span></button></li>)}</ul>
          {!episodes.data.episodes.length && <p className="dx-empty-list">No episodes were returned.</p>}
          <div className="dx-pagination"><button type="button" aria-label="Previous episodes" disabled={offset === 0 || episodes.isFetching} onClick={() => changePage(Math.max(0, offset - pageSize))}><ExplorerIcon kind="left" /></button><span>{episodes.data.episodes.length ? `${offset + 1}–${offset + episodes.data.episodes.length}` : '0'} of {episodes.data.total_episodes.toLocaleString()}</span><button type="button" aria-label="Next episodes" disabled={!episodes.data.episodes.length || offset + episodes.data.episodes.length >= episodes.data.total_episodes || episodes.isFetching} onClick={() => changePage(offset + pageSize)}><ExplorerIcon kind="right" /></button></div>
        </aside><div className="dx-preview-content" aria-busy={preview.isFetching}>
          {preview.isPending && selected !== undefined && <div className="dx-preview-loading" role="status"><ExplorerIcon kind="camera" /><h3>Opening episode {selected}</h3></div>}
          {preview.error && <div className="dx-request-error" role="alert"><p>{preview.error.message}</p><button type="button" onClick={() => void preview.refetch()} disabled={preview.isFetching}>{preview.isFetching ? 'Retrying…' : 'Retry episode preview'}</button></div>}
          {preview.data && <EpisodeDetail key={`${job.id}-${preview.data.episode_index}`} preview={preview.data} active={active} />}
        </div></div>
        <PreviewWarnings warnings={episodes.data.warnings} />
      </>}
    </>}

  </section>;
}

export function DatasetExplorer({ job, active = true }: { job: DatasetJob; active?: boolean }) {
  return <Explorer key={job.id} job={job} active={active} />;
}
