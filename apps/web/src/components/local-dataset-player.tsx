'use client';

import { useQuery } from '@tanstack/react-query';
import { useEffect, useEffectEvent, useId, useState } from 'react';
import { datasetLibrary, sampleImage } from '@/lib/dataset-library';
import type { DatasetJob } from '@/lib/api';
import '@/app/dataset-explorer.css';

export function LocalDatasetPlayer({
  job,
  active,
  selection,
}: {
  job: DatasetJob;
  active: boolean;
  selection?: { keys: string[]; onToggle: (key: string) => void; disabled?: boolean };
}) {
  const libraryId = job.request.library_id;
  const samples = useQuery({
    queryKey: ['local-episode-frames', libraryId, job.result?.metadata_sha256],
    queryFn: ({ signal }) => datasetLibrary.samples(libraryId!, signal),
    enabled: active && !!libraryId,
    staleTime: 30000,
    retry: false,
  });
  const [episode, setEpisode] = useState<number | null>(null);
  const [position, setPosition] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [failed, setFailed] = useState<Record<string, boolean>>({});
  const slider = useId();
  const rows = (samples.data ?? []).filter(
    (row) => !row.metadata_sha256 || row.metadata_sha256 === job.result?.metadata_sha256,
  );
  const episodes = [...new Set(rows.map((row) => row.episode_index))].sort((a, b) => a - b);
  const selected = episode ?? episodes[0];
  const frames = rows.filter((row) => row.episode_index === selected);
  const indices = [...new Set(frames.map((row) => row.frame_index))].sort((a, b) => a - b);
  const cameras = [
    ...new Set([
      ...Object.keys(job.result?.features ?? {}).filter((key) => key.startsWith('observation.image')),
      ...frames.map((row) => row.camera),
    ]),
  ];
  const current = Math.min(position, Math.max(0, indices.length - 1));
  const fps = job.result?.fps ?? 1;
  const sample = frames.find((row) => row.frame_index === indices[current]);
  const time = sample?.timestamp ?? (indices[current] ?? 0) / fps;
  const nextSample = frames.find((row) => row.frame_index === indices[current + 1]);
  const nextTime = nextSample?.timestamp ?? (indices[current + 1] ?? indices[current] + 1) / fps;
  const delayMs = Math.max(1, (nextTime - time) * 1000);
  const pause = useEffectEvent(() => setPlaying(false));
  useEffect(() => {
    if (!active) {
      const timer = window.setTimeout(() => pause(), 0);
      return () => window.clearTimeout(timer);
    }
    if (!playing || !active || indices.length < 2) return;
    const timer = window.setTimeout(() => {
      if (current >= indices.length - 1) setPlaying(false);
      else setPosition(current + 1);
    }, delayMs);
    return () => window.clearTimeout(timer);
  }, [active, playing, current, indices.length, delayMs]);
  useEffect(() => {
    const hidden = () => {
      if (document.hidden) pause();
    };
    document.addEventListener('visibilitychange', hidden);
    return () => document.removeEventListener('visibilitychange', hidden);
  }, []);
  if (samples.isPending) return <p role="status">Loading recorded frames…</p>;
  if (samples.error)
    return (
      <div role="alert">
        <p>{samples.error.message}</p>
        <button onClick={() => void samples.refetch()}>Retry local preview</button>
      </div>
    );
  if (!rows.length) return <p role="status">No saved camera preview frames are available yet.</p>;
  return (
    <div className="dx-camera-player">
      <label>
        Preview episode{' '}
        <select
          aria-label="Preview episode"
          value={selected}
          onChange={(event) => {
            setPlaying(false);
            setPosition(0);
            setEpisode(Number(event.target.value));
          }}
        >
          {episodes.map((value) => (
            <option key={value} value={value}>
              Episode {value + 1}
            </option>
          ))}
        </select>
      </label>
      <div className="dx-camera-grid">
        {cameras.map((key) => {
          const frame = frames.find((row) => row.camera === key && row.frame_index === indices[current]);
          return (
            <figure key={key}>
              {frame && !failed[frame.path] ? (
                <img
                  style={{ width: '100%', aspectRatio: '4 / 3', objectFit: 'contain', borderRadius: 8 }}
                  src={sampleImage(libraryId!, frame.path)}
                  alt={`${key} · episode ${selected + 1} · frame ${frame.frame_index}`}
                  onError={() => setFailed((previous) => ({ ...previous, [frame.path]: true }))}
                />
              ) : (
                <p role="alert">This recorded frame could not be loaded.</p>
              )}
              <figcaption>
                {selection ? (
                  <label>
                    <input
                      type="checkbox"
                      aria-label={key}
                      checked={selection.keys.includes(key)}
                      disabled={selection.disabled}
                      onChange={() => selection.onToggle(key)}
                    />
                    {key.replace(/^observation\.images\./, '')}
                  </label>
                ) : (
                  <strong>{key.replace(/^observation\.images\./, '')}</strong>
                )}
                {' · '}
                {fps} FPS
              </figcaption>
            </figure>
          );
        })}
      </div>
      <div className="dx-playback-controls">
        <button
          type="button"
          aria-label={playing ? 'Pause episode' : 'Play episode'}
          disabled={!active || indices.length < 2}
          onClick={() => {
            if (!playing && current === indices.length - 1) setPosition(0);
            setPlaying(!playing);
          }}
        >
          {playing ? 'Pause' : 'Play'}
        </button>
        <label htmlFor={slider} className="visually-hidden">
          Episode playback position
        </label>
        <input
          id={slider}
          type="range"
          min={0}
          max={Math.max(0, indices.length - 1)}
          value={current}
          onChange={(event) => {
            setPlaying(false);
            setPosition(Number(event.target.value));
          }}
          aria-valuetext={`Frame ${indices[current]} · ${time.toFixed(1)} seconds`}
        />
        <span>
          Frame {indices[current]} · {time.toFixed(1)} s · {indices.length} saved frames
        </span>
      </div>
      {sample && (
        <details>
          <summary>Recorded joint values</summary>
          <p>State: {sample.state?.map((value) => value.toFixed(3)).join(', ') ?? 'Unavailable in this preview'}</p>
          <p>Action: {sample.action?.map((value) => value.toFixed(3)).join(', ') ?? 'Unavailable in this preview'}</p>
        </details>
      )}
      {(episodes.length < (job.result?.total_episodes ?? 0) || rows.some((row) => row.preview_truncated)) && (
        <p>
          Saved preview contains {episodes.length} of {job.result?.total_episodes ?? 0} episodes.
        </p>
      )}
    </div>
  );
}
