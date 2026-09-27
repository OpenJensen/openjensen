'use client';

import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api, augmentationClipUrl, augmentationDownloadUrl, isActive, isAugmentationJob, isDatasetJob, type AugmentationJob, type AugmentationRequest, type DatasetJob, type DatasetProfile, type Job } from '@/lib/api';
import { Icon } from '@/components/icon';
import './augmentation-panel.css';

type InspectedDataset = DatasetJob & { result: DatasetProfile };
type Preset = 'lighting' | 'texture' | 'custom';
const presetDetails = {
  lighting: { label: 'Change lighting', icon: 'spark' },
  texture: { label: 'Change textures', icon: 'layers' },
  custom: { label: 'Custom edit', icon: 'sliders' },
} as const;

function videoCameras(profile?: DatasetProfile) {
  return Object.entries(profile?.features ?? {}).filter(([, feature]) =>
    feature && typeof feature === 'object' && 'dtype' in feature && feature.dtype === 'video',
  ).map(([key]) => key);
}

function dateLabel(timestamp: string) {
  return new Date(timestamp).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
}

function RunStatus({ job }: { job: AugmentationJob }) {
  return <span className={`status status-${job.status}`}><span className="status-dot" />{job.status}</span>;
}

function AugmentationRun({ job, projectId }: { job: AugmentationJob; projectId: string }) {
  const client = useQueryClient();
  const [clipIndex, setClipIndex] = useState(0);
  const [mediaErrors, setMediaErrors] = useState<Record<string, boolean>>({});
  const cancel = useMutation({
    mutationFn: () => api.cancel(job.id),
    onSuccess: (updated) => {
      client.setQueryData<Job[]>(['jobs', projectId], previous => previous?.map(item => item.id === updated.id ? updated : item));
      void client.invalidateQueries({ queryKey: ['jobs', projectId] });
    },
  });
  const result = job.status === 'succeeded' ? job.result : null;
  const clip = result?.clips.find(item => item.index === clipIndex) ?? result?.clips[0];
  return <div className="augmentation-run">
    <div className="augmentation-run-heading"><div><h3>{presetDetails[job.request.preset as Preset]?.label ?? 'Dataset augmentation'}</h3><p>{dateLabel(job.created_at)} · {job.request.episode_indices.length} clip{job.request.episode_indices.length === 1 ? '' : 's'}</p></div><RunStatus job={job} /></div>
    {isActive(job) && <div className="augmentation-progress" role="status"><span className="spinner" /><div><strong>{job.status === 'queued' ? 'Waiting to augment' : 'Creating visual variations'}</strong><p>{job.stage ? job.stage.replaceAll('_', ' ') : 'Preparing clips…'}</p></div><button type="button" className="secondary-button" disabled={cancel.isPending} onClick={() => cancel.mutate()}>{cancel.isPending ? 'Cancelling…' : 'Cancel augmentation'}</button></div>}
    {cancel.error && <p className="error-notice" role="alert">{cancel.error.message}</p>}
    {job.error && <p className="error-notice" role="alert">{job.error}</p>}
    {(job.status === 'cancelled' || job.status === 'interrupted') && <p className="augmentation-hint">Run did not complete.</p>}
    {result && <>
      <div className="augmentation-review"><Icon name="check" size={17} /><div><p>Generated clips may not match the original action labels. Review motion and timing, and validate labels before training.</p></div></div>
      {!!result.warnings.length && <ul className="augmentation-warnings">{result.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul>}
      {clip && <>
        <div className="augmentation-preview-heading"><div><h4>Compare the result</h4><p>{clip.camera_key} · {clip.source_start_seconds}s–{clip.source_end_seconds}s</p></div>{result.clips.length > 1 && <label className="augmentation-clip-picker">Episode<select aria-label="Preview episode" value={clip.index} onChange={event => setClipIndex(Number(event.target.value))}>{result.clips.map(item => <option key={item.index} value={item.index}>Episode {item.episode_index}</option>)}</select></label>}</div>
        <div className="augmentation-comparison">{[true, false].map(original => {
          const label = original ? 'Original' : 'Augmented';
          const url = augmentationClipUrl(job.id, clip.index, original);
          return <figure key={`${clip.index}-${label}`}><figcaption><span>{label}</span><small>Episode {clip.episode_index}</small></figcaption><video controls playsInline preload="metadata" aria-label={`${label} episode ${clip.episode_index}`} src={url} onError={() => setMediaErrors(previous => ({ ...previous, [url]: true }))} />{mediaErrors[url] && <p className="augmentation-hint">Preview could not load. Download the clip to review it.</p>}<a className="text-link" href={url} download>Download {label.toLowerCase()} clip <Icon name="arrow" size={13} /></a></figure>;
        })}</div>
      </>}
      <div className="augmentation-export"><a className="secondary-button" href={augmentationDownloadUrl(job.id)} download>Download review bundle <Icon name="arrow" size={15} /></a></div>
      <details className="augmentation-provenance"><summary>Prompt & source provenance</summary><dl><div><dt>Dataset</dt><dd>{result.repo_id}</dd></div><div><dt>Model</dt><dd>{result.model}</dd></div><div><dt>Source inspection</dt><dd><code>{result.source_job_id}</code></dd></div><div><dt>Revision</dt><dd><code>{result.revision}</code></dd></div><div><dt>Metadata SHA-256</dt><dd><code>{result.metadata_sha256}</code></dd></div></dl><h4>Effective prompt</h4><p className="augmentation-effective-prompt">{result.prompt}</p>{clip && <dl><div><dt>Original SHA-256</dt><dd><code>{clip.input_sha256}</code></dd></div><div><dt>Augmented SHA-256</dt><dd><code>{clip.output_sha256}</code></dd></div></dl>}</details>
    </>}
  </div>;
}

export function AugmentationPanel({ projectId, preferredDatasetId, onChooseDataset }: {
  projectId: string; preferredDatasetId?: string; onChooseDataset: () => void;
}) {
  const client = useQueryClient();
  const [datasetId, setDatasetId] = useState(preferredDatasetId ?? '');
  const [cameraKey, setCameraKey] = useState('');
  const [episodes, setEpisodes] = useState('0');
  const [start, setStart] = useState('0');
  const [duration, setDuration] = useState('5');
  const [preset, setPreset] = useState<Preset>('lighting');
  const [prompt, setPrompt] = useState('');
  const [selectedRunId, setSelectedRunId] = useState('');
  const options = useQuery({ queryKey: ['augmentation-options'], queryFn: api.augmentationOptions, enabled: !!projectId, retry: false });
  const jobs = useQuery({ queryKey: ['jobs', projectId], queryFn: () => api.jobs(projectId), enabled: !!projectId, retry: false,
    refetchInterval: query => query.state.data?.some(isActive) ? 1000 : 5000 });
  const datasets = (jobs.data ?? []).filter(isDatasetJob).filter((job): job is InspectedDataset =>
    job.status === 'succeeded' && job.result?.source === 'huggingface' && videoCameras(job.result).length > 0,
  ).sort((a, b) => b.created_at.localeCompare(a.created_at));
  const dataset = datasets.find(job => job.id === datasetId) ?? datasets[0];
  const cameras = videoCameras(dataset?.result);
  const camera = cameras.includes(cameraKey) ? cameraKey : cameras[0] ?? '';
  const maxClips = Math.min(4, options.data?.max_clips ?? 4);
  const maxDuration = Math.min(10, options.data?.max_duration_seconds ?? 10);
  const episodeParts = episodes.split(',').map(value => value.trim());
  const indices = episodeParts.map(Number);
  const episodeIssue = episodeParts.some(value => !/^\d+$/.test(value)) || indices.some(value => !Number.isSafeInteger(value))
    ? 'Enter episode indices as whole numbers separated by commas.'
    : indices.length > maxClips ? `Choose at most ${maxClips} episodes per run.`
    : new Set(indices).size !== indices.length ? 'Choose each episode only once.'
    : dataset && indices.some(value => value >= dataset.result.total_episodes) ? `Episode indices must be between 0 and ${dataset.result.total_episodes - 1}.` : null;
  const timingIssue = !start.trim() || !Number.isFinite(Number(start)) || Number(start) < 0 || Number(start) > 86400
    ? 'Start time must be between 0 and 86,400 seconds.'
    : !duration.trim() || !Number.isFinite(Number(duration)) || Number(duration) < 1 || Number(duration) > maxDuration
      ? `Clip duration must be between 1 and ${maxDuration} seconds.` : null;
  const blocked = !projectId ? 'Create or select a project to begin.'
    : options.isPending || jobs.isPending ? 'Loading augmentation options…'
    : options.isError || jobs.isError ? 'Load the augmentation options and datasets to continue.'
    : !options.data?.configured ? 'Set up Gemini on the application server to generate clips.'
    : !dataset ? 'Inspect a Hugging Face dataset with a video camera first.'
    : episodeIssue ?? timingIssue ?? (preset === 'custom' && !prompt.trim() ? 'Describe the edit you want to make.' : null);
  const create = useMutation({
    mutationFn: () => {
      if (blocked || !dataset) throw new Error(blocked ?? 'Choose a dataset.');
      const body: AugmentationRequest = { operation: 'dataset.augment', source_job_id: dataset.id,
        episode_indices: indices, camera_key: camera, preset, prompt: prompt.trim(), start_seconds: Number(start), duration_seconds: Number(duration) };
      return api.augment(projectId, body);
    },
    onSuccess: job => {
      setSelectedRunId(job.id);
      client.setQueryData<Job[]>(['jobs', projectId], previous => [job, ...(previous ?? []).filter(item => item.id !== job.id)]);
      void client.invalidateQueries({ queryKey: ['jobs', projectId] });
    },
  });
  const runs = (jobs.data ?? []).filter(isAugmentationJob).sort((a, b) => b.created_at.localeCompare(a.created_at));
  const selectedRun = runs.find(job => job.id === selectedRunId) ?? runs[0];
  return <div className="augmentation-panel">
    <div className="augmentation-heading"><div><h2>Augment dataset</h2></div><span className="augmentation-provider"><Icon name="spark" size={15} /> Gemini Omni</span></div>
    {options.data?.auth_message && <p className="augmentation-hint" role="status">{options.data.auth_message}</p>}
    {options.error && <div className="augmentation-notice"><p className="error-notice" role="alert">{options.error.message}</p><button type="button" className="text-button" disabled={!projectId || options.isFetching} onClick={() => void options.refetch()}>Retry augmentation setup</button></div>}
    {jobs.error && <div className="augmentation-notice"><p className="error-notice" role="alert">{jobs.error.message}</p><button type="button" className="text-button" disabled={!projectId || jobs.isFetching} onClick={() => void jobs.refetch()}>Retry datasets</button></div>}
    {options.data && !options.data.configured && <div className="augmentation-setup" role="status"><Icon name="sliders" size={18} /><div><strong>Connect Gemini to start generating</strong><p>{options.data.setup_message ?? 'Connect Google Cloud in Settings and install ffmpeg and ffprobe on the application server.'}</p><button type="button" className="text-button" disabled={!projectId || options.isFetching} onClick={() => void options.refetch()}>{options.isFetching ? 'Checking…' : 'Check setup again'}</button></div></div>}
    {!datasets.length && !jobs.isPending && !jobs.isError && <div className="augmentation-empty"><Icon name="database" size={25} /><h3>Choose a video dataset first</h3><button className="secondary-button" type="button" onClick={onChooseDataset}>Import a dataset <Icon name="arrow" size={15} /></button></div>}
    <form className="augmentation-form" onSubmit={event => { event.preventDefault(); if (!blocked && !create.isPending) create.mutate(); }}>
      <fieldset className="augmentation-source" disabled={!projectId || create.isPending || !dataset}><legend>Source clips</legend><div className="augmentation-fields"><label className="augmentation-wide" htmlFor="augmentation-dataset">Inspected dataset<select id="augmentation-dataset" value={dataset?.id ?? ''} onChange={event => { setDatasetId(event.target.value); setCameraKey(''); create.reset(); }}><option value="" disabled>Select an inspected dataset</option>{datasets.map(job => <option key={job.id} value={job.id}>{job.result.repo_id} · {job.result.revision.slice(0, 7)} · {dateLabel(job.created_at)}</option>)}</select></label><label htmlFor="augmentation-camera">Video camera<select id="augmentation-camera" value={camera} onChange={event => setCameraKey(event.target.value)}>{!cameras.length && <option value="">No video camera available</option>}{cameras.map(key => <option key={key} value={key}>{key}</option>)}</select></label><label htmlFor="augmentation-episodes">Episode indices<input id="augmentation-episodes" aria-label="Episode indices" type="text" value={episodes} onChange={event => setEpisodes(event.target.value)} placeholder="0, 1, 2" aria-describedby="augmentation-episodes-help" aria-invalid={!!episodeIssue} /><small id="augmentation-episodes-help">Up to {maxClips}, comma-separated.</small></label><label htmlFor="augmentation-start">Episode start (seconds)<input id="augmentation-start" aria-label="Episode start (seconds)" type="number" min="0" max="86400" step="0.1" value={start} onChange={event => setStart(event.target.value)} required /></label><label htmlFor="augmentation-duration">Clip duration (seconds)<input id="augmentation-duration" aria-label="Clip duration (seconds)" type="number" min="1" max={maxDuration} step="0.1" value={duration} onChange={event => setDuration(event.target.value)} required /><small>1–{maxDuration} seconds.</small></label></div>{dataset && (episodeIssue || timingIssue) && <p className="augmentation-field-error" role="status">{episodeIssue ?? timingIssue}</p>}</fieldset>
      <fieldset className="augmentation-edit" disabled={!projectId || create.isPending}><legend>Visual edit</legend><div className="augmentation-presets">{(Object.keys(presetDetails) as Preset[]).map(id => <label className="augmentation-preset" key={id}><input type="radio" name="augmentation-preset" value={id} checked={preset === id} onChange={() => { setPreset(id); create.reset(); }} aria-label={presetDetails[id].label} /><span className="augmentation-preset-heading"><Icon name={presetDetails[id].icon} size={18} /><strong>{presetDetails[id].label}</strong><span className="augmentation-choice-dot" /></span></label>)}</div><label className="augmentation-prompt-label" htmlFor="augmentation-prompt">{preset === 'custom' ? 'Edit instructions' : 'Additional instructions (optional)'}<textarea id="augmentation-prompt" value={prompt} onChange={event => setPrompt(event.target.value)} rows={3} maxLength={2000} required={preset === 'custom'} placeholder={preset === 'texture' ? 'For example, give the tabletop a dark wood texture.' : 'For example, use soft afternoon light from the left.'} /></label>{preset !== 'custom' && options.data?.presets?.find(item => item.id === preset)?.prompt && <details className="augmentation-preset-prompt"><summary>View preset instructions</summary><p>{options.data.presets?.find(item => item.id === preset)!.prompt}</p></details>}</fieldset>
      <div className="augmentation-submit"><div className="augmentation-disclosure"><Icon name="external" size={16} /><p>{options.data?.auth_mode === 'google_cloud' ? `Clips and prompts are sent to Google Cloud. Charges apply to ${options.data.google_cloud_project}.` : 'Clips and prompts are sent to Google’s Gemini API. Charges may apply.'}</p></div>{create.error && <p className="error-notice" role="alert">{create.error.message}</p>}<div className="augmentation-submit-row"><p className="augmentation-hint" role="status">{blocked ?? `${indices.length} clip${indices.length === 1 ? '' : 's'} · ${duration} seconds each`}</p><button type="submit" className="primary-button" disabled={!!blocked || create.isPending}>{create.isPending ? 'Starting augmentation…' : 'Generate augmented clips'}<Icon name="spark" size={16} /></button></div></div>
    </form>
    <section className="augmentation-history" aria-labelledby="augmentation-history-title"><div className="augmentation-history-heading"><div><h2 id="augmentation-history-title">History</h2></div>{!!runs.length && <span>{runs.length} run{runs.length === 1 ? '' : 's'}</span>}</div>{!runs.length ? <p className="augmentation-hint">No augmentations yet.</p> : <><label className="augmentation-history-picker" htmlFor="augmentation-history">Run<select id="augmentation-history" aria-label="Run" value={selectedRun?.id ?? ''} onChange={event => setSelectedRunId(event.target.value)}>{runs.map(job => <option key={job.id} value={job.id}>{dateLabel(job.created_at)} · {presetDetails[job.request.preset as Preset]?.label ?? 'Augmentation'} · {job.status}</option>)}</select></label>{selectedRun && <AugmentationRun key={selectedRun.id} job={selectedRun} projectId={projectId} />}</>}</section>
  </div>;
}
