'use client';

import { useEffect, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  api,
  augmentationClipUrl,
  augmentationDownloadUrl,
  isActive,
  isAugmentationJob,
  type AugmentationJob,
  type AugmentationRequest,
  type Job,
} from '@/lib/api';
import { Icon } from '@/components/icon';
import { WorkflowIntegration } from '@/components/workflow-integration';
import { isAugmentationSource, augmentationVideoCameras as videoCameras } from '@/lib/augmentation-source';
import { useDurableSubmission } from '@/lib/durable-submission';
import { augmentationAcknowledgement, validatedProjectHistory } from '@/lib/dataset-submission';
import { DatasetSubmissionRecovery } from '@/components/dataset-submission-recovery';
import { AugmentationGallery } from '@/components/augmentation-gallery';
import { AugmentationDatasetApply } from '@/components/augmentation-dataset-apply';
import './augmentation-panel.css';

type Preset = 'lighting' | 'texture' | 'custom';
type ClipSource = NonNullable<AugmentationRequest['source_clip']>;
type ReuseSelection = { job: AugmentationJob; clipIndex: number; variant: ClipSource['variant'] };
const presetDetails = {
  lighting: { label: 'Change lighting', icon: 'spark' },
  texture: { label: 'Change textures', icon: 'layers' },
  custom: { label: 'Custom edit', icon: 'sliders' },
} as const;

function cameraLabel(key: string) {
  const name = key.replace(/^observation\.images\./, '').replace(/[_.-]+/g, ' ');
  return name.charAt(0).toUpperCase() + name.slice(1);
}

function AppearancePreview({ preset }: { preset: Preset }) {
  return (
    <span className={`augmentation-look augmentation-look-${preset}`} aria-hidden="true">
      <svg viewBox="0 0 240 100" fill="none">
        <path className="augmentation-look-room" d="M0 0h240v100H0z" />
        <path className="augmentation-look-wall" d="M0 68 120 34l120 34v32H0z" />
        <path className="augmentation-look-table" d="m41 62 79-24 80 24-80 26z" />
        <path className="augmentation-look-table-edge" d="M41 62v8l79 26 80-26v-8l-80 26z" />
        {preset === 'texture' && (
          <g className="augmentation-look-grain">
            <path d="m55 61 77 25M70 57l77 25M85 52l77 25M101 47l76 25M116 43l76 25" />
          </g>
        )}
        <path className="augmentation-look-shadow" d="m92 61 24-7 32 10-24 8z" />
        <path className="augmentation-look-object" d="m115 48 16-5 16 5v16l-16 5-16-5z" />
        <path className="augmentation-look-object-edge" d="m115 48 16 5 16-5m-16 5v16" />
        <path className="augmentation-look-arm" d="m86 61-6-24 22-13 12 14" />
        <circle className="augmentation-look-joint" cx="80" cy="37" r="4" />
        <circle className="augmentation-look-joint" cx="102" cy="24" r="4" />
        <path className="augmentation-look-gripper" d="m109 40 5-2 5 2" />
        {preset === 'lighting' && (
          <>
            <path className="augmentation-look-light" d="m194 12-64 33 96 28z" />
            <circle className="augmentation-look-sun" cx="194" cy="12" r="7" />
          </>
        )}
        {preset === 'custom' && <path className="augmentation-look-spark" d="m183 17 3 8 8 3-8 3-3 8-3-8-8-3 8-3z" />}
      </svg>
    </span>
  );
}

function dateLabel(timestamp: string) {
  return new Date(timestamp).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

function RunStatus({ job }: { job: AugmentationJob }) {
  const label =
    {
      succeeded: 'Ready to review',
      running: 'Generating',
      queued: 'Queued',
      failed: 'Failed',
      cancelled: 'Cancelled',
      interrupted: 'Interrupted',
    }[job.status] ?? job.status;
  return (
    <span className={`status status-${job.status}`}>
      <span className="status-dot" />
      {label}
    </span>
  );
}

function AugmentationRun({
  job,
  projectId,
  sourceRunLabel,
  onViewSource,
}: {
  job: AugmentationJob;
  projectId: string;
  sourceRunLabel?: string;
  onViewSource: (id: string) => void;
}) {
  const client = useQueryClient();
  const cancel = useMutation({
    mutationFn: () => api.cancel(job.id),
    onSuccess: (updated) => {
      client.setQueryData<Job[]>(['jobs', projectId], (previous) =>
        previous?.map((item) => (item.id === updated.id ? updated : item)),
      );
      void client.invalidateQueries({ queryKey: ['jobs', projectId] });
    },
  });
  const result = job.status === 'succeeded' ? job.result : null;
  const clip = result?.clips[0];
  return (
    <div className="augmentation-run">
      <div className="augmentation-run-heading">
        <div>
          <h3>{presetDetails[job.request.preset as Preset]?.label ?? 'Dataset augmentation'}</h3>
          <p>
            {dateLabel(job.created_at)} ·{' '}
            {job.request.dataset_scope
              ? 'Full dataset · ' + job.request.dataset_scope.camera_keys.length + ' cameras'
              : job.request.episode_indices.length + ' preview clips'}
          </p>
        </div>
        <RunStatus job={job} />
      </div>
      {job.request.source_clip && (
        <p className="augmentation-lineage">
          {job.request.source_clip.variant === 'augmented' ? 'Refinement of' : 'Another variation from'}{' '}
          {sourceRunLabel ? (
            <button type="button" className="text-button" onClick={() => onViewSource(job.request.source_clip!.job_id)}>
              {sourceRunLabel}
            </button>
          ) : (
            'a saved augmentation'
          )}
          {' · '}Episode {job.request.episode_indices[0]}
        </p>
      )}
      {isActive(job) && (
        <div className="augmentation-progress" role="status">
          <span className="spinner" />
          <div>
            <strong>{job.status === 'queued' ? 'Waiting to augment' : 'Creating visual variations'}</strong>
            <p>{job.stage ? job.stage.replaceAll('_', ' ') : 'Preparing clips…'}</p>
          </div>
          <button
            type="button"
            className="secondary-button"
            disabled={cancel.isPending}
            onClick={() => cancel.mutate()}
          >
            {cancel.isPending ? 'Cancelling…' : 'Cancel augmentation'}
          </button>
        </div>
      )}
      {cancel.error && (
        <p className="error-notice" role="alert">
          {cancel.error.message}
        </p>
      )}
      {job.error && (
        <p className="error-notice" role="alert">
          {job.error}
        </p>
      )}
      {(job.status === 'cancelled' || job.status === 'interrupted') && (
        <p className="augmentation-hint">Run did not complete.</p>
      )}
      {result && (
        <>
          <div className="augmentation-review">
            <Icon name="check" size={17} />
            <div>
              <p>
                Generated clips may not match the original action labels. Review motion and timing, and validate labels
                before training.
              </p>
            </div>
          </div>
          {!!result.warnings.length && (
            <ul className="augmentation-warnings">
              {result.warnings.map((warning, index) => (
                <li key={index}>{warning}</li>
              ))}
            </ul>
          )}
          <div className="augmentation-export">
            <a className="secondary-button" href={augmentationDownloadUrl(job.id)} download>
              Download review bundle <Icon name="arrow" size={15} />
            </a>
          </div>
          <details className="augmentation-provenance">
            <summary>Prompt & source provenance</summary>
            <dl>
              <div>
                <dt>Dataset</dt>
                <dd>{result.repo_id}</dd>
              </div>
              <div>
                <dt>Model</dt>
                <dd>{result.model}</dd>
              </div>
              <div>
                <dt>Source inspection</dt>
                <dd>
                  <code>{result.source_job_id}</code>
                </dd>
              </div>
              <div>
                <dt>Revision</dt>
                <dd>
                  <code>{result.revision}</code>
                </dd>
              </div>
              <div>
                <dt>Metadata SHA-256</dt>
                <dd>
                  <code>{result.metadata_sha256}</code>
                </dd>
              </div>
            </dl>
            <h4>Effective prompt</h4>
            <p className="augmentation-effective-prompt">{result.prompt}</p>
            {clip && (
              <dl>
                <div>
                  <dt>Original SHA-256</dt>
                  <dd>
                    <code>{clip.input_sha256}</code>
                  </dd>
                </div>
                <div>
                  <dt>Augmented SHA-256</dt>
                  <dd>
                    <code>{clip.output_sha256}</code>
                  </dd>
                </div>
              </dl>
            )}
          </details>
        </>
      )}
    </div>
  );
}

export function AugmentationPanel({
  projectId,
  preferredDatasetId,
  onChooseDataset,
  onOpenSettings,
}: {
  projectId: string;
  preferredDatasetId?: string;
  onChooseDataset: () => void;
  onOpenSettings?: () => void;
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
  const [reuse, setReuse] = useState<ReuseSelection | null>(null);
  const [applyJob, setApplyJob] = useState<AugmentationJob | null>(null);
  const applyRef = useRef<HTMLDivElement>(null);
  const sourceSelection = reuse?.job.project_id === projectId ? reuse : null;
  const selectedSourceClip = sourceSelection?.job.result?.clips.find(
    (item) => item.index === sourceSelection.clipIndex,
  );
  const formRef = useRef<HTMLFormElement>(null);
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const submission = useDurableSubmission({ project: projectId, operation: 'dataset.augment' });
  const selection = useRef({ project: projectId, generation: 0 });
  if (selection.current.project !== projectId)
    selection.current = { project: projectId, generation: selection.current.generation + 1 };
  function chooseRun(id: string) {
    selection.current.generation += 1;
    setSelectedRunId(id);
  }
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  const validate = (value: unknown, body: unknown) => augmentationAcknowledgement(value, projectId, body);
  function accept(job: Job | null, generation: number) {
    if (!job) return;
    if (mounted.current && selection.current.generation === generation) setSelectedRunId(job.id);
    client.setQueryData<Job[]>(['jobs', job.project_id], (previous) => [
      job,
      ...(previous ?? []).filter((item) => item.id !== job.id),
    ]);
    void client.invalidateQueries({ queryKey: ['jobs', job.project_id] });
  }
  const options = useQuery({
    queryKey: ['augmentation-options'],
    queryFn: api.augmentationOptions,
    enabled: !!projectId,
    retry: false,
  });
  const jobs = useQuery({
    queryKey: ['jobs', projectId],
    queryFn: () => api.jobs(projectId),
    enabled: !!projectId,
    retry: false,
    refetchInterval: (query) => (query.state.data?.some(isActive) ? 1000 : 5000),
  });
  const datasets = (jobs.data ?? [])
    .filter(isAugmentationSource)
    .sort((a, b) => b.created_at.localeCompare(a.created_at));
  const dataset = datasets.find((job) => job.id === datasetId) ?? datasets[0];
  const cameras = videoCameras(dataset?.result);
  const camera = cameras.includes(cameraKey) ? cameraKey : (cameras[0] ?? '');
  const maxClips = Math.min(4, options.data?.max_clips ?? 4);
  const maxDuration = Math.min(10, options.data?.max_duration_seconds ?? 10);
  const episodeParts = episodes.split(',').map((value) => value.trim());
  const indices = episodeParts.map(Number);
  const episodeIssue =
    episodeParts.some((value) => !/^\d+$/.test(value)) || indices.some((value) => !Number.isSafeInteger(value))
      ? 'Enter whole episode numbers separated by commas.'
      : indices.length > maxClips
        ? `Choose at most ${maxClips} episodes per run.`
        : new Set(indices).size !== indices.length
          ? 'Choose each episode only once.'
          : dataset && indices.some((value) => value >= dataset.result.total_episodes)
            ? `Episode numbers must be between 0 and ${dataset.result.total_episodes - 1}.`
            : null;
  const timingIssue =
    !start.trim() || !Number.isFinite(Number(start)) || Number(start) < 0 || Number(start) > 86400
      ? 'Start time must be between 0 and 86,400 seconds.'
      : !duration.trim() || !Number.isFinite(Number(duration)) || Number(duration) < 1 || Number(duration) > maxDuration
        ? `Clip duration must be between 1 and ${maxDuration} seconds.`
        : null;
  const blocked = !projectId
    ? 'Create or select a project to begin.'
    : !submission.hydrated
      ? 'Checking saved requests…'
      : !submission.available
        ? 'Restore browser recovery storage before submitting.'
        : submission.attempt
          ? 'Verify the saved request before starting another augmentation.'
          : submission.busy
            ? 'Checking the request…'
            : options.isPending || jobs.isPending
              ? 'Loading augmentation options…'
              : options.isError || jobs.isError
                ? 'Load the augmentation options and datasets to continue.'
                : !options.data?.configured
                  ? 'Set up Gemini on the application server to generate clips.'
                  : !dataset
                    ? 'Inspect a Hugging Face dataset with a video camera first.'
                    : (episodeIssue ??
                      timingIssue ??
                      (preset === 'custom' && !prompt.trim() ? 'Describe the edit you want to make.' : null));
  const create = useMutation({
    mutationFn: async () => {
      if (blocked || !dataset) throw new Error(blocked ?? 'Choose a dataset.');
      const body: AugmentationRequest = {
        operation: 'dataset.augment',
        source_job_id: dataset.id,
        episode_indices: indices,
        camera_key: camera,
        preset,
        prompt: prompt.trim(),
        start_seconds: Number(start),
        duration_seconds: Number(duration),
        ...(sourceSelection
          ? {
              source_clip: {
                job_id: sourceSelection.job.id,
                clip_index: sourceSelection.clipIndex,
                variant: sourceSelection.variant,
              },
            }
          : {}),
      };
      const generation = selection.current.generation;
      const job = await submission.submit(body, validate);
      accept(job, generation);
      return job;
    },
  });
  const runs = (jobs.data ?? []).filter(isAugmentationJob).sort((a, b) => b.created_at.localeCompare(a.created_at));
  const selectedRun = runs.find((job) => job.id === selectedRunId) ?? runs[0];
  const runLabel = (id: string) => {
    const index = runs.findIndex((job) => job.id === id);
    return index < 0 ? undefined : `Run ${runs.length - index}`;
  };
  const editToSend = prompt.trim() || options.data?.presets?.find((item) => item.id === preset)?.prompt || '';
  function reuseClip(job: AugmentationJob, clipIndex: number, variant: ClipSource['variant']) {
    const clip = job.result?.clips.find((item) => item.index === clipIndex);
    if (!clip || job.project_id !== projectId) return;
    setReuse({ job, clipIndex, variant });
    setDatasetId(job.request.source_job_id);
    setCameraKey(clip.camera_key);
    setEpisodes(String(clip.episode_index));
    setStart(String(job.request.start_seconds));
    setDuration(String(job.request.duration_seconds));
    setPreset(variant === 'augmented' ? 'custom' : job.request.preset);
    setPrompt(variant === 'augmented' ? '' : (job.request.prompt ?? ''));
    create.reset();
    requestAnimationFrame(() => {
      formRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      promptRef.current?.focus({ preventScroll: true });
    });
  }
  const toggleEpisode = (index: number) => {
    const valid = episodeIssue ? [] : indices;
    const next = valid.includes(index) ? valid.filter((value) => value !== index) : [...valid, index];
    setEpisodes(next.sort((a, b) => a - b).join(', '));
    create.reset();
  };
  return (
    <div className="augmentation-panel">
      <WorkflowIntegration
        label="Generator"
        name="Gemini Omni"
        icon="spark"
        meta={options.data ? (options.data.auth_mode === 'google_cloud' ? 'Google Cloud' : 'Gemini API') : 'Google'}
        status={
          !projectId
            ? undefined
            : options.isError
              ? 'Unavailable'
              : options.isPending
                ? 'Checking…'
                : options.data?.configured
                  ? 'Configured'
                  : 'Setup required'
        }
      />
      {options.error && (
        <div className="augmentation-notice">
          <p className="error-notice" role="alert">
            {options.error.message}
          </p>
          <button
            type="button"
            className="text-button"
            disabled={!projectId || options.isFetching}
            onClick={() => void options.refetch()}
          >
            Retry augmentation setup
          </button>
        </div>
      )}
      {jobs.error && (
        <div className="augmentation-notice">
          <p className="error-notice" role="alert">
            {jobs.error.message}
          </p>
          <button
            type="button"
            className="text-button"
            disabled={!projectId || jobs.isFetching}
            onClick={() => void jobs.refetch()}
          >
            Retry datasets
          </button>
        </div>
      )}
      {options.data && !options.data.configured && (
        <div className="augmentation-setup" role="status">
          <Icon name="sliders" size={18} />
          <div>
            <strong>Connect Gemini to start generating</strong>
            <details className="augmentation-preset-prompt">
              <summary>Setup details</summary>
              <p>
                {options.data.setup_message ??
                  'Connect Google Cloud in Settings and install ffmpeg and ffprobe on the application server.'}
              </p>
            </details>
            <div className="augmentation-setup-actions">
              {onOpenSettings && (
                <button type="button" className="secondary-button" onClick={onOpenSettings}>
                  Open settings <Icon name="arrow" size={14} />
                </button>
              )}
              <button
                type="button"
                className="text-button"
                disabled={!projectId || options.isFetching}
                onClick={() => void options.refetch()}
              >
                {options.isFetching ? 'Checking…' : 'Check setup again'}
              </button>
            </div>
          </div>
        </div>
      )}
      <form
        ref={formRef}
        className="augmentation-form"
        onSubmit={(event) => {
          event.preventDefault();
          if (!blocked && !create.isPending) create.mutate();
        }}
      >
        <fieldset className="augmentation-source" disabled={!projectId || create.isPending}>
          <legend className="visually-hidden">Source clips</legend>
          {!projectId && (
            <div className="augmentation-empty">
              <Icon name="folder" size={25} />
              <h3>Select a project</h3>
            </div>
          )}
          {jobs.isPending && projectId && (
            <div className="augmentation-loading" role="status">
              <span className="spinner" /> Loading your video datasets…
            </div>
          )}
          {!datasets.length && !jobs.isPending && !jobs.isError && (
            <div className="augmentation-empty">
              <Icon name="database" size={25} />
              <h3>No video datasets</h3>
              <button className="secondary-button" type="button" onClick={onChooseDataset}>
                Import a dataset <Icon name="arrow" size={15} />
              </button>
            </div>
          )}
          {sourceSelection && selectedSourceClip ? (
            <div className="augmentation-saved-source" aria-label="Selected source clip">
              <video
                controls
                playsInline
                preload="metadata"
                aria-label="Selected source video"
                src={augmentationClipUrl(
                  sourceSelection.job.id,
                  sourceSelection.clipIndex,
                  sourceSelection.variant === 'original',
                )}
              />
              <div>
                <h3>
                  {sourceSelection.variant === 'augmented'
                    ? 'Refining the generated result'
                    : 'New variation from the saved input'}
                </h3>
                <p>
                  {runLabel(sourceSelection.job.id)} · Episode {selectedSourceClip.episode_index} ·{' '}
                  {cameraLabel(selectedSourceClip.camera_key)} camera
                </p>
                <p>
                  {selectedSourceClip.source_start_seconds}s–{selectedSourceClip.source_end_seconds}s ·{' '}
                  {sourceSelection.job.result?.repo_id}
                </p>
                <p>
                  {sourceSelection.variant === 'augmented'
                    ? 'This edit starts from the augmented video, including its existing changes.'
                    : 'This edit starts from the input video shown in that run.'}
                </p>
                <button
                  type="button"
                  className="text-button"
                  onClick={() => {
                    setReuse(null);
                    create.reset();
                  }}
                >
                  Choose a different source
                </button>
              </div>
            </div>
          ) : (
            !!datasets.length && (
              <>
                <div className="augmentation-group-heading">
                  <h3 id="augmentation-datasets-title">Dataset</h3>
                  <span>{datasets.length} available</span>
                </div>
                <div className="augmentation-datasets" role="radiogroup" aria-labelledby="augmentation-datasets-title">
                  {datasets.map((job) => (
                    <label className="augmentation-choice augmentation-dataset" key={job.id}>
                      <input
                        type="radio"
                        name="augmentation-dataset"
                        aria-label={`${job.result.repo_id}, inspected ${dateLabel(job.created_at)}, revision ${job.result.revision.slice(0, 7)}`}
                        value={job.id}
                        checked={dataset?.id === job.id}
                        onChange={() => {
                          setDatasetId(job.id);
                          setCameraKey('');
                          setEpisodes('0');
                          create.reset();
                        }}
                      />
                      <span className="augmentation-source-icon">
                        <Icon name="database" size={21} />
                      </span>
                      <span className="augmentation-choice-copy">
                        <strong>{job.result.repo_id}</strong>
                        <span>
                          {job.result.total_episodes.toLocaleString()} episodes · {videoCameras(job.result).length}{' '}
                          camera
                          {videoCameras(job.result).length === 1 ? '' : 's'}
                        </span>
                        <small>
                          Inspected {dateLabel(job.created_at)} · {job.result.revision.slice(0, 7)}
                        </small>
                      </span>
                      <span className="augmentation-choice-dot" />
                    </label>
                  ))}
                </div>
                <div className="augmentation-group-heading">
                  <h3 id="augmentation-cameras-title">Camera</h3>
                </div>
                <div className="augmentation-cameras" role="radiogroup" aria-labelledby="augmentation-cameras-title">
                  {cameras.map((key) => (
                    <label className="augmentation-choice augmentation-camera" key={key}>
                      <input
                        type="radio"
                        name="augmentation-camera"
                        value={key}
                        checked={camera === key}
                        aria-label={`${cameraLabel(key)} camera`}
                        onChange={() => {
                          setCameraKey(key);
                          create.reset();
                        }}
                      />
                      <span className="augmentation-camera-icon" aria-hidden="true">
                        <svg viewBox="0 0 32 24" fill="none">
                          <rect x="3" y="5" width="20" height="15" rx="3" />
                          <path d="m23 10 6-3v12l-6-3M9 5l2-3h6l2 3" />
                          <circle cx="13" cy="12.5" r="3.5" />
                        </svg>
                      </span>
                      <span className="augmentation-choice-copy">
                        <strong>{cameraLabel(key)}</strong>
                      </span>
                      <span className="augmentation-choice-dot" />
                    </label>
                  ))}
                </div>
                <div className="augmentation-group-heading">
                  <h3 id="augmentation-episodes-title">Episodes</h3>
                  <span>Up to {maxClips} per run</span>
                </div>
                <div
                  className="augmentation-episode-choices"
                  role="group"
                  aria-labelledby="augmentation-episodes-title"
                >
                  {Array.from({ length: Math.min(dataset.result.total_episodes, 8) }, (_, index) => (
                    <button
                      type="button"
                      key={index}
                      aria-label={`Episode ${index}`}
                      aria-pressed={!episodeIssue && indices.includes(index)}
                      disabled={!episodeIssue && !indices.includes(index) && indices.length >= maxClips}
                      onClick={() => toggleEpisode(index)}
                    >
                      {index}
                    </button>
                  ))}
                </div>
              </>
            )
          )}
          {dataset && !sourceSelection && (
            <div className="augmentation-fields">
              <label className="augmentation-wide" htmlFor="augmentation-episodes">
                Episode numbers
                <input
                  id="augmentation-episodes"
                  aria-label="Episode numbers"
                  type="text"
                  value={episodes}
                  onChange={(event) => setEpisodes(event.target.value)}
                  placeholder="0, 1, 2"
                  aria-describedby="augmentation-episodes-help"
                  aria-invalid={!!episodeIssue}
                />
                <small id="augmentation-episodes-help">
                  0–{dataset.result.total_episodes - 1} · up to {maxClips}, comma-separated
                </small>
              </label>
              <label htmlFor="augmentation-start">
                Start at (seconds)
                <input
                  id="augmentation-start"
                  aria-label="Start at (seconds)"
                  type="number"
                  min="0"
                  max="86400"
                  step="0.1"
                  value={start}
                  onChange={(event) => setStart(event.target.value)}
                  required
                />
              </label>
              <label htmlFor="augmentation-duration">
                Clip length (seconds)
                <input
                  id="augmentation-duration"
                  aria-label="Clip length (seconds)"
                  type="number"
                  min="1"
                  max={maxDuration}
                  step="0.1"
                  value={duration}
                  onChange={(event) => setDuration(event.target.value)}
                  required
                />
                <small>1–{maxDuration} seconds</small>
              </label>
            </div>
          )}
          {dataset && (episodeIssue || timingIssue) && (
            <p className="augmentation-field-error" role="status">
              {episodeIssue ?? timingIssue}
            </p>
          )}
        </fieldset>
        <fieldset className="augmentation-edit" disabled={!projectId || create.isPending}>
          <legend>Appearance</legend>
          <div className="augmentation-presets">
            {(Object.keys(presetDetails) as Preset[]).map((id) => (
              <label className="augmentation-choice augmentation-preset" key={id}>
                <input
                  type="radio"
                  name="augmentation-preset"
                  value={id}
                  checked={preset === id}
                  onChange={() => {
                    setPreset(id);
                    create.reset();
                  }}
                  aria-label={presetDetails[id].label}
                />
                <AppearancePreview preset={id} />
                <span className="augmentation-preset-copy">
                  <span className="augmentation-preset-heading">
                    <strong>{presetDetails[id].label}</strong>
                    <span className="augmentation-choice-dot" />
                  </span>
                </span>
              </label>
            ))}
          </div>
          <label className="augmentation-prompt-label" htmlFor="augmentation-prompt">
            {preset === 'custom' ? 'Edit instructions' : 'Edit instructions (optional)'}
            <textarea
              id="augmentation-prompt"
              ref={promptRef}
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              rows={3}
              maxLength={2000}
              required={preset === 'custom'}
              placeholder={
                preset === 'texture'
                  ? 'For example, make only the tabletop brushed silver metal. Keep the wall unchanged.'
                  : preset === 'custom'
                    ? 'Describe the visual change you want to see.'
                    : 'For example, use soft afternoon light from the left.'
              }
            />
          </label>
          <p className="augmentation-hint">
            Your instructions replace the preset example. Only the changes you describe will be requested.
          </p>
          <div className="augmentation-edit-preview" role="note" aria-label="Edit to send">
            <strong>Edit to send</strong>
            <p>{editToSend || 'Describe the edit you want to make.'}</p>
          </div>
          {options.data?.preservation_prompt && (
            <details className="augmentation-preset-prompt">
              <summary>Full prompt to send</summary>
              <p>
                {options.data.preservation_prompt}
                {editToSend}
              </p>
            </details>
          )}
        </fieldset>
        <section className="augmentation-submit" aria-label="Generate augmentation">
          <div className="augmentation-disclosure">
            <Icon name="external" size={16} />
            <p>
              {options.data?.auth_mode === 'google_cloud'
                ? `Clips and prompts are sent to Google Cloud. Charges apply to ${options.data.google_cloud_project}.`
                : 'Clips and prompts are sent to Google’s Gemini API. Charges may apply.'}
            </p>
          </div>
          {options.data?.configured && options.data.auth_message && (
            <details className="augmentation-preset-prompt">
              <summary>Connection details</summary>
              <p>{options.data.auth_message}</p>
            </details>
          )}
          {create.error && (
            <p className="error-notice" role="alert">
              {create.error.message}
            </p>
          )}
          <DatasetSubmissionRecovery
            submission={submission}
            onReconcile={async () => {
              const generation = selection.current.generation;
              accept(await submission.reconcile(validate), generation);
            }}
            onRetry={async () => {
              const generation = selection.current.generation;
              accept(await submission.retry(validate), generation);
            }}
            onReviewHistory={async () => {
              const fresh = validatedProjectHistory(await api.jobs(projectId), projectId);
              client.setQueryData(['jobs', projectId], fresh);
              return true;
            }}
          />
          <div className="augmentation-submit-row">
            <p className="augmentation-hint" role="status">
              {blocked ??
                `${indices.length} clip${indices.length === 1 ? '' : 's'} · ${duration} seconds each · ${cameraLabel(camera)} camera`}
            </p>
            <button type="submit" className="primary-button" disabled={!!blocked || create.isPending}>
              {create.isPending ? 'Starting augmentation…' : 'Generate augmented clips'}
              <Icon name="spark" size={16} />
            </button>
          </div>
        </section>
      </form>
      <AugmentationGallery
        runs={runs}
        projectId={projectId}
        selectedRun={selectedRun}
        onReuse={reuseClip}
        onAdd={(job, index) => {
          reuseClip(job, index, 'original');
          setPreset('custom');
          setPrompt('');
        }}
        disabled={!projectId || create.isPending || submission.busy || !!submission.attempt}
        onApply={(job) => {
          setApplyJob(job);
          requestAnimationFrame(() => applyRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }));
        }}
      />
      <div ref={applyRef}>
        {applyJob?.project_id === projectId && (
          <AugmentationDatasetApply
            key={applyJob.id}
            job={applyJob}
            dataset={datasets.find((item) => item.id === applyJob.request.source_job_id)}
            busy={
              create.isPending ||
              submission.busy ||
              !!submission.attempt ||
              !submission.available ||
              !submission.hydrated
            }
            onClose={() => setApplyJob(null)}
            onSubmit={async (body) => {
              const generation = selection.current.generation;
              const accepted = await submission.submit(body, validate);
              accept(accepted, generation);
              if (accepted) setApplyJob(null);
            }}
          />
        )}
      </div>
      <section className="augmentation-history" aria-labelledby="augmentation-history-title">
        <div className="augmentation-history-heading">
          <div>
            <h2 id="augmentation-history-title">Run history</h2>
          </div>
          {!!runs.length && (
            <span>
              {runs.length} run{runs.length === 1 ? '' : 's'}
            </span>
          )}
        </div>
        {jobs.isError ? (
          <p className="augmentation-hint" role="status">
            History could not be refreshed. Saved requests still need verification.
          </p>
        ) : jobs.isPending ? (
          <p className="augmentation-hint" role="status">
            Loading augmentation history…
          </p>
        ) : !runs.length ? (
          <div className="augmentation-history-empty">
            <Icon name="layers" size={21} />
            <p>No augmentations yet.</p>
          </div>
        ) : (
          <>
            {runs.length > 1 && (
              <div className="augmentation-run-choices" role="group" aria-label="Augmentation runs">
                {runs.map((job, index) => (
                  <button
                    type="button"
                    key={job.id}
                    aria-label={`View run ${runs.length - index}`}
                    aria-pressed={selectedRun?.id === job.id}
                    onClick={() => chooseRun(job.id)}
                  >
                    <span className="augmentation-run-choice-title">
                      <strong>
                        Run {runs.length - index} ·{' '}
                        {presetDetails[job.request.preset as Preset]?.label ?? 'Augmentation'}
                      </strong>
                      <RunStatus job={job} />
                    </span>
                    <span>{dateLabel(job.created_at)}</span>
                  </button>
                ))}
              </div>
            )}
            {selectedRun && (
              <AugmentationRun
                key={selectedRun.id}
                job={selectedRun}
                projectId={projectId}
                sourceRunLabel={
                  selectedRun.request.source_clip ? runLabel(selectedRun.request.source_clip.job_id) : undefined
                }
                onViewSource={chooseRun}
              />
            )}
          </>
        )}
      </section>
    </div>
  );
}
