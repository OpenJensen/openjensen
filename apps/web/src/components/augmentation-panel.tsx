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
import { isAugmentationSource, augmentationVideoCameras as videoCameras } from '@/lib/augmentation-source';
import { useDurableSubmission } from '@/lib/durable-submission';
import { augmentationAcknowledgement, validatedProjectHistory } from '@/lib/dataset-submission';
import { DatasetSubmissionRecovery } from '@/components/dataset-submission-recovery';
import { AugmentationGallery } from '@/components/augmentation-gallery';
import { AugmentationDatasetApply } from '@/components/augmentation-dataset-apply';
import './augmentation-panel.css';

type ClipSource = NonNullable<AugmentationRequest['source_clip']>;
type ReuseSelection = { job: AugmentationJob; clipIndex: number; variant: ClipSource['variant'] };

function cameraLabel(key: string) {
  const name = key.replace(/^observation\.images\./, '').replace(/[_.-]+/g, ' ');
  return name.charAt(0).toUpperCase() + name.slice(1);
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
          <h3>{job.request.dataset_scope ? 'Dataset augmentation' : 'Style preview'}</h3>
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
  const [prompt, setPrompt] = useState('');
  const [selectedRunId, setSelectedRunId] = useState('');
  const [reuse, setReuse] = useState<ReuseSelection | null>(null);
  const [applyJob, setApplyJob] = useState<AugmentationJob | null>(null);
  const applyRef = useRef<HTMLDivElement>(null);
  const sourceSelection = reuse?.job.project_id === projectId ? reuse : null;
  const selectedSourceClip = sourceSelection?.job.result?.clips.find(
    (item) => item.index === sourceSelection.clipIndex,
  );
  const [styleOpen, setStyleOpen] = useState(false);
  const [styleProject, setStyleProject] = useState('');
  const [sourcePickerOpen, setSourcePickerOpen] = useState(false);
  const styleDialog = useRef<HTMLDialogElement>(null);
  const styleTrigger = useRef<HTMLElement | null>(null);
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const dialogVisible = styleOpen && (!projectId || styleProject === projectId);
  useEffect(() => {
    if (!dialogVisible) return;
    const dialog = styleDialog.current;
    const trigger = styleTrigger.current;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    dialog?.showModal();
    promptRef.current?.focus({ preventScroll: true });
    return () => {
      dialog?.close();
      document.body.style.overflow = previousOverflow;
      if (trigger?.isConnected) trigger.focus({ preventScroll: true });
    };
  }, [dialogVisible]);
  function openStyle() {
    styleTrigger.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    setStyleProject(projectId);
    setStyleOpen(true);
  }
  function closeStyle() {
    setStyleOpen(false);
  }
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
    if (mounted.current && selection.current.generation === generation) {
      setSelectedRunId(job.id);
      setStyleOpen(false);
      setSourcePickerOpen(false);
    }
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
                      (!prompt.trim() ? 'Describe the style you want to create.' : null));
  const create = useMutation({
    mutationFn: async () => {
      if (blocked || !dataset) throw new Error(blocked ?? 'Choose a dataset.');
      const body: AugmentationRequest = {
        operation: 'dataset.augment',
        source_job_id: dataset.id,
        episode_indices: indices,
        camera_key: camera,
        preset: 'custom',
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
  function reuseClip(job: AugmentationJob, clipIndex: number, variant: ClipSource['variant']) {
    const clip = job.result?.clips.find((item) => item.index === clipIndex);
    if (!clip || job.project_id !== projectId) return;
    setReuse({ job, clipIndex, variant });
    setDatasetId(job.request.source_job_id);
    setCameraKey(clip.camera_key);
    setEpisodes(String(clip.episode_index));
    setStart(String(job.request.start_seconds));
    setDuration(String(job.request.duration_seconds));
    setPrompt('');
    create.reset();
    openStyle();
  }
  const toggleEpisode = (index: number) => {
    const valid = episodeIssue ? [] : indices;
    const next = valid.includes(index) ? valid.filter((value) => value !== index) : [...valid, index];
    setEpisodes(next.sort((a, b) => a - b).join(', '));
    create.reset();
  };
  const recovery = (
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
  );
  return (
    <div className="augmentation-panel">
      <div className="augmentation-generator" role="group" aria-label="Generator">
        <Icon name="spark" size={16} />
        <strong>Gemini Omni</strong>
        <span>{options.data?.auth_mode === 'google_cloud' ? 'Google Cloud' : 'Gemini API'}</span>
        <span>
          {options.isPending
            ? 'Checking…'
            : options.isError
              ? 'Unavailable'
              : options.data?.configured
                ? 'Connected'
                : 'Setup required'}
        </span>
      </div>
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
                  Cloud &amp; APIs <Icon name="arrow" size={14} />
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
      <details
        className="augmentation-source-picker"
        open={sourcePickerOpen || !runs.length}
        onToggle={(event) => {
          if (runs.length) setSourcePickerOpen(event.currentTarget.open);
        }}
      >
        <summary>Choose preview clips</summary>
        <div className="augmentation-form">
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
            {!!datasets.length && (
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
            )}
            {dataset && (
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
            <div className="augmentation-source-actions">
              <p className="augmentation-hint">
                {dataset
                  ? indices.length + ' clips · ' + duration + ' seconds each · ' + cameraLabel(camera) + ' camera'
                  : 'Choose a video dataset to start.'}
              </p>
              <button
                type="button"
                className="primary-button"
                aria-haspopup="dialog"
                disabled={
                  !projectId ||
                  !dataset ||
                  !!episodeIssue ||
                  !!timingIssue ||
                  !options.data?.configured ||
                  create.isPending ||
                  submission.busy ||
                  !!submission.attempt ||
                  !submission.available ||
                  !submission.hydrated
                }
                onClick={() => {
                  setReuse(null);
                  setPrompt('');
                  create.reset();
                  openStyle();
                }}
              >
                Add a style <Icon name="spark" size={16} />
              </button>
            </div>
          </fieldset>
        </div>
      </details>
      {!dialogVisible && recovery}
      {dialogVisible && (
        <dialog
          ref={styleDialog}
          className="augmentation-style-dialog"
          aria-labelledby="augmentation-style-title"
          onCancel={(event) => {
            event.preventDefault();
            closeStyle();
          }}
          onClick={(event) => {
            if (event.target !== event.currentTarget) return;
            const bounds = event.currentTarget.getBoundingClientRect();
            if (
              event.clientX < bounds.left ||
              event.clientX > bounds.right ||
              event.clientY < bounds.top ||
              event.clientY > bounds.bottom
            )
              closeStyle();
          }}
        >
          <header>
            <div>
              <h2 id="augmentation-style-title">
                {sourceSelection?.variant === 'augmented' ? 'Refine style' : 'Add a style'}
              </h2>
              <p>
                {sourceSelection?.variant === 'augmented'
                  ? 'Describe what you want to change in this result.'
                  : 'Describe the style you want to try on your original clip.'}
              </p>
            </div>
            <button
              type="button"
              className="text-button augmentation-dialog-close"
              aria-label="Close style dialog"
              onClick={closeStyle}
            >
              ×
            </button>
          </header>
          {sourceSelection && selectedSourceClip ? (
            <div className="augmentation-dialog-source" aria-label="Selected source clip">
              <video
                controls
                muted
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
                <strong>
                  {sourceSelection.variant === 'augmented'
                    ? 'Result from ' + (runLabel(sourceSelection.job.id) ?? 'saved run')
                    : 'Original clip'}
                </strong>
                <p>
                  Episode {selectedSourceClip.episode_index} · {cameraLabel(selectedSourceClip.camera_key)} camera
                </p>
                <p>
                  {selectedSourceClip.source_start_seconds}s–{selectedSourceClip.source_end_seconds}s
                </p>
              </div>
            </div>
          ) : (
            <p className="augmentation-dialog-selection">
              {dataset?.result.repo_id} · Episode{indices.length === 1 ? '' : 's'} {episodes} · {cameraLabel(camera)}{' '}
              camera · {start}s–{Number(start) + Number(duration)}s
            </p>
          )}
          <form
            className="augmentation-form"
            onSubmit={(event) => {
              event.preventDefault();
              if (!blocked && !create.isPending) create.mutate();
            }}
          >
            <fieldset disabled={!projectId || create.isPending}>
              <label className="augmentation-prompt-label" htmlFor="augmentation-prompt">
                Style prompt
              </label>
              <textarea
                id="augmentation-prompt"
                ref={promptRef}
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                rows={5}
                maxLength={2000}
                required
                placeholder="For example, make only the tabletop brushed metal and keep the wall as it is."
              />
            </fieldset>
            <p className="augmentation-hint">Describe what should change and what should stay the same.</p>
            <p className="augmentation-dialog-billing">
              {options.data?.auth_mode === 'google_cloud'
                ? 'Clips and prompts are sent to Google Cloud. Charges apply to ' +
                  options.data.google_cloud_project +
                  '.'
                : 'Clips and prompts are sent to Google’s Gemini API. Charges may apply.'}
            </p>
            <details className="augmentation-preset-prompt">
              <summary>Generation details</summary>
              {options.data?.auth_message && <p>{options.data.auth_message}</p>}
              {options.data?.preservation_prompt && (
                <p>
                  {options.data.preservation_prompt}
                  {prompt.trim()}
                </p>
              )}
            </details>
            {create.error && (
              <p className="error-notice" role="alert">
                {create.error.message}
              </p>
            )}
            {recovery}
            {blocked && prompt.trim() && (
              <p className="augmentation-field-error" role="status">
                {blocked}
              </p>
            )}
            <footer>
              <button type="button" className="secondary-button" onClick={closeStyle}>
                Close
              </button>
              <button type="submit" className="primary-button" disabled={!!blocked || create.isPending}>
                {create.isPending ? 'Starting…' : 'Generate style'} <Icon name="spark" size={16} />
              </button>
            </footer>
          </form>
        </dialog>
      )}
      <AugmentationGallery
        runs={runs}
        projectId={projectId}
        selectedRun={selectedRun}
        onReuse={reuseClip}
        onAdd={(job, index) => reuseClip(job, index, 'original')}
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
                      <strong>Run {runs.length - index}</strong>
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
