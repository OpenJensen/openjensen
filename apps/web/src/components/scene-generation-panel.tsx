'use client';

import { useEffect, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { AgentRequestError, agentRequest } from '@/lib/agent-api';
import { api, apiOrigin, type Job } from '@/lib/api';
import Link from 'next/link';
import { JobHistory } from '@/components/job-history';
import { Notice } from '@/components/ui/notice';
import { StatusBadge } from '@/components/ui/status-badge';
import './scene-generation-panel.css';

type Model = {
  id: string;
  label: string;
  reasoning_efforts: string[];
  default_reasoning_effort: string;
};
const powerLabels: Record<string, string> = {
  low: 'Low',
  medium: 'Medium',
  high: 'High',
  xhigh: 'Extra high',
  max: 'Max',
  ultra: 'Ultra',
};
type Options = {
  models: Model[];
  default_model: string;
  configured: boolean;
  simulation_available?: boolean;
  codex_available: boolean;
  builder_available: boolean;
  supported_repo: string;
  supported_revision: string;
};
type Source = {
  id: string;
  job_id?: string;
  name: string;
  profile?: { repo_id?: string; revision: string; total_episodes: number };
  status: string;
};
type SceneRequest = {
  request_id: string;
  source_job_id: string;
  source_scene_id?: string | null;
  scope?: 'dataset';
  episode_index?: number;
  camera_key: string;
  max_attempts: number;
  model?: string | null;
  reasoning_effort?: string;
};
type SceneJob = Omit<Job, 'request' | 'result'> & {
  request: SceneRequest;
  result?: { kind: 'scene_generation'; scene_id: string; validation_status: string } | null;
};
type Media = {
  role: 'environment_image' | 'episode_replay' | 'source_image' | 'source_video';
  path: string;
  label: string;
  episode_index?: number;
};
type Run = {
  id: string;
  job_id?: string | null;
  status: string;
  stage: string;
  attempt: number;
  message: string;
  request: SceneRequest;
  gates: Record<string, boolean>;
  events: { at: string; stage: string; message: string }[];
  artifacts: string[];
  media?: Media[];
  source?: {
    total_episodes?: number;
    inspected_episodes?: number[];
    replay_episode?: number;
    resource_cleanup?: 'cluster_removed' | 'not_launched' | 'unverified';
  };
  usage: Record<string, number>;
  agent?: { harness: string; model: string; reasoning_effort: string } | null;
};
const isActive = (job: Pick<Job, 'status'>) => job.status === 'queued' || job.status === 'running';
const labels: Record<string, string> = {
  source_evidence: 'Dataset evidence',
  agent_review: 'Agent review',
  usd_structure: 'USD structure',
  isaac_runtime: 'Isaac runtime',
  visual_match: 'Image match',
  calibrated_replay: 'Calibrated replay',
};
const artifactUrl = (id: string, name: string, jobId?: string) =>
  `${apiOrigin}/api/v1/${jobId ? `jobs/${encodeURIComponent(jobId)}/scene` : `scenes/${encodeURIComponent(id)}`}/artifact?name=${encodeURIComponent(name)}`;

export function SceneGenerationPanel({ projectId }: { projectId?: string }) {
  const client = useQueryClient();
  const options = useQuery({
    queryKey: ['scene-options'],
    queryFn: () => agentRequest<Options>('/scene-generation/options'),
    retry: false,
  });
  const sources = useQuery({
    queryKey: ['scene-sources', projectId],
    queryFn: () => agentRequest<Source[]>(`/datasets?project_id=${encodeURIComponent(projectId!)}`),
    enabled: !!projectId,
    retry: false,
  });
  const jobs = useQuery({
    queryKey: ['jobs', projectId],
    queryFn: () => agentRequest<SceneJob[]>(`/projects/${encodeURIComponent(projectId!)}/jobs`),
    enabled: !!projectId,
    retry: false,
    staleTime: 0,
    refetchInterval: (query) =>
      query.state.data?.some((job) => job.kind === 'scene.generate' && isActive(job)) ? 2500 : false,
  });
  const legacy = useQuery({
    queryKey: ['scene-runs', projectId],
    queryFn: () => agentRequest<Run[]>(`/projects/${encodeURIComponent(projectId!)}/scenes`),
    enabled: !!projectId,
    retry: false,
  });
  const [view, setView] = useState<'jobs' | 'new' | 'detail'>('jobs');
  const [selectedId, setSelectedId] = useState('');
  const [sourceId, setSourceId] = useState('');
  const [attempts, setAttempts] = useState(2);
  const [model, setModel] = useState('');
  const [power, setPower] = useState('');
  const [pending, setPending] = useState<SceneRequest | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const history = (jobs.data ?? [])
    .filter((job) => job.kind === 'scene.generate' && job.project_id === projectId)
    .sort((a, b) => b.created_at.localeCompare(a.created_at));
  const selectedJob = history.find((job) => job.id === selectedId);
  const savedScenes = (legacy.data ?? []).filter((run) => !run.job_id);
  const legacyScene = selectedId.startsWith('legacy:')
    ? savedScenes.find((run) => run.id === selectedId.slice(7))
    : undefined;
  const scene = useQuery({
    // Fetch final media even when the job finishes between scene polls.
    queryKey: ['scene-job', selectedJob?.id, selectedJob?.status],
    queryFn: () => agentRequest<Run>(`/jobs/${encodeURIComponent(selectedJob!.id)}/scene`),
    enabled: view === 'detail' && !!selectedJob,
    retry: false,
    refetchInterval: selectedJob && isActive(selectedJob) ? 2500 : false,
  });
  const run = legacyScene ?? scene.data;
  const candidates =
    sources.data?.filter(
      (item) =>
        item.status === 'ready' &&
        item.job_id &&
        item.profile?.repo_id === options.data?.supported_repo &&
        item.profile?.revision === options.data?.supported_revision,
    ) ?? [];
  const selected = pending
    ? candidates.find((item) => item.job_id === pending.source_job_id)
    : (candidates.find((item) => item.job_id === sourceId) ?? candidates[0]);
  const running = history.some(isActive);
  const storageKey = `jensen-scene-request:${projectId}`;

  useEffect(() => {
    try {
      const saved = localStorage.getItem(storageKey);
      const request = saved ? JSON.parse(saved) : null;
      setPending(request);
      if (request) setView('new');
    } catch {
      setPending(null);
    }
  }, [storageKey]);

  const models = options.data?.models ?? [];
  const selectedModel =
    models.find((item) => item.id === (pending?.model ?? model)) ??
    models.find((item) => item.id === (history[0]?.request.model ?? options.data?.default_model)) ??
    models[0];
  const requestedPower = pending?.reasoning_effort ?? (power || history[0]?.request.reasoning_effort || '');
  const selectedPower = selectedModel?.reasoning_efforts.includes(requestedPower)
    ? requestedPower
    : (selectedModel?.default_reasoning_effort ?? 'medium');
  const simulationReady = options.data?.simulation_available ?? options.data?.configured ?? false;
  const retryPending = !!pending?.source_scene_id;
  const reusedSceneId = selectedJob?.request.source_scene_id ?? run?.request.source_scene_id;
  const sourceJob = reusedSceneId
    ? history.find((job) => job.result?.scene_id === reusedSceneId || job.request.request_id === reusedSceneId)
    : undefined;
  const locked = busy || !!pending;
  const datasetName = (source: string) => sources.data?.find((item) => item.job_id === source)?.name ?? 'Dataset';
  const openNew = () => {
    setError('');
    setView('new');
  };

  async function generate(reuse?: SceneRequest) {
    if (
      !projectId ||
      busy ||
      (reuse && running) ||
      ((reuse || retryPending) && !simulationReady) ||
      (!pending && !reuse && (!selected || !selectedModel || running))
    )
      return;
    setBusy(true);
    setError('');
    try {
      const saved = localStorage.getItem(storageKey);
      const body: SceneRequest = saved
        ? JSON.parse(saved)
        : (reuse ?? {
            request_id: crypto.randomUUID().replaceAll('-', ''),
            source_job_id: selected!.job_id!,
            scope: 'dataset',
            camera_key: 'observation.images.front',
            max_attempts: attempts,
            model: selectedModel!.id,
            reasoning_effort: selectedPower,
          });
      localStorage.setItem(storageKey, JSON.stringify(body));
      setPending(body);
      const created = await agentRequest<SceneJob>(
        `/projects/${encodeURIComponent(projectId)}/scene-jobs`,
        'POST',
        body,
      );
      if (
        typeof created.id !== 'string' ||
        !created.id ||
        created.project_id !== projectId ||
        created.kind !== 'scene.generate' ||
        created.request?.request_id !== body.request_id ||
        created.request?.source_job_id !== body.source_job_id ||
        (created.request?.source_scene_id ?? null) !== (body.source_scene_id ?? null) ||
        created.request?.scope !== 'dataset' ||
        created.request?.camera_key !== body.camera_key ||
        created.request?.max_attempts !== body.max_attempts ||
        (body.model != null && created.request?.model !== body.model) ||
        created.request?.reasoning_effort !== (body.reasoning_effort ?? 'medium')
      )
        throw new Error('Job receipt could not be verified. Retry this request to check its status.');
      client.setQueryData<SceneJob[]>(['jobs', projectId], (previous) => [
        created,
        ...(previous ?? []).filter((job) => job.id !== created.id),
      ]);
      localStorage.removeItem(storageKey);
      setPending(null);
      setSelectedId(created.id);
      setView('detail');
      await jobs.refetch();
    } catch (problem) {
      if (problem instanceof AgentRequestError && problem.status >= 400 && problem.status < 500) {
        localStorage.removeItem(storageKey);
        setPending(null);
      }
      setError(problem instanceof Error ? problem.message : 'Job request could not be verified.');
      await jobs.refetch();
    } finally {
      setBusy(false);
    }
  }

  async function refresh() {
    await jobs.refetch();
    if (selectedJob) await scene.refetch();
    if (legacyScene) await legacy.refetch();
  }

  return (
    <section className="scene-generation" aria-label="Scene generation">
      {view === 'jobs' ? (
        <>
          <JobHistory
            title="Real to sim jobs"
            newLabel="New job"
            entries={history.map((job) => ({
              id: job.id,
              title: datasetName(job.request.source_job_id),
              subtitle: [job.request.model, powerLabels[job.request.reasoning_effort ?? 'medium']]
                .filter(Boolean)
                .join(' · '),
              status: job.status,
              createdAt: job.created_at,
              progress: isActive(job) ? (job.stage ?? job.status).replaceAll('_', ' ') : undefined,
            }))}
            onNew={openNew}
            onSelect={(id) => {
              setSelectedId(id);
              setError('');
              setView('detail');
            }}
            loading={jobs.isPending && !!projectId}
            error={jobs.error}
            disabled={!projectId || busy || !!pending}
            emptyMessage={projectId ? 'No real to sim jobs yet.' : 'Select a project to see its jobs.'}
          />
          {pending && (
            <Notice
              tone="warning"
              title="Unconfirmed request"
              actions={
                <button type="button" className="secondary-button" onClick={openNew}>
                  Review request
                </button>
              }
            />
          )}
          {!!savedScenes.length && (
            <details className="panel scene-previous">
              <summary>Saved scenes ({savedScenes.length})</summary>
              <ul>
                {savedScenes.map((item) => (
                  <li key={item.id}>
                    <button
                      type="button"
                      className="text-button"
                      onClick={() => {
                        setSelectedId(`legacy:${item.id}`);
                        setView('detail');
                      }}
                    >
                      {datasetName(item.request.source_job_id)}
                      {item.request.episode_index !== undefined ? ` · Episode ${item.request.episode_index}` : ''}
                    </button>
                    <StatusBadge tone="warning">{item.status.replaceAll('_', ' ')}</StatusBadge>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </>
      ) : view === 'new' ? (
        <section className="panel scene-compose">
          <header className="scene-heading">
            <button type="button" className="text-button" onClick={() => setView('jobs')}>
              ← Back to jobs
            </button>
            <h2>{pending?.source_scene_id ? 'Retry simulation' : 'New job'}</h2>
          </header>
          {options.isError && <Notice tone="danger" title="Scene setup unavailable" live="alert" />}
          {options.data && !(retryPending ? simulationReady : options.data.configured) && (
            <Notice
              tone="warning"
              title={retryPending ? 'Simulation setup required' : 'Scene setup required'}
              live="status"
              actions={
                <Link className="secondary-button" href="/guide/#real-to-sim">
                  Setup
                </Link>
              }
            />
          )}
          {sources.isError && <Notice tone="danger" title="Datasets unavailable" live="alert" />}
          {!projectId && <Notice title="Select a project" />}
          {projectId && sources.isSuccess && candidates.length === 0 && (
            <Notice
              title="No supported dataset"
              actions={
                <Link className="secondary-button" href="/datasets/">
                  Datasets
                </Link>
              }
            />
          )}
          <form
            onSubmit={(event) => {
              event.preventDefault();
              void generate();
            }}
          >
            <div className="scene-fields">
              <label>
                Dataset
                <select
                  aria-label="Scene dataset"
                  value={pending?.source_job_id ?? selected?.job_id ?? ''}
                  disabled={locked || !candidates.length}
                  onChange={(event) => setSourceId(event.target.value)}
                >
                  {pending && !candidates.some((item) => item.job_id === pending.source_job_id) && (
                    <option value={pending.source_job_id}>Saved dataset request</option>
                  )}
                  {candidates.length ? (
                    candidates.map((source) => (
                      <option key={source.id} value={source.job_id}>
                        {source.name}
                      </option>
                    ))
                  ) : (
                    <option value="">No supported dataset</option>
                  )}
                </select>
                {selected?.profile && (
                  <span className="scene-field-meta">All {selected.profile.total_episodes} episodes</span>
                )}
              </label>
              <label>
                Model
                <select
                  aria-label="Scene agent model"
                  value={pending?.model ?? selectedModel?.id ?? ''}
                  disabled={locked || !models.length}
                  onChange={(event) => {
                    const next = models.find((item) => item.id === event.target.value);
                    setModel(event.target.value);
                    setPower(
                      next?.reasoning_efforts.includes(selectedPower)
                        ? selectedPower
                        : (next?.default_reasoning_effort ?? 'medium'),
                    );
                  }}
                >
                  {pending?.model && !models.some((item) => item.id === pending.model) && (
                    <option value={pending.model}>{pending.model}</option>
                  )}
                  {models.length ? (
                    models.map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.label}
                      </option>
                    ))
                  ) : (
                    <option value="">Loading models…</option>
                  )}
                </select>
              </label>
              <label>
                Power
                <select
                  aria-label="Scene agent power"
                  value={pending?.reasoning_effort ?? selectedPower}
                  disabled={locked || !selectedModel}
                  onChange={(event) => setPower(event.target.value)}
                >
                  {pending?.reasoning_effort &&
                    !selectedModel?.reasoning_efforts.includes(pending.reasoning_effort) && (
                      <option value={pending.reasoning_effort}>
                        {powerLabels[pending.reasoning_effort] ?? pending.reasoning_effort}
                      </option>
                    )}
                  {selectedModel?.reasoning_efforts.map((effort) => (
                    <option key={effort} value={effort}>
                      {powerLabels[effort] ?? effort}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="scene-form-footer">
              <details className="scene-options">
                <summary>Advanced</summary>
                <label>
                  Attempt limit
                  <select
                    aria-label="Scene attempt limit"
                    value={pending?.max_attempts ?? attempts}
                    disabled={locked}
                    onChange={(event) => setAttempts(Number(event.target.value))}
                  >
                    {[1, 2, 3].map((value) => (
                      <option key={value} value={value}>
                        {value}
                      </option>
                    ))}
                  </select>
                </label>
              </details>
              <button
                type="submit"
                className="primary-button"
                disabled={
                  busy ||
                  (retryPending && !simulationReady) ||
                  (!pending && (running || !selected || !selectedModel || !options.data?.configured))
                }
              >
                {busy ? 'Starting…' : pending ? 'Check request' : 'Start job'}
              </button>
            </div>
          </form>
          {running && <Notice title="A job is running" />}
          {pending && (
            <Notice tone="warning" title="Unconfirmed request" live="status">
              {pending.model ?? 'Codex default'} ·{' '}
              {powerLabels[pending.reasoning_effort ?? 'medium'] ?? pending.reasoning_effort}
            </Notice>
          )}
          {error && <Notice tone="danger" title="Request failed" details={error} live="alert" />}
        </section>
      ) : (
        <section className="panel scene-run" aria-label="Real to sim job details">
          <header className="scene-heading">
            <button type="button" className="text-button" onClick={() => setView('jobs')}>
              ← Back to jobs
            </button>
            <div className="scene-actions">
              <button type="button" className="text-button" onClick={() => void refresh()}>
                Refresh
              </button>
              <button type="button" className="secondary-button" onClick={openNew} disabled={busy || !!pending}>
                New job
              </button>
            </div>
          </header>
          <div className="scene-job-heading">
            <h2>
              {selectedJob
                ? datasetName(selectedJob.request.source_job_id)
                : legacyScene
                  ? datasetName(legacyScene.request.source_job_id)
                  : 'Job'}
            </h2>
            {selectedJob && <span className={`status status-${selectedJob.status}`}>{selectedJob.status}</span>}
            {run && !selectedJob && <StatusBadge tone="warning">{run.status.replaceAll('_', ' ')}</StatusBadge>}
          </div>
          {selectedJob && isActive(selectedJob) && (
            <div className="scene-current-stage" role="status">
              <span>{(run?.stage ?? selectedJob.stage ?? 'queued').replaceAll('_', ' ')}</span>
              <progress aria-label="Scene job in progress" />
            </div>
          )}
          {selectedJob?.error && <Notice tone="danger" title="Job failed" details={selectedJob.error} live="alert" />}
          {selectedJob && isActive(selectedJob) && (
            <button
              type="button"
              className="secondary-button"
              disabled={busy}
              onClick={async () => {
                setBusy(true);
                setError('');
                try {
                  await api.cancel(selectedJob.id);
                  await refresh();
                } catch (problem) {
                  setError(problem instanceof Error ? problem.message : 'Cancellation could not be verified.');
                } finally {
                  setBusy(false);
                }
              }}
            >
              {busy ? 'Cancelling…' : 'Cancel job'}
            </button>
          )}
          {selectedJob &&
            run &&
            ['failed', 'interrupted'].includes(selectedJob.status) &&
            ['source_evidence', 'agent_review', 'usd_structure'].every((gate) => run.gates[gate] === true) && (
              <button
                type="button"
                className="primary-button"
                disabled={busy || running || !!pending || !simulationReady}
                onClick={() =>
                  void generate({
                    request_id: crypto.randomUUID().replaceAll('-', ''),
                    source_scene_id: run.id,
                    source_job_id: selectedJob.request.source_job_id,
                    scope: 'dataset',
                    camera_key: selectedJob.request.camera_key,
                    max_attempts: selectedJob.request.max_attempts,
                    model: selectedJob.request.model,
                    reasoning_effort: selectedJob.request.reasoning_effort ?? 'medium',
                  })
                }
              >
                Retry simulation
              </button>
            )}
          {pending && (
            <Notice
              tone="warning"
              title="Unconfirmed request"
              live="status"
              actions={
                <button
                  type="button"
                  className="secondary-button"
                  disabled={busy || (retryPending && !simulationReady)}
                  onClick={() => void generate()}
                >
                  Check request
                </button>
              }
            />
          )}
          {error && <Notice tone="danger" title="Request failed" details={error} live="alert" />}
          {run ? (
            <SceneResults
              run={run}
              job={selectedJob}
              onSourceJob={
                sourceJob
                  ? () => {
                      setSelectedId(sourceJob.id);
                      setError('');
                    }
                  : undefined
              }
            />
          ) : (
            <Notice
              title={
                selectedJob && isActive(selectedJob)
                  ? 'Simulation results pending'
                  : scene.isError
                    ? 'Results unavailable'
                    : 'Loading results…'
              }
            />
          )}
        </section>
      )}
    </section>
  );
}

function SceneResults({ run, job, onSourceJob }: { run: Run; job?: SceneJob; onSourceJob?: () => void }) {
  const media = run.media ?? [];
  const images = media.filter((item) => item.role === 'environment_image');
  const replays = media.filter((item) => item.role === 'episode_replay');
  const evidence = media.filter((item) => item.role === 'source_image' || item.role === 'source_video');
  return (
    <>
      <div className="scene-result-heading">
        <h3>Results</h3>
        <StatusBadge
          tone={run.status === 'failed' ? 'danger' : run.status.startsWith('needs_') ? 'warning' : 'neutral'}
        >
          {(job?.result?.validation_status ?? run.status).replaceAll('_', ' ')}
        </StatusBadge>
        {run.artifacts.includes('scene-bundle.zip') && (
          <a className="secondary-button" href={artifactUrl(run.id, 'scene-bundle.zip', job?.id)}>
            Download scene
          </a>
        )}
      </div>
      {run.source?.resource_cleanup === 'unverified' && (
        <Notice tone="warning" title="Compute cleanup unverified" live="status" />
      )}
      {!!images.length && (
        <section aria-label="Environment images">
          <h3>Environment</h3>
          <div className="scene-environment">
            {images.map((item) => (
              <figure key={item.path}>
                <a href={artifactUrl(run.id, item.path, job?.id)} target="_blank" rel="noreferrer">
                  <img src={artifactUrl(run.id, item.path, job?.id)} alt={item.label} loading="lazy" />
                </a>
                <figcaption>{item.label}</figcaption>
              </figure>
            ))}
          </div>
        </section>
      )}
      {!!replays.length && (
        <section className="scene-replays" aria-label="Episode replay">
          <h3>Episode replay</h3>
          {replays.map((item) => (
            <figure key={item.path}>
              <video
                controls
                muted
                playsInline
                loop
                preload="metadata"
                src={artifactUrl(run.id, item.path, job?.id)}
                aria-label={item.label}
              />
              <figcaption>{item.label}</figcaption>
            </figure>
          ))}
        </section>
      )}
      {!images.length && !replays.length && (
        <Notice title={job && isActive(job) ? 'Simulation media pending' : 'No simulation media'} />
      )}
      <details className="scene-details">
        <summary>Details</summary>
        <p>{run.message}</p>
        {(job?.request.source_scene_id ?? run.request.source_scene_id) && (
          <p>
            Reused reconstruction
            {onSourceJob && (
              <>
                {' '}
                ·{' '}
                <button type="button" className="text-button" onClick={onSourceJob}>
                  Original job
                </button>
              </>
            )}
          </p>
        )}
        <p>
          Agent: {run.agent?.model ?? run.request.model ?? 'Codex default'} · Power:{' '}
          {powerLabels[run.agent?.reasoning_effort ?? run.request.reasoning_effort ?? 'medium']} · Codex CLI
        </p>
        <p>
          Compute:{' '}
          {run.source?.resource_cleanup === 'cluster_removed'
            ? 'Released'
            : run.source?.resource_cleanup === 'not_launched'
              ? 'Not allocated'
              : 'Unverified'}
        </p>
        {run.source?.total_episodes !== undefined && (
          <p>
            Dataset: {run.source.total_episodes} episodes
            {run.source.inspected_episodes?.length ? ` · Inspected: ${run.source.inspected_episodes.join(', ')}` : ''}
            {run.source.replay_episode !== undefined ? ` · Replay: ${run.source.replay_episode}` : ''}
          </p>
        )}
        <ul className="scene-gates" aria-label="Scene validation">
          {Object.entries(run.gates).map(([key, passed]) => (
            <li key={key}>
              <span>{labels[key] ?? key}: </span>
              <StatusBadge tone={passed ? 'success' : 'neutral'}>{passed ? 'passed' : 'unverified'}</StatusBadge>
            </li>
          ))}
        </ul>
        {!!evidence.length && (
          <section aria-label="Dataset evidence">
            <h3>Dataset evidence</h3>
            <div className="scene-evidence">
              {evidence.map((item) =>
                item.role === 'source_video' ? (
                  <video
                    key={item.path}
                    controls
                    muted
                    playsInline
                    preload="metadata"
                    src={artifactUrl(run.id, item.path, job?.id)}
                    aria-label={item.label}
                  />
                ) : (
                  <a key={item.path} href={artifactUrl(run.id, item.path, job?.id)} target="_blank" rel="noreferrer">
                    <img src={artifactUrl(run.id, item.path, job?.id)} alt={item.label} loading="lazy" />
                  </a>
                ),
              )}
            </div>
          </section>
        )}
        <h3>Files</h3>
        <ul className="scene-artifacts">
          {run.artifacts
            .filter((name) => !media.some((item) => item.path === name) && name !== 'scene-bundle.zip')
            .map((name) => (
              <li key={name}>
                <a href={artifactUrl(run.id, name, job?.id)}>{name}</a>
              </li>
            ))}
        </ul>
        <h3>
          History · {run.attempt}/{run.request.max_attempts}
        </h3>
        <ol>
          {run.events.map((event, index) => (
            <li key={index}>{event.message}</li>
          ))}
        </ol>
        {Object.keys(run.usage).length > 0 && (
          <p>
            Tokens: {run.usage.input_tokens ?? 0} input · {run.usage.output_tokens ?? 0} output
          </p>
        )}
        <p>
          Job: {job?.id ?? 'Saved scene'} · Scene: {run.id}
        </p>
      </details>
    </>
  );
}
