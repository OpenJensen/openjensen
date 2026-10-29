'use client';

import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { AgentRequestError, agentRequest } from '@/lib/agent-api';
import { apiOrigin } from '@/lib/api';
import Link from 'next/link';
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
  codex_available: boolean;
  builder_available: boolean;
  supported_repo: string;
  supported_revision: string;
  message: string;
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
  episode_index: number;
  camera_key: string;
  max_attempts: number;
  model?: string | null;
  reasoning_effort?: string;
};
type Run = {
  id: string;
  status: string;
  stage: string;
  attempt: number;
  message: string;
  request: SceneRequest;
  gates: Record<string, boolean>;
  events: { at: string; stage: string; message: string }[];
  artifacts: string[];
  usage: Record<string, number>;
  agent?: { harness: string; model: string; reasoning_effort: string } | null;
};
const active = (run: Run) => run.status === 'queued' || run.status === 'running';
const labels: Record<string, string> = {
  source_evidence: 'Real episode evidence',
  agent_review: 'Agent parameter review',
  usd_structure: 'USD structure',
  isaac_runtime: 'Isaac runtime',
  visual_match: 'Rendered image match',
  calibrated_replay: 'Calibrated action replay',
};
const artifactUrl = (id: string, name: string) =>
  `${apiOrigin}/api/v1/scenes/${id}/artifact?name=${encodeURIComponent(name)}`;

export function SceneGenerationPanel({ projectId }: { projectId?: string }) {
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
  const runs = useQuery({
    queryKey: ['scene-runs', projectId],
    queryFn: () => agentRequest<Run[]>(`/projects/${encodeURIComponent(projectId!)}/scenes`),
    enabled: !!projectId,
    retry: false,
    refetchInterval: (query) => (query.state.data?.some(active) ? 2500 : false),
  });
  const [sourceId, setSourceId] = useState('');
  const [episode, setEpisode] = useState(3);
  const [attempts, setAttempts] = useState(2);
  const [model, setModel] = useState('');
  const [power, setPower] = useState('');
  const [pending, setPending] = useState<SceneRequest | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const candidates =
    sources.data?.filter(
      (item) =>
        item.status === 'ready' &&
        item.job_id &&
        item.profile?.repo_id === options.data?.supported_repo &&
        item.profile?.revision === options.data?.supported_revision,
    ) ?? [];
  const selected = candidates.find((item) => item.job_id === (pending?.source_job_id ?? sourceId)) ?? candidates[0];
  const latest = runs.data?.[0];
  const running = runs.data?.some(active);
  const storageKey = `jensen-scene-request:${projectId}`;

  useEffect(() => {
    try {
      const saved = localStorage.getItem(storageKey);
      setPending(saved ? JSON.parse(saved) : null);
    } catch {
      setPending(null);
    }
  }, [storageKey]);
  const models = options.data?.models ?? [];
  const selectedModel =
    models.find((item) => item.id === (pending?.model ?? model)) ??
    models.find((item) => item.id === (latest?.agent?.model ?? options.data?.default_model)) ??
    models[0];
  const requestedPower = pending?.reasoning_effort ?? (power || latest?.agent?.reasoning_effort || '');
  const selectedPower = selectedModel?.reasoning_efforts.includes(requestedPower)
    ? requestedPower
    : (selectedModel?.default_reasoning_effort ?? 'medium');
  const locked = busy || running || !!pending;

  async function generate() {
    if (!projectId || !selected || !selectedModel || busy || running) return;
    setBusy(true);
    setError('');
    try {
      const saved = localStorage.getItem(storageKey);
      const body: SceneRequest = saved
        ? JSON.parse(saved)
        : {
            request_id: crypto.randomUUID().replaceAll('-', ''),
            source_job_id: selected.job_id!,
            episode_index: episode,
            camera_key: 'observation.images.front',
            max_attempts: attempts,
            model: selectedModel.id,
            reasoning_effort: selectedPower,
          };
      // Retain exact request before the mutation. A retry reconciles the same durable run.
      localStorage.setItem(storageKey, JSON.stringify(body));
      setPending(body);
      await agentRequest<Run>(`/projects/${encodeURIComponent(projectId)}/scenes`, 'POST', body);
      localStorage.removeItem(storageKey);
      setPending(null);
      await runs.refetch();
    } catch (problem) {
      if (problem instanceof AgentRequestError && problem.status >= 400 && problem.status < 500) {
        localStorage.removeItem(storageKey);
        setPending(null);
      }
      setError(problem instanceof Error ? problem.message : 'Scene request could not be verified.');
      await runs.refetch();
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="scene-generation" aria-label="Scene generation">
      <section className="panel scene-compose">
        <header className="scene-heading">
          <h2>New scene</h2>
        </header>
        {options.isError && <Notice tone="danger" title="Scene setup unavailable" live="alert" />}
        {options.data && !options.data.configured && (
          <Notice
            tone="warning"
            title="Scene setup required"
            live="status"
            details={[
              !options.data.codex_available && 'Codex is unavailable.',
              !options.data.builder_available && 'Scene builder is unavailable.',
            ]
              .filter(Boolean)
              .join(' ')}
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
                value={selected?.job_id ?? ''}
                disabled={locked || !candidates.length}
                onChange={(event) => setSourceId(event.target.value)}
              >
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
            </label>
            <label>
              Episode
              <input
                aria-label="Scene episode"
                type="number"
                min={0}
                max={(selected?.profile?.total_episodes ?? 1) - 1}
                value={pending?.episode_index ?? episode}
                disabled={locked}
                onChange={(event) => setEpisode(Number(event.target.value))}
              />
            </label>
            <label>
              Model
              <select
                aria-label="Scene agent model"
                value={selectedModel?.id ?? ''}
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
                value={selectedPower}
                disabled={locked || !selectedModel}
                onChange={(event) => setPower(event.target.value)}
              >
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
                running ||
                !selected ||
                !selectedModel ||
                !options.data?.configured ||
                episode < 0 ||
                episode >= (selected?.profile?.total_episodes ?? 0)
              }
            >
              {busy ? 'Saving…' : running ? 'Generating…' : 'Generate scene'}
            </button>
          </div>
        </form>
        {pending && (
          <Notice tone="warning" title="Unconfirmed request" live="status">
            {pending.model ?? 'Codex default'} ·{' '}
            {powerLabels[pending.reasoning_effort ?? 'medium'] ?? pending.reasoning_effort}
          </Notice>
        )}
        {error && <Notice tone="danger" title="Request failed" details={error} live="alert" />}
      </section>
      {runs.isError && <Notice tone="danger" title="Scene history unavailable" live="alert" />}
      {latest && (
        <section className="panel scene-run" aria-live="polite">
          <header className="scene-heading">
            <div className="scene-run-title">
              <h2>Episode {latest.request.episode_index}</h2>
              <StatusBadge
                tone={
                  latest.status === 'failed'
                    ? 'danger'
                    : active(latest)
                      ? 'info'
                      : latest.status.startsWith('needs_')
                        ? 'warning'
                        : 'neutral'
                }
              >
                {latest.status.replaceAll('_', ' ')}
              </StatusBadge>
            </div>
            <div className="scene-actions">
              <button type="button" className="text-button" onClick={() => void runs.refetch()}>
                Refresh
              </button>
              {latest.artifacts.includes('scene-bundle.zip') && (
                <a className="secondary-button" href={artifactUrl(latest.id, 'scene-bundle.zip')}>
                  Download scene
                </a>
              )}
              {active(latest) && (
                <button
                  type="button"
                  className="secondary-button"
                  disabled={busy}
                  onClick={async () => {
                    setBusy(true);
                    setError('');
                    try {
                      await agentRequest(`/scenes/${latest.id}/cancel`, 'POST', {});
                      await runs.refetch();
                    } catch (problem) {
                      setError(problem instanceof Error ? problem.message : 'Cancellation could not be verified.');
                    } finally {
                      setBusy(false);
                    }
                  }}
                >
                  Cancel
                </button>
              )}
            </div>
          </header>
          {active(latest) && (
            <p className="scene-progress" role="status">
              {latest.stage.replaceAll('_', ' ')}
            </p>
          )}
          {latest.status === 'failed' && (
            <Notice tone="danger" title="Scene failed" details={latest.message} live="alert" />
          )}
          <ul className="scene-gates" aria-label="Scene validation">
            {Object.entries(latest.gates).map(([key, passed]) => (
              <li key={key} data-passed={passed}>
                <span>{labels[key] ?? key}: </span>
                <StatusBadge tone={passed ? 'success' : 'neutral'}>{passed ? 'passed' : 'unverified'}</StatusBadge>
              </li>
            ))}
          </ul>
          {latest.artifacts.some((name) => name.endsWith('.jpg')) && (
            <div className="scene-evidence">
              {latest.artifacts
                .filter((name) => name.endsWith('.jpg'))
                .map((name, index) => (
                  <a key={name} href={artifactUrl(latest.id, name)} target="_blank" rel="noreferrer">
                    <img src={artifactUrl(latest.id, name)} alt={`Real episode sample ${index + 1}`} loading="lazy" />
                  </a>
                ))}
            </div>
          )}
          <details className="scene-details">
            <summary>Details</summary>
            <p>{latest.message}</p>
            <p>
              Agent: {latest.agent?.model ?? latest.request.model ?? 'Codex default (legacy run)'} · Power:{' '}
              {powerLabels[latest.agent?.reasoning_effort ?? latest.request.reasoning_effort ?? 'medium']} · Codex CLI
            </p>
            <h3>Files</h3>
            <ul className="scene-artifacts">
              {latest.artifacts
                .filter((name) => !name.endsWith('.jpg') && name !== 'scene-bundle.zip')
                .map((name) => (
                  <li key={name}>
                    <a href={artifactUrl(latest.id, name)}>{name}</a>
                  </li>
                ))}
            </ul>
            <h3>
              History · {latest.attempt}/{latest.request.max_attempts}
            </h3>
            <ol>
              {latest.events.map((event, index) => (
                <li key={index}>{event.message}</li>
              ))}
            </ol>
            {Object.keys(latest.usage).length > 0 && (
              <p>
                Tokens: {latest.usage.input_tokens ?? 0} input · {latest.usage.output_tokens ?? 0} output
              </p>
            )}
          </details>
        </section>
      )}
      {!!runs.data?.length && runs.data.length > 1 && (
        <details className="panel scene-previous">
          <summary>Previous scenes ({runs.data.length - 1})</summary>
          <ul>
            {runs.data.slice(1).map((run) => (
              <li key={run.id}>
                <span>Episode {run.request.episode_index}</span>
                <StatusBadge>{run.status.replaceAll('_', ' ')}</StatusBadge>
                {run.artifacts.includes('scene-bundle.zip') && (
                  <a href={artifactUrl(run.id, 'scene-bundle.zip')}>Download scene</a>
                )}
              </li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}
