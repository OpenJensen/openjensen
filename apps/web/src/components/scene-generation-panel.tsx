'use client';

import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { AgentRequestError, agentRequest } from '@/lib/agent-api';
import { apiOrigin } from '@/lib/api';
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
      <div className="scene-generation-heading">
        <div>
          <h2>Generate a scene</h2>
          <p>Turn a real dataset episode into an Isaac Sim scene with Codex.</p>
        </div>
        <span className="scene-profile">SO101 pickup</span>
      </div>
      <p>
        Agents fit and review the camera and object layout. A trusted builder creates the robot, colliders and USD
        assets. Isaac rendering and physics validation run separately on a simulation worker.
      </p>
      {options.isError && <p role="alert">Scene runtime status could not be loaded.</p>}
      {options.data && !options.data.configured && (
        <p role="status">
          Scene runtime needs setup:{' '}
          {options.data.codex_available ? 'Codex is available. ' : 'Install and sign in to Codex. '}
          {!options.data.builder_available && 'Configure an OpenUSD builder Python on the service host.'}
        </p>
      )}
      {sources.isError && <p role="alert">Dataset sources could not be loaded.</p>}
      {!projectId && <p>Select a project to generate a scene.</p>}
      {projectId && sources.isSuccess && candidates.length === 0 && (
        <p>
          This first scene profile uses the original codywang/so101_pickup_test dataset at its inspected revision.
          Inspect that dataset in this project to begin.
        </p>
      )}
      <form
        onSubmit={(event) => {
          event.preventDefault();
          void generate();
        }}
      >
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
        <button
          type="submit"
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
      </form>
      <p className="scene-generation-note">
        Model and power apply to both reconstruction and review. Higher power spends more time and tokens reasoning; it
        does not change the simulation GPU. Choices follow the installed Codex catalog; API access depends on your
        account.
      </p>
      {pending && (
        <p role="status">
          Unconfirmed request: {pending.model ?? 'Codex default'} ·{' '}
          {powerLabels[pending.reasoning_effort ?? 'medium'] ?? pending.reasoning_effort}. Retry keeps these saved
          settings.
        </p>
      )}
      <p className="scene-generation-note">
        Sends five source frames and joint samples to OpenAI through Codex. Each attempt uses a reconstruction role and
        an independent review role. Generation uses your configured API key or the service host’s Codex login.
      </p>
      {error && (
        <p role="alert">
          {error} {pending && 'Retry reconciles the saved request.'}
        </p>
      )}
      {runs.isError && <p role="alert">Saved scene runs could not be loaded.</p>}
      {latest && (
        <div className="scene-run" aria-live="polite">
          <div className="scene-run-heading">
            <h3>
              Episode {latest.request.episode_index} · {latest.status.replaceAll('_', ' ')}
            </h3>
            <button type="button" onClick={() => void runs.refetch()}>
              Refresh
            </button>
            {active(latest) && (
              <button
                type="button"
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
          <p>{latest.message}</p>
          <p className="scene-generation-note">
            Agent: {latest.agent?.model ?? latest.request.model ?? 'Codex default (legacy run)'} · Power:{' '}
            {powerLabels[latest.agent?.reasoning_effort ?? latest.request.reasoning_effort ?? 'medium']} · Codex CLI
          </p>
          <ul className="scene-gates">
            {Object.entries(latest.gates).map(([key, passed]) => (
              <li key={key} data-passed={passed}>
                <span aria-hidden="true">{passed ? '✓' : '○'}</span> {labels[key] ?? key}:{' '}
                {passed ? 'passed' : 'unverified'}
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
          <div className="scene-artifacts">
            {latest.artifacts
              .filter((name) => !name.endsWith('.jpg'))
              .map((name) => (
                <a key={name} href={artifactUrl(latest.id, name)}>
                  {name === 'scene-bundle.zip' ? 'Download Isaac scene bundle' : name}
                </a>
              ))}
          </div>
          <details>
            <summary>
              Run history · attempt {latest.attempt}/{latest.request.max_attempts}
            </summary>
            <ol>
              {latest.events.map((event, index) => (
                <li key={index}>{event.message}</li>
              ))}
            </ol>
            {Object.keys(latest.usage).length > 0 && (
              <p>
                Agent tokens: {latest.usage.input_tokens ?? 0} input · {latest.usage.output_tokens ?? 0} output
              </p>
            )}
          </details>
          <p className="scene-generation-note">
            A structurally valid scene still needs an Isaac render and physics check. Image fidelity and action replay
            require their own acceptance evidence.
          </p>
        </div>
      )}
      {!!runs.data?.length && runs.data.length > 1 && (
        <details>
          <summary>Previous scenes ({runs.data.length - 1})</summary>
          <ul>
            {runs.data.slice(1).map((run) => (
              <li key={run.id}>
                Episode {run.request.episode_index} · {run.status.replaceAll('_', ' ')}{' '}
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
