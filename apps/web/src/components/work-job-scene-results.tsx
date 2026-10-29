'use client';

import { useQuery } from '@tanstack/react-query';
import type { components } from '@/lib/api.generated';
import { apiOrigin } from '@/lib/api';
import { AgentRequestError } from '@/lib/agent-api';
import { isWorkActive, type WorkJob } from '@/lib/work-jobs';
import { Notice } from './ui/notice';
import { StatusBadge } from './ui/status-badge';
import './scene-generation-panel.css';

type Scene = components['schemas']['SceneRun'];

function scenePath(job: WorkJob) {
  const kind = job.id.split(':', 1)[0];
  if (kind === 'core') return `/jobs/${encodeURIComponent(job.source_id)}/scene`;
  if (kind === 'scene') return `/scenes/${encodeURIComponent(job.source_id)}`;
  return null;
}

async function readScene(job: WorkJob, path: string, signal: AbortSignal): Promise<Scene> {
  const response = await fetch(`${apiOrigin}/api/v1${path}`, { cache: 'no-store', signal });
  if (!response.ok) throw new AgentRequestError('Simulation results are unavailable.', response.status);
  const value = await response.json();
  if (
    !value ||
    typeof value !== 'object' ||
    Array.isArray(value) ||
    value.project_id !== job.project_id ||
    (job.id.startsWith('core:') ? value.job_id !== job.source_id : value.id !== job.source_id) ||
    typeof value.id !== 'string' ||
    (value.media !== undefined && !Array.isArray(value.media)) ||
    (value.artifacts !== undefined && !Array.isArray(value.artifacts))
  )
    throw new Error('Simulation result ownership could not be verified.');
  return value as Scene;
}

export function WorkJobSceneResults({ job }: { job: WorkJob }) {
  const path = scenePath(job);
  const active = isWorkActive(job);
  const result = useQuery({
    // A terminal transition fetches the final media even between active polls.
    queryKey: ['work-job-scene', job.project_id, job.id, job.status, job.source_status],
    queryFn: ({ signal }) => readScene(job, path!, signal),
    enabled: !!path,
    retry: false,
    refetchInterval: active ? 2000 : false,
  });
  if (!path) return <Notice title="Simulation results unavailable" />;
  if (result.isPending) return <Notice title="Loading simulation results…" live="status" />;
  if (result.error) {
    if (active && result.error instanceof AgentRequestError && result.error.status === 404)
      return <Notice title="Simulation results pending" live="status" />;
    return (
      <Notice
        tone="danger"
        title="Simulation results unavailable"
        live="alert"
        actions={
          <button type="button" className="secondary-button" onClick={() => void result.refetch()}>
            Refresh
          </button>
        }
      />
    );
  }
  const scene = result.data;
  const registered = new Set(scene.artifacts ?? []);
  const media = (scene.media ?? []).filter(
    (item) =>
      item &&
      typeof item.path === 'string' &&
      typeof item.label === 'string' &&
      registered.has(item.path) &&
      !item.path.startsWith('/') &&
      !/[\\\0]/.test(item.path) &&
      item.path.split('/').every((part) => part && part !== '.' && part !== '..'),
  );
  const images = media.filter((item) => item.role === 'environment_image').slice(0, 3);
  const replay = media.find((item) => item.role === 'episode_replay');
  const artifactUrl = (name: string) => `${apiOrigin}/api/v1${path}/artifact?name=${encodeURIComponent(name)}`;
  const runtime = scene.gates?.isaac_runtime === true;
  const calibrated = scene.gates?.calibrated_replay === true;
  const cleanup = scene.source?.resource_cleanup;
  return (
    <div className="work-job-scene-results">
      <div className="workbench-actions">
        <StatusBadge tone={calibrated ? 'success' : runtime ? 'warning' : 'neutral'}>
          {calibrated ? 'Calibrated replay' : runtime ? 'Calibration unverified' : 'Simulation unverified'}
        </StatusBadge>
        {registered.has('scene-bundle.zip') && (
          <a className="secondary-button" href={artifactUrl('scene-bundle.zip')}>
            Download scene
          </a>
        )}
      </div>
      {cleanup === 'unverified' && <Notice tone="warning" title="Compute cleanup unverified" live="status" />}
      {!!images.length && (
        <section aria-label="Environment images">
          <h4>Environment</h4>
          <div className="scene-environment">
            {images.map((item) => (
              <figure key={item.path}>
                <a href={artifactUrl(item.path)} target="_blank" rel="noreferrer">
                  <img src={artifactUrl(item.path)} alt={item.label} loading="lazy" />
                </a>
                <figcaption>{item.label}</figcaption>
              </figure>
            ))}
          </div>
        </section>
      )}
      {replay && (
        <section className="scene-replays" aria-label="Episode replay">
          <h4>Episode replay</h4>
          <figure>
            <video
              controls
              muted
              playsInline
              loop
              preload="metadata"
              src={artifactUrl(replay.path)}
              aria-label={replay.label}
            />
            <figcaption>{replay.label}</figcaption>
          </figure>
        </section>
      )}
      {!images.length && !replay && (
        <Notice
          title={active ? 'Simulation media pending' : 'No simulation media'}
          live={active ? 'status' : undefined}
        />
      )}
      <details className="scene-details">
        <summary>Simulation details</summary>
        <dl className="work-job-facts">
          <div>
            <dt>Compute</dt>
            <dd>
              {cleanup === 'cluster_removed' ? 'Released' : cleanup === 'not_launched' ? 'Not allocated' : 'Unverified'}
            </dd>
          </div>
          {typeof scene.source?.replay_episode === 'number' && (
            <div>
              <dt>Dataset episode</dt>
              <dd>{scene.source.replay_episode}</dd>
            </div>
          )}
        </dl>
      </details>
    </div>
  );
}
