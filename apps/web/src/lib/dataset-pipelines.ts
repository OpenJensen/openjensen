import { apiOrigin } from './api';
import { datasetRequest } from './dataset-library';

export type PipelineStepResult = {
  status?: 'running' | 'succeeded' | 'failed';
  entries?: number;
  kept?: number;
  error?: string;
  summary?: Record<string, unknown>;
};
export type DatasetPipeline = {
  id: string;
  project_id: string;
  name: string;
  dataset: { id: string | null; name: string };
  steps: { processor: string; params: Record<string, unknown> }[];
  results: PipelineStepResult[];
  review: Record<string, boolean>;
  status: 'queued' | 'running' | 'succeeded' | 'failed' | 'interrupted';
  message: string;
  saved_datasets: { dataset_id?: string; name: string; episodes: number; augmented: number }[];
  created_at: string;
  updated_at?: string;
};
export type PipelineEpisode = {
  id: string;
  episode_index: number;
  origin: 'recorded' | 'augmented';
  keep: boolean;
  suggested: boolean;
  reviewed: boolean;
  strip: string | null;
  gripper: { verdict: 'clean' | 'regrasp' | 'no_grasp' | 'not_released'; missed_grasps: number } | null;
  vision: { outcome: 'success' | 'success_after_retry' | 'failed'; grasp_attempts: number; notes: string } | null;
  /** Each stage of the task (reach, grasp, lift, carry, place) and whether it happened. */
  waypoints?: Record<string, boolean>;
};

const root = (project: string) => `/projects/${encodeURIComponent(project)}/dataset-pipelines`;

export const pipelineActive = (pipeline?: DatasetPipeline | null) =>
  pipeline?.status === 'queued' || pipeline?.status === 'running';

export const datasetPipelines = {
  list: (project: string, signal?: AbortSignal) =>
    datasetRequest<{ pipelines: DatasetPipeline[] }>(root(project), { signal }),
  check: (project: string, datasetId: string) =>
    datasetRequest<DatasetPipeline>(root(project), {
      method: 'POST',
      body: JSON.stringify({ dataset_id: datasetId, recipe: 'check' }),
    }),
  episodes: (project: string, id: string, signal?: AbortSignal) =>
    datasetRequest<{ pipeline: DatasetPipeline; episodes: PipelineEpisode[] }>(
      `${root(project)}/${encodeURIComponent(id)}/episodes`,
      { signal },
    ),
  review: (project: string, id: string, decisions: Record<string, boolean>) =>
    datasetRequest<DatasetPipeline>(`${root(project)}/${encodeURIComponent(id)}/review`, {
      method: 'PUT',
      body: JSON.stringify({ decisions }),
    }),
  save: (project: string, id: string, name: string, augmentCopies: number) =>
    datasetRequest<DatasetPipeline>(`${root(project)}/${encodeURIComponent(id)}/save`, {
      method: 'POST',
      body: JSON.stringify({ name, augment_copies: augmentCopies }),
    }),
  resume: (project: string, id: string) =>
    datasetRequest<DatasetPipeline>(`${root(project)}/${encodeURIComponent(id)}/continue`, { method: 'POST' }),
  media: (project: string, id: string, path: string) =>
    `${apiOrigin}/api/v1${root(project)}/${encodeURIComponent(id)}/media?path=${encodeURIComponent(path)}`,
};

/** One word for what the checks found, agreeing or not. */
export function episodeVerdict(episode: PipelineEpisode): { label: string; tone: 'good' | 'warn' | 'bad' } {
  const outcome = episode.vision?.outcome;
  const gripper = episode.gripper?.verdict;
  if (outcome === 'failed' || gripper === 'no_grasp' || (gripper === 'not_released' && !outcome))
    return { label: 'Failed', tone: 'bad' };
  if (outcome === 'success_after_retry' || gripper === 'regrasp') return { label: 'Retried grasp', tone: 'warn' };
  return { label: 'Clean', tone: 'good' };
}

/** The two checks disagree, so a person should look. */
export const episodeDisputed = (episode: PipelineEpisode) => {
  if (!episode.vision || !episode.gripper) return false;
  const fromGripper: Record<string, string> = {
    clean: 'success',
    regrasp: 'success_after_retry',
    no_grasp: 'failed',
    not_released: 'failed',
  };
  return fromGripper[episode.gripper.verdict] !== episode.vision.outcome;
};
