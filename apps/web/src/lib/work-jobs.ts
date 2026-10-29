import type { components } from './api.generated';
import { apiOrigin } from './api';

type WorkJobSchema = components['schemas']['WorkJob'];
export type WorkJob = Omit<WorkJobSchema, 'inputs' | 'outputs' | 'parent_ids' | 'events'> &
  Required<Pick<WorkJobSchema, 'inputs' | 'outputs' | 'parent_ids' | 'events'>>;
export type JobOperation = components['schemas']['JobOperation'];
export type WorkSubmission = Omit<components['schemas']['WorkSubmission'], 'inputs'> & {
  inputs: NonNullable<components['schemas']['WorkSubmission']['inputs']>;
};
export const jobCategoryActions: Record<string, { operation: string; label: string; title: string }> = {
  training: { operation: 'policy.finetune', label: 'Start a new fine-tuning', title: 'Fine-tuning jobs' },
  distillation: { operation: 'policy.distill', label: 'Start a new distillation', title: 'Distillation jobs' },
  quantization: { operation: 'policy.quantize', label: 'Start a new quantization', title: 'Quantization jobs' },
  evaluation: { operation: 'policy.evaluate', label: 'New evaluation', title: 'Evaluation jobs' },
  simulation: { operation: 'policy.run', label: 'New run', title: 'Run jobs' },
  augmentation: { operation: 'dataset.augment', label: 'New augmentation', title: 'Augmentation jobs' },
  'real-to-sim': { operation: 'scene.generate', label: 'New scene', title: 'Real-to-sim jobs' },
  teaching: { operation: 'agent.control', label: 'New teleoperation', title: 'Teleoperation jobs' },
  'decision-lab': { operation: 'decision.score', label: 'New scoring', title: 'Decision scoring jobs' },
};
const jobResultLabels: Record<string, string> = {
  'dataset.inspect': 'View dataset',
  'dataset.import': 'View dataset',
  'dataset.convert': 'View dataset',
  'dataset.augment': 'View augmentation',
  'scene.generate': 'Review scene',
  'policy.finetune': 'View training run',
  'policy.distill': 'View distillation',
  'policy.quantize': 'View quantization',
  'policy.evaluate': 'View evaluation',
  'policy.run': 'View run',
  'teaching.capture': 'Open recording',
  'agent.control': 'Open controls',
  'decision.score': 'View scoring',
};
export const jobResultLabel = (operation: string) => jobResultLabels[operation] ?? 'View result';

export const workStatusLabels: Record<string, string> = {
  queued: 'Queued',
  running: 'Running',
  waiting: 'Needs attention',
  succeeded: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
  interrupted: 'Interrupted',
  cancelling: 'Cancelling',
  preparing: 'Preparing',
};
export const isWorkActive = (job: WorkJob) => ['queued', 'preparing', 'running', 'cancelling'].includes(job.status);

export class WorkJobHttpError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}
export type WorkLogRecord = {
  sequence: number;
  job_id: string;
  attempt_id: string;
  timestamp: string;
  kind: 'event' | 'log';
  component: string;
  stage: string | null;
  severity: 'debug' | 'info' | 'warning' | 'error';
  stream: 'stdout' | 'stderr' | null;
  message: string;
  data: Record<string, unknown>;
};
export type WorkLogPage = {
  records: WorkLogRecord[];
  next_cursor: number;
  has_more: boolean;
  retained_from: number | null;
  truncated: boolean;
  available: boolean;
};
export type WorkFailure = {
  code: string;
  stage: string;
  message: string;
  cause: string | null;
  exit_code: number | null;
  retryable: boolean;
  cleanup: 'unknown' | 'pending' | 'succeeded' | 'failed';
  attempt_id: string;
  timestamp: string;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiOrigin}/api/v1${path}`, { cache: 'no-store', ...init });
  const value = await response.json();
  if (!response.ok) {
    const message =
      response.status >= 500 && init?.method === 'POST'
        ? 'The application could not confirm the job request. Review or recover the saved request before another submission.'
        : typeof value.detail === 'string'
          ? value.detail
          : 'The job request could not be verified.';
    throw new WorkJobHttpError(message, response.status);
  }
  const key = new Headers(init?.headers).get('Idempotency-Key');
  if (
    key &&
    (response.headers.get('Idempotency-Key') !== key ||
      !response.headers
        .get('Cache-Control')
        ?.split(',')
        .some((part) => part.trim().toLowerCase() === 'no-store'))
  )
    throw new Error(
      'Durable submission support could not be verified. Recover the saved request before another submission.',
    );
  return value;
}
const path = (project: string) => `/projects/${encodeURIComponent(project)}/work-jobs`;
export const workJobs = {
  operations: () => request<JobOperation[]>('/job-operations'),
  list: (project: string, signal?: AbortSignal) => request<WorkJob[]>(path(project), { signal }),
  page: async (
    project: string,
    offset: number,
    filters: { module?: string; status?: string; q?: string },
    signal?: AbortSignal,
  ): Promise<{ jobs: WorkJob[]; nextOffset: number | null }> => {
    const params = new URLSearchParams({ offset: String(offset), limit: '50' });
    for (const [key, value] of Object.entries(filters)) if (value && value !== 'all') params.set(key, value);
    const response = await fetch(`${apiOrigin}/api/v1${path(project)}?${params}`, { cache: 'no-store', signal });
    const value: unknown = await response.json();
    if (!response.ok || !Array.isArray(value)) throw new Error('Job history could not be loaded.');
    const next = response.headers.get('X-Next-Offset');
    const nextOffset = next === null ? null : Number(next);
    if (nextOffset !== null && (!Number.isSafeInteger(nextOffset) || nextOffset <= offset))
      throw new Error('Job history pagination could not be verified.');
    return { jobs: value as WorkJob[], nextOffset };
  },
  get: (project: string, id: string, signal?: AbortSignal) =>
    request<WorkJob>(`${path(project)}/${encodeURIComponent(id)}`, { signal }),
  cancel: (project: string, id: string) =>
    request<WorkJob>(`${path(project)}/${encodeURIComponent(id)}/cancel`, { method: 'POST' }),
  logs: (
    project: string,
    id: string,
    after: number,
    filters: { severity?: string; stream?: string; query?: string },
    signal?: AbortSignal,
  ) => {
    const params = new URLSearchParams({ after: String(after), limit: '200' });
    for (const [name, value] of Object.entries(filters)) if (value) params.set(name, value);
    return request<WorkLogPage>(`${path(project)}/${encodeURIComponent(id)}/logs?${params}`, { signal });
  },
  failure: (project: string, id: string, signal?: AbortSignal) =>
    request<WorkFailure | null>(`${path(project)}/${encodeURIComponent(id)}/failure`, { signal }),
  diagnosticsUrl: (project: string, id: string) =>
    `${apiOrigin}/api/v1${path(project)}/${encodeURIComponent(id)}/diagnostics.zip`,
  submit: (project: string, body: WorkSubmission, key: string) =>
    request<WorkJob>(path(project), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Idempotency-Key': key },
      body: JSON.stringify(body),
    }),
};
