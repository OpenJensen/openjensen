import type { components } from './api.generated';

export type PolicyRequest = components['schemas']['PolicyRequest'];
export type PolicyArtifact = components['schemas']['PolicyArtifact'];
export type LifecycleResult = components['schemas']['LifecycleResult'];
export type JobEvent = components['schemas']['JobEvent'];
export type PolicyOptions = {
  runtimes: { id: string; label: string; device: 'cpu' | 'cuda'; training: boolean; simulation: boolean }[];
  sources: { id: string; label: string; task: string }[];
  training_methods: { id: string; label: string; description: string }[];
  default_training_method: string;
  quantization_defaults: { cuda: { language: 'Q8_0'; vision: null }; cpu: { language: 'Q8_0'; vision: null }; note: string };
};

export type Project = components['schemas']['Project'];
export type Job = components['schemas']['Job'];
export type DatasetProfile = components['schemas']['DatasetProfile'];
export type Capability = components['schemas']['Capability'];
export type IntakeRequest = components['schemas']['IntakeRequest'];
export type DatasetJob = Job & { kind: 'dataset.inspect'; request: IntakeRequest; result?: DatasetProfile | null };

export function isDatasetJob(job: Job): job is DatasetJob {
  return job.kind === 'dataset.inspect' && 'source' in job.request &&
    (!job.result || 'inspection_scope' in job.result);
}

export type EpisodePage = components['schemas']['EpisodePage'];
export type EpisodePreview = components['schemas']['EpisodePreview'];
export type CameraPreview = components['schemas']['CameraPreview'];
export type FrameSample = components['schemas']['FrameSample'];

// Static production builds use the Python host's origin. Development uses its
// loopback API unless the developer explicitly provides an alternative origin.
export const apiOrigin = (process.env.NEXT_PUBLIC_API_URL ??
  (process.env.NODE_ENV === 'development' ? 'http://127.0.0.1:8000' : '')).replace(/\/$/, '');

export const apiReferenceUrl = '/docs/';
export const openApiUrl = `${apiOrigin}/openapi.json`;

async function request<T>(path: string, init?: RequestInit, timeoutMs = 15_000): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${apiOrigin}/api/v1${path}`, {
      ...init,
      cache: 'no-store',
      headers: { 'Content-Type': 'application/json', ...init?.headers },
      signal: controller.signal,
    });
    if (!response.ok) {
      let message = `The request failed (${response.status}).`;
      try {
        const body: { detail?: unknown } = await response.json();
        if (typeof body.detail === 'string') message = body.detail;
        else if (Array.isArray(body.detail)) {
          message = body.detail.map((issue: { msg?: string }) => issue.msg ?? 'Invalid input').join('; ');
        }
      } catch { /* Preserve the HTTP error when no JSON detail exists. */ }
      throw new Error(message);
    }
    return await response.json() as T;
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error(`The API did not respond within ${timeoutMs / 1000} seconds. Check the connection and try again.`);
    }
    if (error instanceof TypeError) {
      throw new Error('Cannot reach the application API. Start the Python application and check its address.');
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

export const artifactDownloadUrl = (projectId: string, artifactId: string) =>
  `${apiOrigin}/api/v1/projects/${encodeURIComponent(projectId)}/artifacts/${encodeURIComponent(artifactId)}/download`;

export const api = {
  cloudRuns: () => request<components['schemas']['CloudRunsFeed']>('/cloud-runs'),
  policyOptions: () => request<PolicyOptions>('/policy-options'),
  artifacts: (id: string) => request<PolicyArtifact[]>(`/projects/${encodeURIComponent(id)}/artifacts`),
  events: (id: string) => request<JobEvent[]>(`/jobs/${encodeURIComponent(id)}/events`),
  policyJob: (id: string, body: PolicyRequest) => request<Job>(`/projects/${encodeURIComponent(id)}/policy-jobs`, {
    method: 'POST', body: JSON.stringify(body),
  }),
  health: () => request<{ status: string; version: string }>('/health'),
  capabilities: () => request<Capability[]>('/capabilities'),
  projects: () => request<Project[]>('/projects'),
  createProject: (name: string) => request<Project>('/projects', {
    method: 'POST', body: JSON.stringify({ name }),
  }),
  jobs: (projectId: string) => request<Job[]>(`/projects/${encodeURIComponent(projectId)}/jobs`),
  inspect: (projectId: string, body: IntakeRequest) => request<Job>(
    `/projects/${encodeURIComponent(projectId)}/intakes`, { method: 'POST', body: JSON.stringify(body) },
  ),
  cancel: (jobId: string) => request<Job>(`/jobs/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' }),
  episodes: (jobId: string, offset = 0, limit = 6) => request<EpisodePage>(
    `/jobs/${encodeURIComponent(jobId)}/episodes?offset=${offset}&limit=${limit}`, undefined, 60_000,
  ),
  episode: (jobId: string, episodeIndex: number) => request<EpisodePreview>(
    `/jobs/${encodeURIComponent(jobId)}/episodes/${episodeIndex}`, undefined, 60_000,
  ),
};

export function isActive(job: Job): boolean {
  return job.status === 'queued' || job.status === 'running';
}
