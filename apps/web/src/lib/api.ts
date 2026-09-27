import { publicDemo } from "@/lib/public-demo";
import type { components } from './api.generated';
import { basePath, publicPath } from './base-path';
import type { TrainingModel } from './training-models';

export type CloudProvider = 'gcp';
export type CloudConfig = { project_id: string; region: string };
export type CloudConnection = {
  provider: CloudProvider;
  name: string;
  status: 'disconnected' | 'connected' | 'unverified' | 'setup_required' | 'error';
  config: CloudConfig | null;
  identity: { account: string | null } | null;
  checked_at: string | null;
  message: string | null;
  setup_commands: string[];
};
export type CloudConnections = { providers: CloudConnection[] };
export type HuggingFaceConnectionStatus = {
  configured: boolean;
  username: string | null;
  token_hint: string | null;
  checked_at: string | null;
  message: string | null;
};
export type ComputeProvider = CloudProvider | 'local';
export type LocalComputeSettings = { enabled: boolean; label: string };
export type GcpComputeSettings = { enabled: boolean; default_gpu: string; disk_size_gb: number; idle_minutes: number };
export type CloudGpuOption = { id: string; label: string; accelerator: string; gpu_memory_mib: number; gpu_count: number; supported: boolean; available: boolean; unavailable_reason: string | null };
export type ComputeSettings = {
  local: LocalComputeSettings;
  gcp?: GcpComputeSettings;
  runtimes: PolicyOptions['runtimes'];
  gpu_options?: CloudGpuOption[];
  gcp_status?: { status: 'unchecked' | 'ready' | 'setup_required' | 'error'; configured: boolean; project_id: string | null; region: string | null; workspace?: string | null; skypilot_installed: boolean; checked_at: string | null; message: string | null; setup_commands: string[] };
};

export type PolicyRequest = components['schemas']['PolicyRequest'];
export type PolicyArtifact = components['schemas']['PolicyArtifact'];
export type LifecycleResult = components['schemas']['LifecycleResult'];
export type JobEvent = components['schemas']['JobEvent'];
export type TrainingTelemetry = components['schemas']['TrainingTelemetry'];
export type TrainingMetric = components['schemas']['TrainingMetric'];
export type PolicyOptions = {
  runtimes: { id: string; label: string; device: 'cpu' | 'cuda'; training: boolean; act_export?: boolean; export_only?: boolean; simulation: boolean; engine_evaluation?: boolean; run?: boolean; gpu_name?: string | null; gpu_memory_mib?: number | null; training_gpu_count?: number | null; training_model_ids?: string[]; provider?: ComputeProvider; provider_label?: string; region?: string | null; enabled?: boolean; execution?: 'native' | 'skypilot'; accelerator?: string | null; unavailable_reason?: string | null }[];
  compute?: { local: LocalComputeSettings; gcp?: GcpComputeSettings };
  training_models?: TrainingModel[];
  sources: { id: string; label: string; task: string }[];
  training_methods: { id: string; label: string; description: string }[];
  default_training_method: string;
  quantization_defaults: { cuda: { language: 'Q8_0'; vision: null }; cpu: { language: 'Q8_0'; vision: null }; note: string };
};

export type Project = components['schemas']['Project'];
export type Job = components['schemas']['Job'];
export type DatasetProfile = components['schemas']['DatasetProfile'];
export type Capability = components['schemas']['Capability'];
export type IntakeRequest = Omit<components['schemas']['IntakeRequest'], 'snapshot_for_training'> & { snapshot_for_training?: boolean };
export type DatasetJob = Job & { kind: 'dataset.inspect'; request: IntakeRequest; result?: DatasetProfile | null };
export type AugmentationRequest = components['schemas']['AugmentationRequest'];
export type AugmentationResult = components['schemas']['AugmentationResult'];
export type AugmentationJob = Job & { kind: 'dataset.augment'; request: AugmentationRequest; result?: AugmentationResult | null };
export type AugmentationOptions = components['schemas']['AugmentationOptions'];

export function isAugmentationJob(job: Job): job is AugmentationJob {
  return job.kind === 'dataset.augment' && 'operation' in job.request &&
    job.request.operation === 'dataset.augment' && (!job.result || 'clips' in job.result);
}

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
  (process.env.NODE_ENV === 'development' ? 'http://127.0.0.1:8000' : basePath)).replace(/\/$/, '');

export const apiReferenceUrl = publicPath('/docs/');
export const openApiUrl = `${apiOrigin}/openapi.json`;

// Local preview media uses the API host/prefix; Hub and signed external URLs stay intact.
export const apiMediaUrl = (url: string) => url.startsWith('/api/') ? `${apiOrigin}${url}` : url;

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, init?: RequestInit, timeoutMs = 15_000): Promise<T> {
  const publicInspection = init?.method === "POST" && /^\/projects\/[A-Za-z0-9_-]+\/intakes$/.test(path);
  if (publicDemo && init?.method && !["GET", "HEAD"].includes(init.method.toUpperCase()) && !publicInspection) {
    throw new ApiError("Starting cloud jobs and changing settings require the owner workspace.", 403);
  }
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
      throw new ApiError(message, response.status);
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

export const trainingReproducibilityUrl = (jobId: string) =>
  `${apiOrigin}/api/v1/jobs/${encodeURIComponent(jobId)}/training/reproducibility`;

export const augmentationClipUrl = (jobId: string, index: number, original = false) =>
  `${apiOrigin}/api/v1/jobs/${encodeURIComponent(jobId)}/augmentation/clips/${index}?original=${original}`;

export const augmentationDownloadUrl = (jobId: string) =>
  `${apiOrigin}/api/v1/jobs/${encodeURIComponent(jobId)}/augmentation/download`;

export const api = {
  huggingFaceStatus: () => request<HuggingFaceConnectionStatus>('/huggingface-connection'),
  saveHuggingFaceToken: (token: string) => request<HuggingFaceConnectionStatus>('/huggingface-connection', { method: 'PUT', body: JSON.stringify({ token }) }, 20_000),
  removeHuggingFaceToken: () => request<HuggingFaceConnectionStatus>('/huggingface-connection', { method: 'DELETE' }),
  computeSettings: () => request<ComputeSettings>('/compute-settings'),
  saveCloudCompute: (gcp: GcpComputeSettings) => request<ComputeSettings>('/compute-settings', { method: 'PUT', body: JSON.stringify({ gcp }) }),
  checkCloudCompute: () => request<ComputeSettings>('/compute-settings/gcp/check', { method: 'POST' }, 150_000),
  prepareCloudCompute: () => request<ComputeSettings>('/compute-settings/gcp/prepare', { method: 'POST' }, 260_000),
  saveComputeSettings: (local: LocalComputeSettings) => request<ComputeSettings>('/compute-settings', { method: 'PUT', body: JSON.stringify({ local }) }),
  cloudConnections: () => request<CloudConnections>('/cloud-connections'),
  connectCloud: (provider: CloudProvider, config: CloudConfig) => request<CloudConnection>(`/cloud-connections/${provider}/connect`, { method: 'POST', body: JSON.stringify(config) }, 45_000),
  recheckCloud: (provider: CloudProvider) => request<CloudConnection>(`/cloud-connections/${provider}/recheck`, { method: 'POST' }, 45_000),
  disconnectCloud: (provider: CloudProvider) => request<CloudConnection>(`/cloud-connections/${provider}/disconnect`, { method: 'POST' }),
  augmentationOptions: () => request<AugmentationOptions>('/augmentation-options'),
  augment: (projectId: string, body: AugmentationRequest) => request<Job>(`/projects/${encodeURIComponent(projectId)}/augmentations`, {
    method: 'POST', body: JSON.stringify(body),
  }),

  cloudRuns: () => request<components['schemas']['CloudRunsFeed']>('/cloud-runs'),
  policyOptions: () => request<PolicyOptions>('/policy-options'),
  artifacts: (id: string) => request<PolicyArtifact[]>(`/projects/${encodeURIComponent(id)}/artifacts`),
  trainingTelemetry: (id: string) => request<TrainingTelemetry>(`/jobs/${encodeURIComponent(id)}/training`),
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
    30_000,
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
