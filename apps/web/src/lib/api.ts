import type { components } from './api.generated';

export type Project = components['schemas']['Project'];
export type Job = components['schemas']['Job'];
export type DatasetProfile = components['schemas']['DatasetProfile'];
export type Capability = components['schemas']['Capability'];
export type IntakeRequest = components['schemas']['IntakeRequest'];

// Static production builds use the Python host's origin. Development uses its
// loopback API unless the developer explicitly provides an alternative origin.
export const apiOrigin = (process.env.NEXT_PUBLIC_API_URL ??
  (process.env.NODE_ENV === 'development' ? 'http://127.0.0.1:8000' : '')).replace(/\/$/, '');

export const apiReferenceUrl = '/docs/';
export const openApiUrl = `${apiOrigin}/openapi.json`;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15_000);
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
      throw new Error('The API did not respond within 15 seconds. Check that the local application is running.');
    }
    if (error instanceof TypeError) {
      throw new Error('Cannot reach the application API. Start the Python application and check its address.');
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

export const api = {
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
};

export function isActive(job: Job): boolean {
  return job.status === 'queued' || job.status === 'running';
}
