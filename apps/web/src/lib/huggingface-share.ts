import type { components } from './api.generated';
import { apiOrigin } from './api';
import type { WorkJob } from './work-jobs';

export type ShareKind = 'dataset' | 'model';
export type ShareDestination = components['schemas']['Destination'];
export type ShareAccount = components['schemas']['SharingAccount'];
export type Published = components['schemas']['Published'];

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${apiOrigin}/api/v1${path}`, {
      cache: 'no-store',
      ...init,
      headers: { 'Content-Type': 'application/json', ...init?.headers },
    });
  } catch (error) {
    throw new Error('Cannot reach the application. Check that it is running, then try again.', { cause: error });
  }
  const value = await response.json().catch(() => ({}));
  if (!response.ok)
    throw new Error(
      typeof value?.detail === 'string' ? value.detail : `Hugging Face sharing failed (${response.status}).`,
    );
  return value;
}

const base = (project: string) => `/projects/${encodeURIComponent(project)}/huggingface-sharing`;

export const huggingFaceShare = {
  /** Where this dataset or model would go. Sends nothing. */
  destination: (project: string, kind: ShareKind, id: string, signal?: AbortSignal) =>
    request<ShareDestination>(`${base(project)}/destination?kind=${kind}&id=${encodeURIComponent(id)}`, { signal }),
  publish: (project: string, body: { kind: ShareKind; id: string; repo_id: string; private: boolean }) =>
    request<WorkJob>(`${base(project)}/publish`, { method: 'POST', body: JSON.stringify(body) }),
  fetch: (project: string, kind: ShareKind, link: string) =>
    request<WorkJob>(`${base(project)}/fetch`, { method: 'POST', body: JSON.stringify({ kind, link }) }),
};

/** `owner/name`, as Hugging Face accepts it. The server checks again. */
export const repositoryName = (value: string) =>
  /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/.test(value.trim());

export const repositoryUrl = (kind: ShareKind, repository: string) =>
  `https://huggingface.co/${kind === 'dataset' ? 'datasets/' : ''}${repository}`;

/** Text read from a finished job's result, which the API types as free-form JSON. */
export function resultText(job: WorkJob | undefined, key: string): string | undefined {
  const result = job?.result;
  if (!result || typeof result !== 'object' || Array.isArray(result)) return undefined;
  const value = (result as Record<string, unknown>)[key];
  return typeof value === 'string' ? value : undefined;
}

export function resultFlag(job: WorkJob | undefined, key: string): boolean {
  const result = job?.result;
  if (!result || typeof result !== 'object' || Array.isArray(result)) return false;
  return (result as Record<string, unknown>)[key] === true;
}
