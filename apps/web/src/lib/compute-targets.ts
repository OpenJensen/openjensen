import { apiOrigin } from './api';

export type TargetResources = {
  cpus: number;
  memory_gib: number;
  accelerator: string | null;
  gpu_count: number;
  disk_gib: number;
  timeout_seconds: number;
  concurrency: number;
};
export type ComputeTarget = {
  id: string;
  label: string;
  adapter: 'local' | 'skypilot';
  infrastructure: 'local' | 'ssh' | 'gcp' | 'runpod' | 'nebius' | 'slurm';
  enabled: boolean;
  runtime_id: string;
  connection_ref: string | null;
  location: string | null;
  partition: string | null;
  account: string | null;
  qos: string | null;
  resources: TargetResources;
  operations: string[];
};
export type PublicComputeTarget = {
  definition: ComputeTarget;
  operations: string[];
  status: 'disabled' | 'setup_required' | 'configured' | 'error';
  message: string;
  launchable: boolean;
  checked_at: string | null;
  sha256: string;
};
export type ComputeTargetList = {
  schema_version: 1;
  targets: PublicComputeTarget[];
  runtime_options: {
    id: string;
    label: string;
    operations: string[];
    device?: 'cpu' | 'cuda';
    accelerator?: string | null;
    remote_only?: boolean;
  }[];
  connection_refs: string[];
};
async function request<T>(path = '', init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiOrigin}/api/v1/compute-targets${path}`, {
    cache: 'no-store',
    ...init,
  });
  const value = await response.json();
  if (!response.ok)
    throw new Error(
      typeof value.detail === 'string'
        ? value.detail
        : 'Compute settings could not be saved. Review the fields and try again.',
    );
  return value as T;
}
export const computeTargets = {
  list: async (signal?: AbortSignal) => {
    const value = await request<ComputeTargetList>('', { signal });
    if (
      value.schema_version !== 1 ||
      !Array.isArray(value.targets) ||
      !Array.isArray(value.runtime_options) ||
      !Array.isArray(value.connection_refs)
    )
      throw new Error('Compute targets could not be loaded. Refresh the connection and try again.');
    return value;
  },
  save: (target: ComputeTarget) =>
    request<PublicComputeTarget>(`/${encodeURIComponent(target.id)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(target),
    }),
  check: (id: string) => request<PublicComputeTarget>(`/${encodeURIComponent(id)}/check`, { method: 'POST' }),
  remove: (id: string) => request<{ removed: true }>(`/${encodeURIComponent(id)}`, { method: 'DELETE' }),
};
