import { apiOrigin } from './api';

export type JobComputeSelection = { id: string; label: string; runtimeId: string; operation: string };
type SelectionStore = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>;
const selectionKey = (project: string, operation: string) => {
  const endpoint = new URL(apiOrigin || '/', 'http://same-origin.invalid');
  return `openjensen:job-compute:v1:${encodeURIComponent(endpoint.origin + endpoint.pathname)}:${encodeURIComponent(project)}:${operation}`;
};
export function selectedJobCompute(
  project: string,
  operation: string,
  storage: SelectionStore = sessionStorage,
): JobComputeSelection | null {
  const raw = storage.getItem(selectionKey(project, operation));
  if (!raw) return null;
  const value: unknown = JSON.parse(raw);
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new Error('Saved compute selection is unreadable. Review compute before submitting.');
  const item = value as Record<string, unknown>;
  if (
    typeof item.id !== 'string' ||
    typeof item.label !== 'string' ||
    typeof item.runtimeId !== 'string' ||
    item.operation !== operation
  )
    throw new Error('Saved compute selection is unreadable. Review compute before submitting.');
  return item as JobComputeSelection;
}
export function saveJobCompute(
  project: string,
  operation: string,
  selection: JobComputeSelection | null,
  storage: SelectionStore = sessionStorage,
): void {
  if (selection) storage.setItem(selectionKey(project, operation), JSON.stringify(selection));
  else storage.removeItem(selectionKey(project, operation));
  if (typeof window !== 'undefined') window.dispatchEvent(new Event('openjensen:job-compute-selection'));
}
/** Call only when constructing a new request, before durable journaling. */
export function withJobCompute<T extends object>(project: string, body: T, storage?: SelectionStore): T {
  if (storage === undefined && typeof window === 'undefined') return body;
  const input = body as Record<string, unknown>;
  if (typeof input.operation !== 'string' || !input.operation.startsWith('policy.')) return body;
  // A recovered or explicitly bound request always owns its original selection.
  if (input.execution_target_id != null) return body;
  const selection = selectedJobCompute(project, input.operation, storage ?? sessionStorage);
  if (!selection) return body;
  if (input.simulation != null)
    throw new Error(
      'This simulation uses a multi-worker profile. Clear the compute override or choose a compatible workflow.',
    );
  return { ...body, execution_target_id: selection.id, runtime_id: selection.runtimeId };
}
