import type { Job } from '@/lib/api';

/** Every observer must verify shared cached history for its selected project. */
export function validProjectHistory(value: unknown, project: string): value is Job[] {
  return (
    Boolean(project) &&
    Array.isArray(value) &&
    value.every(
      (item) =>
        item &&
        typeof item === 'object' &&
        !Array.isArray(item) &&
        typeof item.id === 'string' &&
        item.id.length > 0 &&
        item.project_id === project &&
        typeof item.kind === 'string' &&
        item.kind.length > 0 &&
        typeof item.status === 'string' &&
        ['queued', 'running', 'succeeded', 'failed', 'cancelled', 'interrupted'].includes(item.status) &&
        typeof item.created_at === 'string' &&
        Number.isFinite(Date.parse(item.created_at)) &&
        typeof item.updated_at === 'string' &&
        Number.isFinite(Date.parse(item.updated_at)) &&
        item.request &&
        typeof item.request === 'object' &&
        !Array.isArray(item.request),
    )
  );
}

type HistoryQuery = {
  data: Job[] | undefined;
  error: Error | null;
  isError: boolean;
  isSuccess: boolean;
  status: 'pending' | 'success' | 'error';
};

/** A different observer can populate this key without running our query function. */
export function projectHistoryView<T extends HistoryQuery>(query: T, project: string) {
  if (query.data === undefined || validProjectHistory(query.data, project)) return query;
  return {
    ...query,
    data: undefined,
    error: new Error('Project history is malformed or contains another project. Recovery remains paused.'),
    isError: true,
    isSuccess: false,
    status: 'error' as const,
  };
}
