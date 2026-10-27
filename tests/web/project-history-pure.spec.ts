import { expect, test } from '@playwright/test';
import type { Job } from '../../apps/web/src/lib/api';
import { projectHistoryView } from '../../apps/web/src/lib/project-history';

const project = 'selected-project';
const job = {
  id: 'saved-job',
  project_id: project,
  kind: 'policy.finetune',
  status: 'succeeded',
  created_at: '2026-10-01T00:00:00Z',
  updated_at: '2026-10-01T00:00:00Z',
  request: {},
} as Job;

function freshCache(data: unknown) {
  return {
    data: data as Job[],
    error: null as Error | null,
    isError: false,
    isSuccess: true,
    status: 'success' as const,
    dataUpdatedAt: Date.now(),
    refetch: async () => [job],
  };
}

test('valid fresh project history remains unchanged', () => {
  const cache = freshCache([job]);
  expect(projectHistoryView(cache, project)).toBe(cache);
  const empty = freshCache([]);
  expect(projectHistoryView(empty, project)).toBe(empty);
});

test('mixed-project fresh cached history is rejected without altering the cache or its refetch', () => {
  const foreign = { ...job, id: 'foreign-job', project_id: 'another-project' };
  const cache = freshCache([job, foreign]);
  const view = projectHistoryView(cache, project);
  expect(view.data).toBeUndefined();
  expect(view.error?.message).toContain('contains another project');
  expect(view.isError).toBe(true);
  expect(view.isSuccess).toBe(false);
  expect(view.status).toBe('error');
  expect(view.refetch).toBe(cache.refetch);
  expect(cache.data).toEqual([job, foreign]);
  expect(cache.isSuccess).toBe(true);
});

test('malformed shared cache stays unavailable until a valid result recovers it', async () => {
  for (const malformed of [
    null,
    { jobs: [] },
    [null],
    [{ ...job, request: [] }],
    [{ ...job, updated_at: 'invalid' }],
  ]) {
    const cache = freshCache(malformed);
    const invalid = projectHistoryView(cache, project);
    expect(invalid.data).toBeUndefined();
    expect(invalid.isError).toBe(true);
    const recovered = { ...cache, data: await invalid.refetch() };
    expect(projectHistoryView(recovered, project)).toBe(recovered);
    expect(projectHistoryView(recovered, project).error).toBeNull();
  }
});
