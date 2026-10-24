import { expect, test } from '@playwright/test';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { datasetPollingInterval, publishDatasetReceipt } from '../../apps/web/src/lib/dataset-cache';
import type { LibraryDataset } from '../../apps/web/src/lib/dataset-library';

const { QueryClient } = createRequire(resolve('apps/web/package.json'))('@tanstack/react-query');
const dataset = (status: LibraryDataset['status'], id = 'dataset-one'): LibraryDataset => ({
  id,
  project_id: 'project-one',
  name: 'Local data',
  source: 'local',
  status,
  created_at: '2026-10-12T00:00:00Z',
});

test('publishing conversion receipt cancels an older GET without cancelling another dataset', async () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  let releaseOlder!: () => void, releaseOther!: () => void, olderSignal!: AbortSignal, otherSignal!: AbortSignal;
  const olderWaiting = new Promise<void>((resolve) => {
      releaseOlder = resolve;
    }),
    otherWaiting = new Promise<void>((resolve) => {
      releaseOther = resolve;
    });
  client.setQueryData(['dataset-import', 'dataset-one'], dataset('detected'));
  const olderRead = client
    .fetchQuery({
      queryKey: ['dataset-import', 'dataset-one'],
      queryFn: async ({ signal }: { signal: AbortSignal }) => {
        olderSignal = signal;
        await olderWaiting;
        return dataset('detected');
      },
    })
    .catch(() => null);
  const otherRead = client.fetchQuery({
    queryKey: ['dataset-import', 'dataset-two'],
    queryFn: async ({ signal }: { signal: AbortSignal }) => {
      otherSignal = signal;
      await otherWaiting;
      return dataset('ready', 'dataset-two');
    },
  });
  try {
    await publishDatasetReceipt(client, dataset('converting'));
    expect(olderSignal.aborted).toBe(true);
    expect(otherSignal.aborted).toBe(false);
    releaseOlder();
    releaseOther();
    await olderRead;
    await otherRead;
    expect(client.getQueryData(['dataset-import', 'dataset-one']).status).toBe('converting');
    expect(client.getQueryData(['dataset-import', 'dataset-two']).status).toBe('ready');
  } finally {
    releaseOlder();
    releaseOther();
    client.clear();
  }
});

test('accepted conversion keeps polling through earlier detection reads until a terminal state', () => {
  expect(datasetPollingInterval(dataset('detected'))).toBe(false);
  expect(datasetPollingInterval(dataset('uploading'))).toBe(1000);
  expect(datasetPollingInterval(dataset('converting'))).toBe(1000);
  expect(datasetPollingInterval(dataset('detected'), 'dataset-one')).toBe(1000);
  expect(datasetPollingInterval(dataset('detected', 'another'), 'dataset-one')).toBe(false);
  for (const status of ['ready', 'failed', 'unavailable'] as const)
    expect(datasetPollingInterval(dataset(status), 'dataset-one')).toBe(false);
});
