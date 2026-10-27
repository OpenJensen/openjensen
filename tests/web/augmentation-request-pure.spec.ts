import { expect, test } from '@playwright/test';
import { augmentationAcknowledgement } from '../../apps/web/src/lib/dataset-submission';

const request = {
  operation: 'dataset.augment',
  source_job_id: 'dataset',
  episode_indices: [3],
  camera_key: 'front',
  preset: 'texture',
  prompt: 'Make the table metal.',
  start_seconds: 0,
  duration_seconds: 5,
};
function receipt(body: object) {
  return {
    id: 'job',
    project_id: 'project',
    kind: 'dataset.augment',
    status: 'queued',
    request: body,
    created_at: '2026-10-26T12:00:00Z',
    updated_at: '2026-10-26T12:00:00Z',
  };
}

test('legacy requests accept omitted or null clip sources but reject an unexpected saved clip', () => {
  expect(augmentationAcknowledgement(receipt(request), 'project', request).id).toBe('job');
  expect(augmentationAcknowledgement(receipt({ ...request, source_clip: null }), 'project', request).id).toBe('job');
  expect(() =>
    augmentationAcknowledgement(
      receipt({ ...request, source_clip: { job_id: 'parent', clip_index: 0, variant: 'augmented' } }),
      'project',
      request,
    ),
  ).toThrow();
});

test('saved-clip acceptance must match the exact parent, clip and variant', () => {
  const body = { ...request, source_clip: { job_id: 'parent', clip_index: 1, variant: 'augmented' } };
  expect(augmentationAcknowledgement(receipt(body), 'project', body).id).toBe('job');
  for (const change of [{ job_id: 'different' }, { clip_index: 0 }, { variant: 'original' }]) {
    expect(() =>
      augmentationAcknowledgement(
        receipt({ ...body, source_clip: { ...body.source_clip, ...change } }),
        'project',
        body,
      ),
    ).toThrow();
  }
});
