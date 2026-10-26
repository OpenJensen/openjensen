import { expect, test } from '@playwright/test';
import { augmentationGallery } from '../../apps/web/src/lib/augmentation-gallery';
import { augmentationAcknowledgement } from '../../apps/web/src/lib/dataset-submission';
import type { AugmentationJob } from '../../apps/web/src/lib/api';

function run(id: string, overrides: Partial<AugmentationJob> = {}): AugmentationJob {
  return {
    id,
    project_id: 'p',
    kind: 'dataset.augment',
    status: 'succeeded',
    created_at: id,
    updated_at: id,
    request: { source_job_id: 'source', episode_indices: [3], camera_key: 'front', preset: 'texture' },
    result: {
      operation: 'dataset.augment',
      source_job_id: 'source',
      repo_id: 'repo',
      revision: 'a'.repeat(40),
      metadata_sha256: 'b'.repeat(64),
      clips: [
        {
          index: 0,
          episode_index: 3,
          camera_key: 'front',
          source_start_seconds: 15,
          source_end_seconds: 20,
          input_sha256: 'c'.repeat(64),
          output_sha256: 'd'.repeat(64),
        },
      ],
    },
    ...overrides,
  } as AugmentationJob;
}
test('groups variations and refinements with the unedited original while excluding failed and other project jobs', () => {
  const original = run('1');
  const second = run('2', {
    request: { ...original.request, source_clip: { job_id: '1', clip_index: 0, variant: 'original' } },
  });
  const third = run('3', {
    request: { ...original.request, source_clip: { job_id: '2', clip_index: 0, variant: 'augmented' } },
  });
  const groups = augmentationGallery(
    [third, run('4', { status: 'failed' }), run('5', { project_id: 'other' }), second, original],
    'p',
  );
  expect(groups).toHaveLength(1);
  expect(groups[0].original?.job.id).toBe('1');
  expect(groups[0].entries.map((item) => item.job.id)).toEqual(['1', '2', '3']);
});
test('separates cameras, episode intervals and pinned revisions and never labels a refinement input original', () => {
  const root = run('1');
  const other = (id: string, clip: object, result: object = {}) =>
    run(id, { result: { ...root.result!, ...result, clips: [{ ...root.result!.clips[0], ...clip }] } });
  expect(
    augmentationGallery(
      [
        root,
        other('2', { camera_key: 'side' }),
        other('3', { source_start_seconds: 20, source_end_seconds: 25 }),
        other('4', {}, { revision: 'f'.repeat(40) }),
      ],
      'p',
    ),
  ).toHaveLength(4);
  expect(
    augmentationGallery(
      [
        run('2', {
          request: { ...root.request, source_clip: { job_id: 'missing', clip_index: 0, variant: 'augmented' } },
        }),
      ],
      'p',
    )[0].original,
  ).toBeUndefined();
});
test('durable receipts reject an altered dataset camera scope or reviewed plan', () => {
  const body = {
    operation: 'dataset.augment',
    source_job_id: 'source',
    episode_indices: [],
    camera_key: 'front',
    preset: 'texture',
    prompt: '',
    start_seconds: 0,
    duration_seconds: 10,
    dataset_scope: { recipe_job_id: 'recipe', camera_keys: ['front'], plan_sha256: 'f'.repeat(64) },
  };
  const job = run('1', { request: body as AugmentationJob['request'] });
  expect(() => augmentationAcknowledgement(job, 'p', body)).not.toThrow();
  expect(() =>
    augmentationAcknowledgement(
      { ...job, request: { ...body, dataset_scope: { ...body.dataset_scope, camera_keys: ['side'] } } },
      'p',
      body,
    ),
  ).toThrow();
  expect(() => augmentationAcknowledgement({ ...job, request: { ...body, dataset_scope: null } }, 'p', body)).toThrow();
});
