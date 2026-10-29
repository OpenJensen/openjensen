import { expect, test } from '@playwright/test';
import { saveJobCompute, selectedJobCompute, withJobCompute } from '../../apps/web/src/lib/job-compute-selection';

function storage() {
  const values = new Map<string, string>();
  return {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => {
      values.set(key, value);
    },
    removeItem: (key: string) => {
      values.delete(key);
    },
    values,
  };
}
const target = { id: 'gpu-target', label: 'GPU server', runtimeId: 'training-worker', operation: 'policy.finetune' };

test('new recipe captures explicit compute before journaling without mutating its inputs', () => {
  const store = storage();
  const original = { operation: 'policy.finetune', runtime_id: 'old-worker', training: { steps: 100 } };
  saveJobCompute('project', target.operation, target, store);
  const result = withJobCompute('project', original, store);
  expect(result).toEqual({ ...original, execution_target_id: target.id, runtime_id: target.runtimeId });
  expect(original.runtime_id).toBe('old-worker');
  expect(selectedJobCompute('other-project', target.operation, store)).toBeNull();
});

test('recovered bound recipe retains exact target even when the configured selection changes', () => {
  const store = storage();
  saveJobCompute('project', target.operation, target, store);
  const saved = { operation: target.operation, runtime_id: 'accepted-worker', execution_target_id: 'accepted-target' };
  expect(withJobCompute('project', saved, store)).toBe(saved);
});

test('no explicit selection keeps the existing module recipe intact', () => {
  const store = storage();
  const body = { operation: 'policy.quantize', runtime_id: 'conversion-worker' };
  expect(withJobCompute('project', body, store)).toBe(body);
});

test('corrupt compute selection blocks replacement rather than changing execution silently', () => {
  const store = storage();
  saveJobCompute('project', target.operation, target, store);
  for (const key of store.values.keys()) store.values.set(key, '{broken');
  expect(() => withJobCompute('project', { operation: target.operation }, store)).toThrow();
});

test('multi-worker simulation cannot be redirected through a single-runtime override', () => {
  const store = storage();
  saveJobCompute('project', 'policy.run', { ...target, operation: 'policy.run' }, store);
  expect(() =>
    withJobCompute('project', { operation: 'policy.run', simulation: { profile_id: 'scene' } }, store),
  ).toThrow('multi-worker');
});
