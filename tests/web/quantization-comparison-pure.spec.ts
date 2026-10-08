import { test, expect } from '@playwright/test';
import type { Job } from '../../apps/web/src/lib/api';
import { pooledActionRmse, quantizationComparison } from '../../apps/web/src/lib/quantization-comparison';

const job = { kind: 'policy.quantize', status: 'succeeded', request: { artifact_id: 'selected-checkpoint' } } as Job;
const comparison = () => ({ schema_version: 1, scope: 'paired_synthetic_native_actions', reference: 'floating_gguf_before_quantization', backend: 'cpu', samples: 1, coordinates: 300, action_chunk_size: 50, real_action_dim: 6, input_sha256: 'a'.repeat(64), executable_sha256: 'b'.repeat(64), source_model_sha256: 'c'.repeat(64), quantized_model_sha256: 'd'.repeat(64), action_rmse: .01, action_mse: .0001, action_mae: .008, action_max_abs_difference: .03, source_file_bytes: 2000000, quantized_file_bytes: 1200000, validation_loss: null, task_success: null });
const report = () => ({ source_artifact_id: 'selected-checkpoint', source_manifest_sha256: 'e'.repeat(64), comparison: comparison() });

test('paired quantization metrics belong to the selected checkpoint and retain comparable file sizes', () => {
  const value = quantizationComparison(job, [report()])!;
  expect(value.reference).toBe('Original floating GGUF');
  expect(value.sourceBytes).toBe(2000000);
  expect(value.packedBytes).toBe(1200000);
  expect(pooledActionRmse(value)).toBe(.01);
});
test('missing, unpaired, non-finite or inconsistent measurements never produce a chart', () => {
  expect(quantizationComparison(job, [])).toBeNull();
  expect(quantizationComparison({ ...job, status: 'running' }, [report()])).toBeNull();
  expect(quantizationComparison(job, [{ ...report(), source_artifact_id: 'another-checkpoint' }])).toBeNull();
  for (const patch of [{ action_rmse: NaN }, { source_model_sha256: 'missing' }, { coordinates: 1600 }, { source_file_bytes: 0 }, { action_max_abs_difference: -.1 }, { task_success: 1 }]) {
    expect(quantizationComparison(job, [{ ...report(), comparison: { ...comparison(), ...patch } }])).toBeNull();
  }
});
test('identical predictions keep the measured zero rather than treating it as absent', () => {
  const value = quantizationComparison(job, [{ ...report(), comparison: { ...comparison(), action_rmse: 0, action_mse: 0, action_mae: 0, action_max_abs_difference: 0 } }])!;
  expect(pooledActionRmse(value)).toBe(0);
});
test('multiple observations pool squared differences weighted by coordinate count', () => {
  expect(pooledActionRmse({ reference: 'fixture', description: '', sourceBytes: 1, packedBytes: 1, sizeLabel: '', samples: [{ label: 'a', rmse: 3, maximum: 3, coordinates: 1 }, { label: 'b', rmse: 4, maximum: 4, coordinates: 3 }] })).toBeCloseTo(Math.sqrt(57 / 4));
});
