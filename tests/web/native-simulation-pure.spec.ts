import { test, expect } from '@playwright/test';
import type { Job } from '../../apps/web/src/lib/api';
import {
  nativeInput, simulationTarget, simulationOptions, simulationProfileTarget,
  simulationExecutionTarget, type NativeArtifact, type SimulationProfile,
} from '../../apps/web/src/lib/native-simulation';

const legacy: SimulationProfile = { id: 'cup', label: 'Cup simulator', architectures: ['act', 'smolvla'], experimental: true, task_object: 'cup' };
const cpu: SimulationProfile = { ...legacy, id: 'packed-cup', architectures: ['act'], provider: 'gcp', policy_runtime: 'packed-act-cpu', policy_device: 'cpu', policy_formats: ['firebird_quant'], accelerators: ['L4'] };
const cuda: SimulationProfile = { ...legacy, provider: 'gcp', policy_runtime: 'lerobot-cuda', policy_device: 'cuda', policy_formats: ['safetensors'], accelerators: ['L4', 'H100'] };
function artifact(format = 'inference_export', metadata: Record<string, unknown> = {}): NativeArtifact {
  return { id: 'policy', project_id: 'alpha', format, metadata: { architecture: 'act', ...metadata } } as NativeArtifact;
}
function job(target: Record<string, unknown>): Job {
  return { simulation_target: { provider: 'gcp', profile_id: 'cup', accelerators: ['L4', 'H100'], ...target } } as unknown as Job;
}
const originalFetch = globalThis.fetch;
test.afterEach(() => { globalThis.fetch = originalFetch; });

test('legacy and explicit CUDA profiles retain float ACT and SmolVLA choices', () => {
  for (const profile of [undefined, legacy, cuda]) for (const architecture of ['act', 'smolvla']) {
    expect(nativeInput(artifact('inference_export', { architecture }), 'alpha', profile)).toBe(true);
  }
});
test('packed outputs are offered only for an explicit ACT CPU profile', () => {
  const packed = artifact('native_quantized', { format: 'firebird_quant' });
  expect(nativeInput(packed, 'alpha', cpu)).toBe(true);
  for (const profile of [undefined, legacy, cuda]) expect(nativeInput(packed, 'alpha', profile)).toBe(false);
  expect(nativeInput(artifact(), 'alpha', cpu)).toBe(false);
  expect(nativeInput(artifact('native_quantized', { architecture: 'smolvla' }), 'alpha', cpu)).toBe(false);
});
for (const metadata of [
  { format: 'safetensors' }, { checkpoint: { model_format: 'safetensors' } },
  { checkpoint: { model_format: null } }, { checkpoint: { model_format: [] } },
  { checkpoint: { model_format: 'gguf' } },
]) test(`contradictory or malformed packed encoding is hidden: ${JSON.stringify(metadata)}`, () => {
  expect(nativeInput(artifact('native_quantized', metadata), 'alpha', cpu)).toBe(false);
});
test('packed selection retains project and local-storage boundaries', () => {
  const packed = artifact('native_quantized');
  expect(nativeInput(packed, 'beta', cpu)).toBe(false);
  for (const metadata of [{ storage: 'gcs' }, { remote_uri: 'gs://fixture/policy' }]) {
    expect(nativeInput(artifact('native_quantized', metadata), 'alpha', cpu)).toBe(false);
  }
});
test('accepted target and displayed resource choice agree for both runtimes', () => {
  const oldJob = job({});
  const cpuJob = job({ policy_runtime: 'packed-act-cpu', accelerators: ['L4'] });
  expect(simulationTarget(oldJob)).not.toBeNull();
  expect(simulationTarget(cpuJob)).not.toBeNull();
  expect(simulationProfileTarget(legacy)).toBe('L4 + H100 workers');
  expect(simulationProfileTarget(cpu)).toBe('L4 simulator + CPU policy worker');
  expect(simulationExecutionTarget(cpuJob)).toBe('L4 simulator + CPU policy worker');
});
for (const target of [
  { policy_runtime: null }, { policy_runtime: 'unknown' },
  { policy_runtime: 'packed-act-cpu' }, { policy_runtime: 'packed-act-cpu', accelerators: ['H100'] },
  { policy_runtime: 'lerobot-cuda', accelerators: ['L4'] },
]) test(`wrong accepted runtime/resources are rejected: ${JSON.stringify(target)}`, () => {
  expect(simulationTarget(job(target))).toBeNull();
  expect(simulationExecutionTarget(job(target))).toBeNull();
});
test('options admit legacy profiles and both coherent new runtime declarations', async () => {
  globalThis.fetch = async () => Response.json({ profiles: [legacy, cpu, cuda], max_archive_bytes: 1000, scored_evaluation: false });
  expect((await simulationOptions()).profiles).toEqual([legacy, cpu, cuda]);
});
for (const patch of [
  { policy_device: 'cuda' }, { policy_formats: ['safetensors'] },
  { accelerators: ['L4', 'H100'] }, { architectures: ['act', 'smolvla'] },
  { policy_runtime: undefined }, { provider: 'firebird' },
]) test(`options reject a contradictory CPU profile: ${JSON.stringify(patch)}`, async () => {
  globalThis.fetch = async () => Response.json({ profiles: [{ ...cpu, ...patch }], max_archive_bytes: 1000, scored_evaluation: false });
  await expect(simulationOptions()).rejects.toThrow('unsupported simulation options');
});
