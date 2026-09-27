import { expect, test } from '@playwright/test';
import type { Job } from '../../apps/web/src/lib/api';
import type { PackedArtifact, QuantizationReport } from '../../apps/web/src/lib/native-quantization';
import type { SimulationProfile } from '../../apps/web/src/lib/native-simulation';
import { quantizedSimulationHandoff, simulationHandoffArtifact, simulationHandoffKey, simulationHandoffProfiles, type SimulationHandoff } from '../../apps/web/src/lib/native-simulation-handoff';

const modelId = `sha256:${'b'.repeat(64)}`;
const output = (): PackedArtifact => ({ id: 'packed:second', project_id: 'alpha', job_id: 'quant-job', label: 'Same package label', format: 'native_quantized', manifest_sha256: 'a'.repeat(64), metadata: { architecture: 'act', format: 'firebird_quant', format_version: 1, inference_only: true, model_id: modelId, precision: 'int8', fresh_reload_verified: true, cpu_reload_verified: true, source_artifact_id: 'teacher', source_artifact_manifest_sha256: 'c'.repeat(64) } } as PackedArtifact);
const job = { id: 'quant-job', project_id: 'alpha', status: 'succeeded' } as Job;
const report = { model_id: modelId, precision: 'int8', source_artifact_id: 'teacher', source_artifact_manifest_sha256: 'c'.repeat(64), prediction_horizon: 100, execution_horizon: 100, temporal_contract_sha256: null } as QuantizationReport;
const profile: SimulationProfile = { id: 'packed-cpu', label: 'Generated CPU policy profile', architectures: ['act'], experimental: true, task_object: 'cup', provider: 'gcp', accelerators: ['L4'], policy_runtime: 'packed-act-cpu', policy_device: 'cpu', policy_formats: ['firebird_quant'] };

test('simulation continuation binds completed quantization output identity, not its label', () => {
  const source = output(), handoff = quantizedSimulationHandoff(source, job, report)!;
  expect(handoff).toEqual({ projectId: 'alpha', artifactId: source.id, jobId: job.id, manifestSha256: source.manifest_sha256, modelId });
  const sameLabel = { ...source, id: 'packed:first' };
  expect(simulationHandoffArtifact(handoff, 'alpha', [sameLabel, source])).toBe(source);
  expect(simulationHandoffArtifact(handoff, 'alpha', [sameLabel])).toBeNull();
  expect(simulationHandoffArtifact(handoff, 'alpha', [source, source])).toBeNull();
});
test('incomplete or mismatched quantization evidence cannot produce a simulation handoff', () => {
  expect(quantizedSimulationHandoff(output(), job, null)).toBeNull();
  for (const patch of [{ status: 'running' }, { project_id: 'beta' }, { id: 'another-job' }]) expect(quantizedSimulationHandoff(output(), { ...job, ...patch } as Job, report)).toBeNull();
  for (const patch of [{ model_id: `sha256:${'d'.repeat(64)}` }, { precision: 'int4' }, { source_artifact_id: 'another-source' }, { prediction_horizon: 8 }]) expect(quantizedSimulationHandoff(output(), job, { ...report, ...patch } as QuantizationReport)).toBeNull();
  for (const manifest_sha256 of ['', 'not-a-hash']) expect(quantizedSimulationHandoff({ ...output(), manifest_sha256 }, job, report)).toBeNull();
});
test('a current artifact cannot replace the original project, producer, manifest or model', () => {
  const source = output(), handoff = quantizedSimulationHandoff(source, job, report)!;
  expect(simulationHandoffArtifact(handoff, 'beta', [source])).toBeNull();
  for (const patch of [{ project_id: 'beta' }, { job_id: 'another-job' }, { manifest_sha256: 'd'.repeat(64) }, { format: 'native_checkpoint' }]) expect(simulationHandoffArtifact(handoff, 'alpha', [{ ...source, ...patch } as PackedArtifact])).toBeNull();
  for (const patch of [{ model_id: `sha256:${'d'.repeat(64)}` }, { architecture: 'smolvla' }, { format: 'safetensors' }, { remote: true }, { storage: 'gcs' }, { remote_uri: 'gs://generated/policy' }]) expect(simulationHandoffArtifact(handoff, 'alpha', [{ ...source, metadata: { ...source.metadata, ...patch } }])).toBeNull();
});
test('handoff keys preserve strict types and distinguish successor package identities', () => {
  const handoff = quantizedSimulationHandoff(output(), job, report)!;
  expect(simulationHandoffKey(handoff, 'alpha')).not.toBe(simulationHandoffKey({ ...handoff, manifestSha256: 'd'.repeat(64) }, 'alpha'));
  for (const patch of [{ modelId: [modelId] }, { artifactId: '' }, { jobId: '' }, { projectId: 'beta' }, { manifestSha256: ['a'.repeat(64)] }]) expect(simulationHandoffKey({ ...handoff, ...patch } as SimulationHandoff, 'alpha')).toBeNull();
});
test('only explicit packed ACT CPU profiles are offered, with the L4 simulator retained', () => {
  const cuda: SimulationProfile = { id: 'cuda', label: 'Legacy CUDA', architectures: ['act', 'smolvla'], experimental: true, task_object: 'cup' };
  expect(simulationHandoffProfiles(output(), [cuda, profile])).toEqual([profile]);
  for (const patch of [{ policy_device: 'cuda' }, { policy_runtime: 'lerobot-cuda' }, { policy_formats: ['safetensors'] }, { architectures: ['smolvla'] }, { accelerators: [] }, { accelerators: ['L4', 'H100'] }, { experimental: false }]) expect(simulationHandoffProfiles(output(), [{ ...profile, ...patch } as SimulationProfile])).toEqual([]);
  expect(simulationHandoffProfiles(output(), [])).toEqual([]);
});
