import { type Job, type PolicyArtifact, type PolicyOptions } from './api';
import { policyJobRequest, UncertainPolicyJob } from './policy-job-mutation';

// Additive client boundary while the core-owned combined schema is generated.
// Callers choose registered IDs; no executable, file path or provider is accepted.
export type NativeQuantization = { format: 'firebird_quant'; bits: 4 | 8; group_size: 64 };
export type NativeQuantizationRuntime = PolicyOptions['runtimes'][number] & {
  native_quantization?: boolean; native_quantization_only?: boolean; launchable?: boolean;
};
export type PackedArtifact = Omit<PolicyArtifact, 'format'> & { format: PolicyArtifact['format'] | 'native_quantized' };
export class UncertainQuantization extends Error {}
const encode = encodeURIComponent;
const statuses = ['queued', 'running', 'succeeded', 'failed', 'cancelled', 'interrupted'];
export function object(value: unknown): value is Record<string, unknown> { return value !== null && typeof value === 'object' && !Array.isArray(value); }
export function nativeQuantizationOf(job: Job): NativeQuantization | null {
  const request: unknown = job.request;
  if (job.kind !== 'policy.quantize' || !object(request) || !object(request.native_quantization)) return null;
  const spec = request.native_quantization;
  return spec.format === 'firebird_quant' && (spec.bits === 4 || spec.bits === 8) && spec.group_size === 64 ? spec as NativeQuantization : null;
}
export function nativeQuantizationOnly(runtime: unknown): boolean { return object(runtime) && runtime.native_quantization_only === true; }
export function availableNativeQuantizer(runtime: NativeQuantizationRuntime): boolean {
  return runtime.native_quantization === true && runtime.enabled !== false && runtime.launchable !== false &&
    runtime.execution === 'native' && runtime.provider === 'local' && !runtime.unavailable_reason;
}
export function nativeQuantizationInput(artifact: PackedArtifact, projectId: string): boolean {
  return artifact.project_id === projectId && ['native_checkpoint', 'inference_export'].includes(artifact.format) &&
    artifact.metadata?.architecture === 'act' && artifact.metadata?.storage !== 'gcs' && !artifact.metadata?.remote && !artifact.metadata?.remote_uri &&
    artifact.metadata?.inference_only !== false && artifact.metadata?.method !== 'full' && artifact.metadata?.training_backend !== 'lerobot' && artifact.metadata?.use_vae !== true;
}
function uncertain() { return new UncertainQuantization('The submission outcome is unverified. Check recorded jobs before making another request. This request was not retried.'); }
async function request(path: string, body?: unknown): Promise<unknown> {
  try { return await policyJobRequest(path, body); }
  catch (error) { if (error instanceof UncertainPolicyJob) throw uncertain(); throw error; }
}
type Expected = { project: string; runtime: string; artifact: string; bits: 4 | 8; timeout: number };
function accepted(value: unknown, expected: Expected, id?: string): Job {
  if (!object(value) || typeof value.id !== 'string' || !value.id || (id && value.id !== id) || value.project_id !== expected.project ||
      value.kind !== 'policy.quantize' || !statuses.includes(String(value.status)) || !object(value.request) ||
      value.request.operation !== 'policy.quantize' || value.request.runtime_id !== expected.runtime || value.request.artifact_id !== expected.artifact ||
      value.request.timeout_seconds !== expected.timeout || !object(value.request.native_quantization)) throw uncertain();
  const spec = value.request.native_quantization;
  if (spec.format !== 'firebird_quant' || spec.bits !== expected.bits || spec.group_size !== 64) throw uncertain();
  return value as unknown as Job;
}
export async function startNativeQuantization(expected: Expected): Promise<Job> {
  const value = await request(`/projects/${encode(expected.project)}/policy-jobs`, { operation: 'policy.quantize', runtime_id: expected.runtime, artifact_id: expected.artifact, native_quantization: { format: 'firebird_quant', bits: expected.bits, group_size: 64 }, timeout_seconds: expected.timeout });
  return accepted(value, expected);
}
export async function cancelNativeQuantization(job: Job, stillSelected: () => boolean): Promise<Job> {
  const spec = nativeQuantizationOf(job);
  const original: unknown = job.request;
  if (!spec || !object(original) || typeof original.runtime_id !== 'string' || typeof original.artifact_id !== 'string' || typeof original.timeout_seconds !== 'number') throw new Error('This is not a supported native quantization job.');
  const expected = { project: job.project_id, runtime: original.runtime_id, artifact: original.artifact_id, bits: spec.bits, timeout: original.timeout_seconds };
  const current = accepted(await request(`/jobs/${encode(job.id)}`), expected, job.id);
  if (!stillSelected()) throw new Error('Selection changed; cancellation was not submitted.');
  if (!['queued', 'running'].includes(current.status)) return current;
  return accepted(await request(`/jobs/${encode(job.id)}/cancel`, {}), expected, job.id);
}

export type QuantizationReport = {
  model_id: string; precision: 'int4' | 'int8'; source_artifact_id: string;
  source_artifact_manifest_sha256: string; source_weight_bytes: number; packed_weight_bytes: number; policy_package_bytes: number;
  drift_from_fp32: { seed: number; input_sha256: string; raw: Drift; postprocessed: Drift }[];
};
type Drift = { rmse: number; maximum_absolute_difference: number; coordinates: 600 };
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0;
function drift(value: unknown): boolean { return object(value) && finite(value.rmse) && finite(value.maximum_absolute_difference) && value.coordinates === 600; }
export function measuredNativeReport(value: unknown, job: Job): QuantizationReport | null {
  const spec = nativeQuantizationOf(job);
  const original: unknown = job.request;
  if (!spec || !object(value) || !object(original) || value.stage !== 'operation' || value.operation !== 'policy.quantize' ||
      value.architecture !== 'act' || value.format !== 'firebird_quant' || value.precision !== `int${spec.bits}` ||
      typeof value.model_id !== 'string' || !/^sha256:[a-f0-9]{64}$/.test(value.model_id) || value.source_artifact_id !== original.artifact_id ||
      typeof value.source_artifact_manifest_sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(value.source_artifact_manifest_sha256) ||
      value.fresh_reload_verified !== true || value.cpu_reload_verified !== true || value.quality_verified !== false ||
      value.calibration_verified !== false || value.isaac_runtime_verified !== false || value.runtime_verified !== false || value.speedup_verified !== false ||
      value.task_success !== null || value.gpu_memory_bytes !== null || value.inference_speedup !== null ||
      !['source_weight_bytes', 'packed_weight_bytes', 'policy_package_bytes'].every(key => Number.isSafeInteger(value[key]) && Number(value[key]) > 0) ||
      !Array.isArray(value.drift_from_fp32) || value.drift_from_fp32.length !== 2 || value.drift_from_fp32.some((item, index) =>
        !object(item) || item.seed !== [171, 902][index] || typeof item.input_sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(item.input_sha256) || !drift(item.raw) || !drift(item.postprocessed))) return null;
  return value as unknown as QuantizationReport;
}

/** Navigation only: the destination must still admit its worker, dataset and explicit recipe. */
export function replayableNativeOutput(artifact: PackedArtifact, job: Job, report: QuantizationReport | null): boolean {
  const metadata = artifact.metadata;
  return !!report && job.status === 'succeeded' && typeof artifact.id === 'string' && artifact.id.length > 0 &&
    artifact.project_id === job.project_id && artifact.job_id === job.id && artifact.format === 'native_quantized' &&
    metadata?.architecture === 'act' && metadata.format === 'firebird_quant' && metadata.format_version === 1 &&
    metadata.model_id === report.model_id && metadata.precision === report.precision && metadata.inference_only === true &&
    metadata.fresh_reload_verified === true && metadata.cpu_reload_verified === true &&
    metadata.source_artifact_id === report.source_artifact_id && metadata.source_artifact_manifest_sha256 === report.source_artifact_manifest_sha256 &&
    metadata.storage !== 'gcs' && !metadata.remote && !metadata.remote_uri;
}
