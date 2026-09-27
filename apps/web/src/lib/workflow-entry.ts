import { isActive, type Job, type PolicyOptions } from './api';
import { nativeQuantizationOf, nativeQuantizationOnly } from './native-quantization';
import { replayJob } from './native-replay';
import { isSimulationJob } from './native-simulation';

export type RunMode = 'engine' | 'native' | 'replay';
export type QuantizeMode = 'gguf' | 'native';
export type Entry<M extends string> = { mode: M; jobId?: string; origin: 'automatic' | 'manual' | 'handoff' | 'recovery' };
export type ProjectEntry = { run?: Entry<RunMode>; quantize?: Entry<QuantizeMode> };

/** Eligibility is configuration, never proof of a running worker or accepted policy. */
export function engineRuntime(item: PolicyOptions['runtimes'][number], stage: 'Run' | 'Evaluate' | 'Quantize') {
  if (item.enabled === false || item.launchable === false || item.unavailable_reason || item.training_only || item.export_only || item.native_distillation_only || item.native_replay_only || nativeQuantizationOnly(item)) return false;
  const flag = stage === 'Run' ? item.run : stage === 'Evaluate' ? item.engine_evaluation : undefined;
  return stage === 'Quantize' || (item.execution === 'skypilot' ? flag === true : flag !== false);
}
export function runJobMode(job: Job, projectId: string): RunMode | null {
  if (job.project_id !== projectId) return null;
  if (isSimulationJob(job)) return 'native';
  if (replayJob(job)) return 'replay';
  // Unknown specialized requests must not become engine jobs by exclusion.
  if (job.kind !== 'policy.run' || ('native_replay' in job.request && job.request.native_replay != null)) return null;
  return 'engine';
}
export function quantizeJobMode(job: Job, projectId: string): QuantizeMode | null {
  if (job.project_id !== projectId) return null;
  if (nativeQuantizationOf(job)) return 'native';
  if (!['policy.quantize', 'policy.workflow'].includes(job.kind) || ('native_quantization' in job.request && job.request.native_quantization != null)) return null;
  return 'gguf';
}
function ordered(jobs: Job[]) {
  return [...jobs].sort((a, b) => Number(isActive(b)) - Number(isActive(a)) || b.created_at.localeCompare(a.created_at) || a.id.localeCompare(b.id));
}
/** An existing job may restore its workflow; available workers never choose one. */
export function initialRunEntry(projectId: string, jobs: Job[]): Entry<RunMode> | undefined {
  const saved = ordered(jobs).find(job => runJobMode(job, projectId));
  return saved ? { mode: runJobMode(saved, projectId)!, jobId: saved.id, origin: 'automatic' } : undefined;
}
export function initialQuantizeEntry(projectId: string, jobs: Job[]): Entry<QuantizeMode> | undefined {
  const saved = ordered(jobs).find(job => quantizeJobMode(job, projectId));
  return saved ? { mode: quantizeJobMode(saved, projectId)!, jobId: saved.id, origin: 'automatic' } : undefined;
}
