import { type DatasetJob, type Job, type PolicyArtifact, type PolicyOptions, type PolicyRequest } from './api';
import { policyJobRequest, record, sameJson, UncertainPolicyJob } from './policy-job-mutation';

export type DistillationRecipe = NonNullable<PolicyRequest['native_distillation']>;
export type DistillationRequest = Pick<PolicyRequest, 'operation' | 'runtime_id' | 'artifact_id' | 'dataset_job_id' | 'native_distillation' | 'timeout_seconds'>;
export type StudentJob = Job & { request: PolicyRequest & { native_distillation: DistillationRecipe } };
export function studentJob(job: Job): job is StudentJob { const value: unknown = job.request; return job.kind === 'policy.distill' && record(value) && value.operation === 'policy.distill' && record(value.native_distillation) && value.native_distillation.adapter === 'act-act-v1'; }
export function studentTeacher(artifact: PolicyArtifact, project: string): boolean {
  return artifact.project_id === project && ['native_checkpoint', 'inference_export'].includes(artifact.format) && artifact.metadata?.architecture === 'act' && artifact.metadata?.storage !== 'gcs' && !artifact.metadata?.remote && !artifact.metadata?.remote_uri;
}
export function studentDataset(job: DatasetJob, project: string): boolean { return job.project_id === project && job.status === 'succeeded' && job.result?.inspection_scope === 'complete_snapshot' && job.result.format === 'lerobot_v3' && job.result.snapshot?.lineage_validated === true && job.result.snapshot.total_episodes >= 3; }
export function studentRuntime(runtime: PolicyOptions['runtimes'][number]): boolean { return runtime.native_distillation === true && runtime.provider === 'local' && runtime.execution === 'native' && runtime.enabled !== false && runtime.launchable !== false && !runtime.unavailable_reason; }
export function episodeSelection(value: string): number[] {
  if (!/^\s*\d+(\s*,\s*\d+)*\s*$/.test(value)) throw new Error('Enter episode numbers separated by commas in each partition.');
  const ids = value.split(',').map(item => Number(item.trim()));
  if (ids.length > 256 || ids.some(id => !Number.isSafeInteger(id) || id > 19999) || new Set(ids).size !== ids.length) throw new Error('Episode numbers must be unique integers between 0 and 19999.');
  return ids;
}
function accepted(value: unknown, project: string, expected: DistillationRequest, id?: string): Job {
  if (!record(value) || typeof value.id !== 'string' || !value.id || (id && value.id !== id) || value.project_id !== project || value.kind !== 'policy.distill' || typeof value.status !== 'string' || !['queued', 'running', 'succeeded', 'failed', 'cancelled', 'interrupted'].includes(value.status) || !record(value.request) ||
      !['operation', 'runtime_id', 'artifact_id', 'dataset_job_id', 'timeout_seconds', 'native_distillation'].every(key => sameJson(value.request && (value.request as Record<string, unknown>)[key], expected[key as keyof DistillationRequest]))) throw new UncertainPolicyJob();
  return value as unknown as Job;
}
export async function startStudent(project: string, expected: DistillationRequest): Promise<Job> { return accepted(await policyJobRequest(`/projects/${encodeURIComponent(project)}/policy-jobs`, expected), project, expected); }
export async function cancelStudent(job: StudentJob, stillSelected: () => boolean): Promise<Job> {
  const current = accepted(await policyJobRequest(`/jobs/${encodeURIComponent(job.id)}`), job.project_id, job.request, job.id);
  if (!stillSelected()) throw new Error('Selection changed; cancellation was not submitted.');
  if (!['queued', 'running'].includes(current.status)) return current;
  return accepted(await policyJobRequest(`/jobs/${encodeURIComponent(job.id)}/cancel`, {}), job.project_id, job.request, job.id);
}
export type StudentReport = { dataset_kind: 'lerobot' | 'generated_fixture'; student_weights_bytes: number; teacher_inference_tensor_bytes: number; steps: number; trained_student: { validation: { teacher_normalized_l1: number }; final: { teacher_normalized_l1: number } } };
export function studentReport(value: unknown, job: StudentJob): StudentReport | null {
  const kind = job.request.native_distillation.coordinate_attestation === 'generated_fixture' ? 'generated_fixture' : 'lerobot';
  if (!record(value) || value.dataset_kind !== kind || job.status !== 'succeeded') return null;
  if (!record(value) || value.operation !== 'policy.distill' || value.adapter !== 'act-act-v1' || value.teacher_artifact_id !== job.request.artifact_id || value.dataset_job_id !== job.request.dataset_job_id || value.steps !== job.request.native_distillation.steps || value.fresh_reload_verified !== true || value.quality_verified !== false || value.calibration_verified !== false || value.speedup_verified !== false || value.task_success !== null || !['lerobot', 'generated_fixture'].includes(String(value.dataset_kind)) ||
    !['student_weights_bytes', 'teacher_inference_tensor_bytes'].every(key => Number.isSafeInteger(value[key]) && Number(value[key]) > 0) || Number(value.student_weights_bytes) >= Number(value.teacher_inference_tensor_bytes) || !record(value.trained_student)) return null;
  for (const split of ['validation', 'final']) { const result = value.trained_student[split]; if (!record(result) || typeof result.teacher_normalized_l1 !== 'number' || !Number.isFinite(result.teacher_normalized_l1) || result.teacher_normalized_l1 < 0) return null; }
  return value as unknown as StudentReport;
}
