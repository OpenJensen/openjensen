import { type DatasetJob, type Job, type PolicyArtifact, type PolicyOptions, type PolicyRequest } from './api';
import { policyJobRequest, record, sameJson, UncertainPolicyJob } from './policy-job-mutation';

export type ReplayRecipe = NonNullable<PolicyRequest['native_replay']>;
export type ReplayRequest = Pick<PolicyRequest, 'operation' | 'runtime_id' | 'artifact_id' | 'dataset_job_id' | 'native_replay' | 'timeout_seconds'>;
export type ReplayJob = Job & { request: PolicyRequest & { native_replay: ReplayRecipe } };
export function replayJob(job: Job): job is ReplayJob { const value: unknown = job.request; return job.kind === 'policy.run' && record(value) && value.operation === 'policy.run' && record(value.native_replay) && value.native_replay.adapter === 'act-packed-observation-v1'; }
export function replayPolicy(item: PolicyArtifact, project: string): boolean { return item.project_id === project && item.format === 'native_quantized' && item.metadata?.architecture === 'act' && !item.metadata?.remote && !item.metadata?.remote_uri && item.metadata?.storage !== 'gcs'; }
export function replayDataset(job: DatasetJob, project: string): boolean { return job.project_id === project && job.status === 'succeeded' && job.result?.inspection_scope === 'complete_snapshot' && job.result.format === 'lerobot_v3' && !!job.result.snapshot; }
export function replayRuntime(item: PolicyOptions['runtimes'][number]): boolean { return item.native_replay === true && item.provider === 'local' && item.execution === 'native' && item.enabled !== false && item.launchable !== false && !item.unavailable_reason; }
export function replaySelection(value: string): ReplayRecipe['selection'] {
  if (!/^\s*\d+\s*:\s*\d+(\s*,\s*\d+\s*:\s*\d+)*\s*$/.test(value)) throw new Error('Select observations as episode:frame pairs, separated by commas.');
  const pairs = value.split(',').map(item => item.split(':').map(Number));
  if (pairs.length > 32 || pairs.some(([episode, frame]) => !Number.isSafeInteger(episode) || episode > 19999 || !Number.isSafeInteger(frame) || frame > 10000000) || new Set(pairs.map(pair => pair.join(':'))).size !== pairs.length) throw new Error('Choose 1–32 distinct observations with valid episode and frame numbers.');
  return pairs.map(([episode_index, frame_index]) => ({ episode_index, frame_index }));
}
function accepted(value: unknown, project: string, expected: ReplayRequest, id?: string): ReplayJob {
  if (!record(value) || typeof value.id !== 'string' || !value.id || (id && value.id !== id) || value.project_id !== project || value.kind !== 'policy.run' || !['queued', 'running', 'succeeded', 'failed', 'cancelled', 'interrupted'].includes(String(value.status)) || !record(value.request) ||
      !['operation', 'runtime_id', 'artifact_id', 'dataset_job_id', 'timeout_seconds', 'native_replay'].every(key => sameJson((value.request as Record<string, unknown>)[key], expected[key as keyof ReplayRequest]))) throw new UncertainPolicyJob();
  return value as unknown as ReplayJob;
}
export async function startReplay(project: string, request: ReplayRequest): Promise<ReplayJob> { return accepted(await policyJobRequest(`/projects/${encodeURIComponent(project)}/policy-jobs`, request), project, request); }
export async function cancelReplay(job: ReplayJob, stillSelected: () => boolean): Promise<ReplayJob> {
  const current = accepted(await policyJobRequest(`/jobs/${encodeURIComponent(job.id)}`), job.project_id, job.request, job.id);
  if (!stillSelected()) throw new Error('Selection changed; cancellation was not submitted.');
  if (!['queued', 'running'].includes(current.status)) return current;
  return accepted(await policyJobRequest(`/jobs/${encodeURIComponent(job.id)}/cancel`, {}), job.project_id, job.request, job.id);
}
export type ReplayReport = { model_id: string; observations: number; elapsed_seconds: number; observation_source: { kind: 'generated_fixture' | 'lerobot_snapshot' } };
export function replayReport(value: unknown, job: ReplayJob): ReplayReport | null {
  const kind = job.request.native_replay.coordinate_attestation === 'generated_fixture' ? 'generated_fixture' : 'lerobot_snapshot';
  if (job.status !== 'succeeded' || !record(value) || value.operation !== 'policy.run' || value.stage !== 'native_replay' || value.mode !== 'independent_observation_replay' || value.device !== 'cpu' || value.source_artifact_id !== job.request.artifact_id || value.dataset_job_id !== job.request.dataset_job_id || !record(value.observation_source) || value.observation_source.kind !== kind || typeof value.model_id !== 'string' || !/^sha256:[a-f0-9]{64}$/.test(value.model_id) || value.observations !== job.request.native_replay.selection.length || !sameJson(value.action_shape, [100, 6]) || value.reset_repeat_exact !== true || value.server_closed !== true || value.task_success !== null || ['quality_verified', 'calibration_verified', 'speedup_verified', 'isaac_runtime_verified'].some(key => value[key] !== false) || typeof value.elapsed_seconds !== 'number' || !Number.isFinite(value.elapsed_seconds) || value.elapsed_seconds < 0) return null;
  return value as unknown as ReplayReport;
}
export type ReplayRecord = { artifact_id: string; job_id: string; model_id: string; source_kind: string; coordinate_names: string[]; units: string[]; records: { episode_index: number; frame_index: number; actions: number[][]; reset_repeat_exact: true }[] };
export async function readReplay(project: string, artifact: string, job: ReplayJob, report: ReplayReport): Promise<ReplayRecord> {
  const value = await policyJobRequest(`/projects/${encodeURIComponent(project)}/artifacts/${encodeURIComponent(artifact)}/replay`);
  if (!record(value) || value.artifact_id !== artifact || value.job_id !== job.id || value.model_id !== report.model_id || value.source_kind !== report.observation_source.kind || !Array.isArray(value.coordinate_names) || value.coordinate_names.length !== 6 || value.coordinate_names.some(item => typeof item !== 'string' || !item || item.length > 256) || !sameJson(value.units, job.request.native_replay.units) || !Array.isArray(value.records) || value.records.length !== report.observations) throw new Error('The saved action record does not match this replay job.');
  for (const [index, row] of value.records.entries()) {
    const expected = job.request.native_replay.selection[index];
    if (!record(row) || row.episode_index !== expected.episode_index || row.frame_index !== expected.frame_index || row.reset_repeat_exact !== true || !Array.isArray(row.actions) || row.actions.length !== 100 || row.actions.some(action => !Array.isArray(action) || action.length !== 6 || action.some(item => typeof item !== 'number' || !Number.isFinite(item)))) throw new Error('A complete finite 100 × 6 action record is required.');
  }
  return value as unknown as ReplayRecord;
}
