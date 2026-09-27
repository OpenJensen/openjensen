import { type Job } from './api';
import { policyJobRequest, record, UncertainPolicyJob } from './policy-job-mutation';
import { storedAttempt, type PolicyJobAttempt } from './policy-job-attempt';

export type ActExportIdentity = { project: string; artifact: string; runtime: string };
const statuses = ['queued', 'running', 'succeeded', 'failed', 'cancelled', 'interrupted'];
const text = (value: unknown): value is string => typeof value === 'string' && value.length > 0 && value.length <= 512;
const timestamp = (value: unknown): value is string => typeof value === 'string' && value.length <= 64 && Number.isFinite(Date.parse(value));
export function actExportContext(expected: ActExportIdentity): string {
  if (![expected.project, expected.artifact, expected.runtime].every(text)) throw new Error('Export identity is invalid. Refresh the project before submitting.');
  return `ACT export · Project: ${expected.project} · Checkpoint: ${expected.artifact} · Worker: ${expected.runtime} · Recipe: full FP32, 600 seconds.`;
}
/** Preserve the original source when a page reload converts pending into uncertainty. */
export function storedActExportAttempt(project: string): PolicyJobAttempt {
  const raw = sessionStorage.getItem(`firebird:job-attempt:policy.export:${project}`);
  if (raw !== null && raw.length > 16_384) throw new Error('Export recovery record is too large.');
  let value: unknown;
  try { value = raw === null ? null : JSON.parse(raw); } catch { /* Shared recovery handles malformed records conservatively. */ }
  if (record(value) && value.state === 'pending' && typeof value.message === 'string' && value.message.length <= 1800) {
    return { state: 'uncertain', message: `${value.message} ${new UncertainPolicyJob().message}` };
  }
  return storedAttempt('policy.export', project);
}
export function actExportRecipe(expected: ActExportIdentity) {
  return { operation: 'policy.export' as const, runtime_id: expected.runtime, artifact_id: expected.artifact, training_method: 'full' as const, timeout_seconds: 600 };
}
/** Keep a small identity receipt, never a cached claim about output/model verification. */
export function actExportReceipt(value: unknown, expected: ActExportIdentity): Job {
  if (!record(value) || !text(value.id) || value.project_id !== expected.project || value.kind !== 'policy.export' ||
      typeof value.status !== 'string' || !statuses.includes(value.status) || !timestamp(value.created_at) || !timestamp(value.updated_at) ||
      !record(value.request) || Object.entries(actExportRecipe(expected)).some(([key, item]) => value.request && (value.request as Record<string, unknown>)[key] !== item) ||
      (value.stage != null && (typeof value.stage !== 'string' || value.stage.length > 512)) ||
      (value.error != null && (typeof value.error !== 'string' || value.error.length > 2000))) throw new UncertainPolicyJob();
  return { id: value.id, project_id: expected.project, kind: 'policy.export', status: value.status,
    request: actExportRecipe(expected), created_at: value.created_at, updated_at: value.updated_at,
    stage: value.stage ?? null, error: value.error ?? null, result: null } as Job;
}
export async function startActExport(expected: ActExportIdentity): Promise<Job> {
  if (![expected.project, expected.artifact, expected.runtime].every(text)) throw new Error('Export identity is invalid. Refresh the project before submitting.');
  return actExportReceipt(await policyJobRequest(`/projects/${encodeURIComponent(expected.project)}/policy-jobs`, actExportRecipe(expected)), expected);
}
const receiptKey = (project: string, artifact: string) => `firebird:act-export-receipt:${project}:${artifact}`;
export function storeActExportReceipt(job: Job): void {
  const request = job.request;
  if (!('artifact_id' in request) || !request.artifact_id) throw new Error('Export receipt has no source.');
  sessionStorage.setItem(receiptKey(job.project_id, request.artifact_id), JSON.stringify(job));
}
export function storedActExportReceipt(project: string, artifact: string): Job | null {
  const raw = sessionStorage.getItem(receiptKey(project, artifact));
  if (raw === null) return null;
  if (raw.length > 16_384) throw new Error('Export receipt is too large.');
  const value: unknown = JSON.parse(raw);
  if (!record(value) || !record(value.request) || !text(value.request.runtime_id)) throw new Error('Export receipt is invalid.');
  return actExportReceipt(value, { project, artifact, runtime: value.request.runtime_id });
}
