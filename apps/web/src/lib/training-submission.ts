import type { Job, PolicyRequest } from './api';
import { policyJobRequest, record, sameJson, UncertainPolicyJob } from './policy-job-mutation';

/** A successful HTTP response must acknowledge this exact training request. */
export async function startTraining(project: string, body: PolicyRequest, methods: readonly string[]): Promise<Job> {
  const value = await policyJobRequest(`/projects/${encodeURIComponent(project)}/policy-jobs`, body);
  return trainingReceipt(value, project, body, methods);
}

/** Validate POST or saved-key lookup against the persisted original recipe. */
export function trainingReceipt(value: unknown, project: string, originalBody: unknown, methods: readonly string[] = ['lora', 'qlora', 'full']): Job {
  if (!record(originalBody) || originalBody.operation !== 'policy.finetune' || typeof originalBody.runtime_id !== 'string') throw new UncertainPolicyJob();
  const body = originalBody as PolicyRequest;
  if (!record(value) || typeof value.id !== 'string' || !value.id || value.project_id !== project ||
      value.kind !== 'policy.finetune' || typeof value.status !== 'string' || !['queued', 'running', 'succeeded', 'failed', 'cancelled', 'interrupted'].includes(value.status) ||
      !record(value.request) || ![value.created_at, value.updated_at].every(date => typeof date === 'string' && Number.isFinite(Date.parse(date)))) throw new UncertainPolicyJob();
  const request = value.request;
  if (typeof request.dataset_job_id !== 'string' || !request.dataset_job_id || typeof request.training_method !== 'string' ||
      !methods.includes(request.training_method) || (request.training !== null && !record(request.training))) throw new UncertainPolicyJob();
  for (const key of ['operation', 'runtime_id', 'artifact_id', 'resume_job_id', 'timeout_seconds'] as const) {
    if (!sameJson(request[key] ?? null, body[key] ?? null)) throw new UncertainPolicyJob();
  }
  // Resume restores checkpoint-owned dataset/method/recipe. A new run must
  // retain every submitted setting; only the catalog-owned subdirectory may resolve.
  if (!body.resume_job_id && !body.artifact_id) {
    if (request.dataset_job_id !== body.dataset_job_id || request.training_method !== body.training_method || !record(request.training) || !body.training) throw new UncertainPolicyJob();
    for (const [key, expected] of Object.entries(body.training)) {
      if (key !== 'checkpoint_subdirectory' && !sameJson(request.training[key], expected)) throw new UncertainPolicyJob();
    }
  }
  return value as unknown as Job;
}
