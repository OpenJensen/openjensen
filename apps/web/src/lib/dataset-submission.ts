import { recordingRecipe, recordingReceipt, recordingRequest } from './recording-preparation';
import type { AugmentationRequest, IntakeRequest, Job } from './api';
import { record, sameJson, UncertainPolicyJob } from './policy-job-mutation';

const statuses = ['queued', 'running', 'succeeded', 'failed', 'cancelled', 'interrupted'];

function acknowledgement(value: unknown, project: string, operation: string): Record<string, unknown> {
  if (!record(value) || typeof value.id !== 'string' || !value.id || value.id.length > 200 ||
      value.project_id !== project || value.kind !== operation || (typeof value.status !== 'string' || !statuses.includes(value.status)) ||
      !record(value.request) || ![value.created_at, value.updated_at].every(date => typeof date === 'string' && Number.isFinite(Date.parse(date)))) {
    throw new UncertainPolicyJob();
  }
  return value;
}

/** Validate the submitted fields; the echoed durable key binds server-resolved paths/revisions. */
export function intakeAcknowledgement(value: unknown, project: string, original: unknown): Job {
  const job = acknowledgement(value, project, 'dataset.inspect');
  if (!record(original)) throw new UncertainPolicyJob();
  if (original.recordings != null) {
    // Recording intake deliberately has a null host path. Its exact normalized
    // recipe is shared with the recording panel, and the full accepted job stays intact.
    const recipe = recordingRecipe(original.recordings);
    if (!sameJson(recordingRequest(recipe), original)) throw new UncertainPolicyJob();
    recordingReceipt(job, project, recipe);
    return job as unknown as Job;
  }
  const body = original as IntakeRequest;
  const request = job.request as Record<string, unknown>;
  if (request.source !== body.source || (request.snapshot_for_training ?? false) !== (body.snapshot_for_training ?? false) ||
      !sameJson(request.recordings ?? null, body.recordings ?? null)) throw new UncertainPolicyJob();
  if (body.source === 'huggingface') {
    const submittedRevision = body.revision?.trim() || 'main';
    if (request.repo_id !== body.repo_id || request.path != null || typeof request.revision !== 'string' ||
        (/^[a-f0-9]{40}$/i.test(submittedRevision) ? request.revision !== submittedRevision :
          request.revision !== submittedRevision && !/^[a-f0-9]{40}$/i.test(request.revision))) throw new UncertainPolicyJob();
  } else if (body.library_id) {
    if (request.library_id !== body.library_id || request.path != null || request.repo_id != null || request.revision !== 'main') throw new UncertainPolicyJob();
  } else {
    // Metadata inspection resolves allowed relative paths and symlinks on the API host.
    // A training snapshot retains its original path; never guess a host filesystem here.
    if (request.repo_id != null || typeof request.path !== 'string' || !request.path ||
        (body.snapshot_for_training && request.path !== body.path)) throw new UncertainPolicyJob();
  }
  return job as unknown as Job;
}

export function augmentationAcknowledgement(value: unknown, project: string, original: unknown): Job {
  const job = acknowledgement(value, project, 'dataset.augment');
  if (!record(original)) throw new UncertainPolicyJob();
  const request = job.request as Record<string, unknown>;
  const defaults: Partial<AugmentationRequest> = { operation: 'dataset.augment', preset: 'lighting', prompt: '', start_seconds: 0, duration_seconds: 5 };
  for (const [key, expected] of Object.entries({ ...defaults, ...original })) {
    if (!sameJson(request[key], expected)) throw new UncertainPolicyJob();
  }
  return job as unknown as Job;
}

/** History acknowledgment must use a newly fetched complete project response. */
export function validatedProjectHistory(value: unknown, project: string): Job[] {
  if (!project || !Array.isArray(value)) throw new Error('Project history could not be verified.');
  for (const job of value) {
    if (!record(job) || typeof job.kind !== 'string') throw new Error('Project history could not be verified.');
    acknowledgement(job, project, job.kind);
  }
  return value as Job[];
}

/** An unkeyed intake cannot be replaced while a known intake may still be running. */
export function reviewedIntakeHistory(value: unknown, project: string): Job[] {
  const history = validatedProjectHistory(value, project);
  if (history.length > 10000) throw new Error('Project history exceeds its review limit.');
  for (const job of history) {
    if (job.kind !== 'dataset.inspect') continue;
    if ('recordings' in job.request && job.request.recordings != null) recordingReceipt(job, project);
    if (job.status === 'queued' || job.status === 'running') throw new Error(`Dataset intake ${job.id} is still ${job.status}. Inspect its saved job before clearing legacy recovery.`);
  }
  return history;
}
