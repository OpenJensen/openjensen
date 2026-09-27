import { apiOrigin, type Job, type PolicyArtifact } from './api';

// Additive API types until the combined server schema is generated. No operator
// paths, commands, credentials or arbitrary media URLs are accepted by this client.
export type SimulationTarget = { profile_id: string; profile_sha256: string; provider: 'gcp'; accelerators: ('L4' | 'H100')[]; source_manifest_sha256: string; model_id?: string | null };
export type SimulationProfile = { id: string; label: string; architectures: ('act' | 'smolvla')[]; experimental: true; task_object: 'cup' };
export type SimulationOptions = { profiles: SimulationProfile[]; unavailable_reason?: string | null; max_archive_bytes: number; scored_evaluation: false };
export type NativeArtifact = Omit<PolicyArtifact, 'format'> & { format: PolicyArtifact['format'] | 'simulation_record' };
export type SimulationJob = Job & { simulation_target?: SimulationTarget | null };
export const MAX_ARCHIVE_BYTES = 4 * 1024 ** 3;
const prefix = `${apiOrigin}/api/v1`;
const encode = encodeURIComponent;

export class UncertainSubmission extends Error {}
class RejectedRequest extends Error {}
function uncertain() { return new UncertainSubmission('The submission outcome is unverified. Check the job list before making another request; this request was not retried.'); }
export function objectRecord(value: unknown): value is Record<string, unknown> { return typeof value === 'object' && value !== null && !Array.isArray(value); }
function message(value: unknown, status: number) { return objectRecord(value) && typeof value.detail === 'string' ? value.detail.slice(0, 2000) : `The application returned HTTP ${status}.`; }
export function simulationTarget(job: Job): SimulationTarget | null {
  const target = (job as SimulationJob).simulation_target;
  return target?.provider === 'gcp' && typeof target.profile_id === 'string' && Array.isArray(target.accelerators) ? target : null;
}
export function isSimulationJob(job: Job) {
  return simulationTarget(job) !== null || ('simulation' in job.request && objectRecord(job.request.simulation) && typeof job.request.simulation.profile_id === 'string');
}
export function nativeInput(artifact: NativeArtifact, projectId: string, profile?: SimulationProfile) {
  const architecture = artifact.metadata?.architecture;
  return artifact.project_id === projectId && ['inference_export', 'native_checkpoint'].includes(artifact.format) &&
    (architecture === 'act' || architecture === 'smolvla') && (!profile || profile.architectures.includes(architecture)) &&
    artifact.metadata?.storage !== 'gcs' && !artifact.metadata?.remote_uri;
}
export function acceptedJob(value: unknown, projectId: string, operation: string, profileId?: string, expectedId?: string): SimulationJob {
  if (!objectRecord(value) || typeof value.id !== 'string' || !value.id || value.project_id !== projectId || value.kind !== operation ||
      (typeof value.status !== 'string' || !['queued', 'running', 'succeeded', 'failed', 'cancelled', 'interrupted'].includes(value.status)) ||
      (expectedId && value.id !== expectedId) || !objectRecord(value.request) || value.request.operation !== operation ||
      (profileId && (!objectRecord(value.request.simulation) || value.request.simulation.profile_id !== profileId))) throw uncertain();
  return value as unknown as SimulationJob;
}
async function json(path: string, init?: RequestInit) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 15_000);
  const mutation = init?.method === 'POST';
  try {
    const response = await fetch(`${prefix}${path}`, { ...init, cache: 'no-store', redirect: 'error', signal: controller.signal, headers: { 'Content-Type': 'application/json', ...init?.headers } });
    const reader = response.body?.getReader();
    if (!reader) throw new Error('Missing application response');
    const parts: Uint8Array[] = []; let bytes = 0;
    try { while (true) { const part = await reader.read(); if (part.done) break; bytes += part.value.length; if (bytes > 1024 * 1024) throw new Error('Application response exceeds its limit'); parts.push(part.value); } }
    finally { await reader.cancel(); }
    const buffer = new Uint8Array(bytes); let offset = 0;
    for (const part of parts) { buffer.set(part, offset); offset += part.length; }
    const value: unknown = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(buffer));
    if (!response.ok) {
      if (mutation && response.status >= 500) throw uncertain();
      throw new RejectedRequest(message(value, response.status));
    }
    return value;
  } catch (error) {
    if (error instanceof UncertainSubmission) throw error;
    if (mutation && !(error instanceof RejectedRequest)) throw uncertain();
    throw error instanceof Error ? error : new Error('The application could not be reached.');
  } finally { clearTimeout(timer); }
}
export async function simulationOptions(): Promise<SimulationOptions> {
  const value = await json('/simulation-options');
  if (!objectRecord(value) || !Array.isArray(value.profiles) || value.profiles.length > 100 ||
      !Number.isSafeInteger(value.max_archive_bytes) || Number(value.max_archive_bytes) <= 0 || Number(value.max_archive_bytes) > MAX_ARCHIVE_BYTES || value.scored_evaluation !== false ||
      value.profiles.some(p => !objectRecord(p) || typeof p.id !== 'string' || !/^[\w-]{1,100}$/.test(p.id) || typeof p.label !== 'string' || p.experimental !== true || p.task_object !== 'cup' ||
        !Array.isArray(p.architectures) || !p.architectures.length || p.architectures.some(a => a !== 'act' && a !== 'smolvla'))) throw new Error('The application returned unsupported simulation options.');
  return value as SimulationOptions;
}
export async function startSimulation(projectId: string, profileId: string, artifactId: string, timeout: number, manifestSha256: string) {
  if (!/^[a-f0-9]{64}$/.test(manifestSha256)) throw new Error('The selected policy has no valid manifest identity. Refresh saved policies.');
  const result = await json(`/projects/${encode(projectId)}/policy-jobs`, { method: 'POST', body: JSON.stringify({ operation: 'policy.run', runtime_id: profileId, artifact_id: artifactId, simulation: { profile_id: profileId, experimental: true }, timeout_seconds: timeout }) });
  const receipt = acceptedJob(result, projectId, 'policy.run', profileId);
  if (!('artifact_id' in receipt.request) || receipt.request.artifact_id !== artifactId || receipt.request.runtime_id !== profileId || receipt.request.timeout_seconds !== timeout || !objectRecord(receipt.request.simulation) || receipt.request.simulation.experimental !== true) throw uncertain();
  if (!receipt.simulation_target || receipt.simulation_target.profile_id !== profileId || receipt.simulation_target.source_manifest_sha256 !== manifestSha256) throw uncertain();
  return receipt;
}
function sameSimulationRequest(left: Job, right: Job): boolean {
  const a: unknown = left.request, b: unknown = right.request;
  return objectRecord(a) && objectRecord(b) && ['operation', 'runtime_id', 'artifact_id', 'source_id', 'timeout_seconds'].every(key => (a[key] ?? null) === (b[key] ?? null)) &&
    objectRecord(a.simulation) && objectRecord(b.simulation) && a.simulation.profile_id === b.simulation.profile_id && a.simulation.experimental === b.simulation.experimental;
}
export async function cancelSimulation(job: Job, stillSelected: () => boolean = () => true) {
  const current = acceptedJob(await json(`/jobs/${encode(job.id)}`), job.project_id, job.kind, undefined, job.id);
  if (!isSimulationJob(current) || !sameSimulationRequest(current, job)) throw uncertain();
  if (!stillSelected()) throw new Error('Selection changed; cancellation was not submitted.');
  if (!['queued', 'running'].includes(current.status)) return current;
  const receipt = acceptedJob(await json(`/jobs/${encode(job.id)}/cancel`, { method: 'POST' }), job.project_id, job.kind, undefined, job.id);
  if (!sameSimulationRequest(receipt, job)) throw uncertain();
  return receipt;
}
export function uploadModel(projectId: string, profileId: string, file: File, progress: (percent: number) => void): { result: Promise<SimulationJob>; abort: () => void } {
  const xhr = new XMLHttpRequest();
  const result = new Promise<SimulationJob>((resolve, reject) => {
    if (!file.size || file.size > MAX_ARCHIVE_BYTES) { reject(new Error('Choose a nonempty TAR archive no larger than 4 GiB.')); return; }
    const url = new URL(`${prefix}/projects/${encode(projectId)}/model-imports?profile_id=${encode(profileId)}`, window.location.href).href;
    xhr.open('POST', url);
    xhr.setRequestHeader('Content-Type', 'application/x-tar');
    // The server's upload deadline is 300 seconds; leave time for its receipt.
    xhr.timeout = 330_000;
    xhr.upload.onprogress = event => { if (event.lengthComputable) progress(Math.min(100, Math.floor(event.loaded * 100 / event.total))); };
    xhr.onerror = xhr.ontimeout = xhr.onabort = () => reject(uncertain());
    xhr.onload = () => {
      try {
        // XHR cannot disable redirects. Detect changed final URLs; this cannot prevent an already-followed upload.
        if (xhr.responseURL !== url || xhr.responseText.length > 1024 * 1024) throw uncertain();
        const value: unknown = JSON.parse(xhr.responseText);
        if (xhr.status >= 500) throw uncertain();
        if (xhr.status < 200 || xhr.status >= 300) throw new Error(message(value, xhr.status));
        const receipt = acceptedJob(value, projectId, 'policy.import', profileId);
        if (!('source_id' in receipt.request) || receipt.request.runtime_id !== profileId || typeof receipt.request.source_id !== 'string' || !receipt.request.source_id || receipt.request.source_id.length > 512 || receipt.request.timeout_seconds !== 600 || !objectRecord(receipt.request.simulation) || receipt.request.simulation.experimental !== false) throw uncertain();
        resolve(receipt);
      } catch (error) { reject(error instanceof SyntaxError ? uncertain() : error); }
    };
    xhr.send(file);
  });
  return { result, abort: () => xhr.abort() };
}
export const simulationVideoUrl = (id: string) => `${prefix}/jobs/${encode(id)}/simulation-media/video`;

// Match the pinned runner's observed states; ignore extra fields and arbitrary
// strings instead of presenting them as a verified worker status.
const workerStates = new Set(['PENDING', 'SUBMITTED', 'STARTING', 'RUNNING', 'WINDING_DOWN', 'RECOVERING', 'CANCELLING', 'SUCCEEDED', 'CANCELLED', 'FAILED', 'FAILED_SETUP', 'FAILED_PRECHECKS', 'FAILED_NO_RESOURCE', 'FAILED_CONTROLLER']);
export function simulationTaskSummary(data: unknown): string {
  if (!objectRecord(data) || !objectRecord(data.tasks)) return '';
  const tasks = data.tasks;
  return (['isaac', 'vla'] as const).flatMap(role => typeof tasks[role] === 'string' && workerStates.has(tasks[role]) ? [`${role === 'isaac' ? 'Isaac' : 'VLA'}: ${tasks[role]}`] : []).join(' · ');
}
