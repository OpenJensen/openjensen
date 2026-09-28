import { apiOrigin, type Job } from './api';
import { policyJobRequest, record, sameJson } from './policy-job-mutation';

export type ManagedTeachingRequest = { operation: 'teaching.capture'; profile_id: string; profile_sha256: string; timeout_seconds: number };
export type ManagedTeachingProfile = { id: string; label: string; profile_sha256: string; max_seconds: number; max_capture_bytes: number; runtime_verified: false; transport: 'local_owned_process' };
export type ManagedTeachingOptions = { configured: boolean; available: boolean; profiles: ManagedTeachingProfile[]; message: string };
export type ManagedCaptureEpisode = { episode_id: string; receipt_sha256: string; frames: number; termination: 'finish' | 'reset' | 'step_limit' | 'shutdown'; outcome: 'unknown' | 'operator_reported_failure' };
export type ManagedPublishedCapture = {
  project_id: string; job_id: string; profile_id: string; profile_sha256: string;
  session_id: string; session_sha256: string; inventory_sha256: string; recording_configuration_sha256: string;
  episodes: ManagedCaptureEpisode[]; origin: 'recorded' | 'synthetic'; lineage_group: string;
};
export type ManagedTeachingStatus = { job: Job; ready: boolean; session_id: string | null; stop_requested: boolean };
export type TeachingState = { mode: string; episode_id: string | null; revision: number; session_id: string | null; instruction: string; outcome: string; steps: number; sim_time: number; joints: string[]; state_rad: number[] | null; fault: string | null };
export type TeachingConnection = { connected: boolean; state: TeachingState | null; message: string | null; voice_configured: boolean };
export type TeachingControlReceipt = { command_id: string; status: 'queued' | 'executing' | 'acknowledged' | 'rejected'; error?: string };
export type TeachingTransport = {
  identity: string; frameUrl: string;
  request: <T>(path: string, body?: object, beforePost?: () => void) => Promise<T>;
};
const hash = (v: unknown): v is string => typeof v === 'string' && /^[a-f0-9]{64}$/.test(v);
const session = (v: unknown): v is string => typeof v === 'string' && /^[a-f0-9]{32}$/.test(v);
const id = (v: unknown): v is string => typeof v === 'string' && /^[A-Za-z0-9_-]{1,96}$/.test(v);
const text = (v: unknown, maximum = 256): v is string => typeof v === 'string' && v.length > 0 && v.length <= maximum && !/[\x00-\x1f\x7f]/.test(v);
const integer = (v: unknown, min: number, max: number): v is number => typeof v === 'number' && Number.isSafeInteger(v) && v >= min && v <= max;
const statuses = ['queued', 'running', 'succeeded', 'failed', 'cancelled', 'interrupted'];
function need(value: unknown, message = 'Managed teaching response has invalid or changed identity.'): asserts value { if (!value) throw new Error(message); }
const base = (project: string) => { need(text(project)); return `/projects/${encodeURIComponent(project)}/teaching`; };
const endpoint = (project: string, job: string) => { need(text(job)); return `${base(project)}/sessions/${encodeURIComponent(job)}`; };

export function managedTeachingRequest(value: unknown): ManagedTeachingRequest {
  need(record(value) && Object.keys(value).sort().join(',') === 'operation,profile_id,profile_sha256,timeout_seconds');
  need(value.operation === 'teaching.capture' && typeof value.profile_id === 'string' && /^[A-Za-z0-9_-]{1,64}$/.test(value.profile_id) && hash(value.profile_sha256) && integer(value.timeout_seconds, 1, 3600));
  return value as ManagedTeachingRequest;
}
export function managedTeachingOptions(value: unknown): ManagedTeachingOptions {
  need(record(value) && typeof value.configured === 'boolean' && typeof value.available === 'boolean' && text(value.message, 1024) && Array.isArray(value.profiles) && value.profiles.length <= 32);
  const profiles = value.profiles.map(p => {
    need(record(p) && typeof p.id === 'string' && /^[A-Za-z0-9_-]{1,64}$/.test(p.id) && text(p.label, 100) && hash(p.profile_sha256) && integer(p.max_seconds, 1, 3600) && integer(p.max_capture_bytes, 1024, 8 * 1024 ** 3) && p.runtime_verified === false && p.transport === 'local_owned_process');
    return p as ManagedTeachingProfile;
  });
  need(new Set(profiles.map(p => p.id)).size === profiles.length && (value.configured || !profiles.length) && (!value.available || (value.configured && profiles.length > 0)));
  return { configured: value.configured, available: value.available, profiles, message: value.message };
}
export function requireManagedProfile(options: ManagedTeachingOptions, request: ManagedTeachingRequest): void {
  const profile = options.profiles.find(p => p.id === request.profile_id);
  need(options.available && profile && profile.profile_sha256 === request.profile_sha256 && request.timeout_seconds <= profile.max_seconds, 'Teaching profile is unavailable or changed. Refresh and review its limits before starting.');
}

function teachingJobIdentity(value: unknown, project: string, expected?: unknown, jobId?: string): Job {
  need(record(value) && text(project) && text(value.id) && value.project_id === project && value.kind === 'teaching.capture' && typeof value.status === 'string' && statuses.includes(value.status));
  need(typeof value.created_at === 'string' && typeof value.updated_at === 'string' && Number.isFinite(Date.parse(value.created_at)) && Number.isFinite(Date.parse(value.updated_at)) && Date.parse(value.updated_at) >= Date.parse(value.created_at));
  const request = managedTeachingRequest(value.request);
  if (expected !== undefined) need(sameJson(request, managedTeachingRequest(expected)));
  if (jobId !== undefined) need(value.id === jobId);
  return value as Job;
}
/** A retained ACK carries identity only; it never proves publication or readiness. */
export function retainedManagedTeachingReceipt(value: unknown, project: string): Job {
  const job = teachingJobIdentity(value, project);
  need(job.result === null && job.error === null && job.stage === null);
  return job;
}
export function managedTeachingJob(value: unknown, project: string, expected?: unknown, jobId?: string): Job {
  const job = teachingJobIdentity(value, project, expected, jobId);
  if (job.status === 'succeeded') publishedCapture(job);
  return job;
}
export function publishedCapture(job: Job): ManagedPublishedCapture {
  need(job.kind === 'teaching.capture' && job.status === 'succeeded');
  const request = managedTeachingRequest(job.request), result: unknown = job.result;
  need(record(result) && result.schema_version === 1 && result.operation === request.operation && result.profile_id === request.profile_id && result.profile_sha256 === request.profile_sha256);
  need(session(result.session_id) && hash(result.session_sha256) && hash(result.inventory_sha256) && hash(result.recording_configuration_sha256));
  need(result.published === true && result.content_verified === true && result.process_cleanup_verified === true && result.simulator_coordinates === true && result.physical_calibration_verified === false && result.task_success_claimed === false);
  need((result.origin === 'recorded' || result.origin === 'synthetic') && typeof result.lineage_group === 'string' && /^[A-Za-z0-9_.:-]{1,128}$/.test(result.lineage_group) && Array.isArray(result.episodes) && result.episodes.length > 0 && result.episodes.length <= 100);
  const episodes = result.episodes.map(e => {
    need(record(e) && session(e.episode_id) && hash(e.receipt_sha256) && integer(e.frames, 1, 3600) && typeof e.termination === 'string' && ['finish', 'reset', 'step_limit', 'shutdown'].includes(e.termination) && typeof e.outcome === 'string' && ['unknown', 'operator_reported_failure'].includes(e.outcome));
    return { episode_id: e.episode_id, receipt_sha256: e.receipt_sha256, frames: e.frames, termination: e.termination, outcome: e.outcome } as ManagedCaptureEpisode;
  });
  need(new Set(episodes.map(e => e.episode_id)).size === episodes.length);
  return { project_id: job.project_id, job_id: job.id, profile_id: request.profile_id, profile_sha256: request.profile_sha256, session_id: result.session_id, session_sha256: result.session_sha256, inventory_sha256: result.inventory_sha256, recording_configuration_sha256: result.recording_configuration_sha256, episodes, origin: result.origin, lineage_group: result.lineage_group };
}
export function managedTeachingStatus(value: unknown, project: string, job: string, expected?: unknown): ManagedTeachingStatus {
  need(record(value) && typeof value.ready === 'boolean' && typeof value.stop_requested === 'boolean' && (value.session_id === null || session(value.session_id)));
  const saved = managedTeachingJob(value.job, project, expected, job);
  need(!value.ready || (saved.status === 'running' && value.session_id !== null && !value.stop_requested));
  if (saved.status === 'succeeded') need(value.session_id === publishedCapture(saved).session_id);
  return { job: saved, ready: value.ready, session_id: value.session_id, stop_requested: value.stop_requested };
}
export function managedTeachingHistory(value: unknown, project: string): Job[] {
  need(Array.isArray(value) && value.length <= 10000);
  const jobs: Job[] = [];
  for (const job of value) {
    need(record(job) && text(job.id) && job.project_id === project && typeof job.kind === 'string');
    if (job.kind === 'teaching.capture') jobs.push(managedTeachingJob(job, project));
  }
  need(new Set(jobs.map(j => j.id)).size === jobs.length);
  return jobs.sort((a, b) => b.created_at.localeCompare(a.created_at));
}
export async function readManagedOptions(project: string): Promise<ManagedTeachingOptions> { return managedTeachingOptions(await policyJobRequest(`${base(project)}/profiles`)); }
export async function readManagedHistory(project: string): Promise<Job[]> { need(text(project)); return managedTeachingHistory(await policyJobRequest(`/projects/${encodeURIComponent(project)}/jobs`), project); }
export async function readManagedStatus(project: string, job: string, expected?: unknown): Promise<ManagedTeachingStatus> { return managedTeachingStatus(await policyJobRequest(endpoint(project, job)), project, job, expected); }
export async function stopManagedSession(project: string, job: string, expected: unknown): Promise<ManagedTeachingStatus> {
  const status = managedTeachingStatus(await policyJobRequest(`${endpoint(project, job)}/stop`, {}), project, job, expected);
  need(!['queued', 'running'].includes(status.job.status) || status.stop_requested, 'Stop acknowledgement did not confirm this session is stopping.');
  return status;
}

/** A managed transport can never fall through to the manual or provider routes. */
export function managedTeachingTransport(project: string, job: string, sessionId: string, expected: ManagedTeachingRequest): TeachingTransport {
  need(session(sessionId));
  const request = managedTeachingRequest(expected), root = endpoint(project, job);
  async function state(): Promise<TeachingConnection> {
    const status = await readManagedStatus(project, job, request);
    need(status.ready && status.session_id === sessionId, 'This owned teaching session is no longer ready. Review its saved status.');
    const value = await policyJobRequest(`${root}/state`);
    need(record(value) && value.session_id === sessionId && typeof value.mode === 'string' && ['idle', 'running', 'paused', 'faulted', 'closed', 'starting'].includes(value.mode) && (value.episode_id === null || id(value.episode_id)) && integer(value.revision, 0, Number.MAX_SAFE_INTEGER));
    need(typeof value.instruction === 'string' && value.instruction.length <= 256 && typeof value.outcome === 'string' && value.outcome.length <= 256 && integer(value.steps, 0, Number.MAX_SAFE_INTEGER) && typeof value.sim_time === 'number' && Number.isFinite(value.sim_time) && value.sim_time >= 0 && Array.isArray(value.joints) && value.joints.length <= 64 && value.joints.every(v => text(v)) && new Set(value.joints).size === value.joints.length);
    need(value.state_rad === null || (Array.isArray(value.state_rad) && value.state_rad.length === value.joints.length && value.state_rad.every(v => typeof v === 'number' && Number.isFinite(v))));
    need(value.fault === null || typeof value.fault === 'string');
    return { connected: true, state: value as TeachingState, message: null, voice_configured: false };
  }
  return { identity: JSON.stringify([project, job, sessionId]), frameUrl: `${apiOrigin}/api/v1${root}/frame`, request: async <T>(path: string, body?: object, beforePost?: () => void): Promise<T> => {
    if (path === '/state' && body === undefined) return await state() as T;
    need((path === '/commands' && body !== undefined) || (/^\/commands\/[A-Za-z0-9_-]{1,96}$/.test(path) && body === undefined), 'This service is unavailable for a managed teaching session.');
    await state();
    if (body !== undefined) need(record(body) && body.session_id === sessionId && id(body.command_id));
    if (body !== undefined) beforePost?.();
    const result = await policyJobRequest(root + path, body);
    const expectedId = body !== undefined && record(body) ? body.command_id : path.split('/').pop();
    need(record(result) && result.command_id === expectedId && typeof result.status === 'string' && ['queued', 'executing', 'acknowledged', 'rejected'].includes(result.status));
    if (record(result.state)) need(result.state.session_id === sessionId);
    return result as T;
  } };
}
