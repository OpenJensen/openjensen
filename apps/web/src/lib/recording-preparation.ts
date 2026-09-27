import type { components } from './api.generated';
import type { DatasetJob, DatasetProfile } from './api';
import { policyJobRequest, record, sameJson, UncertainPolicyJob } from './policy-job-mutation';
import type { PolicyJobAttempt } from './policy-job-attempt';

export type RecordingRecipe = Required<components['schemas']['RecordingPreparation']>;
export type RecordingCatalog = components['schemas']['RecordingCatalog'];
export type RecordingOptions = Required<components['schemas']['RecordingOptions']>;
export type RecordingIntake = { source: 'local'; repo_id: null; revision: 'main'; path: null; snapshot_for_training: true; recordings: RecordingRecipe };
export type RecordingJob = DatasetJob & { request: RecordingIntake };
export type RecordingState = { schema_version: 1; project_id: string; recipe: RecordingRecipe | null; selected_job_id: string; pending: { attempt_id: string; action: 'submit' | 'cancel'; recipe: RecordingRecipe; job_id: string | null } | null };
export const RECORDING_OPERATION = 'dataset.inspect.recordings';
const statuses = ['queued', 'running', 'succeeded', 'failed', 'cancelled', 'interrupted'];
const ID = /^[a-f0-9]{32}$/, HASH = /^[a-f0-9]{64}$/;
const text = (v: unknown, max = 512): v is string => typeof v === 'string' && v.length > 0 && v.length <= max && !/[\x00-\x1f\x7f]/.test(v);
const hash = (v: unknown): v is string => typeof v === 'string' && HASH.test(v);
const id = (v: unknown): v is string => typeof v === 'string' && ID.test(v);
const integer = (v: unknown, min: number, max: number): v is number => typeof v === 'number' && Number.isSafeInteger(v) && v >= min && v <= max;
const timestamp = (v: unknown): v is string => typeof v === 'string' && /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,9})?(?:Z|[+-]\d\d:\d\d)$/.test(v) && Number.isFinite(Date.parse(v));
function need(ok: unknown, message = 'Recording metadata is invalid. Refresh and review the published captures.'): asserts ok { if (!ok) throw new Error(message); }
function keys(v: unknown, names: string[]): asserts v is Record<string, unknown> { need(record(v) && sameJson(Object.keys(v).sort(), [...names].sort())); }
function strings(value: unknown, max: number): value is string[] { return Array.isArray(value) && value.length <= max && value.every(item => typeof item === 'string' && item.length <= 2000); }

export function recordingRecipe(value: unknown): RecordingRecipe {
  keys(value, ['schema_version', 'configuration_sha256', 'timeout_seconds', 'captures']);
  need(value.schema_version === 1 && hash(value.configuration_sha256) && integer(value.timeout_seconds, 60, 1800));
  need(Array.isArray(value.captures) && value.captures.length > 0 && value.captures.length <= 100);
  const sessions = new Set<string>(), episodes = new Set<string>();
  const captures = value.captures.map(c => {
    keys(c, ['session_id', 'session_sha256', 'episodes']);
    need(id(c.session_id) && hash(c.session_sha256) && !sessions.has(c.session_id)); sessions.add(c.session_id);
    need(Array.isArray(c.episodes) && c.episodes.length > 0 && c.episodes.length <= 100);
    return { session_id: c.session_id, session_sha256: c.session_sha256, episodes: c.episodes.map(e => {
      keys(e, ['episode_id', 'receipt_sha256']); need(id(e.episode_id) && hash(e.receipt_sha256) && !episodes.has(e.episode_id)); episodes.add(e.episode_id);
      return { episode_id: e.episode_id, receipt_sha256: e.receipt_sha256 };
    }) };
  });
  need(episodes.size <= 100, 'Select at most 100 distinct episodes.');
  return { schema_version: 1, configuration_sha256: value.configuration_sha256, timeout_seconds: value.timeout_seconds, captures };
}
export function recordingRequest(value: RecordingRecipe): RecordingIntake {
  return { source: 'local', repo_id: null, revision: 'main', path: null, snapshot_for_training: true, recordings: recordingRecipe(value) };
}
export function recordingOptions(v: unknown): RecordingOptions {
  keys(v, ['configured', 'runtime_verified', 'configuration_sha256', 'max_episodes', 'max_source_bytes', 'setup_message']);
  need(typeof v.configured === 'boolean' && v.runtime_verified === false && (v.configured ? hash(v.configuration_sha256) : v.configuration_sha256 === null) &&
    v.max_episodes === 100 && v.max_source_bytes === 8 * 1024 ** 3 && text(v.setup_message, 2000));
  return v as RecordingOptions;
}
export function recordingCatalog(v: unknown): RecordingCatalog {
  keys(v, ['configuration_sha256', 'captures', 'message']); need(hash(v.configuration_sha256) && text(v.message, 2000) && Array.isArray(v.captures) && v.captures.length <= 100);
  const sessions = new Set<string>(), episodes = new Set<string>();
  for (const c of v.captures) {
    keys(c, ['session_id', 'session_sha256', 'origin', 'lineage_group', 'controller', 'state_units', 'action_units', 'timebase', 'camera_key', 'joint_names', 'width', 'height', 'fps', 'physics_hz', 'scene_sha256', 'camera_prim', 'episodes', 'content_verified']);
    need(id(c.session_id) && !sessions.has(c.session_id) && hash(c.session_sha256)); sessions.add(c.session_id);
    need((c.origin === 'recorded' || c.origin === 'synthetic') && typeof c.lineage_group === 'string' && /^[a-zA-Z0-9_.:-]{1,128}$/.test(c.lineage_group));
    need(c.controller === 'joint_position_targets' && c.state_units === 'radians' && c.action_units === 'radians' && c.timebase === 'simulation_seconds' && c.camera_key === 'observation.images.front' && c.content_verified === false);
    need(Array.isArray(c.joint_names) && c.joint_names.length > 0 && c.joint_names.length <= 32 && c.joint_names.every(n => text(n, 256) && /^[\p{ID_Start}_][\p{ID_Continue}]*$/u.test(n)) && new Set(c.joint_names).size === c.joint_names.length);
    need(integer(c.width, 2, 1920) && c.width % 2 === 0 && integer(c.height, 2, 1920) && c.height % 2 === 0 && integer(c.fps, 1, 60) && integer(c.physics_hz, c.fps, 1000) && hash(c.scene_sha256) && text(c.camera_prim, 256));
    need(Array.isArray(c.episodes) && c.episodes.length > 0 && c.episodes.length <= 1000);
    for (const e of c.episodes) {
      keys(e, ['episode_id', 'receipt_sha256', 'frames', 'outcome', 'termination']);
      need(id(e.episode_id) && !episodes.has(e.episode_id) && hash(e.receipt_sha256) && integer(e.frames, 1, 3600) && (e.outcome === 'unknown' || e.outcome === 'operator_reported_failure') && typeof e.termination === 'string' && ['finish', 'reset', 'step_limit', 'shutdown'].includes(e.termination)); episodes.add(e.episode_id);
    }
  }
  need(episodes.size <= 1000); return v as RecordingCatalog;
}
export function selectionMatches(recipe: RecordingRecipe, catalog: RecordingCatalog): boolean {
  return recipe.configuration_sha256 === catalog.configuration_sha256 && recipe.captures.every(c => {
    const current = catalog.captures.find(item => item.session_id === c.session_id);
    return current?.session_sha256 === c.session_sha256 && c.episodes.every(e => current.episodes.some(item => item.episode_id === e.episode_id && item.receipt_sha256 === e.receipt_sha256));
  });
}
export function recordingReceipt(v: unknown, project: string, expected?: RecordingRecipe, expectedId?: string): RecordingJob {
  try {
    need(record(v) && text(project) && text(v.id) && v.project_id === project && v.kind === 'dataset.inspect' && typeof v.status === 'string' && statuses.includes(v.status) && timestamp(v.created_at) && timestamp(v.updated_at) && Date.parse(v.updated_at) >= Date.parse(v.created_at));
    need(v.compute_target == null && v.simulation_target == null && (!expectedId || v.id === expectedId));
    keys(v.request, ['source', 'repo_id', 'revision', 'path', 'snapshot_for_training', 'recordings']);
    const request = recordingRequest(recordingRecipe(v.request.recordings)); need(sameJson(request, v.request) && (!expected || sameJson(request.recordings, expected)));
    // Admission is not verification of a result. Never persist unsolicited output or paths.
    return { id: v.id, project_id: project, kind: 'dataset.inspect', status: v.status, request, created_at: v.created_at, updated_at: v.updated_at, result: null, error: null, stage: null, compute_target: null, simulation_target: null } as RecordingJob;
  } catch { throw new UncertainPolicyJob(); }
}
function sorted(v: unknown): unknown { return Array.isArray(v) ? v.map(sorted) : record(v) ? Object.fromEntries(Object.keys(v).sort().map(k => [k, sorted(v[k])])) : v; }
export async function recordingSelectionHash(value: RecordingRecipe): Promise<string> {
  const raw = new TextEncoder().encode(JSON.stringify(sorted(recordingRecipe(value))));
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', raw))).map(x => x.toString(16).padStart(2, '0')).join('');
}
export async function recordingResult(v: unknown, job: RecordingJob): Promise<DatasetProfile> {
  need(job.status === 'succeeded' && record(v), 'The saved preparation has no verified complete result.');
  need(v.schema_version === 1 && v.source === 'local' && v.repo_id === null && v.format === 'lerobot_v3' && v.inspection_scope === 'complete_snapshot' && v.preview == null && hash(v.metadata_sha256) && v.revision === `metadata-sha256:${v.metadata_sha256}` && timestamp(v.inspected_at));
  need(integer(v.total_frames, 1, 360000) && integer(v.total_episodes, 1, 100) && typeof v.fps === 'number' && Number.isFinite(v.fps) && v.fps > 0 && v.fps <= 60 && record(v.features) && Object.keys(v.features).length > 0 && strings(v.warnings, 100));
  const s = v.snapshot, r = v.recording_preparation;
  need(record(s) && s.schema_version === 1 && hash(s.manifest_sha256) && s.id === `sha256:${s.manifest_sha256}` && s.format === 'lerobot_v3' && integer(s.total_bytes, 1, 16 * 1024 ** 3) && integer(s.file_count, 1, 4096) && s.total_frames === v.total_frames && s.total_episodes === v.total_episodes && s.lineage_validated === true && strings(s.warnings, 100));
  need(record(r) && r.job_id === job.id && r.source_count === job.request.recordings.captures.length && integer(r.lineage_group_count, 1, r.source_count) && r.writer_readback_verified === true && r.source_preserved === true && r.task_success_verified === false && r.selection_sha256 === await recordingSelectionHash(job.request.recordings));
  need(v.total_episodes === job.request.recordings.captures.reduce((n, c) => n + c.episodes.length, 0));
  return v as DatasetProfile;
}

const projectPath = (project: string) => { need(text(project)); return `/projects/${encodeURIComponent(project)}`; };
export async function readRecordingContext(project: string): Promise<{ options: RecordingOptions; catalog: RecordingCatalog | null }> {
  const options = recordingOptions(await policyJobRequest(`${projectPath(project)}/recordings/options`));
  if (!options.configured) return { options, catalog: null };
  const catalog = recordingCatalog(await policyJobRequest(`${projectPath(project)}/recordings`));
  need(options.configuration_sha256 === catalog.configuration_sha256, 'Recording configuration changed during refresh. Review the catalog again.');
  return { options, catalog };
}
export async function checkRecordingSubmission(project: string, recipe: RecordingRecipe, stillSelected: () => boolean): Promise<void> {
  const checked = recordingRecipe(recipe), context = await readRecordingContext(project);
  need(context.catalog && selectionMatches(checked, context.catalog), 'Selected recording bytes or configuration changed. Reselect and review the published episodes.');
  need(stillSelected(), 'Selection changed; preparation was not submitted.');
}

/** Legacy entry point retained for callers outside the durable browser controller. */
export async function submitRecordings(project: string, recipe: RecordingRecipe, stillSelected: () => boolean): Promise<RecordingJob> {
  await checkRecordingSubmission(project, recipe, stillSelected);
  return recordingReceipt(await policyJobRequest(`${projectPath(project)}/intakes`, recordingRequest(recipe)), project, recipe);
}
export async function readRecordingHistory(project: string): Promise<RecordingJob[]> {
  const raw = await policyJobRequest(`${projectPath(project)}/jobs`);
  need(Array.isArray(raw) && raw.length <= 10000, 'Saved recording history is invalid or exceeds its limit.');
  const jobs: RecordingJob[] = [];
  for (const value of raw) {
    need(record(value) && value.project_id === project, 'Job history contains a foreign or invalid project identity.');
    if (value.kind !== 'dataset.inspect') continue;
    need(record(value.request));
    if (value.request.recordings != null) jobs.push(recordingReceipt(value, project));
  }
  return jobs;
}
export async function readRecordingJob(project: string, jobId: string, recipe?: RecordingRecipe): Promise<RecordingJob> {
  need(text(jobId)); const raw = await policyJobRequest(`/jobs/${encodeURIComponent(jobId)}`), receipt = recordingReceipt(raw, project, recipe, jobId);
  need(record(raw));
  return { ...receipt, error: raw.error == null ? null : text(raw.error, 4000) ? raw.error : 'Saved job error is invalid.', stage: raw.stage == null ? null : text(raw.stage, 512) ? raw.stage : null,
    result: receipt.status === 'succeeded' ? await recordingResult(raw.result, receipt) : null };
}
export async function cancelRecording(job: RecordingJob, stillSelected: () => boolean): Promise<RecordingJob> {
  const current = await readRecordingJob(job.project_id, job.id, job.request.recordings);
  need(stillSelected(), 'Selection changed; cancellation was not submitted.');
  if (!['queued', 'running'].includes(current.status)) return current;
  return recordingReceipt(await policyJobRequest(`/jobs/${encodeURIComponent(job.id)}/cancel`, {}), job.project_id, job.request.recordings, job.id);
}

export class RecordingStorageUnavailable extends Error { constructor() { super('Recording recovery could not be saved or read in this browser session. Further requests are disabled; inspect saved jobs before retrying.'); } }
type Reader = Pick<Storage, 'getItem'>;
type Writer = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>;
const stateKey = (p: string) => `firebird:recording-preparation:${p}`;
export const emptyRecordingState = (project: string): RecordingState => ({ schema_version: 1, project_id: project, recipe: null, selected_job_id: '', pending: null });
function stateValue(project: string, v: unknown): RecordingState {
  keys(v, ['schema_version', 'project_id', 'recipe', 'selected_job_id', 'pending']);
  need(v.schema_version === 1 && v.project_id === project && typeof v.selected_job_id === 'string' && (v.selected_job_id === '' || text(v.selected_job_id)));
  const recipe = v.recipe === null ? null : recordingRecipe(v.recipe);
  let pending: RecordingState['pending'] = null;
  if (v.pending !== null) {
    keys(v.pending, ['attempt_id', 'action', 'recipe', 'job_id']); const p = v.pending; need(id(p.attempt_id));
    need((p.action === 'submit' && p.job_id === null) || (p.action === 'cancel' && text(p.job_id)));
    pending = { attempt_id: p.attempt_id, action: p.action as 'submit' | 'cancel', job_id: p.job_id as string | null, recipe: recordingRecipe(p.recipe) };
  }
  return { schema_version: 1, project_id: project, recipe, selected_job_id: v.selected_job_id, pending };
}
function boundedJSON(raw: string): unknown { need(new TextEncoder().encode(raw).length <= 65536); return JSON.parse(raw) as unknown; }
export function readRecordingState(project: string, storage: Reader = sessionStorage): RecordingState {
  try { need(text(project)); const raw = storage.getItem(stateKey(project)); return raw === null ? emptyRecordingState(project) : stateValue(project, boundedJSON(raw)); } catch { throw new RecordingStorageUnavailable(); }
}
export function writeRecordingState(project: string, v: RecordingState, storage: Writer = sessionStorage): void {
  try { need(text(project)); const raw = JSON.stringify(stateValue(project, v)); boundedJSON(raw); storage.setItem(stateKey(project), raw); } catch { throw new RecordingStorageUnavailable(); }
}
export function readRecordingAttempt(project: string, storage: Reader = sessionStorage): PolicyJobAttempt {
  let raw: string | null;
  try { raw = storage.getItem(`firebird:job-attempt:${RECORDING_OPERATION}:${project}`); }
  catch { throw new RecordingStorageUnavailable(); }
  if (raw === null) return null;
  try {
    need(raw.length <= 16384); const v: unknown = JSON.parse(raw);
    if (record(v) && (v.state === 'pending' || v.state === 'uncertain') && text(v.message, 1800)) return { state: 'uncertain', message: `${v.message}${v.state === 'pending' ? ' Outcome unverified after navigation or reload. Inspect saved jobs; no automatic retry was made.' : ''}` };
  } catch { /* Malformed journals remain uncertain; they never enable another submission. */ }
  return { state: 'uncertain', message: 'An earlier recording request has an unreadable outcome. Refresh and inspect saved jobs before another request.' };
}
export function storeRecordingReceipt(job: RecordingJob, storage: Writer = sessionStorage): void {
  try { const raw = JSON.stringify(recordingReceipt(job, job.project_id)); boundedJSON(raw); storage.setItem(`firebird:recording-receipt:${job.project_id}`, raw); } catch { throw new RecordingStorageUnavailable(); }
}
export function readRecordingReceipt(project: string, storage: Reader = sessionStorage): RecordingJob | null {
  try { const raw = storage.getItem(`firebird:recording-receipt:${project}`); return raw === null ? null : recordingReceipt(boundedJSON(raw), project); } catch { throw new RecordingStorageUnavailable(); }
}
export function recordingIntent(project: string, recipe: RecordingRecipe, action: 'submit' | 'cancel', jobId?: string): string {
  need(text(project) && (action !== 'cancel' || text(jobId)));
  return `Recordings · Project ${project} · ${action === 'cancel' ? `Cancel job ${jobId}` : 'Prepare dataset'} · ${recipe.captures.length} captures / ${recipe.captures.reduce((n, c) => n + c.episodes.length, 0)} episodes · Configuration ${recipe.configuration_sha256} · ${recipe.timeout_seconds} seconds. Exact original selection retained in this session.`;
}

const attemptKey = (project: string) => `firebird:job-attempt:${RECORDING_OPERATION}:${project}`;
export function ownsRecordingAttempt(project: string, attemptId: string, storage: Reader = sessionStorage): boolean {
  need(id(attemptId)); return readRecordingState(project, storage).pending?.attempt_id === attemptId;
}
/** Synchronous session-storage transitions; old mounted controllers cannot mutate a successor. */
export function beginRecordingAttempt(project: string, pending: NonNullable<RecordingState['pending']>, message: string, storage: Writer = sessionStorage): RecordingState {
  const current = readRecordingState(project, storage);
  need(current.pending === null && readRecordingAttempt(project, storage) === null, 'A recording request still needs reconciliation. Refresh its recovery record.');
  need(text(message, 1800)); const next = stateValue(project, { ...current, pending });
  try { writeRecordingState(project, next, storage); storage.setItem(attemptKey(project), JSON.stringify({ state: 'pending', message })); return next; }
  catch { throw new RecordingStorageUnavailable(); }
}
export function finishRecordingAttempt(project: string, attemptId: string, job: RecordingJob, select: boolean, storage: Writer = sessionStorage): { state: RecordingState; receipt: RecordingJob } | null {
  const current = readRecordingState(project, storage);
  if (!current.pending || current.pending.attempt_id !== attemptId) return null;
  const receipt = recordingReceipt(job, project, current.pending.recipe, current.pending.action === 'cancel' ? current.pending.job_id! : undefined);
  const next = { ...current, selected_job_id: select ? receipt.id : current.selected_job_id };
  try {
    storeRecordingReceipt(receipt, storage);
    writeRecordingState(project, next, storage);
    storage.removeItem(attemptKey(project));
    const complete = { ...next, pending: null }; writeRecordingState(project, complete, storage);
    return { state: complete, receipt };
  } catch { throw new RecordingStorageUnavailable(); }
}
export function failRecordingAttempt(project: string, attemptId: string, uncertainMessage: string | null, storage: Writer = sessionStorage): RecordingState | null {
  const current = readRecordingState(project, storage);
  if (!current.pending || current.pending.attempt_id !== attemptId) return null;
  try {
    if (uncertainMessage !== null) { need(text(uncertainMessage, 1800)); storage.setItem(attemptKey(project), JSON.stringify({ state: 'uncertain', message: uncertainMessage })); return current; }
    storage.removeItem(attemptKey(project)); const next = { ...current, pending: null }; writeRecordingState(project, next, storage); return next;
  } catch { throw new RecordingStorageUnavailable(); }
}
export function clearRecordingAttempt(project: string, reviewedAttemptId: string | null, storage: Writer = sessionStorage): RecordingState | null {
  const current = readRecordingState(project, storage);
  if ((current.pending?.attempt_id ?? null) !== reviewedAttemptId) return null;
  try { storage.removeItem(attemptKey(project)); const next = { ...current, pending: null }; writeRecordingState(project, next, storage); return next; }
  catch { throw new RecordingStorageUnavailable(); }
}

/** Older recording submissions share intake admission; cancellation records remain independent. */
export function recordingSubmissionLegacy(project: string, storage: Reader): [string | null, string | null] | null {
  const raw = storage.getItem(stateKey(project)), attempt = storage.getItem(attemptKey(project));
  if (raw === null) return attempt === null ? null : [raw, attempt];
  try {
    const state = stateValue(project, boundedJSON(raw));
    if (state.pending?.action === 'cancel') return null;
    if (!state.pending && attempt === null) return null;
    // Draft edits and manual history selection do not change the old request's
    // ownership. Bind only the exact pending identity plus its journal bytes.
    return [state.pending ? JSON.stringify(state.pending) : null, attempt];
  } catch { /* Unknown saved state cannot silently authorize another intake. */ }
  return [raw, attempt];
}
export function clearRecordingSubmissionLegacy(project: string, expected: [string | null, string | null], storage: Writer): void {
  need(sameJson(recordingSubmissionLegacy(project, storage), expected), 'Recording recovery changed. Review its history again.');
  // Never delete malformed draft data or a cancellation to clear submission recovery.
  const state = readRecordingState(project, storage);
  need(state.pending?.action !== 'cancel');
  if (state.pending) writeRecordingState(project, { ...state, pending: null }, storage);
  storage.removeItem(attemptKey(project));
}
