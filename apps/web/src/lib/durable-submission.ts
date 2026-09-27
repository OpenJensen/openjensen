import { recordingSubmissionLegacy, clearRecordingSubmissionLegacy, checkRecordingSubmission, recordingRecipe } from './recording-preparation';
import { reviewedIntakeHistory } from './dataset-submission';
import { useEffect } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { apiOrigin, type Job } from './api';
import type { PolicyJobAttempt, SubmissionIdentity, SubmissionOperation } from './policy-job-attempt';
import { PolicyJobHttpError, policyJobRequest, record, sameJson } from './policy-job-mutation';

export type SubmissionScope = { project: string; operation: SubmissionOperation };
export type SubmissionValidator = (value: unknown, originalBody: Record<string, unknown>) => Job;
export type SubmissionPreflight = (originalBody: Record<string, unknown>) => Promise<void>;
type Store = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>;
export type SubmissionRecovery = {
  hydrated: boolean; available: boolean; attempt: PolicyJobAttempt; receipt: Job | null;
  error: string; busy: boolean; notFoundKey: string | null; legacyRecords?: { primary: [string | null, string | null]; recording: [string | null, string | null] | null };
};
export const initialSubmissionRecovery: SubmissionRecovery = { hydrated: false, available: false, attempt: null, receipt: null, error: '', busy: false, notFoundKey: null };
export type SubmissionCache = { get(): SubmissionRecovery; set(value: SubmissionRecovery): void };
const paths: Record<SubmissionOperation, string> = { 'dataset.inspect': 'intakes', 'dataset.augment': 'augmentations', 'policy.finetune': 'policy-jobs', 'teaching.capture': 'teaching/sessions' };
const identifier = (value: unknown): value is string => typeof value === 'string' && value.length > 0 && value.length <= 256 && !/[\x00-\x1f\x7f]/.test(value);
const keyPattern = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const maxBody = 1024 * 1024, maxJournal = maxBody + 8192;
const interrupted = 'An earlier request did not return a verified outcome. Check the saved request or recorded jobs; no automatic retry was made.';

export class SubmissionStorageUnavailable extends Error {
  constructor() { super('Submission recovery storage is unavailable. Restore browser storage and reload before submitting again. Any known accepted job is retained in this session.'); }
}
export class SubmissionJournalChanged extends Error {
  constructor() { super('Submission recovery changed or is unreadable. Inspect the recorded jobs; do not replace the unresolved request.'); }
}
function store(): Store { try { return sessionStorage; } catch { throw new SubmissionStorageUnavailable(); } }
function legacyKey(scope: SubmissionScope): string { return `firebird:job-attempt:${scope.operation}:${scope.project}`; }
export function submissionStorageKey(scope: SubmissionScope): string {
  const endpoint = new URL(apiOrigin || '/', 'http://same-origin.invalid');
  return `firebird:submission:v1:${encodeURIComponent(endpoint.origin + endpoint.pathname)}:${scope.operation}:${encodeURIComponent(scope.project)}`;
}
function readRaw(storage: Store, key: string): string | null {
  try { return storage.getItem(key); } catch { throw new SubmissionStorageUnavailable(); }
}
function writeRaw(storage: Store, key: string, value: string | null) {
  try { if (value === null) storage.removeItem(key); else storage.setItem(key, value); }
  catch { throw new SubmissionStorageUnavailable(); }
}
function boundedBody(value: unknown): Record<string, unknown> {
  let nodes = 0;
  const visit = (item: unknown, depth: number): void => {
    if (++nodes > 50000 || depth > 32) throw new Error('Submission recipe is too complex.');
    if (Array.isArray(item)) { for (const child of item) visit(child, depth + 1); return; }
    if (record(item) && (Object.getPrototypeOf(item) === Object.prototype || Object.getPrototypeOf(item) === null)) { for (const child of Object.values(item)) visit(child, depth + 1); return; }
    if (item === null || typeof item === 'string' || typeof item === 'boolean' || (typeof item === 'number' && Number.isFinite(item))) return;
    throw new Error('Submission recipe must contain finite JSON values only.');
  };
  if (!record(value)) throw new Error('Submission recipe must be a JSON object.');
  visit(value, 0);
  const text = JSON.stringify(value);
  if (new TextEncoder().encode(text).length > maxBody) throw new Error('Submission recipe exceeds 1 MiB.');
  return JSON.parse(text) as Record<string, unknown>;
}
function identity(value: unknown, scope: SubmissionScope): SubmissionIdentity {
  if (!record(value) || Object.keys(value).sort().join(',') !== 'body,key,operation,project,version' || value.version !== 1 || value.project !== scope.project || value.operation !== scope.operation || typeof value.key !== 'string' || !keyPattern.test(value.key)) throw new SubmissionJournalChanged();
  const body = boundedBody(value.body);
  if (body.operation !== undefined && body.operation !== scope.operation) throw new SubmissionJournalChanged();
  return { version: 1, key: value.key, project: scope.project, operation: scope.operation, body };
}
export function createSubmissionAttempt(project: string, operation: SubmissionOperation, body: unknown): NonNullable<PolicyJobAttempt> {
  if (!identifier(project) || !Object.hasOwn(paths, operation)) throw new Error('Invalid submission scope.');
  const submission = identity({ version: 1, key: crypto.randomUUID(), project, operation, body }, { project, operation });
  return { state: 'pending', message: 'Submitting one saved request…', submission };
}
export function submissionOf(attempt: PolicyJobAttempt): SubmissionIdentity | null { return attempt?.submission ?? null; }
function sameIdentity(left: PolicyJobAttempt, right: PolicyJobAttempt): boolean {
  return !!left?.submission && !!right?.submission && sameJson(left.submission, right.submission);
}
function parseAttempt(raw: string, scope: SubmissionScope): NonNullable<PolicyJobAttempt> {
  if (raw.length > maxJournal) throw new SubmissionJournalChanged();
  const value: unknown = JSON.parse(raw);
  if (!record(value) || Object.keys(value).sort().join(',') !== 'message,state,submission' || (value.state !== 'pending' && value.state !== 'uncertain') || typeof value.message !== 'string' || value.message.length > 2000) throw new SubmissionJournalChanged();
  return { state: value.state, message: value.message, submission: identity(value.submission, scope) };
}
function legacyRecords(scope: SubmissionScope, storage: Store): NonNullable<SubmissionRecovery['legacyRecords']> {
  try { return { primary: [readRaw(storage, submissionStorageKey(scope)), readRaw(storage, legacyKey(scope))], recording: scope.operation === 'dataset.inspect' ? recordingSubmissionLegacy(scope.project, storage) : null }; }
  catch { throw new SubmissionStorageUnavailable(); }
}
function readAttempt(scope: SubmissionScope, storage: Store): PolicyJobAttempt {
  const raw = readRaw(storage, submissionStorageKey(scope));
  if (raw !== null) {
    try { return parseAttempt(raw, scope); }
    catch { return { state: 'uncertain', message: 'Saved submission identity is unreadable. Inspect recorded jobs before explicitly acknowledging this legacy or damaged recovery record.' }; }
  }
  const records = legacyRecords(scope, storage);
  const legacy = records.primary[1] ?? records.recording?.[1] ?? null;
  if (legacy === null && records.recording === null) return null;
  let message = interrupted;
  try {
    if (legacy !== null && legacy.length <= 16384) {
      const value: unknown = JSON.parse(legacy);
      if (record(value) && typeof value.message === 'string' && value.message.length <= 2000) message = value.message;
    }
  } catch { /* Malformed legacy state is uncertainty, never an empty journal. */ }
  return { state: 'uncertain', message };
}
function minimalReceipt(job: Job, scope: SubmissionScope): Job {
  if (!identifier(job.id) || job.project_id !== scope.project || job.kind !== scope.operation || typeof job.status !== 'string' || !['queued', 'running', 'succeeded', 'failed', 'cancelled', 'interrupted'].includes(job.status) || !record(job.request) || ![job.created_at, job.updated_at].every(value => typeof value === 'string' && Number.isFinite(Date.parse(value)))) throw new Error('Application returned a different or invalid job identity.');
  return { id: job.id, project_id: job.project_id, kind: job.kind, status: job.status, request: job.request, created_at: job.created_at, updated_at: job.updated_at, result: null, error: null, stage: null };
}

/** Pure controller plus a shared cache: old mounts cannot clear a successor's key. */
export function submissionController(scope: SubmissionScope, cache: SubmissionCache, getStorage: () => Store = store) {
  const current = () => cache.get();
  const put = (patch: Partial<SubmissionRecovery>) => cache.set({ ...current(), ...patch });
  const owns = (expected: NonNullable<PolicyJobAttempt>) => sameIdentity(current().attempt, expected);
  function storageFailure(error: unknown, expected?: NonNullable<PolicyJobAttempt>) {
    if (expected && !owns(expected)) return;
    put({ available: false, hydrated: true, busy: false, notFoundKey: null, attempt: current().attempt ? { ...current().attempt!, state: 'uncertain' } : null, error: error instanceof Error ? error.message : new SubmissionStorageUnavailable().message });
  }
  function hydrate() {
    if (current().hydrated || !identifier(scope.project) || !Object.hasOwn(paths, scope.operation)) return;
    try {
      const saved = readAttempt(scope, getStorage());
      const records = saved && !saved.submission ? legacyRecords(scope, getStorage()) : undefined;
      put({ hydrated: true, available: true, attempt: saved ? { ...saved, state: 'uncertain' } : null, legacyRecords: records });
    } catch (error) { storageFailure(error); }
  }
  function storedOwner(expected: NonNullable<PolicyJobAttempt>): boolean {
    if (!owns(expected)) return false;
    if (!sameIdentity(readAttempt(scope, getStorage()), expected)) throw new SubmissionJournalChanged();
    return true;
  }
  function fail(error: unknown, expected: NonNullable<PolicyJobAttempt>) {
    if (!owns(expected)) return;
    if (error instanceof SubmissionStorageUnavailable || error instanceof SubmissionJournalChanged) { storageFailure(error, expected); return; }
    const message = error instanceof Error ? error.message.slice(0, 2000) : interrupted;
    const next: NonNullable<PolicyJobAttempt> = { ...expected, state: 'uncertain', message };
    try {
      if (!storedOwner(expected)) return;
      writeRaw(getStorage(), submissionStorageKey(scope), JSON.stringify(next));
      put({ attempt: next, error: message, busy: false });
    } catch (fault) { storageFailure(fault, expected); }
  }
  function accept(value: unknown, expected: NonNullable<PolicyJobAttempt>, validate: SubmissionValidator): Job | null {
    const original = expected.submission!;
    const accepted = validate(value, boundedBody(original.body));
    const receipt = minimalReceipt(accepted, scope);
    if (!owns(expected)) return null;
    // The cache fence cannot throw on denied storage. Retain the verified ACK first.
    put({ receipt });
    try {
      if (!storedOwner(expected)) return null;
      writeRaw(getStorage(), submissionStorageKey(scope), null);
      put({ attempt: null, error: '', busy: false, notFoundKey: null });
    } catch (fault) { storageFailure(fault, expected); }
    return accepted;
  }
  async function lookup(expected: NonNullable<PolicyJobAttempt>): Promise<{ found: true; value: unknown } | { found: false }> {
    const submission = expected.submission!;
    try {
      const value = await policyJobRequest(`/projects/${encodeURIComponent(scope.project)}/submissions/${encodeURIComponent(submission.key)}?operation=${encodeURIComponent(scope.operation)}`, undefined, { idempotencyKey: submission.key });
      return { found: true, value };
    } catch (error) {
      if (error instanceof PolicyJobHttpError && error.status === 404) return { found: false };
      throw error;
    }
  }
  async function execute(expected: NonNullable<PolicyJobAttempt>, validate: SubmissionValidator, postIfMissing: boolean, initial = false, beforePost?: SubmissionPreflight): Promise<Job | null> {
    try {
      if (!storedOwner(expected)) return null;
      const result = await lookup(expected);
      if (result.found) return accept(result.value, expected, validate);
      if (!storedOwner(expected)) return null;
      put({ notFoundKey: expected.submission!.key });
      if (!postIfMissing) throw new Error('No saved binding is visible yet. The earlier request may still commit. Check again or explicitly retry the same saved request; a new request remains blocked.');
      const savedBody = expected.submission!.body;
      if (beforePost || (scope.operation === 'dataset.inspect' && savedBody.recordings != null)) {
        try {
          if (scope.operation === 'dataset.inspect' && savedBody.recordings != null) await checkRecordingSubmission(scope.project, recordingRecipe(savedBody.recordings), () => owns(expected));
          if (beforePost) await beforePost(boundedBody(savedBody));
        }
        catch (error) {
          if (!initial) throw error; // A prior POST may still commit on explicit retry.
          if (!storedOwner(expected)) return null;
          // This newly admitted identity has never been POSTed. A rejected local
          // preflight may permit an edited request, unlike an uncertain retry.
          writeRaw(getStorage(), submissionStorageKey(scope), null);
          put({ attempt: null, error: error instanceof Error ? error.message : 'Submission preflight failed.', busy: false, notFoundKey: null });
          return null;
        }
        if (!storedOwner(expected)) return null;
      }
      // Older recording pages must not bypass this shared admission while a
      // durable request is awaiting its catalog read or support response.
      const older = legacyRecords(scope, getStorage());
      if (older.primary[1] !== null || older.recording !== null) throw new SubmissionJournalChanged();
      // Every POST is initiated by submit()/retry(), after this exact endpoint
      // echoed the saved key on 404. No fetch/query library mutation retry exists.
      let value: unknown;
      try {
        value = await policyJobRequest(`/projects/${encodeURIComponent(scope.project)}/${paths[scope.operation]}`, expected.submission!.body, { idempotencyKey: expected.submission!.key, allowInitialRejection: initial });
      } catch (error) {
        if (initial && error instanceof PolicyJobHttpError && error.initialRejection && [400, 422].includes(error.status)) {
          if (!storedOwner(expected)) return null;
          writeRaw(getStorage(), submissionStorageKey(scope), null);
          put({ attempt: null, error: error.message, busy: false, notFoundKey: null });
          return null;
        }
        throw error;
      }
      return accept(value, expected, validate);
    } catch (error) { fail(error, expected); return null; }
    finally { if (owns(expected)) put({ busy: false }); }
  }
  async function submit(body: unknown, validate: SubmissionValidator, beforePost?: SubmissionPreflight): Promise<Job | null> {
    if (!current().hydrated || !current().available || current().attempt || current().busy) return null;
    let attempt: NonNullable<PolicyJobAttempt> | undefined;
    try {
      attempt = createSubmissionAttempt(scope.project, scope.operation, body);
      if (readAttempt(scope, getStorage()) !== null) throw new SubmissionJournalChanged();
      writeRaw(getStorage(), submissionStorageKey(scope), JSON.stringify(attempt));
      put({ attempt, receipt: null, error: '', busy: true, notFoundKey: null });
    } catch (error) {
      if (error instanceof SubmissionStorageUnavailable || error instanceof SubmissionJournalChanged) storageFailure(error);
      else put({ error: error instanceof Error ? error.message : 'Invalid submission recipe.' });
      return null;
    }
    return execute(attempt, validate, true, true, beforePost);
  }
  async function reconcile(validate: SubmissionValidator): Promise<Job | null> {
    const attempt = current().attempt;
    if (!current().available || current().busy || !submissionOf(attempt)) return null;
    put({ busy: true, error: '', notFoundKey: null });
    return execute(attempt!, validate, false);
  }
  const canRetry = () => current().available && !current().busy && !!current().attempt?.submission && current().notFoundKey === current().attempt?.submission?.key;
  async function retry(validate: SubmissionValidator, beforePost?: SubmissionPreflight): Promise<Job | null> {
    if (!canRetry()) return null;
    const expected = current().attempt!;
    put({ busy: true, error: '', notFoundKey: null });
    return execute(expected, validate, true, false, beforePost);
  }
  function clearLegacy(expected: PolicyJobAttempt, history?: unknown): boolean {
    if (!expected || expected.submission || !current().available || current().busy || current().attempt !== expected) return false;
    if (scope.operation === 'dataset.inspect') {
      try { reviewedIntakeHistory(history, scope.project); }
      catch (error) { put({ error: error instanceof Error ? error.message : 'Project history could not be verified.' }); return false; }
    }
    try {
      // Only legacy/damaged records can be acknowledged following the caller's
      // explicit fresh history review. A valid saved key is never discarded here.
      const storage = getStorage(), saved = readAttempt(scope, storage);
      const records = legacyRecords(scope, storage);
      if (saved?.submission || !sameJson(saved, expected) || !sameJson(records, current().legacyRecords)) throw new SubmissionJournalChanged();
      if (records.recording) clearRecordingSubmissionLegacy(scope.project, records.recording, storage);
      writeRaw(storage, submissionStorageKey(scope), null);
      writeRaw(storage, legacyKey(scope), null);
      put({ attempt: null, error: '', notFoundKey: null, legacyRecords: undefined });
      return true;
    } catch (error) { storageFailure(error); return false; }
  }
  return { hydrate, submit, reconcile, retry, canRetry, clearLegacy };
}

export function useDurableSubmission(scope: SubmissionScope) {
  const client = useQueryClient(), key = ['durable-submission', submissionStorageKey(scope)];
  const query = useQuery<SubmissionRecovery>({ queryKey: key, queryFn: async () => initialSubmissionRecovery, initialData: initialSubmissionRecovery, enabled: false, gcTime: Infinity });
  const controller = submissionController(scope, { get: () => client.getQueryData<SubmissionRecovery>(key) ?? initialSubmissionRecovery, set: value => { client.setQueryData(key, value); } });
  useEffect(() => { controller.hydrate(); }, [client, scope.project, scope.operation]);
  const run = async (operation: () => Promise<Job | null>) => {
    const receipt = await operation();
    if (receipt) client.setQueryData<Job[]>(['jobs', scope.project], previous => [receipt, ...(previous ?? []).filter(job => job.id !== receipt.id)]);
    void client.invalidateQueries({ queryKey: ['jobs', scope.project] }).catch(() => {});
    return receipt;
  };
  return { ...query.data, canRetry: controller.canRetry(), submit: (body: unknown, validate: SubmissionValidator, beforePost?: SubmissionPreflight) => run(() => controller.submit(body, validate, beforePost)), reconcile: (validate: SubmissionValidator) => run(() => controller.reconcile(validate)), retry: (validate: SubmissionValidator, beforePost?: SubmissionPreflight) => run(() => controller.retry(validate, beforePost)), clearLegacy: controller.clearLegacy };
}
