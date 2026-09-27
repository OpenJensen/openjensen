import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api, apiOrigin, type Job } from './api';
import { record } from './policy-job-mutation';

export type CancellationWorkflow = 'distillation' | 'replay';
export type CancellationScope = { project: string; workflow: CancellationWorkflow };
export type CancellationAttempt = CancellationScope & {
  version: 1; action: 'cancel'; id: string; jobId: string; requestSha256: string;
  state: 'pending' | 'uncertain';
};
export type CancellationReceipt = { attemptId: string; jobId: string; status: Job['status'] };
type Recovery = { available: boolean; hydrated: boolean; attempt: CancellationAttempt | null; receipt: CancellationReceipt | null; error: string };
const initial: Recovery = { available: false, hydrated: false, attempt: null, receipt: null, error: '' };
const statuses = ['queued', 'running', 'succeeded', 'failed', 'cancelled', 'interrupted'];
const identifier = (value: unknown): value is string => typeof value === 'string' && value.length > 0 && value.length <= 256 && !/[\x00-\x1f\x7f]/.test(value);

export class CancellationStorageError extends Error {
  constructor() { super('Cancellation recovery storage is unavailable. Restore browser storage and reload, then inspect the original job. No automatic retry was made.'); }
}
export class CancellationJournalError extends Error {
  constructor() { super('Cancellation recovery is unreadable or changed. Inspect recorded jobs and restore this tab’s recovery storage before cancelling again.'); }
}

// sessionStorage already separates browser origins. Strip URL credentials/query;
// the API origin and mount path further separate installations on the same origin.
export function cancellationKey(scope: CancellationScope): string {
  const endpoint = new URL(apiOrigin || '/', 'http://same-origin.invalid');
  return `firebird:cancel-attempt:v1:${encodeURIComponent(endpoint.origin + endpoint.pathname)}:${scope.workflow}:${encodeURIComponent(scope.project)}`;
}
function storage(): Storage { try { return sessionStorage; } catch { throw new CancellationStorageError(); } }
export function readCancellation(scope: CancellationScope, store: Pick<Storage, 'getItem'> = storage()): CancellationAttempt | null {
  let raw: string | null;
  try { raw = store.getItem(cancellationKey(scope)); } catch { throw new CancellationStorageError(); }
  if (raw === null) return null;
  try {
    if (raw.length > 4096) throw new Error();
    const value: unknown = JSON.parse(raw);
    if (!record(value) || value.version !== 1 || value.action !== 'cancel' || value.project !== scope.project || value.workflow !== scope.workflow || !identifier(value.id) || !identifier(value.jobId) || typeof value.requestSha256 !== 'string' || !/^[a-f0-9]{64}$/.test(value.requestSha256) || (value.state !== 'pending' && value.state !== 'uncertain')) throw new Error();
    return { version: 1, action: 'cancel', project: scope.project, workflow: scope.workflow, id: value.id, jobId: value.jobId, requestSha256: value.requestSha256, state: value.state };
  } catch { throw new CancellationJournalError(); }
}
export function sameCancellation(left: CancellationAttempt | null | undefined, right: CancellationAttempt): boolean {
  return !!left && left.id === right.id && left.state === right.state && left.project === right.project && left.workflow === right.workflow && left.jobId === right.jobId && left.requestSha256 === right.requestSha256;
}
export function transitionCancellation(scope: CancellationScope, expected: CancellationAttempt | null, next: CancellationAttempt | null, store: Pick<Storage, 'getItem' | 'setItem' | 'removeItem'> = storage()): boolean {
  const current = readCancellation(scope, store);
  if (expected ? !sameCancellation(current, expected) : current !== null) return false;
  if (next && (next.project !== scope.project || next.workflow !== scope.workflow)) throw new CancellationJournalError();
  try { if (next) store.setItem(cancellationKey(scope), JSON.stringify(next)); else store.removeItem(cancellationKey(scope)); }
  catch { throw new CancellationStorageError(); }
  return true;
}
export async function cancellationRequestSha(request: unknown): Promise<string> {
  const normalize = (value: unknown, depth = 0): unknown => {
    if (depth > 20) throw new CancellationJournalError();
    if (Array.isArray(value)) return value.map(item => normalize(item, depth + 1));
    if (record(value)) return Object.fromEntries(Object.keys(value).sort().map(key => [key, normalize(value[key], depth + 1)]));
    if (value === null || typeof value === 'string' || typeof value === 'boolean' || (typeof value === 'number' && Number.isFinite(value))) return value;
    throw new CancellationJournalError();
  };
  const text = JSON.stringify(normalize(request));
  if (text.length > 32768) throw new CancellationJournalError();
  const bytes = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
}

/** Cancellation-only recovery; submission journals and server history are separate. */
export function useNativeCancellation(scope: CancellationScope) {
  const client = useQueryClient(), key = ['native-cancellation', cancellationKey(scope)];
  const query = useQuery<Recovery>({ queryKey: key, queryFn: async () => initial, initialData: initial, enabled: false, gcTime: Infinity });
  const mounted = useRef(true), reviewing = useRef(false);
  const [review, setReview] = useState<{ attempt: CancellationAttempt; generation: number; status: Job['status'] } | null>(null), [refreshing, setRefreshing] = useState(false);
  const cached = () => client.getQueryData<Recovery>(key) ?? initial;
  const update = (value: Recovery) => client.setQueryData<Recovery>(key, value);
  const invalidate = () => { void client.invalidateQueries({ queryKey: ['jobs', scope.project] }).catch(() => {}); };
  function unavailable(cause: unknown, expected?: CancellationAttempt) {
    const current = cached();
    if (expected && current.attempt?.id !== expected.id) return;
    update({ ...current, hydrated: true, available: false, attempt: current.attempt ? { ...current.attempt, state: 'uncertain' } : null, error: cause instanceof Error ? cause.message : new CancellationStorageError().message });
    if (mounted.current) setReview(null);
  }
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  useEffect(() => {
    try {
      let saved = readCancellation(scope); const current = cached();
      if (current.hydrated && !current.available) return; // A storage fault stays latched until reload.
      if (current.attempt && !sameCancellation(saved, current.attempt)) throw new CancellationJournalError();
      if (saved?.state === 'pending' && !sameCancellation(current.attempt, saved)) {
        const uncertain: CancellationAttempt = { ...saved, state: 'uncertain' };
        if (!transitionCancellation(scope, saved, uncertain)) throw new CancellationJournalError();
        saved = uncertain;
      }
      update({ ...current, hydrated: true, available: true, attempt: saved, error: '' });
    } catch (cause) { unavailable(cause); }
  }, [client, scope.project, scope.workflow]);
  function owns(expected: CancellationAttempt): boolean {
    return sameCancellation(cached().attempt, expected) && sameCancellation(readCancellation(scope), expected);
  }
  async function run(job: Job, stillSelected: () => boolean, send: (stillSelected: () => boolean) => Promise<Job>): Promise<Job | undefined> {
    let attempt: CancellationAttempt | undefined;
    try {
      if (!cached().available || cached().attempt || !stillSelected()) return;
      const requestSha256 = await cancellationRequestSha(job.request);
      if (!cached().available || cached().attempt || !stillSelected()) return;
      if (!identifier(job.id) || job.project_id !== scope.project) throw new CancellationJournalError();
      const candidate: CancellationAttempt = { version: 1, action: 'cancel', ...scope, id: crypto.randomUUID(), jobId: job.id, requestSha256, state: 'pending' };
      if (!transitionCancellation(scope, null, candidate)) throw new CancellationJournalError();
      attempt = candidate;
      update({ ...cached(), attempt, receipt: null, error: '' });
      if (mounted.current) setReview(null);
      const expected = attempt;
      const response = await send(() => stillSelected() && owns(expected));
      if (!sameCancellation(cached().attempt, expected)) { invalidate(); return; }
      // Cache ownership cannot throw on a storage read. The existing workflow helper validated this ACK. Retain only identity/status
      // before fallible cleanup. Its result/measurements/media are never cached here.
      const receipt: CancellationReceipt = { attemptId: expected.id, jobId: job.id, status: response.status };
      update({ ...cached(), receipt });
      try {
        if (!transitionCancellation(scope, expected, null)) throw new CancellationJournalError();
        update({ ...cached(), attempt: null, error: '' });
      } catch (cause) { unavailable(cause, expected); }
      invalidate();
      return { id: job.id, project_id: job.project_id, kind: job.kind, request: job.request, status: response.status, created_at: job.created_at, updated_at: job.updated_at, result: null };
    } catch (cause) {
      if (attempt) {
        try {
          if (!owns(attempt)) { invalidate(); return; }
          const uncertain: CancellationAttempt = { ...attempt, state: 'uncertain' };
          if (transitionCancellation(scope, attempt, uncertain)) update({ ...cached(), attempt: uncertain, error: cause instanceof Error ? cause.message.slice(0, 2000) : 'Cancellation outcome is unverified.' });
        } catch (fault) { unavailable(fault, attempt); }
      } else unavailable(cause);
      if (mounted.current) setReview(null);
    }
  }
  async function refresh(generation: number, stillCurrent: () => boolean) {
    const attempt = cached().attempt;
    if (!cached().available || !attempt || attempt.state !== 'uncertain' || reviewing.current) return;
    reviewing.current = true; if (mounted.current) { setReview(null); setRefreshing(true); }
    try {
      if (!owns(attempt)) return;
      // Direct read: never acknowledge using an initial/polling request that began
      // before uncertainty and was deduplicated by the shared query observer.
      const list: unknown = await api.jobs(scope.project);
      const original = Array.isArray(list) ? list.find(value => record(value) && value.id === attempt.jobId && value.project_id === scope.project) : undefined;
      if (!record(original) || !record(original.request) || original.kind !== (scope.workflow === 'replay' ? 'policy.run' : 'policy.distill') || typeof original.status !== 'string' || !statuses.includes(original.status) || await cancellationRequestSha(original.request) !== attempt.requestSha256) throw new Error('Fresh history did not verify the original cancellation target. Keep inspecting the original job; cancellation remains blocked.');
      if (mounted.current && stillCurrent() && owns(attempt)) { update({ ...cached(), error: '' }); setReview({ attempt, generation, status: original.status as Job['status'] }); }
    } catch (cause) {
      if (cause instanceof CancellationStorageError || cause instanceof CancellationJournalError) unavailable(cause, attempt);
      else if (mounted.current && sameCancellation(cached().attempt, attempt)) update({ ...cached(), error: cause instanceof Error ? cause.message : 'Cancellation history could not be refreshed.' });
    } finally { reviewing.current = false; if (mounted.current) setRefreshing(false); }
  }
  function acknowledge(generation: number) {
    if (!review || review.generation !== generation || !cached().available) return;
    try {
      if (!owns(review.attempt)) return;
      if (transitionCancellation(scope, review.attempt, null)) update({ ...cached(), attempt: null, error: '' });
      setReview(null);
    } catch (cause) { unavailable(cause, review.attempt); }
  }
  const currentReview = review && sameCancellation(query.data.attempt, review.attempt) ? review : null;
  return { ...query.data, refreshing, reviewedTarget: currentReview ? { jobId: currentReview.attempt.jobId, status: currentReview.status } : null, canAcknowledge: !!currentReview, run, refresh, acknowledge, selectionChanged: () => setReview(null) };
}
