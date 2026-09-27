import { expect, test } from '@playwright/test';
import { policyJobRequest } from '../../apps/web/src/lib/policy-job-mutation';

const originalFetch = globalThis.fetch;
test.afterEach(() => { globalThis.fetch = originalFetch; });

test('keyed transport sends exact key once and requires its echoed identity', async () => {
  let count = 0;
  globalThis.fetch = async (_url, init) => {
    count += 1;
    expect(new Headers(init?.headers).get('Idempotency-Key')).toBe('saved-key');
    return Response.json({ id: 'accepted' }, { headers: { 'Idempotency-Key': 'saved-key', 'Cache-Control': 'no-store' } });
  };
  expect(await policyJobRequest('/projects/p/policy-jobs', { operation: 'policy.finetune' }, { idempotencyKey: 'saved-key' })).toEqual({ id: 'accepted' });
  expect(count).toBe(1);
});

test('old backend without echoed key never becomes a verified keyed response', async () => {
  globalThis.fetch = async () => Response.json({ id: 'accepted' });
  await expect(policyJobRequest('/projects/p/submissions/saved-key?operation=policy.finetune', undefined, { idempotencyKey: 'saved-key' })).rejects.toThrow();
});

import type { Job } from '../../apps/web/src/lib/api';
import { createSubmissionAttempt, initialSubmissionRecovery, submissionController, submissionOf, submissionStorageKey, type SubmissionRecovery, type SubmissionScope } from '../../apps/web/src/lib/durable-submission';
import { trainingReceipt } from '../../apps/web/src/lib/training-submission';

const scope: SubmissionScope = { project: 'alpha', operation: 'policy.finetune' };
const body = () => ({ operation: 'policy.finetune', runtime_id: 'owned-runtime', dataset_job_id: 'dataset-one', training_method: 'full', timeout_seconds: 600, training: { model_id: 'act', steps: 4, camera_keys: ['camera.top'] } });
function job(request: unknown = body(), overrides: Record<string, unknown> = {}) {
  return { id: 'accepted-job', project_id: 'alpha', kind: 'policy.finetune', status: 'queued', created_at: '2026-09-27T12:00:00Z', updated_at: '2026-09-27T12:00:01Z', request, result: { unverified: true }, ...overrides };
}
const validator = (value: unknown, original: Record<string, unknown>) => trainingReceipt(value, 'alpha', original);
function fixture(seed = new Map<string, string>()) {
  let state: SubmissionRecovery = { ...initialSubmissionRecovery };
  const storage = { getItem: (key: string) => seed.get(key) ?? null, setItem: (key: string, value: string) => { seed.set(key, value); }, removeItem: (key: string) => { seed.delete(key); } };
  const cache = { get: () => state, set: (value: SubmissionRecovery) => { state = value; } };
  const controller = submissionController(scope, cache, () => storage);
  controller.hydrate();
  return { seed, storage, cache, controller, state: () => state };
}
function response(value: unknown, key: string, status = 200) { return Response.json(value, { status, headers: { 'Idempotency-Key': key, 'Cache-Control': 'no-store' } }); }
function missing(key: string) { return response({ detail: 'No saved submission' }, key, 404); }
function requestKey(init?: RequestInit) { return new Headers(init?.headers).get('Idempotency-Key')!; }
function deferred<T>() { let resolve!: (value: T) => void; const promise = new Promise<T>(r => { resolve = r; }); return { promise, resolve }; }

for (const operation of ['dataset.inspect', 'dataset.augment', 'policy.finetune'] as const) test(`generic ${operation} persists exact original before any I/O and uses only its fixed endpoint`, async () => {
  const f = fixture(), ownScope = { ...scope, operation };
  const c = submissionController(ownScope, f.cache, () => f.storage);
  const recipe = operation === 'policy.finetune' ? body() : { repo_id: 'owned/dataset', fraction: 0.5, ordered: [2, 1] };
  const seen: string[] = [];
  globalThis.fetch = async (url, init) => {
    const key = requestKey(init), stored = JSON.parse(f.seed.get(submissionStorageKey(ownScope))!);
    expect(stored.submission.key).toBe(key); expect(stored.submission.body).toEqual(recipe);
    seen.push(`${init?.method} ${String(url)}`);
    return init?.method === 'GET' ? missing(key) : response(job(recipe, { kind: operation }), key);
  };
  const accepted = await c.submit(recipe, value => value as Job);
  expect(accepted?.id).toBe('accepted-job'); expect(accepted?.result).toEqual({ unverified: true }); expect(f.state().receipt?.result).toBeNull(); expect(seen).toHaveLength(2);
  expect(seen[0]).toContain(`/submissions/`); expect(seen[0]).toContain(`operation=${encodeURIComponent(operation)}`);
  expect(seen[1]).toMatch(new RegExp(`/${operation === 'dataset.inspect' ? 'intakes' : operation === 'dataset.augment' ? 'augmentations' : 'policy-jobs'}$`));
  expect(f.state().attempt).toBeNull(); expect(f.seed.size).toBe(0);
});

test('lost ACK survives reload; authoritative lookup validates saved recipe and never POSTs', async () => {
  const f = fixture(), recipe = body(); let key = '', posts = 0;
  globalThis.fetch = async (_url, init) => { key = requestKey(init); if (init?.method === 'GET') return missing(key); posts++; throw new Error('Lost response after server acceptance'); };
  expect(await f.controller.submit(recipe, validator)).toBeNull(); expect(posts).toBe(1);
  const saved = submissionOf(f.state().attempt)!; recipe.training.steps = 99;
  expect(saved.body).toEqual(body()); expect(f.state().attempt?.state).toBe('uncertain');
  const reloaded = fixture(f.seed); expect(reloaded.controller.canRetry()).toBe(false);
  globalThis.fetch = async (_url, init) => { expect(init?.method).toBe('GET'); expect(requestKey(init)).toBe(key); return response(job(body(), { status: 'succeeded' }), key); };
  expect((await reloaded.controller.reconcile(validator))?.id).toBe('accepted-job'); expect(reloaded.state().attempt).toBeNull(); expect(posts).toBe(1);
});

test('404 reconciliation retains identity; explicit retry uses identical key/body and no new request', async () => {
  const f = fixture(); let posts = 0; const keys: string[] = [], bodies: unknown[] = [];
  globalThis.fetch = async (_url, init) => { keys.push(requestKey(init)); if (init?.method === 'GET') return missing(requestKey(init)); posts++; bodies.push(JSON.parse(String(init.body))); throw new Error('lost'); };
  await f.controller.submit(body(), validator); const original = submissionOf(f.state().attempt)!;
  const reload = fixture(f.seed); expect(await reload.controller.retry(validator)).toBeNull(); expect(posts).toBe(1);
  expect(await reload.controller.reconcile(validator)).toBeNull(); expect(reload.controller.canRetry()).toBe(true); expect(submissionOf(reload.state().attempt)?.key).toBe(original.key);
  expect(await reload.controller.submit({ ...body(), timeout_seconds: 999 }, validator)).toBeNull(); expect(posts).toBe(1);
  globalThis.fetch = async (_url, init) => { keys.push(requestKey(init)); if (init?.method === 'GET') return missing(requestKey(init)); posts++; bodies.push(JSON.parse(String(init.body))); return response(job(), requestKey(init)); };
  expect((await reload.controller.retry(validator))?.id).toBe('accepted-job'); expect(posts).toBe(2); expect(new Set(keys)).toEqual(new Set([original.key])); expect(bodies).toEqual([body(), body()]);
});

test('retry finds server commitment after earlier404 and does not POST again', async () => {
  const f = fixture(); let posts = 0;
  globalThis.fetch = async (_url, init) => { if (init?.method === 'GET') return missing(requestKey(init)); posts++; throw new Error('lost'); };
  await f.controller.submit(body(), validator);
  globalThis.fetch = async (_url, init) => { expect(init?.method).toBe('GET'); return response(job(), requestKey(init)); };
  expect((await f.controller.retry(validator))?.id).toBe('accepted-job'); expect(posts).toBe(1);
});

for (const change of ['missing key', 'wrong key', 'missing no-store', 'unsupported endpoint', 'invalid JSON']) test(`unverified contract ${change} makes zero POSTs`, async () => {
  const f = fixture(); let reads = 0;
  globalThis.fetch = async (_url, init) => {
    expect(init?.method).toBe('GET'); reads++;
    if (change === 'invalid JSON') return new Response('{', { status: 404, headers: { 'Idempotency-Key': requestKey(init), 'Cache-Control': 'no-store' } });
    const headers = { ...(change === 'missing key' || change === 'unsupported endpoint' ? {} : { 'Idempotency-Key': change === 'wrong key' ? 'foreign' : requestKey(init) }), ...(change === 'missing no-store' ? {} : { 'Cache-Control': 'no-store' }) };
    return Response.json({ detail: 'Not found' }, { status: 404, headers });
  };
  expect(await f.controller.submit(body(), validator)).toBeNull(); expect(reads).toBe(1); expect(f.state().attempt?.state).toBe('uncertain'); expect(f.controller.canRetry()).toBe(false);
});

for (const raw of ['{', JSON.stringify({ state: 'pending', message: 'Original source A' }), JSON.stringify({ state: 'uncertain', message: 'lost' })]) test(`legacy journal is retained without key generation or I/O: ${raw}`, async () => {
  const f = fixture(new Map([['firebird:job-attempt:policy.finetune:alpha', raw]]));
  globalThis.fetch = async () => { throw new Error('No network allowed'); };
  expect(f.state().attempt?.state).toBe('uncertain'); expect(submissionOf(f.state().attempt)).toBeNull();
  expect(await f.controller.submit(body(), validator)).toBeNull(); expect(await f.controller.retry(validator)).toBeNull(); expect(f.seed.size).toBe(1);
  const expected = f.state().attempt; expect(f.controller.clearLegacy(expected)).toBe(true); expect(f.seed.size).toBe(0);
});

test('corrupt keyed identity becomes uncertainty, never trusted lookup or replacement', async () => {
  const attempt = createSubmissionAttempt('alpha', 'policy.finetune', body()); attempt.submission!.project = 'foreign';
  const f = fixture(new Map([[submissionStorageKey(scope), JSON.stringify(attempt)]]));
  expect(submissionOf(f.state().attempt)).toBeNull(); expect(await f.controller.reconcile(validator)).toBeNull(); expect(await f.controller.submit(body(), validator)).toBeNull(); expect(f.seed.size).toBe(1);
});

test('shared cache excludes double clicks and remounted controllers during admission', async () => {
  const f = fixture(), gate = deferred<Response>(); let calls = 0;
  globalThis.fetch = async (_url, init) => { calls++; return init?.method === 'GET' ? gate.promise : response(job(), requestKey(init)); };
  const first = f.controller.submit(body(), validator);
  const remount = submissionController(scope, f.cache, () => f.storage); remount.hydrate();
  expect(await remount.submit(body(), validator)).toBeNull(); expect(await remount.reconcile(validator)).toBeNull(); expect(calls).toBe(1);
  gate.resolve(missing(submissionOf(f.state().attempt)!.key)); expect((await first)?.id).toBe('accepted-job'); expect(calls).toBe(2);
});

for (const outcome of ['ACK', 'network failure']) test(`old mount late ${outcome} cannot erase same-recipe successor`, async () => {
  const f = fixture(), old = deferred<Response>(); let oldKey = '';
  globalThis.fetch = async (_url, init) => { if (init?.method === 'GET') { oldKey = requestKey(init); return missing(oldKey); } return old.promise; };
  const pending = f.controller.submit(body(), validator); await expect.poll(() => f.state().notFoundKey).toBeTruthy();
  const successor = createSubmissionAttempt('alpha', 'policy.finetune', body());
  f.seed.set(submissionStorageKey(scope), JSON.stringify(successor)); f.cache.set({ ...f.state(), attempt: successor, receipt: null, busy: true });
  const snapshot = JSON.stringify(f.state()), saved = [...f.seed];
  old.resolve(outcome === 'ACK' ? response(job(), oldKey) : new Response('failed', { status: 503 }));
  expect(await pending).toBeNull(); expect(JSON.stringify(f.state())).toBe(snapshot); expect([...f.seed]).toEqual(saved);
});

for (const phase of ['initial read', 'admission write']) test(`${phase} storage denial blocks all I/O and latches admission`, async () => {
  const f = fixture(); let calls = 0;
  globalThis.fetch = async () => { calls++; throw new Error(); };
  if (phase === 'initial read') f.storage.getItem = () => { throw new Error('denied'); };
  else f.storage.setItem = () => { throw new Error('quota'); };
  expect(await f.controller.submit(body(), validator)).toBeNull(); expect(calls).toBe(0); expect(f.state().available).toBe(false); expect(f.state().error).toContain('storage');
  expect(await f.controller.submit(body(), validator)).toBeNull(); expect(calls).toBe(0);
});

for (const failure of ['read', 'remove']) test(`valid POST ACK is retained before cleanup ${failure} failure`, async () => {
  const f = fixture();
  globalThis.fetch = async (_url, init) => {
    if (init?.method === 'GET') return missing(requestKey(init));
    if (failure === 'read') f.storage.getItem = () => { throw new Error('denied after ACK'); };
    else f.storage.removeItem = () => { throw new Error('failed cleanup'); };
    return response(job(), requestKey(init));
  };
  expect((await f.controller.submit(body(), validator))?.id).toBe('accepted-job'); expect(f.state().receipt?.id).toBe('accepted-job'); expect(f.state().receipt?.result).toBeNull(); expect(f.state().available).toBe(false); expect(f.state().attempt?.state).toBe('uncertain');
});

test('valid recovered GET ACK is retained before denied storage read', async () => {
  const attempt = createSubmissionAttempt('alpha', 'policy.finetune', body()), f = fixture(new Map([[submissionStorageKey(scope), JSON.stringify(attempt)]]));
  globalThis.fetch = async (_url, init) => { f.storage.getItem = () => { throw new Error('denied after GET ACK'); }; return response(job(), requestKey(init)); };
  expect((await f.controller.reconcile(validator))?.id).toBe('accepted-job'); expect(f.state().receipt?.id).toBe('accepted-job'); expect(f.state().available).toBe(false);
});

for (const [index, bad] of [undefined, NaN, Infinity, { nested: undefined }, new Date(), { value: Array(33).fill(0).reduce(value => ({ value }), {}) }].entries()) test(`non-JSON/unbounded recipe refuses before persistence: ${index}`, () => {
  expect(() => createSubmissionAttempt('alpha', 'policy.finetune', bad)).toThrow();
});
test('saved identity is a bounded deep copy and excludes cross-project/operation journals', () => {
  const recipe = body(), attempt = createSubmissionAttempt('alpha', 'policy.finetune', recipe);
  recipe.training.camera_keys.push('new'); expect(submissionOf(attempt)?.body).toEqual(body());
  expect(() => createSubmissionAttempt('alpha', 'dataset.inspect', body())).toThrow();
  expect(() => createSubmissionAttempt('alpha', 'policy.finetune', { value: 'é'.repeat(524288) })).toThrow('1 MiB');
  expect(submissionStorageKey(scope)).not.toBe(submissionStorageKey({ ...scope, project: 'beta' }));
  expect(submissionStorageKey(scope)).not.toBe(submissionStorageKey({ ...scope, operation: 'dataset.inspect' }));
});

for (const override of [{ id: '' }, { project_id: 'beta' }, { kind: 'policy.quantize' }, { status: ['queued'] }, { updated_at: 'yesterday' }, { request: { ...body(), runtime_id: 'foreign' } }, { request: { ...body(), training: { ...body().training, steps: 5 } } }]) test(`lookup rejects wrong acknowledged identity ${JSON.stringify(override)}`, async () => {
  const attempt = createSubmissionAttempt('alpha', 'policy.finetune', body()), f = fixture(new Map([[submissionStorageKey(scope), JSON.stringify(attempt)]]));
  globalThis.fetch = async (_url, init) => response(job(body(), override), requestKey(init));
  expect(await f.controller.reconcile(validator)).toBeNull(); expect(f.state().receipt).toBeNull(); expect(submissionOf(f.state().attempt)?.key).toBe(attempt.submission!.key);
});

test('training recovery admits saved resume enrichment and new-job checkpoint normalization', () => {
  const resume = { ...body(), resume_job_id: 'original-training', training: null };
  expect(trainingReceipt(job({ ...resume, dataset_job_id: 'saved-dataset', training_method: 'lora', training: { steps: 12 } }), 'alpha', resume).id).toBe('accepted-job');
  const recipe = { ...body(), training: { ...body().training, checkpoint_subdirectory: 'catalog/path' } };
  expect(trainingReceipt(job({ ...recipe, training: { ...recipe.training, checkpoint_subdirectory: 'normalized/path' } }), 'alpha', recipe).id).toBe('accepted-job');
  expect(() => trainingReceipt(job({ ...resume, resume_job_id: 'other' }), 'alpha', resume)).toThrow();
});

test('unkeyed existing transport remains compatible and does not add submission headers', async () => {
  globalThis.fetch = async (_url, init) => { expect(new Headers(init?.headers).has('Idempotency-Key')).toBe(false); return Response.json({ id: 'legacy' }); };
  expect(await policyJobRequest('/jobs/legacy/cancel', {})).toEqual({ id: 'legacy' });
});

for (const status of [400, 422]) test(`initial well-formed${status} rejection permits corrected explicit submission`, async () => {
  const f = fixture(); const keys: string[] = []; let posts = 0;
  globalThis.fetch = async (_url, init) => {
    if (init?.method === 'GET') return missing(requestKey(init));
    posts++; keys.push(requestKey(init));
    return Response.json({ detail: 'Correct the original recipe' }, { status });
  };
  expect(await f.controller.submit(body(), validator)).toBeNull(); expect(f.state().attempt).toBeNull(); expect(f.state().error).toBe('Correct the original recipe'); expect(f.state().available).toBe(true);
  const corrected = { ...body(), timeout_seconds: 900 };
  globalThis.fetch = async (_url, init) => { if (init?.method === 'GET') return missing(requestKey(init)); posts++; keys.push(requestKey(init)); expect(JSON.parse(String(init.body))).toEqual(corrected); return response(job(corrected), requestKey(init)); };
  expect((await f.controller.submit(corrected, validator))?.id).toBe('accepted-job'); expect(posts).toBe(2); expect(keys[0]).not.toBe(keys[1]);
});
for (const status of [422, 409, 500]) test(`lost ACK then explicit retry${status} retains original identity`, async () => {
  const f = fixture(); let posts = 0;
  globalThis.fetch = async (_url, init) => { if (init?.method === 'GET') return missing(requestKey(init)); posts++; throw new Error('Lost ACK'); };
  await f.controller.submit(body(), validator); const original = submissionOf(f.state().attempt)!;
  globalThis.fetch = async (_url, init) => { if (init?.method === 'GET') return missing(requestKey(init)); posts++; return Response.json({ detail: 'Rejected retry' }, { status }); };
  expect(await f.controller.retry(validator)).toBeNull(); expect(submissionOf(f.state().attempt)).toEqual(original); expect(posts).toBe(2); expect(f.state().attempt?.state).toBe('uncertain');
});
for (const detail of [null, [], ['error'], { msg: 'Invalid' }, '']) test(`malformed initial422 detail remains uncertain: ${JSON.stringify(detail)}`, async () => {
  const f = fixture();
  globalThis.fetch = async (_url, init) => init?.method === 'GET' ? missing(requestKey(init)) : Response.json({ detail }, { status: 422 });
  await f.controller.submit(body(), validator); expect(f.state().attempt?.state).toBe('uncertain');
});
test('empty project does not read browser storage or authorize submission', () => {
  const f = fixture(); f.cache.set({ ...initialSubmissionRecovery }); let reads = 0;
  submissionController({ ...scope, project: '' }, f.cache, () => { reads++; throw new Error(); }).hydrate();
  expect(reads).toBe(0); expect(f.state().available).toBe(false);
});

test('malformed echoed initial422 is still uncertainty', async () => {
  const f = fixture();
  globalThis.fetch = async (_url, init) => init?.method === 'GET' ? missing(requestKey(init)) : response({ detail: [] }, requestKey(init), 422);
  await f.controller.submit(body(), validator); expect(f.state().attempt?.state).toBe('uncertain');
});
test('late denied storage cannot latch off a successor after old ACK', async () => {
  const f = fixture(), old = deferred<Response>(); let oldKey = '';
  globalThis.fetch = async (_url, init) => { if (init?.method === 'GET') { oldKey = requestKey(init); return missing(oldKey); } return old.promise; };
  const pending = f.controller.submit(body(), validator); await expect.poll(() => f.state().notFoundKey).toBeTruthy();
  const successor = createSubmissionAttempt('alpha', 'policy.finetune', body()); f.cache.set({ ...f.state(), attempt: successor, receipt: null, busy: true });
  f.storage.getItem = () => { throw new Error('Old completion must not inspect successor storage'); };
  const before = f.state(); old.resolve(response(job(), oldKey)); expect(await pending).toBeNull(); expect(f.state()).toBe(before);
});
test('legacy acknowledgement cannot clear a newer durable submission', () => {
  const f = fixture(new Map([['firebird:job-attempt:policy.finetune:alpha', '{']])); const expected = f.state().attempt;
  const successor = createSubmissionAttempt('alpha', 'policy.finetune', body()); f.cache.set({ ...f.state(), attempt: successor });
  expect(f.controller.clearLegacy(expected)).toBe(false); expect(f.state().attempt).toBe(successor);
});

test('fresh-history acknowledgement cannot remove a different malformed legacy record', () => {
  const key = 'firebird:job-attempt:policy.finetune:alpha', f = fixture(new Map([[key, '{']]));
  const expected = f.state().attempt; f.seed.set(key, '{changed');
  expect(f.controller.clearLegacy(expected)).toBe(false); expect(f.seed.get(key)).toBe('{changed'); expect(f.state().available).toBe(false);
});

import { intakeAcknowledgement } from '../../apps/web/src/lib/dataset-submission';
import { emptyRecordingState, recordingRequest, type RecordingRecipe } from '../../apps/web/src/lib/recording-preparation';
const intakeScope: SubmissionScope = { project: 'alpha', operation: 'dataset.inspect' };
const recordingRecipeFixture: RecordingRecipe = { schema_version: 1, configuration_sha256: 'a'.repeat(64), timeout_seconds: 600, captures: [{ session_id: 'b'.repeat(32), session_sha256: 'c'.repeat(64), episodes: [{ episode_id: 'd'.repeat(32), receipt_sha256: 'e'.repeat(64) }] }] };
const recordingBody = () => recordingRequest(recordingRecipeFixture);
const intakeValidator = (value: unknown, original: Record<string, unknown>) => intakeAcknowledgement(value, 'alpha', original);
const recordingAlias = 'firebird:job-attempt:dataset.inspect.recordings:alpha', recordingDraft = 'firebird:recording-preparation:alpha';
function intakeFixture(seed = new Map<string, string>()) {
  const f = fixture(seed); f.cache.set({ ...initialSubmissionRecovery });
  const controller = submissionController(intakeScope, f.cache, () => f.storage); controller.hydrate();
  return { ...f, controller };
}
function oldRecording(action: 'submit' | 'cancel' = 'submit') {
  return { ...emptyRecordingState('alpha'), recipe: recordingRecipeFixture, selected_job_id: 'manual-selection', pending: { attempt_id: 'f'.repeat(32), action, recipe: recordingRecipeFixture, job_id: action === 'cancel' ? 'existing-job' : null } };
}
const plainIntake = { source: 'local', path: '/recorded/dataset', snapshot_for_training: true, repo_id: null, revision: 'main' };

for (const body of [recordingBody(), plainIntake]) test(`intake recovery shares full exact job across surfaces: ${body.path}`, async () => {
  const attempt = createSubmissionAttempt('alpha', 'dataset.inspect', body), f = intakeFixture(new Map([[submissionStorageKey(intakeScope), JSON.stringify(attempt)]]));
  const accepted = job(body, { kind: 'dataset.inspect', status: 'succeeded', result: { preserved: 'full profile' } });
  globalThis.fetch = async (_url, init) => { expect(init?.method).toBe('GET'); return response(accepted, requestKey(init)); };
  expect(await f.controller.reconcile(intakeValidator)).toEqual(accepted);
  expect(f.state().attempt).toBeNull();
});
for (const change of ['foreign project', 'changed recipe', 'host path', 'extra original field']) test(`recording intake shared ACK refuses ${change}`, () => {
  const original: Record<string, unknown> = recordingBody(), accepted = job(original, { kind: 'dataset.inspect' });
  if (change === 'foreign project') accepted.project_id = 'beta';
  if (change === 'changed recipe') accepted.request = { ...original, recordings: { ...recordingRecipeFixture, timeout_seconds: 601 } };
  if (change === 'host path') accepted.request = { ...original, path: '/foreign' };
  if (change === 'extra original field') original.extra = true;
  expect(() => intakeValidator(accepted, original)).toThrow();
});
for (const variant of ['submit marker', 'orphan attempt', 'malformed attempt', 'marker only', 'malformed draft']) test(`generic intake cannot bypass legacy recording ${variant}`, async () => {
  const seed = new Map<string, string>();
  if (variant !== 'orphan attempt' && variant !== 'malformed attempt') seed.set(recordingDraft, variant === 'malformed draft' ? '{' : JSON.stringify(oldRecording()));
  if (variant !== 'marker only' && variant !== 'malformed draft') seed.set(recordingAlias, variant === 'malformed attempt' ? '{' : JSON.stringify({ state: 'pending', message: 'Original recording request' }));
  const f = intakeFixture(seed), before = [...seed]; let calls = 0;
  globalThis.fetch = async () => { calls++; throw new Error(); };
  expect(f.state().attempt?.state).toBe('uncertain'); expect(submissionOf(f.state().attempt)).toBeNull();
  expect(await f.controller.submit(plainIntake, intakeValidator)).toBeNull(); expect(await f.controller.retry(intakeValidator)).toBeNull();
  expect(calls).toBe(0); expect([...seed]).toEqual(before);
});
test('explicit legacy recording cleanup preserves recipe, selected job, receipt and other project', () => {
  const state = oldRecording(), seed = new Map([[recordingDraft, JSON.stringify(state)], [recordingAlias, '{'], ['firebird:recording-receipt:alpha', 'unchanged-receipt'], ['firebird:job-attempt:dataset.inspect.recordings:beta', 'other-project']]);
  const f = intakeFixture(seed); expect(f.controller.clearLegacy(f.state().attempt, [])).toBe(true);
  expect(JSON.parse(seed.get(recordingDraft)!)).toEqual({ ...state, pending: null });
  expect(seed.get('firebird:recording-receipt:alpha')).toBe('unchanged-receipt'); expect(seed.get('firebird:job-attempt:dataset.inspect.recordings:beta')).toBe('other-project'); expect(seed.has(recordingAlias)).toBe(false);
});
test('old cancellation state and journal are excluded from intake migration and never deleted', () => {
  const seed = new Map([[recordingDraft, JSON.stringify(oldRecording('cancel'))], [recordingAlias, '{'], ['firebird:job-attempt:dataset.inspect:alpha', '{']]), f = intakeFixture(seed);
  const draft = seed.get(recordingDraft), alias = seed.get(recordingAlias);
  expect(f.controller.clearLegacy(f.state().attempt, [])).toBe(true); expect(seed.get(recordingDraft)).toBe(draft); expect(seed.get(recordingAlias)).toBe(alias);
  const reloaded = intakeFixture(seed); expect(reloaded.state().attempt).toBeNull();
});
for (const replacement of ['changed submit', 'cancel']) test(`legacy review cannot clear a newer recording ${replacement}`, () => {
  const seed = new Map([[recordingDraft, JSON.stringify(oldRecording())], [recordingAlias, '{']]), f = intakeFixture(seed);
  seed.set(recordingDraft, JSON.stringify(replacement === 'cancel' ? oldRecording('cancel') : { ...oldRecording(), pending: { ...oldRecording().pending, attempt_id: 'a'.repeat(32) } })); const before = [...seed];
  expect(f.controller.clearLegacy(f.state().attempt, [])).toBe(false); expect([...seed]).toEqual(before);
});
test('malformed recording draft remains preserved after explicit history acknowledgement', () => {
  const seed = new Map([[recordingDraft, '{']]), f = intakeFixture(seed);
  expect(f.controller.clearLegacy(f.state().attempt, [])).toBe(false); expect(seed.get(recordingDraft)).toBe('{'); expect(f.state().available).toBe(false);
});
test('newly appearing legacy recording submit blocks a keyed POST after support lookup', async () => {
  const f = intakeFixture(); let calls = 0;
  globalThis.fetch = async (_url, init) => { calls++; expect(init?.method).toBe('GET'); f.seed.set(recordingDraft, JSON.stringify(oldRecording())); return missing(requestKey(init)); };
  expect(await f.controller.submit(plainIntake, intakeValidator)).toBeNull(); expect(calls).toBe(1); expect(f.state().available).toBe(false);
});
test('preflight sees persisted immutable body after support lookup; ownership is rechecked before POST', async () => {
  const f = fixture(), gate = deferred<void>(), events: string[] = [];
  globalThis.fetch = async (_url, init) => { events.push(init!.method!); expect(f.seed.has(submissionStorageKey(scope))).toBe(true); return missing(requestKey(init)); };
  const pending = f.controller.submit(body(), validator, async original => { events.push('preflight'); expect(original).toEqual(body()); await gate.promise; });
  await expect.poll(() => events).toEqual(['GET', 'preflight']);
  const successor = createSubmissionAttempt('alpha', 'policy.finetune', body()); f.seed.set(submissionStorageKey(scope), JSON.stringify(successor)); f.cache.set({ ...f.state(), attempt: successor, busy: true });
  const before = f.state(); gate.resolve(); expect(await pending).toBeNull(); expect(f.state()).toBe(before); expect(events).toEqual(['GET', 'preflight']);
});
test('never-POSTed preflight refusal allows corrected request, uncertain retry refusal retains original key', async () => {
  const f = fixture(); let posts = 0;
  globalThis.fetch = async (_url, init) => { if (init?.method === 'GET') return missing(requestKey(init)); posts++; throw new Error('lost ACK'); };
  expect(await f.controller.submit(body(), validator, async () => { throw new Error('Selection changed'); })).toBeNull();
  expect(posts).toBe(0); expect(f.state().attempt).toBeNull(); expect(f.state().available).toBe(true);
  await f.controller.submit(body(), validator); const original = submissionOf(f.state().attempt);
  await f.controller.retry(validator, async () => { throw new Error('Old captures no longer available'); });
  expect(posts).toBe(1); expect(submissionOf(f.state().attempt)).toEqual(original);
});
test('generic surface cannot skip recording catalog validation on retry; found lookup needs no catalog', async () => {
  const body = recordingBody(), attempt = createSubmissionAttempt('alpha', 'dataset.inspect', body), f = intakeFixture(new Map([[submissionStorageKey(intakeScope), JSON.stringify(attempt)]]));
  const reads: string[] = []; let found = false;
  globalThis.fetch = async (url, init) => {
    expect(init?.method).toBe('GET'); reads.push(String(url));
    if (String(url).includes('/submissions/')) return found ? response(job(body, { kind: 'dataset.inspect' }), requestKey(init)) : missing(requestKey(init));
    return Response.json({ detail: 'Catalog unavailable' }, { status: 503 });
  };
  await f.controller.reconcile(intakeValidator); expect(f.controller.canRetry()).toBe(true);
  await f.controller.retry(intakeValidator); expect(reads.at(-1)).toContain('/recordings/options'); expect(submissionOf(f.state().attempt)).toEqual(attempt.submission);
  found = true; const count = reads.length; expect((await f.controller.reconcile(intakeValidator))?.id).toBe('accepted-job'); expect(reads.length).toBe(count + 1);
});

for (const value of [undefined, true, [{ ...job(), project_id: 'foreign' }], [job(plainIntake, { kind: 'dataset.inspect', status: 'running' })], [job(recordingBody(), { kind: 'dataset.inspect', status: 'queued' })]]) test(`shared legacy intake clearing requires verified inactive project history: ${JSON.stringify(value)}`, () => {
  const seed = new Map([[recordingDraft, JSON.stringify(oldRecording())]]), f = intakeFixture(seed), before = [...seed];
  expect(f.controller.clearLegacy(f.state().attempt, value)).toBe(false); expect([...seed]).toEqual(before); expect(f.state().attempt).not.toBeNull();
});

test('legacy recording review retains later manual history selection while clearing only the same pending identity', () => {
  const seed = new Map([[recordingDraft, JSON.stringify(oldRecording())]]), f = intakeFixture(seed);
  const current = { ...oldRecording(), selected_job_id: 'reviewed-job' }; seed.set(recordingDraft, JSON.stringify(current));
  expect(f.controller.clearLegacy(f.state().attempt, [])).toBe(true);
  expect(JSON.parse(seed.get(recordingDraft)!)).toEqual({ ...current, pending: null });
});
