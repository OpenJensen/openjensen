import { test, expect } from '@playwright/test';
import { recordingRecipe, recordingCatalog, recordingOptions, recordingRequest, recordingReceipt, recordingResult, recordingSelectionHash, selectionMatches, readRecordingState, writeRecordingState, readRecordingAttempt, RECORDING_OPERATION, submitRecordings, cancelRecording, readRecordingHistory, storeRecordingReceipt, readRecordingReceipt, RecordingStorageUnavailable, beginRecordingAttempt, finishRecordingAttempt, failRecordingAttempt, clearRecordingAttempt, ownsRecordingAttempt } from '../../apps/web/src/lib/recording-preparation';

import { UncertainPolicyJob } from '../../apps/web/src/lib/policy-job-mutation';

const hex = (c: string, n = 64) => c.repeat(n);
const stamp = '2026-09-27T00:00:00Z';
function recipe() { return { schema_version: 1, configuration_sha256: hex('a'), timeout_seconds: 600, captures: [{ session_id: hex('b', 32), session_sha256: hex('c'), episodes: [{ episode_id: hex('d', 32), receipt_sha256: hex('e') }] }] }; }
function catalog() { return { configuration_sha256: hex('a'), message: 'Finalized metadata only', captures: [{ session_id: hex('b', 32), session_sha256: hex('c'), origin: 'synthetic', lineage_group: 'fixture-one', controller: 'joint_position_targets', state_units: 'radians', action_units: 'radians', timebase: 'simulation_seconds', camera_key: 'observation.images.front', joint_names: ['joint_one'], width: 32, height: 32, fps: 10, physics_hz: 60, scene_sha256: hex('f'), camera_prim: '/World/Camera', content_verified: false, episodes: [{ episode_id: hex('d', 32), receipt_sha256: hex('e'), frames: 12, outcome: 'unknown', termination: 'finish' }] }] }; }
function job() { return { id: 'job-one', project_id: 'alpha', kind: 'dataset.inspect', status: 'queued', request: recordingRequest(recordingRecipe(recipe())), created_at: stamp, updated_at: stamp, result: { arbitrary: true }, compute_target: null, simulation_target: null }; }

for (const bad of [true, '1', 2, null]) test(`reject schema ${JSON.stringify(bad)}`, () => expect(() => recordingRecipe({ ...recipe(), schema_version: bad })).toThrow());
for (const bad of [true, 59, 1801, 60.5, '600']) test(`reject timeout ${JSON.stringify(bad)}`, () => expect(() => recordingRecipe({ ...recipe(), timeout_seconds: bad })).toThrow());
test('strict selection never adds client paths or overrides source semantics', () => {
  const r = recordingRecipe(recipe()); expect(recordingRequest(r)).toEqual({ source: 'local', repo_id: null, revision: 'main', path: null, snapshot_for_training: true, recordings: recipe() });
  expect(() => recordingRecipe({ ...recipe(), path: '/private/capture' })).toThrow();
  expect(() => recordingRecipe({ ...recipe(), captures: [recipe().captures[0], recipe().captures[0]] })).toThrow();
  const c = recipe().captures[0]; expect(() => recordingRecipe({ ...recipe(), captures: [{ ...c, episodes: [c.episodes[0], c.episodes[0]] }] })).toThrow();
});
test('catalog is metadata only and stale hashes never select replacement bytes', () => {
  const c = recordingCatalog(catalog()), r = recordingRecipe(recipe()); expect(selectionMatches(r, c)).toBe(true);
  expect(selectionMatches(r, { ...c, configuration_sha256: hex('f') })).toBe(false);
  expect(selectionMatches(r, { ...c, captures: [{ ...c.captures[0], session_sha256: hex('f') }] })).toBe(false);
  expect(selectionMatches(r, { ...c, captures: [] })).toBe(false);
  expect(() => recordingCatalog({ ...catalog(), captures: [{ ...catalog().captures[0], content_verified: true }] })).toThrow();
  expect(() => recordingCatalog({ ...catalog(), captures: [{ ...catalog().captures[0], state_units: 'normalized' }] })).toThrow();
});
test('configured options do not promote runtime verified', () => {
  const o = { configured: true, runtime_verified: false, configuration_sha256: hex('a'), max_episodes: 100, max_source_bytes: 8 * 1024 ** 3, setup_message: 'Configuration present only' };
  expect(recordingOptions(o)).toEqual(o); expect(() => recordingOptions({ ...o, runtime_verified: true })).toThrow(); expect(() => recordingOptions({ ...o, configured: true, configuration_sha256: null })).toThrow();
});
for (const bad of [['queued'], true, {}, 'unknown', null]) test(`reject coerced ACK status ${JSON.stringify(bad)}`, () => expect(() => recordingReceipt({ ...job(), status: bad }, 'alpha')).toThrow());
test('ACK is exact admission only and removes unsolicited data', () => {
  const r = recordingReceipt(job(), 'alpha', recordingRecipe(recipe())); expect(r.result).toBeNull(); expect(r.error).toBeNull(); expect(r.stage).toBeNull();
  expect(() => recordingReceipt(job(), 'beta')).toThrow();
  expect(() => recordingReceipt({ ...job(), created_at: 'yesterday' }, 'alpha')).toThrow();
  expect(() => recordingReceipt({ ...job(), request: { ...job().request, path: '/private' } }, 'alpha')).toThrow();
  expect(() => recordingReceipt({ ...job(), request: { ...job().request, recordings: { ...recipe(), timeout_seconds: 601 } } }, 'alpha', recordingRecipe(recipe()))).toThrow();
});
test('canonical selection identity matches Python sorted compact JSON', async () => {
  expect(await recordingSelectionHash(recordingRecipe(recipe()))).toBe('f330388c8f5187dad8dcb4552cb79c2f2daf5310effeab11e0b3843b2cc76cf4');
});
test('historical result requires exact job/selection/snapshot identity and claims', async () => {
  const j = recordingReceipt({ ...job(), status: 'succeeded' }, 'alpha');
  const profile = { schema_version: 1, source: 'local', repo_id: null, revision: `metadata-sha256:${hex('f')}`, format: 'lerobot_v3', total_frames: 12, total_episodes: 1, fps: 10, features: { 'observation.state': {} }, metadata_sha256: hex('f'), inspected_at: stamp, warnings: [], inspection_scope: 'complete_snapshot', snapshot: { schema_version: 1, id: `sha256:${hex('c')}`, manifest_sha256: hex('c'), format: 'lerobot_v3', total_bytes: 1000, file_count: 5, total_frames: 12, total_episodes: 1, lineage_validated: true, warnings: [] }, recording_preparation: { job_id: j.id, selection_sha256: await recordingSelectionHash(j.request.recordings), source_count: 1, lineage_group_count: 1, writer_readback_verified: true, source_preserved: true, task_success_verified: false } };
  expect(await recordingResult(profile, j)).toEqual(profile);
  await expect(recordingResult({ ...profile, total_frames: 13 }, j)).rejects.toThrow();
  await expect(recordingResult({ ...profile, recording_preparation: { ...profile.recording_preparation, task_success_verified: true } }, j)).rejects.toThrow();
  await expect(recordingResult({ ...profile, recording_preparation: { ...profile.recording_preparation, selection_sha256: hex('f') } }, j)).rejects.toThrow();
});
test('session draft remains exact/project bound and corrupt state fails closed', () => {
  const memory = new Map<string, string>(); const storage = { getItem: (k: string) => memory.get(k) ?? null, setItem: (k: string, v: string) => { memory.set(k, v); }, removeItem: (k: string) => { memory.delete(k); } };
  const state = { schema_version: 1 as const, project_id: 'alpha', recipe: recordingRecipe(recipe()), selected_job_id: 'job-one', pending: null };
  writeRecordingState('alpha', state, storage); expect(readRecordingState('alpha', storage)).toEqual(state); expect(readRecordingState('beta', storage).recipe).toBeNull();
  const key = [...memory.keys()][0]; memory.set(key, '{'); expect(() => readRecordingState('alpha', storage)).toThrow(); memory.set(key, ' '.repeat(65537)); expect(() => readRecordingState('alpha', storage)).toThrow();
});
test('reload preserves pending original context as uncertainty and never retries', () => {
  const key = `firebird:job-attempt:${RECORDING_OPERATION}:alpha`;
  const storage = { getItem: (k: string) => k === key ? JSON.stringify({ state: 'pending', message: 'Original project alpha, capture b, request digest c' }) : null };
  const got = readRecordingAttempt('alpha', storage); expect(got?.state).toBe('uncertain'); expect(got?.message).toContain('Original project alpha');
});

const originalFetch = globalThis.fetch;
test.afterEach(() => { globalThis.fetch = originalFetch; });
function options() { return { configured: true, runtime_verified: false, configuration_sha256: hex('a'), max_episodes: 100, max_source_bytes: 8 * 1024 ** 3, setup_message: 'Configured, not runtime verified' }; }
function fetchFixture(overrides: { catalog?: unknown; response?: unknown; status?: number; lost?: boolean; history?: unknown } = {}) {
  const requests: { path: string; method: string; body: unknown }[] = [];
  globalThis.fetch = async (url, init) => {
    const path = new URL(String(url), 'http://127.0.0.1').pathname, method = init?.method ?? 'GET';
    const body: unknown = typeof init?.body === 'string' ? JSON.parse(init.body) : null; requests.push({ path, method, body });
    if (method === 'POST') {
      if (overrides.lost) throw new TypeError('Generated lost response');
      return Response.json(overrides.response ?? job(), { status: overrides.status ?? 202 });
    }
    if (path.endsWith('/recordings/options')) return Response.json(options());
    if (path.endsWith('/recordings')) return Response.json(overrides.catalog ?? catalog());
    if (path.endsWith('/jobs')) return Response.json(overrides.history ?? [job()]);
    if (path.endsWith('/jobs/job-one')) return Response.json({ ...job(), status: 'running' });
    throw new Error(`Unexpected fixed route ${path}`);
  };
  return requests;
}
test('fresh preflight submits exact identity once and discards returned results', async () => {
  const requests = fetchFixture(); const result = await submitRecordings('alpha', recordingRecipe(recipe()), () => true);
  expect(requests.map(r => r.method)).toEqual(['GET', 'GET', 'POST']); expect(requests[2].body).toEqual(job().request); expect(result.result).toBeNull();
});
test('changed capture receipt before submission makes zero POSTs', async () => {
  const c = catalog(); c.captures[0].episodes[0].receipt_sha256 = hex('f'); const requests = fetchFixture({ catalog: c });
  await expect(submitRecordings('alpha', recordingRecipe(recipe()), () => true)).rejects.toThrow('changed'); expect(requests.every(r => r.method === 'GET')).toBe(true);
});
test('manual change during fresh context read makes zero POSTs', async () => {
  const requests = fetchFixture(); await expect(submitRecordings('alpha', recordingRecipe(recipe()), () => false)).rejects.toThrow('Selection changed'); expect(requests).toHaveLength(2);
});
test('lost POST acknowledgement stays uncertain with no automatic retry', async () => {
  const requests = fetchFixture({ lost: true }); await expect(submitRecordings('alpha', recordingRecipe(recipe()), () => true)).rejects.toBeInstanceOf(UncertainPolicyJob); expect(requests.filter(r => r.method === 'POST')).toHaveLength(1);
});
test('definitive rejection is distinguishable from unknown submission', async () => {
  fetchFixture({ response: { detail: 'Generated stale receipt' }, status: 409 });
  try { await submitRecordings('alpha', recordingRecipe(recipe()), () => true); throw new Error('Expected rejection'); }
  catch (e) { expect(e).toBeInstanceOf(Error); expect(e).not.toBeInstanceOf(UncertainPolicyJob); expect((e as Error).message).toContain('Generated stale receipt'); }
});
test('cancellation freshly binds original job and rejects selection changes before POST', async () => {
  const requests = fetchFixture(); await expect(cancelRecording(recordingReceipt({ ...job(), status: 'running' }, 'alpha'), () => false)).rejects.toThrow('Selection changed'); expect(requests.map(r => r.method)).toEqual(['GET']);
});
test('cancellation acknowledgement must keep exact original request', async () => {
  const wrong = { ...job(), request: { ...job().request, recordings: { ...recipe(), timeout_seconds: 601 } } }; const requests = fetchFixture({ response: wrong });
  await expect(cancelRecording(recordingReceipt({ ...job(), status: 'running' }, 'alpha'), () => true)).rejects.toBeInstanceOf(UncertainPolicyJob); expect(requests.map(r => r.method)).toEqual(['GET', 'POST']);
});
test('history rejects a foreign project instead of authorizing review of an empty list', async () => {
  fetchFixture({ history: [{ ...job(), project_id: 'beta' }] }); await expect(readRecordingHistory('alpha')).rejects.toThrow('foreign');
});
test('history never promotes stored result fields into dataset readiness', async () => {
  fetchFixture(); expect((await readRecordingHistory('alpha'))[0].result).toBeNull();
});
test('catalog rejects malformed origins, metadata arrays, duplicate episodes and oversized bounds', () => {
  for (const change of [{ origin: ['recorded'] }, { width: true }, { width: 1922 }, { joint_names: [] }, { scene_sha256: [] }, { fps: Infinity }]) expect(() => recordingCatalog({ ...catalog(), captures: [{ ...catalog().captures[0], ...change }] })).toThrow();
  const c = catalog(); c.captures[0].episodes.push(c.captures[0].episodes[0]); expect(() => recordingCatalog(c)).toThrow();
});
test('aggregate selection bound applies across separate sessions', () => {
  const captures = Array.from({ length: 100 }, (_, i) => ({ session_id: i.toString(16).padStart(32, '0'), session_sha256: hex('c'), episodes: [{ episode_id: (i + 100).toString(16).padStart(32, '0'), receipt_sha256: hex('e') }] }));
  expect(recordingRecipe({ ...recipe(), captures }).captures).toHaveLength(100);
  captures[0].episodes.push({ episode_id: (300).toString(16).padStart(32, '0'), receipt_sha256: hex('e') }); expect(() => recordingRecipe({ ...recipe(), captures })).toThrow('100');
});
test('receipt storage keeps only normalized admission and latches write failures', () => {
  const data = new Map<string, string>(); const storage = { getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => { data.set(k, v); }, removeItem: (k: string) => { data.delete(k); } };
  storeRecordingReceipt(recordingReceipt(job(), 'alpha'), storage); const restored = readRecordingReceipt('alpha', storage)!; expect(restored.result).toBeNull(); expect(restored.request).toEqual(job().request); expect(readRecordingReceipt('beta', storage)).toBeNull();
  expect(() => storeRecordingReceipt(restored, { ...storage, setItem: () => { throw new Error('Quota'); } })).toThrow(RecordingStorageUnavailable);
  const key = [...data.keys()][0]; data.set(key, JSON.stringify({ ...job(), project_id: 'beta' })); expect(() => readRecordingReceipt('alpha', storage)).toThrow(RecordingStorageUnavailable);
});
test('pending cancellation preserves its own recipe separately from the edited draft', () => {
  const data = new Map<string, string>(); const storage = { getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => { data.set(k, v); }, removeItem: (k: string) => { data.delete(k); } };
  const original = recordingRecipe(recipe()), edited = { ...original, timeout_seconds: 900 };
  const value = { schema_version: 1 as const, project_id: 'alpha', recipe: edited, selected_job_id: 'job-two', pending: { attempt_id: hex('9', 32), action: 'cancel' as const, recipe: original, job_id: 'job-one' } };
  writeRecordingState('alpha', value, storage); expect(readRecordingState('alpha', storage).pending?.recipe.timeout_seconds).toBe(600); expect(readRecordingState('alpha', storage).recipe?.timeout_seconds).toBe(900);
});

function journalFixture() {
  const values = new Map<string, string>(), writes: string[] = [];
  const storage = { getItem: (k: string) => values.get(k) ?? null, setItem: (k: string, v: string) => { writes.push(`set:${k}`); values.set(k, v); }, removeItem: (k: string) => { writes.push(`remove:${k}`); values.delete(k); } };
  writeRecordingState('alpha', { schema_version: 1, project_id: 'alpha', recipe: recordingRecipe(recipe()), selected_job_id: 'manual-job', pending: null }, storage);
  const a = { attempt_id: hex('1', 32), action: 'submit' as const, recipe: recordingRecipe(recipe()), job_id: null };
  const b = { ...a, attempt_id: hex('2', 32) };
  return { storage, values, writes, a, b };
}
for (const outcome of ['ack', 'uncertain error', 'definitive error']) test(`remounted same-recipe successor is never erased by old ${outcome}`, () => {
  const f = journalFixture(); beginRecordingAttempt('alpha', f.a, 'Original A', f.storage);
  expect(clearRecordingAttempt('alpha', f.a.attempt_id, f.storage)).not.toBeNull(); beginRecordingAttempt('alpha', f.b, 'Remounted B, same recipe', f.storage);
  const before = [...f.values], writeCount = f.writes.length;
  const result = outcome === 'ack' ? finishRecordingAttempt('alpha', f.a.attempt_id, recordingReceipt(job(), 'alpha'), true, f.storage) : failRecordingAttempt('alpha', f.a.attempt_id, outcome === 'uncertain error' ? 'Late old error' : null, f.storage);
  expect(result).toBeNull(); expect([...f.values]).toEqual(before); expect(f.writes).toHaveLength(writeCount); expect(ownsRecordingAttempt('alpha', f.b.attempt_id, f.storage)).toBe(true);
});
test('stale ACK cannot invoke failing storage writes or replace a retained receipt', () => {
  const f = journalFixture(); beginRecordingAttempt('alpha', f.a, 'A', f.storage); clearRecordingAttempt('alpha', f.a.attempt_id, f.storage); beginRecordingAttempt('alpha', f.b, 'B', f.storage);
  const writesFail = { ...f.storage, setItem: () => { throw new Error('Do not write'); }, removeItem: () => { throw new Error('Do not clear'); } };
  expect(finishRecordingAttempt('alpha', f.a.attempt_id, recordingReceipt(job(), 'alpha'), true, writesFail)).toBeNull(); expect(failRecordingAttempt('alpha', f.a.attempt_id, 'old failure', writesFail)).toBeNull();
});
test('stale history-review token cannot clear a remounted new attempt', () => {
  const f = journalFixture(); beginRecordingAttempt('alpha', f.a, 'A', f.storage); clearRecordingAttempt('alpha', f.a.attempt_id, f.storage); beginRecordingAttempt('alpha', f.b, 'B', f.storage);
  const before = [...f.values]; expect(clearRecordingAttempt('alpha', f.a.attempt_id, f.storage)).toBeNull(); expect([...f.values]).toEqual(before);
});
test('current ACK saves admission before cleanup and preserves a newer manual history choice', () => {
  const f = journalFixture(); beginRecordingAttempt('alpha', f.a, 'A', f.storage); const before = f.writes.length;
  const result = finishRecordingAttempt('alpha', f.a.attempt_id, recordingReceipt(job(), 'alpha'), false, f.storage)!;
  expect(result.state.selected_job_id).toBe('manual-job'); expect(result.state.pending).toBeNull(); expect(readRecordingReceipt('alpha', f.storage)?.id).toBe('job-one');
  const operations = f.writes.slice(before); expect(operations[0]).toContain('recording-receipt'); expect(operations.findIndex(v => v.startsWith('remove:'))).toBeGreaterThan(0);
});
test('receipt survives failed journal cleanup and original attempt remains reconcilable', () => {
  const f = journalFixture(); beginRecordingAttempt('alpha', f.a, 'A', f.storage);
  expect(() => finishRecordingAttempt('alpha', f.a.attempt_id, recordingReceipt(job(), 'alpha'), true, { ...f.storage, removeItem: () => { throw new Error('Storage cleanup failed'); } })).toThrow(RecordingStorageUnavailable);
  expect(readRecordingReceipt('alpha', f.storage)?.id).toBe('job-one'); expect(readRecordingState('alpha', f.storage).pending?.attempt_id).toBe(f.a.attempt_id); expect(readRecordingAttempt('alpha', f.storage)?.state).toBe('uncertain');
});
test('new begin cannot overwrite an unreconciled attempt even for identical inputs', () => {
  const f = journalFixture(); beginRecordingAttempt('alpha', f.a, 'A', f.storage); const before = [...f.values];
  expect(() => beginRecordingAttempt('alpha', f.b, 'B', f.storage)).toThrow('reconciliation'); expect([...f.values]).toEqual(before);
});
test('partial begin failure preserves unique pending ownership before any POST', () => {
  const f = journalFixture(); const original = f.storage.setItem;
  expect(() => beginRecordingAttempt('alpha', f.a, 'A', { ...f.storage, setItem: (k, v) => { if (k.includes('job-attempt')) throw new Error('Generated quota failure'); original(k, v); } })).toThrow(RecordingStorageUnavailable);
  expect(ownsRecordingAttempt('alpha', f.a.attempt_id, f.storage)).toBe(true);
  expect(() => beginRecordingAttempt('alpha', f.b, 'B', f.storage)).toThrow();
});

test('first pending-state write failure leaves no owner or journal and reports storage admission failure', () => {
  const f = journalFixture(), before = [...f.values];
  expect(() => beginRecordingAttempt('alpha', f.a, 'A', { ...f.storage, setItem: () => { throw new Error('Generated first write failure'); } })).toThrow(RecordingStorageUnavailable);
  expect([...f.values]).toEqual(before); expect(ownsRecordingAttempt('alpha', f.a.attempt_id, f.storage)).toBe(false); expect(readRecordingAttempt('alpha', f.storage)).toBeNull();
});
test('first receipt write failure retains original pending ownership and never clears its journal', () => {
  const f = journalFixture(); beginRecordingAttempt('alpha', f.a, 'A', f.storage); const before = [...f.values];
  const receipt = recordingReceipt(job(), 'alpha', f.a.recipe);
  expect(() => finishRecordingAttempt('alpha', f.a.attempt_id, receipt, true, { ...f.storage, setItem: () => { throw new Error('Generated receipt write failure'); } })).toThrow(RecordingStorageUnavailable);
  expect([...f.values]).toEqual(before); expect(readRecordingReceipt('alpha', f.storage)).toBeNull(); expect(ownsRecordingAttempt('alpha', f.a.attempt_id, f.storage)).toBe(true); expect(readRecordingAttempt('alpha', f.storage)?.state).toBe('uncertain');
  // This already validated admission remains available to the mounted caller.
  expect(receipt.id).toBe('job-one'); expect(receipt.result).toBeNull();
});

test('journal read denial is a dedicated storage failure before pending ownership or a request', () => {
  const f = journalFixture(), before = [...f.values];
  const denied = { ...f.storage, getItem: (key: string) => { if (key.includes('job-attempt')) throw new Error('Generated journal-only read denial'); return f.storage.getItem(key); } };
  expect(() => readRecordingAttempt('alpha', denied)).toThrow(RecordingStorageUnavailable);
  expect(() => beginRecordingAttempt('alpha', f.a, 'A', denied)).toThrow(RecordingStorageUnavailable);
  expect([...f.values]).toEqual(before); expect(ownsRecordingAttempt('alpha', f.a.attempt_id, f.storage)).toBe(false);
});
