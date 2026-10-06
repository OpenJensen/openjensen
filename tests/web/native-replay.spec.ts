import { selectProject } from './project-controls';
import { startReplay, cancelReplay } from '../../apps/web/src/lib/native-replay';
import { expect, test, type Page } from '@playwright/test';

const time = '2026-09-27T10:30:00Z', model = `sha256:${'f'.repeat(64)}`;
const runtime = { id: 'replay-cpu', label: 'Local observation replay', execution: 'native', provider: 'local', device: 'cpu', enabled: true, native_replay: true, native_replay_only: true, training: false, simulation: false, run: false, engine_evaluation: false };
const policy = { id: 'packed:policy', project_id: 'alpha', job_id: 'packed', label: 'Packed ACT fixture', format: 'native_quantized', path: 'owned', manifest_sha256: 'a'.repeat(64), file_bytes: 10000, metadata: { architecture: 'act' } };
const dataset = { id: 'data', project_id: 'alpha', kind: 'dataset.inspect', status: 'succeeded', created_at: time, updated_at: time, request: { source: 'local', path: 'fixture' }, result: { source: 'local', format: 'lerobot_v3', repo_id: null, revision: `metadata-sha256:${'c'.repeat(64)}`, metadata_sha256: 'c'.repeat(64), fps: 30, robot_type: 'generated-so101', license: null, inspected_at: time, warnings: [], total_episodes: 6, total_frames: 24, inspection_scope: 'complete_snapshot', features: {}, snapshot: { id: `sha256:${'b'.repeat(64)}`, manifest_sha256: 'b'.repeat(64), lineage_validated: true, total_episodes: 6 } } };
const panel = (page: Page) => page.getByRole('region', { name: 'CPU observation replay', exact: true });
const submit = (page: Page) => page.getByRole('button', { name: 'Run CPU observation replay', exact: true });
async function fixture(page: Page) {
  const state = { jobs: [dataset] as Record<string, any>[], posts: [] as Record<string, any>[], cancelled: [] as string[], outcome: 'ok', postGate: null as Promise<void> | null, invalid: false, prediction: 100, workers: [runtime] as Record<string, unknown>[] };
  await page.route('**/api/v1/**', async route => {
    const req = route.request(), path = new URL(req.url()).pathname;
    if (path === '/api/v1/projects') return route.fulfill({ json: [{ id: 'alpha', name: 'CPU replay fixture', created_at: time }] });
    if (path === '/api/v1/policy-options') return route.fulfill({ json: { runtimes: state.workers, sources: [], training_models: [], training_methods: [], quantization_defaults: { cpu: { language: 'Q8_0' }, cuda: { language: 'Q8_0' }, note: '' } } });
    if (path.endsWith('/artifacts')) return route.fulfill({ json: [policy, { ...policy, id: 'remote', metadata: { architecture: 'act', storage: 'gcs' } }, { ...policy, id: 'wrong', project_id: 'beta' }, { ...policy, id: 'float', format: 'native_checkpoint' }] });
    if (path.endsWith('/policy-jobs') && req.method() === 'POST') {
      const body = req.postDataJSON(); state.posts.push(body);
      if (state.postGate) await state.postGate;
      const job = { id: 'replay-001', project_id: 'alpha', kind: 'policy.run', status: 'running', request: body, created_at: time, updated_at: time, result: null };
      state.jobs.push(job);
      if (state.outcome === 'lost') return route.abort('failed');
      if (state.outcome === 'wrong') return route.fulfill({ status: 202, json: { ...job, project_id: 'beta' } });
      return route.fulfill({ status: 202, json: job });
    }
    if (path.endsWith('/cancel') && req.method() === 'POST') { const id = path.split('/').at(-2)!; state.cancelled.push(id); const job = state.jobs.find(item => item.id === id)!; job.status = 'cancelled'; return route.fulfill({ json: job }); }
    if (path.endsWith('/jobs') && path.includes('/projects/')) return route.fulfill({ json: state.jobs });
    if (path.endsWith('/events')) return route.fulfill({ json: [] });
    if (path.endsWith('/replay')) {
      const job = state.jobs.find(item => item.id === 'replay-001')!;
      return route.fulfill({ json: { artifact_id: 'replay-001:operation', job_id: job.id, model_id: state.invalid ? 'wrong' : model, source_kind: 'generated_fixture', coordinate_names: ['j0', 'j1', 'j2', 'j3', 'j4', 'gripper'], units: job.request.native_replay.units, records: job.request.native_replay.selection.map((row: object) => ({ ...row, reset_repeat_exact: true, actions: Array.from({ length: state.prediction }, (_, step) => Array.from({ length: 6 }, (_, joint) => joint + step / 100)) })) } });
    }
    if (path.startsWith('/api/v1/jobs/')) { const job = state.jobs.find(item => item.id === path.split('/').at(-1)); if (job) return route.fulfill({ json: job }); }
    if (req.method() !== 'GET') return route.fulfill({ status: 405, json: {} });
    return route.continue();
  });
  await page.goto('/'); await expect(page.getByLabel('Current project')).toHaveAttribute('data-project-id', 'alpha');
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  await page.getByRole('button', { name: 'Replay observations', exact: true }).click();
  await expect(page.getByRole('region', { name: 'CPU observation replay', exact: true })).toBeVisible();
  return state;
}
async function prepare(page: Page) {
  await page.getByRole('radio', { name: policy.label, exact: true }).check();
  await page.getByRole('radio', { name: 'Local robotics dataset', exact: true }).check();
  await page.getByLabel('Episode and frame pairs', { exact: true }).fill('0:3, 2:1');
  await page.getByLabel('Six replay coordinate units', { exact: true }).fill('degrees, degrees, degrees, degrees, degrees, recorded_gripper');
  await page.getByRole('radio', { name: 'Generated test observations', exact: true }).check();
  await page.getByRole('checkbox', { name: /I verified that these/ }).check();
}
function complete(state: Awaited<ReturnType<typeof fixture>>) {
  const job = state.jobs.find(item => item.id === 'replay-001')!;
  job.status = 'succeeded'; job.stage = 'replaying'; job.result = { artifacts: [{ ...policy, id: 'replay-001:operation', job_id: job.id, format: 'native_run_record', metadata: { recipe: 'native-observation-replay-v1', model_id: model } }], reports: [{ operation: 'policy.run', stage: 'native_replay', mode: 'independent_observation_replay', device: 'cpu', source_artifact_id: policy.id, dataset_job_id: 'data', observation_source: { kind: 'generated_fixture' }, model_id: model, observations: 2, action_shape: [state.prediction, 6], reset_repeat_exact: true, server_closed: true, task_success: null, quality_verified: false, calibration_verified: false, speedup_verified: false, isaac_runtime_verified: false, elapsed_seconds: 2 }] };
  return job;
}
test('only local packed policies and explicit bounded observations can be replayed', async ({ page }) => {
  const state = await fixture(page); await expect(submit(page)).toBeDisabled();
  await expect(page.getByRole('radiogroup', { name: 'Packed ACT policy', exact: true }).getByRole('radio')).toHaveCount(1);
  await prepare(page); await expect(submit(page)).toBeEnabled();
  for (const invalid of ['0:3, 0:3', '6:1', '0:24', 'foo', '0:-1']) { await page.getByLabel('Episode and frame pairs', { exact: true }).fill(invalid); await expect(submit(page)).toBeDisabled(); }
  expect(state.posts).toHaveLength(0);
});
test('replay submits once with full identity and cancellation is explicit', async ({ page }) => {
  const state = await fixture(page); await prepare(page); await submit(page).dblclick();
  await expect(page.getByRole('article', { name: 'Observation replay details' })).toHaveAttribute('data-job-id', 'replay-001');
  expect(state.posts).toHaveLength(1); expect(state.posts[0].native_replay.selection).toEqual([{ episode_index: 0, frame_index: 3 }, { episode_index: 2, frame_index: 1 }]);
  expect(state.posts[0].native_replay.coordinate_attestation).toBe('generated_fixture'); expect(state.posts[0]).not.toHaveProperty('simulation');
  await page.getByRole('button', { name: 'Cancel selected replay', exact: true }).click(); expect(state.cancelled).toHaveLength(0);
  await page.getByRole('button', { name: 'Confirm cancellation', exact: true }).click();
  await expect(page.getByRole('article', { name: 'Observation replay details' })).toContainText('cancelled'); expect(state.cancelled).toEqual(['replay-001']);
});
for (const outcome of ['lost', 'wrong']) test(`${outcome} mutation remains paused after reload`, async ({ page }) => {
  const state = await fixture(page); state.outcome = outcome; await prepare(page); await submit(page).click();
  await expect(page.getByText(/request outcome is unverified/).first()).toBeVisible();
  await page.reload(); await page.getByRole('button', { name: 'Run', exact: true }).click(); await page.getByRole('button', { name: 'Replay observations', exact: true }).click();
  const acknowledgment = page.getByRole('button', { name: 'I checked replay jobs; allow a new request' }); await expect(acknowledgment).toBeDisabled(); await expect(submit(page)).toBeDisabled();
  await page.getByRole('button', { name: 'Refresh replay jobs' }).click(); await expect(acknowledgment).toBeEnabled(); expect(state.posts).toHaveLength(1);
});
test('saved full action chunks render without a simulation quality claim', async ({ page }, testInfo) => {
  const state = await fixture(page); await prepare(page); await submit(page).click();
  await expect(page.getByRole('article', { name: 'Observation replay details' })).toHaveAttribute('data-job-id', 'replay-001'); complete(state);
  await page.getByRole('button', { name: 'Refresh replay jobs' }).click();
  await expect(page.getByRole('region', { name: 'Predicted action chunks' }).getByRole('img')).toHaveCount(6);
  await page.getByLabel('Recorded observation', { exact: true }).selectOption('1');
  await expect(page.getByRole('link', { name: 'Download verified replay record' })).toHaveAttribute('href', /replay-001%3Aoperation\/download$/);
  await expect(page.getByText(/does not measure task success/)).toBeVisible();
  await page.getByRole('button', { name: 'Dark', exact: true }).click();
  const width = await page.evaluate(() => [document.documentElement.clientWidth, document.documentElement.scrollWidth]); expect(width[1]).toBeLessThanOrEqual(width[0] + 1);
  await page.screenshot({ path: testInfo.outputPath('replay-actions-dark.png'), fullPage: true });
});
test('a mismatched saved record is withheld, and unmeasured report claims are rejected', async ({ page }) => {
  const state = await fixture(page); await prepare(page); await submit(page).click();
  await expect(page.getByRole('article', { name: 'Observation replay details' })).toHaveAttribute('data-job-id', 'replay-001'); state.invalid = true; const job = complete(state);
  await page.getByRole('button', { name: 'Refresh replay jobs' }).click(); await expect(panel(page).getByRole('alert')).toContainText('does not match'); await expect(page.getByRole('region', { name: 'Predicted action chunks' })).toHaveCount(0);
  job.result.reports[0].quality_verified = true; await page.getByRole('button', { name: 'Refresh replay jobs' }).click();
  await expect(panel(page).getByRole('alert')).toContainText('Complete replay evidence is unavailable'); await expect(page.getByRole('link', { name: 'Download verified replay record' })).toHaveCount(0);
});


test('journal write failure blocks submission even when removal still works', async ({ page }) => {
  const state = await fixture(page); await prepare(page);
  await page.evaluate(() => {
    const original = Storage.prototype.setItem;
    Storage.prototype.setItem = function (key: string, value: string) {
      if (key.startsWith('firebird:job-attempt:')) throw new DOMException('Journal write denied', 'SecurityError');
      return original.call(this, key, value);
    };
  });
  await submit(page).click();
  await expect(submit(page)).toBeDisabled();
  expect(state.posts).toHaveLength(0);
  await page.getByRole('button', { name: 'Refresh replay jobs' }).click();
  await expect(submit(page)).toBeDisabled();
  expect(state.posts).toHaveLength(0);
});

test('acknowledged job remains visible when journal cleanup fails', async ({ page }) => {
  const state = await fixture(page); await prepare(page);
  await page.evaluate(() => {
    const original = Storage.prototype.removeItem;
    Storage.prototype.removeItem = function (key: string) {
      if (key.startsWith('firebird:job-attempt:')) throw new DOMException('Journal removal denied', 'SecurityError');
      return original.call(this, key);
    };
  });
  await submit(page).click();
  await expect(page.getByRole('article', { name: 'Observation replay details' })).toHaveAttribute('data-job-id', 'replay-001');
  expect(state.posts).toHaveLength(1);
  await page.getByText('Prepare another replay', { exact: true }).click();
  await expect(submit(page)).toBeDisabled();
  const pending = await page.evaluate(() => Object.keys(sessionStorage).filter(key => key.startsWith('firebird:job-attempt:')).map(key => JSON.parse(sessionStorage.getItem(key)!)));
  expect(pending).toHaveLength(1);
  expect(pending[0].state).toBe('pending');
  await expect(page.getByText(/^Submitting one /)).toHaveCount(0);
});


test('journal cleanup failure after navigation preserves recovery guidance', async ({ page }) => {
  const state = await fixture(page); await prepare(page);
  let release!: () => void; state.postGate = new Promise<void>(resolve => { release = resolve; });
  await page.evaluate(() => {
    const original = Storage.prototype.removeItem;
    Storage.prototype.removeItem = function (key: string) {
      if (key.startsWith('firebird:job-attempt:')) {
        (window as unknown as { journalCleanupFailures: number }).journalCleanupFailures = 1;
        throw new DOMException('Journal removal denied', 'SecurityError');
      }
      return original.call(this, key);
    };
  });
  await submit(page).click(); await expect.poll(() => state.posts.length).toBe(1);
  await page.getByRole('button', { name: 'Dataset', exact: true }).click(); release();
  await expect.poll(() => page.evaluate(() => (window as unknown as { journalCleanupFailures?: number }).journalCleanupFailures)).toBe(1);
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  await page.getByRole('button', { name: 'Replay observations', exact: true }).click();
  await expect(page.getByRole('button', { name: 'I checked replay jobs; allow a new request' })).toBeVisible();
  await expect(page.getByText(/^Submitting one /)).toHaveCount(0);
  await expect(submit(page)).toBeDisabled();
  expect(state.posts).toHaveLength(1);
});

test('visible observation source cards preserve the distinction between recorded and generated data', async ({ page }) => {
  const state = await fixture(page); await prepare(page);
  await expect(submit(page)).toBeEnabled();
  await page.getByRole('radio', { name: 'Generated test observations', exact: true }).focus();
  await page.keyboard.press('ArrowLeft');
  await expect(page.getByRole('radio', { name: 'Recorded dataset', exact: true })).toBeChecked();
  await expect(page.getByRole('checkbox', { name: /I verified that these/ })).not.toBeChecked();
  await expect(submit(page)).toBeDisabled();
  await expect(page.getByRole('radio', { name: policy.label, exact: true })).toBeVisible();
  expect(state.posts).toEqual([]);
});

test('a missing replay worker explains how to reconnect and cannot start a job', async ({ page }) => {
  const state = await fixture(page); state.workers = [];
  await page.getByRole('button', { name: 'Refresh replay jobs' }).click();
  await expect(page.getByText('Replay worker not connected', { exact: true })).toBeVisible();
  await expect(submit(page)).toBeDisabled();
  expect(state.posts).toEqual([]);
});

test('replay mutation ACK status contract accepts only six exact strings', async () => {
  const request: Parameters<typeof startReplay>[1] = { operation: 'policy.run', runtime_id: runtime.id, artifact_id: policy.id, dataset_job_id: 'data', timeout_seconds: 600, native_replay: { adapter: 'act-packed-observation-v1', selection: [{ episode_index: 0, frame_index: 3 }], coordinate_attestation: 'generated_fixture', units: ['degrees', 'degrees', 'degrees', 'degrees', 'degrees', 'recorded_gripper'] } };
  const valid = ['queued', 'running', 'succeeded', 'failed', 'cancelled', 'interrupted'];
  const original = { id: 'contract', project_id: 'alpha', kind: request.operation, status: 'running', request } as Parameters<typeof cancelReplay>[0];
  const realFetch = globalThis.fetch;
  let methods: string[] = [], status: unknown = 'running', cancelResponse = false;
  globalThis.fetch = async (_input, init) => {
    methods.push(init?.method ?? 'GET');
    return new Response(JSON.stringify({ ...original, status: cancelResponse && init?.method === 'GET' ? 'running' : status }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  };
  try {
    for (status of valid) { methods = []; expect((await startReplay('alpha', request)).status).toBe(status); expect(methods).toEqual(['POST']); }
    for (status of [['running'], ['succeeded'], [['running']], [], null, true, 1, {}, undefined, 'unknown']) {
      methods = []; await expect(startReplay('alpha', request)).rejects.toThrow(/outcome is unverified/); expect(methods).toEqual(['POST']);
      methods = []; await expect(cancelReplay(original, () => true)).rejects.toThrow(/outcome is unverified/); expect(methods).toEqual(['GET']);
    }
    cancelResponse = true; status = ['cancelled']; methods = [];
    await expect(cancelReplay(original, () => true)).rejects.toThrow(/outcome is unverified/); expect(methods).toEqual(['GET', 'POST']);
    status = 'cancelled'; methods = [];
    expect((await cancelReplay(original, () => true)).status).toBe('cancelled'); expect(methods).toEqual(['GET', 'POST']);
  } finally { globalThis.fetch = realFetch; }
});

for (const phase of ['submit', 'cancel-preflight', 'cancel-receipt'] as const) test(`replay malformed ${phase} status is rejected without automatic retry`, async ({ page }) => {
  const state = await fixture(page); await prepare(page);
  if (phase === 'submit') {
    await page.route('**/api/v1/projects/alpha/policy-jobs', route => {
      const request = route.request().postDataJSON(); state.posts.push(request);
      return route.fulfill({ status: 202, json: { id: 'malformed', project_id: 'alpha', kind: request.operation, status: ['running'], request } });
    });
    await submit(page).click();
  } else {
    await submit(page).click();
    await expect(page.getByRole('article', { name: 'Observation replay details' })).toHaveAttribute('data-job-id', 'replay-001');
    const original = state.jobs.find(item => item.id === 'replay-001')!;
    if (phase === 'cancel-preflight') await page.route('**/api/v1/jobs/replay-001', route => route.fulfill({ json: { ...original, status: ['running'] } }));
    else await page.route('**/api/v1/jobs/replay-001/cancel', route => { state.cancelled.push('replay-001'); return route.fulfill({ json: { ...original, status: ['cancelled'] } }); });
    await page.getByRole('button', { name: 'Cancel selected replay', exact: true }).click();
    await page.getByRole('button', { name: 'Confirm cancellation', exact: true }).click();
  }
  await expect(panel(page).getByRole('alert')).toContainText('outcome is unverified');
  expect(state.posts).toHaveLength(1);
  expect(state.cancelled).toHaveLength(phase === 'cancel-receipt' ? 1 : 0);
  const details = page.getByRole('article', { name: 'Observation replay details' });
  if (phase === 'submit') {
    expect(await page.evaluate(() => sessionStorage.getItem('firebird:job-attempt:policy.run.replay:alpha'))).toContain('uncertain');
    await expect(details).toHaveCount(0);
  } else {
    await expect(details).toHaveAttribute('data-job-id', 'replay-001');
    await expect(details.locator('.status')).toHaveText('running');
    await expect(page.getByRole('button', { name: 'Confirm cancellation', exact: true })).toHaveCount(0);
    const recovery = await page.evaluate(() => Object.keys(sessionStorage).filter(key => key.startsWith('firebird:cancel-attempt:')).map(key => JSON.parse(sessionStorage.getItem(key)!)));
    expect(recovery).toHaveLength(1); expect(recovery[0]).toMatchObject({ jobId: 'replay-001', state: 'uncertain', action: 'cancel' });
  }
});

// Pure journal contract checks run without a browser or application server.
test('cancellation journal contract keeps same-job successor ownership distinct', async () => {
  const { readCancellation, transitionCancellation } = await import('../../apps/web/src/lib/native-cancellation-attempt');
  const values = new Map<string, string>();
  const store = { getItem: (key: string) => values.get(key) ?? null, setItem: (key: string, value: string) => { values.set(key, value); }, removeItem: (key: string) => { values.delete(key); } };
  const scope = { project: 'alpha', workflow: 'replay' as const };
  const first = { ...scope, version: 1 as const, action: 'cancel' as const, id: 'first', jobId: 'same-job', requestSha256: 'a'.repeat(64), state: 'pending' as const };
  expect(transitionCancellation(scope, null, first, store)).toBe(true);
  const uncertain = { ...first, state: 'uncertain' as const };
  expect(transitionCancellation(scope, first, uncertain, store)).toBe(true);
  expect(transitionCancellation(scope, first, null, store)).toBe(false);
  expect(transitionCancellation(scope, uncertain, null, store)).toBe(true);
  const next = { ...first, id: 'different-attempt' };
  expect(transitionCancellation(scope, null, next, store)).toBe(true);
  expect(transitionCancellation(scope, first, uncertain, store)).toBe(false);
  expect(transitionCancellation(scope, first, null, store)).toBe(false);
  expect(readCancellation(scope, store)).toEqual(next);
});

test('cancellation journal contract distinguishes denied reads and never clears on failed cleanup', async () => {
  const { readCancellation, transitionCancellation, CancellationStorageError } = await import('../../apps/web/src/lib/native-cancellation-attempt');
  const scope = { project: 'alpha', workflow: 'replay' as const };
  const attempt = { ...scope, version: 1 as const, action: 'cancel' as const, id: 'owned', jobId: 'job', requestSha256: 'b'.repeat(64), state: 'pending' as const };
  const denied = () => { throw new Error('Denied'); };
  expect(() => readCancellation(scope, { getItem: denied })).toThrow(CancellationStorageError);
  let value: string | null = null;
  const store = { getItem: () => value, setItem: (_: string, next: string) => { value = next; }, removeItem: denied };
  expect(() => transitionCancellation(scope, null, attempt, { ...store, setItem: denied })).toThrow(CancellationStorageError);
  expect(value).toBeNull();
  expect(transitionCancellation(scope, null, attempt, store)).toBe(true);
  expect(() => transitionCancellation(scope, attempt, null, store)).toThrow(CancellationStorageError);
  expect(readCancellation(scope, store)).toEqual(attempt);
});

test('cancellation journal contract rejects malformed and oversized records and isolates scopes', async () => {
  const { cancellationKey, readCancellation, transitionCancellation, CancellationJournalError } = await import('../../apps/web/src/lib/native-cancellation-attempt');
  const scope = { project: 'alpha', workflow: 'replay' as const };
  for (const raw of ['{', 'null', '[]', 'x'.repeat(4097), JSON.stringify({ version: 1, action: 'cancel', ...scope, id: 'id', jobId: 'job', requestSha256: 'a'.repeat(64), state: ['pending'] })]) expect(() => readCancellation(scope, { getItem: () => raw })).toThrow(CancellationJournalError);
  expect(cancellationKey(scope)).not.toBe(cancellationKey({ ...scope, project: 'beta' }));
  expect(cancellationKey(scope)).not.toBe(cancellationKey({ ...scope, workflow: 'distillation' }));
  const other = { version: 1 as const, action: 'cancel' as const, project: 'beta', workflow: 'replay' as const, id: 'foreign', jobId: 'job', requestSha256: 'a'.repeat(64), state: 'pending' as const };
  let writes = 0;
  expect(() => transitionCancellation(scope, null, other, { getItem: () => null, setItem: () => { writes++; }, removeItem: () => { writes++; } })).toThrow(CancellationJournalError);
  expect(writes).toBe(0);
});

test('cancellation journal contract hashes exact finite recipe values without storing them', async () => {
  const { cancellationRequestSha } = await import('../../apps/web/src/lib/native-cancellation-attempt');
  const first = await cancellationRequestSha({ steps: 1, source: 'owned', recipe: [true, 0.5] });
  expect(first).toMatch(/^[a-f0-9]{64}$/);
  expect(await cancellationRequestSha({ recipe: [true, 0.5], source: 'owned', steps: 1 })).toBe(first);
  expect(await cancellationRequestSha({ steps: true, source: 'owned', recipe: [true, 0.5] })).not.toBe(first);
  await expect(cancellationRequestSha({ value: Infinity })).rejects.toThrow();
  await expect(cancellationRequestSha({ value: 'x'.repeat(32769) })).rejects.toThrow();
});

async function cancellationFixture(page: Page) {
  const state = await fixture(page); await prepare(page); await submit(page).click();
  await expect(page.getByRole('article', { name: 'Observation replay details' })).toHaveAttribute('data-job-id', 'replay-001');
  return state;
}
async function confirmCancellation(page: Page) {
  await page.getByRole('button', { name: 'Cancel selected replay', exact: true }).click();
  await page.getByRole('button', { name: 'Confirm cancellation', exact: true }).click();
}
async function reopenCancellationLane(page: Page) {
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  const choice = page.getByRole('button', { name: 'Replay observations', exact: true });
  await expect(panel(page).or(choice).first()).toBeVisible();
  if (await choice.isVisible()) await choice.click();
  await expect(panel(page)).toBeVisible();
}
const cancellationRecovery = (page: Page) => page.getByRole('region', { name: 'Cancellation recovery', exact: true });
const acknowledgeCancellation = (page: Page) => page.getByRole('button', { name: 'I reviewed cancellation history; allow another cancellation', exact: true });

test('cancellation outcome survives stage navigation and reload without another POST', async ({ page }) => {
  const state = await cancellationFixture(page);
  await page.route('**/api/v1/jobs/replay-001/cancel', async route => { state.cancelled.push('replay-001'); await route.abort('failed'); });
  await confirmCancellation(page);
  await expect(cancellationRecovery(page)).toContainText('Cancellation outcome is unverified for replay-001');
  await page.getByRole('button', { name: 'Dataset', exact: true }).click(); await reopenCancellationLane(page);
  await expect(cancellationRecovery(page)).toContainText('replay-001');
  await page.reload(); await reopenCancellationLane(page);
  await expect(cancellationRecovery(page)).toContainText('replay-001');
  await expect(acknowledgeCancellation(page)).toBeDisabled();
  await page.getByRole('button', { name: 'Refresh replay jobs', exact: true }).click();
  await expect(acknowledgeCancellation(page)).toBeDisabled();
  await page.getByRole('button', { name: 'Refresh cancellation history', exact: true }).click();
  await expect(acknowledgeCancellation(page)).toBeEnabled();
  expect(state.cancelled).toEqual(['replay-001']); expect(state.posts).toHaveLength(1);
});

for (const method of ['getItem', 'setItem'] as const) test(`cancellation ${method} denial blocks all cancellation I/O and remains latched`, async ({ page }) => {
  const state = await cancellationFixture(page); let reads = 0;
  await page.route('**/api/v1/jobs/replay-001', route => { reads++; return route.fulfill({ json: state.jobs.find(item => item.id === 'replay-001') }); });
  await page.evaluate(method => {
    const original = Storage.prototype[method];
    Object.defineProperty(Storage.prototype, method, { configurable: true, value: function (key: string, ...args: string[]) {
      if (key.startsWith('firebird:cancel-attempt:')) throw new DOMException('Cancellation journal denied', 'SecurityError');
      return Reflect.apply(original, this, [key, ...args]);
    } });
  }, method);
  await confirmCancellation(page);
  await expect(page.getByText(/Cancellation recovery storage is unavailable/)).toBeVisible();
  await expect(page.getByRole('button', { name: 'Cancel selected replay', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Dataset', exact: true }).click(); await reopenCancellationLane(page);
  await page.getByLabel('Saved replay', { exact: true }).selectOption('replay-001');
  await expect(page.getByText(/Cancellation recovery storage is unavailable/)).toBeVisible();
  await expect(page.getByRole('button', { name: 'Cancel selected replay', exact: true })).toBeDisabled();
  expect(reads).toBe(0); expect(state.cancelled).toEqual([]);
});

test('cancellation acknowledgment survives failed journal cleanup and failed history', async ({ page }) => {
  const state = await cancellationFixture(page); const original = state.jobs.find(item => item.id === 'replay-001');
  await page.evaluate(() => { const original = Storage.prototype.removeItem; Storage.prototype.removeItem = function (key: string) { if (key.startsWith('firebird:cancel-attempt:')) throw new DOMException('Denied', 'SecurityError'); return original.call(this, key); }; });
  let outage = false;
  await page.route('**/api/v1/jobs/replay-001/cancel', route => { state.cancelled.push('replay-001'); outage = true; return route.fulfill({ json: { ...original, status: 'cancelled', result: { reports: [{ quality_verified: true }] } } }); });
  await page.route('**/api/v1/projects/alpha/jobs', route => route.fulfill(outage ? { status: 503, json: { detail: 'History unavailable' } } : { json: state.jobs }));
  await confirmCancellation(page);
  await expect(page.getByText('Cancellation response for replay-001: cancelled. Recorded job history remains authoritative.', { exact: true })).toBeVisible();
  await expect(page.getByText(/Cancellation recovery storage is unavailable/)).toBeVisible();
  await expect(cancellationRecovery(page)).toContainText('replay-001');
  await expect(page.getByRole('button', { name: 'Cancel selected replay', exact: true })).toBeDisabled();
  expect(state.cancelled).toEqual(['replay-001']);
  const stored = await page.evaluate(() => Object.keys(sessionStorage).filter(key => key.startsWith('firebird:cancel-attempt:')).map(key => sessionStorage.getItem(key)));
  expect(stored).toHaveLength(1); expect(stored[0]).not.toContain('reports'); expect(stored[0]).not.toContain('quality_verified');
});

test('validated cancellation acknowledgment survives a newly denied storage read', async ({ page }) => {
  const state = await cancellationFixture(page); const original = state.jobs.find(item => item.id === 'replay-001');
  let storedBeforeDenial: string[] = [];
  await page.route('**/api/v1/jobs/replay-001/cancel', async route => {
    state.cancelled.push('replay-001');
    storedBeforeDenial = await page.evaluate(() => {
      const original = Storage.prototype.getItem;
      const entries = Object.keys(sessionStorage).filter(key => key.startsWith('firebird:cancel-attempt:')).map(key => original.call(sessionStorage, key)!);
      Storage.prototype.getItem = function (key: string) { if (key.startsWith('firebird:cancel-attempt:')) throw new DOMException('Denied after acknowledgment', 'SecurityError'); return original.call(this, key); };
      return entries;
    });
    return route.fulfill({ json: { ...original, status: 'cancelled', result: { reports: [{ quality_verified: true }] } } });
  });
  await confirmCancellation(page);
  await expect(page.getByText('Cancellation response for replay-001: cancelled. Recorded job history remains authoritative.', { exact: true })).toBeVisible();
  await expect(page.getByText(/Cancellation recovery storage is unavailable/)).toBeVisible();
  await expect(cancellationRecovery(page)).toContainText('replay-001');
  await page.getByRole('button', { name: 'Dataset', exact: true }).click(); await reopenCancellationLane(page);
  await page.getByLabel('Saved replay', { exact: true }).selectOption('replay-001');
  await expect(page.getByText('Cancellation response for replay-001: cancelled. Recorded job history remains authoritative.', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Cancel selected replay', exact: true })).toBeDisabled();
  expect(state.cancelled).toEqual(['replay-001']); expect(state.posts).toHaveLength(1);
  expect(storedBeforeDenial).toHaveLength(1); expect(JSON.parse(storedBeforeDenial[0])).toMatchObject({ jobId: 'replay-001', state: 'pending' });
  expect(storedBeforeDenial[0]).not.toContain('reports'); expect(storedBeforeDenial[0]).not.toContain('quality_verified');
});

test('cancellation selection A to B to A invalidates the old preflight', async ({ page }) => {
  const state = await cancellationFixture(page); const original = state.jobs.find(item => item.id === 'replay-001');
  state.jobs.push({ ...original, id: 'replay-002' }); await page.getByRole('button', { name: 'Refresh replay jobs', exact: true }).click();
  let reads = 0, release!: () => void; const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/api/v1/jobs/replay-001', async route => { reads++; await gate; return route.fulfill({ json: original }); });
  await confirmCancellation(page); await expect.poll(() => reads).toBe(1);
  await page.getByLabel('Saved replay', { exact: true }).selectOption('replay-002');
  await page.getByLabel('Saved replay', { exact: true }).selectOption('replay-001'); release();
  await expect(cancellationRecovery(page)).toContainText('Cancellation outcome is unverified');
  expect(state.cancelled).toEqual([]);
});

test('late cancellation receipt survives unmount without replacing a manual saved job', async ({ page }) => {
  const state = await cancellationFixture(page); const original = state.jobs.find(item => item.id === 'replay-001');
  state.jobs.push({ ...original, id: 'replay-002' }); await page.getByRole('button', { name: 'Refresh replay jobs', exact: true }).click();
  let release!: () => void; const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/api/v1/jobs/replay-001/cancel', async route => { state.cancelled.push('replay-001'); await gate; return route.fulfill({ json: { ...original, status: 'cancelled' } }); });
  await confirmCancellation(page); await expect.poll(() => state.cancelled.length).toBe(1);
  await page.getByRole('button', { name: 'Dataset', exact: true }).click(); await reopenCancellationLane(page);
  await page.getByLabel('Saved replay', { exact: true }).selectOption('replay-002'); release();
  await expect(page.getByText('Cancellation response for replay-001: cancelled. Recorded job history remains authoritative.', { exact: true })).toBeVisible();
  await expect(page.getByRole('article', { name: 'Observation replay details' })).toHaveAttribute('data-job-id', 'replay-002');
  expect(state.cancelled).toEqual(['replay-001']); expect(state.posts).toHaveLength(1);
});

test('cancellation recovery requires a fresh owned history read after uncertainty', async ({ page }) => {
  const state = await cancellationFixture(page);
  let releaseCancel!: () => void; const cancelGate = new Promise<void>(resolve => { releaseCancel = resolve; });
  await page.route('**/api/v1/jobs/replay-001/cancel', async route => { state.cancelled.push('replay-001'); await cancelGate; return route.abort('failed'); });
  await confirmCancellation(page); await expect.poll(() => state.cancelled.length).toBe(1);
  let reads = 0, releaseHistory!: () => void; const historyGate = new Promise<void>(resolve => { releaseHistory = resolve; });
  await page.route('**/api/v1/projects/alpha/jobs', async route => { const index = ++reads; if (index === 1) await historyGate; return route.fulfill({ json: index === 1 ? state.jobs : state.jobs.map(item => item.id === 'replay-001' ? { ...item, status: 'cancelled' } : item) }); });
  // A normal shared history poll begins before the cancellation becomes uncertain.
  await expect.poll(() => reads).toBe(1);
  releaseCancel(); await expect(cancellationRecovery(page)).toContainText('Cancellation outcome is unverified');
  await expect(acknowledgeCancellation(page)).toBeDisabled();
  // The initial shared query remains unresolved; recovery must make its own GET.
  await page.getByRole('button', { name: 'Refresh cancellation history', exact: true }).click();
  await expect.poll(() => reads).toBeGreaterThanOrEqual(2); await expect(acknowledgeCancellation(page)).toBeEnabled();
  await expect(cancellationRecovery(page).getByRole('status')).toHaveText('Fresh cancellation history for replay-001: cancelled. This is the status returned by your recovery refresh.');
  releaseHistory();
  await expect(cancellationRecovery(page).getByRole('status')).toContainText('replay-001: cancelled');
  expect(state.cancelled).toEqual(['replay-001']);
});

test('cancellation review cannot authorize a changed selection generation', async ({ page }) => {
  const state = await cancellationFixture(page); const original = state.jobs.find(item => item.id === 'replay-001');
  state.jobs.push({ ...original, id: 'replay-002' }); await page.getByRole('button', { name: 'Refresh replay jobs', exact: true }).click();
  await page.route('**/api/v1/jobs/replay-001/cancel', async route => { state.cancelled.push('replay-001'); return route.abort('failed'); });
  await confirmCancellation(page); await expect(cancellationRecovery(page)).toContainText('Cancellation outcome is unverified');
  let release!: () => void; const gate = new Promise<void>(resolve => { release = resolve; }); let reads = 0;
  await page.route('**/api/v1/projects/alpha/jobs', async route => { reads++; await gate; return route.fulfill({ json: state.jobs }); });
  await page.getByRole('button', { name: 'Refresh cancellation history', exact: true }).click(); await expect.poll(() => reads).toBeGreaterThan(0);
  await page.getByLabel('Saved replay', { exact: true }).selectOption('replay-002'); await page.getByLabel('Saved replay', { exact: true }).selectOption('replay-001'); release();
  await expect(page.getByRole('button', { name: 'Refresh cancellation history', exact: true })).toBeEnabled();
  await expect(acknowledgeCancellation(page)).toBeDisabled(); expect(state.cancelled).toEqual(['replay-001']);
});

test('pending cancellation reload keeps the original unique attempt and never resends', async ({ page }) => {
  const state = await cancellationFixture(page);
  let release!: () => void; const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/api/v1/jobs/replay-001/cancel', async route => { state.cancelled.push('replay-001'); await gate; try { await route.abort('failed'); } catch { /* The original page was explicitly reloaded. */ } });
  await confirmCancellation(page); await expect.poll(() => state.cancelled.length).toBe(1);
  const before = await page.evaluate(() => JSON.parse(sessionStorage.getItem(Object.keys(sessionStorage).find(key => key.startsWith('firebird:cancel-attempt:'))!)!));
  expect(before.state).toBe('pending'); expect(before.jobId).toBe('replay-001');
  await page.reload(); await reopenCancellationLane(page);
  await expect(cancellationRecovery(page)).toContainText('Cancellation outcome is unverified for replay-001');
  const after = await page.evaluate(() => JSON.parse(sessionStorage.getItem(Object.keys(sessionStorage).find(key => key.startsWith('firebird:cancel-attempt:'))!)!));
  expect(after).toEqual({ ...before, state: 'uncertain' });
  await expect(acknowledgeCancellation(page)).toBeDisabled(); release(); expect(state.cancelled).toEqual(['replay-001']);
});

test('cancellation review rejects changed original identity and survives another project', async ({ page }) => {
  const state = await cancellationFixture(page); const original = state.jobs.find(item => item.id === 'replay-001');
  await page.route('**/api/v1/jobs/replay-001/cancel', async route => { state.cancelled.push('replay-001'); return route.abort('failed'); });
  await confirmCancellation(page); await expect(cancellationRecovery(page)).toContainText('replay-001');
  await page.route('**/api/v1/projects/alpha/jobs', route => route.fulfill({ json: [{ ...original, request: { ...original.request, artifact_id: 'changed-source' } }] }));
  await page.getByRole('button', { name: 'Refresh cancellation history', exact: true }).click();
  await expect(page.getByText(/Fresh history did not verify the original cancellation target/)).toBeVisible();
  await expect(acknowledgeCancellation(page)).toBeDisabled();
  await page.route('**/api/v1/projects', route => route.fulfill({ json: [{ id: 'alpha', name: 'Original project', created_at: time }, { id: 'beta', name: 'Another project', created_at: time }] }));
  await page.reload(); await reopenCancellationLane(page);
  await selectProject(page, 'beta'); await reopenCancellationLane(page);
  await expect(cancellationRecovery(page)).toHaveCount(0);
  await selectProject(page, 'alpha'); await reopenCancellationLane(page);
  await expect(cancellationRecovery(page)).toContainText('replay-001');
  await expect(acknowledgeCancellation(page)).toBeDisabled(); expect(state.cancelled).toEqual(['replay-001']);
});


test('a later history failure disables cancellation recovery acknowledgment', async ({ page }) => {
  const state = await cancellationFixture(page);
  await page.route('**/api/v1/jobs/replay-001/cancel', async route => { state.cancelled.push('replay-001'); return route.abort('failed'); });
  await confirmCancellation(page); await expect(cancellationRecovery(page)).toContainText('replay-001');
  await page.getByRole('button', { name: 'Refresh cancellation history', exact: true }).click();
  await expect(acknowledgeCancellation(page)).toBeEnabled();
  await page.route('**/api/v1/projects/alpha/jobs', route => route.fulfill({ status: 503, json: { detail: 'Current history unavailable' } }));
  await page.getByRole('button', { name: 'Refresh replay jobs', exact: true }).click();
  await expect(page.getByText(/Job updates are unavailable/)).toBeVisible();
  await expect(acknowledgeCancellation(page)).toBeDisabled(); expect(state.cancelled).toEqual(['replay-001']);
});


test('8-step replay shows all predicted actions and chart endpoints without executing3 as the full horizon', async ({ page }) => {
  const state = await fixture(page); state.prediction = 8; await prepare(page); await submit(page).click();
  await expect(page.getByRole('article', { name: 'Observation replay details' })).toHaveAttribute('data-job-id', 'replay-001'); complete(state);
  await page.getByRole('button', { name: 'Refresh replay jobs' }).click();
  const charts = page.getByRole('region', { name: 'Predicted action chunks' });
  await expect(charts.getByRole('img')).toHaveCount(6);
  await expect(charts.getByRole('img').first()).toHaveAccessibleName(/8 predicted actions/);
  await expect(page.getByRole('article', { name: 'Observation replay details' })).toContainText('8 × 6');
  await expect(charts.getByText(/actions 1–8$/)).toHaveCount(6);
  const points = await charts.locator('polyline').first().getAttribute('points');
  expect(points?.split(' ')).toHaveLength(8); expect(points?.split(' ').at(-1)?.split(',')[0]).toBe('256');
  await expect(page.getByText(/does not measure task success/)).toBeVisible();
  expect(state.posts).toHaveLength(1);
});
