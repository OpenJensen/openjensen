import { startReplay, cancelReplay } from '../../apps/web/src/lib/native-replay';
import { expect, test, type Page } from '@playwright/test';

const time = '2026-09-27T10:30:00Z', model = `sha256:${'f'.repeat(64)}`;
const runtime = { id: 'replay-cpu', label: 'Local observation replay', execution: 'native', provider: 'local', device: 'cpu', enabled: true, native_replay: true, native_replay_only: true, training: false, simulation: false, run: false, engine_evaluation: false };
const policy = { id: 'packed:policy', project_id: 'alpha', job_id: 'packed', label: 'Packed ACT fixture', format: 'native_quantized', path: 'owned', manifest_sha256: 'a'.repeat(64), file_bytes: 10000, metadata: { architecture: 'act' } };
const dataset = { id: 'data', project_id: 'alpha', kind: 'dataset.inspect', status: 'succeeded', created_at: time, updated_at: time, request: { source: 'local', path: 'fixture' }, result: { source: 'local', format: 'lerobot_v3', repo_id: null, revision: `metadata-sha256:${'c'.repeat(64)}`, metadata_sha256: 'c'.repeat(64), fps: 30, robot_type: 'generated-so101', license: null, inspected_at: time, warnings: [], total_episodes: 6, total_frames: 24, inspection_scope: 'complete_snapshot', features: {}, snapshot: { id: `sha256:${'b'.repeat(64)}`, manifest_sha256: 'b'.repeat(64), lineage_validated: true, total_episodes: 6 } } };
const panel = (page: Page) => page.getByRole('region', { name: 'CPU observation replay', exact: true });
const submit = (page: Page) => page.getByRole('button', { name: 'Run CPU observation replay', exact: true });
async function fixture(page: Page) {
  const state = { jobs: [dataset] as Record<string, any>[], posts: [] as Record<string, any>[], cancelled: [] as string[], outcome: 'ok', postGate: null as Promise<void> | null, invalid: false, workers: [runtime] as Record<string, unknown>[] };
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
      return route.fulfill({ json: { artifact_id: 'replay-001:operation', job_id: job.id, model_id: state.invalid ? 'wrong' : model, source_kind: 'generated_fixture', coordinate_names: ['j0', 'j1', 'j2', 'j3', 'j4', 'gripper'], units: job.request.native_replay.units, records: job.request.native_replay.selection.map((row: object) => ({ ...row, reset_repeat_exact: true, actions: Array.from({ length: 100 }, (_, step) => Array.from({ length: 6 }, (_, joint) => joint + step / 100)) })) } });
    }
    if (path.startsWith('/api/v1/jobs/')) { const job = state.jobs.find(item => item.id === path.split('/').at(-1)); if (job) return route.fulfill({ json: job }); }
    if (req.method() !== 'GET') return route.fulfill({ status: 405, json: {} });
    return route.continue();
  });
  await page.goto('/'); await expect(page.getByLabel('Current project')).toHaveValue('alpha');
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  await page.getByRole('button', { name: 'Observation replay · ACT', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Inspect the actions your policy predicts' })).toBeVisible();
  return state;
}
async function prepare(page: Page) {
  await page.getByLabel('Packed ACT policy', { exact: true }).selectOption(policy.id);
  await page.getByLabel('Observation dataset', { exact: true }).selectOption('data');
  await page.getByLabel('Episode and frame pairs', { exact: true }).fill('0:3, 2:1');
  await page.getByLabel('Six replay coordinate units', { exact: true }).fill('degrees, degrees, degrees, degrees, degrees, recorded_gripper');
  await page.getByRole('checkbox', { name: 'These are generated test observations.' }).check();
  await page.getByRole('checkbox', { name: /I verified that these/ }).check();
}
function complete(state: Awaited<ReturnType<typeof fixture>>) {
  const job = state.jobs.find(item => item.id === 'replay-001')!;
  job.status = 'succeeded'; job.stage = 'replaying'; job.result = { artifacts: [{ ...policy, id: 'replay-001:operation', job_id: job.id, format: 'native_run_record', metadata: { recipe: 'native-observation-replay-v1', model_id: model } }], reports: [{ operation: 'policy.run', stage: 'native_replay', mode: 'independent_observation_replay', device: 'cpu', source_artifact_id: policy.id, dataset_job_id: 'data', observation_source: { kind: 'generated_fixture' }, model_id: model, observations: 2, action_shape: [100, 6], reset_repeat_exact: true, server_closed: true, task_success: null, quality_verified: false, calibration_verified: false, speedup_verified: false, isaac_runtime_verified: false, elapsed_seconds: 2 }] };
  return job;
}
test('only local packed policies and explicit bounded observations can be replayed', async ({ page }) => {
  const state = await fixture(page); await expect(submit(page)).toBeDisabled();
  await expect(page.getByLabel('Packed ACT policy', { exact: true }).locator('option')).toHaveCount(2);
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
  await page.reload(); await page.getByRole('button', { name: 'Run', exact: true }).click(); await page.getByRole('button', { name: 'Observation replay · ACT', exact: true }).click();
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
  await page.getByRole('button', { name: 'Observation replay · ACT', exact: true }).click();
  await expect(page.getByRole('button', { name: 'I checked replay jobs; allow a new request' })).toBeVisible();
  await expect(page.getByText(/^Submitting one /)).toHaveCount(0);
  await expect(submit(page)).toBeDisabled();
  expect(state.posts).toHaveLength(1);
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
  }
});
