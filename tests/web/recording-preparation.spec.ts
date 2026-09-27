import { expect, test, type Page } from '@playwright/test';
import { createHash } from 'node:crypto';

const stamp = '2026-09-27T00:00:00Z', hex = (c: string, n = 64) => c.repeat(n);
const operation = 'dataset.inspect.recordings';
const options = { configured: true, runtime_verified: false, configuration_sha256: hex('a'), max_episodes: 100, max_source_bytes: 8589934592, setup_message: 'Published fixture metadata; writer checks happen during preparation.' };
function capture(c: string, e: string) { return { session_id: hex(c, 32), session_sha256: hex(c), origin: 'synthetic', lineage_group: `declared-${c}`, controller: 'joint_position_targets', state_units: 'radians', action_units: 'radians', timebase: 'simulation_seconds', camera_key: 'observation.images.front', joint_names: ['joint_one'], width: 32, height: 32, fps: 10, physics_hz: 60, scene_sha256: hex('f'), camera_prim: '/World/Camera', content_verified: false, episodes: [{ episode_id: hex(e, 32), receipt_sha256: hex(e), frames: 12, outcome: 'unknown', termination: 'finish' }] }; }
const catalog = () => ({ configuration_sha256: hex('a'), message: 'Finalized metadata only', captures: [capture('b', 'd'), capture('c', 'e')] });
const recipe = (two = true) => ({ schema_version: 1, configuration_sha256: hex('a'), timeout_seconds: 600, captures: catalog().captures.slice(0, two ? 2 : 1).map(c => ({ session_id: c.session_id, session_sha256: c.session_sha256, episodes: c.episodes.map(e => ({ episode_id: e.episode_id, receipt_sha256: e.receipt_sha256 })) })) });
const request = (two = true) => ({ source: 'local', repo_id: null, revision: 'main', path: null, snapshot_for_training: true, recordings: recipe(two) });
function sorted(v: any): any { return Array.isArray(v) ? v.map(sorted) : v && typeof v === 'object' ? Object.fromEntries(Object.keys(v).sort().map(k => [k, sorted(v[k])])) : v; }
function job(id: string, project = 'alpha', two = true, status = 'succeeded'): any {
  const req = request(two), count = two ? 2 : 1;
  return { id, project_id: project, kind: 'dataset.inspect', status, created_at: stamp, updated_at: stamp, request: req, compute_target: null, simulation_target: null, error: null, stage: status === 'succeeded' ? 'completed' : 'preparing', result: status !== 'succeeded' ? null : { schema_version: 1, source: 'local', repo_id: null, revision: `metadata-sha256:${hex('f')}`, format: 'lerobot_v3', robot_type: 'generated', total_frames: 12 * count, total_episodes: count, fps: 10, features: { 'observation.state': { dtype: 'float32', shape: [1] }, action: { dtype: 'float32', shape: [1] }, 'observation.images.front': { dtype: 'video', shape: [32, 32, 3] } }, license: null, metadata_sha256: hex('f'), inspected_at: stamp, warnings: [], inspection_scope: 'complete_snapshot', snapshot: { schema_version: 1, id: `sha256:${hex('c')}`, manifest_sha256: hex('c'), format: 'lerobot_v3', total_bytes: 1000, file_count: 5, total_frames: 12 * count, total_episodes: count, lineage_validated: true, warnings: [] }, recording_preparation: { job_id: id, source_count: count, lineage_group_count: count, writer_readback_verified: true, source_preserved: true, task_success_verified: false, selection_sha256: createHash('sha256').update(JSON.stringify(sorted(req.recordings))).digest('hex') } } };
}
const panel = (page: Page) => page.getByRole('region', { name: 'Recording dataset preparation', exact: true });
const submit = (page: Page) => panel(page).getByRole('button', { name: 'Prepare selected recordings', exact: true });
const refresh = (page: Page) => panel(page).getByRole('button', { name: 'Refresh recordings and jobs', exact: true });
const ack = (page: Page) => panel(page).getByRole('button', { name: 'I checked recording jobs; allow a new request', exact: true });
function deferred() { let release!: () => void; const promise = new Promise<void>(r => { release = r; }); return { promise, release }; }
async function fixture(page: Page) {
  const state = { configured: true, failCatalog: false, failHistory: false, omit: '' as string, captures: catalog(), jobs: [] as any[], posts: [] as { path: string; body: any }[], lost: false, bad: null as null | string, gate: null as ReturnType<typeof deferred> | null, getGate: null as ReturnType<typeof deferred> | null, getGateId: '', nextId: 'new-preparation', reads: 0 };
  await page.route('**/api/v1/**', async route => {
    const req = route.request(), path = new URL(req.url()).pathname, project = path.split('/')[4];
    if (req.method() === 'POST') {
      state.posts.push({ path, body: req.postDataJSON() });
      const pendingGate = state.gate, lost = state.lost, bad = state.bad, id = state.nextId;
      const value = path.endsWith('/cancel') ? { ...state.jobs.find(j => j.id === path.split('/').at(-2)), status: 'cancelled' } : { ...job(id, project, true, 'queued'), request: req.postDataJSON() };
      if (pendingGate) await pendingGate.promise;
      if (lost) return route.abort('failed');
      if (bad === 'array') value.status = ['queued'];
      if (bad === 'foreign') value.project_id = 'beta';
      if (bad === 'recipe') value.request = { ...value.request, recordings: { ...value.request.recordings, timeout_seconds: 601 } };
      return route.fulfill({ status: 202, json: value });
    }
    if (path.endsWith('/recordings/options')) return route.fulfill({ json: { ...options, configured: state.configured, configuration_sha256: state.configured ? options.configuration_sha256 : null } });
    if (path.endsWith('/recordings')) return route.fulfill({ status: state.failCatalog ? 503 : 200, json: state.failCatalog ? { detail: 'Generated catalog unavailable' } : state.captures });
    if (path.endsWith('/jobs') && path.includes('/projects/')) { state.reads++; return route.fulfill({ status: state.failHistory ? 503 : 200, json: state.failHistory ? { detail: 'Generated history unavailable' } : state.jobs.filter(j => j.project_id === project && j.id !== state.omit) }); }
    if (path.startsWith('/api/v1/jobs/')) {
      const id = path.split('/')[4]; if (state.getGate && id === state.getGateId) await state.getGate.promise;
      const value = state.jobs.find(j => j.id === id);
      if (path.endsWith('/events')) return route.fulfill({ json: [] });
      if (path.endsWith('/episodes')) return route.fulfill({ json: { episodes: [], total: 0, offset: 0, limit: 6 } });
      return route.fulfill({ status: value ? 200 : 404, json: value ?? { detail: 'No saved fixture' } });
    }
    const replies: Record<string, unknown> = {
      '/api/v1/health': { status: 'ok', version: 'test' }, '/api/v1/capabilities': [],
      '/api/v1/projects': ['alpha', 'beta'].map(id => ({ id, name: `Recording ${id}`, created_at: stamp })),
      '/api/v1/policy-options': { runtimes: [], sources: [], training_models: [], training_methods: [], default_training_method: 'full' },
      '/api/v1/teaching/state': { connected: false, state: null, message: 'Generated disconnected executor', voice_configured: false },
      '/api/v1/teaching/intelligence/status': { broker_reachable: false, configured: false, busy: false },
      '/api/v1/teaching/intelligence/settings': { saved: false, revision: null, voice_model: null, message: 'Not configured' },
      '/api/v1/teaching/voice/status': { broker_reachable: false, configuration_present: false, dependencies_present: false, message: 'Not configured' },
    };
    if (path in replies) return route.fulfill({ json: replies[path] });
    if (path.endsWith('/artifacts')) return route.fulfill({ json: [] });
    return route.fulfill({ status: 404, json: { detail: `Unexpected read ${path}` } });
  });
  await page.goto('/'); await expect(page.getByLabel('Current project')).toHaveValue('alpha');
  await page.getByRole('button', { name: 'Teaching', exact: true }).click(); await expect(panel(page)).toBeVisible();
  await expect(panel(page).getByText(options.setup_message)).toBeVisible();
  return state;
}
async function selectEpisodes(page: Page, two = true) {
  await panel(page).getByRole('checkbox', { name: `Episode ${hex('d', 32)} · 12 frames · outcome unknown`, exact: true }).check();
  if (two) await panel(page).getByRole('checkbox', { name: `Episode ${hex('e', 32)} · 12 frames · outcome unknown`, exact: true }).check();
  await panel(page).getByRole('checkbox', { name: /I reviewed these exact/ }).check();
}
async function openSaved(page: Page, state: Awaited<ReturnType<typeof fixture>>, value = job('saved-recording')) {
  state.jobs.push(value); await refresh(page).click(); await panel(page).getByLabel('Saved recording preparation', { exact: true }).selectOption(value.id);
  await expect(panel(page).getByRole('article')).toHaveAttribute('data-job-id', value.id); return value;
}
async function storageFault(page: Page, key: string, method: 'setItem' | 'getItem' | 'removeItem') {
  await page.evaluate(({ key, method }) => { const saved = Storage.prototype[method] as Function; (window as any).__restoreRecordingStorage = () => { (Storage.prototype as any)[method] = saved; }; (Storage.prototype as any)[method] = function (k: string, ...args: string[]) { if (this === sessionStorage && k.includes(key)) throw new DOMException('Generated storage failure', 'QuotaExceededError'); return saved.call(this, k, ...args); }; }, { key, method });
}

test('configured catalog is metadata only with zero automatic selection or submissions', async ({ page }) => {
  const s = await fixture(page); await expect(submit(page)).toBeDisabled(); await expect(panel(page).getByText(/Synthetic capture/)).toHaveCount(2); await expect(panel(page).getByRole('checkbox', { checked: true })).toHaveCount(0); expect(s.posts).toEqual([]);
  await page.setViewportSize({ width: 320, height: 800 }); const [w, v] = await page.evaluate(() => [document.documentElement.scrollWidth, document.documentElement.clientWidth]); expect(w).toBeLessThanOrEqual(v + 1);
});
test('exact preparation submits once and minimal acknowledgement survives reload', async ({ page }) => {
  const s = await fixture(page); await selectEpisodes(page); await submit(page).dblclick(); await expect(panel(page).getByRole('article')).toHaveAttribute('data-job-id', 'new-preparation');
  expect(s.posts).toEqual([{ path: '/api/v1/projects/alpha/intakes', body: request() }]);
  await page.reload(); await page.getByRole('button', { name: 'Teaching', exact: true }).click(); await expect(panel(page).getByRole('article')).toHaveAttribute('data-job-id', 'new-preparation'); await expect(panel(page).getByRole('button', { name: 'Train on this dataset' })).toHaveCount(0); expect(s.posts).toHaveLength(1);
});
test('draft and selected job are project isolated and restore without consent', async ({ page }) => {
  const s = await fixture(page); await selectEpisodes(page); await page.getByLabel('Current project').selectOption('beta'); await expect(panel(page).getByRole('checkbox', { checked: true })).toHaveCount(0); await page.getByLabel('Current project').selectOption('alpha'); await expect(panel(page).getByRole('checkbox', { checked: true })).toHaveCount(2); await expect(submit(page)).toBeDisabled(); expect(s.posts).toEqual([]);
});
test('changed metadata before preparation refuses all POST and does not substitute bytes', async ({ page }) => {
  const s = await fixture(page); await selectEpisodes(page); s.captures.captures[0].session_sha256 = hex('f'); await submit(page).click(); await expect(panel(page).getByRole('alert').filter({ hasText: 'changed' })).toBeVisible(); expect(s.posts).toEqual([]);
});
for (const unavailable of ['unconfigured', 'empty', 'error']) test(`${unavailable} recording catalog has explicit state and no mutation`, async ({ page }) => {
  const s = await fixture(page); if (unavailable === 'unconfigured') s.configured = false; else if (unavailable === 'empty') s.captures.captures = []; else s.failCatalog = true; await refresh(page).click();
  await expect(panel(page).getByText(unavailable === 'unconfigured' ? /Remote Isaac captures/ : unavailable === 'empty' ? 'No finalized episodes are published for this project.' : /Recording availability is unverified/)).toBeVisible(); await expect(submit(page)).toBeDisabled(); expect(s.posts).toEqual([]);
});
for (const bad of ['array', 'foreign', 'recipe']) test(`${bad} invalid acknowledgement remains uncertain after reload`, async ({ page }) => {
  const s = await fixture(page); s.bad = bad; await selectEpisodes(page); await submit(page).click(); await expect(ack(page)).toBeDisabled(); await page.reload(); await page.getByRole('button', { name: 'Teaching', exact: true }).click(); await expect(ack(page)).toBeDisabled(); await expect(submit(page)).toBeDisabled(); expect(s.posts).toHaveLength(1);
});
test('lost response requires fresh successful review and does not silently retry', async ({ page }) => {
  const s = await fixture(page); s.lost = true; await selectEpisodes(page); await submit(page).click(); await expect(ack(page)).toBeDisabled(); s.failHistory = true; await refresh(page).click(); await expect(panel(page).getByText(/Saved preparation history is unavailable/)).toBeVisible(); await expect(ack(page)).toBeDisabled(); s.failHistory = false; await refresh(page).click(); await expect(ack(page)).toBeEnabled(); await ack(page).click(); await expect(submit(page)).toBeDisabled(); expect(s.posts).toHaveLength(1);
});
for (const failure of ['first pending write', 'journal read']) test(`${failure} latches admission off before POST until explicit recovery`, async ({ page }) => {
  const s = await fixture(page); await selectEpisodes(page); await storageFault(page, failure === 'journal read' ? 'job-attempt' : 'recording-preparation', failure === 'journal read' ? 'getItem' : 'setItem'); await submit(page).click(); await expect(panel(page).getByText('Recording session recovery is not ready. No preparation or cancellation can be submitted.')).toBeVisible(); await expect(submit(page)).toBeDisabled(); expect(s.posts).toEqual([]); await page.evaluate(() => (window as any).__restoreRecordingStorage()); await panel(page).getByRole('button', { name: 'Read session recovery again' }).click(); await expect(submit(page)).toBeDisabled(); await panel(page).getByRole('checkbox', { name: /I reviewed these exact/ }).check(); await expect(submit(page)).toBeEnabled();
});
test('known accepted ID stays visible when first receipt persistence fails and history is unavailable', async ({ page }) => {
  const s = await fixture(page); await selectEpisodes(page); await storageFault(page, 'recording-receipt', 'setItem'); s.failHistory = true; await submit(page).click(); await expect(panel(page).getByText(/Retained acknowledgement: job new-preparation/)).toBeVisible(); await expect(submit(page)).toBeDisabled(); expect(s.posts).toHaveLength(1);
});
for (const outcome of ['ack', 'error']) test(`old same-recipe ${outcome} after remount cannot replace a new pending request`, async ({ page }) => {
  const s = await fixture(page), first = deferred(); s.gate = first; s.lost = outcome === 'error'; s.nextId = 'old-job'; await selectEpisodes(page); await submit(page).click(); await expect.poll(() => s.posts.length).toBe(1);
  await page.getByRole('button', { name: 'Dataset', exact: true }).click(); await page.getByRole('button', { name: 'Teaching', exact: true }).click(); await refresh(page).click(); await ack(page).click();
  const second = deferred(); s.gate = second; s.lost = false; s.nextId = 'new-job'; await panel(page).getByRole('checkbox', { name: /I reviewed these exact/ }).check(); await submit(page).click(); await expect.poll(() => s.posts.length).toBe(2);
  const pending = await page.evaluate(() => sessionStorage.getItem('firebird:recording-preparation:alpha')); first.release(); await expect.poll(() => page.evaluate(() => sessionStorage.getItem('firebird:recording-preparation:alpha'))).toBe(pending); await expect(panel(page).getByText(/Retained acknowledgement: job old-job/)).toHaveCount(0);
  second.release(); await expect(panel(page).getByRole('article')).toHaveAttribute('data-job-id', 'new-job'); expect(s.posts).toHaveLength(2);
});
test('late accepted job never replaces a manual saved selection', async ({ page }) => {
  const s = await fixture(page), gate = deferred(); s.gate = gate; const saved = job('manual-job'); s.jobs.push(saved); await refresh(page).click(); await selectEpisodes(page); await submit(page).click(); await expect.poll(() => s.posts.length).toBe(1); await panel(page).getByLabel('Saved recording preparation', { exact: true }).selectOption(saved.id); gate.release(); await expect(panel(page).getByText(/Retained acknowledgement: job new-preparation/)).toBeVisible(); await expect(panel(page).getByRole('article')).toHaveAttribute('data-job-id', saved.id);
});
test('single-group result remains inspectable with explicit training limitation', async ({ page }) => {
  const s = await fixture(page); await openSaved(page, s, job('one-group', 'alpha', false)); await expect(panel(page).getByRole('button', { name: 'Train on this dataset' })).toBeDisabled(); await expect(panel(page).getByText(/one lineage group/)).toBeVisible(); await panel(page).getByRole('button', { name: 'View this dataset' }).click(); await expect(page.getByRole('heading', { name: 'Dataset', exact: true, level: 1 })).toBeVisible(); expect(s.posts).toEqual([]);
});
test('two-group exact training handoff refuses omitted history and preserves later manual choice', async ({ page }) => {
  const s = await fixture(page), chosen = await openSaved(page, s); s.jobs.push(job('alternative')); s.omit = chosen.id; const omitted = page.waitForResponse(r => new URL(r.url()).pathname === '/api/v1/projects/alpha/jobs'); await panel(page).getByRole('button', { name: 'Train on this dataset' }).click(); await omitted;
  await expect(page.getByRole('alert').filter({ hasText: `Selected dataset ${chosen.id} is unavailable` })).toBeVisible();
  await expect(page.locator('input[name="training-dataset"]:checked')).toHaveCount(0);
  // The native radio is visually hidden; activate its exact enclosing label as a user would.
  await page.locator('label').filter({ has: page.locator('input[name="training-dataset"][value="alternative"]') }).click();
  await expect(page.locator('input[name="training-dataset"]:checked')).toHaveValue('alternative');
  const restored = page.waitForResponse(r => new URL(r.url()).pathname === '/api/v1/projects/alpha/jobs'); s.omit = '';
  expect((await (await restored).json()).some((item: { id: string }) => item.id === chosen.id)).toBe(true);
  await expect(page.locator(`input[name="training-dataset"][value="${chosen.id}"]`)).toBeVisible();
  await expect(page.locator('input[name="training-dataset"]:checked')).toHaveValue('alternative'); expect(s.posts).toEqual([]);
});
test('View exact dataset refuses omitted history instead of showing another inspection', async ({ page }) => {
  const s = await fixture(page), chosen = await openSaved(page, s); s.jobs.push(job('alternative')); s.omit = chosen.id; const omitted = page.waitForResponse(r => new URL(r.url()).pathname === '/api/v1/projects/alpha/jobs'); await panel(page).getByRole('button', { name: 'View this dataset' }).click(); await omitted;
  await expect(page.getByRole('alert').filter({ hasText: `Selected inspection ${chosen.id} is unavailable` })).toBeVisible(); await expect(page.getByRole('button', { name: 'Train on this dataset', exact: true })).toHaveCount(0); expect(s.posts).toEqual([]);
});
test('cancellation takes fresh exact identity and selection change prevents POST', async ({ page }) => {
  const s = await fixture(page), running = await openSaved(page, s, job('running-job', 'alpha', true, 'running')); s.jobs.push(job('other-job')); await refresh(page).click(); await panel(page).getByRole('button', { name: 'Cancel selected preparation' }).click(); const gate = deferred(); s.getGate = gate; s.getGateId = running.id; await panel(page).getByRole('button', { name: 'Confirm preparation cancellation' }).click(); await panel(page).getByLabel('Saved recording preparation', { exact: true }).selectOption('other-job'); gate.release(); await expect(panel(page).getByRole('alert').filter({ hasText: 'Selection changed' })).toBeVisible(); expect(s.posts).toEqual([]);
});
test('lost cancellation preserves exact original job across reload and blocks duplicate cancellation', async ({ page }) => {
  const s = await fixture(page); await openSaved(page, s, job('running-job', 'alpha', true, 'running')); s.lost = true; await panel(page).getByRole('button', { name: 'Cancel selected preparation' }).click(); await panel(page).getByRole('button', { name: 'Confirm preparation cancellation' }).click(); await expect(ack(page)).toBeDisabled(); await page.reload(); await page.getByRole('button', { name: 'Teaching', exact: true }).click(); await expect(panel(page).getByRole('alert').filter({ hasText: 'Cancel job running-job' })).toBeVisible(); await refresh(page).click(); await expect(ack(page)).toBeDisabled(); expect(s.posts).toEqual([{ path: '/api/v1/jobs/running-job/cancel', body: {} }]);
});

test('Teaching draft is not remounted by recording project changes', async ({ page }) => {
  const s = await fixture(page);
  await page.route('**/api/v1/teaching/state', route => route.fulfill({ json: { connected: true, voice_configured: false, state: { session_id: 'generated-live-session', revision: 1, episode_id: null, mode: 'idle', instruction: '', joints: ['joint_one'], state_rad: [0], steps: 0, sim_time: 0, outcome: 'unknown', fault: null } } }));
  const instruction = page.getByRole('textbox', { name: 'Task instruction', exact: true }); await expect(instruction).toBeEnabled(); await instruction.fill('Unsubmitted live teaching draft');
  await page.getByLabel('Current project').selectOption('beta'); await expect(instruction).toHaveValue('Unsubmitted live teaching draft'); expect(s.posts).toEqual([]);
});

for (const action of ['submit', 'cancel'] as const) test(`post-ACK storage read denial retains exact ${action} identity and disables further mutation`, async ({ page }) => {
  const s = await fixture(page), gate = deferred(); s.gate = gate;
  if (action === 'submit') {
    await selectEpisodes(page); await submit(page).click();
  } else {
    await openSaved(page, s, job('cancel-target', 'alpha', true, 'running'));
    await panel(page).getByRole('button', { name: 'Cancel selected preparation' }).click();
    await panel(page).getByRole('button', { name: 'Confirm preparation cancellation' }).click();
  }
  await expect.poll(() => s.posts.length).toBe(1);
  const saved = await page.evaluate(() => sessionStorage.getItem('firebird:recording-preparation:alpha'));
  s.failHistory = true;
  await storageFault(page, 'recording-preparation', 'getItem');
  gate.release();
  const id = action === 'submit' ? 'new-preparation' : 'cancel-target';
  // Even the selected cancellation target must retain its new ACK status when history fails.
  await expect(panel(page).getByText(`Retained acknowledgement: job ${id} · ${action === 'submit' ? 'queued' : 'cancelled'}. Restore session recovery and inspect saved jobs to verify its current result.`)).toBeVisible();
  await expect(panel(page).getByText('Recording session recovery is not ready. No preparation or cancellation can be submitted.')).toBeVisible();
  await expect(submit(page)).toBeDisabled(); expect(s.posts).toHaveLength(1);
  await page.evaluate(() => (window as any).__restoreRecordingStorage());
  expect(await page.evaluate(() => sessionStorage.getItem('firebird:recording-preparation:alpha'))).toBe(saved);
});

test('post-ACK read failure after successor takeover preserves only the new known receipt', async ({ page }) => {
  const s = await fixture(page), first = deferred(); s.gate = first; s.nextId = 'old-job';
  await selectEpisodes(page); await submit(page).click(); await expect.poll(() => s.posts.length).toBe(1);
  await page.getByRole('button', { name: 'Dataset', exact: true }).click(); await page.getByRole('button', { name: 'Teaching', exact: true }).click();
  await refresh(page).click(); await ack(page).click();
  const second = deferred(); s.gate = second; s.nextId = 'new-job';
  await panel(page).getByRole('checkbox', { name: /I reviewed these exact/ }).check(); await submit(page).click();
  await expect.poll(() => s.posts.length).toBe(2);
  const saved = await page.evaluate(() => sessionStorage.getItem('firebird:recording-preparation:alpha'));
  s.failHistory = true; await storageFault(page, 'recording-preparation', 'getItem');
  const oldResponse = page.waitForResponse(r => new URL(r.url()).pathname === '/api/v1/projects/alpha/intakes' && r.request().method() === 'POST');
  first.release(); expect((await (await oldResponse).json()).id).toBe('old-job');
  second.release();
  await expect(panel(page).getByText(/Retained acknowledgement: job new-job/)).toBeVisible();
  await expect(panel(page).getByText(/Retained acknowledgement: job old-job/)).toHaveCount(0);
  await expect(submit(page)).toBeDisabled(); expect(s.posts).toHaveLength(2);
  await page.evaluate(() => (window as any).__restoreRecordingStorage());
  expect(await page.evaluate(() => sessionStorage.getItem('firebird:recording-preparation:alpha'))).toBe(saved);
});
