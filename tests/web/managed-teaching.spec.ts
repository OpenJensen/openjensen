import { selectProject } from './project-controls';
import { expect, test, type Page } from '@playwright/test';
import { createHash } from 'node:crypto';

// Browser fixtures prove request and identity boundaries, not an Isaac runtime.
const stamp = '2026-09-28T00:00:00Z', hex = (c: string, size = 64) => c.repeat(size);
const recipe = () => ({ operation: 'teaching.capture', profile_id: 'local-isaac', profile_sha256: hex('a'), timeout_seconds: 300 });
const profiles = () => ({ configured: true, available: true, message: 'Generated configured profile; readiness is checked after start.', profiles: [{ id: 'local-isaac', label: 'Local Isaac fixture', profile_sha256: hex('a'), max_seconds: 300, max_capture_bytes: 1048576, runtime_verified: false, transport: 'local_owned_process' }] });
const episode = () => ({ episode_id: hex('1', 32), receipt_sha256: hex('2'), frames: 3, termination: 'finish', outcome: 'unknown' });
const capture = () => ({ session_id: hex('b', 32), session_sha256: hex('c'), origin: 'recorded', lineage_group: 'one-session', controller: 'joint_position_targets', state_units: 'radians', action_units: 'radians', timebase: 'simulation_seconds', camera_key: 'observation.images.front', joint_names: ['joint'], width: 32, height: 32, fps: 10, physics_hz: 60, scene_sha256: hex('f'), camera_prim: '/World/Camera', content_verified: false, episodes: [episode()] });
const publication = () => ({ schema_version: 1, operation: 'teaching.capture', profile_id: 'local-isaac', profile_sha256: hex('a'), session_id: hex('b', 32), session_sha256: hex('c'), inventory_sha256: hex('d'), recording_configuration_sha256: hex('e'), episodes: [episode()], origin: 'recorded', lineage_group: 'one-session', published: true, content_verified: true, process_cleanup_verified: true, simulator_coordinates: true, physical_calibration_verified: false, task_success_claimed: false });
const job = (id = 'capture-a', project = 'alpha', status = 'running'): any => ({ id, project_id: project, kind: 'teaching.capture', status, request: recipe(), result: status === 'succeeded' ? publication() : null, error: null, stage: status === 'succeeded' ? 'published' : 'teaching', created_at: stamp, updated_at: stamp, compute_target: null, simulation_target: null });
const executorState = () => ({ mode: 'running', episode_id: hex('1', 32), revision: 4, session_id: hex('b', 32), instruction: 'Teach one motion', outcome: 'unknown', steps: 3, sim_time: .3, joints: ['joint'], state_rad: [.1], fault: null });
const managed = (page: Page) => page.getByRole('region', { name: 'Managed teaching sessions', exact: true });
const recordings = (page: Page) => page.getByRole('region', { name: 'Recording dataset preparation', exact: true });
const start = (page: Page) => managed(page).getByRole('button', { name: 'Start teaching session', exact: true });
const check = (page: Page) => managed(page).getByRole('button', { name: 'Check saved teaching request', exact: true });
const retry = (page: Page) => managed(page).getByRole('button', { name: 'Retry same teaching request', exact: true });
function deferred() { let release!: () => void; const promise = new Promise<void>(resolve => { release = resolve; }); return { promise, release }; }
function canonical(value: any): any { return Array.isArray(value) ? value.map(canonical) : value && typeof value === 'object' ? Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])])) : value; }
function prepared(body: any) {
  return { id: 'prepared-a', project_id: 'alpha', kind: 'dataset.inspect', status: 'succeeded', request: body, created_at: stamp, updated_at: stamp, error: null, stage: 'completed', compute_target: null, simulation_target: null,
    result: { schema_version: 1, source: 'local', repo_id: null, revision: `metadata-sha256:${hex('f')}`, format: 'lerobot_v3', robot_type: 'generated', total_frames: 3, total_episodes: 1, fps: 10, features: { 'observation.state': { dtype: 'float32', shape: [1] }, action: { dtype: 'float32', shape: [1] } }, license: null, metadata_sha256: hex('f'), inspected_at: stamp, warnings: [], inspection_scope: 'complete_snapshot', snapshot: { schema_version: 1, id: `sha256:${hex('c')}`, manifest_sha256: hex('c'), format: 'lerobot_v3', total_bytes: 1000, file_count: 5, total_frames: 3, total_episodes: 1, lineage_validated: true, warnings: [] }, recording_preparation: { job_id: 'prepared-a', source_count: 1, lineage_group_count: 1, writer_readback_verified: true, source_preserved: true, task_success_verified: false, selection_sha256: createHash('sha256').update(JSON.stringify(canonical(body.recordings))).digest('hex') } } };
}
async function enterManaged(page: Page) {
  await page.getByRole('button', { name: 'Teaching', exact: true }).click();
  await page.getByRole('group', { name: 'Teaching mode', exact: true }).getByRole('button', { name: 'Managed session', exact: true }).click();
  await expect(managed(page)).toBeVisible();
}
async function fixture(page: Page, initial: any[] = []) {
  const s = { options: profiles(), jobs: initial, catalog: { configuration_sha256: hex('e'), message: 'Finalized fixture metadata only', captures: [capture()] }, bindings: new Map<string, any>(), posts: [] as { path: string; body: any; key: string }[], manual: [] as string[], unexpected: [] as string[], failHistory: false, failStatus: false, lostStart: false, commitLost: false, lostStop: false, support: true, stopping: new Set<string>(), startGate: null as ReturnType<typeof deferred> | null };
  await page.route('**/api/v1/**', async route => {
    const request = route.request(), path = new URL(request.url()).pathname, project = path.split('/')[4];
    if (path.startsWith('/api/v1/teaching/')) {
      s.manual.push(path);
      if (path.endsWith('/state')) return route.fulfill({ json: { connected: false, state: null, message: 'Manual fixture is disconnected', voice_configured: false } });
      if (path.endsWith('/settings')) return route.fulfill({ json: { saved: false, revision: null, voice_model: null, message: 'Not configured' } });
      return route.fulfill({ json: { broker_reachable: false, configured: false, configuration_present: false, dependencies_present: false, busy: false, message: 'Not configured' } });
    }
    if (path.includes('/submissions/')) {
      const key = path.split('/').at(-1)!, saved = s.bindings.get(key);
      return route.fulfill({ status: saved ? 200 : 404, json: saved ?? { detail: 'No saved binding' }, headers: s.support ? { 'Idempotency-Key': key, 'Cache-Control': 'no-store' } : {} });
    }
    if (request.method() === 'POST') {
      const body = request.postDataJSON(), key = request.headers()['idempotency-key'] ?? '';
      s.posts.push({ path, body, key });
      const headers = key ? { 'Idempotency-Key': key, 'Cache-Control': 'no-store' } : {};
      if (path.endsWith('/teaching/sessions')) {
        const saved = s.bindings.get(key) ?? { ...job('capture-a', project), request: body };
        if (s.startGate) await s.startGate.promise;
        if (!s.lostStart || s.commitLost) { s.bindings.set(key, saved); if (!s.jobs.some(j => j.id === saved.id)) s.jobs.push(saved); }
        if (s.lostStart) return route.abort('failed');
        return route.fulfill({ status: 202, json: saved, headers });
      }
      if (path.endsWith('/stop')) {
        const id = path.split('/').at(-2)!, saved = s.jobs.find(j => j.id === id && j.project_id === project);
        s.stopping.add(id);
        if (s.lostStop) return route.abort('failed');
        return route.fulfill({ json: { job: saved, ready: false, session_id: hex('b', 32), stop_requested: true } });
      }
      if (path.endsWith('/commands')) return route.fulfill({ status: 202, json: { command_id: body.command_id, status: 'queued' } });
      if (path.endsWith('/intakes')) { const saved = prepared(body); s.bindings.set(key, saved); s.jobs.push(saved); return route.fulfill({ status: 202, json: saved, headers }); }
      s.unexpected.push(`${request.method()} ${path}`); return route.fulfill({ status: 405, json: { detail: 'Unexpected mutation' } });
    }
    if (path.endsWith('/teaching/profiles')) return route.fulfill({ json: s.options });
    if (path.includes('/teaching/sessions/')) {
      const id = path.split('/')[7], saved = s.jobs.find(j => j.id === id && j.project_id === project);
      if (path.endsWith('/frame')) return route.fulfill({ status: 503, json: { detail: 'No generated image in this request fixture' } });
      if (path.endsWith('/state')) return route.fulfill({ json: executorState() });
      if (path.includes('/commands/')) return route.fulfill({ json: { command_id: path.split('/').at(-1), status: 'acknowledged' } });
      return route.fulfill({ status: s.failStatus ? 503 : saved ? 200 : 404, json: s.failStatus ? { detail: 'Status unavailable' } : !saved ? { detail: 'Missing session' } : { job: saved, ready: saved.status === 'running' && !s.stopping.has(id), session_id: saved.status === 'interrupted' ? null : hex('b', 32), stop_requested: s.stopping.has(id) } });
    }
    if (path.endsWith('/recordings/options')) return route.fulfill({ json: { configured: true, runtime_verified: false, configuration_sha256: hex('e'), max_episodes: 100, max_source_bytes: 8589934592, setup_message: 'Published fixture metadata; writer verification is required.' } });
    if (path.endsWith('/recordings')) return route.fulfill({ json: s.catalog });
    if (path.endsWith('/jobs') && path.includes('/projects/')) return route.fulfill({ status: s.failHistory ? 503 : 200, json: s.failHistory ? { detail: 'History unavailable' } : s.jobs.filter(j => j.project_id === project) });
    if (path.startsWith('/api/v1/jobs/')) { const saved = s.jobs.find(j => j.id === path.split('/')[4]); if (path.endsWith('/events')) return route.fulfill({ json: [] }); return route.fulfill({ status: saved ? 200 : 404, json: saved ?? { detail: 'Missing job' } }); }
    const replies: Record<string, unknown> = { '/api/v1/health': { status: 'ok', version: 'fixture' }, '/api/v1/projects': ['alpha', 'beta'].map(id => ({ id, name: `Teaching ${id}`, created_at: stamp })), '/api/v1/capabilities': [], '/api/v1/policy-options': { runtimes: [], sources: [], training_models: [], training_methods: [], default_training_method: 'full' } };
    if (path in replies) return route.fulfill({ json: replies[path] });
    if (path.endsWith('/artifacts')) return route.fulfill({ json: [] });
    s.unexpected.push(`${request.method()} ${path}`); return route.fulfill({ status: 404, json: { detail: 'Unexpected fixture read' } });
  });
  await page.goto('/'); await expect(page.getByLabel('Current project')).toHaveAttribute('data-project-id', 'alpha'); await enterManaged(page);
  await expect(managed(page).getByText(s.options.message)).toBeVisible();
  return s;
}
async function reviewStart(page: Page) {
  await managed(page).getByLabel('Teaching profile', { exact: true }).selectOption('local-isaac');
  await managed(page).getByRole('checkbox', { name: /Start this configured session/ }).check();
  await expect(start(page)).toBeEnabled();
}
async function chooseSaved(page: Page, id = 'capture-a') { await managed(page).getByLabel('Saved teaching session', { exact: true }).selectOption(id); }

test('managed start is explicit and scoped controls never call manual voice or intelligence', async ({ page }) => {
  const s = await fixture(page); await expect(start(page)).toBeDisabled(); expect(s.posts).toEqual([]);
  s.manual.length = 0; await reviewStart(page); await start(page).click();
  const controls = managed(page).getByRole('region', { name: 'Teaching controls', exact: true });
  await expect(controls.getByRole('button', { name: '+ 0.05 rad', exact: true })).toBeEnabled();
  await controls.getByRole('button', { name: '+ 0.05 rad', exact: true }).click(); await expect(controls.getByText('Command: acknowledged', { exact: true })).toBeVisible();
  await controls.getByRole('button', { name: 'Finish episode', exact: true }).click(); await expect.poll(() => s.posts.filter(p => p.path.endsWith('/commands')).length).toBe(2);
  expect(s.posts[0]).toMatchObject({ path: '/api/v1/projects/alpha/teaching/sessions', body: recipe() }); expect(s.posts[0].key).toMatch(/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/);
  expect(s.posts[1].body).toMatchObject({ operation: 'correct', session_id: hex('b', 32), expected_revision: 4 }); expect(s.posts[2].body.operation).toBe('finish');
  expect(s.posts.some(p => p.path.endsWith('/stop') || p.path.endsWith('/intakes'))).toBe(false); expect(s.manual).toEqual([]); expect(s.unexpected).toEqual([]);
  await expect(controls.getByRole('button', { name: 'Connect voice', exact: true })).toHaveCount(0);
});
test('unavailable configured profiles never pretend runtime readiness', async ({ page }) => {
  const s = await fixture(page); s.options.available = false;
  await managed(page).getByRole('button', { name: 'Refresh teaching sessions', exact: true }).click();
  await expect(managed(page).getByLabel('Teaching profile', { exact: true })).toBeDisabled(); await expect(start(page)).toBeDisabled(); expect(s.posts).toEqual([]);
});
test('lost start retains exact key through reload; 404 permits only explicit same-request retry', async ({ page }) => {
  const s = await fixture(page); s.lostStart = true; await reviewStart(page); await start(page).click(); await expect(check(page)).toBeEnabled();
  const original = structuredClone(s.posts[0]); await page.reload(); await enterManaged(page); await expect(check(page)).toBeEnabled(); expect(s.posts).toHaveLength(1);
  await check(page).click(); await expect(retry(page)).toBeEnabled(); await expect(start(page)).toBeDisabled(); expect(s.posts).toHaveLength(1);
  s.lostStart = false; await retry(page).click(); await expect(managed(page).getByLabel('Saved teaching session', { exact: true })).toHaveValue('capture-a');
  expect(s.posts).toEqual([original, original]);
});
test('completed reconciliation retains minimal accepted identity when history and status fail after remount', async ({ page }) => {
  const s = await fixture(page); s.lostStart = true; s.commitLost = true; await reviewStart(page); await start(page).click(); await expect(check(page)).toBeEnabled();
  const saved = job('capture-a', 'alpha', 'succeeded'); s.bindings.set(s.posts[0].key, saved); s.jobs = [saved]; s.failHistory = true; s.failStatus = true;
  await check(page).click(); await expect(managed(page).getByLabel('Saved teaching session', { exact: true })).toHaveValue('capture-a');
  await page.getByRole('button', { name: 'Dataset', exact: true }).click(); await enterManaged(page);
  await expect(managed(page).getByText(/Acknowledgement retained for job capture-a/)).toBeVisible(); await chooseSaved(page);
  await expect(managed(page).getByText(/selected session is unavailable or changed/)).toBeVisible();
  await expect(managed(page).getByRole('button', { name: 'Review for dataset preparation', exact: true })).toHaveCount(0); await expect(managed(page).getByRole('region', { name: 'Teaching controls', exact: true })).toHaveCount(0); expect(s.posts).toHaveLength(1);
});
test('denied durable journal storage prevents the first start request', async ({ page }) => {
  const s = await fixture(page); await reviewStart(page);
  await page.evaluate(() => { const original = Storage.prototype.setItem; Storage.prototype.setItem = function (key, value) { if (this === sessionStorage && key.startsWith('firebird:submission:v1:')) throw new DOMException('Generated write denial'); return original.call(this, key, value); }; });
  await start(page).click(); await expect(start(page)).toBeDisabled(); await expect(managed(page).getByRole('alert').filter({ hasText: /storage|recovery/i })).toBeVisible(); expect(s.posts).toEqual([]);
});
test('late start acknowledgement never moves a different project or stops a session on navigation', async ({ page }) => {
  const s = await fixture(page); const gate = deferred(); s.startGate = gate; await reviewStart(page); await start(page).click(); await expect.poll(() => s.posts.length).toBe(1);
  await expect(managed(page)).toHaveCount(1); await expect(recordings(page)).toHaveCount(1);
  await selectProject(page, 'beta'); await enterManaged(page);
  await expect(managed(page)).toHaveCount(1); await expect(recordings(page)).toHaveCount(1);
  await expect(managed(page).getByLabel('Saved teaching session', { exact: true })).toHaveValue(''); gate.release();
  await expect.poll(() => s.bindings.size).toBe(1);
  await expect(managed(page)).toHaveCount(1); await expect(recordings(page)).toHaveCount(1);
  await expect(managed(page).getByLabel('Saved teaching session', { exact: true })).toHaveValue(''); expect(s.posts).toHaveLength(1);
  await selectProject(page, 'alpha'); await enterManaged(page);
  await expect(managed(page)).toHaveCount(1); await expect(recordings(page)).toHaveCount(1);
  await chooseSaved(page); await expect(managed(page).getByRole('button', { name: 'Stop and publish', exact: true })).toBeVisible(); expect(s.posts).toHaveLength(1);
});
test('stop ambiguity and remount retain saved stopping state without claiming publication or reposting', async ({ page }) => {
  const s = await fixture(page, [job()]); await chooseSaved(page); s.lostStop = true;
  await managed(page).getByRole('button', { name: 'Stop and publish', exact: true }).click(); expect(s.posts).toEqual([]);
  await managed(page).getByRole('button', { name: 'Confirm stop and publish', exact: true }).click(); await expect(managed(page).getByText(/Stop outcome is unverified/)).toBeVisible();
  await page.reload(); await enterManaged(page); await chooseSaved(page); await expect(managed(page).getByText(/Publication is not confirmed yet/)).toBeVisible();
  await expect(managed(page).getByRole('button', { name: 'Review for dataset preparation', exact: true })).toHaveCount(0); expect(s.posts).toEqual([{ path: '/api/v1/projects/alpha/teaching/sessions/capture-a/stop', body: {}, key: '' }]);
});
test('interrupted history is visible but never offers controls or published handoff', async ({ page }) => {
  const s = await fixture(page, [job('capture-a', 'alpha', 'interrupted')]); await chooseSaved(page);
  await expect(managed(page).getByText(/Capture publication is not confirmed/)).toBeVisible(); await expect(managed(page).getByRole('region', { name: 'Teaching controls', exact: true })).toHaveCount(0); expect(s.posts).toEqual([]);
});
test('verified publication requires a separate exact recording selection and preparation consent', async ({ page }) => {
  const s = await fixture(page, [job('capture-a', 'alpha', 'succeeded')]); await chooseSaved(page);
  await managed(page).getByRole('button', { name: 'Review for dataset preparation', exact: true }).click();
  const notice = recordings(page).getByRole('complementary', { name: 'Published teaching capture', exact: true }); await expect(notice).toBeVisible();
  await expect(recordings(page).getByRole('checkbox', { checked: true })).toHaveCount(0); expect(s.posts).toEqual([]);
  await notice.getByRole('button', { name: 'Select these recorded episodes', exact: true }).click();
  await expect(recordings(page).getByRole('checkbox', { name: `Episode ${hex('1', 32)} · 3 frames · outcome unknown`, exact: true })).toBeChecked();
  const consent = recordings(page).getByRole('checkbox', { name: /I reviewed these exact/ }); await expect(consent).not.toBeChecked(); await expect(recordings(page).getByRole('button', { name: 'Prepare selected recordings', exact: true })).toBeDisabled(); expect(s.posts).toEqual([]);
  await consent.check(); await recordings(page).getByRole('button', { name: 'Prepare selected recordings', exact: true }).click();
  await expect(recordings(page).getByText(/This dataset has one lineage group/)).toBeVisible(); await expect(recordings(page).getByRole('button', { name: 'Train on this dataset', exact: true })).toBeDisabled();
  expect(s.posts).toHaveLength(1); expect(s.posts[0].path).toBe('/api/v1/projects/alpha/intakes'); expect(s.posts[0].body.recordings.captures).toEqual([{ session_id: hex('b', 32), session_sha256: hex('c'), episodes: [{ episode_id: hex('1', 32), receipt_sha256: hex('2') }] }]);
});
test('changed published catalog refuses handoff without substituting episodes or starting preparation', async ({ page }) => {
  const s = await fixture(page, [job('capture-a', 'alpha', 'succeeded')]); await chooseSaved(page);
  await managed(page).getByRole('button', { name: 'Review for dataset preparation', exact: true }).click(); s.catalog.captures[0].session_sha256 = hex('f');
  await recordings(page).getByRole('button', { name: 'Select these recorded episodes', exact: true }).click();
  await expect(recordings(page).getByRole('alert').filter({ hasText: /no longer matches/ })).toBeVisible(); await expect(recordings(page).getByRole('checkbox', { checked: true })).toHaveCount(0); expect(s.posts).toEqual([]);
});
