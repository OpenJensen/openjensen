import { expect, test, type Page } from '@playwright/test';

const time = '2026-09-27T10:30:00Z';
const runtime = { id: 'student-cpu', label: 'Local ACT student', execution: 'native', provider: 'local', device: 'cpu', enabled: true, native_distillation: true, native_distillation_only: true, training: false, simulation: false, run: false, engine_evaluation: false };
const teacher = { id: 'teacher:policy', project_id: 'alpha', job_id: 'teacher', label: 'Imported ACT teacher', format: 'native_checkpoint', path: 'owned', manifest_sha256: 'a'.repeat(64), file_bytes: 10000, metadata: { architecture: 'act', use_vae: true } };
function dataset(id = 'data', snapshot = true) { return { id, project_id: 'alpha', kind: 'dataset.inspect', status: 'succeeded', created_at: time, updated_at: time, request: { source: 'local', path: 'fixture' }, result: { source: 'local', format: 'lerobot_v3', repo_id: null, revision: 'metadata-sha256:' + 'c'.repeat(64), metadata_sha256: 'c'.repeat(64), fps: 30, robot_type: 'generated-so101', license: null, inspected_at: time, warnings: [], total_episodes: 6, total_frames: 24, inspection_scope: snapshot ? 'complete_snapshot' : 'metadata_only', features: { 'observation.state': { names: ['j0', 'j1', 'j2', 'j3', 'j4', 'gripper'] }, action: { names: ['j0', 'j1', 'j2', 'j3', 'j4', 'gripper'] } }, ...(snapshot ? { snapshot: { id: 'sha256:' + 'b'.repeat(64), manifest_sha256: 'b'.repeat(64), lineage_validated: true, total_episodes: 6 } } : {}) } }; }
async function fixture(page: Page) {
  const state = { workers: [runtime] as Record<string, unknown>[], jobs: [dataset(), dataset('metadata-only', false)] as Record<string, any>[], posts: [] as Record<string, any>[], cancel: [] as string[], outcome: 'ok', outage: false, postGate: null as Promise<void> | null, cancelGate: null as Promise<void> | null };
  await page.route('**/api/v1/**', async route => {
    const req = route.request(), path = new URL(req.url()).pathname;
    if (path === '/api/v1/projects') return route.fulfill({ json: [{ id: 'alpha', name: 'Local student fixture', created_at: time }, { id: 'beta', name: 'Second project', created_at: time }] });
    if (path === '/api/v1/policy-options') return route.fulfill({ json: { runtimes: state.workers, sources: [], training_models: [], training_methods: [], default_training_method: 'lora', quantization_defaults: { cpu: { language: 'Q8_0', vision: null }, cuda: { language: 'Q8_0', vision: null }, note: '' } } });
    if (path.endsWith('/artifacts')) return route.fulfill({ json: path.includes('/alpha/') ? [teacher, { ...teacher, id: 'wrong-project', project_id: 'beta' }, { ...teacher, id: 'remote', metadata: { architecture: 'act', storage: 'gcs' } }, { ...teacher, id: 'packed', format: 'native_quantized' }] : [] });
    if (path.endsWith('/policy-jobs') && req.method() === 'POST') {
      const body = req.postDataJSON(); state.posts.push(body);
      if (state.outcome === 'reject') return route.fulfill({ status: 422, json: { detail: 'Lineage group spans training and final partitions.' } });
      if (state.postGate) await state.postGate;
      const created = { id: 'student-001', project_id: 'alpha', kind: 'policy.distill', status: 'running', stage: 'preparing', request: body, created_at: time, updated_at: time, result: null };
      state.jobs.push(created);
      if (state.outcome === 'lost') return route.abort('failed');
      if (state.outcome === 'wrong-teacher') return route.fulfill({ status: 202, json: { ...created, request: { ...body, artifact_id: 'other' } } });
      return route.fulfill({ status: 202, json: created });
    }
    if (path.endsWith('/cancel') && req.method() === 'POST') { const id = path.split('/').at(-2)!; state.cancel.push(id); const item = state.jobs.find(item => item.id === id)!; item.status = 'cancelled'; if (state.cancelGate) await state.cancelGate; return route.fulfill({ json: item }); }
    if (path.endsWith('/jobs') && path.includes('/projects/')) return route.fulfill(state.outage ? { status: 503, json: { detail: 'History temporarily unavailable' } } : { json: path.includes('/alpha/') ? state.jobs : [] });
    if (path.endsWith('/events')) return route.fulfill({ json: [{ sequence: 1, timestamp: time, stage: 'preparing', message: 'Verifying explicit partitions', data: {} }] });
    if (path.startsWith('/api/v1/jobs/')) { const item = state.jobs.find(item => item.id === path.split('/').at(-1)); if (item) return route.fulfill({ json: item }); }
    if (req.method() !== 'GET') return route.fulfill({ status: 405, json: { detail: 'Unexpected mutation' } });
    return route.continue();
  });
  await page.goto('/'); await expect(page.getByLabel('Current project')).toHaveValue('alpha');
  await page.getByRole('button', { name: 'Distill', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Teach a smaller ACT policy' })).toBeVisible();
  return state;
}
const panel = (page: Page) => page.getByRole('region', { name: 'ACT distillation', exact: true });
const submit = (page: Page) => page.getByRole('button', { name: 'Train ACT256 student', exact: true });
const refresh = (page: Page) => page.getByRole('button', { name: 'Refresh distillation jobs' }).click();
async function prepare(page: Page) {
  await page.getByLabel('ACT teacher', { exact: true }).selectOption(teacher.id);
  await page.getByLabel('Verified dataset', { exact: true }).selectOption('data');
  await page.getByLabel('Training episodes', { exact: true }).fill('0, 1');
  await page.getByLabel('Validation episodes', { exact: true }).fill('2, 3');
  await page.getByLabel('Final episodes', { exact: true }).fill('4, 5');
  await page.getByLabel('Six coordinate units', { exact: true }).fill('degrees, degrees, degrees, degrees, degrees, recorded_gripper');
  await page.getByRole('checkbox', { name: /I verified that/ }).check();
}
test('owned local teacher, complete dataset, explicit splits and coordinate attestation are required', async ({ page }) => {
  const state = await fixture(page);
  await expect(submit(page)).toBeDisabled();
  await expect(page.getByLabel('ACT teacher', { exact: true }).locator('option')).toHaveCount(2);
  await expect(page.getByLabel('Verified dataset', { exact: true }).locator('option')).toHaveCount(2);
  await prepare(page); await expect(submit(page)).toBeEnabled();
  await page.getByLabel('Final episodes', { exact: true }).fill('0, 5'); await expect(submit(page)).toBeDisabled();
  await page.getByLabel('Final episodes', { exact: true }).fill('4, 6'); await expect(submit(page)).toBeDisabled();
  await page.getByLabel('Final episodes', { exact: true }).fill('4, 5');
  await page.getByLabel('Six coordinate units', { exact: true }).fill('radians'); await expect(submit(page)).toBeDisabled();
  expect(state.posts).toEqual([]);
});
test('submits one complete local recipe, shows progress and cancels only the selected job', async ({ page }) => {
  const state = await fixture(page); await prepare(page); await submit(page).dblclick();
  await expect(page.getByRole('article', { name: 'Distillation job details' })).toHaveAttribute('data-job-id', 'student-001');
  expect(state.posts).toHaveLength(1); expect(state.posts[0]).toEqual({ operation: 'policy.distill', runtime_id: runtime.id, artifact_id: teacher.id, dataset_job_id: 'data', timeout_seconds: 600, native_distillation: { adapter: 'act-act-v1', student: 'act-256', steps: 100, learning_rate: .0001, seed: 1729, frame_stride: 30, splits: { train: [0, 1], validation: [2, 3], final: [4, 5] }, coordinate_attestation: 'teacher_recorded_coordinates', units: ['degrees', 'degrees', 'degrees', 'degrees', 'degrees', 'recorded_gripper'] } });
  await page.getByRole('button', { name: 'Cancel selected distillation', exact: true }).click(); expect(state.cancel).toEqual([]);
  await page.getByRole('button', { name: 'Confirm cancellation', exact: true }).click();
  await expect(page.getByRole('article', { name: 'Distillation job details' })).toContainText('cancelled'); expect(state.cancel).toEqual(['student-001']);
});
for (const outcome of ['lost', 'wrong-teacher']) test(`${outcome} does not retry and survives stage navigation until history review`, async ({ page }) => {
  const state = await fixture(page); state.outcome = outcome; await prepare(page); await submit(page).click();
  await expect(page.getByText(/request outcome is unverified/).first()).toBeVisible(); expect(state.posts).toHaveLength(1);
  await page.getByRole('button', { name: 'Dataset', exact: true }).click(); await page.getByRole('button', { name: 'Distill', exact: true }).click();
  await page.reload(); await page.getByRole('button', { name: 'Distill', exact: true }).click();
  await expect(submit(page)).toBeDisabled();
  const resume = page.getByRole('button', { name: 'I checked recorded jobs; allow a new request' }); await expect(resume).toBeDisabled();
  await refresh(page); await expect(resume).toBeEnabled(); expect(state.posts).toHaveLength(1);
});
test('clear admission failures stay actionable and missing workers never enable a launch', async ({ page }) => {
  const state = await fixture(page); state.outcome = 'reject'; await prepare(page); await submit(page).click();
  await expect(panel(page).getByRole('alert')).toContainText('Lineage group spans'); await expect(submit(page)).toBeEnabled();
  state.workers = []; await refresh(page); await expect(submit(page)).toBeDisabled();
  await expect(page.getByText(/No local ACT distillation worker/)).toBeVisible(); expect(state.posts).toHaveLength(1);
});
test('completed generated evidence has scoped metrics and missing report never implies quality', async ({ page }, testInfo) => {
  const state = await fixture(page); await prepare(page); await page.getByRole('checkbox', { name: 'This snapshot contains generated test observations.' }).check(); await page.getByRole('checkbox', { name: /I verified that/ }).check(); await submit(page).click();
  const saved = state.jobs.find(item => item.id === 'student-001')!;
  saved.status = 'succeeded'; saved.stage = 'distilling'; saved.result = { artifacts: [{ ...teacher, id: 'student-001:distilled', job_id: 'student-001', label: 'ACT256 student', metadata: { architecture: 'act', recipe: 'act-action-distillation-v1' } }], reports: [{ operation: 'policy.distill', adapter: 'act-act-v1', teacher_artifact_id: teacher.id, dataset_job_id: 'data', steps: 100, fresh_reload_verified: true, quality_verified: false, calibration_verified: false, speedup_verified: false, task_success: null, dataset_kind: 'generated_fixture', student_weights_bytes: 55971416, teacher_inference_tensor_bytes: 136972568, trained_student: { validation: { teacher_normalized_l1: .33 }, final: { teacher_normalized_l1: .39 } } }] };
  await refresh(page); await expect(page.getByText('Generated observations · software verification only')).toBeVisible();
  await expect(page.getByRole('article', { name: 'Distillation job details' }).locator('.native-result-summary')).toHaveText('Recorded job · student-001');
  await expect(page.getByRole('link', { name: 'Download tested student package' })).toHaveAttribute('href', /student-001%3Adistilled\/download$/);
  await expect(page.getByText(/do not prove task success/)).toBeVisible();
  const widths = await page.evaluate(() => ({ inner: document.documentElement.clientWidth, scroll: document.documentElement.scrollWidth })); expect(widths.scroll).toBeLessThanOrEqual(widths.inner + 1);
  await page.screenshot({ path: testInfo.outputPath('distillation-result.png'), fullPage: true });
  saved.result.reports[0].dataset_kind = 'lerobot'; await refresh(page);
  await expect(page.getByText('Generated observations · software verification only')).toHaveCount(0);
  await expect(panel(page).getByRole('alert')).toContainText('Complete distillation measurements are unavailable');
  saved.result.reports = []; await refresh(page); await expect(panel(page).getByRole('alert')).toContainText('Complete distillation measurements are unavailable');
});


test('an earlier history refresh cannot authorize retry of a later ambiguous request', async ({ page }) => {
  const state = await fixture(page); await prepare(page);
  let release!: () => void; state.postGate = new Promise<void>(resolve => { release = resolve; }); state.outcome = 'lost';
  await submit(page).click(); await expect.poll(() => state.posts.length).toBe(1);
  await refresh(page); await expect(page.getByRole('button', { name: 'Refresh distillation jobs' })).toBeEnabled(); release();
  const acknowledgement = page.getByRole('button', { name: 'I checked recorded jobs; allow a new request' });
  await expect(acknowledgement).toBeDisabled(); await refresh(page); await expect(acknowledgement).toBeEnabled();
  expect(state.posts).toHaveLength(1);
});

test('a delayed cancellation never steals the newly selected job', async ({ page }) => {
  const state = await fixture(page); await prepare(page); await submit(page).click();
  await expect(page.getByRole('article', { name: 'Distillation job details' })).toHaveAttribute('data-job-id', 'student-001');
  state.jobs.push({ ...state.jobs.find(item => item.id === 'student-001'), id: 'student-002' }); await refresh(page);
  let release!: () => void; state.cancelGate = new Promise<void>(resolve => { release = resolve; });
  await page.getByRole('button', { name: 'Cancel selected distillation', exact: true }).click();
  await page.getByRole('button', { name: 'Confirm cancellation', exact: true }).click();
  await expect.poll(() => state.cancel.length).toBe(1);
  await page.getByLabel('Saved distillation job', { exact: true }).selectOption('student-002'); release();
  await expect(page.getByRole('article', { name: 'Distillation job details' })).toHaveAttribute('data-job-id', 'student-002');
  await expect(page.getByRole('button', { name: 'Cancel selected distillation', exact: true })).toBeEnabled();
  expect(state.cancel).toEqual(['student-001']);
});

test('stale history and disabled local compute cannot launch or cancel', async ({ page }) => {
  const state = await fixture(page); await prepare(page);
  state.workers = [{ ...runtime, launchable: false }]; await refresh(page); await expect(submit(page)).toBeDisabled();
  state.workers = [runtime]; await refresh(page); await expect(submit(page)).toBeEnabled();
  state.outage = true; await refresh(page); await expect(submit(page)).toBeDisabled();
  state.outage = false; await refresh(page); await expect(submit(page)).toBeEnabled(); await submit(page).click();
  await expect(page.getByRole('article', { name: 'Distillation job details' })).toHaveAttribute('data-job-id', 'student-001');
  state.outage = true; await refresh(page); await expect(page.getByRole('button', { name: 'Cancel selected distillation', exact: true })).toBeDisabled();
  expect(state.posts).toHaveLength(1); expect(state.cancel).toHaveLength(0);
});

test('switching project during admission never displays another project job', async ({ page }) => {
  const state = await fixture(page); await prepare(page);
  let release!: () => void; state.postGate = new Promise<void>(resolve => { release = resolve; });
  await submit(page).click(); await expect.poll(() => state.posts.length).toBe(1);
  await page.getByLabel('Current project').selectOption('beta'); release();
  await expect(page.getByRole('article', { name: 'Distillation job details' })).toHaveCount(0);
  await expect(submit(page)).toBeDisabled();
  await page.getByLabel('Current project').selectOption('alpha'); await refresh(page);
  await page.getByLabel('Saved distillation job', { exact: true }).selectOption('student-001');
  await expect(page.getByRole('article', { name: 'Distillation job details' })).toHaveAttribute('data-job-id', 'student-001');
  expect(state.posts).toHaveLength(1);
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
  await page.getByRole('button', { name: 'Refresh distillation jobs' }).click();
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
  await expect(page.getByRole('article', { name: 'Distillation job details' })).toHaveAttribute('data-job-id', 'student-001');
  expect(state.posts).toHaveLength(1);
  await page.getByText('Prepare another student', { exact: true }).click();
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
  await page.getByRole('button', { name: 'Distill', exact: true }).click();

  await expect(page.getByRole('button', { name: 'I checked recorded jobs; allow a new request' })).toBeVisible();
  await expect(page.getByText(/^Submitting one /)).toHaveCount(0);
  await expect(submit(page)).toBeDisabled();
  expect(state.posts).toHaveLength(1);
});
