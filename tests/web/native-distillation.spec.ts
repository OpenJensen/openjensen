import { chooseTransformationModel, openTransformationJob, transformationJobs } from './lifecycle-controls';
import { selectProject } from './project-controls';
import { startStudent, cancelStudent } from '../../apps/web/src/lib/native-distillation';
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
  await page.goto('/datasets/'); await expect(page.getByLabel('Current project')).toHaveAttribute('data-project-id', 'alpha');
  await page.getByRole('link', { name: 'Distill', exact: true }).click();
  await chooseTransformationModel(page, 'distillation', 'Choose Imported ACT teacher · teacher:policy');
  await expect(page.getByRole('region', { name: 'ACT distillation', exact: true })).toBeVisible();
  return state;
}
const panel = (page: Page) => page.getByRole('region', { name: 'ACT distillation', exact: true });
const submit = (page: Page) => page.getByRole('button', { name: 'Train ACT256 student', exact: true });
const refresh = (page: Page) => page.getByRole('button', { name: 'Refresh distillation jobs' }).click();
async function prepare(page: Page) {
  await page.getByRole('group', { name: 'Teacher', exact: true }).getByRole('radio', { name: teacher.label, exact: true }).check();
  await page.getByRole('group', { name: 'Dataset', exact: true }).locator('input[value="data"]').check();
  await page.getByLabel('Training episodes', { exact: true }).fill('0, 1');
  await page.getByLabel('Validation episodes', { exact: true }).fill('2, 3');
  await page.getByLabel('Final episodes', { exact: true }).fill('4, 5');
  await page.getByLabel('Six coordinate units', { exact: true }).fill('degrees, degrees, degrees, degrees, degrees, recorded_gripper');
  await page.getByRole('checkbox', { name: /I verified that/ }).check();
}
test('owned local teacher, complete dataset, explicit splits and coordinate attestation are required', async ({ page }) => {
  const state = await fixture(page);
  await expect(submit(page)).toBeDisabled();
  await expect(page.getByRole('group', { name: 'Teacher', exact: true }).getByRole('radio')).toHaveCount(1);
  await expect(page.getByRole('group', { name: 'Dataset', exact: true }).getByRole('radio')).toHaveCount(1);
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
  await page.getByRole('link', { name: 'Dataset', exact: true }).click(); await page.getByRole('link', { name: 'Distill', exact: true }).click();
  await page.reload(); await page.getByRole('link', { name: 'Distill', exact: true }).click();
  await page.getByRole('button', { name: 'Review request', exact: true }).click();
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
  await expect(page.getByRole('article', { name: 'Distillation job details' })).toHaveAttribute('data-job-id', 'student-001');
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
  await openTransformationJob(page, 'distillation', 'student-002'); release();
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
  await selectProject(page, 'beta'); release();
  await expect(page.getByRole('article', { name: 'Distillation job details' })).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Distillation workspace', exact: true })).toBeVisible();
  await expect(panel(page)).toHaveCount(0);
  await expect(submit(page)).toHaveCount(0);
  await selectProject(page, 'alpha');
  await openTransformationJob(page, 'distillation', 'student-001');
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
  await chooseTransformationModel(page,'distillation','Choose Imported ACT teacher · teacher:policy');
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
  await page.getByRole('link', { name: 'Dataset', exact: true }).click(); release();
  await expect.poll(() => page.evaluate(() => (window as unknown as { journalCleanupFailures?: number }).journalCleanupFailures)).toBe(1);
  await page.getByRole('link', { name: 'Distill', exact: true }).click();
  await page.getByRole('button', { name: 'Review request', exact: true }).click();

  await expect(page.getByRole('button', { name: 'I checked recorded jobs; allow a new request' })).toBeVisible();
  await expect(page.getByText(/^Submitting one /)).toHaveCount(0);
  await expect(submit(page)).toBeDisabled();
  expect(state.posts).toHaveLength(1);
});


test('distillation mutation ACK status contract accepts only six exact strings', async () => {
  const request: Parameters<typeof startStudent>[1] = { operation: 'policy.distill', runtime_id: runtime.id, artifact_id: teacher.id, dataset_job_id: 'data', timeout_seconds: 600, native_distillation: { adapter: 'act-act-v1', student: 'act-256', steps: 100, learning_rate: .0001, seed: 1729, frame_stride: 30, splits: { train: [0], validation: [1], final: [2] }, coordinate_attestation: 'generated_fixture', units: ['degrees', 'degrees', 'degrees', 'degrees', 'degrees', 'recorded_gripper'] } };
  const valid = ['queued', 'running', 'succeeded', 'failed', 'cancelled', 'interrupted'];
  const original = { id: 'contract', project_id: 'alpha', kind: request.operation, status: 'running', request } as Parameters<typeof cancelStudent>[0];
  const realFetch = globalThis.fetch;
  let methods: string[] = [], status: unknown = 'running', cancelResponse = false;
  globalThis.fetch = async (_input, init) => {
    methods.push(init?.method ?? 'GET');
    return new Response(JSON.stringify({ ...original, status: cancelResponse && init?.method === 'GET' ? 'running' : status }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  };
  try {
    for (status of valid) { methods = []; expect((await startStudent('alpha', request)).status).toBe(status); expect(methods).toEqual(['POST']); }
    for (status of [['running'], ['succeeded'], [['running']], [], null, true, 1, {}, undefined, 'unknown']) {
      methods = []; await expect(startStudent('alpha', request)).rejects.toThrow(/outcome is unverified/); expect(methods).toEqual(['POST']);
      methods = []; await expect(cancelStudent(original, () => true)).rejects.toThrow(/outcome is unverified/); expect(methods).toEqual(['GET']);
    }
    cancelResponse = true; status = ['cancelled']; methods = [];
    await expect(cancelStudent(original, () => true)).rejects.toThrow(/outcome is unverified/); expect(methods).toEqual(['GET', 'POST']);
    status = 'cancelled'; methods = [];
    expect((await cancelStudent(original, () => true)).status).toBe('cancelled'); expect(methods).toEqual(['GET', 'POST']);
  } finally { globalThis.fetch = realFetch; }
});

for (const phase of ['submit', 'cancel-preflight', 'cancel-receipt'] as const) test(`distillation malformed ${phase} status is rejected without automatic retry`, async ({ page }) => {
  const state = await fixture(page); await prepare(page);
  if (phase === 'submit') {
    await page.route('**/api/v1/projects/alpha/policy-jobs', route => {
      const request = route.request().postDataJSON(); state.posts.push(request);
      return route.fulfill({ status: 202, json: { id: 'malformed', project_id: 'alpha', kind: request.operation, status: ['running'], request } });
    });
    await submit(page).click();
  } else {
    await submit(page).click();
    await expect(page.getByRole('article', { name: 'Distillation job details' })).toHaveAttribute('data-job-id', 'student-001');
    const original = state.jobs.find(item => item.id === 'student-001')!;
    if (phase === 'cancel-preflight') await page.route('**/api/v1/jobs/student-001', route => route.fulfill({ json: { ...original, status: ['running'] } }));
    else await page.route('**/api/v1/jobs/student-001/cancel', route => { state.cancel.push('student-001'); return route.fulfill({ json: { ...original, status: ['cancelled'] } }); });
    await page.getByRole('button', { name: 'Cancel selected distillation', exact: true }).click();
    await page.getByRole('button', { name: 'Confirm cancellation', exact: true }).click();
  }
  await expect(panel(page).getByRole('alert')).toContainText('outcome is unverified');
  expect(state.posts).toHaveLength(1);
  expect(state.cancel).toHaveLength(phase === 'cancel-receipt' ? 1 : 0);
  const details = page.getByRole('article', { name: 'Distillation job details' });
  if (phase === 'submit') {
    expect(await page.evaluate(() => sessionStorage.getItem('firebird:job-attempt:policy.distill:alpha'))).toContain('uncertain');
    await expect(details).toHaveCount(0);
  } else {
    await expect(details).toHaveAttribute('data-job-id', 'student-001');
    await expect(details.locator('.status')).toHaveText('running');
    await expect(page.getByRole('button', { name: 'Confirm cancellation', exact: true })).toHaveCount(0);
    const recovery = await page.evaluate(() => Object.keys(sessionStorage).filter(key => key.startsWith('firebird:cancel-attempt:')).map(key => JSON.parse(sessionStorage.getItem(key)!)));
    expect(recovery).toHaveLength(1); expect(recovery[0]).toMatchObject({ jobId: 'student-001', state: 'uncertain', action: 'cancel' });
  }
});

test('expanded training recipe keeps labels above full-width inputs with space between fields', async ({ page }) => {
  const state = await fixture(page); await prepare(page);
  await page.getByText('Training recipe', { exact: true }).click();
  const names = ['Training steps', 'Frame stride', 'Learning rate', 'Random seed', 'Distillation timeout (seconds)'];
  await expect(page.getByLabel(names[0], { exact: true })).toBeVisible();
  const fields = await Promise.all(names.map(name => page.getByLabel(name, { exact: true }).evaluate(element => {
    const input = element.getBoundingClientRect(), label = element.closest('label')!.getBoundingClientRect();
    return { labelTop: label.top, labelWidth: label.width, inputTop: input.top, inputBottom: input.bottom, inputWidth: input.width };
  })));
  for (const [index, field] of fields.entries()) {
    expect(field.inputTop - field.labelTop, `${names[index]} label above its input`).toBeGreaterThanOrEqual(12);
    expect(Math.abs(field.inputWidth - field.labelWidth), `${names[index]} uses its available field width`).toBeLessThanOrEqual(1);
    if (index) expect(field.labelTop - fields[index - 1].inputBottom, `space before ${names[index]}`).toBeGreaterThanOrEqual(15);
  }
  expect(state.posts).toEqual([]);
});

async function cancellationFixture(page: Page) {
  const state = await fixture(page); await prepare(page); await submit(page).click();
  await expect(page.getByRole('article', { name: 'Distillation job details' })).toHaveAttribute('data-job-id', 'student-001');
  return state;
}
async function confirmCancellation(page: Page) {
  await page.getByRole('button', { name: 'Cancel selected distillation', exact: true }).click();
  await page.getByRole('button', { name: 'Confirm cancellation', exact: true }).click();
}
async function reopenCancellationLane(page: Page) {
  await page.getByRole('link', { name: 'Distill', exact: true }).click();
  const history = page.getByRole('region', { name: 'Distillation jobs', exact: true });
  await expect(history).toBeVisible();
  if (await page.getByLabel('Current project').getAttribute('data-project-id') === 'alpha') {
    await history.locator('[data-job-id="student-001"]').click();
    await expect(panel(page)).toBeVisible();
  }
}
const cancellationRecovery = (page: Page) => page.getByRole('region', { name: 'Cancellation recovery', exact: true });
const acknowledgeCancellation = (page: Page) => page.getByRole('button', { name: 'I reviewed cancellation history; allow another cancellation', exact: true });

test('cancellation outcome survives stage navigation and reload without another POST', async ({ page }) => {
  const state = await cancellationFixture(page);
  await page.route('**/api/v1/jobs/student-001/cancel', async route => { state.cancel.push('student-001'); await route.abort('failed'); });
  await confirmCancellation(page);
  await expect(cancellationRecovery(page)).toContainText('Cancellation outcome is unverified for student-001');
  await page.getByRole('link', { name: 'Dataset', exact: true }).click(); await reopenCancellationLane(page);
  await expect(cancellationRecovery(page)).toContainText('student-001');
  await page.reload(); await reopenCancellationLane(page);
  await expect(cancellationRecovery(page)).toContainText('student-001');
  await expect(acknowledgeCancellation(page)).toBeDisabled();
  await page.getByRole('button', { name: 'Refresh distillation jobs', exact: true }).click();
  await expect(acknowledgeCancellation(page)).toBeDisabled();
  await page.getByRole('button', { name: 'Refresh cancellation history', exact: true }).click();
  await expect(acknowledgeCancellation(page)).toBeEnabled();
  expect(state.cancel).toEqual(['student-001']); expect(state.posts).toHaveLength(1);
});

for (const method of ['getItem', 'setItem'] as const) test(`cancellation ${method} denial blocks all cancellation I/O and remains latched`, async ({ page }) => {
  const state = await cancellationFixture(page); let reads = 0;
  await page.route('**/api/v1/jobs/student-001', route => { reads++; return route.fulfill({ json: state.jobs.find(item => item.id === 'student-001') }); });
  await page.evaluate(method => {
    const original = Storage.prototype[method];
    Object.defineProperty(Storage.prototype, method, { configurable: true, value: function (key: string, ...args: string[]) {
      if (key.startsWith('firebird:cancel-attempt:')) throw new DOMException('Cancellation journal denied', 'SecurityError');
      return Reflect.apply(original, this, [key, ...args]);
    } });
  }, method);
  await confirmCancellation(page);
  await expect(page.getByText(/Cancellation recovery storage is unavailable/)).toBeVisible();
  await expect(page.getByRole('button', { name: 'Cancel selected distillation', exact: true })).toBeDisabled();
  await page.getByRole('link', { name: 'Dataset', exact: true }).click(); await reopenCancellationLane(page);
  await openTransformationJob(page, 'distillation', 'student-001');
  await expect(page.getByText(/Cancellation recovery storage is unavailable/)).toBeVisible();
  await expect(page.getByRole('button', { name: 'Cancel selected distillation', exact: true })).toBeDisabled();
  expect(reads).toBe(0); expect(state.cancel).toEqual([]);
});

test('cancellation acknowledgment survives failed journal cleanup and failed history', async ({ page }) => {
  const state = await cancellationFixture(page); const original = state.jobs.find(item => item.id === 'student-001');
  await page.evaluate(() => { const original = Storage.prototype.removeItem; Storage.prototype.removeItem = function (key: string) { if (key.startsWith('firebird:cancel-attempt:')) throw new DOMException('Denied', 'SecurityError'); return original.call(this, key); }; });
  let outage = false;
  await page.route('**/api/v1/jobs/student-001/cancel', route => { state.cancel.push('student-001'); outage = true; return route.fulfill({ json: { ...original, status: 'cancelled', result: { reports: [{ quality_verified: true }] } } }); });
  await page.route('**/api/v1/projects/alpha/jobs', route => route.fulfill(outage ? { status: 503, json: { detail: 'History unavailable' } } : { json: state.jobs }));
  await confirmCancellation(page);
  await expect(page.getByText('Cancellation response for student-001: cancelled. Recorded job history remains authoritative.', { exact: true })).toBeVisible();
  await expect(page.getByText(/Cancellation recovery storage is unavailable/)).toBeVisible();
  await expect(cancellationRecovery(page)).toContainText('student-001');
  await expect(page.getByRole('button', { name: 'Cancel selected distillation', exact: true })).toBeDisabled();
  expect(state.cancel).toEqual(['student-001']);
  const stored = await page.evaluate(() => Object.keys(sessionStorage).filter(key => key.startsWith('firebird:cancel-attempt:')).map(key => sessionStorage.getItem(key)));
  expect(stored).toHaveLength(1); expect(stored[0]).not.toContain('reports'); expect(stored[0]).not.toContain('quality_verified');
});

test('validated cancellation acknowledgment survives a newly denied storage read', async ({ page }) => {
  const state = await cancellationFixture(page); const original = state.jobs.find(item => item.id === 'student-001');
  let storedBeforeDenial: string[] = [];
  await page.route('**/api/v1/jobs/student-001/cancel', async route => {
    state.cancel.push('student-001');
    storedBeforeDenial = await page.evaluate(() => {
      const original = Storage.prototype.getItem;
      const entries = Object.keys(sessionStorage).filter(key => key.startsWith('firebird:cancel-attempt:')).map(key => original.call(sessionStorage, key)!);
      Storage.prototype.getItem = function (key: string) { if (key.startsWith('firebird:cancel-attempt:')) throw new DOMException('Denied after acknowledgment', 'SecurityError'); return original.call(this, key); };
      return entries;
    });
    return route.fulfill({ json: { ...original, status: 'cancelled', result: { reports: [{ quality_verified: true }] } } });
  });
  await confirmCancellation(page);
  await expect(page.getByText('Cancellation response for student-001: cancelled. Recorded job history remains authoritative.', { exact: true })).toBeVisible();
  await expect(page.getByText(/Cancellation recovery storage is unavailable/)).toBeVisible();
  await expect(cancellationRecovery(page)).toContainText('student-001');
  await page.getByRole('link', { name: 'Dataset', exact: true }).click(); await reopenCancellationLane(page);
  await openTransformationJob(page, 'distillation', 'student-001');
  await expect(page.getByText('Cancellation response for student-001: cancelled. Recorded job history remains authoritative.', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Cancel selected distillation', exact: true })).toBeDisabled();
  expect(state.cancel).toEqual(['student-001']); expect(state.posts).toHaveLength(1);
  expect(storedBeforeDenial).toHaveLength(1); expect(JSON.parse(storedBeforeDenial[0])).toMatchObject({ jobId: 'student-001', state: 'pending' });
  expect(storedBeforeDenial[0]).not.toContain('reports'); expect(storedBeforeDenial[0]).not.toContain('quality_verified');
});

test('cancellation selection A to B to A invalidates the old preflight', async ({ page }) => {
  const state = await cancellationFixture(page); const original = state.jobs.find(item => item.id === 'student-001');
  state.jobs.push({ ...original, id: 'student-002' }); await page.getByRole('button', { name: 'Refresh distillation jobs', exact: true }).click();
  let reads = 0, release!: () => void; const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/api/v1/jobs/student-001', async route => { reads++; await gate; return route.fulfill({ json: original }); });
  await confirmCancellation(page); await expect.poll(() => reads).toBe(1);
  await openTransformationJob(page, 'distillation', 'student-002');
  await openTransformationJob(page, 'distillation', 'student-001'); release();
  await expect(cancellationRecovery(page)).toContainText('Cancellation outcome is unverified');
  expect(state.cancel).toEqual([]);
});

test('late cancellation receipt survives unmount without replacing a manual saved job', async ({ page }) => {
  const state = await cancellationFixture(page); const original = state.jobs.find(item => item.id === 'student-001');
  state.jobs.push({ ...original, id: 'student-002' }); await page.getByRole('button', { name: 'Refresh distillation jobs', exact: true }).click();
  let release!: () => void; const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/api/v1/jobs/student-001/cancel', async route => { state.cancel.push('student-001'); await gate; return route.fulfill({ json: { ...original, status: 'cancelled' } }); });
  await confirmCancellation(page); await expect.poll(() => state.cancel.length).toBe(1);
  await page.getByRole('link', { name: 'Dataset', exact: true }).click(); await reopenCancellationLane(page);
  await openTransformationJob(page, 'distillation', 'student-002'); release();
  await expect(page.getByText('Cancellation response for student-001: cancelled. Recorded job history remains authoritative.', { exact: true })).toBeVisible();
  await expect(page.getByRole('article', { name: 'Distillation job details' })).toHaveAttribute('data-job-id', 'student-002');
  expect(state.cancel).toEqual(['student-001']); expect(state.posts).toHaveLength(1);
});

test('cancellation recovery requires a fresh owned history read after uncertainty', async ({ page }) => {
  const state = await cancellationFixture(page);
  let releaseCancel!: () => void; const cancelGate = new Promise<void>(resolve => { releaseCancel = resolve; });
  await page.route('**/api/v1/jobs/student-001/cancel', async route => { state.cancel.push('student-001'); await cancelGate; return route.abort('failed'); });
  await confirmCancellation(page); await expect.poll(() => state.cancel.length).toBe(1);
  let reads = 0, releaseHistory!: () => void; const historyGate = new Promise<void>(resolve => { releaseHistory = resolve; });
  await page.route('**/api/v1/projects/alpha/jobs', async route => { const index = ++reads; if (index === 1) await historyGate; return route.fulfill({ json: index === 1 ? state.jobs : state.jobs.map(item => item.id === 'student-001' ? { ...item, status: 'cancelled' } : item) }); });
  // A normal shared history poll begins before the cancellation becomes uncertain.
  await expect.poll(() => reads).toBe(1);
  releaseCancel(); await expect(cancellationRecovery(page)).toContainText('Cancellation outcome is unverified');
  await expect(acknowledgeCancellation(page)).toBeDisabled();
  // The initial shared query remains unresolved; recovery must make its own GET.
  await page.getByRole('button', { name: 'Refresh cancellation history', exact: true }).click();
  await expect.poll(() => reads).toBeGreaterThanOrEqual(2); await expect(acknowledgeCancellation(page)).toBeEnabled();
  await expect(cancellationRecovery(page).getByRole('status')).toHaveText('Fresh cancellation history for student-001: cancelled. This is the status returned by your recovery refresh.');
  releaseHistory();
  await expect(cancellationRecovery(page).getByRole('status')).toContainText('student-001: cancelled');
  expect(state.cancel).toEqual(['student-001']);
});

test('cancellation review cannot authorize a changed selection generation', async ({ page }) => {
  const state = await cancellationFixture(page); const original = state.jobs.find(item => item.id === 'student-001');
  state.jobs.push({ ...original, id: 'student-002' }); await page.getByRole('button', { name: 'Refresh distillation jobs', exact: true }).click();
  await page.route('**/api/v1/jobs/student-001/cancel', async route => { state.cancel.push('student-001'); return route.abort('failed'); });
  await confirmCancellation(page); await expect(cancellationRecovery(page)).toContainText('Cancellation outcome is unverified');
  let release!: () => void; const gate = new Promise<void>(resolve => { release = resolve; }); let reads = 0;
  await page.route('**/api/v1/projects/alpha/jobs', async route => { reads++; await gate; return route.fulfill({ json: state.jobs }); });
  await page.getByRole('button', { name: 'Refresh cancellation history', exact: true }).click(); await expect.poll(() => reads).toBeGreaterThan(0);
  await openTransformationJob(page, 'distillation', 'student-002'); await openTransformationJob(page, 'distillation', 'student-001'); release();
  await expect(page.getByRole('button', { name: 'Refresh cancellation history', exact: true })).toBeEnabled();
  await expect(acknowledgeCancellation(page)).toBeDisabled(); expect(state.cancel).toEqual(['student-001']);
});

test('pending cancellation reload keeps the original unique attempt and never resends', async ({ page }) => {
  const state = await cancellationFixture(page);
  let release!: () => void; const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/api/v1/jobs/student-001/cancel', async route => { state.cancel.push('student-001'); await gate; try { await route.abort('failed'); } catch { /* The original page was explicitly reloaded. */ } });
  await confirmCancellation(page); await expect.poll(() => state.cancel.length).toBe(1);
  const before = await page.evaluate(() => JSON.parse(sessionStorage.getItem(Object.keys(sessionStorage).find(key => key.startsWith('firebird:cancel-attempt:'))!)!));
  expect(before.state).toBe('pending'); expect(before.jobId).toBe('student-001');
  await page.reload(); await reopenCancellationLane(page);
  await expect(cancellationRecovery(page)).toContainText('Cancellation outcome is unverified for student-001');
  const after = await page.evaluate(() => JSON.parse(sessionStorage.getItem(Object.keys(sessionStorage).find(key => key.startsWith('firebird:cancel-attempt:'))!)!));
  expect(after).toEqual({ ...before, state: 'uncertain' });
  await expect(acknowledgeCancellation(page)).toBeDisabled(); release(); expect(state.cancel).toEqual(['student-001']);
});

test('cancellation review rejects changed original identity and survives another project', async ({ page }) => {
  const state = await cancellationFixture(page); const original = state.jobs.find(item => item.id === 'student-001');
  await page.route('**/api/v1/jobs/student-001/cancel', async route => { state.cancel.push('student-001'); return route.abort('failed'); });
  await confirmCancellation(page); await expect(cancellationRecovery(page)).toContainText('student-001');
  await page.route('**/api/v1/projects/alpha/jobs', route => route.fulfill({ json: [{ ...original, request: { ...original.request, artifact_id: 'changed-source' } }] }));
  await page.getByRole('button', { name: 'Refresh cancellation history', exact: true }).click();
  await expect(page.getByText(/Fresh history did not verify the original cancellation target/)).toBeVisible();
  await expect(acknowledgeCancellation(page)).toBeDisabled();
  await page.route('**/api/v1/projects', route => route.fulfill({ json: [{ id: 'alpha', name: 'Original project', created_at: time }, { id: 'beta', name: 'Another project', created_at: time }] }));
  await page.reload(); await reopenCancellationLane(page);
  await selectProject(page, 'beta'); await reopenCancellationLane(page);
  await expect(cancellationRecovery(page)).toHaveCount(0);
  await selectProject(page, 'alpha'); await reopenCancellationLane(page);
  await expect(cancellationRecovery(page)).toContainText('student-001');
  await expect(acknowledgeCancellation(page)).toBeDisabled(); expect(state.cancel).toEqual(['student-001']);
});


test('a later history failure disables cancellation recovery acknowledgment', async ({ page }) => {
  const state = await cancellationFixture(page);
  await page.route('**/api/v1/jobs/student-001/cancel', async route => { state.cancel.push('student-001'); return route.abort('failed'); });
  await confirmCancellation(page); await expect(cancellationRecovery(page)).toContainText('student-001');
  await page.getByRole('button', { name: 'Refresh cancellation history', exact: true }).click();
  await expect(acknowledgeCancellation(page)).toBeEnabled();
  await page.route('**/api/v1/projects/alpha/jobs', route => route.fulfill({ status: 503, json: { detail: 'Current history unavailable' } }));
  await page.getByRole('button', { name: 'Refresh distillation jobs', exact: true }).click();
  await expect(page.getByText(/Job updates are unavailable/)).toBeVisible();
  await expect(acknowledgeCancellation(page)).toBeDisabled(); expect(state.cancel).toEqual(['student-001']);
});
