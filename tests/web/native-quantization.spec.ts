import { chooseTransformationModel, openTransformationJob, transformationJobs } from './lifecycle-controls';
import { selectProject } from './project-controls';
import { startNativeQuantization, cancelNativeQuantization } from '../../apps/web/src/lib/native-quantization';
import { expect, test, type Page } from '@playwright/test';
import { measuredNativeReport, replayableNativeOutput, type PackedArtifact } from '../../apps/web/src/lib/native-quantization';
import type { Job } from '../../apps/web/src/lib/api';

const timestamp = '2026-09-27T12:00:00Z';
const runtime = { id: 'act-cpu', label: 'Generated CPU quantizer', device: 'cpu', provider: 'local', execution: 'native', enabled: true, launchable: true, native_quantization: true, native_quantization_only: true, training: false, simulation: false, engine_evaluation: false, run: false };
const engine = { id: 'engine', label: 'Existing Smol engine', device: 'cpu', provider: 'local', execution: 'native', enabled: true, training: false, simulation: false, engine_evaluation: true, run: true };
function artifact(id: string, format = 'inference_export', metadata: Record<string, unknown> = {}, project = 'alpha') {
  return { id, project_id: project, job_id: metadata.architecture === 'smolvla' ? 'smol-source' : 'source', label: `Generated ${id}`, format, path: 'fixture', manifest_sha256: 'a'.repeat(64), file_bytes: 4096, parent_ids: [], metadata: { architecture: 'act', inference_only: true, ...metadata } };
}
function job(id: string, status = 'running', bits = 8, source = 'act-export') {
  return { id, project_id: 'alpha', kind: 'policy.quantize', status, stage: 'quantizing', created_at: timestamp, updated_at: timestamp, error: null as string | null, result: null as Record<string, unknown> | null,
    request: { operation: 'policy.quantize', runtime_id: runtime.id, artifact_id: source, native_quantization: { format: 'firebird_quant', bits, group_size: 64 }, timeout_seconds: 600 } };
}
function report() {
  return { stage: 'operation', operation: 'policy.quantize', architecture: 'act', format: 'firebird_quant', format_version: 1, inference_only: true, training_resume_supported: false, precision: 'int8', model_id: 'sha256:' + 'b'.repeat(64), source_artifact_id: 'act-export', source_artifact_manifest_sha256: 'a'.repeat(64), source_weight_bytes: 136991488, packed_weight_bytes: 36694284, policy_package_bytes: 36700000,
    fresh_reload_verified: true, cpu_reload_verified: true, runtime_verified: false, isaac_runtime_verified: false, quality_verified: false, calibration_verified: false, speedup_verified: false, task_success: null, gpu_memory_bytes: null, inference_speedup: null,
    drift_from_fp32: [171, 902].map(seed => ({ seed, input_sha256: String(seed === 171 ? 'c' : 'd').repeat(64), raw: { rmse: .0035, maximum_absolute_difference: .0065, coordinates: 600 }, postprocessed: { rmse: .09482, maximum_absolute_difference: .141285, coordinates: 600 } })) };
}
async function fixture(page: Page) {
  const state = { runtimes: [runtime, engine] as Record<string, unknown>[], optionsError: false, jobsError: false, artifactsError: false, submit: 'ok', postGate: null as Promise<void> | null, cancelGate: null as Promise<void> | null, posts: [] as Record<string, any>[], jobs: [] as Record<string, any>[], events: [] as string[], cancels: [] as string[], gets: [] as string[],
    artifacts: [artifact('act-export'), artifact('act-native', 'native_checkpoint'), artifact('smol', 'native_checkpoint', { architecture: 'smolvla' }), artifact('training', 'training_checkpoint', { method: 'full' }), artifact('wrong-project', 'inference_export', {}, 'beta'), artifact('remote', 'inference_export', { storage: 'gcs', remote_uri: 'gs://fixture' }), artifact('packed', 'native_quantized'), artifact('vae', 'native_checkpoint', { use_vae: true }), artifact('full', 'native_checkpoint', { method: 'full', training_backend: 'lerobot' })] };
  await page.route('**/api/v1/**', async route => {
    const req = route.request(), path = new URL(req.url()).pathname;
    if (req.method() === 'POST') {
      if (path.endsWith('/policy-jobs')) {
        const body = req.postDataJSON(); state.posts.push(body);
        const created = job('submitted', 'running', body.native_quantization.bits, body.artifact_id); created.request = body;
        if (state.submit === 'reject') return route.fulfill({ status: 422, json: { detail: 'ACT source still contains training-only VAE weights; create an inference export first.' } });
        if (state.postGate) await state.postGate;
        state.jobs.unshift(created);
        if (state.submit === 'lost') return route.abort('failed');
        if (state.submit === 'bad-json') return route.fulfill({ status: 202, contentType: 'application/json', body: '{' });
        if (state.submit === 'wrong-project') return route.fulfill({ status: 202, json: { ...created, project_id: 'beta' } });
        if (state.submit === 'wrong-source') return route.fulfill({ status: 202, json: { ...created, request: { ...body, artifact_id: 'other' } } });
        if (state.submit === 'wrong-bits') return route.fulfill({ status: 202, json: { ...created, request: { ...body, native_quantization: { ...body.native_quantization, bits: 4 } } } });
        return route.fulfill({ status: 202, json: created });
      }
      if (path.endsWith('/cancel')) { const id = path.split('/').at(-2)!; state.cancels.push(id); const current = state.jobs.find(item => item.id === id)!; current.status = 'cancelled'; if (state.cancelGate) await state.cancelGate; return route.fulfill({ json: current }); }
      return route.fulfill({ status: 405, json: { detail: 'Unexpected fixture mutation' } });
    }
    if (path === '/api/v1/projects') return route.fulfill({ json: [{ id: 'alpha', name: 'Generated ACT quantization', created_at: timestamp }, { id: 'beta', name: 'Second project', created_at: timestamp }] });
    if (path === '/api/v1/policy-options') return route.fulfill(state.optionsError ? { status: 503, json: { detail: 'Generated capability outage' } } : { json: { runtimes: state.runtimes, sources: [], training_models: [], training_methods: [], default_training_method: 'lora', quantization_defaults: { cpu: { language: 'Q8_0', vision: null }, cuda: { language: 'Q8_0', vision: null }, note: '' } } });
    if (path.endsWith('/jobs') && path.includes('/projects/')) return route.fulfill(state.jobsError ? { status: 503, json: { detail: 'Generated history outage' } } : { json: path.includes('/alpha/') ? state.jobs : [] });
    if (path.endsWith('/artifacts')) return route.fulfill(state.artifactsError ? { status: 503, json: { detail: 'Generated artifact outage' } } : { json: path.includes('/alpha/') ? state.artifacts : [] });
    if (path.endsWith('/events')) { const id = path.split('/').at(-2)!; state.events.push(id); return route.fulfill({ json: [{ sequence: 1, timestamp, stage: 'quantizing', message: `Generated activity for ${id}`, data: {} }] }); }
    if (path.startsWith('/api/v1/jobs/')) { const id = path.split('/').at(-1)!; state.gets.push(id); const current = state.jobs.find(item => item.id === id); if (current) return route.fulfill({ json: current }); }
    return route.continue();
  });
  await page.goto('/datasets/'); await expect(page.getByLabel('Current project')).toHaveAttribute('data-project-id', 'alpha');
  await page.getByRole('link', { name: 'Quantize', exact: true }).click();
  await chooseTransformationModel(page, 'quantization', 'Choose Generated act-export · act-export');
  await expect(page.getByRole('region', { name: 'Native ACT quantization', exact: true })).toBeVisible();
  return state;
}
async function choosePolicy(page: Page, id: string) {
  const checkpoint = page.getByLabel('Checkpoint', { exact: true });
  if (await checkpoint.count()) await checkpoint.selectOption(id);
  else await page.getByRole('group', { name: 'Policy', exact: true }).locator(`input[value="${id}"]`).check();
}
async function expectPolicy(page: Page, id: string) {
  await expect.poll(async () => {
    const checkpoint = page.getByLabel('Checkpoint', { exact: true });
    if (await checkpoint.count()) return await checkpoint.inputValue();
    const checked = page.getByRole('group', { name: 'Policy', exact: true }).locator('input:checked');
    return await checked.count() ? await checked.inputValue() : '';
  }).toBe(id);
}
const submit = (page: Page) => page.getByRole('button', { name: 'Create ACT quantized package', exact: true });
const refresh = (page: Page) => page.getByRole('button', { name: 'Refresh ACT quantization jobs', exact: true }).click();

test('capability is required and native-only workers cannot leak into the existing engine workflow', async ({ page }) => {
  const state = await fixture(page); state.runtimes = [{ ...runtime, native_quantization: false }, engine]; await refresh(page);
  await expect(page.getByText('No local ACT quantization worker is configured.', { exact: false })).toBeVisible();
  await expect(submit(page)).toBeDisabled();
  await chooseTransformationModel(page, 'quantization', 'Choose Generated smol · smol');
  await expect(page.getByRole('group', { name: 'My model', exact: true })).toBeVisible();
  await expect(page.getByRole('group', { name: 'Compute', exact: true }).locator('input[value="act-cpu"]')).toHaveCount(0);
  await expect(page.getByRole('group', { name: 'Compute', exact: true }).locator('input[value="engine"]')).toHaveCount(1);
  expect(state.posts).toEqual([]);
});

for (const bits of [8, 4]) test(`INT${bits} requires an owned compatible source and submits exact bounded payload only once`, async ({ page }) => {
  const state = await fixture(page); const picker = page.getByRole('group', { name: 'Policy', exact: true });
  await expect(picker.getByRole('radio')).toHaveCount(0);
  await expect(picker.getByLabel('Checkpoint').locator('option:enabled')).toHaveCount(2);
  await expect(picker.getByLabel('Checkpoint').locator('option[value="smol"]')).toHaveCount(0);
  await expect(page.getByRole('group', { name: 'Compression', exact: true }).locator('input:checked')).toHaveValue('8'); await expect(submit(page)).toBeEnabled();
  await choosePolicy(page, 'act-export'); if (bits === 4) await page.getByRole('group', { name: 'Compression', exact: true }).locator('input[value="4"]').check();
  await submit(page).dblclick();
  await expect(page.getByRole('article', { name: 'ACT quantization job details' })).toHaveAttribute('data-job-id', 'submitted');
  expect(state.posts).toEqual([{ operation: 'policy.quantize', runtime_id: 'act-cpu', artifact_id: 'act-export', native_quantization: { format: 'firebird_quant', bits, group_size: 64 }, timeout_seconds: 600 }]);
  await page.getByText('Activity and recorded report', { exact: true }).click();
  await expect(page.getByRole('region', { name: 'ACT quantization event log' })).toContainText('Generated activity for submitted');
  await chooseTransformationModel(page, 'quantization', 'Choose Generated smol · smol');
  await expect(page.getByLabel('Checkpoint',{exact:true})).toHaveValue('smol');
  await expect(page.getByRole('article', { name: 'ACT quantization job details' })).toHaveCount(0);
  expect(state.posts).toHaveLength(1);
});

test('admission rejection remains visible and does not retry or imply conversion', async ({ page }) => {
  const state = await fixture(page); state.submit = 'reject'; await choosePolicy(page, 'act-export'); await submit(page).click();
  await expect(page.getByRole('region', { name: 'Native ACT quantization', exact: true }).getByRole('alert')).toContainText('create an inference export first'); await expect(submit(page)).toBeEnabled();
  await expect(page.getByRole('article', { name: 'ACT quantization job details' })).toHaveCount(0); expect(state.posts).toHaveLength(1);
});

for (const outcome of ['lost', 'bad-json', 'wrong-project', 'wrong-source', 'wrong-bits']) test(`${outcome} response pauses further submissions until an explicit history review`, async ({ page }) => {
  const state = await fixture(page); state.submit = outcome; await choosePolicy(page, 'act-export'); await submit(page).click();
  await expect(page.getByRole('region', { name: 'Native ACT quantization', exact: true }).getByRole('alert')).toContainText('submission outcome is unverified'); await expect(submit(page)).toBeDisabled();
  const allow = page.getByRole('button', { name: 'I checked the jobs; allow a new request' }); await expect(allow).toBeDisabled();
  await refresh(page); await expect(allow).toBeEnabled(); expect(state.posts).toHaveLength(1);
  await openTransformationJob(page, 'quantization', 'submitted'); await expect(page.getByRole('article', { name: 'ACT quantization job details' })).toBeVisible();
  await refresh(page); await expect(allow).toBeEnabled(); await allow.click(); await chooseTransformationModel(page,'quantization','Choose Generated act-export · act-export'); await expect(submit(page)).toBeEnabled(); expect(state.posts).toHaveLength(1);
});

test('completed candidate shows measured storage and drift separately from exact reload with download only', async ({ page }, testInfo) => {
  const state = await fixture(page); const done = job('complete', 'succeeded'); done.result = { reports: [report()], artifacts: [{ ...artifact('packed-result', 'native_quantized'), job_id: done.id }, { ...artifact('other-project', 'native_quantized', {}, 'beta'), job_id: done.id }] }; state.jobs.push(done); state.artifacts.push(artifact('packed-result', 'native_quantized'));
  await refresh(page); await openTransformationJob(page, 'quantization', done.id);
  const measured = page.getByRole('region', { name: 'Measured ACT quantization results' });
  await expect(measured).toContainText('136,991,488 bytes'); await expect(measured).toContainText('36,694,284 bytes');
  const comparison = measured.getByRole('region', { name: 'Comparison with original model', exact: true });
  await expect(comparison.getByRole('img', { name: 'Action difference from original model on fixed inputs' })).toBeVisible();
  await expect(comparison).toContainText('0.0035');
  await expect(comparison).toContainText('Original FP32');
  await expect(comparison).toContainText('2 fixed inputs');
  await page.getByText('Action differences', { exact: true }).focus(); await page.keyboard.press('Enter');
  await expect(page.getByRole('region', { name: 'FP32 action differences', exact: true })).toBeVisible();
  await expect(measured).toContainText('Exact agreement with the packed candidate'); await expect(measured).toContainText('0.0948200');
  await expect(measured).toContainText('physical units are unverified');
  await expect(page.getByRole('link', { name: 'Download INT8 package' })).toHaveCount(1);
  await expect(page.getByRole('link', { name: 'Download INT8 package' })).toHaveAttribute('href', '/api/v1/projects/alpha/artifacts/packed-result/download');
  await expect(page.getByRole('group',{name:'ACT quantization setup',exact:true})).toHaveCount(0);
  await expect(page.getByText('Prepare another ACT candidate',{exact:true})).toHaveCount(0);
  await expect(submit(page)).toHaveCount(0);
  await page.screenshot({ path: testInfo.outputPath('quantization-result.png'), fullPage: true });
  await expect(page.getByRole('group', { name: 'Policy', exact: true }).locator('input[value="packed-result"]')).toHaveCount(0);
  await page.setViewportSize({ width: 320, height: 900 }); const widths = await page.evaluate(() => ({ width: document.documentElement.clientWidth, scroll: document.documentElement.scrollWidth, offenders: [...document.querySelectorAll('body *')].filter(item => item.getBoundingClientRect().right > document.documentElement.clientWidth + 1).slice(-15).map(item => ({ tag: item.tagName, cls: item.className, right: item.getBoundingClientRect().right })) })); expect(widths.scroll, JSON.stringify(widths)).toBeLessThanOrEqual(widths.width + 1);
  await page.getByRole('link', { name: 'Run', exact: true }).click(); await page.getByRole('button', { name: '3D simulation', exact: true }).click();
  await expect(page.locator('.native-simulation').locator('input[type=radio][value="packed-result"]')).toHaveCount(0); expect(state.posts).toEqual([]);
});

test('missing or incomplete report never displays inferred reload or numeric acceptance', async ({ page }) => {
  const state = await fixture(page); const done = job('incomplete', 'succeeded'); done.result = { reports: [{ ...report(), drift_from_fp32: [] }], artifacts: [] }; state.jobs.push(done);
  await refresh(page); await openTransformationJob(page, 'quantization', done.id);
  await expect(page.getByRole('region', { name: 'Native ACT quantization', exact: true }).getByRole('alert')).toContainText('complete measured quantization report is unavailable');
  await expect(page.getByRole('region', { name: 'Measured ACT quantization results' })).toHaveCount(0);
  await expect(page.getByRole('link', { name: 'Download INT8 package' })).toHaveCount(0);
});

test('cancellation requires confirmation bound to the freshly read selected job', async ({ page }) => {
  const state = await fixture(page); state.jobs.push(job('one'), job('two')); await refresh(page); await openTransformationJob(page, 'quantization', 'one');
  await page.getByRole('button', { name: 'Cancel selected ACT quantization' }).click(); expect(state.cancels).toEqual([]);
  await openTransformationJob(page, 'quantization', 'two'); await expect(page.getByRole('group', { name: 'Confirm ACT quantization cancellation' })).toHaveCount(0);
  await page.getByRole('button', { name: 'Cancel selected ACT quantization' }).click(); await page.getByRole('button', { name: 'Confirm cancellation', exact: true }).click();
  await expect(page.getByRole('article', { name: 'ACT quantization job details' })).toContainText('cancelled'); expect(state.gets).toContain('two'); expect(state.cancels).toEqual(['two']);
});

test('selection changing during cancellation preflight does not cancel the previous job', async ({ page }) => {
  const state = await fixture(page); state.jobs.push(job('one'), job('two')); await refresh(page); await openTransformationJob(page, 'quantization', 'one');
  let entered!: () => void, release!: () => void; const enteredPromise = new Promise<void>(resolve => { entered = resolve; }); const barrier = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/api/v1/jobs/one', async route => { entered(); await barrier; await route.fulfill({ json: state.jobs.find(item => item.id === 'one') }); });
  await page.getByRole('button', { name: 'Cancel selected ACT quantization' }).click(); await page.getByRole('button', { name: 'Confirm cancellation', exact: true }).click(); await enteredPromise;
  await openTransformationJob(page, 'quantization', 'two');
  const response = page.waitForResponse('**/api/v1/jobs/one'); release(); await response;
  await expect(page.getByRole('article', { name: 'ACT quantization job details' })).toHaveAttribute('data-job-id', 'two');
  await expect(page.getByRole('button', { name: 'Cancel selected ACT quantization' })).toBeEnabled();
  expect(state.cancels).toEqual([]);
});

test('project switching resets precision/source and failed history blocks mutation', async ({ page }) => {
  const state = await fixture(page); await choosePolicy(page, 'act-export'); await page.getByRole('group', { name: 'Compression', exact: true }).locator('input[value="4"]').check();
  state.jobsError = true; await refresh(page); await expect(page.getByText(/Job updates are unavailable/)).toBeVisible(); await expect(submit(page)).toBeDisabled();
  state.jobsError = false; await selectProject(page, 'beta');
  await page.getByRole('button', { name: 'Start a new quantization', exact: true }).click();
  await expect(page.getByText('No saved models in this project yet')).toBeVisible();
  await expect(page.getByRole('group', { name: 'Policy', exact: true })).toHaveCount(0);
  await expect(submit(page)).toHaveCount(0); expect(state.posts).toEqual([]);
});

test('disabled runtime and timeout bounds block submission without changing precision or source', async ({ page }) => {
  const state = await fixture(page); await choosePolicy(page, 'act-export');
  for (const seconds of ['29', '601', '30.5', '']) { await page.getByLabel('Quantization timeout (seconds)', { exact: true }).fill(seconds); await expect(submit(page)).toBeDisabled(); }
  await page.getByLabel('Quantization timeout (seconds)', { exact: true }).fill('30'); await expect(submit(page)).toBeEnabled();
  state.runtimes = [{ ...runtime, enabled: false }, engine]; await refresh(page); await expect(page.getByText(/configured ACT quantization worker is unavailable/)).toBeVisible(); await expect(submit(page)).toBeDisabled(); expect(state.posts).toEqual([]);
});

test('a fast terminal transition refetches the final events after an empty active response', async ({ page }) => {
  const state = await fixture(page); const quick = job('quick', 'queued'); state.jobs.push(quick);
  let armed = false, terminal = false, finalReads = 0;
  let emptyRead!: () => void; const empty = new Promise<void>(resolve => { emptyRead = resolve; });
  await page.route('**/api/v1/jobs/quick/events', async route => {
    if (terminal) { finalReads++; return route.fulfill({ json: [{ sequence: 1, timestamp, stage: 'verifying', message: 'Generated final packed reload receipt', data: {} }] }); }
    await route.fulfill({ json: [] }); if (armed) emptyRead();
  });
  await page.route('**/api/v1/projects/alpha/jobs', async route => {
    if (armed) { await empty; terminal = true; quick.status = 'succeeded'; }
    await route.fulfill({ json: state.jobs });
  });
  await refresh(page); await openTransformationJob(page, 'quantization', 'quick');
  await page.getByText('Activity and recorded report', { exact: true }).click();
  await expect(page.getByRole('region', { name: 'ACT quantization event log' })).toHaveText('No recorded events yet.'); armed = true;
  await expect(page.getByRole('article', { name: 'ACT quantization job details' })).toContainText('succeeded', { timeout: 7000 });
  await expect(page.getByRole('region', { name: 'ACT quantization event log' })).toContainText('Generated final packed reload receipt'); expect(finalReads).toBeGreaterThan(0); expect(state.posts).toEqual([]);
});

test('failed worker output never becomes a successful download or rerun', async ({ page }) => {
  const state = await fixture(page); const failed = job('failed', 'failed'); failed.error = 'Generated fresh reload mismatch'; failed.result = { reports: [report()], artifacts: [{ ...artifact('partial', 'native_quantized'), job_id: failed.id }] }; state.jobs.push(failed);
  await refresh(page); await openTransformationJob(page, 'quantization', failed.id);
  await expect(page.getByRole('article', { name: 'ACT quantization job details' }).getByRole('alert')).toHaveText('Generated fresh reload mismatch');
  await expect(page.getByRole('region', { name: 'Measured ACT quantization results' })).toHaveCount(0); await expect(page.getByRole('link', { name: 'Download INT8 package' })).toHaveCount(0); expect(state.posts).toEqual([]);
});

for (const invalid of ['model-id', 'duplicate-seeds', 'reversed-seeds']) test(`invalid model or fixture identity ${invalid} cannot promote a measured report`, async ({ page }) => {
  const state = await fixture(page); const done = job('invalid-report', 'succeeded'); const value = report();
  if (invalid === 'model-id') value.model_id = 'unbound-model';
  else if (invalid === 'duplicate-seeds') value.drift_from_fp32[1].seed = value.drift_from_fp32[0].seed;
  else value.drift_from_fp32.reverse();
  done.result = { reports: [value], artifacts: [] }; state.jobs.push(done); await refresh(page);
  await openTransformationJob(page, 'quantization', done.id);
  await expect(page.getByRole('region', { name: 'Measured ACT quantization results' })).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Native ACT quantization', exact: true }).getByRole('alert')).toContainText('complete measured quantization report is unavailable');
});


test('a completed earlier history refresh cannot authorize an ambiguous quantization retry', async ({ page }) => {
  const state = await fixture(page); await choosePolicy(page, 'act-export');
  let release!: () => void; state.postGate = new Promise<void>(resolve => { release = resolve; }); state.submit = 'lost';
  await submit(page).click(); await expect.poll(() => state.posts.length).toBe(1);
  await refresh(page); await expect(page.getByRole('button', { name: 'Refresh ACT quantization jobs', exact: true })).toBeEnabled(); release();
  const allow = page.getByRole('button', { name: 'I checked the jobs; allow a new request' });
  await expect(allow).toBeDisabled(); await refresh(page); await expect(allow).toBeEnabled(); expect(state.posts).toHaveLength(1);
});

test('a delayed cancel response keeps the newly selected quantization job', async ({ page }) => {
  const state = await fixture(page); state.jobs.push(job('one'), job('two')); await refresh(page);
  await openTransformationJob(page, 'quantization', 'one');
  let release!: () => void; state.cancelGate = new Promise<void>(resolve => { release = resolve; });
  await page.getByRole('button', { name: 'Cancel selected ACT quantization' }).click(); await page.getByRole('button', { name: 'Confirm cancellation', exact: true }).click();
  await expect.poll(() => state.cancels.length).toBe(1);
  await openTransformationJob(page, 'quantization', 'two'); release();
  await expect(page.getByRole('button', { name: 'Cancel selected ACT quantization' })).toBeEnabled();
  await expect(page.getByRole('article', { name: 'ACT quantization job details' })).toHaveAttribute('data-job-id', 'two'); expect(state.cancels).toEqual(['one']);
});

test('uncertain quantization survives stage navigation and reload until fresh history review', async ({ page }) => {
  const state = await fixture(page); state.submit = 'lost'; await choosePolicy(page, 'act-export'); await submit(page).click();
  await expect(page.getByRole('button', { name: 'I checked the jobs; allow a new request' })).toBeVisible();
  await page.getByRole('link', { name: 'Dataset', exact: true }).click(); await page.getByRole('link', { name: 'Quantize', exact: true }).click();
  await page.getByRole('button', { name: 'Review request', exact: true }).click();
  await expect(submit(page)).toBeDisabled();
  await page.reload(); await page.getByRole('link', { name: 'Quantize', exact: true }).click(); await chooseTransformationModel(page, 'quantization', 'Choose Generated act-export · act-export');
  const allow = page.getByRole('button', { name: 'I checked the jobs; allow a new request' }); await expect(allow).toBeDisabled();
  state.jobsError = true; await refresh(page); await expect(allow).toBeDisabled();
  state.jobsError = false; await refresh(page); await expect(allow).toBeEnabled(); expect(state.posts).toHaveLength(1);
});


async function fromStudent(page: Page, state: Awaited<ReturnType<typeof fixture>>, id: string) {
  const output = { ...artifact(id, 'native_checkpoint', { recipe: 'act-action-distillation-v1' }), job_id: 'distilled' };
  state.jobs.push({ id: 'distilled', project_id: 'alpha', kind: 'policy.distill', status: 'succeeded', stage: 'completed', created_at: timestamp, updated_at: timestamp,
    request: { operation: 'policy.distill', runtime_id: 'student', artifact_id: 'teacher', dataset_job_id: 'data', timeout_seconds: 600, native_distillation: { adapter: 'act-act-v1', coordinate_attestation: 'generated_fixture', steps: 1 } }, result: { reports: [], artifacts: [output] } });
  await page.getByRole('link', { name: 'Distill', exact: true }).click();
  await openTransformationJob(page, 'distillation', 'distilled');
  await page.getByRole('button', { name: 'Open ACT quantization', exact: true }).click();
}

test('preferred distilled policy is selected once and polling preserves a later explicit choice', async ({ page }) => {
  const state = await fixture(page); await fromStudent(page, state, 'act-native');
  const picker = page.getByRole('group', { name: 'Policy', exact: true }); await expectPolicy(page, 'act-native');
  await choosePolicy(page, 'act-export'); await refresh(page); await expect(page.getByRole('button', { name: 'Refresh ACT quantization jobs', exact: true })).toBeEnabled();
  await expectPolicy(page, 'act-export'); await submit(page).click();
  await expect.poll(() => state.posts.length).toBe(1); expect(state.posts[0].artifact_id).toBe('act-export');
});

test('preferred policy from another project is never selected and never enables submission', async ({ page }) => {
  const state = await fixture(page); await fromStudent(page, state, 'wrong-project');
  await expect(page.getByRole('group', { name: 'Policy', exact: true }).locator('input:checked')).toHaveCount(0); await expect(submit(page)).toBeDisabled(); expect(state.posts).toHaveLength(0);
});

test('a delayed preferred policy does not replace a user choice made while it was absent', async ({ page }) => {
  const state = await fixture(page); await fromStudent(page, state, 'late-student');
  const picker = page.getByRole('group', { name: 'Policy', exact: true }); await expect(picker.locator('input:checked')).toHaveCount(0);
  await choosePolicy(page, 'act-export'); state.artifacts.push(artifact('late-student', 'native_checkpoint'));
  await refresh(page); await expect(page.getByLabel('Checkpoint').locator('option[value="late-student"]')).toHaveCount(1); await expectPolicy(page, 'act-export'); expect(state.posts).toHaveLength(0);
});

test('unavailable session storage blocks a request instead of dropping its recovery journal', async ({ page }) => {
  await page.addInitScript(() => { const original = Storage.prototype.setItem; Storage.prototype.setItem = function(key, value) { if (key.startsWith('firebird:job-attempt:')) throw new DOMException('Storage unavailable', 'SecurityError'); return original.call(this, key, value); }; });
  const state = await fixture(page); await choosePolicy(page, 'act-export'); await submit(page).click();
  await expect(submit(page)).toBeDisabled(); expect(state.posts).toHaveLength(0);
});

test('activity refresh failure remains visible outside collapsed details', async ({ page }) => {
  const state = await fixture(page); state.jobs.push(job('watched')); await refresh(page); await openTransformationJob(page, 'quantization', 'watched');
  await page.route('**/api/v1/jobs/watched/events', route => route.fulfill({ status: 503, json: { detail: 'Generated events outage' } }));
  await refresh(page); await expect(page.getByRole('article', { name: 'ACT quantization job details' }).getByRole('alert')).toContainText('previously received activity may be stale');
  await expect(page.getByRole('region', { name: 'ACT quantization event log' })).toBeHidden(); expect(state.posts).toHaveLength(0);
});


test('a preferred policy waits for its compatible owned artifact to arrive', async ({ page }) => {
  const state = await fixture(page); await fromStudent(page, state, 'arriving-student');
  const picker = page.getByRole('group', { name: 'Policy', exact: true }); await expect(picker.locator('input:checked')).toHaveCount(0);
  state.artifacts.push(artifact('arriving-student', 'native_checkpoint')); await refresh(page);
  await expectPolicy(page, 'arriving-student'); await expect(submit(page)).toBeEnabled(); expect(state.posts).toHaveLength(0);
});


test('an acknowledged job remains visible when clearing its recovery journal fails', async ({ page }) => {
  await page.addInitScript(() => {
    const original = Storage.prototype.removeItem;
    (window as unknown as { journalRemovals: number }).journalRemovals = 0;
    Storage.prototype.removeItem = function(key) {
      if (key.startsWith('firebird:job-attempt:')) { (window as unknown as { journalRemovals: number }).journalRemovals++; throw new DOMException('Storage unavailable', 'SecurityError'); }
      return original.call(this, key);
    };
  });
  const state = await fixture(page); await choosePolicy(page, 'act-export'); await submit(page).click();
  await expect(page.getByRole('article', { name: 'ACT quantization job details' })).toHaveAttribute('data-job-id', 'submitted');
  await expect(page.getByRole('region', { name: 'Native ACT quantization', exact: true }).getByRole('alert')).toContainText('Browser session storage is unavailable');
  await expect(page.getByText('Submitting one local quantization job…', { exact: true })).toHaveCount(0);
  expect(await page.evaluate(() => (window as unknown as { journalRemovals: number }).journalRemovals)).toBe(1);
  await chooseTransformationModel(page,'quantization','Choose Generated act-export · act-export'); await expect(submit(page)).toBeDisabled(); expect(state.posts).toHaveLength(1);
  await page.reload(); await page.getByRole('link', { name: 'Quantize', exact: true }).click(); await chooseTransformationModel(page, 'quantization', 'Choose Generated act-export · act-export');
  await expect(page.getByRole('button', { name: 'I checked the jobs; allow a new request' })).toBeDisabled(); expect(state.posts).toHaveLength(1);
});


test('journal cleanup after navigation exposes recovery instead of a permanent pending state', async ({ page }) => {
  await page.addInitScript(() => { const original = Storage.prototype.removeItem; Storage.prototype.removeItem = function(key) { if (key.startsWith('firebird:job-attempt:')) throw new DOMException('Storage unavailable', 'SecurityError'); return original.call(this, key); }; });
  const state = await fixture(page); await choosePolicy(page, 'act-export');
  let release!: () => void; state.postGate = new Promise<void>(resolve => { release = resolve; });
  await submit(page).click(); await expect.poll(() => state.posts.length).toBe(1);
  await page.getByRole('link', { name: 'Dataset', exact: true }).click(); await page.getByRole('link', { name: 'Quantize', exact: true }).click();
  await page.getByRole('button', { name: 'Review request', exact: true }).click();
  await expect(page.getByText('Submitting one local quantization job…', { exact: true })).toBeVisible(); release();
  await expect(page.getByText(/Browser session storage is unavailable/)).toBeVisible();
  await expect(page.getByText('Submitting one local quantization job…', { exact: true })).toHaveCount(0);
  const allow = page.getByRole('button', { name: 'I checked the jobs; allow a new request' }); await expect(allow).toBeDisabled();
  await refresh(page); await expect(allow).toBeEnabled(); await openTransformationJob(page, 'quantization', 'submitted');
  await expect(page.getByRole('article', { name: 'ACT quantization job details' })).toHaveAttribute('data-job-id', 'submitted'); expect(state.posts).toHaveLength(1);
});


function replayCandidate(id = 'ready-packed') {
  const proof = report();
  return { ...artifact(id, 'native_quantized', { format: 'firebird_quant', format_version: 1, model_id: proof.model_id, precision: proof.precision, fresh_reload_verified: true, cpu_reload_verified: true, source_artifact_id: proof.source_artifact_id, source_artifact_manifest_sha256: proof.source_artifact_manifest_sha256 }), job_id: 'ready-job' };
}

test('replay eligibility binds a completed packed artifact to its job and measured report', () => {
  const done = job('ready-job', 'succeeded') as unknown as Job;
  const measured = measuredNativeReport(report(), done);
  expect(measured).not.toBeNull();
  expect(replayableNativeOutput(replayCandidate() as PackedArtifact, done, measured)).toBe(true);
  expect(replayableNativeOutput(replayCandidate() as PackedArtifact, { ...done, status: 'running' }, measured)).toBe(false);
  expect(replayableNativeOutput(replayCandidate() as PackedArtifact, done, null)).toBe(false);
});

test('replay eligibility rejects remote, incomplete and mismatched packed artifact identities', () => {
  const done = job('ready-job', 'succeeded') as unknown as Job;
  const measured = measuredNativeReport(report(), done);
  const invalid = [
    { ...replayCandidate(), id: '' },
    { ...replayCandidate(), project_id: 'beta' },
    { ...replayCandidate(), job_id: 'other-job' },
    { ...replayCandidate(), format: 'inference_export' },
    ...[
      { architecture: 'smolvla' }, { format: 'other' }, { format_version: 2 },
      { inference_only: false }, { fresh_reload_verified: false }, { cpu_reload_verified: false },
      { model_id: 'sha256:' + 'f'.repeat(64) }, { precision: 'int4' },
      { source_artifact_id: 'another-source' }, { source_artifact_manifest_sha256: 'f'.repeat(64) },
      { storage: 'gcs' }, { remote_uri: 'gs://generated' }, { remote: true }
    ].map(change => ({ ...replayCandidate(), metadata: { ...replayCandidate().metadata, ...change } }))
  ];
  for (const candidate of invalid) expect(replayableNativeOutput(candidate as PackedArtifact, done, measured), JSON.stringify(candidate)).toBe(false);
});


test('quantization mutation ACK status contract accepts only six exact strings', async () => {
  const request = job('contract').request;
  const valid = ['queued', 'running', 'succeeded', 'failed', 'cancelled', 'interrupted'];
  const original = { id: 'contract', project_id: 'alpha', kind: request.operation, status: 'running', request } as Parameters<typeof cancelNativeQuantization>[0];
  const realFetch = globalThis.fetch;
  let methods: string[] = [], status: unknown = 'running', cancelResponse = false;
  globalThis.fetch = async (_input, init) => {
    methods.push(init?.method ?? 'GET');
    return new Response(JSON.stringify({ ...original, status: cancelResponse && init?.method === 'GET' ? 'running' : status }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  };
  try {
    for (status of valid) { methods = []; expect((await startNativeQuantization({ project: 'alpha', runtime: runtime.id, artifact: 'act-export', bits: 8, timeout: 600 })).status).toBe(status); expect(methods).toEqual(['POST']); }
    for (status of [['running'], ['succeeded'], [['running']], [], null, true, 1, {}, undefined, 'unknown']) {
      methods = []; await expect(startNativeQuantization({ project: 'alpha', runtime: runtime.id, artifact: 'act-export', bits: 8, timeout: 600 })).rejects.toThrow(/outcome is unverified/); expect(methods).toEqual(['POST']);
      methods = []; await expect(cancelNativeQuantization(original, () => true)).rejects.toThrow(/outcome is unverified/); expect(methods).toEqual(['GET']);
    }
    cancelResponse = true; status = ['cancelled']; methods = [];
    await expect(cancelNativeQuantization(original, () => true)).rejects.toThrow(/outcome is unverified/); expect(methods).toEqual(['GET', 'POST']);
    status = 'cancelled'; methods = [];
    expect((await cancelNativeQuantization(original, () => true)).status).toBe('cancelled'); expect(methods).toEqual(['GET', 'POST']);
  } finally { globalThis.fetch = realFetch; }
});

for (const phase of ['submit', 'cancel-preflight', 'cancel-receipt'] as const) test(`quantization malformed ${phase} status never clears request recovery`, async ({ page }) => {
  const state = await fixture(page); await choosePolicy(page, 'act-export');
  if (phase === 'submit') {
    await page.route('**/api/v1/projects/alpha/policy-jobs', route => {
      const request = route.request().postDataJSON(); state.posts.push(request);
      return route.fulfill({ status: 202, json: { id: 'malformed', project_id: 'alpha', kind: request.operation, status: ['running'], request } });
    });
    await submit(page).click();
  } else {
    await submit(page).click();
    await expect(page.getByRole('article', { name: 'ACT quantization job details' })).toHaveAttribute('data-job-id', 'submitted');
    const original = state.jobs.find(item => item.id === 'submitted')!;
    if (phase === 'cancel-preflight') await page.route('**/api/v1/jobs/submitted', route => route.fulfill({ json: { ...original, status: ['running'] } }));
    else await page.route('**/api/v1/jobs/submitted/cancel', route => { state.cancels.push('submitted'); return route.fulfill({ json: { ...original, status: ['cancelled'] } }); });
    await page.getByRole('button', { name: 'Cancel selected ACT quantization', exact: true }).click();
    await page.getByRole('button', { name: 'Confirm cancellation', exact: true }).click();
  }
  await expect(page.getByRole('region', { name: 'Native ACT quantization', exact: true }).getByRole('alert')).toContainText('outcome is unverified');
  expect(state.posts).toHaveLength(1);
  expect(state.cancels).toHaveLength(phase === 'cancel-receipt' ? 1 : 0);
  expect(await page.evaluate(() => sessionStorage.getItem('firebird:job-attempt:policy.quantize:alpha'))).toContain('uncertain');
  if (phase === 'submit') await expect(page.getByRole('article', { name: 'ACT quantization job details' })).toHaveCount(0);
});


test('8-step packed report displays its full prediction dimensions without quality acceptance', async ({ page }) => {
  const state = await fixture(page), done = job('temporal-packed', 'succeeded');
  const measured = { ...report(), prediction_horizon: 8, execution_horizon: 3, temporal_contract_sha256: 'e'.repeat(64) };
  for (const row of measured.drift_from_fp32) { row.raw.coordinates = 48; row.postprocessed.coordinates = 48; }
  done.result = { reports: [measured], artifacts: [] }; state.jobs.push(done); await refresh(page);
  await openTransformationJob(page, 'quantization', done.id);
  await expect(page.getByRole('region', { name: 'Measured ACT quantization results' })).toBeVisible();
  await page.getByText('Action differences', { exact: true }).click();
  await expect(page.getByRole('table')).toHaveAccessibleName('Difference from FP32 across each full 8 × 6 action chunk');
  await expect(page.getByText('Task quality, calibration, speed and GPU memory savings remain unverified.', { exact: true })).toBeVisible();
  expect(state.posts).toEqual([]);
});
