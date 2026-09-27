import { createServer } from 'node:http';
import { expect, test, type Page } from '@playwright/test';

const timestamp = '2026-09-27T12:00:00Z';
const profile = { id: 'cup-fixture', label: 'Generated test cup profile', architectures: ['act', 'smolvla'], experimental: true, task_object: 'cup' };
function artifact(id: string, architecture = 'act', format = 'inference_export', metadata = {}) {
  return { id, project_id: 'alpha', job_id: 'source', label: `Generated ${id}`, format, path: 'fixture', manifest_sha256: 'a'.repeat(64), file_bytes: 4096, parent_ids: [], metadata: { architecture, ...metadata } };
}
function job(id: string, kind = 'policy.run', status = 'running', source = 'act-export') {
  return { id, project_id: 'alpha', kind, status, created_at: timestamp, updated_at: timestamp, stage: 'simulation', error: null,
    request: { operation: kind, runtime_id: profile.id, artifact_id: kind === 'policy.run' ? source : null, source_id: kind === 'policy.import' ? 'upload-id' : null, simulation: { profile_id: profile.id, experimental: kind === 'policy.run' }, timeout_seconds: kind === 'policy.import' ? 600 : 7200 },
    compute_target: null, simulation_target: kind === 'policy.run' ? { profile_id: profile.id, profile_sha256: 'b'.repeat(64), provider: 'gcp', accelerators: ['L4', 'H100'], source_manifest_sha256: 'a'.repeat(64) } : null, result: null as null | Record<string, unknown> };
}
async function fixture(page: Page) {
  const state = { profiles: [profile], optionsError: false, jobsError: false, submit: 'ok', posts: [] as { path: string; body: unknown }[], jobs: [] as ReturnType<typeof job>[], artifacts: [artifact('act-export'), artifact('smol-export', 'smolvla'), artifact('periodic', 'act', 'training_checkpoint'), artifact('remote', 'act', 'native_checkpoint', { storage: 'gcs', remote_uri: 'gs://fixture' }), artifact('unsupported', 'diffusion')], uploaded: Buffer.alloc(0), cancels: [] as string[], events: [] as string[], tasks: { isaac: 'RUNNING', vla: 'STARTING' } };
  await page.route('**/api/v1/**', async route => {
    const req = route.request(), url = new URL(req.url()), path = url.pathname;
    if (req.method() === 'POST') {
      state.posts.push({ path, body: path.endsWith('/model-imports') ? null : req.postDataJSON() });
      if (path.endsWith('/model-imports')) {
        state.uploaded = req.postDataBuffer() ?? Buffer.alloc(0);
        expect(req.headers()['content-type']).toBe('application/x-tar'); expect(url.searchParams.get('profile_id')).toBe(profile.id);
        const imported = job('imported', 'policy.import', 'succeeded'); state.jobs.unshift(imported); state.artifacts.push(artifact('uploaded-policy', 'smolvla'));
        return route.fulfill({ status: 202, json: imported });
      }
      if (path.endsWith('/policy-jobs')) {
        const created = job('submitted', 'policy.run', 'running', String(req.postDataJSON().artifact_id));
        created.request.timeout_seconds = req.postDataJSON().timeout_seconds;
        if (state.submit === 'wrong-project') return route.fulfill({ status: 202, json: { ...created, project_id: 'beta' } });
        if (state.submit === 'reject') return route.fulfill({ status: 422, json: { detail: 'Fixture camera is incompatible with the cup scene' } });
        state.jobs.unshift(created);
        if (state.submit === 'lost') return route.abort('failed');
        return route.fulfill({ status: 202, json: created });
      }
      if (path.endsWith('/cancel')) { const id = path.split('/').at(-2)!; state.cancels.push(id); const selected = state.jobs.find(item => item.id === id)!; selected.status = 'cancelled'; return route.fulfill({ json: selected }); }
      return route.fulfill({ status: 405, json: { detail: 'Unexpected test mutation' } });
    }
    if (path === '/api/v1/simulation-options') return route.fulfill(state.optionsError ? { status: 503, json: { detail: 'Fixture profile failure' } } : { json: { profiles: state.profiles, unavailable_reason: state.profiles.length ? null : 'No Isaac simulation profile is configured.', max_archive_bytes: 4 * 1024 ** 3, scored_evaluation: false } });
    if (path === '/api/v1/projects') return route.fulfill({ json: [{ id: 'alpha', name: 'Generated simulation fixture', created_at: timestamp }, { id: 'beta', name: 'Second fixture', created_at: timestamp }] });
    if (path.endsWith('/jobs') && path.includes('/projects/')) return route.fulfill(state.jobsError ? { status: 503, json: { detail: 'Fixture job outage' } } : { json: path.includes('/alpha/') ? state.jobs : [] });
    if (path.endsWith('/artifacts')) return route.fulfill({ json: path.includes('/alpha/') ? state.artifacts : [] });
    if (path.endsWith('/events')) { const id = path.split('/').at(-2)!; state.events.push(id); return route.fulfill({ json: [{ sequence: 1, stage: 'simulation', timestamp, message: `Generated event for ${id}`, data: { tasks: state.tasks } }] }); }
    if (path.startsWith('/api/v1/jobs/')) { const id = path.split('/').at(-1)!; const selected = state.jobs.find(item => item.id === id); if (selected) return route.fulfill({ json: selected }); }
    return route.continue();
  });
  await page.goto('/'); await expect(page.getByLabel('Current project')).toHaveValue('alpha');
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  await page.getByRole('button', { name: '3D simulation', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Native Isaac simulation', exact: true })).toBeVisible();
  return state;
}
const acknowledge = (page: Page) => page.getByRole('checkbox', { name: /experimental, paid cloud rollout/ }).check();

test('a packed CPU profile is never selected automatically on direct Run entry', async ({ page }) => {
  const state = await fixture(page);
  const cpu = { ...profile, id: 'packed-cpu', label: 'Generated packed CPU profile', architectures: ['act'], policy_runtime: 'packed-act-cpu', policy_device: 'cpu', policy_formats: ['firebird_quant'], provider: 'gcp', accelerators: ['L4'] };
  state.profiles = [cpu];
  state.artifacts.push(artifact('packed-policy', 'act', 'native_quantized', { format: 'firebird_quant' }));
  await page.reload(); await page.getByRole('button', { name: 'Run', exact: true }).click();
  await page.getByRole('button', { name: '3D simulation', exact: true }).click();
  const profiles = page.getByRole('radiogroup', { name: 'Isaac profile', exact: true });
  const policies = page.getByRole('radiogroup', { name: 'Native policy', exact: true });
  await expect(profiles.getByRole('radio', { name: cpu.label, exact: true })).toBeEnabled();
  await expect(profiles.locator('input:checked')).toHaveCount(0);
  await expect(policies.getByRole('radio')).toHaveCount(0);
  await expect(page.getByRole('checkbox', { name: /experimental, paid cloud rollout/ })).toBeDisabled();
  await profiles.getByRole('radio', { name: cpu.label, exact: true }).check();
  await expect(policies.getByRole('radio', { name: 'Generated packed-policy', exact: true })).toBeVisible();
  await expect(policies.locator('input:checked')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Start experimental simulation', exact: true })).toBeDisabled();
  expect(state.posts).toEqual([]);
});

test('native Run preserves engine navigation and fails closed when profiles are missing', async ({ page }) => {
  const state = await fixture(page); state.profiles = [];
  await page.reload(); await page.getByRole('button', { name: 'Run', exact: true }).click();
  await expect(page.getByRole('group', { name: 'Run mode', exact: true })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Run jobs', exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: '3D simulation' }).click();
  await expect(page.getByText('Simulator not connected', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Start experimental simulation' })).toBeDisabled();
  await expect(page.getByRole('radiogroup', { name: 'Policy source', exact: true })).toHaveCount(0);
  expect(state.posts).toEqual([]);
  await page.getByRole('button', { name: 'Evaluate', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Evaluation purpose' })).toContainText('scored ACT / Isaac evaluation is not configured');
});

test('native import sends exact TAR bytes once and does not start a GPU run', async ({ page }) => {
  const state = await fixture(page);
  const bytes = Buffer.from('Explicit generated TAR transport fixture; worker validation is covered by API tests.');
  await page.getByRole('radio', { name: 'Import package', exact: true }).check();
  await page.getByLabel('Native policy TAR', { exact: true }).setInputFiles({ name: 'synthetic-policy.tar', mimeType: 'application/x-tar', buffer: bytes });
  await page.getByRole('button', { name: 'Upload and validate policy' }).dblclick();
  await expect(page.getByRole('article', { name: 'Native simulation job details' })).toContainText('The native package is saved');
  expect(state.uploaded).toEqual(bytes); expect(state.posts).toHaveLength(1);
  await page.getByText('Prepare another run', { exact: true }).click();
  await expect(page.getByRole('radiogroup', { name: 'Native policy', exact: true }).locator('input:checked')).toHaveCount(0);
  await expect(page.getByRole('radio', { name: 'Generated uploaded-policy', exact: true })).toHaveCount(1);
  await expect(page.getByRole('button', { name: 'Start experimental simulation' })).toBeDisabled();
});

for (const architecture of ['act', 'smol']) test(`native ${architecture} launch requires explicit source and consent with bounded exact payload`, async ({ page }) => {
  const state = await fixture(page);
  const choices = page.getByRole('radiogroup', { name: 'Native policy', exact: true });
  await expect(choices.getByRole('radio')).toHaveCount(2);
  await expect(choices).toContainText('Generated act-export');
  await expect(choices).toContainText('Generated smol-export');
  await choices.getByRole('radio', { name: `Generated ${architecture}-export`, exact: true }).check(); await acknowledge(page);
  await page.getByLabel('Simulation timeout (seconds)').fill('7201');
  await expect(page.getByRole('button', { name: 'Start experimental simulation' })).toBeDisabled();
  await page.getByLabel('Simulation timeout (seconds)').fill('600');
  await expect(page.getByRole('checkbox', { name: /experimental, paid cloud rollout/ })).not.toBeChecked();
  await acknowledge(page);
  await page.getByRole('button', { name: 'Start experimental simulation' }).dblclick();
  await expect(page.getByRole('article', { name: 'Native simulation job details' })).toHaveAttribute('data-job-id', 'submitted');
  expect(state.posts).toEqual([{ path: '/api/v1/projects/alpha/policy-jobs', body: { operation: 'policy.run', runtime_id: profile.id, artifact_id: `${architecture}-export`, simulation: { profile_id: profile.id, experimental: true }, timeout_seconds: 600 } }]);
  await page.getByText('Activity and technical details', { exact: true }).click();
  await expect(page.getByRole('region', { name: 'Native simulation event log' })).toContainText('Generated event for submitted');
  await expect(page.getByText('Cup pickup success', { exact: true }).locator('..')).toContainText('Not measured');
});

test('lost launch receipt pauses submissions and recovers the actual job without retry', async ({ page }) => {
  const state = await fixture(page); state.submit = 'lost';
  await page.getByRole('radio', { name: 'Generated act-export', exact: true }).check(); await acknowledge(page);
  await page.getByRole('button', { name: 'Start experimental simulation' }).click();
  await expect(page.locator('.native-simulation').getByRole('alert')).toContainText('outcome is unverified');
  await expect(page.getByRole('button', { name: 'Start experimental simulation' })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'I checked the jobs; allow a new request' })).toBeDisabled();
  await page.getByRole('button', { name: 'Refresh simulation jobs' }).click();
  await page.getByLabel('Saved simulation job').selectOption('submitted');
  await expect(page.getByRole('article', { name: 'Native simulation job details' })).toHaveAttribute('data-job-id', 'submitted');
  expect(state.posts).toHaveLength(1);
});

test('mismatched receipt is ambiguous while explicit admission failure remains visible', async ({ page }) => {
  const state = await fixture(page); state.submit = 'reject';
  await page.getByRole('radio', { name: 'Generated act-export', exact: true }).check(); await acknowledge(page);
  await page.getByRole('button', { name: 'Start experimental simulation' }).click();
  await expect(page.locator('.native-simulation').getByRole('alert')).toHaveText('Fixture camera is incompatible with the cup scene');
  expect(state.posts).toHaveLength(1);
  state.submit = 'wrong-project';
  await page.getByRole('button', { name: 'Start experimental simulation' }).click();
  await expect(page.locator('.native-simulation').getByRole('alert')).toContainText('outcome is unverified');
  await expect(page.getByRole('article', { name: 'Native simulation job details' })).toHaveCount(0);
  expect(state.posts).toHaveLength(2);
});

test('native cancellation is confirmed for the selected job and cannot follow a selection switch', async ({ page }) => {
  const state = await fixture(page); state.jobs.push(job('first'), job('second'));
  await page.getByRole('button', { name: 'Refresh simulation jobs' }).click();
  await page.getByLabel('Saved simulation job').selectOption('first');
  await page.getByRole('button', { name: 'Cancel selected native job' }).click();
  await page.getByLabel('Saved simulation job').selectOption('second');
  await expect(page.getByRole('button', { name: 'Confirm cancellation' })).toHaveCount(0);
  expect(state.cancels).toEqual([]);
  await page.getByRole('button', { name: 'Cancel selected native job' }).click();
  await page.getByRole('button', { name: 'Confirm cancellation' }).click();
  await expect(page.getByRole('article', { name: 'Native simulation job details' })).toContainText('cancelled');
  expect(state.cancels).toEqual(['second']);
});

test('verified native record exposes fixed video/download paths without claiming task success', async ({ page }) => {
  const state = await fixture(page), done = job('done', 'policy.run', 'succeeded');
  done.result = { decision: 'completed', artifacts: [{ ...artifact('record', 'act', 'simulation_record'), job_id: 'done' }], reports: [{ stage: 'simulation', task_success: null, calibration_verified: false, artifacts: [{ path: 'artifacts/outputs/video.mp4', sha256: 'e'.repeat(64), bytes: 12, generation: '1' }] }] };
  state.jobs.push(done);
  await page.route('**/api/v1/jobs/done/simulation-media/video', route => route.abort());
  await page.getByRole('button', { name: 'Refresh simulation jobs' }).click(); await page.getByLabel('Saved simulation job').selectOption('done');
  await expect(page.getByRole('link', { name: 'Download simulation record' })).toHaveAttribute('href', '/api/v1/projects/alpha/artifacts/record/download');
  await expect(page.getByText(/The recorded video is unavailable/)).toBeVisible();
  await expect(page.getByText('Execution completed. This is not a scored evaluation or proof of cup pickup.')).toBeVisible();
  await page.getByRole('button', { name: 'Cloud runs', exact: true }).click();
  await expect(page.getByLabel('Application cloud job', { exact: true })).toHaveValue('done');
  await expect(page.getByText('Execution target', { exact: true }).locator('..')).toContainText('L4 + H100');
  await page.getByRole('button', { name: 'Open simulation job' }).click();
  await expect(page.getByRole('article', { name: 'Native simulation job details' })).toHaveAttribute('data-job-id', 'done');
  expect(state.posts).toHaveLength(0);
});

test('failed run and project changes cannot reuse selected source or expose success outputs', async ({ page }) => {
  const state = await fixture(page); const failed = job('failed', 'policy.run', 'failed'); failed.error = 'Synthetic worker failure' as never; state.jobs.push(failed);
  await page.getByRole('button', { name: 'Refresh simulation jobs' }).click(); await page.getByLabel('Saved simulation job').selectOption('failed');
  await expect(page.locator('.native-simulation').getByRole('alert')).toHaveText('Synthetic worker failure');
  await expect(page.getByRole('link', { name: 'Download simulation record' })).toHaveCount(0);
  await page.getByText('Prepare another run', { exact: true }).click();
  await page.getByRole('radio', { name: 'Generated smol-export', exact: true }).check(); await acknowledge(page);
  await page.getByLabel('Current project').selectOption('beta');
  await expect(page.getByRole('button', { name: 'Check inference', exact: true })).toHaveAttribute('aria-pressed', 'false');
  await page.getByRole('button', { name: '3D simulation', exact: true }).click();
  await expect(page.getByRole('radiogroup', { name: 'Native policy', exact: true }).locator('input:checked')).toHaveCount(0);
  await expect(page.getByRole('checkbox', { name: /experimental, paid cloud rollout/ })).not.toBeChecked();
  await expect(page.getByRole('button', { name: 'Start experimental simulation' })).toBeDisabled();
  expect(state.posts).toHaveLength(0);
  await page.setViewportSize({ width: 320, height: 900 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
});


test('changing the available profile cannot carry consent to a different cloud target', async ({ page }) => {
  const state = await fixture(page);
  await page.getByRole('radio', { name: 'Generated act-export', exact: true }).check(); await acknowledge(page);
  state.profiles = [{ ...profile, id: 'replacement-cup', label: 'Different generated profile' }];
  await page.getByRole('button', { name: 'Refresh simulation jobs' }).click();
  await expect(page.getByRole('radio', { name: 'Different generated profile', exact: true })).toBeChecked();
  await expect(page.getByRole('checkbox', { name: /experimental, paid cloud rollout/ })).not.toBeChecked();
  await expect(page.getByRole('button', { name: 'Start experimental simulation' })).toBeDisabled();
  expect(state.posts).toHaveLength(0);
});


test('native and Cloud event views display bounded observed worker states as they change', async ({ page }) => {
  const state = await fixture(page); state.jobs.push(job('observed'));
  await page.getByRole('button', { name: 'Refresh simulation jobs' }).click();
  await page.getByLabel('Saved simulation job').selectOption('observed');
  await expect(page.getByLabel('Observed simulation workers')).toHaveText('Isaac: RUNNING · VLA: STARTING');
  await page.getByText('Activity and technical details', { exact: true }).click();
  await expect(page.getByRole('region', { name: 'Native simulation event log' })).toContainText('Isaac: RUNNING · VLA: STARTING');
  state.tasks = { isaac: 'SUCCEEDED', vla: 'CANCELLING' };
  await expect(page.getByRole('region', { name: 'Native simulation event log' })).toContainText('Isaac: SUCCEEDED · VLA: CANCELLING', { timeout: 7000 });
  await page.getByRole('button', { name: 'Cloud runs', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Application job event log', exact: true })).toContainText('Isaac: SUCCEEDED · VLA: CANCELLING');
  state.tasks = { isaac: 'UNRECOGNIZED_STATUS', vla: 'arbitrary-not-a-worker-state' };
  await page.getByRole('button', { name: 'Refresh application jobs' }).click();
  await expect(page.getByRole('region', { name: 'Application job event log', exact: true })).toHaveText(`${timestamp} · simulation · Generated event for observed`);
  expect(state.posts).toHaveLength(0);
});

for (const view of ['native', 'cloud'] as const) test(`${view} final events refresh when a quick job completes after an empty active poll`, async ({ page }) => {
  const state = await fixture(page);
  const quick = job('quick-completion', view === 'native' ? 'policy.import' : 'policy.run', 'queued');
  state.jobs.push(quick);
  let armed = false, terminalPublished = false, finalReads = 0;
  let finishEmptyPoll!: () => void;
  const emptyPollFinished = new Promise<void>(resolve => { finishEmptyPoll = resolve; });
  await page.route('**/api/v1/jobs/quick-completion/events', async route => {
    if (terminalPublished) {
      finalReads += 1;
      return route.fulfill({ json: [{ sequence: 1, stage: 'model_import', timestamp, message: 'Generated final integrity receipt', data: {} }] });
    }
    await route.fulfill({ json: [] });
    if (armed) finishEmptyPoll();
  });
  await page.route('**/api/v1/projects/alpha/jobs', async route => {
    if (armed) {
      // Let an in-flight active-event read finish empty before publishing the
      // terminal status. No event timer is advanced or manually refreshed.
      await emptyPollFinished;
      quick.status = 'succeeded';
      terminalPublished = true;
    }
    return route.fulfill({ json: state.jobs });
  });
  await page.getByRole('button', { name: 'Refresh simulation jobs' }).click();
  if (view === 'native') {
    await page.getByLabel('Saved simulation job').selectOption(quick.id);
    await page.getByText('Activity and technical details', { exact: true }).click();
  }
  else await page.getByRole('button', { name: 'Cloud runs', exact: true }).click();
  const log = page.getByRole('region', { name: view === 'native' ? 'Native simulation event log' : 'Application job event log', exact: true });
  await expect(log).toContainText(view === 'native' ? 'No recorded events yet.' : 'No events received for this job yet.');
  armed = true;
  const status = view === 'native' ? page.getByRole('article', { name: 'Native simulation job details' }) : page.getByText('Recorded status', { exact: true }).locator('..');
  await expect(status).toContainText('succeeded', { timeout: 8000 });
  await expect(log).toContainText('Generated final integrity receipt', { timeout: 5000 });
  expect(finalReads).toBeGreaterThan(0);
  expect(state.posts).toHaveLength(0);
});


test('completed recording is first, technical details stay secondary, and another run keeps explicit consent', async ({ page }) => {
  const state = await fixture(page), done = job('result-first', 'policy.run', 'succeeded');
  done.result = { decision: 'completed', artifacts: [{ ...artifact('record', 'act', 'simulation_record'), job_id: done.id }], reports: [{ stage: 'simulation', task_success: null, calibration_verified: false, artifacts: [{ path: 'artifacts/outputs/video.mp4', sha256: 'e'.repeat(64), bytes: 12, generation: '1' }] }] };
  state.jobs.push(done);
  let releaseVideo!: () => void;
  const videoHeld = new Promise<void>(resolve => { releaseVideo = resolve; });
  await page.route('**/api/v1/jobs/result-first/simulation-media/video', async route => { await videoHeld; await route.abort(); });
  try {
    await page.getByRole('button', { name: 'Refresh simulation jobs' }).click();
    await page.getByLabel('Saved simulation job').selectOption(done.id);
    const result = page.getByRole('article', { name: 'Native simulation job details' });
    const video = page.getByLabel('Recorded cup rollout', { exact: true });
    await expect(video).toBeVisible();
    await expect(result.getByRole('region', { name: 'Native simulation event log' })).toBeHidden();
    await expect(page.getByRole('button', { name: 'Start experimental simulation' })).toBeHidden();
    const mediaBox = await video.boundingBox();
    const preparation = page.getByText('Prepare another run', { exact: true });
    const prepareBox = await preparation.boundingBox();
    expect(mediaBox!.y + mediaBox!.height).toBeLessThan(prepareBox!.y);
    await page.getByText('Activity and technical details', { exact: true }).focus();
    await page.keyboard.press('Enter');
    await expect(result.getByRole('region', { name: 'Native simulation event log' })).toContainText('Generated event for result-first');
    await preparation.focus(); await page.keyboard.press('Enter');
    await expect(page.getByRole('radiogroup', { name: 'Native policy', exact: true }).locator('input:checked')).toHaveCount(0);
    await expect(page.getByRole('checkbox', { name: /experimental, paid cloud rollout/ })).not.toBeChecked();
    await expect(page.getByRole('button', { name: 'Start experimental simulation' })).toBeDisabled();
    expect(state.posts).toEqual([]);
  } finally { releaseVideo(); }
});

test('collapsed activity retains observed states and user disclosure choice across refresh', async ({ page }) => {
  const state = await fixture(page); state.jobs.push(job('quiet-progress'));
  await page.getByRole('button', { name: 'Refresh simulation jobs' }).click();
  await page.getByLabel('Saved simulation job').selectOption('quiet-progress');
  const summary = page.getByText('Activity and technical details', { exact: true });
  await expect(page.getByLabel('Observed simulation workers')).toHaveText('Isaac: RUNNING · VLA: STARTING');
  await expect(page.getByRole('region', { name: 'Native simulation event log' })).toBeHidden();
  await summary.click();
  await expect(page.getByRole('region', { name: 'Native simulation event log' })).toBeVisible();
  state.tasks = { isaac: 'SUCCEEDED', vla: 'CANCELLING' };
  await page.getByRole('button', { name: 'Refresh simulation jobs' }).click();
  await expect(page.getByLabel('Observed simulation workers')).toHaveText('Isaac: SUCCEEDED · VLA: CANCELLING');
  await expect(page.getByRole('region', { name: 'Native simulation event log' })).toBeVisible();
  await summary.click();
  await page.getByRole('button', { name: 'Refresh simulation jobs' }).click();
  await expect(page.getByRole('region', { name: 'Native simulation event log' })).toBeHidden();
  expect(state.posts).toEqual([]);
});

test('failed activity refresh visibly marks cached worker states stale while details stay closed', async ({ page }) => {
  const state = await fixture(page); state.jobs.push(job('stale-progress'));
  await page.getByRole('button', { name: 'Refresh simulation jobs' }).click();
  await page.getByLabel('Saved simulation job').selectOption('stale-progress');
  const workers = page.getByLabel('Observed simulation workers');
  const log = page.getByRole('region', { name: 'Native simulation event log' });
  await expect(workers).toHaveText('Isaac: RUNNING · VLA: STARTING');
  await expect(log).toBeHidden();
  await page.route('**/api/v1/jobs/stale-progress/events', route => route.fulfill({ status: 503, json: { detail: 'Generated activity outage' } }));
  await page.getByRole('button', { name: 'Refresh simulation jobs' }).click();
  await expect(page.getByRole('article', { name: 'Native simulation job details' }).getByRole('alert')).toHaveText('Activity updates are unavailable. Worker states and event history may be stale.');
  await expect(workers).toHaveText('Last observed: Isaac: RUNNING · VLA: STARTING');
  await expect(log).toBeHidden();
  await page.getByText('Activity and technical details', { exact: true }).click();
  await expect(log).toContainText('Generated event for stale-progress');
  await expect(page.getByText('Generated activity outage', { exact: true })).toBeVisible();
  expect(state.posts).toEqual([]);
});


for (const width of [320, 390]) test(`compact ${width}px navigation keeps stage content visible and every action keyboard reachable`, async ({ page }) => {
  await page.setViewportSize({ width, height: 844 });
  const state = await fixture(page);
  await page.getByRole('button', { name: 'Dataset', exact: true }).click();
  const heading = page.getByRole('heading', { name: 'Dataset', exact: true });
  const bounds = await heading.boundingBox();
  // The grouped navigation exposes all destinations; the stage heading and
  // introduction must still fit in the first screen without scrolling.
  expect(bounds!.y + bounds!.height).toBeLessThan(844 * 0.6);
  const navigation = page.getByRole('navigation', { name: 'Policy lifecycle' });
  // The compact navigation scrolls internally; its viewport must remain on-screen.
  const navigationBounds = await navigation.boundingBox();
  expect(navigationBounds!.x).toBeGreaterThanOrEqual(0);
  expect(navigationBounds!.x + navigationBounds!.width).toBeLessThanOrEqual(width);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
  await page.getByRole('button', { name: 'Run', exact: true }).focus();
  await page.keyboard.press('Tab');
  await expect(page.getByRole('button', { name: 'Decision lab', exact: true })).toBeFocused();
  await page.keyboard.press('Tab');
  const cloud = page.getByRole('button', { name: 'Cloud runs', exact: true });
  await expect(cloud).toBeFocused();
  await page.keyboard.press('Tab');
  await expect(page.getByRole('button', { name: 'Settings & diagnostics', exact: true })).toBeFocused();
  await page.keyboard.press('Shift+Tab');
  await expect(cloud).toBeFocused();
  const buttonBounds = await cloud.boundingBox();
  expect(buttonBounds!.x).toBeGreaterThanOrEqual(0);
  expect(buttonBounds!.x + buttonBounds!.width).toBeLessThanOrEqual(width);
  await page.keyboard.press('Enter');
  await expect(page.getByRole('heading', { name: 'Cloud runs', exact: true })).toBeVisible();
  await page.getByLabel('Current project').focus();
  await page.keyboard.press('Tab');
  await expect(page.getByLabel('New project', { exact: true })).toBeFocused();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
  expect(state.posts).toEqual([]);
});

test('scene cards support keyboard selection and changing the cloud target clears policy and consent', async ({ page }) => {
  const state = await fixture(page);
  state.profiles.push({ ...profile, id: 'second-scene', label: 'Second cup scene' });
  await page.getByRole('button', { name: 'Refresh simulation jobs' }).click();
  await expect(page.getByRole('radio', { name: 'Second cup scene', exact: true })).toBeVisible();
  await page.getByRole('radio', { name: 'Generated act-export', exact: true }).check();
  await acknowledge(page);
  await expect(page.getByRole('button', { name: 'Start experimental simulation' })).toBeEnabled();
  await page.getByRole('radio', { name: profile.label, exact: true }).focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('radio', { name: 'Second cup scene', exact: true })).toBeChecked();
  await expect(page.getByRole('radiogroup', { name: 'Native policy', exact: true }).locator('input:checked')).toHaveCount(0);
  await expect(page.getByRole('checkbox', { name: /experimental, paid cloud rollout/ })).not.toBeChecked();
  await expect(page.getByRole('button', { name: 'Start experimental simulation' })).toBeDisabled();
  expect(state.posts).toEqual([]);
});

test('an empty policy library has a direct import path without opening a technical menu', async ({ page }) => {
  const state = await fixture(page); state.artifacts = [];
  await page.getByRole('button', { name: 'Refresh simulation jobs' }).click();
  await expect(page.getByText('No compatible policy yet', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Import a policy', exact: true }).click();
  await expect(page.getByRole('radio', { name: 'Import package', exact: true })).toBeChecked();
  await expect(page.getByLabel('Native policy TAR', { exact: true })).toBeVisible();
  await expectImportBlocked(page);
  expect(state.posts).toEqual([]);
});

test('upload cancellation stays reachable when its simulation profile disappears', async ({ page }) => {
  const state = await fixture(page);
  let release!: () => void, uploads = 0;
  const heldUpload = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/api/v1/projects/alpha/model-imports?*', async route => {
    uploads += 1;
    await heldUpload;
    // The browser may already have cancelled this held request via XHR.abort().
    try { await route.abort(); } catch { /* No response is needed for an aborted upload. */ }
  });
  try {
    await page.getByRole('radio', { name: 'Import package', exact: true }).check();
    await page.getByLabel('Native policy TAR', { exact: true }).setInputFiles({ name: 'held-policy.tar', mimeType: 'application/x-tar', buffer: Buffer.from('Generated upload cancellation fixture') });
    await page.getByRole('button', { name: 'Upload and validate policy', exact: true }).click();
    await expect.poll(() => uploads).toBe(1);
    await expect(page.getByRole('button', { name: 'Stop upload', exact: true })).toBeVisible();

    state.profiles = [];
    await page.getByRole('button', { name: 'Refresh simulation jobs', exact: true }).click();
    await expect(page.getByText('Simulator not connected', { exact: true })).toBeVisible();
    const stop = page.getByRole('button', { name: 'Stop upload', exact: true });
    await expect(stop).toBeEnabled();
    await stop.click();

    await expect(page.getByRole('region', { name: 'Native Isaac simulation', exact: true }).getByRole('alert')).toContainText('submission outcome is unverified');
    await expect(stop).toHaveCount(0);
    const acknowledgment = page.getByRole('button', { name: 'I checked the jobs; allow a new request', exact: true });
    await expect(acknowledgment).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Start experimental simulation', exact: true })).toBeDisabled();
    await page.getByRole('button', { name: 'Refresh simulation jobs', exact: true }).click();
    await expect(acknowledgment).toBeEnabled();
    expect(uploads).toBe(1);
    expect(state.posts).toEqual([]);
  } finally { release(); }
});

async function expectImportBlocked(page: Page) {
  const upload = page.getByRole('button', { name: 'Upload and validate policy' });
  if (await upload.count()) await expect(upload).toBeDisabled();
  else await expect(page.getByRole('radio', { name: 'Import package', exact: true })).toBeDisabled();
}

async function openNativeAgain(page: Page, reload = false) {
  if (reload) { await page.reload(); await expect(page.getByLabel('Current project')).toHaveValue('alpha'); }
  else await page.getByRole('button', { name: 'Dataset', exact: true }).click();
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  await page.getByRole('button', { name: '3D simulation', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Native Isaac simulation', exact: true })).toBeVisible();
}
async function prepareExplicitRun(page: Page) {
  await page.getByRole('radio', { name: 'Generated act-export', exact: true }).check();
  await page.getByLabel('Simulation timeout (seconds)').fill('600');
  await acknowledge(page);
}

for (const boundary of ['navigation', 'reload'] as const) test(`native launch uncertainty survives ${boundary} without another submission`, async ({ page }) => {
  const state = await fixture(page); state.submit = 'lost';
  await page.route('**/api/v1/projects/alpha/jobs', route => route.fulfill({ json: [] }));
  await prepareExplicitRun(page);
  await page.getByRole('button', { name: 'Start experimental simulation' }).click();
  await expect(page.locator('.native-simulation')).toContainText('outcome is unverified');
  await openNativeAgain(page, boundary === 'reload');
  await expect(page.locator('.native-simulation')).toContainText('outcome is unverified');
  await expect(page.locator('.native-simulation')).toContainText('act-export');
  await expect(page.locator('.native-simulation')).toContainText('cup-fixture');
  await expect(page.getByRole('button', { name: 'Start experimental simulation' })).toBeDisabled();
  await expectImportBlocked(page);
  await expect(page.getByRole('button', { name: 'I checked the jobs; allow a new request' })).toBeDisabled();
  expect(state.posts).toHaveLength(1);
});

test('native pending launch remains owned after leaving the panel before its lost receipt', async ({ page }) => {
  const state = await fixture(page); let release!: () => void;
  const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/api/v1/projects/alpha/policy-jobs', async route => {
    state.posts.push({ path: new URL(route.request().url()).pathname, body: route.request().postDataJSON() });
    await gate; await route.abort().catch(() => undefined);
  });
  try {
    await prepareExplicitRun(page); await page.getByRole('button', { name: 'Start experimental simulation' }).click();
    await expect.poll(() => state.posts.length).toBe(1);
    await openNativeAgain(page);
    await expectImportBlocked(page);
    await expect(page.getByRole('radio', { name: 'Generated act-export', exact: true })).toBeDisabled();
    release();
    await expect(page.locator('.native-simulation')).toContainText('outcome is unverified');
    await expect(page.getByRole('button', { name: 'I checked the jobs; allow a new request' })).toBeDisabled();
    expect(state.posts).toHaveLength(1);
  } finally { release(); }
});

test('native history refresh started before uncertainty cannot authorize a new request', async ({ page }) => {
  const state = await fixture(page); state.submit = 'lost';
  await prepareExplicitRun(page);
  let release!: () => void; let held = 0; const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/api/v1/projects/alpha/jobs', async route => { held += 1; await gate; await route.fulfill({ json: [] }); });
  try {
    const refresh = page.getByRole('button', { name: 'Refresh simulation jobs' });
    await refresh.click(); await expect.poll(() => held).toBeGreaterThan(0);
    await page.getByRole('button', { name: 'Start experimental simulation' }).click();
    await expect(page.locator('.native-simulation')).toContainText('outcome is unverified');
    release(); await expect(refresh).toBeEnabled();
    await expect(page.getByRole('button', { name: 'I checked the jobs; allow a new request' })).toBeDisabled();
    expect(state.posts).toHaveLength(1);
  } finally { release(); }
});

for (const mismatch of ['runtime', 'timeout', 'experimental', 'manifest', 'array-status'] as const) test(`native run rejects ${mismatch} acknowledgment without clearing recovery`, async ({ page }) => {
  const state = await fixture(page);
  await page.route('**/api/v1/projects/alpha/policy-jobs', route => {
    const body = route.request().postDataJSON(); state.posts.push({ path: '/api/v1/projects/alpha/policy-jobs', body });
    const ack = job('mismatched', 'policy.run', 'running', body.artifact_id);
    ack.request.timeout_seconds = body.timeout_seconds;
    if (mismatch === 'runtime') ack.request.runtime_id = 'another-profile';
    if (mismatch === 'timeout') ack.request.timeout_seconds = 7200;
    if (mismatch === 'experimental') ack.request.simulation.experimental = false;
    if (mismatch === 'manifest') ack.simulation_target!.source_manifest_sha256 = 'd'.repeat(64);
    return route.fulfill({ status: 202, json: mismatch === 'array-status' ? { ...ack, status: ['running'] } : ack });
  });
  await prepareExplicitRun(page); await page.getByRole('button', { name: 'Start experimental simulation' }).click();
  await expect(page.getByRole('region', { name: 'Native simulation recovery' })).toContainText('act-export');
  await expect(page.getByRole('article', { name: 'Native simulation job details' })).toHaveCount(0);
  await openNativeAgain(page, true);
  await expect(page.getByRole('button', { name: 'Start experimental simulation' })).toBeDisabled();
  expect(state.posts).toHaveLength(1);
});

for (const outcome of ['lost', 'wrong-timeout', 'redirect'] as const) test(`native upload ${outcome} preserves explicit file context on reload`, async ({ page }) => {
  const state = await fixture(page); let followed = 0;
  // Playwright routes intercept only the first redirect request. Use a real
  // disposable loopback target to exercise XHR's final responseURL guard.
  const server = createServer((request, response) => {
    response.setHeader('Access-Control-Allow-Origin', '*');
    response.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
    response.setHeader('Access-Control-Allow-Headers', 'content-type');
    if (request.method === 'OPTIONS') { response.writeHead(204); response.end(); return; }
    followed += 1; request.resume();
    response.writeHead(202, { 'Content-Type': 'application/json' });
    response.end(JSON.stringify(job('redirected', 'policy.import', 'queued')));
  });
  let redirectUrl = '';
  if (outcome === 'redirect') {
    await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
    const address = server.address();
    if (!address || typeof address === 'string') throw new Error('Disposable redirect fixture did not bind');
    redirectUrl = `http://127.0.0.1:${address.port}/redirected`;
  }
  try {
    await page.route('**/api/v1/projects/alpha/model-imports?*', route => {
      state.posts.push({ path: '/api/v1/projects/alpha/model-imports', body: null });
      if (outcome === 'lost') return route.abort();
      if (outcome === 'redirect') return route.fulfill({ status: 307, headers: { location: redirectUrl } });
      const ack = job('upload-mismatch', 'policy.import', 'queued'); ack.request.timeout_seconds = 7200;
      return route.fulfill({ status: 202, json: ack });
    });
    await page.getByRole('radio', { name: 'Import package', exact: true }).check();
    const bytes = Buffer.from('Generated upload transport fixture');
    await page.getByLabel('Native policy TAR', { exact: true }).setInputFiles({ name: 'explicit-generated.tar', mimeType: 'application/x-tar', buffer: bytes });
    await page.getByRole('button', { name: 'Upload and validate policy' }).click();
    await expect(page.getByRole('region', { name: 'Native simulation recovery' })).toContainText('explicit-generated.tar');
    await expect(page.getByRole('article', { name: 'Native simulation job details' })).toHaveCount(0);
    await openNativeAgain(page, true);
    const recovery = page.getByRole('region', { name: 'Native simulation recovery' });
    await expect(recovery).toContainText(` ${bytes.length} bytes`); await expect(recovery).toContainText('cup-fixture');
    await expectImportBlocked(page);
    expect(state.posts).toHaveLength(1);
    // XHR itself follows a 307. Refusing the receipt cannot undo that transfer.
    expect(followed).toBe(outcome === 'redirect' ? 1 : 0);
  } finally {
    server.closeAllConnections();
    if (server.listening) await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
  }
});

test('native fetch launch refuses redirects without replaying the POST', async ({ page }) => {
  const state = await fixture(page); let followed = 0;
  await page.route('**/api/v1/projects/alpha/policy-jobs', route => { state.posts.push({ path: '/api/v1/projects/alpha/policy-jobs', body: route.request().postDataJSON() }); return route.fulfill({ status: 307, headers: { location: '/api/v1/fixture-run-redirect' } }); });
  await page.route('**/api/v1/fixture-run-redirect', route => { followed += 1; return route.fulfill({ json: {} }); });
  await prepareExplicitRun(page); await page.getByRole('button', { name: 'Start experimental simulation' }).click();
  await expect(page.getByRole('region', { name: 'Native simulation recovery' })).toBeVisible();
  expect(state.posts).toHaveLength(1); expect(followed).toBe(0);
});

test('native cancellation uncertainty survives reload with the exact selected job', async ({ page }) => {
  const state = await fixture(page); state.jobs.push(job('cancel-owned'));
  await page.route('**/api/v1/jobs/cancel-owned/cancel', route => { state.cancels.push('cancel-owned'); return route.abort(); });
  await page.getByRole('button', { name: 'Refresh simulation jobs' }).click();
  await page.getByLabel('Saved simulation job').selectOption('cancel-owned');
  await page.getByRole('button', { name: 'Cancel selected native job' }).click(); await page.getByRole('button', { name: 'Confirm cancellation' }).click();
  await expect(page.getByRole('region', { name: 'Native simulation recovery' })).toContainText('Action: cancel');
  await openNativeAgain(page, true);
  await expect(page.getByRole('region', { name: 'Native simulation recovery' })).toContainText('Job: cancel-owned');
  await page.getByLabel('Saved simulation job').selectOption('cancel-owned');
  await expect(page.getByRole('button', { name: 'Cancel selected native job' })).toBeDisabled();
  expect(state.cancels).toEqual(['cancel-owned']);
});

test('native cancellation rechecks selection generation after A to B to A', async ({ page }) => {
  const state = await fixture(page); const first = job('cancel-first'); state.jobs.push(first, job('cancel-second'));
  await page.getByRole('button', { name: 'Refresh simulation jobs' }).click();
  let release!: () => void; let reads = 0; const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/api/v1/jobs/cancel-first', async route => { reads += 1; await gate; await route.fulfill({ json: first }); });
  try {
    await page.getByLabel('Saved simulation job').selectOption(first.id);
    await page.getByRole('button', { name: 'Cancel selected native job' }).click(); await page.getByRole('button', { name: 'Confirm cancellation' }).click();
    await expect.poll(() => reads).toBe(1);
    await page.getByLabel('Saved simulation job').selectOption('cancel-second');
    await page.getByLabel('Saved simulation job').selectOption(first.id); release();
    await expect(page.locator('.native-simulation').getByRole('alert')).toHaveText('Selection changed; cancellation was not submitted.');
    expect(state.cancels).toEqual([]); expect(state.posts).toEqual([]);
  } finally { release(); }
});

for (const failure of ['pending-write', 'receipt-write', 'cleanup'] as const) test(`native ${failure} storage failure preserves request ownership and acknowledged ID`, async ({ page }) => {
  const state = await fixture(page);
  await page.evaluate(which => {
    const set = Storage.prototype.setItem, remove = Storage.prototype.removeItem;
    Storage.prototype.setItem = function(key, value) {
      if ((which === 'pending-write' && key.includes('job-attempt:policy.run.simulation:')) || (which === 'receipt-write' && key.includes('native-simulation-receipt:'))) throw new Error('Generated storage failure');
      return set.call(this, key, value);
    };
    Storage.prototype.removeItem = function(key) { if (which === 'cleanup' && key.includes('job-attempt:policy.run.simulation:')) throw new Error('Generated cleanup failure'); return remove.call(this, key); };
  }, failure);
  await prepareExplicitRun(page); await page.getByRole('button', { name: 'Start experimental simulation' }).click();
  await expect(page.locator('.native-simulation').getByRole('alert')).toContainText('Browser session storage is unavailable');
  if (failure !== 'pending-write') await expect(page.getByRole('article', { name: 'Native simulation job details' })).toHaveAttribute('data-job-id', 'submitted');
  expect(state.posts).toHaveLength(failure === 'pending-write' ? 0 : 1);
  await openNativeAgain(page);
  if (failure !== 'pending-write') {
    await expect(page.getByRole('article', { name: 'Native simulation job details' })).toHaveAttribute('data-job-id', 'submitted');
    await expect(page.getByRole('button', { name: 'Cancel selected native job' })).toBeDisabled();
    await page.getByText('Prepare another run', { exact: true }).click();
  }
  await expectImportBlocked(page);
  await expect(page.getByRole('button', { name: 'Start experimental simulation' })).toBeDisabled();
  expect(state.posts).toHaveLength(failure === 'pending-write' ? 0 : 1);
});

test('native late acknowledged receipt survives unmount without replacing a manually selected recording', async ({ page }) => {
  const state = await fixture(page); const done = job('saved-video', 'policy.run', 'succeeded');
  done.result = { artifacts: [{ ...artifact('record', 'act', 'simulation_record'), job_id: done.id }], reports: [{ stage: 'simulation', artifacts: [{ path: 'artifacts/outputs/video.mp4' }] }] };
  state.jobs.push(done); await page.getByRole('button', { name: 'Refresh simulation jobs' }).click();
  let release!: () => void; const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/api/v1/projects/alpha/policy-jobs', async route => {
    const body = route.request().postDataJSON(); state.posts.push({ path: '/api/v1/projects/alpha/policy-jobs', body });
    const ack = job('late-ack', 'policy.run', 'running', body.artifact_id); ack.request.timeout_seconds = body.timeout_seconds;
    await gate; await route.fulfill({ status: 202, json: ack });
  });
  await page.route('**/api/v1/jobs/saved-video/simulation-media/video', route => route.fulfill({ status: 200, contentType: 'video/mp4', body: Buffer.alloc(0) }));
  try {
    await prepareExplicitRun(page); await page.getByRole('button', { name: 'Start experimental simulation' }).click();
    await expect.poll(() => state.posts.length).toBe(1); await openNativeAgain(page);
    await page.getByLabel('Saved simulation job').selectOption(done.id); release();
    await expect(page.getByRole('region', { name: 'Native submission receipt' })).toContainText('late-ack');
    await expect(page.getByRole('article', { name: 'Native simulation job details' })).toHaveAttribute('data-job-id', done.id);
    await expect(page.getByRole('link', { name: 'Download simulation record' })).toHaveAttribute('href', '/api/v1/projects/alpha/artifacts/record/download');
    await page.getByRole('button', { name: 'View acknowledged job' }).click();
    await expect(page.getByRole('article', { name: 'Native simulation job details' })).toHaveAttribute('data-job-id', 'late-ack');
    await expect(page.getByText('Execution target', { exact: true }).locator('..')).toContainText('Awaiting recorded cloud target');
    await expect(page.getByText('Local package validation', { exact: true })).toHaveCount(0);
    await expect(page.getByLabel('Recorded cup rollout', { exact: true })).toHaveCount(0);
    await expect(page.getByRole('link', { name: 'Download simulation record' })).toHaveCount(0);
    expect(state.posts).toHaveLength(1);
  } finally { release(); }
});

test('native pending restart preserves original context and project isolation without trusting cached media', async ({ page }) => {
  const state = await fixture(page);
  await page.evaluate(() => {
    sessionStorage.setItem('firebird:job-attempt:policy.run.simulation:alpha', JSON.stringify({ state: 'pending', message: 'Project: alpha · Profile: cup-fixture · Policy: act-export · Budget: 600 seconds' }));
  });
  await openNativeAgain(page, true);
  await expect(page.getByRole('button', { name: '3D simulation', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('region', { name: 'Native simulation recovery' })).toContainText('Policy: act-export');
  await expect(page.getByRole('button', { name: 'I checked the jobs; allow a new request' })).toBeDisabled();
  await page.getByLabel('Current project').selectOption('beta');
  await page.getByRole('button', { name: '3D simulation', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Native simulation recovery' })).toHaveCount(0);
  await page.getByLabel('Current project').selectOption('alpha');
  await expect(page.getByRole('region', { name: 'Native simulation recovery' })).toContainText('Policy: act-export');
  expect(state.posts).toEqual([]);
});

for (const both of [false, true]) test(`native recovery entry ${both ? 'asks which unresolved workflow to inspect' : 'beats newer engine history and preserves manual mode'}`, async ({ page }) => {
  const state = await fixture(page);
  await page.route('**/api/v1/projects/alpha/jobs', route => route.fulfill({ json: [{ ...job('newer-engine', 'policy.run', 'succeeded'), simulation_target: null, request: { operation: 'policy.run', runtime_id: 'browser-success' } }] }));
  await page.evaluate(multiple => {
    sessionStorage.setItem('firebird:job-attempt:policy.run.simulation:alpha', JSON.stringify({ state: 'pending', message: 'Original native launch · act-export · 600 seconds' }));
    if (multiple) sessionStorage.setItem('firebird:job-attempt:policy.run.replay:alpha', JSON.stringify({ state: 'pending', message: 'Original CPU replay' }));
  }, both);
  await page.reload(); await expect(page.getByLabel('Current project')).toHaveValue('alpha');
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  const nativeMode = page.getByRole('button', { name: '3D simulation', exact: true });
  if (both) {
    await expect(page.getByRole('region', { name: 'Workflow selection status' })).toContainText('Both observation replay and native simulation have unresolved requests');
    await expect(nativeMode).toHaveAttribute('aria-pressed', 'false');
    await nativeMode.click();
  } else await expect(nativeMode).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('region', { name: 'Native simulation recovery' })).toContainText('Original native launch');
  await page.getByRole('button', { name: 'Check inference', exact: true }).click();
  await page.getByRole('button', { name: 'Dataset', exact: true }).click();
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Check inference', exact: true })).toHaveAttribute('aria-pressed', 'true');
  expect(await page.evaluate(() => sessionStorage.getItem('firebird:job-attempt:policy.run.simulation:alpha'))).toContain('Original native launch');
  expect(state.posts).toEqual([]);
});
