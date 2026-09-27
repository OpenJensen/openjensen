import { expect, test, type Page } from '@playwright/test';

const timestamp = '2026-09-27T12:00:00Z';
const profile = { id: 'cup-fixture', label: 'Generated test cup profile', architectures: ['act', 'smolvla'], experimental: true, task_object: 'cup' };
function artifact(id: string, architecture = 'act', format = 'inference_export', metadata = {}) {
  return { id, project_id: 'alpha', job_id: 'source', label: `Generated ${id}`, format, path: 'fixture', manifest_sha256: 'a'.repeat(64), file_bytes: 4096, parent_ids: [], metadata: { architecture, ...metadata } };
}
function job(id: string, kind = 'policy.run', status = 'running', source = 'act-export') {
  return { id, project_id: 'alpha', kind, status, created_at: timestamp, updated_at: timestamp, stage: 'simulation', error: null,
    request: { operation: kind, runtime_id: profile.id, artifact_id: kind === 'policy.run' ? source : null, source_id: kind === 'policy.import' ? 'upload-id' : null, simulation: { profile_id: profile.id, experimental: kind === 'policy.run' }, timeout_seconds: 7200 },
    compute_target: null, simulation_target: kind === 'policy.run' ? { profile_id: profile.id, profile_sha256: 'b'.repeat(64), provider: 'gcp', accelerators: ['L4', 'H100'], source_manifest_sha256: 'a'.repeat(64) } : null, result: null as null | Record<string, unknown> };
}
async function fixture(page: Page) {
  const state = { profiles: [profile], optionsError: false, jobsError: false, submit: 'ok', posts: [] as { path: string; body: unknown }[], jobs: [] as ReturnType<typeof job>[], artifacts: [artifact('act-export'), artifact('smol-export', 'smolvla'), artifact('periodic', 'act', 'training_checkpoint'), artifact('remote', 'act', 'native_checkpoint', { storage: 'gcs', remote_uri: 'gs://fixture' }), artifact('unsupported', 'diffusion')], uploaded: Buffer.alloc(0), cancels: [] as string[], events: [] as string[] };
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
    if (path.endsWith('/events')) { const id = path.split('/').at(-2)!; state.events.push(id); return route.fulfill({ json: [{ sequence: 1, stage: 'simulation', timestamp, message: `Generated event for ${id}`, data: {} }] }); }
    if (path.startsWith('/api/v1/jobs/')) { const id = path.split('/').at(-1)!; const selected = state.jobs.find(item => item.id === id); if (selected) return route.fulfill({ json: selected }); }
    return route.continue();
  });
  await page.goto('/'); await expect(page.getByLabel('Current project')).toHaveValue('alpha');
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  await page.getByRole('button', { name: 'Native Isaac · ACT / SmolVLA', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Native Isaac simulation', exact: true })).toBeVisible();
  return state;
}
const acknowledge = (page: Page) => page.getByRole('checkbox', { name: /experimental, paid cloud rollout/ }).check();

test('native Run preserves engine navigation and fails closed when profiles are missing', async ({ page }) => {
  const state = await fixture(page); state.profiles = [];
  await page.reload(); await page.getByRole('button', { name: 'Run', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Run jobs', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Native Isaac · ACT / SmolVLA' }).click();
  await expect(page.getByText(/No Isaac simulation profile is configured/)).toBeVisible();
  await expect(page.getByRole('button', { name: 'Start experimental simulation' })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Upload and validate policy' })).toBeDisabled();
  expect(state.posts).toEqual([]);
  await page.getByRole('button', { name: 'Evaluate', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Evaluation purpose' })).toContainText('scored Isaac evaluation is not configured');
});

test('native import sends exact TAR bytes once and does not start a GPU run', async ({ page }) => {
  const state = await fixture(page);
  const bytes = Buffer.from('Explicit generated TAR transport fixture; worker validation is covered by API tests.');
  await page.getByLabel('Native policy TAR', { exact: true }).setInputFiles({ name: 'synthetic-policy.tar', mimeType: 'application/x-tar', buffer: bytes });
  await page.getByRole('button', { name: 'Upload and validate policy' }).dblclick();
  await expect(page.getByRole('article', { name: 'Native simulation job details' })).toContainText('The native package is saved');
  expect(state.uploaded).toEqual(bytes); expect(state.posts).toHaveLength(1);
  await expect(page.getByLabel('Native policy', { exact: true })).toHaveValue('');
  await expect(page.getByLabel('Native policy', { exact: true }).locator('option[value="uploaded-policy"]')).toHaveCount(1);
  await expect(page.getByRole('button', { name: 'Start experimental simulation' })).toBeDisabled();
});

for (const architecture of ['act', 'smol']) test(`native ${architecture} launch requires explicit source and consent with bounded exact payload`, async ({ page }) => {
  const state = await fixture(page);
  const select = page.getByLabel('Native policy', { exact: true });
  await expect(select.locator('option')).toHaveText(['Choose a saved native policy', 'Generated act-export · ACT · act-expo', 'Generated smol-export · SMOLVLA · smol-exp']);
  await select.selectOption(`${architecture}-export`); await acknowledge(page);
  await page.getByLabel('Simulation timeout (seconds)').fill('7201');
  await expect(page.getByRole('button', { name: 'Start experimental simulation' })).toBeDisabled();
  await page.getByLabel('Simulation timeout (seconds)').fill('600');
  await page.getByRole('button', { name: 'Start experimental simulation' }).dblclick();
  await expect(page.getByRole('article', { name: 'Native simulation job details' })).toHaveAttribute('data-job-id', 'submitted');
  expect(state.posts).toEqual([{ path: '/api/v1/projects/alpha/policy-jobs', body: { operation: 'policy.run', runtime_id: profile.id, artifact_id: `${architecture}-export`, simulation: { profile_id: profile.id, experimental: true }, timeout_seconds: 600 } }]);
  await expect(page.getByRole('region', { name: 'Native simulation event log' })).toContainText('Generated event for submitted');
  await expect(page.getByText('Cup pickup success', { exact: true }).locator('..')).toContainText('Not measured');
});

test('lost launch receipt pauses submissions and recovers the actual job without retry', async ({ page }) => {
  const state = await fixture(page); state.submit = 'lost';
  await page.getByLabel('Native policy', { exact: true }).selectOption('act-export'); await acknowledge(page);
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
  await page.getByLabel('Native policy', { exact: true }).selectOption('act-export'); await acknowledge(page);
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
  await page.getByLabel('Native policy', { exact: true }).selectOption('smol-export'); await acknowledge(page);
  await page.getByLabel('Current project').selectOption('beta');
  await expect(page.getByLabel('Native policy', { exact: true })).toHaveValue('');
  await expect(page.getByRole('checkbox', { name: /experimental, paid cloud rollout/ })).not.toBeChecked();
  await expect(page.getByRole('button', { name: 'Start experimental simulation' })).toBeDisabled();
  expect(state.posts).toHaveLength(0);
  await page.setViewportSize({ width: 320, height: 900 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
});


test('changing the available profile cannot carry consent to a different cloud target', async ({ page }) => {
  const state = await fixture(page);
  await page.getByLabel('Native policy', { exact: true }).selectOption('act-export'); await acknowledge(page);
  state.profiles = [{ ...profile, id: 'replacement-cup', label: 'Different generated profile' }];
  await page.getByRole('button', { name: 'Refresh simulation jobs' }).click();
  await expect(page.getByLabel('Isaac profile', { exact: true })).toHaveValue('replacement-cup');
  await expect(page.getByRole('checkbox', { name: /experimental, paid cloud rollout/ })).not.toBeChecked();
  await expect(page.getByRole('button', { name: 'Start experimental simulation' })).toBeDisabled();
  expect(state.posts).toHaveLength(0);
});
