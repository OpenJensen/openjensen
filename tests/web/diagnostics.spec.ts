import { expect, test, type APIRequestContext, type Page } from '@playwright/test';

const configured = 'http://127.0.0.1:8766';

async function project(request: APIRequestContext, origin = '') {
  const response = await request.post(`${origin}/api/v1/projects`, { data: { name: 'Diagnostics browser test' } });
  expect(response.ok()).toBeTruthy();
  return (await response.json()).id as string;
}

async function openDiagnostics(page: Page, id: string, origin = '', preferences = {}) {
  await page.addInitScript(({ id, preferences }) => {
    localStorage.setItem('firebird.project', id);
    localStorage.setItem(`firebird.workflow.${id}`, JSON.stringify(preferences));
  }, { id, preferences });
  await page.goto(`${origin}/`);
  await page.getByRole('button', { name: 'Settings & diagnostics', exact: true }).click();
  await page.getByRole('button', { name: 'Diagnostics', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Run diagnostics', exact: true })).toBeVisible();
}

async function policy(request: APIRequestContext, id: string) {
  const response = await request.post(`${configured}/api/v1/projects/${id}/policy-jobs`, {
    data: { operation: 'policy.import', runtime_id: 'fixture', source_id: 'fixture-source' },
  });
  expect(response.status()).toBe(202);
  const job = await response.json();
  await expect.poll(async () => (await (await request.get(`${configured}/api/v1/jobs/${job.id}`)).json()).status).toBe('succeeded');
  const artifacts = await (await request.get(`${configured}/api/v1/projects/${id}/artifacts`)).json();
  expect(artifacts[0].metadata.fixture_only).toBe(true);
  return artifacts[0].id as string;
}

test('shows complete reference numbers with separate hardware and honest missing values', async ({ page, request }, testInfo) => {
  await openDiagnostics(page, await project(request));
  const table = page.getByRole('table', { name: /NVIDIA L4 · 8 vCPUs/ });
  const native = table.getByRole('row').filter({ has: page.getByRole('rowheader', { name: 'native-bf16', exact: true }) });
  await expect(native).toContainText('342.59');
  await expect(native).toContainText('37.96');
  await expect(native).toContainText('1186');
  await expect(native).toContainText('16/20');
  const q4 = table.getByRole('row').filter({ has: page.getByRole('rowheader', { name: 'cpp-Q4_0', exact: true }) });
  await expect(q4).toContainText('4/20');
  await expect(q4).toContainText('13 lost / 1 gained');
  await expect(table.getByRole('row')).toHaveCount(11);
  await expect(page.getByRole('button', { name: 'Start diagnostics', exact: true })).toBeDisabled();
  await expect(page.getByText('No execution target is configured.', { exact: false })).toBeVisible();
  await expect(page.getByRole('link', { name: 'View source report' })).toHaveAttribute('href', /e5866f0.*dedicated-l4\/REPORT.md$/);
  await page.getByRole('combobox', { name: 'Reference hardware' }).selectOption('existingL4');
  await expect(page.getByRole('table', { name: /NVIDIA L4 · 12 vCPUs/ })).toContainText('348.80');
  await expect(page.getByRole('columnheader', { name: 'Process to first action (s)', exact: true })).toHaveCount(0);
  await page.getByRole('combobox', { name: 'Reference hardware' }).selectOption('rtx3070');
  const rtx = page.getByRole('table', { name: /NVIDIA RTX 3070/ });
  await expect(rtx.getByRole('row').filter({ has: page.getByRole('rowheader', { name: 'cpp-Q4_0', exact: true }) })).toContainText('8/20');
  await expect(rtx).not.toContainText(/Not recorded|Not collected|Deferred|host failed/);
  const missing = rtx.getByRole('row').filter({ has: page.getByRole('rowheader', { name: 'trtllm-fp16', exact: true }) });
  await expect(missing.getByRole('cell')).toHaveText(['—', '—', '—', '—', '—']);
  await page.getByRole('combobox', { name: 'Reference hardware' }).selectOption('dedicatedL4');
  const sizes = await page.evaluate(() => ({ width: document.documentElement.clientWidth, scroll: document.documentElement.scrollWidth }));
  expect(sizes.scroll).toBeLessThanOrEqual(sizes.width + 1);
  await page.screenshot({ path: testInfo.outputPath('diagnostics.png'), fullPage: true });
});

test('keeps diagnostics open when the current project finishes loading', async ({ page, request }) => {
  const id = await project(request);
  let releaseProjects!: () => void;
  const projectsReady = new Promise<void>(resolve => { releaseProjects = resolve; });
  await page.route('**/api/v1/projects', async route => {
    await projectsReady;
    await route.continue();
  });
  try {
    await openDiagnostics(page, id);
    await expect(page.getByText('Select or create a project to save your diagnostic results.', { exact: true })).toBeVisible();
  } finally {
    releaseProjects();
  }
  await expect(page.getByRole('combobox', { name: 'Current project', exact: true })).toHaveValue(id);
  await expect(page.getByText('No execution target is configured.', { exact: false })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Run diagnostics', exact: true })).toBeVisible();
});

test('keeps the Spatial protocol locked and labels the selected suite', async ({ page, request }) => {
  await openDiagnostics(page, await project(request), '', {
    suite: 'libero_spatial', mode: 'engine', steps: 10, taskIds: '1,2',
  });
  const mode = page.getByRole('combobox', { name: 'Diagnostic mode', exact: true });
  await expect(mode).toHaveValue('libero');
  await expect(mode).toBeDisabled();
  await expect(page.getByText('Evaluates LIBERO Spatial tasks 1,2', { exact: false })).toContainText('full 280-step horizon');
  await page.getByRole('button', { name: 'Edit diagnostic settings', exact: true }).click();
  await page.getByRole('combobox', { name: 'Task suite', exact: true }).selectOption('libero_object');
  await page.getByRole('button', { name: 'Diagnostics', exact: true }).click();
  await expect(mode).toBeEnabled();
  await mode.selectOption('engine');
  await expect(page.getByText('Checks loading, finite actions', { exact: false })).toBeVisible();
});

test('submits the selected policy through the real API and shows subprocess results', async ({ page, request }) => {
  const id = await project(request, configured);
  const artifact = await policy(request, id);
  await openDiagnostics(page, id, configured, { warmup: 2, repetitions: 4 });
  await page.getByRole('combobox', { name: 'Diagnostic policy', exact: true }).selectOption(artifact);
  const accepted = page.waitForResponse(response => response.url().endsWith(`/projects/${id}/policy-jobs`) && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Start diagnostics', exact: true }).click();
  const response = await accepted;
  expect(response.status()).toBe(202);
  expect(response.request().postDataJSON()).toMatchObject({ operation: 'policy.evaluate', runtime_id: 'fixture', artifact_id: artifact, evaluation: { mode: 'engine', warmups: 2, repetitions: 4 } });
  const job = await response.json();
  await expect.poll(async () => (await (await request.get(`${configured}/api/v1/jobs/${job.id}`)).json()).status).toBe('succeeded');
  await expect(page.getByRole('table', { name: 'Recorded policy measurements' })).toBeVisible();
  // This proves UI → API → worker wiring only. The worker is explicitly synthetic.
  const completed = await (await request.get(`${configured}/api/v1/jobs/${job.id}`)).json();
  expect(completed.result.reports[0].scope).toBe('test_fixture');
  await expect(page.getByRole('combobox', { name: 'Run', exact: true })).toHaveValue(job.id);
  await page.reload();
  await page.getByRole('button', { name: 'Settings & diagnostics', exact: true }).click();
  await page.getByRole('button', { name: 'Diagnostics', exact: true }).click();
  await expect(page.getByRole('table', { name: 'Recorded policy measurements' })).toBeVisible();
});

test('guides an empty project to Quantize and blocks unsupported simulation', async ({ page, request }) => {
  const id = await project(request, configured);
  await openDiagnostics(page, id, configured, { mode: 'libero' });
  await expect(page.getByText('Evaluates LIBERO Object task 0', { exact: false })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Start diagnostics', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Open Quantize', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Create a smaller policy', exact: true })).toBeVisible();
  const artifact = await policy(request, id);
  await page.getByRole('button', { name: 'Settings & diagnostics', exact: true }).click();
  await page.getByRole('button', { name: 'Diagnostics', exact: true }).click();
  await page.getByRole('combobox', { name: 'Diagnostic policy', exact: true }).selectOption(artifact);
  await page.getByRole('combobox', { name: 'Diagnostic execution target' }).selectOption('cpu-only');
  await expect(page.getByText('This target has no LIBERO simulator.', { exact: false })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Start diagnostics', exact: true })).toBeDisabled();
  await page.getByRole('combobox', { name: 'Diagnostic mode' }).selectOption('engine');
  await expect(page.getByRole('button', { name: 'Start diagnostics', exact: true })).toBeEnabled();
});

test('cancels an active diagnostic worker from the diagnostics tab', async ({ page, request }) => {
  const id = await project(request, configured);
  const artifact = await policy(request, id);
  await openDiagnostics(page, id, configured);
  await page.getByRole('combobox', { name: 'Diagnostic policy', exact: true }).selectOption(artifact);
  await page.getByRole('combobox', { name: 'Diagnostic execution target' }).selectOption('slow');
  const accepted = page.waitForResponse(response => response.url().endsWith(`/projects/${id}/policy-jobs`) && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Start diagnostics', exact: true }).click();
  const job = await (await accepted).json();
  await expect(page.getByRole('button', { name: 'Cancel run', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Cancel run', exact: true }).click();
  await expect.poll(async () => (await (await request.get(`${configured}/api/v1/jobs/${job.id}`)).json()).status).toBe('cancelled');
  await expect(page.getByRole('button', { name: 'Cancel run', exact: true })).toHaveCount(0);
});

test('keeps reference results visible and blocks launch when target discovery fails', async ({ page, request }) => {
  const id = await project(request);
  await page.route('**/api/v1/policy-options', route => route.fulfill({ status: 503, json: { detail: 'Target discovery unavailable' } }));
  await openDiagnostics(page, id);
  await expect(page.getByRole('alert').filter({ hasText: 'Target discovery unavailable' })).toBeVisible({ timeout: 15_000 });
  await expect(page.getByRole('button', { name: 'Start diagnostics', exact: true })).toBeDisabled();
  await expect(page.getByRole('table', { name: /NVIDIA L4 · 8 vCPUs/ })).toBeVisible();
});
