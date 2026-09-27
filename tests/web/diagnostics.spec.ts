import { createServer } from 'node:http';
import { expect, test, type APIRequestContext, type Page } from '@playwright/test';
import { waitForJob } from './job-waiter';

const configured = `http://127.0.0.1:${process.env.FIREBIRD_DIAGNOSTICS_CONFIGURED_PORT ?? '8766'}`;

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
  await waitForJob(request, job.id, 'succeeded', { origin: configured });
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
  await page.getByText('Protocol details', { exact: true }).click();
  await expect(page.getByText('Evaluates LIBERO Spatial tasks 1,2', { exact: false })).toContainText('full 280-step horizon');
  await page.getByRole('button', { name: 'Edit diagnostic settings', exact: true }).click();
  await page.getByRole('combobox', { name: 'Task suite', exact: true }).selectOption('libero_object');
  await page.getByRole('button', { name: 'Diagnostics', exact: true }).click();
  await expect(mode).toBeEnabled();
  await mode.selectOption('engine');
  await page.getByText('Protocol details', { exact: true }).click();
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
  await waitForJob(request, job.id, 'succeeded', { origin: configured });
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
  await page.getByText('Protocol details', { exact: true }).click();
  await expect(page.getByText('Evaluates LIBERO Object task 0', { exact: false })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Start diagnostics', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Open Quantize', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Quantization jobs', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'New quantization', exact: true })).toBeEnabled();
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
  await waitForJob(request, job.id, 'cancelled', { origin: configured });
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

test('worker waits reject wrong terminal states and report bounded timeouts without cancelling work', async ({ request }) => {
  const id = await project(request, configured);
  const artifact = await policy(request, id);
  const response = await request.post(`${configured}/api/v1/projects/${id}/policy-jobs`, {
    data: { operation: 'policy.evaluate', runtime_id: 'slow', artifact_id: artifact },
  });
  expect(response.status()).toBe(202);
  const job = await response.json();
  try {
    await expect(waitForJob(request, job.id, 'succeeded', {
      origin: configured, timeoutMs: 500,
    })).rejects.toThrow(new RegExp(`within 500ms[.] Last observation: (Job ${job.id}: no snapshot received|.*${job.id}.*"status":"(queued|running)")`));
    const pending = await request.get(`${configured}/api/v1/jobs/${job.id}`);
    expect(['queued', 'running']).toContain((await pending.json()).status);
  } finally {
    // The wait observes only. Always clean up the owned slow fixture explicitly.
    const cancelled = await request.post(`${configured}/api/v1/jobs/${job.id}/cancel`);
    expect(cancelled.status()).toBe(200);
    await waitForJob(request, job.id, 'cancelled', { origin: configured });
  }
  await expect(waitForJob(request, job.id, 'succeeded', {
    origin: configured,
  })).rejects.toThrow(`Job reached cancelled; expected succeeded`);
});

test('worker deadline retains diagnostics when its HTTP observation stalls', async ({ request }) => {
  // This transport-only fixture has no model metrics or application side effects.
  for (const firstSnapshot of [false, true]) {
    let reads = 0;
    const server = createServer((_incoming, response) => {
      reads += 1;
      if (firstSnapshot && reads === 1) {
        response.writeHead(200, { 'Content-Type': 'application/json' });
        response.end(JSON.stringify({ id: 'http-timeout-fixture', status: 'running', stage: 'evaluation' }));
      }
      // Other reads remain pending until the observer's own deadline aborts them.
    });
    await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
    try {
      const address = server.address();
      if (!address || typeof address === 'string') throw new Error('Missing test server address');
      const observation = firstSnapshot ? '"status":"running"' : 'no snapshot received';
      await expect(waitForJob(request, 'http-timeout-fixture', 'succeeded', {
        origin: `http://127.0.0.1:${address.port}`, timeoutMs: 500,
      })).rejects.toThrow(new RegExp(`within 500ms[.] Last observation:.*${observation}`));
    } finally {
      server.closeAllConnections();
      await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
    }
  }
});
