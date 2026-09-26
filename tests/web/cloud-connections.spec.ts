import { expect, test, type Page } from '@playwright/test';

type Provider = 'gcp';
type Connection = {
  provider: Provider;
  name: string;
  status: 'disconnected' | 'connected' | 'unverified' | 'setup_required' | 'error';
  config: { project_id: string; region: string } | null;
  identity: { account?: string | null; account_id?: string | null; principal?: string | null } | null;
  checked_at: string | null;
  message: string | null;
  setup_commands: string[];
};
type CloudPreferences = { enabled: boolean; default_gpu: string; disk_size_gb: number; idle_minutes: number };
type ComputeStatus = {
  status: 'unchecked' | 'ready' | 'setup_required' | 'error';
  configured: boolean;
  skypilot_installed: boolean;
  project_id: string | null;
  region: string | null;
  checked_at: string | null;
  message: string;
  setup_commands: string[];
};
const projectId = 'cloud-fixture';
const timestamp = '2026-09-26T12:00:00Z';
const cloudGpus = [
  { id: 'L4', label: 'NVIDIA L4 · 24 GB', gpu_memory_mib: 24576, instance_type: 'g2-standard-4', supported: true },
  { id: 'T4', label: 'NVIDIA T4 · 16 GB', gpu_memory_mib: 16384, instance_type: 'n1-highmem-4', supported: true },
  { id: 'A100', label: 'NVIDIA A100 · 40 GB', gpu_memory_mib: 40960, instance_type: 'a2-highgpu-1g', supported: true },
];

function disconnected(provider: Provider): Connection {
  return {
    provider, name: 'Google Cloud',
    status: 'disconnected', config: null, identity: null, checked_at: null, message: null,
    setup_commands: ['gcloud auth login'],
  };
}

function connected(provider: Provider, config: NonNullable<Connection['config']>): Connection {
  return {
    ...disconnected(provider), config, status: 'connected', checked_at: timestamp,
    identity: { account: 'robotics@example.test', account_id: null, principal: null },
  };
}

async function openCompute(page: Page) {
  await page.getByRole('button', { name: 'Settings & diagnostics', exact: true }).click();
  await page.getByRole('button', { name: 'Compute', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Google Cloud', exact: true })).toBeVisible();
}

async function mockWorkspace(page: Page, initial: Connection[] = [], initialCompute: { gcp?: Partial<CloudPreferences>; status?: Partial<ComputeStatus> } = {}) {
  const providers: Record<Provider, Connection> = { gcp: disconnected('gcp') };
  for (const item of initial) providers[item.provider] = item;
  const writes: { provider: Provider; action: string; body: unknown }[] = [];
  const computeWrites: unknown[] = [];
  let local = { enabled: false, label: 'Local machine' };
  let gcp: CloudPreferences = { enabled: true, default_gpu: 'A100', disk_size_gb: 200, idle_minutes: 10, ...initialCompute.gcp };
  let checkedStatus: Partial<ComputeStatus> | null = initialCompute.status ?? null;
  let availableGpus = ['L4', 'T4', 'A100'];
  const mutations: string[] = [];
  const unexpected: string[] = [];
  const reads: string[] = [];
  const replies: { status?: number; connection?: Connection; detail?: string }[] = [];
  function computeResponse() {
    const config = providers.gcp.config;
    const gcp_status: ComputeStatus = {
      status: 'unchecked', configured: !!config, skypilot_installed: true,
      project_id: config?.project_id ?? null, region: config?.region ?? null, checked_at: null,
      message: config ? 'Check SkyPilot setup to verify application credentials and regional GPU offerings.' : 'Connect Google Cloud in Settings first.',
      setup_commands: config ? ['gcloud auth application-default login'] : [],
      ...checkedStatus,
    };
    const gpu_options = cloudGpus.map(gpu => {
      const unavailable_reason = !gcp.enabled ? 'Enable Google Cloud training in Settings.'
        : gcp_status.status !== 'ready' ? gcp_status.message
        : !availableGpus.includes(gpu.id) ? `SkyPilot does not offer this single-GPU machine in ${gcp_status.region}.`
        : null;
      return { ...gpu, accelerator: gpu.id, gpu_count: 1, available: unavailable_reason === null, unavailable_reason };
    });
    return { local, gcp, runtimes: [], gcp_status, gpu_options };
  }
  // Every application request is intercepted. These tests cannot authenticate
  // with a provider, alter real cloud configuration, or provision compute.
  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() !== 'GET') mutations.push(`${request.method()} ${path}`);
    if (request.method() === 'PUT' && path === '/api/v1/compute-settings') {
      const body = request.postDataJSON();
      computeWrites.push(body);
      if (body.local) local = body.local;
      if (body.gcp) gcp = body.gcp;
      await route.fulfill({ json: computeResponse() });
      return;
    }
    const cloudAction = /^\/api\/v1\/cloud-connections\/(gcp)\/(connect|recheck|disconnect)$/.exec(path);
    if (request.method() === 'POST' && cloudAction) {
      const provider = cloudAction[1] as Provider;
      const action = cloudAction[2];
      const body = request.postData() ? request.postDataJSON() : null;
      writes.push({ provider, action, body });
      const reply = replies.shift();
      if (reply?.status && reply.status >= 400) {
        await route.fulfill({ status: reply.status, json: { detail: reply.detail } });
        return;
      }
      const response = reply?.connection ?? (action === 'disconnect' ? disconnected(provider)
        : connected(provider, action === 'connect' ? body : providers[provider].config!));
      // Failed connect attempts leave the previously saved connection intact.
      if (action !== 'connect' || response.status === 'connected') {
        providers[provider] = response;
        checkedStatus = null;
      }
      await route.fulfill({ json: response });
      return;
    }
    if (request.method() === 'GET') {
      reads.push(path);
      const responses: Record<string, unknown> = {
        '/api/v1/health': { status: 'ok', version: 'browser-fixture' },
        '/api/v1/capabilities': [],
        '/api/v1/projects': [{ id: projectId, name: 'Cloud review', created_at: timestamp }],
        [`/api/v1/projects/${projectId}/jobs`]: [],
        [`/api/v1/projects/${projectId}/artifacts`]: [],
        '/api/v1/huggingface-connection': { configured: false, username: null, token_hint: null, checked_at: null, message: null },
        '/api/v1/cloud-connections': { providers: Object.values(providers) },
        '/api/v1/compute-settings': computeResponse(),
        '/api/v1/policy-options': {
          runtimes: [], sources: [], training_models: [], training_methods: [], default_training_method: 'lora', compute: { local, gcp },
          quantization_defaults: { cuda: { language: 'Q8_0', vision: null }, cpu: { language: 'Q8_0', vision: null }, note: '' },
        },
      };
      if (path in responses) { await route.fulfill({ json: responses[path] }); return; }
    }
    unexpected.push(`${request.method()} ${path}`);
    await route.fulfill({ status: 405, json: { detail: 'Blocked by cloud connection browser fixture.' } });
  });
  await page.goto('/');
  await openCompute(page);
  return { providers, writes, unexpected, replies, reads, computeWrites, mutations };
}

async function noOverflow(page: Page) {
  const sizes = await page.evaluate(() => ({ width: document.documentElement.clientWidth, scroll: document.documentElement.scrollWidth }));
  expect(sizes.scroll).toBeLessThanOrEqual(sizes.width + 1);
}

test('Google Cloud opens a cancellable keyboard-accessible form without verifying or asking for secrets', async ({ page }) => {
  const { writes, unexpected } = await mockWorkspace(page);
  const gcp = page.getByRole('region', { name: 'Google Cloud', exact: true });
  await expect(gcp.getByRole('button', { name: 'Connect', exact: true })).toBeVisible();
  await expect(page.getByRole('region', { name: /Amazon|AWS/i })).toHaveCount(0);
  await expect(page.locator('.cloud-settings').getByText(/Amazon|AWS/i)).toHaveCount(0);
  await expect(page.locator('.cloud-settings').getByText('Connected', { exact: true })).toHaveCount(0);
  const connect = gcp.getByRole('button', { name: 'Connect', exact: true });
  await connect.focus();
  await page.keyboard.press('Enter');
  const dialog = page.getByRole('dialog', { name: 'Connect Google Cloud', exact: true });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByLabel('Project ID', { exact: true })).toBeVisible();
  await expect(dialog.getByLabel('Region', { exact: true })).toBeVisible();
  await expect(dialog.locator('input[type=password]')).toHaveCount(0);
  await expect(page.getByLabel(/secret|private key|access key/i)).toHaveCount(0);
  await page.keyboard.press('Escape');
  await expect(dialog).toHaveCount(0);
  await expect(connect).toBeFocused();
  expect(writes).toEqual([]);
  expect(unexpected).toEqual([]);
});

test('verifies Google Cloud through the API, restores server state after reload, and disconnects', async ({ page }) => {
  const { providers, writes, unexpected, reads } = await mockWorkspace(page);
  const gcp = page.getByRole('region', { name: 'Google Cloud', exact: true });
  await gcp.getByRole('button', { name: 'Connect', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Connect Google Cloud', exact: true });
  await dialog.getByLabel('Project ID', { exact: true }).fill('robot-training-123');
  await dialog.getByLabel('Region', { exact: true }).fill('europe-west4');
  await dialog.getByRole('button', { name: 'Verify & save', exact: true }).click();
  await expect(dialog).toHaveCount(0);
  await expect(gcp.getByText('Connected', { exact: true })).toBeVisible();
  expect(writes).toEqual([{ provider: 'gcp', action: 'connect', body: { project_id: 'robot-training-123', region: 'europe-west4' } }]);
  expect(providers.gcp.config).toEqual({ project_id: 'robot-training-123', region: 'europe-west4' });
  await page.evaluate(() => localStorage.clear());
  const priorReads = reads.filter(path => path === '/api/v1/cloud-connections').length;
  await page.reload();
  await openCompute(page);
  await expect(gcp.getByText('Connected', { exact: true })).toBeVisible();
  await expect(gcp.getByText('robot-training-123', { exact: true })).toBeVisible();
  expect(reads.filter(path => path === '/api/v1/cloud-connections').length).toBeGreaterThan(priorReads);
  await gcp.getByRole('button', { name: 'Disconnect', exact: true }).click();
  await expect(gcp.getByRole('button', { name: 'Connect', exact: true })).toBeVisible();
  await expect(gcp.getByText('Connected', { exact: true })).toHaveCount(0);
  expect(writes.at(-1)).toEqual({ provider: 'gcp', action: 'disconnect', body: null });
  expect(providers.gcp.config).toBeNull();
  expect(unexpected).toEqual([]);
});

test('missing authentication stays unsaved and offers a retry with the entered values', async ({ page }) => {
  const { providers, writes, unexpected, replies } = await mockWorkspace(page);
  const gcp = page.getByRole('region', { name: 'Google Cloud', exact: true });
  await gcp.getByRole('button', { name: 'Connect', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Connect Google Cloud', exact: true });
  await dialog.getByLabel('Project ID', { exact: true }).fill('robot-training-123');
  replies.push({ connection: { ...disconnected('gcp'), status: 'setup_required', checked_at: timestamp,
    message: 'Sign in to Google Cloud on the application server.', setup_commands: ['gcloud auth login'] } });
  await dialog.getByRole('button', { name: 'Verify & save', exact: true }).click();
  await expect(dialog.getByText('Sign in to Google Cloud on the application server.', { exact: true })).toBeVisible();
  await expect(dialog.getByText('gcloud auth login', { exact: true })).toBeVisible();
  await expect(page.locator('.cloud-settings').getByText('Connected', { exact: true })).toHaveCount(0);
  await expect(dialog.getByLabel('Project ID', { exact: true })).toHaveValue('robot-training-123');
  expect(providers.gcp.config).toBeNull();
  await dialog.getByRole('button', { name: 'Verify & save', exact: true }).click();
  await expect(dialog).toHaveCount(0);
  await expect(gcp.getByText('Connected', { exact: true })).toBeVisible();
  expect(writes).toHaveLength(2);
  expect(unexpected).toEqual([]);
});

test('a failed verification request leaves the form recoverable and never marks it connected', async ({ page }) => {
  const { providers, writes, unexpected, replies } = await mockWorkspace(page);
  const gcp = page.getByRole('region', { name: 'Google Cloud', exact: true });
  await gcp.getByRole('button', { name: 'Connect', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Connect Google Cloud', exact: true });
  await dialog.getByLabel('Project ID', { exact: true }).fill('robot-training-123');
  replies.push({ status: 503, detail: 'Cloud verification is temporarily unavailable.' });
  await dialog.getByRole('button', { name: 'Verify & save', exact: true }).click();
  await expect(dialog.getByText('Cloud verification is temporarily unavailable.', { exact: true })).toBeVisible();
  await expect(dialog.getByLabel('Project ID', { exact: true })).toHaveValue('robot-training-123');
  await expect(page.locator('.cloud-settings').getByText('Connected', { exact: true })).toHaveCount(0);
  expect(providers.gcp.config).toBeNull();
  await dialog.getByRole('button', { name: 'Verify & save', exact: true }).click();
  await expect(dialog).toHaveCount(0);
  await expect(gcp.getByText('Connected', { exact: true })).toBeVisible();
  expect(writes).toHaveLength(2);
  expect(unexpected).toEqual([]);
});

test('an unsuccessful account edit preserves the saved connection', async ({ page }) => {
  const saved = connected('gcp', { project_id: 'existing-project-123', region: 'us-central1' });
  const { providers, writes, unexpected, replies } = await mockWorkspace(page, [saved]);
  const gcp = page.getByRole('region', { name: 'Google Cloud', exact: true });
  await gcp.getByRole('button', { name: 'Edit', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Connect Google Cloud', exact: true });
  await expect(dialog.getByLabel('Project ID', { exact: true })).toHaveValue('existing-project-123');
  await dialog.getByLabel('Project ID', { exact: true }).fill('inaccessible-project-456');
  replies.push({ connection: { ...disconnected('gcp'), status: 'error', checked_at: timestamp,
    message: 'This account cannot access that Google Cloud project.' } });
  await dialog.getByRole('button', { name: 'Verify & save', exact: true }).click();
  await expect(dialog.getByText('This account cannot access that Google Cloud project.', { exact: true })).toBeVisible();
  expect(providers.gcp).toEqual(saved);
  await dialog.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(gcp.getByText('Connected', { exact: true })).toBeVisible();
  await expect(gcp.getByText('existing-project-123', { exact: true })).toBeVisible();
  await expect(gcp.getByText('inaccessible-project-456', { exact: true })).toHaveCount(0);
  expect(writes).toEqual([{ provider: 'gcp', action: 'connect', body: { project_id: 'inaccessible-project-456', region: 'us-central1' } }]);
  expect(unexpected).toEqual([]);
});

test('unverified saved accounts require an explicit recheck and expired authentication clears connected status', async ({ page }) => {
  const saved = { ...connected('gcp', { project_id: 'robot-training-123', region: 'us-central1' }), status: 'unverified' as const };
  const { writes, unexpected, replies } = await mockWorkspace(page, [saved]);
  const gcp = page.getByRole('region', { name: 'Google Cloud', exact: true });
  await expect(gcp.getByText('Connected', { exact: true })).toHaveCount(0);
  await expect(gcp.getByRole('button', { name: 'Recheck', exact: true })).toBeVisible();
  expect(writes).toEqual([]);
  await gcp.getByRole('button', { name: 'Recheck', exact: true }).click();
  await expect(gcp.getByText('Connected', { exact: true })).toBeVisible();
  replies.push({ connection: { ...saved, status: 'setup_required', identity: null, checked_at: timestamp,
    message: 'Google Cloud sign-in expired.', setup_commands: ['gcloud auth login'] } });
  await gcp.getByRole('button', { name: 'Recheck', exact: true }).click();
  await expect(gcp.getByText('Connected', { exact: true })).toHaveCount(0);
  await expect(gcp.getByText('Google Cloud sign-in expired.', { exact: true })).toBeVisible();
  await expect(gcp.getByRole('button', { name: 'Disconnect', exact: true })).toBeVisible();
  expect(writes).toEqual([{ provider: 'gcp', action: 'recheck', body: null }, { provider: 'gcp', action: 'recheck', body: null }]);
  expect(unexpected).toEqual([]);
});

test('cloud cards and authentication setup fit at 320px in dark mode', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 320, height: 740 });
  const { unexpected, replies } = await mockWorkspace(page);
  await page.getByRole('button', { name: 'Dark', exact: true }).click();
  await noOverflow(page);
  await page.getByRole('region', { name: 'Google Cloud', exact: true }).getByRole('button', { name: 'Connect', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Connect Google Cloud', exact: true });
  await dialog.getByLabel('Project ID', { exact: true }).fill('robotics-development-team');
  replies.push({ connection: { ...disconnected('gcp'), status: 'setup_required', checked_at: timestamp,
    message: 'Sign in to Google Cloud on the application server.', setup_commands: ['gcloud auth login --project robotics-development-team'] } });
  await dialog.getByRole('button', { name: 'Verify & save', exact: true }).click();
  await expect(dialog.getByText('Sign in to Google Cloud on the application server.', { exact: true })).toBeVisible();
  await noOverflow(page);
  const screenshot = testInfo.outputPath('cloud-setup-mobile.png');
  await page.screenshot({ path: screenshot, fullPage: true });
  await testInfo.attach('cloud-setup-mobile', { path: screenshot, contentType: 'image/png' });
  expect(unexpected).toEqual([]);
});


test('local run preferences and machine label persist after saving and reloading', async ({ page }, testInfo) => {
  const { writes, unexpected, computeWrites } = await mockWorkspace(page);
  const local = page.getByRole('region', { name: 'Local runs', exact: true });
  await expect(local.getByRole('checkbox', { name: 'Enable local runs', exact: true })).not.toBeChecked();
  await local.getByRole('textbox', { name: 'Machine label', exact: true }).fill('Robotics lab RTX 3070');
  await local.getByRole('checkbox', { name: 'Enable local runs', exact: true }).check();
  await local.getByRole('button', { name: 'Save local settings', exact: true }).click();
  await expect(local.getByRole('status')).toHaveText('Local settings saved.');
  expect(computeWrites).toEqual([{ local: { enabled: true, label: 'Robotics lab RTX 3070' } }]);
  await page.reload();
  await openCompute(page);
  await expect(local.getByRole('textbox', { name: 'Machine label', exact: true })).toHaveValue('Robotics lab RTX 3070');
  await expect(local.getByRole('checkbox', { name: 'Enable local runs', exact: true })).toBeChecked();
  await local.getByRole('checkbox', { name: 'Enable local runs', exact: true }).uncheck();
  await local.getByRole('button', { name: 'Save local settings', exact: true }).click();
  await expect.poll(() => computeWrites.length).toBe(2);
  await expect(local.getByText('Disabled', { exact: true })).toBeVisible();
  await noOverflow(page);
  await page.screenshot({ path: testInfo.outputPath('local-compute-settings.png'), fullPage: true });
  expect(writes).toEqual([]);
  expect(unexpected).toEqual([]);
});

async function openTrainingPreferences(page: Page) {
  const panel = page.locator('.cloud-gpu-settings');
  await expect(panel.locator(':scope > summary')).toContainText('Training preferences');
  await panel.locator(':scope > summary').click();
  await expect(panel.getByRole('combobox', { name: 'Default GPU', exact: true })).toBeVisible();
  return panel;
}

for (const gpu of ['L4', 'T4', 'A100']) {
  test(`optional ${gpu} training preferences persist without manual setup or renting compute`, async ({ page }) => {
    const saved = connected('gcp', { project_id: 'robot-training-123', region: 'us-central1' });
    const { computeWrites, mutations, writes, unexpected } = await mockWorkspace(page, [saved]);
    const preferences = page.locator('.cloud-gpu-settings');
    await expect(preferences).not.toHaveAttribute('open', '');
    const panel = await openTrainingPreferences(page);
    const picker = panel.getByRole('combobox', { name: 'Default GPU', exact: true });
    await picker.click();
    const choices = panel.getByRole('listbox', { name: 'Default GPU', exact: true });
    await expect(choices.getByRole('option')).toHaveCount(3);
    for (const option of ['L4', 'T4', 'A100']) await expect(choices.getByRole('option', { name: option, exact: true })).toBeVisible();
    await choices.getByRole('option', { name: gpu, exact: true }).click();
    await expect(picker).toHaveAttribute('value', gpu);
    await panel.locator('.cloud-gpu-advanced > summary').click();
    await expect(panel.getByRole('checkbox', { name: 'Allow cloud training', exact: true })).toBeChecked();
    await panel.getByRole('spinbutton', { name: 'Disk size (GB)', exact: true }).fill('300');
    await panel.getByRole('spinbutton', { name: 'Idle shutdown (minutes)', exact: true }).fill('15');
    await expect(panel.getByRole('button', { name: /Prepare|Check/ })).toHaveCount(0);
    await expect(panel.getByText(/SkyPilot|gcloud|setup commands/i)).toHaveCount(0);
    await panel.getByRole('button', { name: 'Save preferences', exact: true }).click();
    await expect(panel.getByRole('status')).toHaveText('Preferences saved.');
    expect(computeWrites).toEqual([{ gcp: { enabled: true, default_gpu: gpu, disk_size_gb: 300, idle_minutes: 15 } }]);
    await expect(page.getByRole('region', { name: 'Local runs', exact: true }).getByRole('checkbox', { name: 'Enable local runs', exact: true })).not.toBeChecked();
    await page.reload();
    await openCompute(page);
    await expect(preferences).not.toHaveAttribute('open', '');
    await openTrainingPreferences(page);
    await expect(picker).toHaveAttribute('value', gpu);
    await panel.locator('.cloud-gpu-advanced > summary').click();
    await expect(panel.getByRole('checkbox', { name: 'Allow cloud training', exact: true })).toBeChecked();
    await expect(panel.getByRole('spinbutton', { name: 'Disk size (GB)', exact: true })).toHaveValue('300');
    await expect(panel.getByRole('spinbutton', { name: 'Idle shutdown (minutes)', exact: true })).toHaveValue('15');
    await expect(panel.getByRole('button', { name: 'Save preferences', exact: true })).toBeDisabled();
    expect(writes).toEqual([]);
    expect(mutations).toEqual(['PUT /api/v1/compute-settings']);
    expect(unexpected).toEqual([]);
  });
}

test('connected cloud account requires no manual training setup screens', async ({ page }) => {
  const saved = connected('gcp', { project_id: 'robot-training-123', region: 'us-central1' });
  const { mutations, unexpected } = await mockWorkspace(page, [saved], {
    status: { status: 'unchecked', message: 'Internal preparation will happen automatically.', setup_commands: ['gcloud auth application-default login'] },
  });
  const preferences = page.locator('.cloud-gpu-settings');
  await expect(preferences).not.toHaveAttribute('open', '');
  await expect(page.getByRole('region', { name: 'Google Cloud', exact: true }).getByText('Connected', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: /Prepare SkyPilot|Check SkyPilot/ })).toHaveCount(0);
  await expect(page.getByText('Training preferences', { exact: false })).toBeVisible();
  await openTrainingPreferences(page);
  await expect(preferences.getByText(/SkyPilot|application-default|Internal preparation|Setup required/)).toHaveCount(0);
  await expect(preferences.locator('.cloud-gpu-advanced')).not.toHaveAttribute('open', '');
  await expect(preferences.getByRole('spinbutton')).toHaveCount(0);
  expect(mutations).toEqual([]);
  expect(unexpected).toEqual([]);
});

test('advanced cloud opt-out is saved without changing the connected account', async ({ page }) => {
  const saved = connected('gcp', { project_id: 'robot-training-123', region: 'us-central1' });
  const { providers, mutations, computeWrites, unexpected } = await mockWorkspace(page, [saved]);
  const panel = await openTrainingPreferences(page);
  await panel.locator('.cloud-gpu-advanced > summary').click();
  await panel.getByRole('checkbox', { name: 'Allow cloud training', exact: true }).uncheck();
  await panel.getByRole('button', { name: 'Save preferences', exact: true }).click();
  await expect(panel.getByRole('status')).toHaveText('Preferences saved.');
  expect(computeWrites).toEqual([{ gcp: { enabled: false, default_gpu: 'A100', disk_size_gb: 200, idle_minutes: 10 } }]);
  await page.reload();
  await openCompute(page);
  await openTrainingPreferences(page);
  await panel.locator('.cloud-gpu-advanced > summary').click();
  await expect(panel.getByRole('checkbox', { name: 'Allow cloud training', exact: true })).not.toBeChecked();
  expect(providers.gcp).toEqual(saved);
  expect(mutations).toEqual(['PUT /api/v1/compute-settings']);
  expect(unexpected).toEqual([]);
});

test('training preferences GPU popover fits the screen without infrastructure details', async ({ page }, testInfo) => {
  const { mutations, unexpected } = await mockWorkspace(page, [connected('gcp', { project_id: 'robotics-training-123', region: 'us-central1' })]);
  const panel = await openTrainingPreferences(page);
  await panel.getByRole('combobox', { name: 'Default GPU', exact: true }).click();
  await expect(panel.getByRole('listbox', { name: 'Default GPU', exact: true })).toBeVisible();
  await expect(panel.getByRole('option', { name: 'T4', exact: true })).toBeVisible();
  await noOverflow(page);
  const screenshot = testInfo.outputPath('cloud-gpu-preferences.png');
  await page.screenshot({ path: screenshot, fullPage: true });
  await testInfo.attach('cloud-gpu-preferences', { path: screenshot, contentType: 'image/png' });
  expect(mutations).toEqual([]);
  expect(unexpected).toEqual([]);
});
