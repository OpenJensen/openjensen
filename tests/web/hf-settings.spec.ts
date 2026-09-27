import { expect, test, type Page } from '@playwright/test';

const fakeToken = 'hf_BrowserFixtureTokenNotARealCredential1234567890';
const timestamp = '2026-09-26T12:00:00Z';
const blank = { configured: false, username: null, token_hint: null, checked_at: null, message: null };
const configured = { configured: true, username: 'robotic-user', token_hint: '••••7890', checked_at: timestamp, message: null };

async function settings(page: Page, reject = false) {
  let status: typeof blank | typeof configured = blank;
  const requests: { method: string; body: unknown }[] = [];
  await page.route('**/api/v1/**', async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname.replace('/api/v1', '');
    let value: unknown;
    if (path === '/huggingface-connection') {
      if (request.method() === 'PUT') {
        requests.push({ method: 'PUT', body: request.postDataJSON() });
        if (reject) return route.fulfill({ status: 422, json: { detail: 'Hugging Face rejected this token.' } });
        status = configured;
      } else if (request.method() === 'DELETE') {
        requests.push({ method: 'DELETE', body: null });
        status = blank;
      }
      value = status;
    } else if (path === '/health') value = { status: 'ok', version: '0.1.0' };
    else if (path === '/projects') value = [{ id: 'hf-fixture', name: 'HF settings test', created_at: timestamp }];
    else if (path === '/capabilities' || path.endsWith('/jobs') || path.endsWith('/artifacts')) value = [];
    else if (path === '/policy-options') value = {
      runtimes: [], sources: [], training_models: [], training_methods: [], default_training_method: 'lora',
      compute: { local: { enabled: false, label: 'Local machine' } },
      quantization_defaults: { cuda: { language: 'Q8_0', vision: null }, cpu: { language: 'Q8_0', vision: null }, note: '' },
    };
    else if (path === '/cloud-connections') value = { providers: [{
      provider: 'gcp', name: 'Google Cloud', status: 'disconnected', config: null, identity: null,
      checked_at: null, message: null, setup_commands: [],
    }] };
    else if (path === '/compute-settings') value = {
      local: { enabled: false, label: 'Local machine' }, runtimes: [], gpu_options: [],
      gcp: { enabled: true, default_gpu: 'A100', disk_size_gb: 200, idle_minutes: 10 },
      gcp_status: { status: 'unchecked', configured: false, project_id: null, region: null, skypilot_installed: true, checked_at: null, message: 'Connect Google Cloud.', setup_commands: [] },
    };
    else return route.fulfill({ status: 404, json: { detail: 'Not used by this fixture' } });
    return route.fulfill({ json: value });
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Settings & diagnostics', exact: true }).click();
  await page.getByRole('button', { name: 'Compute', exact: true }).click();
  return { panel: page.getByRole('region', { name: 'Hugging Face', exact: true }), requests };
}

test('HF token is optional, masked, cleared after save, and removable', async ({ page }) => {
  const { panel, requests } = await settings(page);
  await expect(panel.getByText('Optional', { exact: true })).toBeVisible();
  await expect(panel.getByText('For private or gated models.', { exact: true })).toBeVisible();
  const input = panel.getByLabel('Hugging Face access token');
  await expect(input).toHaveAttribute('type', 'password');
  await input.fill(fakeToken);
  await panel.getByRole('button', { name: 'Save token', exact: true }).click();
  await expect(input).toHaveValue('');
  await expect(panel.getByText('Token saved', { exact: true })).toBeVisible();
  await expect(panel.getByText('robotic-user', { exact: true })).toBeVisible();
  await expect(panel.getByText('••••7890', { exact: true })).toBeVisible();
  await expect(panel).not.toContainText(fakeToken);
  expect(requests).toEqual([{ method: 'PUT', body: { token: fakeToken } }]);
  expect(await page.evaluate(() => JSON.stringify({ local: { ...localStorage }, session: { ...sessionStorage } }))).not.toContain(fakeToken);
  await panel.getByRole('button', { name: 'Remove token', exact: true }).click();
  await expect(panel.getByText('Optional', { exact: true })).toBeVisible();
  await expect(panel.getByText('Hugging Face token removed.', { exact: true })).toBeVisible();
  expect(requests[1]).toEqual({ method: 'DELETE', body: null });
});

test('failed token verification shows a safe error without storing it', async ({ page }) => {
  const { panel } = await settings(page, true);
  await panel.getByLabel('Hugging Face access token').fill(fakeToken);
  await panel.getByRole('button', { name: 'Save token', exact: true }).click();
  await expect(panel.getByRole('alert')).toHaveText('Hugging Face rejected this token.');
  await expect(panel.getByText('Optional', { exact: true })).toBeVisible();
  await expect(panel).not.toContainText(fakeToken);
  expect(await page.evaluate(() => JSON.stringify({ ...localStorage }))).not.toContain(fakeToken);
});
