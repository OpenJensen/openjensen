import { expect, test } from '@playwright/test';
import type { ComputeTarget, PublicComputeTarget } from '../../apps/web/src/lib/compute-targets';
import { installedModules } from './module-fixtures';
import { fulfillApplicationUpdateRead } from './application-update-controls';

async function settings(page: import('@playwright/test').Page) {
  const saved: ComputeTarget[] = [];
  const actions: string[] = [];
  let entries: PublicComputeTarget[] = [];
  await page.route('**/api/v1/**', async (route) => {
    if (await fulfillApplicationUpdateRead(route)) return;
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (path === '/api/v1/compute-targets') {
      await route.fulfill({
        json: {
          schema_version: 1,
          targets: entries,
          connection_refs: ['team-compute'],
          runtime_options: [
            { id: 'worker', label: 'Registered policy worker', operations: ['policy.finetune', 'policy.quantize'] },
          ],
        },
      });
      return;
    }
    if (path.startsWith('/api/v1/compute-targets/')) {
      if (request.method() === 'PUT') {
        const definition = request.postDataJSON() as ComputeTarget;
        saved.push(definition);
        const entry: PublicComputeTarget = {
          definition,
          operations: definition.operations.length ? definition.operations : ['policy.finetune', 'policy.quantize'],
          status: definition.enabled ? 'configured' : 'disabled',
          launchable: definition.enabled,
          message: 'Configuration saved.',
          checked_at: null,
          sha256: 'a'.repeat(64),
        };
        entries = [...entries.filter((item) => item.definition.id !== definition.id), entry];
        await route.fulfill({ json: entry });
      } else if (request.method() === 'POST') {
        actions.push('check');
        await route.fulfill({
          json: { ...entries[0], checked_at: '2026-01-01T12:00:00Z', message: 'Access verified without provisioning.' },
        });
      } else {
        actions.push('remove');
        entries = [];
        await route.fulfill({ json: { removed: true } });
      }
      return;
    }
    const replies: Record<string, unknown> = {
      '/api/v1/health': { status: 'ok', version: 'test' },
      '/api/v1/modules': installedModules,
      '/api/v1/capabilities': [],
      '/api/v1/projects': [],
      '/api/v1/policy-options': {
        runtimes: [],
        sources: [],
        compute: { local: { enabled: false, label: 'Local machine' } },
      },
    };
    await route.fulfill({ json: replies[path] ?? [] });
  });
  await page.goto('/settings/');
  await expect(page.getByRole('region', { name: 'Compute targets', exact: true })).toBeVisible();
  return { saved, actions };
}

test('targets are explicit opt-in and can be edited, checked and removed', async ({ page }) => {
  const state = await settings(page);
  await page.getByRole('button', { name: 'Add compute target', exact: true }).click();
  const form = page.getByRole('form', { name: 'Configure compute target', exact: true });
  await form.getByLabel('Target name', { exact: true }).fill('GPU server');
  await form.getByLabel('Worker runtime', { exact: true }).selectOption('worker');
  await expect(form.getByLabel('Allow jobs on this target', { exact: true })).not.toBeChecked();
  await form.getByRole('button', { name: 'Save compute target', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'GPU server', exact: true })).toBeVisible();
  expect(state.saved[0].enabled).toBe(false);
  await page.getByRole('button', { name: 'Enable', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Disable', exact: true })).toBeVisible();
  expect(state.saved[1].enabled).toBe(true);
  await page.getByRole('button', { name: 'Check configuration', exact: true }).click();
  await expect(page.getByRole('status').filter({ hasText: 'Access verified without provisioning' })).toBeVisible();
  await page.getByRole('button', { name: 'Edit', exact: true }).click();
  await form.getByLabel('Target name', { exact: true }).fill('Policy worker');
  await form.getByRole('button', { name: 'Save compute target', exact: true }).click();
  expect(state.saved[2].id).toBe(state.saved[0].id);
  await page.getByRole('button', { name: 'Remove', exact: true }).click();
  await expect(page.getByText('No compute targets configured.', { exact: false })).toBeVisible();
  expect(state.actions).toEqual(['check', 'remove']);
});

test('Slurm settings retain resource limits and connection references', async ({ page }) => {
  const state = await settings(page);
  await page.getByRole('button', { name: 'Add compute target', exact: true }).click();
  const form = page.getByRole('form', { name: 'Configure compute target', exact: true });
  await form.getByLabel('Target name', { exact: true }).fill('Research cluster');
  await form.getByLabel('Infrastructure', { exact: true }).selectOption('slurm');
  await form.getByLabel('Worker runtime', { exact: true }).selectOption('worker');
  await form.getByLabel('Connection', { exact: true }).selectOption('team-compute');
  await form.getByLabel('Registered cluster', { exact: true }).fill('research');
  await form.getByLabel('Partition (optional)', { exact: true }).fill('gpu');
  await form.getByLabel('Account (optional)', { exact: true }).fill('robotics');
  await form.getByLabel('QoS (optional)', { exact: true }).fill('normal');
  await form.getByLabel('CPUs', { exact: true }).fill('8');
  await form.getByLabel('Memory (GiB)', { exact: true }).fill('32');
  await form.getByLabel('GPU count', { exact: true }).selectOption('1');
  await form.getByLabel('Accelerator type', { exact: true }).fill('L4');
  await form.getByLabel('Concurrent jobs', { exact: true }).fill('2');
  await form.getByLabel('Allow jobs on this target', { exact: true }).check();
  await form.getByRole('button', { name: 'Save compute target', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Research cluster', exact: true })).toBeVisible();
  expect(state.saved[0]).toMatchObject({
    adapter: 'skypilot',
    infrastructure: 'slurm',
    connection_ref: 'team-compute',
    location: 'research',
    partition: 'gpu',
    account: 'robotics',
    qos: 'normal',
    resources: { cpus: 8, memory_gib: 32, gpu_count: 1, accelerator: 'L4', concurrency: 2 },
  });
  expect(Object.keys(state.saved[0])).not.toContain('command');
  expect(Object.keys(state.saved[0])).not.toContain('credentials');
});
