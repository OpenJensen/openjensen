import { expect, test, type Page } from '@playwright/test';
import { fulfillApplicationUpdateRead } from './application-update-controls';

async function fixture(page: Page) {
  const state = {
    ready: false,
    cancelled: false,
    operation: null as any,
    writes: [] as string[],
    installs: [] as any[],
  };
  const modules = [
    { id: 'robot-lab', name: 'Initial setup' },
    { id: 'teaching', name: 'AI teleoperation' },
    { id: 'augmentation', name: 'Augmentation' },
    { id: 'decision-lab', name: 'Decision Lab' },
    { id: 'cloud-runs', name: 'Cloud runs' },
  ].map((item) => ({
    ...item,
    installed: false,
    configured: false,
    managed: false,
    setup: 'Configure the external service.',
    setup_url: '/guide/',
  }));
  await page.route('**/api/v1/**', async (route) => {
    if (await fulfillApplicationUpdateRead(route)) return;
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() !== 'GET') state.writes.push(path);
    if (path === '/api/v1/modules') return route.fulfill({ json: modules });
    if (path.startsWith('/api/v1/modules/')) {
      const item = modules.find((item) => path.endsWith('/' + item.id))!;
      item.installed = request.postDataJSON().enabled;
      return route.fulfill({ json: { enabled: item.installed } });
    }
    if (path === '/api/v1/module-installations/cancel') {
      state.cancelled = true;
      state.operation = {
        ...state.operation,
        status: 'cancelled',
        message: 'Installation stopped; dependencies kept.',
      };
      return route.fulfill({ json: state.operation });
    }
    if (path === '/api/v1/module-installations') {
      if (request.method() === 'POST') {
        const payload = request.postDataJSON();
        state.installs.push(payload);
        state.operation = {
          id: payload.operation_id,
          profile: payload.profile,
          status: 'running',
          stage: 'Installing dataset readers',
          message: 'Installing on fixture-server.',
          started_at: 'fixture',
          finished_at: null,
        };
        return route.fulfill({ status: 202, json: state.operation });
      }
      if (state.ready && state.operation)
        state.operation = { ...state.operation, status: 'succeeded', message: 'Dataset readers installed.' };
      return route.fulfill({
        json: {
          host: 'fixture-server',
          operation: state.operation,
          profiles: [
            {
              id: 'datasets',
              name: 'Dataset tools',
              ready: state.ready,
              installable: true,
              message: 'Pinned readers.',
              minimum_free_gib: 11,
            },
          ],
        },
      });
    }
    if (path === '/api/v1/compute-settings/local/setup')
      return route.fulfill({ json: { status: 'idle', stage: '', message: '' } });
    if (path === '/api/v1/projects' || path === '/api/v1/datasets') return route.fulfill({ json: [] });
    return route.continue();
  });
  await page.goto('/settings/');
  await page.getByRole('button', { name: 'Modules', exact: true }).click();
  return state;
}

test('add and hide modules update navigation and survive reload without installation', async ({ page }) => {
  const state = await fixture(page);
  const nav = page.getByRole('navigation', { name: 'Policy lifecycle' });
  await expect(nav.getByRole('link', { name: 'AI teleoperation', exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Add AI teleoperation', exact: true }).click();
  await expect(nav.getByRole('link', { name: 'AI teleoperation', exact: true })).toBeVisible();
  await page.reload();
  await page.getByRole('button', { name: 'Modules', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Hide AI teleoperation' })).toBeVisible();
  await page.getByRole('button', { name: 'Hide AI teleoperation' }).click();
  await expect(nav.getByRole('link', { name: 'AI teleoperation', exact: true })).toHaveCount(0);
  expect(state.installs).toEqual([]);
  expect(state.writes).toEqual(['/api/v1/modules/teaching', '/api/v1/modules/teaching']);
});

test('installation names its server, reports progress and survives navigation', async ({ page }) => {
  const state = await fixture(page);
  await expect(page.getByText('Server: fixture-server')).toBeVisible();
  expect(state.writes).toEqual([]);
  await page.getByRole('button', { name: 'Install Dataset tools' }).click();
  await expect(page.getByText('Installing dataset readers', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Compute', exact: true }).click();
  await page.getByRole('button', { name: 'Modules', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Cancel installation' })).toBeVisible();
  state.ready = true;
  await expect(page.getByText('Dataset readers installed.', { exact: true })).toBeVisible();
  expect(state.installs).toHaveLength(1);
  expect(state.installs[0].profile).toBe('datasets');
  expect(state.installs[0].operation_id).toMatch(/^[a-f0-9]{32}$/);
  expect(Object.keys(state.installs[0]).sort()).toEqual(['operation_id', 'profile']);
  await expect(page.getByRole('button', { name: 'Install Dataset tools' })).toHaveCount(0);
});

test('cancel stops installation without hiding the module or reloading the page', async ({ page }) => {
  const state = await fixture(page);
  await page.evaluate(() => {
    (window as any).__moduleDocument = 'same';
  });
  await page.getByRole('button', { name: 'Install Dataset tools' }).click();
  await page.getByRole('button', { name: 'Cancel installation' }).click();
  await expect(page.getByText('Installation stopped; dependencies kept.')).toBeVisible();
  expect(state.cancelled).toBe(true);
  expect(await page.evaluate(() => (window as any).__moduleDocument)).toBe('same');
  await expect(page.getByRole('button', { name: 'Install Dataset tools' })).toBeEnabled();
});
