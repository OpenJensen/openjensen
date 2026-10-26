import { expect, test, type Page } from '@playwright/test';
import { fulfillApplicationUpdateRead } from './application-update-controls';

async function fixture(page: Page, options: { companion?: 'connected' | 'offline'; modulesError?: boolean } = {}) {
  const state = {
    ready: false,
    modulesError: options.modulesError ?? false,
    cancelled: false,
    operation: null as any,
    writes: [] as string[],
    installs: [] as any[],
  };
  const modules = [
    { id: 'robot-lab', name: 'Robot Lab' },
    { id: 'teaching', name: 'Teaching' },
    { id: 'augmentation', name: 'Augmentation' },
    { id: 'decision-lab', name: 'Decision Lab' },
    { id: 'cloud-runs', name: 'Cloud runs' },
  ].map((item) => ({
    ...item,
    installed: item.id === 'robot-lab' && Boolean(options.companion),
    available: item.id === 'teaching' || (item.id === 'robot-lab' && Boolean(options.companion)),
    enabled: item.id === 'robot-lab' && Boolean(options.companion),
    configured: item.id === 'robot-lab' && Boolean(options.companion),
    managed: false,
    setup: 'Configure the external service.',
    setup_url: '/guide/',
  }));
  await page.route('**/api/v1/**', async (route) => {
    if (await fulfillApplicationUpdateRead(route)) return;
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() !== 'GET') state.writes.push(path);
    if (path === '/api/v1/modules')
      return state.modulesError
        ? route.fulfill({ status: 503, json: { detail: 'Module list unavailable.' } })
        : route.fulfill({ json: modules });
    if (path.startsWith('/api/v1/modules/')) {
      const item = modules.find((item) => path.endsWith('/' + item.id))!;
      item.enabled = request.postDataJSON().enabled;
      item.installed = item.available && item.enabled;
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
            ...(options.companion
              ? [
                  {
                    id: 'robot',
                    name: 'Robot Lab',
                    ready: true,
                    installable: false,
                    location: 'companion',
                    host: 'Paired computer',
                    connected: options.companion === 'connected',
                    message: 'Reconnect the paired computer to use the arm.',
                    minimum_free_gib: 11,
                  },
                ]
              : []),
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
  return Object.assign(state, { modules });
}

test('add and hide modules update navigation and survive reload without installation', async ({ page }) => {
  const state = await fixture(page);
  const nav = page.getByRole('navigation', { name: 'Policy lifecycle' });
  await expect(nav.getByRole('link', { name: 'Teaching', exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Add Teaching', exact: true }).click();
  await expect(nav.getByRole('link', { name: 'Teaching', exact: true })).toBeVisible();
  await page.reload();
  await page.getByRole('button', { name: 'Modules', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Hide Teaching' })).toBeVisible();
  await page.getByRole('button', { name: 'Hide Teaching' }).click();
  await expect(nav.getByRole('link', { name: 'Teaching', exact: true })).toHaveCount(0);
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

test('paired Robot Lab shows its computer and never offers another installation', async ({ page }) => {
  const state = await fixture(page, { companion: 'connected' });
  const robot = page
    .getByRole('list', { name: 'Runtime packages' })
    .getByRole('listitem')
    .filter({ hasText: 'Robot Lab' });
  await expect(robot).toContainText('Installed on Paired computer.');
  await expect(robot).toContainText('Companion connected.');
  await expect(robot).toContainText('Installed');
  await expect(page.getByRole('button', { name: 'Install Robot Lab', exact: true })).toHaveCount(0);
  await expect(
    page.getByRole('navigation', { name: 'Policy lifecycle' }).getByRole('link', { name: 'Robot Lab', exact: true }),
  ).toBeVisible();
  expect(state.installs).toEqual([]);
});

test('offline paired Robot Lab stays installed and asks for reconnection', async ({ page }) => {
  const state = await fixture(page, { companion: 'offline' });
  const robot = page
    .getByRole('list', { name: 'Runtime packages' })
    .getByRole('listitem')
    .filter({ hasText: 'Robot Lab' });
  await expect(robot).toContainText('Installed · Offline');
  await expect(robot).toContainText('Reconnect the paired computer');
  await expect(page.getByRole('button', { name: 'Install Robot Lab', exact: true })).toHaveCount(0);
  expect(state.installs).toEqual([]);
});

test('unavailable modules cannot be added and direct visits stay out of navigation', async ({ page }) => {
  const state = await fixture(page);
  const cards = page.getByRole('list', { name: 'Workspace modules' });
  const robot = cards.getByRole('listitem').filter({ hasText: 'Robot Lab' });
  await expect(robot).toContainText('Not installed');
  await expect(robot.getByRole('button', { name: 'Add Robot Lab', exact: true })).toHaveCount(0);
  await expect(robot.getByRole('link', { name: 'Set up Robot Lab', exact: true })).toBeVisible();
  await page.goto('/robot-lab/');
  await expect(
    page.getByRole('navigation', { name: 'Policy lifecycle' }).getByRole('link', { name: 'Robot Lab', exact: true }),
  ).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Module setup' })).toBeVisible();
  await page.getByRole('button', { name: 'Manage modules', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Modules', exact: true })).toBeVisible();
  expect(state.writes).toEqual([]);
});

test('an unreadable module list does not reveal optional navigation from browser history', async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem(
      'jensen.modules',
      JSON.stringify(['robot-lab', 'teaching', 'augmentation', 'decision-lab', 'cloud-runs']),
    );
  });
  const state = await fixture(page, { modulesError: true });
  await expect(page.getByText('Module list unavailable.', { exact: true })).toBeVisible();
  const nav = page.getByRole('navigation', { name: 'Policy lifecycle' });
  for (const name of ['Robot Lab', 'Teaching', 'Augmentation', 'Decision Lab', 'Cloud runs']) {
    await expect(nav.getByRole('link', { name, exact: true })).toHaveCount(0);
  }
  expect(state.writes).toEqual([]);
});
