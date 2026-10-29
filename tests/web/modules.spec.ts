import { expect, test, type Page } from '@playwright/test';
import { fulfillApplicationUpdateRead } from './application-update-controls';

const catalog = [
  ['robot-lab', 'Initial setup', 'Robot Lab'],
  ['teaching', 'AI teleoperation', 'Robot Lab'],
  ['datasets', 'Datasets', 'Data'],
  ['augmentation', 'Augmentation', 'Data'],
  ['training', 'Fine-tuning', 'Train'],
  ['distillation', 'Distillation', 'Train'],
  ['quantization', 'Quantization', 'Train'],
  ['evaluation', 'Evaluation', 'Test'],
  ['simulation', 'Simulation', 'Test'],
  ['decision-lab', 'Decision Lab', 'Test'],
  ['models', 'Models', 'Workspace'],
  ['cloud-runs', 'Cloud runs', 'Workspace'],
  ['google-cloud', 'Google Cloud', 'Cloud & APIs'],
  ['openai', 'OpenAI', 'Cloud & APIs'],
  ['gemini', 'Gemini', 'Cloud & APIs'],
  ['huggingface', 'Hugging Face', 'Cloud & APIs'],
] as const;

async function fixture(page: Page, options: { companion?: 'connected' | 'offline'; modulesError?: boolean } = {}) {
  const modules = catalog.map(([id, name, group]) => ({
    id,
    name,
    group,
    installed: ['datasets', 'models'].includes(id) || (id === 'robot-lab' && Boolean(options.companion)),
    enabled: ['datasets', 'models'].includes(id) || (id === 'robot-lab' && Boolean(options.companion)),
    available: true,
    configured: ['datasets', 'models'].includes(id) || (id === 'robot-lab' && Boolean(options.companion)),
    default_installed: ['datasets', 'models'].includes(id),
    managed: false,
    setup: 'Configure the external service.',
    setup_url: '/guide/',
  }));
  const state = {
    modules,
    ready: false,
    modulesError: options.modulesError ?? false,
    cancelled: false,
    operation: null as any,
    writes: [] as string[],
    runtimeInstalls: [] as any[],
  };
  await page.route('**/api/v1/**', async (route) => {
    if (await fulfillApplicationUpdateRead(route)) return;
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() !== 'GET') state.writes.push(`${request.method()} ${path}`);
    if (path === '/api/v1/modules')
      return state.modulesError
        ? route.fulfill({ status: 503, json: { detail: 'Module list unavailable.' } })
        : route.fulfill({ json: modules });
    if (path.startsWith('/api/v1/modules/')) {
      const id = path.split('/')[4];
      const item = modules.find((item) => item.id === id)!;
      if (request.method() === 'DELETE') {
        item.installed = item.enabled = false;
      } else if (path.endsWith('/install')) {
        const profiles: Record<string, string> = {
          datasets: 'datasets',
          'robot-lab': 'robot',
          'decision-lab': 'decision',
        };
        if (profiles[id]) {
          const payload = { profile: profiles[id], operation_id: 'a'.repeat(32), module: id };
          state.runtimeInstalls.push(payload);
          state.operation = {
            ...payload,
            id: payload.operation_id,
            status: 'running',
            stage: 'Downloading from upstream',
            message: 'Installing on fixture-server.',
            started_at: 'fixture',
          };
          return route.fulfill({ status: 202, json: state.operation });
        }
        item.installed = item.enabled = true;
      } else {
        item.enabled = request.postDataJSON().enabled;
      }
      return route.fulfill({ json: item });
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
        state.runtimeInstalls.push(payload);
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
      if (state.ready && state.operation) {
        state.operation = { ...state.operation, status: 'succeeded', message: 'Dataset readers installed.' };
        if (state.operation.module) {
          const item = modules.find((value) => value.id === state.operation.module)!;
          item.installed = item.enabled = true;
        }
      }
      return route.fulfill({
        json: {
          host: 'fixture-server',
          operation: state.operation,
          profiles: [
            ...(!options.companion
              ? [
                  {
                    id: 'robot',
                    name: 'Robot Lab',
                    ready: false,
                    installable: true,
                    message: 'Pinned USB dependencies.',
                    minimum_free_gib: 11,
                  },
                ]
              : []),
            {
              id: 'decision',
              name: 'Decision scorer',
              ready: false,
              installable: true,
              message: 'CPU packages.',
              minimum_free_gib: 14,
            },
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
  await page.goto('/modules/');
  return state;
}

function moduleCard(page: Page, name: string) {
  return page
    .getByLabel('Workspace modules', { exact: true })
    .getByRole('listitem')
    .filter({ has: page.getByRole('heading', { name, exact: true }) });
}

test('module manager groups every feature and clearly marks the fresh defaults', async ({ page }) => {
  const state = await fixture(page);
  for (const group of ['Robot Lab', 'Data', 'Train', 'Test', 'Workspace', 'Cloud & APIs']) {
    await expect(page.getByRole('list', { name: `${group} modules`, exact: true })).toBeVisible();
  }
  await expect(page.getByRole('list', { name: 'Train modules' }).getByRole('heading')).toHaveText([
    'Fine-tuning',
    'Distillation',
    'Quantization',
  ]);
  for (const name of ['Datasets', 'Models']) await expect(moduleCard(page, name)).toContainText('Included by default');
  const nav = page.getByRole('navigation', { name: 'Policy lifecycle' });
  for (const name of ['Dashboard', 'Datasets', 'Models', 'Modules', 'Cloud & APIs', 'Settings'])
    await expect(nav.getByRole('link', { name, exact: true })).toBeVisible();
  for (const name of [
    'Initial setup',
    'AI teleoperation',
    'Fine-tune',
    'Distill',
    'Quantize',
    'Evaluate',
    'Run',
    'Cloud runs',
  ])
    await expect(nav.getByRole('link', { name, exact: true })).toHaveCount(0);
  const robot = moduleCard(page, 'Initial setup');
  await expect(robot.getByRole('button', { name: 'Install module Initial setup', exact: true })).toBeVisible();
  await expect(robot.getByText('Setup required', { exact: true })).toHaveCount(0);
  await expect(robot.getByRole('link')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Install Training', exact: true })).toHaveCount(0);
  await expect(page.getByRole('list', { name: 'Runtime dependencies' }).getByRole('listitem')).toHaveCount(1);
  expect(state.writes).toEqual([]);
});

test('install, hide, show and uninstall persist independently of readiness and dependencies', async ({ page }) => {
  const state = await fixture(page);
  const card = moduleCard(page, 'AI teleoperation');
  const nav = page.getByRole('navigation', { name: 'Policy lifecycle' });
  await card.getByRole('button', { name: 'Enable module AI teleoperation', exact: true }).click();
  await expect(nav.getByRole('link', { name: 'AI teleoperation', exact: true })).toBeVisible();
  await expect(card.getByText('Setup required', { exact: true })).toBeVisible();
  expect(state.modules.find((item) => item.id === 'teaching')?.configured).toBe(false);
  await page.reload();
  await card.getByRole('button', { name: 'Hide AI teleoperation', exact: true }).click();
  await expect(card).toContainText('Enabled · Hidden');
  await expect(nav.getByRole('link', { name: 'AI teleoperation', exact: true })).toHaveCount(0);
  await page.reload();
  await expect(card.getByRole('button', { name: 'Show AI teleoperation', exact: true })).toBeVisible();
  await card.getByRole('button', { name: 'Show AI teleoperation', exact: true }).click();
  await expect(nav.getByRole('link', { name: 'AI teleoperation', exact: true })).toBeVisible();
  await card.getByRole('button', { name: 'Disable AI teleoperation', exact: true }).click();
  await expect(nav.getByRole('link', { name: 'AI teleoperation', exact: true })).toHaveCount(0);
  await expect(card.getByRole('button', { name: 'Enable module AI teleoperation', exact: true })).toBeVisible();
  await expect(card.getByText('Setup required', { exact: true })).toHaveCount(0);
  await expect(page.getByRole('status').filter({ hasText: 'Your data is kept.' })).toBeVisible();
  await page.reload();
  await expect(card.getByRole('button', { name: 'Enable module AI teleoperation', exact: true })).toBeVisible();
  expect(state.runtimeInstalls).toEqual([]);
  expect(state.writes).toEqual([
    'POST /api/v1/modules/teaching/install',
    'PUT /api/v1/modules/teaching',
    'PUT /api/v1/modules/teaching',
    'DELETE /api/v1/modules/teaching',
  ]);
});

test('all feature pages require installation while Modules stays reachable', async ({ page }) => {
  const state = await fixture(page);
  await moduleCard(page, 'Datasets').getByRole('button', { name: 'Uninstall Datasets', exact: true }).click();
  await page.goto('/datasets/');
  const nav = page.getByRole('navigation', { name: 'Policy lifecycle' });
  await expect(nav.getByRole('link', { name: 'Datasets', exact: true })).toHaveCount(0);
  await expect(page.getByRole('textbox', { name: 'Dataset repository', exact: true })).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Module setup' })).toBeVisible();
  await page.getByRole('button', { name: 'Manage modules', exact: true }).click();
  await expect(page).toHaveURL(/\/modules\/$/);
  await expect(
    moduleCard(page, 'Datasets').getByRole('button', { name: 'Install module Datasets', exact: true }),
  ).toBeVisible();
  expect(state.runtimeInstalls).toEqual([]);
});

test('integration modules install separately and open the permanent Cloud & APIs page', async ({ page }) => {
  const state = await fixture(page);
  const card = moduleCard(page, 'OpenAI');
  await expect(card.getByText('Setup required', { exact: true })).toHaveCount(0);
  await card.getByRole('button', { name: 'Enable module OpenAI', exact: true }).click();
  await expect(card.getByRole('button', { name: 'Hide OpenAI', exact: true })).toHaveCount(0);
  await card.getByRole('button', { name: 'Open OpenAI', exact: true }).click();
  await expect(page).toHaveURL(/\/cloud-apis\/$/);
  await expect(page.getByRole('heading', { name: 'Cloud & APIs', level: 1, exact: true })).toBeVisible();
  expect(state.runtimeInstalls).toEqual([]);
});

test('runtime installation reports its server and progress without a duplicate request', async ({ page }) => {
  const state = await fixture(page);
  await expect(page.getByText('Server: fixture-server')).toBeVisible();
  await page.getByRole('button', { name: 'Install Dataset tools', exact: true }).click();
  await expect(page.getByText('Installing dataset readers', { exact: true })).toBeVisible();
  await page.getByRole('link', { name: 'Settings', exact: true }).click();
  await page.getByRole('link', { name: 'Modules', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Cancel installation', exact: true })).toBeVisible();
  state.ready = true;
  await expect(page.getByText('Dataset readers installed.', { exact: true })).toBeVisible();
  expect(state.runtimeInstalls).toHaveLength(1);
  expect(state.runtimeInstalls[0].profile).toBe('datasets');
  expect(state.runtimeInstalls[0].operation_id).toMatch(/^[a-f0-9]{32}$/);
  await expect(page.getByRole('button', { name: 'Install Dataset tools', exact: true })).toHaveCount(0);
});

test('cancel stops only the dependency installation', async ({ page }) => {
  const state = await fixture(page);
  await page.getByRole('button', { name: 'Install Dataset tools', exact: true }).click();
  await page.getByRole('button', { name: 'Cancel installation', exact: true }).click();
  await expect(page.getByText('Installation stopped; dependencies kept.')).toBeVisible();
  expect(state.cancelled).toBe(true);
  await expect(page.getByRole('button', { name: 'Install Dataset tools', exact: true })).toBeEnabled();
  expect(state.modules.find((item) => item.id === 'datasets')?.installed).toBe(true);
});

for (const connection of ['connected', 'offline'] as const) {
  test(`paired Robot Lab stays installed when the companion is ${connection}`, async ({ page }) => {
    const state = await fixture(page, { companion: connection });
    const robot = page
      .getByRole('list', { name: 'Runtime dependencies' })
      .getByRole('listitem')
      .filter({ hasText: 'Robot Lab' });
    await expect(robot).toContainText('Installed on Paired computer.');
    await expect(robot).toContainText(connection === 'connected' ? 'Companion connected.' : 'Installed · Offline');
    await expect(page.getByRole('button', { name: 'Install Robot Lab', exact: true })).toHaveCount(0);
    expect(state.runtimeInstalls).toEqual([]);
  });
}

test('unreadable module state reveals no feature navigation from stale browser history', async ({ page }) => {
  await page.addInitScript(
    (ids) => localStorage.setItem('jensen.modules', JSON.stringify(ids)),
    catalog.map(([id]) => id),
  );
  const state = await fixture(page, { modulesError: true });
  await expect(page.getByText('Module list unavailable.', { exact: true })).toBeVisible();
  const nav = page.getByRole('navigation', { name: 'Policy lifecycle' });
  for (const name of [
    'Initial setup',
    'AI teleoperation',
    'Datasets',
    'Models',
    'Fine-tune',
    'Distill',
    'Quantize',
    'Evaluate',
    'Run',
  ])
    await expect(nav.getByRole('link', { name, exact: true })).toHaveCount(0);
  for (const name of ['Modules', 'Cloud & APIs', 'Settings'])
    await expect(nav.getByRole('link', { name, exact: true })).toBeVisible();
  expect(state.writes).toEqual([]);
});

test('the module Install button downloads first and only then adds the page to the sidebar', async ({ page }) => {
  const state = await fixture(page);
  const card = moduleCard(page, 'Datasets');
  await card.getByRole('button', { name: 'Uninstall Datasets', exact: true }).click();
  const nav = page.getByRole('navigation', { name: 'Policy lifecycle' });
  await card.getByRole('button', { name: 'Install module Datasets', exact: true }).click();
  await expect(card).toContainText('Installing…');
  await expect(nav.getByRole('link', { name: 'Datasets', exact: true })).toHaveCount(0);
  await expect(page.getByText('Downloading from upstream', { exact: true })).toBeVisible();
  expect(state.runtimeInstalls).toHaveLength(1);
  state.ready = true;
  await expect(nav.getByRole('link', { name: 'Datasets', exact: true })).toBeVisible();
  await expect(card).toContainText('Installed');
});

test('a failed upstream installation stays uninstalled and can be retried', async ({ page }) => {
  const state = await fixture(page);
  const card = moduleCard(page, 'Decision Lab');
  await card.getByRole('button', { name: 'Install module Decision Lab', exact: true }).click();
  state.operation = { ...state.operation, status: 'failed', message: 'Upstream download failed.' };
  await page.getByRole('button', { name: 'Refresh', exact: true }).click();
  await expect(page.getByText('Upstream download failed.', { exact: true })).toBeVisible();
  await expect(card.getByRole('button', { name: 'Install module Decision Lab', exact: true })).toBeEnabled();
  await expect(
    page.getByRole('navigation', { name: 'Policy lifecycle' }).getByRole('link', { name: 'Decision Lab', exact: true }),
  ).toHaveCount(0);
});
