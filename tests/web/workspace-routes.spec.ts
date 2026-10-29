import { openModuleTask } from './work-job-controls';
import { fulfillApplicationUpdateRead } from './application-update-controls';
import { expect, test, type Page } from '@playwright/test';
import { selectProject } from './project-controls';

const destinations = [
  ['dashboard', 'Dashboard'],
  ['datasets', 'Datasets'],
  ['training', 'Fine-tune'],
  ['distillation', 'Distill'],
  ['quantization', 'Quantize'],
  ['evaluation', 'Evaluate'],
  ['simulation', 'Run'],
  ['settings', 'Settings'],
  ['modules', 'Modules'],
  ['cloud-apis', 'Cloud & APIs'],
  ['cloud-runs', 'Cloud runs'],
  ['augmentation', 'Augmentation'],
  ['real-to-sim', 'Real to sim'],
  ['ai-teleoperation', 'AI teleoperation'],
  ['decision-lab', 'Decision Lab'],
  ['models', 'Models'],
  ['robot-lab', 'Initial setup'],
] as const;

const moduleGroups: Record<string, string> = {
  'robot-lab': 'Robot Lab',
  'ai-teleoperation': 'Robot Lab',
  datasets: 'Data',
  augmentation: 'Data',
  'real-to-sim': 'Data',
  training: 'Train',
  distillation: 'Train',
  quantization: 'Train',
  evaluation: 'Test',
  simulation: 'Test',
  'decision-lab': 'Test',
  models: 'Workspace',
  'cloud-runs': 'Workspace',
};
const moduleFixture = (visible?: string[]) =>
  destinations
    .filter(([id]) => id in moduleGroups)
    .map(([id, name]) => ({
      id: id === 'ai-teleoperation' ? 'teaching' : id,
      name,
      group: moduleGroups[id],
      installed: !visible || visible.includes(id),
      available: true,
      enabled: !visible || visible.includes(id),
      configured: !visible || visible.includes(id),
      managed: false,
      setup: 'Configure this feature.',
      setup_url: '/guide/',
      default_installed: ['datasets', 'models'].includes(id),
    }));

async function fixture(page: Page) {
  const writes: string[] = [];
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.route('**/api/v1/**', async (route) => {
    if (await fulfillApplicationUpdateRead(route)) return;
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() !== 'GET') {
      writes.push(path);
      return route.fulfill({ status: 405, json: { detail: 'Navigation must not submit work' } });
    }
    if (path === '/api/v1/modules') return route.fulfill({ json: moduleFixture() });
    if (path === '/api/v1/projects')
      return route.fulfill({
        json: [
          { id: 'routes-a', name: 'Route project A', created_at: '2026-10-07T08:00:00Z' },
          { id: 'routes-b', name: 'Route project B', created_at: '2026-10-07T08:00:00Z' },
        ],
      });
    if (path.endsWith('/jobs') || path.endsWith('/artifacts') || path === '/api/v1/datasets') {
      return route.fulfill({ json: [] });
    }
    return route.continue();
  });
  return { writes, errors };
}

test('home opens the dashboard and every section supports a direct visit and refresh', async ({ page }) => {
  test.setTimeout(90_000);
  const { writes, errors } = await fixture(page);
  const home = await page.goto('/');
  // The server must redirect before hydration, so an immediate sidebar click
  // cannot be overwritten by a later client-side dashboard redirect.
  expect(home?.url()).toMatch(/\/dashboard\/$/);
  await page.getByRole('link', { name: 'Settings', exact: true }).click();
  await expect(page).toHaveURL(/\/settings\/$/);
  await page.getByRole('button', { name: 'Diagnostics', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Run diagnostics', exact: true })).toBeVisible();
  for (const [path, title] of destinations) {
    const response = await page.goto(`/${path}/`);
    expect(response?.status()).toBe(200);
    await expect(page.getByRole('heading', { name: title, level: 1, exact: true })).toBeVisible();
    const link = page
      .getByRole('navigation', { name: 'Policy lifecycle' })
      .getByRole('link', { name: title, exact: true });
    await expect(link).toHaveAttribute('href', `/${path}/`);
    await expect(link).toHaveAttribute('aria-current', 'page');
    await expect(page).toHaveTitle(`OpenJensen · ${title}`);
    await page.reload();
    await expect(link).toHaveAttribute('aria-current', 'page');
    await expect(page.getByRole('heading', { name: title, level: 1, exact: true })).toBeVisible();
  }
  expect(writes).toEqual([]);
  expect(errors).toEqual([]);
});

test('the sidebar lists installed modules and Modules shows how to install the rest', async ({ page }) => {
  const { writes, errors } = await fixture(page);
  await page.route('**/api/v1/modules', (route) => route.fulfill({ json: moduleFixture(['robot-lab', 'datasets']) }));
  await page.goto('/dashboard/');
  const nav = page.getByRole('navigation', { name: 'Policy lifecycle' });
  await expect(nav.getByRole('group', { name: 'Robot Lab' }).getByRole('link')).toHaveText(['Initial setup']);
  await expect(nav.getByRole('group', { name: 'Data' }).getByRole('link')).toHaveText(['Datasets']);
  for (const name of ['AI teleoperation', 'Augmentation', 'Real to sim', 'Decision Lab', 'Cloud runs']) {
    await expect(nav.getByRole('link', { name, exact: true })).toHaveCount(0);
  }
  await nav.getByRole('link', { name: 'Modules', exact: true }).click();
  const modules = page.getByLabel('Workspace modules', { exact: true }).getByRole('listitem');
  await expect(modules).toHaveCount(13);
  await expect(modules.filter({ hasText: 'Initial setup' })).toContainText('Installed');
  const teaching = modules.filter({ hasText: 'AI teleoperation' });
  await expect(teaching).toContainText('Not installed');
  await expect(teaching.getByText('Setup required', { exact: true })).toHaveCount(0);
  await expect(teaching.getByRole('button', { name: 'Install module AI teleoperation', exact: true })).toBeVisible();
  // The legacy teaching URL explains setup without exposing an uninstalled module in navigation.
  await page.goto('/teaching/');
  await expect(nav.getByRole('link', { name: 'AI teleoperation', exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Manage modules', exact: true })).toBeVisible();
  expect(writes).toEqual([]);
  expect(errors).toEqual([]);
});

test('sidebar links, Back and Forward preserve project, theme and unfinished dataset input', async ({ page }) => {
  const { writes, errors } = await fixture(page);
  await page.goto('/datasets/');
  await selectProject(page, 'routes-b');
  await page.getByRole('button', { name: 'Dark', exact: true }).click();
  const repository = page.getByRole('textbox', { name: 'Dataset repository', exact: true });
  await repository.fill('our/unfinished-dataset');
  await page.evaluate(() => {
    (window as any).__routeDocument = 'same document';
  });
  const nav = page.getByRole('navigation', { name: 'Policy lifecycle' });
  await nav.getByRole('link', { name: 'Fine-tune', exact: true }).click();
  await expect(page).toHaveURL(/\/training\/$/);
  await nav.getByRole('link', { name: 'Settings', exact: true }).click();
  await expect(page).toHaveURL(/\/settings\/$/);
  await page.goBack();
  await expect(page).toHaveURL(/\/training\/$/);
  await expect(nav.getByRole('link', { name: 'Fine-tune', exact: true })).toHaveAttribute('aria-current', 'page');
  await page.goBack();
  await expect(page).toHaveURL(/\/datasets\/$/);
  await expect(repository).toHaveValue('our/unfinished-dataset');
  await expect(page.getByLabel('Current project')).toHaveAttribute('data-project-id', 'routes-b');
  await expect(page.getByRole('button', { name: 'Dark', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await page.goForward();
  await expect(page).toHaveURL(/\/training\/$/);
  expect(await page.evaluate(() => (window as any).__routeDocument)).toBe('same document');
  await page.reload();
  await expect(page).toHaveURL(/\/training\/$/);
  await expect(page.getByLabel('Current project')).toHaveAttribute('data-project-id', 'routes-b');
  expect(writes).toEqual([]);
  expect(errors).toEqual([]);
});

test('section links can open in a new tab and unknown pages remain 404s', async ({ page, context }, testInfo) => {
  test.skip(testInfo.project.name === 'mobile', 'Desktop modifier-click behavior');
  const { writes, errors } = await fixture(page);
  await page.goto('/datasets/');
  const nav = page.getByRole('navigation', { name: 'Policy lifecycle' });
  // Installed modules join the sidebar once listed; click after it settles.
  await expect(nav).toHaveAttribute('aria-busy', 'false');
  const link = nav.getByRole('link', { name: 'Fine-tune', exact: true });
  await expect(link).toHaveAttribute('href', '/training/');
  // A modifier click belongs to the browser: the workspace must neither cancel it nor
  // navigate this tab. Chromium does not reliably report the background tab it opens
  // under load, so the section's own address is loaded in a second tab below.
  await page.evaluate(() => {
    window.addEventListener('click', (event) => {
      document.documentElement.dataset.modifierClickPrevented = String(event.defaultPrevented);
    });
  });
  await link.click({ modifiers: ['ControlOrMeta'] });
  await expect(page.locator('html')).toHaveAttribute('data-modifier-click-prevented', 'false');
  await expect(page).toHaveURL(/\/datasets\/$/);
  const other = await context.newPage();
  await other.goto('/training/');
  await expect(other.getByRole('heading', { name: 'Fine-tune', level: 1, exact: true })).toBeVisible();
  await other.close();
  const response = await page.request.get('/training/missing-page/');
  expect(response.status()).toBe(404);
  expect(writes).toEqual([]);
  expect(errors).toEqual([]);
});

test('legacy teaching URL keeps the canonical AI teleoperation navigation', async ({ page }) => {
  const { writes, errors } = await fixture(page);
  await page.goto('/teaching/');
  await expect(page.getByRole('heading', { name: 'AI teleoperation', level: 1, exact: true })).toBeVisible();
  await expect(
    page
      .getByRole('navigation', { name: 'Policy lifecycle' })
      .getByRole('link', { name: 'AI teleoperation', exact: true }),
  ).toHaveAttribute('href', '/ai-teleoperation/');
  expect(writes).toEqual([]);
  expect(errors).toEqual([]);
});

test('sidebar preloads routes without mounting their tools or submitting work', async ({ page }) => {
  const { writes, errors } = await fixture(page);
  const destinationReads: string[] = [];
  page.on('request', (request) => {
    const path = new URL(request.url()).pathname;
    if (path === '/api/v1/augmentation-options') destinationReads.push(path);
  });
  const prefetched = page.waitForResponse((response) => {
    const path = new URL(response.url()).pathname;
    return path.startsWith('/augmentation/') && path.endsWith('.txt');
  });
  await page.goto('/dashboard/');
  const link = page
    .getByRole('navigation', { name: 'Policy lifecycle' })
    .getByRole('link', { name: 'Augmentation', exact: true });
  await link.scrollIntoViewIfNeeded();
  expect((await prefetched).status()).toBe(200);
  await expect(page).toHaveURL(/\/dashboard\/$/);
  expect(destinationReads).toEqual([]);
  expect(writes).toEqual([]);
  await page.evaluate(() => {
    (window as any).__preloadedNavigation = 'same document';
  });
  await link.click();
  await expect(page.getByRole('heading', { name: 'Augmentation', level: 1, exact: true })).toBeVisible();
  expect(destinationReads).toEqual([]);
  await openModuleTask(page);
  await expect.poll(() => destinationReads.length).toBeGreaterThan(0);
  expect(await page.evaluate(() => (window as any).__preloadedNavigation)).toBe('same document');
  expect(writes).toEqual([]);
  expect(errors).toEqual([]);
});
