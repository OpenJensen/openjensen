import { fulfillApplicationUpdateRead } from './application-update-controls';
import { expect, test, type Page } from '@playwright/test';
import { selectProject } from './project-controls';

const destinations = [
  ['dashboard', 'Dashboard'],
  ['datasets', 'Dataset'],
  ['training', 'Fine-tune'],
  ['distillation', 'Distill'],
  ['quantization', 'Quantize'],
  ['evaluation', 'Evaluate'],
  ['simulation', 'Run'],
  ['settings', 'Settings & diagnostics'],
  ['cloud-runs', 'Cloud runs'],
  ['augmentation', 'Augmentation'],
  ['teaching', 'Teaching'],
  ['decision-lab', 'Decision lab'],
  ['models', 'My models'],
  ['robot-lab', 'Robot Lab'],
] as const;

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
  await page.getByRole('link', { name: 'Settings & diagnostics', exact: true }).click();
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
    await expect(page).toHaveTitle(`Open Jensen · ${title}`);
    await page.reload();
    await expect(link).toHaveAttribute('aria-current', 'page');
    await expect(page.getByRole('heading', { name: title, level: 1, exact: true })).toBeVisible();
  }
  expect(writes).toEqual([]);
  expect(errors).toEqual([]);
});

test('the sidebar lists installed modules and Settings shows how to add the rest', async ({ page }) => {
  const { writes, errors } = await fixture(page);
  await page.route('**/api/v1/modules', (route) =>
    route.fulfill({
      json: [
        { id: 'robot-lab', name: 'Robot Lab', installed: true, setup: 'Install the robot profile.' },
        { id: 'teaching', name: 'Teaching', installed: false, setup: 'Set JENSEN_TEACHING_URL.' },
        { id: 'augmentation', name: 'Augmentation', installed: false, setup: 'Connect Google Cloud.' },
        { id: 'decision-lab', name: 'Decision lab', installed: false, setup: 'Configure the scorer.' },
        { id: 'cloud-runs', name: 'Cloud runs', installed: false, setup: 'Set JENSEN_CLOUD_RUNS_DIR.' },
      ],
    }),
  );
  await page.goto('/dashboard/');
  const nav = page.getByRole('navigation', { name: 'Policy lifecycle' });
  await expect(nav.getByRole('group', { name: 'Robot' }).getByRole('link')).toHaveText(['Robot Lab']);
  await expect(nav.getByRole('group', { name: 'Data' }).getByRole('link')).toHaveText(['Dataset']);
  for (const name of ['Teaching', 'Augmentation', 'Decision lab', 'Cloud runs']) {
    await expect(nav.getByRole('link', { name, exact: true })).toHaveCount(0);
  }
  await nav.getByRole('link', { name: 'Settings & diagnostics', exact: true }).click();
  await page.getByRole('button', { name: 'Modules', exact: true }).click();
  const modules = page.getByRole('region', { name: 'Modules' }).getByRole('listitem');
  await expect(modules).toHaveCount(5);
  await expect(modules.filter({ hasText: 'Robot Lab' })).toContainText('Installed');
  await expect(modules.filter({ hasText: 'Teaching' })).toContainText('Set JENSEN_TEACHING_URL.Not installed');
  // An uninstalled module stays reachable by URL and is listed while open.
  await page.goto('/teaching/');
  await expect(nav.getByRole('link', { name: 'Teaching', exact: true })).toHaveAttribute('aria-current', 'page');
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
  await nav.getByRole('link', { name: 'Settings & diagnostics', exact: true }).click();
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
  const [other] = await Promise.all([context.waitForEvent('page'), link.click({ modifiers: ['ControlOrMeta'] })]);
  await expect(other).toHaveURL(/\/training\/$/);
  await expect(other.getByRole('heading', { name: 'Fine-tune', level: 1, exact: true })).toBeVisible();
  await expect(page).toHaveURL(/\/datasets\/$/);
  await other.close();
  const response = await page.request.get('/training/missing-page/');
  expect(response.status()).toBe(404);
  expect(writes).toEqual([]);
  expect(errors).toEqual([]);
});
