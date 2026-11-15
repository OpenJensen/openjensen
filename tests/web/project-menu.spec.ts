import { fulfillApplicationUpdateRead } from './application-update-controls';
import { expect, test, type Page } from '@playwright/test';
import { openCreateProject, selectProject } from './project-controls';
import { installedModules } from './module-fixtures';

async function fixture(page: Page, empty = false) {
  const projects = empty
    ? []
    : [
        { id: 'alpha', name: 'Robotics examples', created_at: '2026-10-06T00:00:00Z' },
        { id: 'beta', name: 'PushT benchmark', created_at: '2026-10-06T00:00:00Z' },
        { id: 'gamma', name: 'Psi-Zero validation', created_at: '2026-10-06T00:00:00Z' },
      ];
  const writes: unknown[] = [];
  const control = { fail: false, held: null as Promise<void> | null };
  await page.route('**/api/v1/**', async (route) => {
    if (await fulfillApplicationUpdateRead(route)) return;
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() === 'POST') {
      writes.push({ path, body: request.postDataJSON() });
      if (path !== '/api/v1/projects')
        return route.fulfill({ status: 405, json: { detail: 'No jobs permitted in project fixture.' } });
      if (control.held) await control.held;
      if (control.fail)
        return route.fulfill({ status: 503, json: { detail: 'Project service is temporarily unavailable.' } });
      const created = { id: 'created-project', name: request.postDataJSON().name, created_at: '2026-10-06T00:00:00Z' };
      projects.push(created);
      return route.fulfill({ status: 201, json: created });
    }
    if (path === '/api/v1/health') return route.fulfill({ json: { status: 'ok', version: 'project-menu-fixture' } });
    if (path === '/api/v1/projects') return route.fulfill({ json: projects });
    if (path === '/api/v1/modules')
      return route.fulfill({ json: installedModules.filter((item) => item.id === 'datasets') });
    return route.fulfill({ json: [] });
  });
  await page.goto('/datasets/');
  await expect(page.getByRole('button', { name: 'Current project', exact: true })).toBeEnabled();
  return { writes, control, projects };
}

const current = (page: Page) => page.getByRole('button', { name: 'Current project', exact: true });
const menu = (page: Page) => page.getByRole('menu', { name: 'Projects', exact: true });
const dialog = (page: Page) => page.getByRole('dialog', { name: 'Create project', exact: true });

test('project switching and keyboard navigation preserve scope without creating jobs', async ({ page }, testInfo) => {
  const { writes } = await fixture(page);
  await expect(current(page)).toContainText('Robotics examples');
  await expect(page.getByRole('textbox', { name: 'Project name', exact: true })).toHaveCount(0);
  await current(page).click();
  await expect(menu(page).getByRole('menuitemradio', { name: 'Robotics examples', exact: true })).toHaveAttribute(
    'aria-checked',
    'true',
  );
  await expect(menu(page).getByRole('menuitem', { name: 'Create project…', exact: true })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('project-menu.png') });
  await page.keyboard.press('Escape');
  await expect(current(page)).toBeFocused();
  await selectProject(page, 'beta');
  await page.reload();
  await expect(current(page)).toHaveAttribute('data-project-id', 'beta');
  await current(page).focus();
  await page.keyboard.press('ArrowDown');
  await expect(menu(page).getByRole('menuitemradio', { name: 'PushT benchmark', exact: true })).toBeFocused();
  await page.keyboard.press('Home');
  await expect(menu(page).getByRole('menuitemradio', { name: 'Robotics examples', exact: true })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(current(page)).toHaveAttribute('data-project-id', 'alpha');
  await expect(current(page)).toBeFocused();
  await page.keyboard.press('ArrowDown');
  await expect(menu(page).getByRole('menuitemradio', { name: 'Robotics examples', exact: true })).toBeFocused();
  await page.keyboard.press('p');
  await expect(menu(page).getByRole('menuitemradio', { name: 'PushT benchmark', exact: true })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(current(page)).toHaveAttribute('data-project-id', 'beta');
  await current(page).click();
  await page.keyboard.press('Tab');
  await expect(menu(page)).toHaveCount(0);
  await expect(page.getByRole('link', { name: 'Home', exact: true })).toBeFocused();
  await current(page).click();
  // The popover may cover the page heading on a narrow screen. Exercise a
  // genuine outside click at a visible content point below its bounded menu.
  const bounds = await menu(page).boundingBox();
  const outside = await page.evaluate(() => ({
    x: document.documentElement.clientWidth - 8,
    y: window.innerHeight - 8,
  }));
  expect(outside.y).toBeGreaterThan(bounds!.y + bounds!.height);
  await page.mouse.click(outside.x, outside.y);
  await expect(menu(page)).toHaveCount(0);
  expect(writes).toEqual([]);
});

test('creating from an empty menu focuses the name field and restores the new project after reload', async ({
  page,
}) => {
  const { writes } = await fixture(page, true);
  await expect(current(page)).toContainText('Select or create a project');
  await openCreateProject(page);
  await expect(dialog(page).getByLabel('Project name', { exact: true })).toBeFocused();
  await dialog(page).getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(dialog(page)).toHaveCount(0);
  await expect(current(page)).toBeFocused();
  expect(writes).toEqual([]);
  await openCreateProject(page);
  await dialog(page).getByLabel('Project name', { exact: true }).fill('  Robot lab  ');
  await page.keyboard.press('Enter');
  await expect(dialog(page)).toHaveCount(0);
  await expect(current(page)).toHaveAttribute('data-project-id', 'created-project');
  await expect(current(page)).toContainText('Robot lab');
  await page.reload();
  await expect(current(page)).toHaveAttribute('data-project-id', 'created-project');
  expect(writes).toEqual([{ path: '/api/v1/projects', body: { name: 'Robot lab' } }]);
});

test('a first visit shows the whole menu, so the first project can be created without scrolling', async ({ page }) => {
  await fixture(page, true);
  await current(page).click();
  const create = menu(page).getByRole('menuitem', { name: 'Create project…', exact: true });
  await expect(menu(page).getByText('No projects yet.', { exact: true })).toBeVisible();
  await expect(create).toBeVisible();
  // Visible to a person, not only present: nothing in the menu is cut off or scrolled away.
  expect(await menu(page).evaluate((node) => node.scrollHeight - node.clientHeight)).toBeLessThanOrEqual(0);
  const outer = (await menu(page).boundingBox())!;
  for (const part of [menu(page).getByText('No projects yet.', { exact: true }), create]) {
    const inner = (await part.boundingBox())!;
    expect(inner.y).toBeGreaterThanOrEqual(outer.y);
    expect(inner.y + inner.height).toBeLessThanOrEqual(outer.y + outer.height);
  }
});

test('a failed creation retains the draft and pending retries block duplicate submissions', async ({ page }) => {
  const { control, writes } = await fixture(page);
  control.fail = true;
  await openCreateProject(page);
  await dialog(page).getByLabel('Project name', { exact: true }).fill('Retry lab');
  await dialog(page).getByRole('button', { name: 'Create project', exact: true }).click();
  await expect(dialog(page).getByRole('alert')).toContainText('temporarily unavailable');
  await expect(dialog(page).getByLabel('Project name', { exact: true })).toHaveValue('Retry lab');
  await expect(current(page)).toHaveAttribute('data-project-id', 'alpha');
  control.fail = false;
  let release!: () => void;
  control.held = new Promise<void>((resolve) => {
    release = resolve;
  });
  try {
    await dialog(page).getByRole('button', { name: 'Create project', exact: true }).click();
    await expect(dialog(page).getByRole('status')).toHaveText('Creating project…');
    await page.keyboard.press('Enter');
    await page.keyboard.press('Escape');
    await expect(dialog(page)).toBeVisible();
    await expect(dialog(page).getByRole('button', { name: 'Cancel', exact: true })).toBeDisabled();
    expect(writes).toHaveLength(2);
  } finally {
    release();
    control.held = null;
  }
  await expect(dialog(page)).toHaveCount(0);
  await expect(current(page)).toHaveAttribute('data-project-id', 'created-project');
  expect(writes).toHaveLength(2);
});

test('long names and creation fit a narrow dark screen without clipping the menu', async ({ page }, testInfo) => {
  const { projects, writes } = await fixture(page);
  projects[1].name = 'Robot lab · ' + 'long project name '.repeat(4);
  await page.setViewportSize({ width: 320, height: 720 });
  await page.reload();
  await page.getByRole('button', { name: 'Dark', exact: true }).click();
  await current(page).click();
  const bounds = await menu(page).boundingBox();
  expect(bounds!.x).toBeGreaterThanOrEqual(0);
  expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(320);
  expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(720);
  await expect(menu(page).getByRole('menuitem', { name: 'Create project…', exact: true })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('project-menu-dark-mobile.png') });
  await menu(page).getByRole('menuitem', { name: 'Create project…', exact: true }).click();
  await expect(dialog(page).getByLabel('Project name', { exact: true })).toBeFocused();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1),
  ).toBe(true);
  await page.keyboard.press('Escape');
  await expect(current(page)).toBeFocused();
  expect(writes).toEqual([]);
});

test('projects move in from another workspace and are removed only on the second click', async ({ page }, testInfo) => {
  const { projects, writes } = await fixture(page);
  const calls: { method: string; path: string; body?: unknown }[] = [];
  const nearby = [
    {
      path: '/home/robot/.local/share/older-workspace',
      label: '~/.local/share/older-workspace',
      available: true,
      message: null,
      projects: [{ id: 'pen', name: 'SO101 pen pickup', created_at: '2026-10-31T00:00:00Z', datasets: 2, jobs: 10 }],
    },
    {
      path: '/home/robot/busy-workspace',
      label: '~/busy-workspace',
      available: false,
      message: 'Open in another OpenJensen app. Close it to move projects from here.',
      projects: [{ id: 'busy', name: 'Busy project', created_at: '2026-10-31T00:00:00Z', datasets: 0, jobs: 1 }],
    },
  ];
  await page.route('**/api/v1/projects/**', async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (path === '/api/v1/projects/nearby') return route.fulfill({ json: nearby });
    if (path === '/api/v1/projects/move' && request.method() === 'POST') {
      calls.push({ method: 'POST', path, body: request.postDataJSON() });
      const moved = nearby[0].projects.shift()!;
      const project = { id: moved.id, name: moved.name, created_at: moved.created_at };
      projects.push(project);
      return route.fulfill({ json: project });
    }
    if (request.method() === 'DELETE') {
      calls.push({ method: 'DELETE', path });
      const id = path.split('/').pop()!;
      const [removed] = projects.splice(
        projects.findIndex((project) => project.id === id),
        1,
      );
      return route.fulfill({ json: { id, name: removed.name, datasets: 0, jobs: 0 } });
    }
    return route.fallback();
  });
  await current(page).click();
  await menu(page).getByRole('menuitem', { name: 'Manage projects…', exact: true }).click();
  const manage = page.getByRole('dialog', { name: 'Projects', exact: true });
  await expect(manage.getByText('~/.local/share/older-workspace', { exact: true })).toBeVisible();
  await expect(manage.getByText('2 datasets · 10 jobs', { exact: true })).toBeVisible();
  await expect(manage.getByRole('button', { name: 'Move Busy project here', exact: true })).toBeDisabled();
  await expect(manage.getByText('Open in another OpenJensen app.', { exact: false })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('project-manage.png') });

  await manage.getByRole('button', { name: 'Move SO101 pen pickup here', exact: true }).click();
  await expect(manage.getByRole('status')).toHaveText('Moved SO101 pen pickup here.');
  await expect(current(page)).toHaveAttribute('data-project-id', 'pen');
  await expect(manage.getByRole('button', { name: 'Remove SO101 pen pickup', exact: true })).toBeVisible();

  await manage.getByRole('button', { name: 'Remove PushT benchmark', exact: true }).click();
  await manage.getByRole('button', { name: 'Keep', exact: true }).click();
  expect(calls.filter((call) => call.method === 'DELETE')).toEqual([]);
  await manage.getByRole('button', { name: 'Remove PushT benchmark', exact: true }).click();
  await expect(manage.getByText('Datasets, jobs and recordings are deleted.', { exact: true })).toBeVisible();
  await manage.getByRole('button', { name: 'Remove PushT benchmark', exact: true }).click();
  await expect(manage.getByRole('status')).toHaveText('Removed PushT benchmark.');
  await expect(manage.getByRole('button', { name: 'Remove PushT benchmark', exact: true })).toHaveCount(0);
  expect(calls).toEqual([
    {
      method: 'POST',
      path: '/api/v1/projects/move',
      body: { workspace: '/home/robot/.local/share/older-workspace', project_id: 'pen' },
    },
    { method: 'DELETE', path: '/api/v1/projects/beta' },
  ]);
  expect(writes).toEqual([]);
});
