import { expect, test } from '@playwright/test';

const old = '1'.repeat(40);
const target = '2'.repeat(40);
const initial = {
  installed: { version: '0.1.0', commit: old, host: 'Workspace server' },
  channel: 'main',
  upstream_commit: target,
  latest_release: 'v0.0.1',
  state: 'available',
  checked_at: '2026-10-24T14:00:00Z',
  message: 'An update from upstream main is available.',
  automatic_update: true,
  setup_message: null,
  operation: null,
};

test('update discovery is read only and confirmation identifies the server and exact commit', async ({ page }) => {
  const writes: unknown[] = [];
  await page.route('**/api/v1/updates**', (route) => {
    if (route.request().method() === 'POST') writes.push(route.request().postDataJSON());
    return route.fulfill({ json: initial });
  });
  await page.goto('/dashboard/');
  await page.getByRole('button', { name: 'Version and updates' }).click();
  const dialog = page.getByRole('dialog', { name: 'OpenJensen updates' });
  await expect(dialog).toContainText('Workspace server');
  await expect(dialog).toContainText('v0.1.0');
  await expect(dialog).toContainText('v0.0.1');
  await expect(dialog.getByRole('link', { name: target.slice(0, 8) })).toHaveAttribute(
    'href',
    `https://github.com/OpenJensen/openjensen/commit/${target}`,
  );
  await expect(dialog.getByRole('button', { name: 'Update & restart' })).toBeVisible();
  await dialog.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(dialog).not.toBeVisible();
  expect(writes).toEqual([]);
});

test('accepted update reports progress and reloads once the confirmed version is running', async ({ page }) => {
  let state: any = initial;
  let writes = 0;
  let complete = false;

  await page.route('**/api/v1/updates**', (route) => {
    if (route.request().method() === 'POST') {
      writes += 1;
      expect(route.request().postDataJSON()).toMatchObject({
        target_commit: target,
        confirmation: 'update-and-restart',
      });
      state = {
        ...initial,
        operation: {
          id: route.request().postDataJSON().operation_id,
          phase: 'building',
          target_commit: target,
          message: 'Building the new version…',
        },
      };
    }
    if (complete)
      state = {
        ...state,
        installed: { ...initial.installed, commit: target },
        state: 'current',
        operation: { ...state.operation, phase: 'complete' },
      };
    return route.fulfill({ status: route.request().method() === 'POST' ? 202 : 200, json: state });
  });
  await page.goto('/dashboard/');
  const initialOrigin = await page.evaluate(() => performance.timeOrigin);
  await page.getByRole('button', { name: 'Version and updates' }).click();
  await page.getByRole('button', { name: 'Update & restart' }).click();
  await expect(page.getByRole('dialog')).toContainText('Building the new version…');
  complete = true;
  await expect.poll(() => page.evaluate(() => performance.timeOrigin)).not.toBe(initialOrigin);
  await expect(page.getByRole('button', { name: 'Version and updates' })).toHaveText('v0.1.0');
  await expect(page.getByRole('button', { name: 'Version and updates' })).toHaveText('v0.1.0');
  expect(writes).toBe(1);
  expect(await page.evaluate(() => sessionStorage.getItem('openjensen-pending-update'))).toBeNull();
});

test('busy server error stays in the confirmation dialog without claiming an update', async ({ page }) => {
  await page.route('**/api/v1/updates**', (route) =>
    route.request().method() === 'POST'
      ? route.fulfill({
          status: 409,
          json: { detail: 'Stop Robot Lab and release the motors before updating this server.' },
        })
      : route.fulfill({ json: initial }),
  );
  await page.goto('/dashboard/');
  await page.getByRole('button', { name: 'Version and updates' }).click();
  await page.getByRole('button', { name: 'Update & restart' }).click();
  await expect(page.getByRole('dialog').getByRole('alert')).toContainText('Stop Robot Lab');
  await expect(page.getByRole('button', { name: 'Update & restart' })).toBeEnabled();
});

test('development builds are identified without offering a downgrade to main', async ({ page }) => {
  await page.route('**/api/v1/updates**', (route) =>
    route.fulfill({
      json: { ...initial, state: 'ahead', message: 'This development build is ahead of upstream main.' },
    }),
  );
  await page.goto('/dashboard/');
  await page.getByRole('button', { name: 'Version and updates' }).click();
  await expect(page.getByRole('dialog')).toContainText('ahead of upstream main');
  await expect(page.getByRole('button', { name: 'Update & restart' })).toHaveCount(0);
});

test('lost acceptance reply reconnects to the same update without submitting twice', async ({ page }) => {
  let state: any = initial;
  let writes = 0;
  let complete = false;

  await page.route('**/api/v1/updates**', (route) => {
    if (route.request().method() === 'POST') {
      writes += 1;
      state = {
        ...initial,
        operation: {
          id: route.request().postDataJSON().operation_id,
          phase: 'building',
          target_commit: target,
          message: 'Building the new version…',
        },
      };
      return route.abort('connectionreset');
    }
    if (complete)
      state = {
        ...state,
        installed: { ...initial.installed, commit: target },
        state: 'current',
        operation: { ...state.operation, phase: 'complete' },
      };
    return route.fulfill({ json: state });
  });
  await page.goto('/dashboard/');
  const initialOrigin = await page.evaluate(() => performance.timeOrigin);
  await page.getByRole('button', { name: 'Version and updates' }).click();
  await page.getByRole('button', { name: 'Update & restart' }).click();
  await expect(page.getByRole('dialog')).toContainText('Building the new version…');
  complete = true;
  await expect.poll(() => page.evaluate(() => performance.timeOrigin)).not.toBe(initialOrigin);
  await expect(page.getByRole('button', { name: 'Version and updates' })).toHaveText('v0.1.0');
  expect(writes).toBe(1);
  expect(await page.evaluate(() => sessionStorage.getItem('openjensen-pending-update'))).toBeNull();
});

test('unavailable browser storage keeps the workspace and update dialog usable', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.addInitScript(() => {
    Object.defineProperty(window, 'sessionStorage', {
      get() {
        throw new DOMException('Storage disabled', 'SecurityError');
      },
    });
  });
  await page.route('**/api/v1/updates**', (route) => route.fulfill({ json: initial }));
  await page.goto('/dashboard/');
  await expect(page.getByRole('heading', { name: 'Dashboard', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Version and updates' }).click();
  await expect(page.getByRole('dialog', { name: 'OpenJensen updates' })).toBeVisible();
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  await page.getByRole('link', { name: 'Dataset', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Dataset', exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});
