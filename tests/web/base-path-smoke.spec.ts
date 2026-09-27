import { expect, test } from '@playwright/test';

const origin = `http://127.0.0.1:${process.env.FIREBIRD_PREFIX_SMOKE_PORT ?? '18766'}`;
const prefix = (process.env.NEXT_PUBLIC_BASE_PATH ?? '/firebird').replace(/\/+$/, '');

test('prefixed export loads bundles, API, dataset posters and docs without escaping its mount', async ({ page }) => {
  const escaped: string[] = [];
  const failures: string[] = [];
  page.on('request', request => {
    const url = new URL(request.url());
    if (url.origin === origin && !url.pathname.startsWith(`${prefix}/`)) escaped.push(url.pathname);
  });
  page.on('response', response => {
    if (response.url().startsWith(origin) && response.status() >= 400) failures.push(`${response.status()} ${response.url()}`);
  });
  await page.goto(`${prefix}/`);
  await expect(page.getByRole('link', { name: 'OPEN JENSEN workspace home' })).toHaveAttribute('href', `${prefix}/`);
  await expect(page.getByRole('button', { name: 'Fine-tune', exact: true })).toBeVisible();
  const posters = page.locator('.starter-image img');
  await expect(posters).toHaveCount(2);
  for (const poster of await posters.all()) {
    await expect(poster).toHaveAttribute('src', new RegExp(`^${prefix}/datasets/`));
    await expect.poll(() => poster.evaluate(node => (node as HTMLImageElement).complete && (node as HTMLImageElement).naturalWidth > 0)).toBe(true);
  }
  await expect.poll(() => page.locator('.connection-notice').count()).toBe(0);
  await page.getByRole('button', { name: 'Cloud runs', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Cloud run monitor', exact: true })).toBeVisible();
  await expect(page.getByText('Cloud monitoring is not configured.', { exact: false })).toBeVisible();
  await page.getByRole('button', { name: 'Dataset', exact: true }).click();
  const teachingState = page.waitForResponse(response => new URL(response.url()).pathname === `${prefix}/api/v1/teaching/state`);
  await page.getByRole('button', { name: 'Teaching', exact: true }).click();
  expect((await teachingState).status()).toBe(200);
  await expect(page.getByRole('heading', { name: 'Teach in simulation', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Start recording', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Connect voice', exact: true })).toBeDisabled();
  await page.getByRole('link', { name: 'API reference', exact: true }).first().click();
  await expect(page).toHaveURL(new RegExp(`${prefix}/docs/$`));
  await expect(page.getByRole('heading', { name: 'API reference', exact: true })).toBeVisible();
  await expect(page.getByText(origin + prefix + '/api/v1', { exact: true })).toBeVisible();
  await expect(page.getByRole('link', { name: 'OpenAPI JSON', exact: true })).toHaveAttribute('href', `${prefix}/openapi.json`);
  await expect(page.locator('.reference-endpoint')).not.toHaveCount(0);
  await page.locator('.reference-endpoint > summary').first().click();
  await expect(page.locator('.reference-example .reference-code').first()).toContainText(`${origin}${prefix}/`);
  await page.getByRole('link', { name: 'Back to workspace', exact: true }).click();
  await expect(page).toHaveURL(new RegExp(`${prefix}/$`));
  expect(escaped).toEqual([]);
  expect(failures).toEqual([]);
});
