import { expect, test, type Page } from '@playwright/test';
import { datasetStarters } from '../../apps/web/src/lib/dataset-starters';

async function openExamples(page: Page) {
  const timestamp = '2026-09-26T12:00:00Z';
  await page.route('**/api/v1/projects', route => route.fulfill({ json: [{ id: 'starter-test', name: 'Examples', created_at: timestamp }] }));
  await page.route('**/api/v1/projects/starter-test/jobs', route => route.fulfill({ json: [] }));
  await page.goto('/');
  return page.getByRole('region', { name: 'Example datasets' });
}

test('every example fills its pinned source without starting an inspection', async ({ page }) => {
  const writes: unknown[] = [];
  await page.route('**/api/v1/projects/starter-test/intakes', route => {
    writes.push(route.request().postDataJSON());
    return route.fulfill({ status: 503, json: { detail: 'Fixture stops after validating the submitted source.' } });
  });
  await page.route('**/api/v1/projects/starter-test/submissions/*?operation=dataset.inspect', route => {
    const key = decodeURIComponent(new URL(route.request().url()).pathname.split('/').at(-1)!);
    return route.fulfill({ status: 404, headers: { 'Idempotency-Key': key, 'Cache-Control': 'no-store' }, json: { detail: 'No accepted submission found for this project, operation and key' } });
  });
  const gallery = await openExamples(page);
  await expect(gallery.locator('.starter-card')).toHaveCount(2);
  for (const starter of datasetStarters) {
    const card = gallery.getByRole('button').filter({ has: page.getByRole('heading', { name: starter.title, exact: true }) });
    await card.click();
    await expect(card).toHaveAttribute('aria-pressed', 'true');
    await expect(gallery.locator('.starter-card[aria-pressed="true"]')).toHaveCount(1);
    await expect(page.getByLabel('Dataset repository')).toHaveValue(starter.repoId);
    await expect(page.getByLabel('Revision', { exact: true })).toHaveValue(starter.revision);
  }
  expect(writes).toEqual([]);
  await page.getByRole('button', { name: 'Inspect dataset', exact: true }).click();
  await expect.poll(() => writes).toEqual([{ source: 'huggingface', repo_id: 'lerobot/svla_so100_pickplace', revision: '728583b5eaf9e739a7f119e2def466fa1d552402' }]);
  await page.getByLabel('Dataset repository').fill('my-org/my-dataset');
  await expect(gallery.locator('.starter-card[aria-pressed="true"]')).toHaveCount(0);
});

test('example camera stills load and the gallery fits the viewport', async ({ page }, testInfo) => {
  const gallery = await openExamples(page);
  for (const image of await gallery.getByRole('img').all()) {
    await image.scrollIntoViewIfNeeded();
    await expect.poll(() => image.evaluate(node => node instanceof HTMLImageElement && node.complete && node.naturalWidth > 0)).toBe(true);
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(await page.evaluate(() => document.documentElement.clientWidth + 1));
  await gallery.scrollIntoViewIfNeeded();
  const screenshot = testInfo.outputPath('dataset-examples.png');
  await gallery.screenshot({ path: screenshot });
  await testInfo.attach('dataset-examples', { path: screenshot, contentType: 'image/png' });
});
