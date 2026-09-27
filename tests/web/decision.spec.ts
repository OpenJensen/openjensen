import { expect, test, type Page } from '@playwright/test';

async function lab(page: Page, configured = true) {
  const posts: any[] = [];
  const controls = { fail: false, slow: false };
  await page.route('**/api/v1/**', async route => {
    const req = route.request(), path = new URL(req.url()).pathname;
    if (path === '/api/v1/decision/score') {
      posts.push(req.postDataJSON());
      if (controls.slow) await new Promise(resolve => setTimeout(resolve, 1000));
      if (controls.fail) { await route.fulfill({ status: 503, json: { detail: 'PRIVATE must not show' } }); return; }
      await route.fulfill({ json: { selected_id: 'criterion-1', revision: '5afb8eeff127621fea2d66fc63f56798ada12eda', request_sha256: 'a'.repeat(64), scores: posts.at(-1).criteria.map((item: any) => ({ id: item.id, logit: 0, relative_weight: .5, tokens: 12 })), timing_ms: { load: 123, score: 10 } } }); return;
    }
    const replies: Record<string, unknown> = {
      '/api/v1/health': { status: 'ok', version: 'test' }, '/api/v1/capabilities': [],
      '/api/v1/projects': [],
      '/api/v1/decision/status': { configured, available: configured, busy: false, message: configured ? 'Configured to attempt local scoring; quality is experimental.' : 'The operator must configure the local scorer and accept its license.' },
    };
    await route.fulfill({ json: replies[path] ?? [] });
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Decision lab', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Decision lab', level: 1 })).toBeVisible();
  await expect(page.getByRole('group', { name: 'Advisory model', exact: true })).toContainText('Muose-50M');
  return { posts, controls };
}
async function fill(page: Page) {
  await page.getByLabel('State to compare').fill('The dataset is ready.');
  await page.getByLabel('Criterion 1', { exact: true }).fill('Inspect the dataset');
  await page.getByLabel('Criterion 2', { exact: true }).fill('Train a policy');
}

test('decision lab is disabled without explicit operator configuration', async ({ page }) => {
  const { posts } = await lab(page, false); await fill(page);
  await expect(page.getByRole('button', { name: 'Score criteria' })).toBeDisabled();
  await expect(page.getByText('Experimental · advisory only', { exact: true })).toBeVisible();
  await expect(page.getByText(/Only 3 of 6 workflow/)).not.toBeVisible();
  await page.getByText('Model details', { exact: true }).click();
  await expect(page.getByText(/Only 3 of 6 workflow/)).toBeVisible();
  expect(posts).toHaveLength(0);
});

test('manual scoring submits only bounded text and displays uncalibrated result', async ({ page }) => {
  const { posts } = await lab(page); await fill(page);
  await page.getByRole('button', { name: 'Score criteria' }).click();
  await expect(page.getByRole('region', { name: 'Advisory score result' })).toBeVisible();
  await expect(page.getByText(/no action was taken/)).toBeVisible();
  expect(posts).toHaveLength(1);
  expect(Object.keys(posts[0]).sort()).toEqual(['criteria', 'instructions', 'schema_version', 'state']);
  await page.getByLabel('State to compare').fill('Changed');
  await expect(page.getByRole('region', { name: 'Advisory score result' })).toHaveCount(0);
  const sizes = await page.evaluate(() => [document.documentElement.scrollWidth, document.documentElement.clientWidth]);
  expect(sizes[0]).toBeLessThanOrEqual(sizes[1] + 1);
});

test('failed scoring does not retry or expose private diagnostics', async ({ page }) => {
  const { posts, controls } = await lab(page); controls.fail = true; await fill(page);
  await page.getByRole('button', { name: 'Score criteria' }).click();
  await expect(page.getByRole('alert').filter({ hasText: 'No score was accepted' })).toBeVisible();
  await expect(page.getByText('PRIVATE must not show')).toHaveCount(0);
  await page.waitForTimeout(350); expect(posts).toHaveLength(1);
});

test('criteria enforce uniqueness and two to eight choices', async ({ page }) => {
  await lab(page); await fill(page);
  await page.getByLabel('Criterion 2', { exact: true }).fill('Inspect the dataset');
  await expect(page.getByRole('button', { name: 'Score criteria' })).toBeDisabled();
  for (let i = 0; i < 6; i++) await page.getByRole('button', { name: 'Add criterion' }).click();
  await expect(page.getByRole('button', { name: 'Add criterion' })).toBeDisabled();
  for (let i = 8; i > 2; i--) await page.getByRole('button', { name: `Remove criterion ${i}`, exact: true }).click();
  await expect(page.getByRole('button', { name: 'Remove criterion 2', exact: true })).toHaveCount(0);
});
