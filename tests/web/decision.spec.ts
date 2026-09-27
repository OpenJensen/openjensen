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

for (const width of [1440, 560, 320]) for (const theme of ['Light', 'Dark'] as const) test(`decision form has aligned full-width fields at ${width}px in ${theme.toLowerCase()} mode`, async ({ page }, testInfo) => {
  await page.setViewportSize({ width, height: 1100 });
  const { posts } = await lab(page, false);
  await page.getByRole('button', { name: theme, exact: true }).click();
  const region = page.getByRole('region', { name: 'Decision advisory', exact: true });
  await expect(region.getByRole('status')).toHaveText('Setup required');
  await expect(region.getByRole('group', { name: 'Comparison', exact: true })).toHaveCSS('border-top-width', '0px');
  await expect(region.locator('.workbench-field')).toHaveCount(4);
  expect(await page.getByLabel('Comparison instructions').evaluate(element => element.scrollHeight <= element.clientHeight + 1)).toBe(true);
  const fields = await region.locator('.workbench-field').evaluateAll(elements => elements.map(element => {
    const field = element.getBoundingClientRect();
    const label = element.querySelector('label')!.getBoundingClientRect();
    const control = element.querySelector('textarea')!.getBoundingClientRect();
    return { fieldX: field.x, fieldWidth: field.width, labelBottom: label.bottom, controlX: control.x, controlWidth: control.width, controlTop: control.top, controlBottom: control.bottom };
  }));
  for (const field of fields) {
    expect(field.controlTop - field.labelBottom).toBeGreaterThanOrEqual(6);
    expect(Math.abs(field.controlX - field.fieldX)).toBeLessThanOrEqual(1);
    expect(Math.abs(field.controlWidth - field.fieldWidth)).toBeLessThanOrEqual(1);
    expect(field.controlWidth).toBeGreaterThan(180);
  }
  expect(fields[1].labelBottom).toBeGreaterThan(fields[0].controlBottom + 16);
  if (width <= 700) expect(fields[3].controlTop).toBeGreaterThan(fields[2].controlBottom + 16);
  else expect(Math.abs(fields[2].controlTop - fields[3].controlTop)).toBeLessThanOrEqual(1);
  const add = await page.getByRole('button', { name: 'Add criterion', exact: true }).boundingBox();
  expect(add!.y - Math.max(fields[2].controlBottom, fields[3].controlBottom)).toBeGreaterThanOrEqual(12);
  const submit = page.getByRole('button', { name: 'Score criteria', exact: true });
  await expect(submit).toBeDisabled();
  expect((await submit.boundingBox())!.height).toBeGreaterThanOrEqual(44);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
  const screenshot = testInfo.outputPath(`decision-form-${width}-${theme.toLowerCase()}.png`);
  await page.screenshot({ path: screenshot, fullPage: true });
  await testInfo.attach('Decision form layout', { path: screenshot, contentType: 'image/png' });
  expect(posts).toEqual([]);
});

test('adding and removing criteria preserves keyboard focus without submitting', async ({ page }) => {
  const { posts } = await lab(page);
  await fill(page);
  await page.getByRole('button', { name: 'Add criterion', exact: true }).click();
  await expect(page.getByLabel('Criterion 3', { exact: true })).toBeFocused();
  await page.getByLabel('Criterion 3', { exact: true }).fill('Review the policy');
  await page.getByRole('button', { name: 'Remove criterion 2', exact: true }).click();
  await expect(page.getByLabel('Criterion 2', { exact: true })).toBeFocused();
  await expect(page.getByLabel('Criterion 2', { exact: true })).toHaveValue('Review the policy');
  await expect(page.getByLabel('Criterion 1', { exact: true })).toHaveValue('Inspect the dataset');
  expect(posts).toEqual([]);
});

test('pending scoring locks fields while cancellation stays reachable', async ({ page }) => {
  const { posts } = await lab(page);
  let release!: () => void;
  const held = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/api/v1/decision/score', async route => {
    posts.push(route.request().postDataJSON());
    await held;
    try { await route.abort(); } catch { /* The client may already have stopped scoring. */ }
  });
  try {
    await fill(page);
    await page.getByRole('button', { name: 'Score criteria', exact: true }).click();
    await expect.poll(() => posts.length).toBe(1);
    await expect(page.getByLabel('State to compare')).toBeDisabled();
    await expect(page.getByLabel('Criterion 1', { exact: true })).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Add criterion', exact: true })).toBeDisabled();
    const stop = page.getByRole('button', { name: 'Stop scoring', exact: true });
    await expect(stop).toBeEnabled();
    await stop.click();
    await expect(page.getByRole('alert').filter({ hasText: 'Scoring stopped' })).toBeVisible();
    await expect(page.getByLabel('State to compare')).toBeEnabled();
    expect(posts).toHaveLength(1);
  } finally { release(); }
});
