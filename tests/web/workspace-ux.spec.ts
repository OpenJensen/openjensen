import { expect, test, type Page } from '@playwright/test';

async function workspace(page: Page) {
  const mutations: string[] = [];
  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() !== 'GET') {
      mutations.push(path);
      return route.fulfill({ status: 405, json: { detail: 'Navigation must not submit a job.' } });
    }
    if (path === '/api/v1/projects') return route.fulfill({ json: [{ id: 'ux-review', name: 'Robot workspace', created_at: '2026-09-27T12:00:00Z' }] });
    if (path.endsWith('/projects/ux-review/jobs') || path.endsWith('/projects/ux-review/artifacts')) return route.fulfill({ json: [] });
    if (path === '/api/v1/policy-options') return route.fulfill({ json: {
      runtimes: [], sources: [], training_models: [], training_methods: [],
      default_training_method: 'lora', quantization_defaults: { cpu: { language: 'Q8_0', vision: null }, cuda: { language: 'Q8_0', vision: null }, note: '' },
    } });
    return route.continue();
  });
  await page.goto('/');
  await expect(page.getByLabel('Current project')).toHaveValue('ux-review');
  return mutations;
}

test('data tools are first-class pages with clear return paths and no job submission', async ({ page }) => {
  const mutations = await workspace(page);
  const navigation = page.getByRole('navigation', { name: 'Policy lifecycle' });
  for (const name of ['Augmentation', 'Teaching', 'Decision lab']) {
    const entry = navigation.getByRole('button', { name, exact: true });
    await entry.click();
    await expect(entry).toHaveAttribute('aria-current', 'page');
    await expect(page.getByRole('heading', { name, level: 1, exact: true })).toBeVisible();
    await expect(page.locator('.workflow-panel')).toHaveCount(0);
  }
  await navigation.getByRole('button', { name: 'Augmentation', exact: true }).click();
  await page.getByRole('button', { name: 'Import a dataset', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Dataset', level: 1, exact: true })).toBeVisible();
  expect(mutations).toEqual([]);
});

test('run choices are visible, keyboard accessible and stay within a narrow screen', async ({ page }, testInfo) => {
  const mutations = await workspace(page);
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  const choices = page.getByRole('group', { name: 'Run mode', exact: true });
  await expect(choices.getByRole('button')).toHaveCount(3);
  for (const name of ['3D simulation', 'Replay observations', 'Check inference']) {
    const choice = choices.getByRole('button', { name, exact: true });
    await choice.focus();
    await page.keyboard.press('Enter');
    await expect(choice).toHaveAttribute('aria-pressed', 'true');
  }
  await choices.getByRole('button', { name: '3D simulation', exact: true }).click();
  await page.screenshot({ path: testInfo.outputPath('run-workspace.png'), fullPage: false });
  await page.setViewportSize({ width: 320, height: 900 });
  await expect(page.getByRole('navigation', { name: 'Policy lifecycle' }).getByRole('button', { name: 'Augmentation', exact: true })).toBeVisible();
  await expect(choices.getByRole('button', { name: '3D simulation', exact: true })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
  expect(mutations).toEqual([]);
});


test('workspace pages use a single title without introductory subtitles', async ({ page }, testInfo) => {
  await workspace(page);
  const navigation = page.getByRole('navigation', { name: 'Policy lifecycle' });
  for (const name of ['Dataset', 'Augmentation', 'Teaching', 'Fine-tune', 'Distill', 'Quantize', 'Evaluate', 'Run', 'Decision lab', 'Cloud runs', 'Settings & diagnostics']) {
    await navigation.getByRole('button', { name, exact: true }).click();
    await expect(page.getByRole('heading', { name, level: 1, exact: true })).toBeVisible();
    await expect(page.locator('.page-heading p')).toHaveCount(0);
    await expect(page.locator('.page-heading .page-guide')).toBeVisible();
  }
  await navigation.getByRole('button', { name: 'Distill', exact: true }).click();
  await expect(page.getByText('Teach a smaller ACT policy', { exact: true })).toHaveCount(0);
  await expect(page.getByText('Train an ACT256 student to imitate your teacher’s action chunks, then reload the saved student in a fresh process.', { exact: true })).toHaveCount(0);
  await expect(page.getByRole('group', { name: 'Teacher', exact: true })).toHaveCount(0);
  await page.screenshot({ path: testInfo.outputPath('clean-distill.png'), fullPage: false });
});

test('the contextual guide covers every section and keeps API docs separate', async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await workspace(page);
  await page.getByRole('button', { name: 'Distill', exact: true }).click();
  await page.locator('.page-heading .page-guide').click();
  await expect(page).toHaveURL(/\/guide\/#distill$/);
  await expect(page.getByRole('heading', { name: 'Workspace guide', exact: true })).toBeVisible();
  const sections = page.locator('.guide-sections > section');
  await expect(sections).toHaveCount(11);
  for (const name of ['Dataset', 'Augmentation', 'Teaching', 'Fine-tune', 'Distill', 'Quantize', 'Evaluate', 'Run', 'Decision lab', 'Cloud runs', 'Settings & diagnostics']) {
    await expect(sections.getByRole('heading', { name, exact: true })).toBeVisible();
  }
  await expect(page.locator('#distill')).toContainText('ACT256');
  await expect(page.getByRole('link', { name: 'Distillation setup', exact: true })).toHaveAttribute('href', 'https://github.com/sobhanb-eth/firebird-hackathon-codebase/blob/main/workers/policy_distillation/README.md');
  await page.reload();
  await expect(page.locator('#distill')).toContainText('ACT256');
  await page.setViewportSize({ width: 320, height: 900 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
  await page.getByRole('navigation', { name: 'Guide sections' }).getByRole('link', { name: 'Augmentation', exact: true }).click();
  await expect(page).toHaveURL(/#augmentation$/);
  await page.getByRole('link', { name: 'API reference', exact: true }).filter({ visible: true }).click();
  await expect(page).toHaveURL(/\/docs\/$/);
  await expect(page.getByRole('heading', { name: 'API reference', exact: true })).toBeVisible();
  await page.getByRole('link', { name: 'Workspace guide', exact: true }).click();
  await expect(page).toHaveURL(/\/guide\/$/);
  await page.screenshot({ path: testInfo.outputPath('workspace-guide.png'), fullPage: false });
  expect(errors).toEqual([]);
});


test('fresh workflows require explicit model and runner choices', async ({ page }, testInfo) => {
  const mutations = await workspace(page);
  await page.getByRole('button', { name: 'Quantize', exact: true }).click();
  const quantizers = page.getByRole('group', { name: 'Quantization mode', exact: true });
  await expect(quantizers.getByRole('button', { name: 'SmolVLA', exact: true })).toHaveAttribute('aria-pressed', 'false');
  await expect(quantizers.getByRole('button', { name: 'ACT', exact: true })).toHaveAttribute('aria-pressed', 'false');
  await page.getByRole('region', { name: 'Quantization workflow', exact: true }).screenshot({ path: testInfo.outputPath('quantization-model-cards.png') });
  await expect(page.locator('.workflow-panel')).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Native ACT quantization', exact: true })).toHaveCount(0);

  await page.getByRole('button', { name: 'Run', exact: true }).click();
  const runners = page.getByRole('group', { name: 'Run mode', exact: true });
  for (const name of ['3D simulation', 'Replay observations', 'Check inference']) {
    await expect(runners.getByRole('button', { name, exact: true })).toHaveAttribute('aria-pressed', 'false');
  }
  await expect(page.locator('.workflow-panel')).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Native Isaac simulation', exact: true })).toHaveCount(0);

  await page.getByRole('button', { name: 'Fine-tune', exact: true }).click();
  await page.getByRole('button', { name: 'Start a new fine-tuning', exact: true }).click();
  await page.getByRole('navigation', { name: 'Training setup' }).getByRole('button', { name: 'Model', exact: true }).click();
  const models = page.getByRole('group', { name: 'Base model', exact: true });
  await expect(models.getByRole('radio', { checked: true })).toHaveCount(0);
  await page.getByRole('radio', { name: 'SmolVLA', exact: true }).locator('..').click();
  await expect(page.getByRole('radio', { name: 'SmolVLA', exact: true })).toBeChecked();
  expect(mutations).toEqual([]);
});
