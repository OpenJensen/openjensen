import { selectProject } from './project-controls';
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
  await page.goto('/datasets/');
  await expect(page.getByLabel('Current project')).toHaveAttribute('data-project-id', 'ux-review');
  return mutations;
}

test('data tools are first-class pages with clear return paths and no job submission', async ({ page }) => {
  const mutations = await workspace(page);
  await expect(page.locator('.stage-order')).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Workflow context' })).toHaveCount(0);
  const navigation = page.getByRole('navigation', { name: 'Policy lifecycle' });
  for (const name of ['Augmentation', 'Teaching', 'Decision lab']) {
    const entry = navigation.getByRole('link', { name, exact: true });
    await entry.click();
    await expect(entry).toHaveAttribute('aria-current', 'page');
    await expect(page.getByRole('heading', { name, level: 1, exact: true })).toBeVisible();
    await expect(page.locator('.workflow-panel')).toHaveCount(0);
  }
  await navigation.getByRole('link', { name: 'Augmentation', exact: true }).click();
  await page.getByRole('button', { name: 'Import a dataset', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Dataset', level: 1, exact: true })).toBeVisible();
  expect(mutations).toEqual([]);
});

test('run choices are visible, keyboard accessible and stay within a narrow screen', async ({ page }, testInfo) => {
  const mutations = await workspace(page);
  await page.getByRole('link', { name: 'Run', exact: true }).click();
  const choices = page.getByRole('group', { name: 'Run mode', exact: true });
  await expect(choices.getByRole('button')).toHaveCount(3);
  for (const name of ['3D simulation', 'Replay observations', 'Check inference']) {
    const choice = choices.getByRole('button', { name, exact: true });
    await choice.focus();
    await page.keyboard.press('Enter');
    await expect(choice).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByRole('region', { name: 'Workflow context', exact: true })).toHaveCount(0);
  }
  await choices.getByRole('button', { name: '3D simulation', exact: true }).click();
  await page.screenshot({ path: testInfo.outputPath('run-workspace.png'), fullPage: false });
  await page.setViewportSize({ width: 320, height: 900 });
  await expect(page.getByRole('navigation', { name: 'Policy lifecycle' }).getByRole('link', { name: 'Augmentation', exact: true })).toBeVisible();
  await expect(choices.getByRole('button', { name: '3D simulation', exact: true })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
  expect(mutations).toEqual([]);
});


test('workspace pages use a single title without introductory subtitles', async ({ page }, testInfo) => {
  await workspace(page);
  const navigation = page.getByRole('navigation', { name: 'Policy lifecycle' });
  for (const name of ['Dataset', 'Augmentation', 'Teaching', 'Fine-tune', 'Distill', 'Quantize', 'Evaluate', 'Run', 'Decision lab', 'Cloud runs', 'Settings & diagnostics', 'Dashboard']) {
    await navigation.getByRole('link', { name, exact: true }).click();
    await expect(page.getByRole('heading', { name, level: 1, exact: true })).toBeVisible();
    await expect(page.locator('.page-heading p')).toHaveCount(0);
    await expect(page.locator('.page-heading .page-guide')).toBeVisible();
    await expect(page.locator('.journey-context')).toHaveCount(0);
    await expect(page.getByText('Latest inspection', { exact: true })).toHaveCount(0);
    await expect(page.getByRole('region', { name: 'Workflow context', exact: true })).toHaveCount(0);
    await expect(page.locator('.journey-step')).toHaveCount(0);
    await expect(page.getByText('Project activity', { exact: true })).toHaveCount(0);
    await expect(page.getByRole('status', { name: 'Application API connection', exact: true })).toHaveCount(0);
    await expect(page.getByText('App connected', { exact: true })).toHaveCount(0);
  }
  await page.getByRole('link', { name: 'My models', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'My models', level: 1, exact: true })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Workflow context', exact: true })).toHaveCount(0);
  await navigation.getByRole('link', { name: 'Distill', exact: true }).click();
  await expect(page.getByText('Teach a smaller ACT policy', { exact: true })).toHaveCount(0);
  await expect(page.getByText('Train an ACT256 student to imitate your teacher’s action chunks, then reload the saved student in a fresh process.', { exact: true })).toHaveCount(0);
  await expect(page.getByRole('group', { name: 'Teacher', exact: true })).toHaveCount(0);
  await page.screenshot({ path: testInfo.outputPath('clean-distill.png'), fullPage: false });
});

test('the contextual guide covers every section and keeps API docs separate', async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await workspace(page);
  await page.getByRole('link', { name: 'Distill', exact: true }).click();
  await page.locator('.page-heading .page-guide').click();
  await expect(page).toHaveURL(/\/guide\/#distill$/);
  await expect(page.getByRole('heading', { name: 'Workspace guide', exact: true })).toBeVisible();
  const sections = page.locator('.guide-sections > section');
  await expect(sections).toHaveCount(13);
  for (const name of ['Dashboard', 'My models', 'Dataset', 'Augmentation', 'Teaching', 'Fine-tune', 'Distill', 'Quantize', 'Evaluate', 'Run', 'Decision lab', 'Cloud runs', 'Settings & diagnostics']) {
    await expect(sections.getByRole('heading', { name, exact: true })).toBeVisible();
  }
  await expect(page.locator('#distill')).toContainText('ACT256');
  await expect(page.getByRole('link', { name: 'Distillation setup', exact: true })).toHaveAttribute('href', 'https://github.com/OpenJensen/openjensen/blob/main/workers/policy_distillation/README.md');
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
  await page.getByRole('link', { name: 'Quantize', exact: true }).click();
  await page.getByRole('button',{name:'Start a new quantization',exact:true}).click();
  await expect(page.getByText('No saved models in this project yet')).toBeVisible();
  await page.getByRole('region', { name: 'Your quantization models', exact: true }).screenshot({ path: testInfo.outputPath('quantization-owned-models.png') });
  await expect(page.locator('.workflow-panel')).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Native ACT quantization', exact: true })).toHaveCount(0);

  await page.getByRole('link', { name: 'Run', exact: true }).click();
  const runners = page.getByRole('group', { name: 'Run mode', exact: true });
  for (const name of ['3D simulation', 'Replay observations', 'Check inference']) {
    await expect(runners.getByRole('button', { name, exact: true })).toHaveAttribute('aria-pressed', 'false');
  }
  await expect(page.locator('.workflow-panel')).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Native Isaac simulation', exact: true })).toHaveCount(0);

  await page.getByRole('link', { name: 'Fine-tune', exact: true }).click();
  await page.getByRole('button', { name: 'Start a new fine-tuning', exact: true }).click();
  await page.getByRole('navigation', { name: 'Training setup' }).getByRole('button', { name: 'Model', exact: true }).click();
  const models = page.getByRole('group', { name: 'Base model', exact: true });
  await expect(models.getByRole('checkbox', { checked: true })).toHaveCount(0);
  await page.getByRole('checkbox', { name: 'SmolVLA', exact: true }).locator('..').click();
  await expect(page.getByRole('checkbox', { name: 'SmolVLA', exact: true })).toBeChecked();
  expect(mutations).toEqual([]);
});


async function workflowFixture(page: Page) {
  const mutations = await workspace(page);
  const time = '2026-09-27T12:00:00Z';
  const inspection = (id: string, project_id = 'ux-review', snapshot = true) => ({
    id, project_id, kind: 'dataset.inspect', status: 'succeeded', created_at: time, updated_at: time,
    request: { source: 'local', path: 'generated-fixture' },
    result: { source: 'local', format: 'lerobot_v3', repo_id: null, revision: 'fixture', metadata_sha256: 'a'.repeat(64), fps: 30, robot_type: 'generated-so101', inspected_at: time, warnings: [], total_episodes: 6, total_frames: 24, inspection_scope: 'complete_snapshot', features: {}, snapshot: snapshot ? { id: `sha256:${'b'.repeat(64)}`, manifest_sha256: 'b'.repeat(64), lineage_validated: true, total_episodes: 6, file_count: 4 } : null },
  });
  const state = { jobs: [inspection('dataset-first'), { ...inspection('dataset-second'), created_at: '2026-09-27T13:00:00Z' }, inspection('foreign-dataset', 'other'), { id: 'run-job', project_id: 'ux-review', kind: 'policy.run', status: 'failed', request: { operation: 'policy.run', runtime_id: 'fixture' }, result: null, created_at: time, updated_at: time }, { id: 'training-job', project_id: 'ux-review', kind: 'policy.finetune', status: 'running', request: { operation: 'policy.finetune', runtime_id: 'fixture' }, result: null, created_at: time, updated_at: time }] as Record<string, any>[], failed: false, gate: null as Promise<void> | null };
  await page.route('**/api/v1/projects', route => route.fulfill({ json: [{ id: 'ux-review', name: 'Robot workspace', created_at: time }, { id: 'beta', name: 'Empty second project', created_at: time }] }));
  await page.route('**/api/v1/projects/*/jobs', async route => {
    if (state.gate) await state.gate;
    return state.failed ? route.fulfill({ status: 503, json: { detail: 'Generated unavailable history' } }) : route.fulfill({ json: state.jobs.filter(job => job.project_id === new URL(route.request().url()).pathname.split('/')[4]) });
  });
  await page.route('**/api/v1/datasets**',route=>route.fulfill({json:state.jobs.filter(job=>job.kind==='dataset.inspect').map(job=>({id:`inspection:${job.id}`,job_id:job.id,project_id:job.project_id,name:job.id,source:'local',status:'ready',created_at:job.created_at,profile:job.result}))}));
  await page.reload();
  await expect(page.getByLabel('Current project')).toHaveAttribute('data-project-id', 'ux-review');
  return { state, mutations };
}

test('compact mobile navigation leaves useful stage content visible and every tool keyboard reachable', async ({ page }, testInfo) => {
  const mutations = await workspace(page);
  const navigation = page.getByRole('navigation', { name: 'Policy lifecycle' });
  for (const width of [320, 390, 645]) {
    await page.setViewportSize({ width, height: 900 });
    await page.evaluate(() => window.scrollTo(0, 0));
    const sidebar = await page.getByRole('complementary', { name: 'Workspace navigation' }).boundingBox();
    expect(sidebar!.height).toBeLessThan(235);
    const intake = await page.getByRole('heading', { name: 'Import a dataset', exact: true }).boundingBox();
    expect(intake!.y).toBeLessThan(590);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
    await page.screenshot({ path: testInfo.outputPath(`workspace-empty-${width}.png`), fullPage: false });
  }
  for (const name of ['Teaching', 'Fine-tune', 'Distill', 'Quantize', 'Evaluate', 'Run', 'Decision lab', 'Cloud runs', 'Settings & diagnostics', 'Dataset']) {
    const button = navigation.getByRole('link', { name, exact: true });
    await button.focus();
    await page.keyboard.press('Enter');
    await expect(button).toHaveAttribute('aria-current', 'page');
    await expect(page.getByRole('heading', { name, exact: true, level: 1 })).toBeVisible();
  }
  expect(mutations).toEqual([]);
});

test('workflow pages keep their job history without the duplicate project activity panel', async ({ page }, testInfo) => {
  const { mutations } = await workflowFixture(page);
  await page.getByRole('link', { name: 'Fine-tune', exact: true }).click();
  await expect(page.getByText('Project activity', { exact: true })).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Fine-tuning jobs', exact: true }).locator('[data-job-id="training-job"]')).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('clean-fine-tune.png'), fullPage: true });
  await page.getByRole('link', { name: 'Quantize', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Quantization jobs',exact:true })).toBeVisible();
  await selectProject(page, 'beta');
  await page.getByRole('link', { name: 'Fine-tune', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Fine-tuning jobs', exact: true }).getByText('No fine-tuning jobs yet.')).toBeVisible();
  await expect(page.locator('[data-job-id="training-job"]')).toHaveCount(0);
  expect(mutations).toEqual([]);
});

test('the clean dataset page continues with the exact viewed inspection without submitting training', async ({ page }) => {
  const { mutations } = await workflowFixture(page);
  await page.getByRole('button', { name: 'Open dataset dataset-first',exact:true }).click();
  await page.locator('.inspection-provenance > summary').click();
  await page.getByLabel('History', { exact: true }).selectOption('dataset-first');
  const journey = page.getByRole('region', { name: 'Workflow context', exact: true });
  await expect(journey).toHaveCount(0);
  await expect(page.getByText('Project activity', { exact: true })).toHaveCount(0);
  await expect(page.getByText('Latest inspection', { exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Train on this dataset', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Fine-tune', exact: true, level: 1 })).toBeVisible();
  await expect(page.getByRole('group', { name: 'Inspected dataset', exact: true }).locator('input:checked')).toHaveValue('dataset-first');
  expect(mutations).toEqual([]);
});

test('missing history stays visible in the run list and cannot enable metadata-only training', async ({ page }) => {
  const { state, mutations } = await workflowFixture(page);
  state.failed = true;
  await page.reload();
  await page.getByRole('link', { name: 'Fine-tune', exact: true }).click();
  const history = page.getByRole('region', { name: 'Fine-tuning jobs', exact: true });
  await expect(history.getByRole('alert')).toContainText('Generated unavailable history');
  await expect(history.getByText('No fine-tuning jobs yet.')).toHaveCount(0);
  state.failed = false;
  state.jobs = state.jobs.filter(job => job.id === 'dataset-first');
  state.jobs[0].result.snapshot = null;
  await page.reload();
  await page.getByRole('link', { name: 'Dataset', exact: true }).click();
  await page.getByRole('button', { name: 'Open dataset dataset-first',exact:true }).click();
  await expect(page.getByRole('region', { name: 'Dataset inspection', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Train on this dataset', exact: true })).toHaveCount(0);
  expect(mutations).toEqual([]);
});

test('loading history does not claim the run list is empty', async ({ page }) => {
  const { state, mutations } = await workflowFixture(page);
  let release!: () => void;
  state.gate = new Promise<void>(resolve => { release = resolve; });
  await page.reload();
  await page.getByRole('link', { name: 'Fine-tune', exact: true }).click();
  const history = page.getByRole('region', { name: 'Fine-tuning jobs', exact: true });
  await expect(history.getByRole('status')).toHaveText('Loading jobs…');
  await expect(history.getByText('No fine-tuning jobs yet.')).toHaveCount(0);
  release(); state.gate = null;
  await expect(history.locator('[data-job-id="training-job"]')).toBeVisible();
  expect(mutations).toEqual([]);
});
