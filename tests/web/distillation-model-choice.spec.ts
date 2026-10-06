import { selectProject } from './project-controls';
import { expect, test, type Page } from '@playwright/test';

const time = '2026-09-27T12:00:00Z';
const worker = { id: 'act-student', label: 'Local ACT worker', execution: 'native', provider: 'local', device: 'cpu', enabled: true, native_distillation: true, native_distillation_only: true, training: false, simulation: false, run: false, engine_evaluation: false };
const teacher = { id: 'teacher', project_id: 'alpha', job_id: 'import', label: 'Saved ACT teacher', format: 'native_checkpoint', path: 'owned', manifest_sha256: 'a'.repeat(64), file_bytes: 4096, metadata: { architecture: 'act' } };
const inspectedDataset = { id: 'dataset', project_id: 'alpha', kind: 'dataset.inspect', status: 'succeeded', created_at: time, updated_at: time, request: { source: 'local', path: 'generated' }, result: { source: 'local', format: 'lerobot_v3', repo_id: null, revision: `metadata-sha256:${'c'.repeat(64)}`, metadata_sha256: 'c'.repeat(64), fps: 30, robot_type: 'generated', inspected_at: time, warnings: [], total_episodes: 6, total_frames: 24, inspection_scope: 'complete_snapshot', features: {}, snapshot: { id: `sha256:${'b'.repeat(64)}`, manifest_sha256: 'b'.repeat(64), lineage_validated: true, total_episodes: 6 } } };
function savedJob(id: string, project = 'alpha', adapter = 'act-act-v1') {
  return { id, project_id: project, kind: 'policy.distill', status: 'failed', stage: 'completed', created_at: time, updated_at: time, error: 'Generated failed job', result: null,
    request: { operation: 'policy.distill', runtime_id: worker.id, artifact_id: teacher.id, dataset_job_id: inspectedDataset.id, timeout_seconds: 600, native_distillation: { adapter, student: 'act-256', steps: 100, learning_rate: .0001, seed: 1729, frame_stride: 30, splits: { train: [0, 1], validation: [2, 3], final: [4, 5] }, coordinate_attestation: 'generated_fixture', units: ['degrees', 'degrees', 'degrees', 'degrees', 'degrees', 'recorded_gripper'] } } };
}
const overview = (page: Page) => page.getByRole('region', { name: 'Distillation models', exact: true });
const act = (page: Page) => page.getByRole('button', { name: 'Choose Saved ACT teacher · teacher', exact: true });
const panel = (page: Page) => page.getByRole('region', { name: 'ACT distillation', exact: true });
const openDistill = (page: Page) => page.getByRole('navigation', { name: 'Policy lifecycle' }).getByRole('button', { name: 'Distill', exact: true }).click();

async function fixture(page: Page, initialJobs: Record<string, any>[] = [inspectedDataset], workers: Record<string, unknown>[] = [worker]) {
  const state = { jobs: initialJobs, workers, mutations: [] as string[] };
  await page.route('**/api/v1/**', async route => {
    const request = route.request(), path = new URL(request.url()).pathname;
    if (request.method() !== 'GET') { state.mutations.push(path); return route.fulfill({ status: 405, json: { detail: 'Model navigation must not submit jobs.' } }); }
    if (path === '/api/v1/projects') return route.fulfill({ json: [{ id: 'alpha', name: 'First project', created_at: time }, { id: 'beta', name: 'Second project', created_at: time }] });
    if (path === '/api/v1/policy-options') return route.fulfill({ json: { runtimes: state.workers, sources: [], training_models: [], training_methods: [], default_training_method: 'lora', quantization_defaults: { cpu: { language: 'Q8_0', vision: null }, cuda: { language: 'Q8_0', vision: null }, note: '' } } });
    // Deliberately return foreign records so the overview must filter ownership.
    if (path.endsWith('/jobs') && path.includes('/projects/')) return route.fulfill({ json: state.jobs });
    if (path.endsWith('/artifacts')) return route.fulfill({ json: [teacher] });
    if (path.endsWith('/events')) return route.fulfill({ json: [] });
    return route.continue();
  });
  await page.goto('/');
  await expect(page.getByLabel('Current project')).toHaveAttribute('data-project-id', 'alpha');
  await openDistill(page);
  return state;
}

test('Distill has no default model even with an ACT worker and dataset, and model entry is keyboard accessible', async ({ page }, testInfo) => {
  const state = await fixture(page);
  await expect(overview(page)).toBeVisible();
  await expect(overview(page)).toContainText('Saved ACT teacher');
  await expect(overview(page)).not.toContainText('Other model families are not supported yet.');
  await expect(act(page)).toBeEnabled();
  await expect(panel(page)).toHaveCount(0);
  await expect(page.getByRole('group', { name: 'Teacher', exact: true })).toHaveCount(0);
  await page.screenshot({ path: testInfo.outputPath('distillation-model-overview.png'), fullPage: false });
  await act(page).focus(); await page.keyboard.press('Enter');
  await expect(panel(page)).toBeVisible();
  await expect(page.getByRole('group', { name: 'Teacher', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Choose another teacher', exact: true }).click();
  await expect(overview(page)).toBeVisible();
  await expect(act(page)).toBeEnabled();
  await expect(panel(page)).toHaveCount(0);
  state.jobs.push(savedJob('arrived-later'));
  await expect(page.getByRole('button', { name: 'Open distillation arrived-later', exact: true })).toBeVisible({ timeout: 8000 });
  await expect(panel(page)).toHaveCount(0);
  await page.setViewportSize({ width: 320, height: 900 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
  expect(state.mutations).toEqual([]);
});

test('explicit model choice stays with its project and survives stage navigation but not reload', async ({ page }) => {
  const state = await fixture(page);
  await act(page).click(); await expect(panel(page)).toBeVisible();
  await page.getByRole('button', { name: 'Dataset', exact: true }).click(); await openDistill(page);
  await expect(panel(page)).toBeVisible();
  await selectProject(page, 'beta');
  await expect(overview(page)).toBeVisible(); await expect(panel(page)).toHaveCount(0);
  await expect(act(page)).toHaveCount(0);
  await expect(page.getByText('No saved models in this project yet')).toBeVisible();
  await selectProject(page, 'alpha');
  await expect(panel(page)).toBeVisible();
  await page.reload(); await openDistill(page);
  await expect(overview(page)).toBeVisible(); await expect(panel(page)).toHaveCount(0);
  expect(state.mutations).toEqual([]);
});

test('supported owned history opens the exact saved job without requiring a worker or choosing a default', async ({ page }) => {
  const owned = savedJob('owned-act'), foreign = savedJob('foreign-act', 'beta'), unsupported = savedJob('other-adapter', 'alpha', 'future-adapter');
  const state = await fixture(page, [owned, foreign, unsupported], []);
  await expect(overview(page)).toBeVisible(); await expect(panel(page)).toHaveCount(0);
  await expect(act(page)).toBeEnabled();
  await expect(page.getByRole('button', { name: `Open distillation ${owned.id}`, exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: `Open distillation ${foreign.id}`, exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: `Open distillation ${unsupported.id}`, exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: `Open distillation ${owned.id}`, exact: true }).click();
  await expect(page.getByRole('article', { name: 'Distillation job details', exact: true })).toHaveAttribute('data-job-id', owned.id);
  await expect(page.getByLabel('Saved distillation job', { exact: true })).toHaveValue(owned.id);
  await page.getByText('Prepare another student', { exact: true }).click();
  await expect(page.getByText('No local ACT distillation worker is configured.', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Train ACT256 student', exact: true })).toBeDisabled();
  expect(state.mutations).toEqual([]);
});

for (const status of ['pending', 'uncertain'] as const) test(`${status} request recovery remains visible on the unselected overview and requires fresh review`, async ({ page }) => {
  const state = await fixture(page);
  const key = 'firebird:job-attempt:policy.distill:alpha';
  const journal = JSON.stringify({ state: status, message: 'Generated unresolved distillation request' });
  await page.evaluate(({ key, journal }) => sessionStorage.setItem(key, journal), { key, journal });
  await page.reload(); await openDistill(page);
  await expect(overview(page)).toBeVisible(); await expect(panel(page)).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Review request', exact: true })).toBeVisible();
  expect(await page.evaluate(key => sessionStorage.getItem(key), key)).toBe(journal);
  await page.getByRole('button', { name: 'Review request', exact: true }).click();
  const acknowledgment = page.getByRole('button', { name: 'I checked recorded jobs; allow a new request', exact: true });
  await expect(acknowledgment).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Train ACT256 student', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Refresh distillation jobs', exact: true }).click();
  await expect(acknowledgment).toBeEnabled();
  expect(await page.evaluate(key => sessionStorage.getItem(key), key)).toBe(journal);
  await page.getByRole('button', { name: 'Choose another teacher', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Review request', exact: true })).toBeVisible();
  await selectProject(page, 'beta');
  await expect(overview(page)).toBeVisible();
  await expect(page.getByRole('button', { name: 'Review request', exact: true })).toHaveCount(0);
  expect(state.mutations).toEqual([]);
});

test('unreadable recovery is surfaced before model choice and still blocks submission after opening ACT', async ({ page }) => {
  await page.addInitScript(() => {
    const original = Storage.prototype.getItem;
    Storage.prototype.getItem = function(key: string) {
      if (key === 'firebird:job-attempt:policy.distill:alpha') throw new DOMException('Generated journal access failure', 'SecurityError');
      return original.call(this, key);
    };
  });
  const state = await fixture(page);
  await expect(overview(page).getByRole('alert')).toContainText('Saved request status could not be read.');
  await expect(panel(page)).toHaveCount(0);
  await page.getByRole('button', { name: 'Review request', exact: true }).click();
  await expect(panel(page).getByRole('alert')).toContainText('Browser session storage is unavailable.');
  await expect(page.getByRole('button', { name: 'Train ACT256 student', exact: true })).toBeDisabled();
  expect(state.mutations).toEqual([]);
});
