import { openCreateProject } from './project-controls';
import { execFile } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { promisify } from 'node:util';
import { expect, test, type Page } from '@playwright/test';
import { waitForJob } from './job-waiter';

const execute = promisify(execFile);
const python = process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python';
// Cancellation and failure/retry each need two independently bounded worker waits.
// This only extends the real-workflow test budget, never an application timeout.
test.describe.configure({ timeout: 120_000 });

async function cli(...args: string[]) {
  const { stdout } = await execute(python, ['-m', 'vla_platform.cli', ...args], {
    env: { ...process.env, FIREBIRD_API_URL: `http://127.0.0.1:${process.env.FIREBIRD_BROWSER_PORT ?? '8765'}` },
    timeout: 15_000,
  });
  return JSON.parse(stdout);
}

class WorkflowPage {
  projectId = '';
  modelId = '';
  constructor(readonly page: Page) {}

  async createProject(name: string) {
    await this.page.goto('/datasets/');
    await openCreateProject(this.page);
    await this.page.getByLabel('Project name', { exact: true }).fill(name);
    const created = this.page.waitForResponse(response => response.url().endsWith('/api/v1/projects') && response.request().method() === 'POST');
    await this.page.getByRole('button', { name: 'Create project', exact: true }).click();
    const response = await created;
    expect(response.status()).toBe(201);
    const project = await response.json();
    this.projectId = project.id;
    await expect(this.page.getByLabel('Current project')).toHaveAttribute('data-project-id', project.id);
    return project;
  }

  async quantize(runtime: string) {
    if (!this.modelId) {
      const imported = await this.page.request.post(`/api/v1/projects/${this.projectId}/policy-jobs`, { data: { operation: 'policy.import', runtime_id: 'browser-success', source_id: 'synthetic-source' } });
      expect(imported.status()).toBe(202);
      const receipt = await imported.json();
      const completed = await waitForJob(this.page.request, receipt.id, 'succeeded');
      this.modelId = completed.result.artifacts.find((item: { format: string }) => item.format === 'gguf').id;
    }
    await this.page.getByRole('link', { name: 'Quantize', exact: true }).click();
    await this.page.getByRole('button', { name: `Choose float · ${this.modelId}`, exact: true }).click();
    await this.page.getByRole('group', { name: 'Compute', exact: true }).locator(`input[value="${runtime}"]`).check();
    await this.page.getByRole('group', { name: 'My model', exact: true }).locator(`input[value="${this.modelId}"]`).check();
    const created = this.page.waitForResponse(response => response.url().endsWith('/policy-jobs') && response.request().method() === 'POST');
    await this.page.getByRole('button', { name: 'Run quantization workflow', exact: true }).click();
    const response = await created;
    expect(response.status()).toBe(202);
    return response.json();
  }
}

test('browser intake reads local metadata and preserves its limits after reload', async ({ page, request }, testInfo) => {
  const workflow = new WorkflowPage(page);
  const project = await workflow.createProject(`Synthetic intake ${testInfo.testId}`);
  await page.getByRole('radio', { name: 'Local files' }).check();
  await page.getByText('Folder on the app host', {exact:true}).click();
  await page.getByLabel('Dataset directory', { exact: true }).fill('lerobot_v3_preview');
  const created = page.waitForResponse(response => response.url().endsWith('/intakes') && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Inspect dataset', exact: true }).click();
  const response = await created;
  expect(response.status()).toBe(202);
  const history = await request.get(`/api/v1/projects/${project.id}/jobs`);
  expect(history.status()).toBe(200);
  const receipts = await history.json();
  expect(receipts).toHaveLength(1);
  const submitted = receipts[0];
  const job = await waitForJob(request, submitted.id, 'succeeded');
  await expect(page.getByRole('heading', { name: 'Local dataset', exact: true })).toBeVisible();
  expect(job.result.inspection_scope).toBe('metadata_only');
  expect(job.result.robot_type).toBe('synthetic_fixture');
  expect(await cli('jobs', 'show', submitted.id)).toEqual(job);
  await page.reload();
  await expect(page.getByLabel('Current project')).toHaveAttribute('data-project-id', project.id);
  await page.getByRole('button', { name: 'Open dataset Local dataset', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Local dataset', exact: true })).toBeVisible();
  await page.getByText('Advanced', { exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Inspection notes', exact: true })).toBeVisible();
  await expect(page.getByText('Metadata-only inspection: episode counts and schemas are source-declared, not validated against frames.', { exact: true })).toBeVisible();
  await expect(page.getByText(job.result.revision, { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Load visual preview', exact: true })).toHaveCount(0);
});

test('browser workflow persists real subprocess results and downloads the same artifacts seen by CLI', async ({ page, request }, testInfo) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  const workflow = new WorkflowPage(page);
  const project = await workflow.createProject(`Synthetic workflow ${testInfo.testId}`);
  const submitted = await workflow.quantize('browser-success');
  const job = await waitForJob(request, submitted.id, 'succeeded');
  await expect(page.getByText('Diagnostics complete.', { exact: true })).toBeVisible();
  expect(job.result.decision).toBe('diagnostics_only');
  expect(job.result.selected_artifact_id).toBeNull();
  expect(job.result.artifacts.length).toBeGreaterThanOrEqual(1);
  expect(job.request.artifact_id).toBe(workflow.modelId);
  expect(job.result.artifacts.every((artifact: { parent_ids: string[] }) => artifact.parent_ids.includes(workflow.modelId))).toBeTruthy();
  expect(job.result.artifacts.every((artifact: { metadata: { fixture_only: boolean } }) => artifact.metadata.fixture_only)).toBeTruthy();
  expect(await cli('jobs', 'show', submitted.id)).toEqual(job);
  const allArtifacts = await cli('policy', 'artifacts', project.id);
  expect(allArtifacts).toEqual(expect.arrayContaining(job.result.artifacts));
  expect(allArtifacts.some((artifact: { id: string }) => artifact.id === workflow.modelId)).toBeTruthy();
  const events = await cli('jobs', 'events', submitted.id);
  expect(events.some((event: { message: string }) => event.message === 'Optimizer step 1')).toBeTruthy();

  await page.reload();
  await expect(page.getByLabel('Current project')).toHaveAttribute('data-project-id', project.id);
  await page.getByRole('link', { name: 'Quantize', exact: true }).click();
  await page.locator(`.job-history-entry[data-job-id="${submitted.id}"]`).click();
  await expect(page.getByRole('article', { name: 'Quantize job details' })).toBeVisible();
  await expect(page.getByText('Diagnostics complete.', { exact: true })).toBeVisible();
  const artifact = job.result.artifacts.find((item: { parent_ids: string[] }) => item.parent_ids.length > 0);
  expect(artifact).toBeTruthy();
  const path = `/api/v1/projects/${project.id}/artifacts/${encodeURIComponent(artifact.id)}/download`;
  const downloadPromise = page.waitForEvent('download');
  await page.locator(`a[href="${path}"]`).click();
  const download = await downloadPromise;
  const downloaded = testInfo.outputPath('synthetic-policy.tar');
  await download.saveAs(downloaded);
  const apiDownload = await request.get(path);
  expect(await readFile(downloaded)).toEqual(await apiDownload.body());
  const { stdout } = await execute(python, ['-c', `import hashlib,json,sys,tarfile
with tarfile.open(sys.argv[1]) as archive:
 manifest_bytes=archive.extractfile('policy/manifest.json').read()
 manifest=json.loads(manifest_bytes)
 assert manifest['metadata']['fixture_only'] is True
 for name,sha in manifest['files'].items():
  assert hashlib.sha256(archive.extractfile('policy/'+name).read()).hexdigest()==sha
 print(hashlib.sha256(manifest_bytes).hexdigest())`, downloaded]);
  expect(stdout.trim()).toBe(artifact.manifest_sha256);
  expect(errors).toEqual([]);
});

test('cancelling a started worker remains cancelled after browser reload and in CLI', async ({ page, request }, testInfo) => {
  const workflow = new WorkflowPage(page);
  await workflow.createProject(`Synthetic cancellation ${testInfo.testId}`);
  const submitted = await workflow.quantize('browser-slow');
  // A recorded stdout event proves the registered worker process actually started.
  await waitForJob(request, submitted.id, 'worker_started');
  await page.getByRole('button', { name: 'Cancel run', exact: true }).click();
  const cancelled = await waitForJob(request, submitted.id, 'cancelled');
  expect(await cli('jobs', 'show', submitted.id)).toEqual(cancelled);
  await page.reload();
  await page.getByRole('link', { name: 'Quantize', exact: true }).click();
  await expect(page.locator(`.job-history-entry[data-job-id="${submitted.id}"]`)).toContainText('cancelled');
  await page.locator(`.job-history-entry[data-job-id="${submitted.id}"]`).click();
  await expect(page.getByRole('button', { name: 'Cancel run', exact: true })).toHaveCount(0);
  await expect(page.getByRole('link', { name: /^Download / })).toHaveCount(0);
  expect((await cli('jobs', 'show', submitted.id)).result).toBeNull();
});

test('a real worker failure is visible and does not prevent a subsequent successful run', async ({ page, request }, testInfo) => {
  const workflow = new WorkflowPage(page);
  await workflow.createProject(`Synthetic failure ${testInfo.testId}`);
  const failed = await workflow.quantize('browser-failure');
  const failure = await waitForJob(request, failed.id, 'failed');
  expect(await cli('jobs', 'show', failed.id)).toEqual(failure);
  // A known wrong terminal state must fail as such, not as a completion timeout.
  await expect(waitForJob(request, failed.id, 'succeeded')).rejects.toThrow('Job reached failed; expected succeeded');
  await page.getByText('Details and logs', { exact: true }).click();
  await expect(page.locator('.workflow-job-technical')).toContainText('Intentional synthetic browser worker failure');
  expect((await cli('jobs', 'show', failed.id)).status).toBe('failed');
  await page.reload();
  await page.getByRole('link', { name: 'Quantize', exact: true }).click();
  await page.locator(`.job-history-entry[data-job-id="${failed.id}"]`).click();
  await page.getByText('Details and logs', { exact: true }).click();
  await expect(page.locator('.workflow-job-technical')).toContainText('Intentional synthetic browser worker failure');
  const retryStarted = performance.now();
  const retried = await workflow.quantize('browser-delayed-success');
  const completed = await waitForJob(request, retried.id, 'succeeded');
  expect(await cli('jobs', 'show', retried.id)).toEqual(completed);
  await expect(page.getByText('Diagnostics complete.', { exact: true })).toBeVisible();
  expect(performance.now() - retryStarted).toBeGreaterThanOrEqual(7_000);
  expect((await cli('jobs', 'show', failed.id)).status).toBe('failed');
});
