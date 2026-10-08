import { selectProject } from './project-controls';
import { expect, test, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { gradientAccumulationAvailable, type TrainingModel } from '../../apps/web/src/lib/training-models';

const projectId = 'training-monitor-fixture';
const revision = 'a'.repeat(40);
const timestamp = () => new Date().toISOString();
type AccumulationCatalog = 'supported' | 'missing' | 'disabled' | 'wrong-runtime' | 'wrong-unit' | 'world-size';
function accumulationCapabilities(mode: AccumulationCatalog) {
  return mode === 'missing' ? {} : { gradient_accumulation_supported: mode !== 'disabled',
    gradient_accumulation_runtime_ids: mode === 'wrong-runtime' ? ['not-the-selected-runtime'] : ['skypilot-gcp-A100', 'all-models-a100'],
    training_step_unit: mode === 'wrong-unit' ? 'microbatches' : 'optimizer_updates', training_world_size: mode === 'world-size' ? 2 : 1 };
}

async function workspace(page: Page, mode: 'running' | 'preparing' | 'failed' | 'stale' | 'unavailable' | 'legacy' | 'validating' | 'history' | 'empty' | 'local-snapshot' | 'local-unprepared' = 'running', openDetails = true, exportMode?: 'local' | 'remote' | 'remote-complete' | 'unconfigured' | 'failed' | 'local-dataset', accumulationCatalog: AccumulationCatalog = 'supported') {
  const submitted: Record<string, any>[] = [];
  const unexpected: string[] = [];
  const cancelled: string[] = [];
  const telemetryRequests: string[] = [];
  const savedSubmissions = new Map<string, Record<string, any>>();
  const submissionLookups: string[] = [];
  const dataset = {
    id: 'dataset', project_id: projectId, kind: 'dataset.inspect', status: 'succeeded',
    request: { source: 'huggingface', repo_id: 'fixture/robot', revision: 'main' },
    created_at: timestamp(), updated_at: timestamp(),
    result: {
      source: 'huggingface', repo_id: 'fixture/robot', revision, format: 'lerobot_v3',
      inspection_scope: 'metadata_only', total_episodes: 10, total_frames: 1000, fps: 30,
      features: { 'observation.images.front': { dtype: 'video', shape: [480, 640, 3] }, 'observation.state': { dtype: 'float32', shape: [6] }, action: { dtype: 'float32', shape: [6] } },
      metadata_sha256: 'b'.repeat(64), inspected_at: timestamp(), warnings: [],
    },
  };
  const localDataset = mode === 'local-snapshot' || mode === 'local-unprepared';
  if (localDataset) {
    Object.assign(dataset, { request: { source: 'local', path: 'fixture/local', revision: 'main', snapshot_for_training: mode === 'local-snapshot' } });
    Object.assign(dataset.result, { source: 'local', repo_id: null, revision: `local:${'b'.repeat(64)}` });
    if (mode === 'local-snapshot') Object.assign(dataset.result, { inspection_scope: 'complete_snapshot', snapshot: {
      schema_version: 1, id: `sha256:${'d'.repeat(64)}`, manifest_sha256: 'd'.repeat(64), format: 'lerobot_v3',
      total_bytes: 1000, file_count: 5, total_episodes: 10, total_frames: 1000, lineage_validated: true, warnings: [],
    } });
  }
  const job: Record<string, any> = {
    id: 'run-001', project_id: projectId, kind: 'policy.finetune', status: mode === 'failed' ? 'failed' : 'running',
    stage: mode === 'preparing' ? 'preparing' : 'training', created_at: timestamp(), updated_at: timestamp(),
    request: { operation: 'policy.finetune', runtime_id: 'skypilot-gcp-A100', dataset_job_id: 'dataset', training_method: 'lora',
      training: { steps: 100, batch_size: 4, seed: 42, learning_rate: 0.0001, model_id: 'lerobot/smolvla_base', model_revision: revision, camera_keys: ['observation.images.front'] } },
    error: mode === 'failed' ? 'GPU ran out of memory at step 25.' : null,
  };
  const firstMetric = { step: 10, train_loss: 0.8, validation_loss: 0.9, learning_rate: 0.0001, grad_norm: 0.7, elapsed_seconds: 20, timestamp: timestamp() };
  const latest = { step: 25, train_loss: 0.4, validation_loss: null, learning_rate: 0.00009, grad_norm: 0.5, elapsed_seconds: 50,
    timestamp: mode === 'stale' ? new Date(Date.now() - 180000).toISOString() : timestamp() };
  const telemetry: Record<string, any> = {
    job_id: job.id, status: job.status, phase: mode === 'failed' ? 'failed' : job.stage, current_action: 'Optimizing adapters from robot demonstrations',
    updated_at: timestamp(), completed_steps: 25, total_steps: 100, percent: 25,
    elapsed_seconds: 50, wall_seconds: 80, eta_seconds: 150, latest, metrics: [firstMetric, latest], metrics_truncated: false,
    checkpoints: [{ step: 20, name: 'checkpoint-000020', timestamp: timestamp() }],
    events: [
      { sequence: 1, stage: 'preparing', message: 'Loading pinned dataset snapshot', timestamp: timestamp(), data: {} },
      { sequence: 2, stage: 'training', message: 'Optimizer step 25 completed', timestamp: timestamp(), data: {} },
      { sequence: 3, stage: 'checkpoint', message: 'Checkpoint 20 saved locally', timestamp: timestamp(), data: {} },
    ],
    logs: ['Trying another GCP availability zone after capacity was unavailable.', 'Training worker connected.'],
    reproducibility: { schema_version: 1, job_id: job.id, recipe: { ...job.request.training, dataset_id: 'fixture/robot', dataset_revision: revision },
      recipe_source: 'worker_resolved', model: { repository: 'lerobot/smolvla_base', revision }, dataset: dataset.result,
      runtime: { id: 'skypilot-gcp-A100', label: 'A100' }, lineage: {}, evidence: { splits: { train: [0, 1, 2, 3, 4, 5, 6, 7], validation: [8, 9] } },
      limitations: ['Validation loss is held-out imitation loss, not robot task success.'],
    },
  };
  if (mode === 'preparing') Object.assign(telemetry, {
    current_action: 'Preparing the cloud GPU', completed_steps: null, percent: null, elapsed_seconds: null, eta_seconds: null,
    latest: null, metrics: [], checkpoints: [], events: [],
  });
  if (mode === 'validating') {
    const phaseMetric = { step: 20, elapsed_seconds: 60, train_loss: null, validation_loss: null, learning_rate: null, timestamp: timestamp() };
    Object.assign(telemetry, { phase: 'validation', completed_steps: 20, percent: 20, elapsed_seconds: 60, latest: phaseMetric, metrics: [firstMetric, phaseMetric] });
  }
  const jobs = mode === 'empty' || localDataset ? [dataset] : [job, dataset];
  if (mode === 'history') jobs.push({ ...job, id: 'older-run', status: 'succeeded',
    created_at: new Date(Date.now() - 3600000).toISOString(),
    request: { ...job.request, training_method: 'full', training: { ...job.request.training, model_id: 'code://lerobot/act', steps: 40 } },
    result: { artifacts: [], reports: [{ steps: 40 }], decision: 'completed' },
  });
  const artifact: Record<string, any> = { id: 'checkpoint-artifact', project_id: projectId, job_id: job.id, format: 'training_checkpoint', label: 'Checkpoint step 20', file_bytes: 100, path: 'checkpoint', manifest_sha256: 'c'.repeat(64), parent_ids: [], metadata: {} };
  if (exportMode) artifact.metadata = { architecture: 'act', training_backend: 'lerobot', method: 'full',
    dataset: { source: exportMode === 'local-dataset' ? 'local' : 'huggingface' },
    ...(exportMode?.startsWith('remote') ? { storage: 'gcs', reload_verified: exportMode === 'remote-complete', step: 20 } : {}) };
  const artifacts: Record<string, any>[] = [artifact];
  const extraRuntimes: Record<string, any>[] = [];
  if (exportMode === 'remote-complete') {
    job.status = 'succeeded';
    telemetry.status = 'succeeded';
    artifacts.unshift({ ...artifact, id: 'periodic-same-step', metadata: { ...artifact.metadata, reload_verified: false } });
  }
  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() === 'POST' && path === `/api/v1/projects/${projectId}/policy-jobs`) {
      const body = request.postDataJSON();
      submitted.push(body);
      if (body.operation === 'policy.export') {
        if (exportMode === 'remote-complete') artifacts.push({ ...artifact, id: 'local-copy', job_id: 'export-job', parent_ids: [artifact.id], metadata: { ...artifact.metadata, storage: undefined } });
        const created = { ...job, id: 'export-job', kind: 'policy.export', request: body,
          status: exportMode === 'failed' ? 'failed' : 'succeeded', error: exportMode === 'failed' ? 'Synthetic CPU parity did not match.' : null,
          stage: 'operation', created_at: timestamp() };
        jobs.unshift(created);
        if (exportMode !== 'failed') artifacts.push({ ...artifact, id: 'inference-export', job_id: created.id,
          format: 'inference_export', label: 'ACT inference export', parent_ids: exportMode === 'remote-complete' ? ['local-copy'] : [artifact.id], metadata: { inference_only: true } });
        await route.fulfill({ status: 202, json: created });
        return;
      }
      const created = { ...job, id: `new-run-${submitted.length}`, request: body, status: 'queued', stage: 'preparing', error: null, created_at: timestamp() };
      jobs.unshift(created);
      const key = request.headers()['idempotency-key'];
      if (key) savedSubmissions.set(key, created);
      await route.fulfill({ status: 202, json: created, headers: key ? { 'Idempotency-Key': key, 'Cache-Control': 'no-store' } : {} });
      return;
    }
    if (request.method() === 'POST' && path === `/api/v1/jobs/${job.id}/cancel`) {
      cancelled.push(job.id);
      job.status = 'cancelled';
      telemetry.status = 'cancelled';
      telemetry.phase = 'cancelled';
      telemetry.eta_seconds = null;
      await route.fulfill({ json: job });
      return;
    }
    if (request.method() === 'GET') {
      if (path.startsWith(`/api/v1/projects/${projectId}/submissions/`)) {
        const key = decodeURIComponent(path.split('/').at(-1)!);
        submissionLookups.push(key);
        const saved = savedSubmissions.get(key);
        await route.fulfill({ status: saved ? 200 : 404, json: saved ?? { detail: 'No matching saved request.' }, headers: { 'Idempotency-Key': key, 'Cache-Control': 'no-store' } });
        return;
      }
      const replies: Record<string, unknown> = {
        '/api/v1/health': { status: 'ok', version: 'training-fixture' }, '/api/v1/capabilities': [],
        '/api/v1/simulation-options': { profiles: [] },
        '/api/v1/projects': [{ id: projectId, name: 'Training visibility', created_at: timestamp() }],
        [`/api/v1/projects/${projectId}/jobs`]: jobs,
        [`/api/v1/projects/${projectId}/artifacts`]: artifacts,
        '/api/v1/jobs/dataset/episodes': { repo_id: 'fixture/robot', revision, episodes: [], total_episodes: 10, offset: 0, limit: 6, warnings: [] },
        '/api/v1/datasets': [],
        '/api/v1/policy-options': {
          runtimes: [...extraRuntimes, ...(exportMode && exportMode !== 'unconfigured' ? [{ id: 'act-cpu', label: 'Local CPU export', provider: 'local', execution: 'native', device: 'cpu', enabled: true, training: false, act_export: true, export_only: true, engine_evaluation: false, run: false, simulation: false }] : []), { id: 'skypilot-gcp-A100', label: 'A100', accelerator: 'A100', execution: 'skypilot', provider: 'gcp', device: 'cuda', enabled: true, training: true, simulation: false, training_model_ids: localDataset ? ['smolvla', 'act'] : ['smolvla'] }],
          compute: { local: { enabled: false, label: 'Local' }, gcp: { enabled: true, default_gpu: 'A100', disk_size_gb: 200, idle_minutes: 10 } },
          training_models: [
            { id: 'smolvla', label: 'SmolVLA', description: 'Compact policy', model_id: 'lerobot/smolvla_base', model_revision: revision, methods: ['lora', 'qlora'], suggested_gpu_memory_gb: 16, ...accumulationCapabilities(accumulationCatalog) },
            { id: 'pi05', label: 'π₀.₅', description: 'Flow policy', model_id: 'lerobot/pi05_base', model_revision: revision, methods: ['full'], minimum_gpu_memory_gb: 40, runtime_ids: ['all-models-a100'] },
            ...(localDataset ? [{ id: 'act', label: 'ACT', description: 'Native policy', model_id: 'code://lerobot/act', model_revision: revision, methods: ['full'], backend: 'lerobot', minimum_gpu_memory_gb: 16, ...accumulationCapabilities(accumulationCatalog) }] : []),
          ], sources: [], training_methods: [{ id: 'lora', label: 'LoRA', description: 'Train adapters.' }, { id: 'qlora', label: 'QLoRA', description: 'Train quantized adapters.' }, { id: 'full', label: 'Full training', description: 'Train policy.' }], default_training_method: 'lora',
          quantization_defaults: { cuda: { language: 'Q8_0', vision: null }, cpu: { language: 'Q8_0', vision: null }, note: '' },
        },
      };
      if (path in replies) { await route.fulfill({ json: replies[path] }); return; }
      if (path.endsWith('/events')) { await route.fulfill({ json: telemetry.events }); return; }
      if (/\/jobs\/[^/]+\/training$/.test(path)) {
        telemetryRequests.push(path);
        if (mode === 'legacy') { await route.fulfill({ status: 404, json: { detail: 'Not Found' } }); return; }
        await route.fulfill(mode === 'unavailable' ? { status: 503, json: { detail: 'Training telemetry temporarily unavailable.' } } : { json: telemetry });
        return;
      }
      if (path.endsWith('/training/reproducibility')) {
        await route.fulfill({ contentType: 'application/json', headers: { 'content-disposition': 'attachment; filename="reproducibility.json"' }, body: JSON.stringify(telemetry.reproducibility) });
        return;
      }
    }
    unexpected.push(`${request.method()} ${path}`);
    await route.fulfill({ status: 405, json: { detail: 'Blocked by training monitor fixture.' } });
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Fine-tune', exact: true }).click();
  const monitor = page.getByRole('article', { name: 'Training run monitor' });
  if (openDetails) {
    await page.locator('.job-history-entry[data-job-id="run-001"]').click();
    await expect(monitor).toBeVisible();
  }
  return { telemetry, job, jobs, artifacts, extraRuntimes, monitor, submitted, unexpected, cancelled, telemetryRequests, savedSubmissions, submissionLookups };
}

async function noOverflow(page: Page) {
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(await page.evaluate(() => document.documentElement.clientWidth) + 1);
}

test('Fine-tune opens newest jobs first and opens details only after choosing a job', async ({ page }, testInfo) => {
  const { monitor, telemetryRequests, submitted, unexpected } = await workspace(page, 'history', false);
  const history = page.getByRole('region', { name: 'Fine-tuning jobs', exact: true });
  await expect(history).toBeVisible();
  const entries = history.locator('.job-history-entry');
  await expect(entries).toHaveCount(2);
  await expect(entries.first()).toHaveAttribute('data-job-id', 'run-001');
  await expect(entries.first()).toContainText('SmolVLA · LORA');
  await expect(entries.first()).toContainText('fixture/robot');
  await expect(entries.first()).toContainText('Saved step 20 / 100');
  await expect(entries.first()).toContainText('running');
  await expect(entries.last()).toContainText('ACT · FULL');
  await expect(entries.last()).toContainText('40 steps completed');
  await expect(monitor).toHaveCount(0);
  await expect(page.getByRole('navigation', { name: 'Training setup' })).toHaveCount(0);
  expect(telemetryRequests).toEqual([]);
  await page.setViewportSize({ width: 320, height: 740 });
  await noOverflow(page);
  await page.screenshot({ path: testInfo.outputPath('fine-tuning-jobs-320.png'), fullPage: true });
  await entries.first().click();
  await expect(monitor).toBeVisible();
  await expect(monitor).toHaveAttribute('data-run-id', 'run-001');
  await page.getByRole('button', { name: 'Back to jobs', exact: true }).click();
  await expect(history).toBeVisible();
  await expect(monitor).toHaveCount(0);
  expect(submitted).toEqual([]);
  expect(unexpected).toEqual([]);
});

test('new fine-tuning is a separate view with bold catalog memory budgets and a preserved draft', async ({ page }) => {
  const { submitted, unexpected } = await workspace(page, 'history', false);
  await page.getByRole('button', { name: 'Start a new fine-tuning', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Fine-tuning jobs', exact: true })).toHaveCount(0);
  const setup = page.getByRole('navigation', { name: 'Training setup' });
  await expect(setup).toBeVisible();
  await setup.getByRole('button', { name: 'Model', exact: true }).click();
  const budget = (model: string) => page.getByRole('checkbox', { name: model, exact: true }).locator('..').locator('.training-model-memory strong');
  await expect(budget('SmolVLA')).toHaveText('16 GB+');
  await expect(budget('π₀.₅')).toHaveText('40 GB+');
  await expect(budget('OpenVLA')).toHaveText('GPU budget not verified');
  expect(await budget('SmolVLA').evaluate(element => Number(getComputedStyle(element).fontWeight))).toBeGreaterThanOrEqual(700);
  await chooseOnlyModel(page, 'SmolVLA');
  await setup.getByRole('button', { name: 'Compute', exact: true }).click();
  await page.locator('summary').filter({ hasText: /^Advanced settings/ }).click();
  await page.getByRole('spinbutton', { name: 'Batch size', exact: true }).fill('3');
  await page.getByRole('button', { name: 'Back to jobs', exact: true }).click();
  await page.getByRole('button', { name: 'Start a new fine-tuning', exact: true }).click();
  await setup.getByRole('button', { name: 'Compute', exact: true }).click();
  await page.locator('summary').filter({ hasText: /^Advanced settings/ }).click();
  await expect(page.getByRole('spinbutton', { name: 'Batch size', exact: true })).toHaveValue('3');
  await page.getByRole('button', { name: 'Fine-tune', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Fine-tuning jobs', exact: true })).toBeVisible();
  await expect(setup).toHaveCount(0);
  expect(submitted).toEqual([]);
  expect(unexpected).toEqual([]);
});

test('empty history offers an explicit new job and repeated dataset shortcuts open the selected dataset', async ({ page }) => {
  const { submitted, unexpected } = await workspace(page, 'empty', false);
  await expect(page.getByText('No fine-tuning jobs yet.', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Start a new fine-tuning', exact: true })).toBeEnabled();
  await expect(page.getByRole('navigation', { name: 'Training setup' })).toHaveCount(0);
  for (let attempt = 0; attempt < 2; attempt += 1) {
    await page.getByRole('button', { name: 'Dataset', exact: true }).click();
    await page.getByRole('button', { name: 'Explore dataset', exact: true }).click();
    await page.getByRole('button', { name: 'Train on this dataset', exact: true }).click();
    await expect(page.getByRole('navigation', { name: 'Training setup' })).toBeVisible();
    await expect(page.getByRole('checkbox', { name: 'fixture/robot', exact: true })).toBeChecked();
    await page.getByRole('button', { name: 'Back to jobs', exact: true }).click();
  }
  expect(submitted).toEqual([]);
  expect(unexpected).toEqual([]);
});

test('local training copies open the jobs-first wizard and retain native model admission', async ({ page }) => {
  const { submitted, unexpected } = await workspace(page, 'local-snapshot', false);
  await page.getByRole('button', { name: 'Dataset', exact: true }).click();
  await page.getByRole('button', { name: 'Explore dataset', exact: true }).click();
  await page.getByRole('button', { name: 'Train on this dataset', exact: true }).click();
  await expect(page.getByRole('checkbox', { name: 'Local dataset', exact: true })).toBeChecked();
  const setup = page.getByRole('navigation', { name: 'Training setup' });
  await setup.getByRole('button', { name: 'Model', exact: true }).click();
  await chooseOnlyModel(page, 'SmolVLA');
  await expect(page.getByText('Local snapshots currently support native LeRobot models, including ACT.')).toBeVisible();
  await chooseOnlyModel(page, 'ACT');
  await setup.getByRole('button', { name: 'Compute', exact: true }).click();
  await page.getByRole('button', { name: 'Start fine-tuning', exact: true }).click();
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0]).toMatchObject({ operation: 'policy.finetune', dataset_job_id: 'dataset', training_method: 'full', training: { model_id: 'code://lerobot/act' } });
  expect(submitted[0].training).not.toHaveProperty('dataset_id');
  await expect(page.getByRole('article', { name: 'Training run monitor' })).toHaveAttribute('data-run-id', 'new-run-1');
  expect(unexpected).toEqual([]);
});

test('metadata-only local inspections cannot enter the training creation flow', async ({ page }) => {
  const { submitted, unexpected } = await workspace(page, 'local-unprepared', false);
  await page.getByRole('button', { name: 'Dataset', exact: true }).click();
  await page.getByRole('button', { name: 'Explore dataset', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Train on this dataset', exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Fine-tune', exact: true }).click();
  await page.getByRole('button', { name: 'Start a new fine-tuning', exact: true }).click();
  await expect(page.getByRole('checkbox', { name: 'Local dataset', exact: true })).toBeDisabled();
  await expect(page.getByText('Prepare an immutable training copy when importing this local dataset.')).toBeVisible();
  expect(submitted).toEqual([]);
  expect(unexpected).toEqual([]);
});

test('shows observed progress, loss curves, saved activity, worker output and downloadable run evidence', async ({ page }, testInfo) => {
  const { telemetry, monitor, unexpected } = await workspace(page);
  await expect(monitor.getByRole('progressbar')).toHaveAttribute('value', '25');
  await expect(monitor.getByText('25 / 100 optimizer steps', { exact: true })).toBeVisible();
  await expect(monitor.getByRole('img', { name: /Training and validation loss/ })).toBeVisible();
  await expect(monitor.locator('.training-metrics > div').filter({ has: page.getByText('Validation loss', { exact: true }) })).toContainText('Step 10');
  await expect(monitor.getByText('2m 30s', { exact: true })).toBeVisible();
  await expect(monitor.getByText('Latest checkpoint: step 20', { exact: true })).toBeVisible();
  await expect(monitor.getByLabel('Persisted training activity')).toHaveCount(0);
  await monitor.locator('.training-details > summary').click();
  await monitor.getByLabel('Filter activity').fill('checkpoint');
  await expect(monitor.getByLabel('Persisted training activity').locator('li')).toHaveCount(1);
  await expect(monitor.getByLabel('Persisted training activity')).toContainText('Checkpoint 20 saved locally');
  await monitor.locator('.training-worker-logs > summary').click();
  await expect(monitor.getByLabel('Worker output')).toContainText('capacity was unavailable');
  await monitor.locator('summary').filter({ hasText: /^Reproducibility/ }).click();
  await expect(monitor.getByText('8 train · 2 validation episodes')).toBeVisible();
  await expect(monitor.getByRole('link', { name: 'Download reproducibility JSON' })).toHaveAttribute('href', /\/jobs\/run-001\/training\/reproducibility$/);
  await expect(monitor.getByRole('link', { name: 'Download Checkpoint step 20' })).toHaveAttribute('href', /\/artifacts\/checkpoint-artifact\/download$/);
  const downloadPromise = page.waitForEvent('download');
  await monitor.getByRole('link', { name: 'Download reproducibility JSON' }).click();
  const downloaded = await downloadPromise;
  const record = JSON.parse(await readFile((await downloaded.path())!, 'utf8'));
  expect(record.recipe.seed).toBe(42);
  expect(record.dataset.revision).toBe(revision);
  await monitor.getByRole('heading', { name: 'Training', exact: true }).click();
  telemetry.completed_steps = 40;
  await expect(monitor.getByRole('progressbar')).toHaveAttribute('value', '40');
  await noOverflow(page);
  const fullScreenshot = testInfo.outputPath(`training-monitor-${testInfo.project.name}.png`);
  await monitor.screenshot({ path: fullScreenshot });
  await testInfo.attach(`training-monitor-${testInfo.project.name}`, { path: fullScreenshot, contentType: 'image/png' });
  await page.setViewportSize({ width: 320, height: 740 });
  await noOverflow(page);
  const screenshot = testInfo.outputPath('training-monitor-320.png');
  await monitor.screenshot({ path: screenshot });
  await testInfo.attach('training-monitor-320', { path: screenshot, contentType: 'image/png' });
  expect(unexpected).toEqual([]);
});

test('preparation has unknown progress and invalid metric values never become invented numbers', async ({ page }) => {
  const { telemetry, monitor, unexpected } = await workspace(page, 'preparing');
  await expect(monitor.getByRole('heading', { name: 'Preparing your GPU' })).toBeVisible();
  await expect(monitor.getByRole('progressbar')).not.toHaveAttribute('value');
  await expect(monitor.getByText('Waiting for the first reported step')).toBeVisible();
  await expect(monitor.getByText('Loss curves appear when the worker reports its first metrics.')).toBeVisible();
  Object.assign(telemetry, { completed_steps: -1, percent: 'NaN', latest: { step: 1, train_loss: 'NaN', learning_rate: 'Infinity' }, metrics: [{ step: 1, train_loss: 'NaN', validation_loss: null, timestamp: null }], eta_seconds: 'Infinity' });
  await expect(monitor.locator('.training-metrics').getByText('Awaiting a metric', { exact: true })).toBeVisible();
  await expect(monitor.getByRole('progressbar')).not.toHaveAttribute('value');
  await expect(monitor.locator('.training-metrics dd')).toHaveText(['—', '—', '—', '—', '—']);
  await expect(monitor).not.toContainText('NaN');
  await expect(monitor).not.toContainText('Infinity');
  expect(unexpected).toEqual([]);
});

test('stale metrics remain identified and suppress the remaining-time estimate', async ({ page }) => {
  const { monitor, unexpected } = await workspace(page, 'stale');
  await expect(monitor.getByText(/No new training metric for/)).toBeVisible();
  await expect(monitor.locator('.training-metrics > div').filter({ has: page.getByText('Estimated remaining', { exact: true }) })).toContainText('Awaiting a fresh metric');
  await expect(monitor).not.toContainText('2m 30s');
  await expect(monitor.getByRole('progressbar')).toHaveAttribute('value', '25');
  expect(unexpected).toEqual([]);
});

test('validation phase updates keep each last reported optimizer metric and its source step', async ({ page }) => {
  const { monitor, unexpected } = await workspace(page, 'validating');
  const metric = (name: string) => monitor.locator('.training-metrics > div').filter({ has: page.getByText(name, { exact: true }) });
  await expect(monitor.getByRole('heading', { name: 'Validating on held-out episodes' })).toBeVisible();
  await expect(monitor.getByRole('progressbar')).toHaveAttribute('value', '20');
  await expect(metric('Training loss')).toContainText('0.8');
  await expect(metric('Training loss')).toContainText('Step 10');
  await expect(metric('Validation loss')).toContainText('0.9');
  await expect(metric('Validation loss')).toContainText('Step 10');
  await expect(metric('Learning rate')).toContainText('0.0001');
  await expect(metric('Learning rate')).toContainText('Step 10');
  await expect(metric('Worker elapsed')).toContainText('1m 0s');
  expect(unexpected).toEqual([]);
});

test('failed runs retain evidence and resume the original recipe from their checkpoint', async ({ page }) => {
  const { monitor, submitted, unexpected } = await workspace(page, 'failed');
  await expect(monitor.getByRole('alert')).toHaveText('The GPU ran out of memory. Try a smaller batch size.');
  await expect(monitor.getByText('Training stopped', { exact: true })).toBeVisible();
  await expect(monitor.getByText('GPU ran out of memory at step 25.', { exact: true })).toHaveCount(0);
  await monitor.locator('.training-details > summary').click();
  await expect(monitor.getByText('GPU ran out of memory at step 25.', { exact: true })).toBeVisible();
  await expect(monitor.getByRole('button', { name: 'Cancel run' })).toHaveCount(0);
  await expect(monitor.getByRole('progressbar')).toHaveAttribute('value', '25');
  await monitor.getByRole('button', { name: 'Resume from checkpoint' }).click();
  await page.locator('summary').filter({ hasText: /^Advanced settings/ }).click();
  await expect(page.getByText('Original recipe preserved.')).toBeVisible();
  await page.getByRole('button', { name: 'Resume fine-tuning', exact: true }).click();
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0]).toMatchObject({ operation: 'policy.finetune', artifact_id: 'checkpoint-artifact', training: null, dataset_job_id: 'dataset' });
  expect(unexpected).toEqual([]);
});

test('can cancel an active run and preserves its last recorded progress', async ({ page }) => {
  const { monitor, cancelled, unexpected } = await workspace(page);
  await monitor.getByRole('button', { name: 'Cancel run' }).click();
  await expect.poll(() => cancelled).toEqual(['run-001']);
  await expect(monitor.getByRole('heading', { name: 'Cancelled', exact: true })).toBeVisible();
  await expect(monitor.getByRole('progressbar')).toHaveAttribute('value', '25');
  await expect(monitor.getByRole('button', { name: 'Resume from checkpoint' })).toBeVisible();
  expect(unexpected).toEqual([]);
});

test('reports telemetry failure without misrepresenting progress', async ({ page }) => {
  const { monitor, unexpected } = await workspace(page, 'unavailable');
  await expect(monitor.getByRole('alert')).toContainText('Training telemetry temporarily unavailable.', { timeout: 15000 });
  await expect(monitor.getByRole('button', { name: 'Retry telemetry' })).toBeVisible();
  await expect(monitor.getByRole('progressbar')).not.toHaveAttribute('value');
  expect(unexpected).toEqual([]);
});

test('submits explicit reproducible training settings supported by the worker', async ({ page }) => {
  const { submitted, unexpected } = await workspace(page);
  await page.getByRole('button', { name: 'Back to jobs', exact: true }).click();
  await page.getByRole('button', { name: 'Start a new fine-tuning', exact: true }).click();
  await page.getByRole('navigation', { name: 'Training setup' }).getByRole('button', { name: 'Model', exact: true }).click();
  const models = page.getByRole('group', { name: 'Base model', exact: true });
  if (!await models.locator('input:checked').count()) await chooseOnlyModel(page, 'SmolVLA');
  await page.getByRole('navigation', { name: 'Training setup' }).getByRole('button', { name: 'Compute', exact: true }).click();
  await page.locator('summary').filter({ hasText: /^Advanced settings/ }).click();
  await page.getByLabel('Steps', { exact: true }).fill('200');
  await page.getByLabel('Batch size', { exact: true }).fill('8');
  await page.getByLabel('Random seed', { exact: true }).fill('7');
  await page.getByLabel('Learning rate', { exact: true }).fill('0.0002');
  await page.getByLabel('Gradient accumulation', { exact: true }).fill('2');
  await page.getByLabel('Validation fraction', { exact: true }).fill('0.3');
  await page.getByLabel('Validate every (steps)', { exact: true }).fill('20');
  await page.getByRole('button', { name: 'Start fine-tuning', exact: true }).click();
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0].training).toMatchObject({ steps: 200, batch_size: 8, learning_rate: 0.0002, gradient_accumulation_steps: 2, seed: 7, validation_fraction: 0.3, eval_every: 20 });
  await expect(page.getByRole('article', { name: 'Training run monitor' })).toHaveAttribute('data-run-id', 'new-run-1');
  await expect(page.getByRole('navigation', { name: 'Training setup' })).toHaveCount(0);
  expect(unexpected).toEqual([]);
});


test('an older running application keeps observed progress and activity without retrying unsupported telemetry', async ({ page }) => {
  await page.clock.install();
  const { monitor, telemetryRequests, unexpected } = await workspace(page, 'legacy');
  await expect(monitor.getByText(/Detailed telemetry becomes available after the application is restarted once active runs finish./)).toBeVisible();
  await expect(monitor.getByRole('progressbar')).toHaveAttribute('value', '25');
  await expect(monitor.getByLabel('Persisted training activity')).toHaveCount(0);
  await expect(monitor.getByText('Latest checkpoint: step 20', { exact: true })).toBeVisible();
  await expect(monitor.locator('.training-metrics dd')).toHaveText(['—', '—', '—', '—', '—']);
  await expect(monitor.getByRole('alert')).toHaveCount(0);
  await expect(monitor.getByRole('link', { name: 'Download reproducibility JSON' })).toHaveCount(0);
  await page.clock.fastForward(10000);
  await expect(monitor.getByRole('progressbar')).toHaveAttribute('value', '25');
  expect(telemetryRequests).toHaveLength(1);
  expect(unexpected).toEqual([]);
});


test('diagnostics summarizes a stopped run and opens repeated events only on request', async ({ page }, testInfo) => {
  const { telemetry, job, unexpected } = await workspace(page, 'failed');
  job.error = 'operation: worker skypilot-gcp-A100-012345 exited on cluster internal-cluster-id';
  telemetry.events = Array.from({ length: 120 }, (_, index) => ({ sequence: index + 1, stage: 'operation', message: `Optimizer step ${index + 1} completed`, timestamp: timestamp() }));
  telemetry.events.push({ sequence: 121, stage: 'operation', message: 'Checkpoint 100 saved locally', timestamp: timestamp() });
  telemetry.events.push({ sequence: 122, stage: 'operation', message: 'worker cluster internal-cluster-id stopped', timestamp: timestamp() });
  await page.getByRole('button', { name: 'Diagnostics', exact: true }).click();
  const diagnostics = page.locator('.workflow-panel');
  await expect(diagnostics.getByRole('heading', { name: 'Run diagnostics' })).toBeVisible();
  await expect(diagnostics.locator('.workflow-run-summary')).toContainText('Training stopped');
  await expect(diagnostics.locator('.workflow-run-summary')).toContainText('Last reported step: 120');
  await expect(diagnostics.locator('.workflow-run-summary')).toContainText('Latest checkpoint: step 100');
  await expect(diagnostics.getByLabel('Detailed run activity')).toHaveCount(0);
  await expect(diagnostics.getByText('worker cluster internal-cluster-id stopped', { exact: true })).toHaveCount(0);
  await noOverflow(page);
  const screenshot = testInfo.outputPath(`training-diagnostics-compact-${testInfo.project.name}.png`);
  await diagnostics.screenshot({ path: screenshot });
  await testInfo.attach('compact-training-diagnostics', { path: screenshot, contentType: 'image/png' });
  await diagnostics.locator('.workflow-activity-details > summary').click();
  const activity = diagnostics.getByLabel('Detailed run activity');
  await expect(activity.locator('li')).toHaveCount(122);
  await expect(activity).toContainText('Optimizer step 120 completed');
  await expect(activity).toContainText('worker cluster internal-cluster-id stopped');
  await diagnostics.locator('.workflow-activity-details > summary').click();
  await expect(activity).toHaveCount(0);
  expect(unexpected).toEqual([]);
});


test('ACT checkpoint exports inference-only package through local registered runtime', async ({ page }) => {
  const { monitor, submitted, unexpected } = await workspace(page, 'running', true, 'local');
  await expect(monitor.getByRole('button', { name: 'Quantize checkpoint' })).toHaveCount(0);
  await expect(monitor.getByText('This does not establish robot task success', { exact: false })).toBeVisible();
  await monitor.getByRole('button', { name: 'Export ACT inference package' }).click();
  await expect(monitor.getByRole('link', { name: 'Download ACT inference package' })).toHaveAttribute('href', /artifacts\/inference-export\/download$/);
  expect(submitted).toEqual([{ operation: 'policy.export', runtime_id: 'act-cpu', artifact_id: 'checkpoint-artifact', training_method: 'full', timeout_seconds: 600 }]);
  expect(unexpected).toEqual([]);
  await noOverflow(page);
});

for (const mode of ['remote', 'unconfigured'] as const) test(`ACT export refuses ${mode} checkpoint without a job`, async ({ page }) => {
  const { monitor, submitted } = await workspace(page, 'running', true, mode);
  await expect(monitor.getByRole('button', { name: mode === 'remote' ? 'Download checkpoint and export' : 'Export ACT inference package' })).toBeDisabled();
  await expect(monitor.getByText(mode === 'remote' ? 'Wait for the completed, reload-verified ACT checkpoint.' : 'Ask the app operator to configure the local ACT export worker.', { exact: false })).toBeVisible();
  expect(submitted).toEqual([]);
});

test('ACT export failure stays visible without a download or quality claim', async ({ page }) => {
  const { monitor, submitted } = await workspace(page, 'running', true, 'failed');
  await monitor.getByRole('button', { name: 'Export ACT inference package' }).click();
  await expect(monitor.getByRole('alert').filter({ hasText: 'Synthetic CPU parity did not match.' })).toBeVisible();
  await expect(monitor.getByRole('link', { name: 'Download ACT inference package' })).toHaveCount(0);
  expect(submitted).toHaveLength(1);
});


test('ACT export refuses local lineage without a verified complete snapshot', async ({ page }) => {
  const { monitor, submitted } = await workspace(page, 'running', true, 'local-dataset');
  await expect(monitor.getByRole('button', { name: 'Export ACT inference package' })).toBeDisabled();
  await expect(monitor.getByText('Export needs Hugging Face lineage or a complete, reload-verified local LeRobot v3 snapshot with matching snapshot identity.')).toBeVisible();
  expect(submitted).toEqual([]);
});


function localExportMetadata(): Record<string, any> {
  return { architecture: 'act', training_backend: 'lerobot', method: 'full', reload_verified: true,
    dataset_snapshot_id: `sha256:${'d'.repeat(64)}`, dataset_manifest_sha256: 'd'.repeat(64),
    dataset: { source: 'local', format: 'lerobot_v3', inspection_scope: 'complete_snapshot', repo_id: null, revision: `local:${'b'.repeat(64)}`,
      snapshot: { schema_version: 1, id: `sha256:${'d'.repeat(64)}`, manifest_sha256: 'd'.repeat(64), format: 'lerobot_v3',
        total_bytes: 1000, file_count: 5, total_episodes: 10, total_frames: 1000, lineage_validated: true, warnings: [] } } };
}

test('completed local ACT snapshot exports its exact checkpoint without rewriting lineage', async ({ page }) => {
  const state = await workspace(page, 'running', true, 'local-dataset');
  const checkpoint = state.artifacts[0]; checkpoint.metadata = localExportMetadata();
  state.job.status = 'succeeded'; state.telemetry.status = 'succeeded';
  const original = structuredClone(checkpoint.metadata);
  await page.reload(); await reopenExport(page);
  await expect(state.monitor.getByRole('button', { name: 'Export ACT inference package', exact: true })).toBeEnabled();
  expect(state.submitted).toEqual([]);
  await state.monitor.getByRole('button', { name: 'Export ACT inference package', exact: true }).click();
  await expect(state.monitor.getByRole('link', { name: 'Download ACT inference package' })).toBeVisible();
  expect(state.submitted).toEqual([{ operation: 'policy.export', runtime_id: 'act-cpu', artifact_id: checkpoint.id, training_method: 'full', timeout_seconds: 600 }]);
  expect(checkpoint.metadata).toEqual(original);
  expect(checkpoint.metadata.dataset.source).toBe('local');
  expect(state.unexpected).toEqual([]);
});

for (const fault of ['snapshot-identity', 'manifest', 'incomplete', 'not-reloaded', 'format'] as const) test(`local ACT export refuses ${fault} snapshot metadata`, async ({ page }) => {
  const state = await workspace(page, 'running', true, 'local-dataset');
  const metadata = localExportMetadata(); state.artifacts[0].metadata = metadata;
  state.job.status = 'succeeded'; state.telemetry.status = 'succeeded';
  if (fault === 'snapshot-identity') metadata.dataset.snapshot.id = `sha256:${'e'.repeat(64)}`;
  if (fault === 'manifest') metadata.dataset.snapshot.manifest_sha256 = 'invalid';
  if (fault === 'incomplete') metadata.dataset.inspection_scope = 'metadata_only';
  if (fault === 'not-reloaded') metadata.reload_verified = false;
  if (fault === 'format') metadata.dataset.format = 'lerobot_v2';
  await page.reload(); await reopenExport(page);
  await expect(state.monitor.getByRole('button', { name: 'Export ACT inference package', exact: true })).toBeDisabled();
  await expect(state.monitor.getByText('Export needs Hugging Face lineage or a complete, reload-verified local LeRobot v3 snapshot with matching snapshot identity.')).toBeVisible();
  expect(state.submitted).toEqual([]);
});


test('completed cloud ACT checkpoint explicitly downloads and exports with separate ancestry', async ({ page }) => {
  const { monitor, submitted, unexpected } = await workspace(page, 'running', true, 'remote-complete');
  const checkpoint = monitor.getByRole('combobox', { name: 'Checkpoint', exact: true });
  await expect(checkpoint.locator('option')).toHaveCount(3);
  await expect(checkpoint.locator('option').first()).toContainText('Reload-verified bundle');
  await checkpoint.selectOption('periodic-same-step');
  await expect(monitor.getByRole('button', { name: 'Download checkpoint and export' })).toBeDisabled();
  await expect(monitor.getByText('Wait for the completed, reload-verified ACT checkpoint.', { exact: false })).toBeVisible();
  await checkpoint.selectOption('checkpoint-artifact');
  await expect(monitor.getByText('It does not start a cloud GPU.', { exact: false })).toBeVisible();
  await monitor.getByRole('button', { name: 'Download checkpoint and export' }).click();
  await expect(monitor.getByRole('link', { name: 'Download ACT inference package' })).toBeVisible();
  expect(submitted).toEqual([{ operation: 'policy.export', runtime_id: 'act-cpu', artifact_id: 'checkpoint-artifact', training_method: 'full', timeout_seconds: 600 }]);
  expect(unexpected).toEqual([]);
  await noOverflow(page);
});

test('ACT export-only computer never appears as an engine execution target', async ({ page }) => {
  const { submitted } = await workspace(page, 'running', true, 'local');
  for (const [stage, create] of [['Run', 'New run'], ['Evaluate', 'New evaluation'], ['Quantize', 'New quantization']]) {
    await page.getByRole('button', { name: stage, exact: true }).click();
    if (stage === 'Run') await page.getByRole('button', { name: 'Check inference', exact: true }).click();
    if (stage === 'Quantize') {
      await expect(page.getByRole('button', { name: 'Choose Checkpoint step 20 · checkpoint-artifact', exact: true })).toBeDisabled();
      await expect(page.getByRole('button', { name: 'ACT', exact: true })).toHaveCount(0);
      continue;
    }
    await page.getByRole('button', { name: create, exact: true }).click();
    const target = page.getByRole('group', { name: 'Compute', exact: true });
    await expect(target).not.toContainText('Local CPU export');
    if (stage !== 'Quantize') await expect(target.getByRole('radio')).toHaveCount(0);
  }
  expect(submitted).toEqual([]);
});

async function newTraining(page: Page) {
  await page.getByRole('button', { name: 'Start a new fine-tuning', exact: true }).click();
  await page.getByRole('navigation', { name: 'Training setup' }).getByRole('button', { name: 'Model', exact: true }).click();
  const models = page.getByRole('group', { name: 'Base model', exact: true });
  if (!await models.locator('input:checked').count()) await chooseOnlyModel(page, 'SmolVLA');
  await page.getByRole('navigation', { name: 'Training setup' }).getByRole('button', { name: 'Compute', exact: true }).click();
  return page.getByRole('button', { name: 'Start fine-tuning', exact: true });
}

for (const fault of ['lost', 'wrong-project', 'changed-budget', 'redirect'] as const) {
  test(`training submission recovery retains ${fault} uncertainty across navigation and reload`, async ({ page }, testInfo) => {
    const { jobs, savedSubmissions } = await workspace(page, 'empty', false);
    let posts = 0, redirected = 0;
    await page.route('**/api/v1/training-redirect-target', async route => {
      redirected += 1;
      await route.fulfill({ json: { detail: 'Must not follow a mutation redirect' } });
    });
    await page.route(`**/api/v1/projects/${projectId}/policy-jobs`, async route => {
      posts += 1;
      const body = route.request().postDataJSON();
      const key = route.request().headers()['idempotency-key'];
      expect(key).toMatch(/^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,127}$/);
      const accepted = { id: 'accepted-on-server', project_id: projectId, kind: 'policy.finetune', status: 'queued', request: body, created_at: timestamp(), updated_at: timestamp() };
      jobs.unshift(accepted); savedSubmissions.set(key, accepted);
      const headers = { 'Idempotency-Key': key, 'Cache-Control': 'no-store' };
      if (fault === 'lost') await route.abort();
      else if (fault === 'redirect') await route.fulfill({ status: 307, headers: { ...headers, location: '/api/v1/training-redirect-target' }, body: '' });
      else await route.fulfill({ status: 202, headers, json: fault === 'wrong-project' ? { ...accepted, project_id: 'another-project' } : { ...accepted, request: { ...body, training: { ...body.training, steps: body.training.steps + 1 } } } });
    });
    const start = await newTraining(page);
    await start.click();
    const recovery = page.getByRole('region', { name: 'Training submission recovery' });
    await expect(recovery).toContainText('unverified');
    await expect(start).toBeDisabled();
    await expect(recovery.getByRole('button', { name: 'I checked the jobs; allow a new request' })).toHaveCount(0);
    await page.getByRole('navigation', { name: 'Policy lifecycle' }).getByRole('button', { name: 'Dataset', exact: true }).click();
    await page.getByRole('button', { name: 'Fine-tune', exact: true }).click();
    await expect(recovery).toBeVisible();
    await page.reload();
    await page.getByRole('button', { name: 'Fine-tune', exact: true }).click();
    await expect(recovery).toBeVisible();
    await expect(recovery.getByRole('button', { name: 'Retry same request', exact: true })).toBeDisabled();
    if (fault === 'lost') {
      await noOverflow(page);
      await page.screenshot({ path: testInfo.outputPath('training-submission-recovery.png'), fullPage: true });
    }
    await recovery.getByRole('button', { name: 'Check saved request', exact: true }).click();
    await expect(page.getByRole('article', { name: 'Training run monitor' })).toHaveAttribute('data-run-id', 'accepted-on-server');
    await expect(recovery).toHaveCount(0);
    expect(posts).toBe(1);
    expect(redirected).toBe(0);
  });
}

test('training journal write failure sends no request', async ({ page }) => {
  await page.addInitScript(() => {
    const set = Storage.prototype.setItem;
    Storage.prototype.setItem = function (key, value) {
      if (key.startsWith('firebird:submission:v1:')) throw new Error('Storage unavailable');
      return set.call(this, key, value);
    };
  });
  const { submitted } = await workspace(page, 'empty', false);
  const start = await newTraining(page);
  await start.click();
  await expect(page.getByRole('region', { name: 'Training submission recovery' })).toContainText('storage is unavailable');
  await expect(start).toBeDisabled();
  expect(submitted).toEqual([]);
});

test('training acknowledgement survives journal cleanup failure', async ({ page }) => {
  await page.addInitScript(() => {
    const remove = Storage.prototype.removeItem;
    Storage.prototype.removeItem = function (key) {
      if (key.startsWith('firebird:submission:v1:')) throw new Error('Storage unavailable');
      return remove.call(this, key);
    };
  });
  const { submitted } = await workspace(page, 'empty', false);
  await (await newTraining(page)).click();
  await expect(page.getByRole('article', { name: 'Training run monitor' })).toHaveAttribute('data-run-id', 'new-run-1');
  await expect(page.getByRole('region', { name: 'Training submission recovery' })).toContainText('storage is unavailable');
  expect(submitted).toHaveLength(1);
});

for (const malformed of ['status', 'method', 'dataset', 'timestamp', 'recipe'] as const) {
  test(`training resume rejects malformed ${malformed} acknowledgement without losing recovery`, async ({ page }) => {
    const { monitor } = await workspace(page, 'failed');
    let posts = 0;
    await page.route(`**/api/v1/projects/${projectId}/policy-jobs`, async route => {
      posts += 1;
      const body = route.request().postDataJSON();
      expect(body.artifact_id || body.resume_job_id).toBeTruthy();
      const job: Record<string, any> = { id: 'resumed', project_id: projectId, kind: 'policy.finetune', status: 'queued', request: body, created_at: timestamp(), updated_at: timestamp() };
      if (malformed === 'status') job.status = ['queued'];
      if (malformed === 'method') delete job.request.training_method;
      if (malformed === 'dataset') job.request.dataset_job_id = null;
      if (malformed === 'timestamp') job.updated_at = 'invalid';
      if (malformed === 'recipe') job.request.training = ['invalid'];
      await route.fulfill({ status: 202, json: job, headers: { 'Idempotency-Key': route.request().headers()['idempotency-key'], 'Cache-Control': 'no-store' } });
    });
    await monitor.getByRole('button', { name: 'Resume from checkpoint' }).click();
    await page.getByRole('button', { name: 'Resume fine-tuning', exact: true }).click();
    await expect(page.getByRole('region', { name: 'Training submission recovery' })).toContainText('unverified');
    await expect(page.getByRole('button', { name: 'Resume fine-tuning', exact: true })).toBeDisabled();
    expect(posts).toBe(1);
  });
}

test('legacy training recovery requires fresh history and explicit acknowledgement without POST', async ({ page }) => {
  const { submitted } = await workspace(page, 'empty', false);
  await page.evaluate(project => sessionStorage.setItem(`firebird:job-attempt:policy.finetune:${project}`, JSON.stringify({ state: 'uncertain', message: 'Legacy request outcome is unverified.' })), projectId);
  await page.reload();
  await page.getByRole('button', { name: 'Fine-tune', exact: true }).click();
  const recovery = page.getByRole('region', { name: 'Training submission recovery' });
  await expect(recovery).toContainText('no server request key');
  const acknowledge = recovery.getByRole('button', { name: 'I checked the jobs; allow a new request' });
  await expect(acknowledge).toBeDisabled();
  await expect(recovery.getByRole('button', { name: 'Retry same request' })).toHaveCount(0);
  await page.route(`**/api/v1/projects/${projectId}/jobs`, async route => route.fulfill({ status: 503, json: { detail: 'History unavailable' } }));
  await recovery.getByRole('button', { name: 'Refresh training jobs' }).click();
  await expect(recovery).toContainText('Training history could not refresh');
  await expect(acknowledge).toBeDisabled();
  await page.unroute(`**/api/v1/projects/${projectId}/jobs`);
  await recovery.getByRole('button', { name: 'Refresh training jobs' }).click();
  await expect(acknowledge).toBeEnabled();
  await acknowledge.click();
  await expect(recovery).toHaveCount(0);
  expect(submitted).toEqual([]);
});

test('a definite training rejection reports its reason without inventing a saved job', async ({ page }) => {
  const { jobs } = await workspace(page, 'empty', false);
  let posts = 0;
  await page.route(`**/api/v1/projects/${projectId}/policy-jobs`, async route => {
    posts += 1;
    await route.fulfill({ status: 422, json: { detail: 'The selected dataset is not admitted for training.' }, headers: { 'Idempotency-Key': route.request().headers()['idempotency-key'], 'Cache-Control': 'no-store' } });
  });
  const start = await newTraining(page);
  await start.click();
  await expect(page.getByRole('main').getByRole('alert')).toContainText('The selected dataset is not admitted for training.');
  await expect(start).toBeEnabled();
  await expect(page.getByRole('region', { name: 'Training submission recovery' })).toHaveCount(0);
  expect(jobs.filter(job => job.kind === 'policy.finetune')).toEqual([]);
  expect(posts).toBe(1);
});


async function exportRecoveryFixture(page: Page) {
  const state = await workspace(page, 'running', true, 'remote-complete');
  const originalJobs = structuredClone(state.jobs);
  await page.route(`**/api/v1/projects/${projectId}/jobs`, route => route.fulfill({ json: originalJobs }));
  return { ...state, originalJobs };
}
const exportButton = (page: Page) => page.getByRole('button', { name: 'Download checkpoint and export', exact: true });
async function reopenExport(page: Page) {
  await page.getByRole('button', { name: 'Fine-tune', exact: true }).click();
  await page.locator('.job-history-entry[data-job-id="run-001"]').click();
  await expect(page.getByRole('article', { name: 'Training run monitor' })).toBeVisible();
}
function exportReceipt(state: Awaited<ReturnType<typeof exportRecoveryFixture>>, body: Record<string, unknown>) {
  return { ...state.job, id: 'acknowledged-export', kind: 'policy.export', request: body, status: 'queued', stage: 'preparing', error: null };
}

test('ACT export lost acknowledgment remains paused after reload without another POST', async ({ page }, testInfo) => {
  const state = await exportRecoveryFixture(page);
  await page.route(`**/api/v1/projects/${projectId}/policy-jobs`, async route => {
    state.submitted.push(route.request().postDataJSON()); await route.abort('failed');
  });
  await exportButton(page).click();
  await expect(page.getByRole('region', { name: 'ACT inference export' })).toContainText('outcome is unverified');
  await expect(exportButton(page)).toBeDisabled();
  await page.reload(); await reopenExport(page);
  await expect(exportButton(page)).toBeDisabled();
  await expect(page.getByRole('button', { name: 'I checked the export jobs; allow a new request' })).toBeDisabled();
  expect(state.submitted).toHaveLength(1);
  const image = testInfo.outputPath('export-recovery.png');
  await page.getByRole('region', { name: 'ACT inference export' }).screenshot({ path: image });
  await testInfo.attach('Export recovery', { path: image, contentType: 'image/png' });
});

test('ACT export retains acknowledged job while history is stale and prevents another submission', async ({ page }, testInfo) => {
  const state = await exportRecoveryFixture(page);
  await page.route(`**/api/v1/projects/${projectId}/policy-jobs`, async route => {
    const body = route.request().postDataJSON(); state.submitted.push(body);
    await route.fulfill({ status: 202, json: exportReceipt(state, body) });
  });
  await exportButton(page).click();
  await expect(page.getByRole('region', { name: 'ACT export receipt' })).toContainText('acknowledged-export');
  await expect(page.getByRole('region', { name: 'ACT export receipt' })).toContainText('queued');
  await expect(page.getByRole('button', { name: 'Exporting ACT policy…', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Dataset', exact: true }).click(); await reopenExport(page);
  await expect(page.getByRole('region', { name: 'ACT export receipt' })).toContainText('acknowledged-export');
  await expect(page.getByRole('button', { name: 'Exporting ACT policy…', exact: true })).toBeDisabled();
  expect(state.submitted).toHaveLength(1);
  const image = testInfo.outputPath('export-receipt.png');
  await page.getByRole('region', { name: 'ACT inference export' }).screenshot({ path: image });
  await testInfo.attach('Export receipt', { path: image, contentType: 'image/png' });
});

test('ACT export unmount during submission preserves pending identity and late acknowledgment', async ({ page }) => {
  const state = await exportRecoveryFixture(page); let release!: () => void;
  const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route(`**/api/v1/projects/${projectId}/policy-jobs`, async route => {
    const body = route.request().postDataJSON(); state.submitted.push(body); await gate;
    await route.fulfill({ status: 202, json: exportReceipt(state, body) });
  });
  try {
    await exportButton(page).click(); await expect.poll(() => state.submitted.length).toBe(1);
    await page.getByRole('button', { name: 'Dataset', exact: true }).click(); await reopenExport(page);
    await expect(page.getByRole('button', { name: 'Exporting ACT policy…', exact: true })).toBeDisabled();
    release();
    await expect(page.getByRole('region', { name: 'ACT export receipt' })).toContainText('acknowledged-export');
    expect(state.submitted).toHaveLength(1);
  } finally { release(); }
});

test('ACT export mismatched acknowledgment is uncertain and never promoted to a receipt', async ({ page }) => {
  const state = await exportRecoveryFixture(page);
  await page.route(`**/api/v1/projects/${projectId}/policy-jobs`, async route => {
    const body = route.request().postDataJSON(); state.submitted.push(body);
    await route.fulfill({ status: 202, json: exportReceipt(state, { ...body, artifact_id: 'another-checkpoint' }) });
  });
  await exportButton(page).click();
  await expect(page.getByRole('region', { name: 'ACT inference export' })).toContainText('outcome is unverified');
  await expect(page.getByRole('region', { name: 'ACT export receipt' })).toHaveCount(0);
  await expect(exportButton(page)).toBeDisabled(); expect(state.submitted).toHaveLength(1);
});

for (const field of ['project', 'runtime', 'operation', 'budget', 'status', 'timestamp', 'empty'] as const) {
  test(`ACT export rejects a malformed ${field} acknowledgment without retry`, async ({ page }) => {
    const state = await exportRecoveryFixture(page);
    await page.route(`**/api/v1/projects/${projectId}/policy-jobs`, async route => {
      const body = route.request().postDataJSON(); state.submitted.push(body);
      const receipt: Record<string, any> = exportReceipt(state, { ...body });
      if (field === 'project') receipt.project_id = 'foreign-project';
      if (field === 'runtime') receipt.request.runtime_id = 'another-worker';
      if (field === 'operation') receipt.kind = receipt.request.operation = 'policy.quantize';
      if (field === 'budget') receipt.request.timeout_seconds = 7200;
      if (field === 'status') receipt.status = ['queued'];
      if (field === 'timestamp') receipt.updated_at = 'not-a-date';
      await route.fulfill({ status: 202, json: field === 'empty' ? {} : receipt });
    });
    await exportButton(page).click();
    await expect(page.getByRole('region', { name: 'ACT export recovery' })).toContainText('unverified');
    await expect(page.getByRole('region', { name: 'ACT export receipt' })).toHaveCount(0);
    await expect(exportButton(page)).toBeDisabled(); expect(state.submitted).toHaveLength(1);
  });
}

test('ACT export cannot POST when its pending journal cannot be written', async ({ page }) => {
  await page.addInitScript(() => {
    const write = Storage.prototype.setItem;
    Storage.prototype.setItem = function (key, value) {
      if (key.startsWith('firebird:job-attempt:policy.export:')) throw new Error('Storage unavailable');
      return write.call(this, key, value);
    };
  });
  const state = await exportRecoveryFixture(page);
  await exportButton(page).click();
  await expect(page.getByRole('region', { name: 'ACT inference export' })).toContainText('storage is unavailable');
  await expect(exportButton(page)).toBeDisabled();
  await page.getByRole('button', { name: 'Dataset', exact: true }).click(); await reopenExport(page);
  await expect(exportButton(page)).toBeDisabled(); expect(state.submitted).toHaveLength(0);
});

for (const failure of ['receipt-write', 'cleanup', 'unmounted-cleanup'] as const) {
  test(`ACT export retains accepted receipt after ${failure} storage failure`, async ({ page }) => {
    await page.addInitScript(failure => {
      const write = Storage.prototype.setItem, remove = Storage.prototype.removeItem;
      Storage.prototype.setItem = function (key, value) {
        if (failure === 'receipt-write' && key.startsWith('firebird:act-export-receipt:')) throw new Error('Storage unavailable');
        return write.call(this, key, value);
      };
      Storage.prototype.removeItem = function (key) {
        if (failure !== 'receipt-write' && key.startsWith('firebird:job-attempt:policy.export:')) throw new Error('Storage unavailable');
        return remove.call(this, key);
      };
    }, failure);
    const state = await exportRecoveryFixture(page); let release!: () => void;
    const gate = new Promise<void>(resolve => { release = resolve; });
    await page.route(`**/api/v1/projects/${projectId}/policy-jobs`, async route => {
      const body = route.request().postDataJSON(); state.submitted.push(body);
      if (failure === 'unmounted-cleanup') await gate;
      await route.fulfill({ status: 202, json: exportReceipt(state, body) });
    });
    try {
      await exportButton(page).click(); await expect.poll(() => state.submitted.length).toBe(1);
      if (failure === 'unmounted-cleanup') {
        await page.getByRole('button', { name: 'Dataset', exact: true }).click(); release(); await reopenExport(page);
      }
      await expect(page.getByRole('region', { name: 'ACT export receipt' })).toContainText('acknowledged-export');
      await expect(page.getByRole('region', { name: 'ACT export recovery' })).toContainText('storage is unavailable');
      await expect(page.getByRole('button', { name: 'Exporting ACT policy…', exact: true })).toBeDisabled();
      await expect(page.getByRole('region', { name: 'ACT inference export' })).not.toContainText('Submitting one');
      expect(await page.evaluate(project => JSON.parse(sessionStorage.getItem(`firebird:job-attempt:policy.export:${project}`)!).state, projectId)).toBe('pending');
      expect(state.submitted).toHaveLength(1);
    } finally { release(); }
  });
}

test('ACT export retained active receipt survives reload even when history omits it', async ({ page }) => {
  const state = await exportRecoveryFixture(page);
  await page.route(`**/api/v1/projects/${projectId}/policy-jobs`, async route => {
    const body = route.request().postDataJSON(); state.submitted.push(body);
    await route.fulfill({ status: 202, json: exportReceipt(state, body) });
  });
  await exportButton(page).click();
  await expect(page.getByRole('region', { name: 'ACT export receipt' })).toContainText('acknowledged-export');
  await page.reload(); await reopenExport(page);
  await expect(page.getByRole('region', { name: 'ACT export receipt' })).toContainText('acknowledged-export');
  await expect(page.getByRole('button', { name: 'Exporting ACT policy…', exact: true })).toBeDisabled();
  expect(state.submitted).toHaveLength(1);
});

test('ACT export history refresh begun before uncertainty cannot authorize another request', async ({ page }) => {
  const state = await exportRecoveryFixture(page); let release!: () => void; let reads = 0;
  const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route(`**/api/v1/projects/${projectId}/jobs`, async route => {
    reads += 1; if (reads === 1) await gate; await route.fulfill({ json: state.originalJobs });
  });
  await page.route(`**/api/v1/projects/${projectId}/policy-jobs`, async route => { state.submitted.push(route.request().postDataJSON()); await route.abort(); });
  try {
    await page.getByRole('button', { name: 'Refresh export jobs', exact: true }).click();
    await expect.poll(() => reads).toBe(1); await exportButton(page).click();
    const allow = page.getByRole('button', { name: 'I checked the export jobs; allow a new request' });
    await expect(page.getByRole('region', { name: 'ACT export recovery' })).toContainText('unverified');
    release();
    await expect(page.getByRole('button', { name: 'Refresh export jobs', exact: true })).toBeEnabled();
    await expect(allow).toBeDisabled();
    await page.getByRole('button', { name: 'Refresh export jobs', exact: true }).click();
    await expect(allow).toBeEnabled(); await allow.click();
    await expect(exportButton(page)).toBeEnabled(); expect(state.submitted).toHaveLength(1);
  } finally { release(); }
});

test('ACT export failed history refresh cannot clear uncertainty', async ({ page }) => {
  const state = await exportRecoveryFixture(page);
  await page.route(`**/api/v1/projects/${projectId}/policy-jobs`, async route => { state.submitted.push(route.request().postDataJSON()); await route.abort(); });
  await exportButton(page).click();
  await expect(page.getByRole('region', { name: 'ACT export recovery' })).toContainText('unverified');
  await page.route(`**/api/v1/projects/${projectId}/jobs`, route => route.fulfill({ status: 503, json: { detail: 'History unavailable' } }));
  await page.getByRole('button', { name: 'Refresh export jobs', exact: true }).click();
  await expect(page.getByRole('region', { name: 'ACT inference export' })).toContainText('job updates are unavailable');
  await expect(page.getByRole('button', { name: 'I checked the export jobs; allow a new request' })).toBeDisabled();
  expect(state.submitted).toHaveLength(1);
});

test('ACT export definite rejection permits an explicit request without a false receipt', async ({ page }) => {
  const state = await exportRecoveryFixture(page);
  await page.route(`**/api/v1/projects/${projectId}/policy-jobs`, async route => {
    state.submitted.push(route.request().postDataJSON()); await route.fulfill({ status: 422, json: { detail: 'Checkpoint does not meet export requirements.' } });
  });
  await exportButton(page).click();
  await expect(page.getByRole('region', { name: 'ACT inference export' })).toContainText('Checkpoint does not meet export requirements.');
  await expect(exportButton(page)).toBeEnabled();
  await expect(page.getByRole('region', { name: 'ACT export receipt' })).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'ACT export recovery' })).toHaveCount(0);
  expect(state.submitted).toHaveLength(1);
});

test('ACT export double activation and checkpoint switching keep the original request identity', async ({ page }) => {
  const state = await exportRecoveryFixture(page);
  state.artifacts.push({ ...state.artifacts.find(item => item.id === 'checkpoint-artifact')!, id: 'second-complete-checkpoint', label: 'Second completed checkpoint' });
  await page.reload(); await reopenExport(page);
  await page.getByRole('combobox', { name: 'Checkpoint', exact: true }).selectOption('checkpoint-artifact');
  let release!: () => void; const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route(`**/api/v1/projects/${projectId}/policy-jobs`, async route => {
    const body = route.request().postDataJSON(); state.submitted.push(body); await gate;
    await route.fulfill({ status: 202, json: exportReceipt(state, body) });
  });
  try {
    await exportButton(page).evaluate((element: HTMLButtonElement) => { element.click(); element.click(); });
    await expect.poll(() => state.submitted.length).toBe(1);
    await page.getByRole('combobox', { name: 'Checkpoint', exact: true }).selectOption('second-complete-checkpoint');
    await expect(page.getByRole('button', { name: 'Exporting ACT policy…', exact: true })).toBeDisabled();
    release();
    await expect(exportButton(page)).toBeEnabled();
    await expect(page.getByRole('region', { name: 'ACT export receipt' })).toHaveCount(0);
    await page.getByRole('combobox', { name: 'Checkpoint', exact: true }).selectOption('checkpoint-artifact');
    await expect(page.getByRole('region', { name: 'ACT export receipt' })).toContainText('acknowledged-export');
    await expect(page.getByRole('button', { name: 'Exporting ACT policy…', exact: true })).toBeDisabled();
    expect(state.submitted).toEqual([{ operation: 'policy.export', runtime_id: 'act-cpu', artifact_id: 'checkpoint-artifact', training_method: 'full', timeout_seconds: 600 }]);
  } finally { release(); }
});

test('ACT export ignores another project receipt and download ancestry', async ({ page }) => {
  const state = await workspace(page, 'running', true, 'remote-complete');
  state.jobs.unshift({ ...state.job, id: 'foreign-export', project_id: 'foreign-project', kind: 'policy.export', status: 'queued', request: { operation: 'policy.export', artifact_id: 'checkpoint-artifact', runtime_id: 'act-cpu', training_method: 'full', timeout_seconds: 600 } });
  state.artifacts.push({ ...state.artifacts.find(item => item.id === 'checkpoint-artifact')!, id: 'foreign-output', project_id: 'foreign-project', format: 'inference_export', parent_ids: ['checkpoint-artifact'] });
  state.artifacts.push({ ...state.artifacts.find(item => item.id === 'checkpoint-artifact')!, id: 'foreign-copy', project_id: 'foreign-project', format: 'native_checkpoint', parent_ids: ['checkpoint-artifact'] });
  state.artifacts.push({ ...state.artifacts.find(item => item.id === 'checkpoint-artifact')!, id: 'unbound-output', format: 'inference_export', parent_ids: ['foreign-copy'] });
  await page.reload(); await reopenExport(page);
  await expect(exportButton(page)).toBeEnabled();
  const control = page.getByRole('region', { name: 'ACT inference export' });
  await expect(control.getByRole('region', { name: 'ACT export receipt' })).toHaveCount(0);
  await expect(control.getByRole('link', { name: 'Download ACT inference package' })).toHaveCount(0);
  expect(state.submitted).toHaveLength(0);
});

test('ACT export corrupt stored receipt blocks submission without claiming acceptance', async ({ page }) => {
  await page.addInitScript(project => sessionStorage.setItem(`firebird:act-export-receipt:${project}:checkpoint-artifact`, '{broken'), projectId);
  const state = await exportRecoveryFixture(page);
  await expect(page.getByRole('region', { name: 'ACT inference export' })).toContainText('receipt is unreadable');
  await expect(exportButton(page)).toBeDisabled();
  await expect(page.getByRole('region', { name: 'ACT export receipt' })).toHaveCount(0);
  expect(state.submitted).toHaveLength(0);
});

test('ACT export lost acknowledgment retains its original checkpoint and worker after another selection and reload', async ({ page }) => {
  const state = await exportRecoveryFixture(page);
  state.artifacts.push({ ...state.artifacts.find(item => item.id === 'checkpoint-artifact')!, id: 'second-complete-checkpoint', label: 'Second completed checkpoint' });
  await page.reload(); await reopenExport(page);
  await page.getByRole('combobox', { name: 'Checkpoint', exact: true }).selectOption('checkpoint-artifact');
  await page.route(`**/api/v1/projects/${projectId}/policy-jobs`, async route => { state.submitted.push(route.request().postDataJSON()); await route.abort(); });
  await exportButton(page).click();
  const recovery = page.getByRole('region', { name: 'ACT export recovery' });
  await expect(recovery).toContainText('Checkpoint: checkpoint-artifact');
  await page.getByRole('combobox', { name: 'Checkpoint', exact: true }).selectOption('second-complete-checkpoint');
  await expect(recovery).toContainText(`Project: ${projectId}`);
  await expect(recovery).toContainText('Checkpoint: checkpoint-artifact');
  await expect(recovery).toContainText('Worker: act-cpu');
  await expect(recovery).toContainText('full FP32, 600 seconds');
  await expect(exportButton(page)).toBeDisabled();
  await page.reload(); await reopenExport(page);
  await page.getByRole('combobox', { name: 'Checkpoint', exact: true }).selectOption('second-complete-checkpoint');
  await expect(recovery).toContainText('Checkpoint: checkpoint-artifact');
  await expect(recovery).toContainText('Worker: act-cpu');
  await expect(page.getByRole('button', { name: 'I checked the export jobs; allow a new request' })).toBeDisabled();
  expect(state.submitted).toHaveLength(1);
});

test('ACT export pending reload retains exact original request context as uncertain', async ({ page }) => {
  const state = await exportRecoveryFixture(page); let release!: () => void;
  const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route(`**/api/v1/projects/${projectId}/policy-jobs`, async route => {
    state.submitted.push(route.request().postDataJSON()); await gate;
    await route.abort().catch(() => undefined); // Reload may already have closed this one request.
  });
  try {
    await exportButton(page).click(); await expect.poll(() => state.submitted.length).toBe(1);
    expect(await page.evaluate(project => JSON.parse(sessionStorage.getItem(`firebird:job-attempt:policy.export:${project}`)!).state, projectId)).toBe('pending');
    await page.reload(); await reopenExport(page);
    const recovery = page.getByRole('region', { name: 'ACT export recovery' });
    await expect(recovery).toContainText(`Project: ${projectId}`);
    await expect(recovery).toContainText('Checkpoint: checkpoint-artifact');
    await expect(recovery).toContainText('Worker: act-cpu');
    await expect(recovery).toContainText('full FP32, 600 seconds');
    await expect(recovery).toContainText('outcome is unverified');
    await expect(exportButton(page)).toBeDisabled();
    await expect(page.getByRole('button', { name: 'I checked the export jobs; allow a new request' })).toBeDisabled();
    expect(state.submitted).toHaveLength(1);
  } finally { release(); }
});

test('ACT export recovery at another checkpoint cannot clear a known active project export', async ({ page }) => {
  const state = await exportRecoveryFixture(page);
  state.artifacts.push({ ...state.artifacts.find(item => item.id === 'checkpoint-artifact')!, id: 'second-complete-checkpoint', label: 'Second completed checkpoint' });
  await page.reload(); await reopenExport(page);
  await page.getByRole('combobox', { name: 'Checkpoint', exact: true }).selectOption('checkpoint-artifact');
  await page.route(`**/api/v1/projects/${projectId}/policy-jobs`, async route => {
    const body = route.request().postDataJSON(); state.submitted.push(body);
    state.originalJobs.unshift({ ...exportReceipt(state, body), status: 'running' });
    await route.abort();
  });
  await exportButton(page).click();
  const recovery = page.getByRole('region', { name: 'ACT export recovery' });
  await expect(recovery).toContainText('outcome is unverified');
  await page.getByRole('combobox', { name: 'Checkpoint', exact: true }).selectOption('second-complete-checkpoint');
  await page.getByRole('button', { name: 'Refresh export jobs', exact: true }).click();
  await expect(recovery).toContainText('Recorded export acknowledged-export for checkpoint checkpoint-artifact is running');
  await expect(recovery).toContainText('Wait for active project exports to finish');
  await expect(page.getByRole('button', { name: 'I checked the export jobs; allow a new request' })).toBeDisabled();
  await expect(exportButton(page)).toBeDisabled(); expect(state.submitted).toHaveLength(1);
});


async function exportedPackageFixture(page: Page, configured = true) {
  const state = await workspace(page, 'running', true, 'remote-complete');
  if (configured) state.extraRuntimes.push({ id: 'native-act', label: 'Generated native ACT CPU worker', provider: 'local', execution: 'native', device: 'cpu', enabled: true, launchable: true, native_quantization: true, native_quantization_only: true, act_export: false, training: false, simulation: false, engine_evaluation: false, run: false });
  const packages = ['first', 'second'].map((suffix, index) => ({
    id: `export-${suffix}:operation`, project_id: projectId, job_id: `export-${suffix}`,
    format: 'inference_export', label: 'ACT inference export', file_bytes: 100,
    path: `private/${suffix}`, manifest_sha256: (index ? 'd' : 'e').repeat(64), parent_ids: ['checkpoint-artifact'],
    metadata: { architecture: 'act', inference_only: true, use_vae: false },
  }));
  state.artifacts.push(...packages);
  state.jobs.push(...packages.map(item => ({ ...state.job, id: item.job_id, kind: 'policy.export', status: 'succeeded', stage: 'operation', request: { operation: 'policy.export', runtime_id: 'act-cpu', artifact_id: 'checkpoint-artifact', training_method: 'full', timeout_seconds: 600 }, result: { decision: 'completed', artifacts: [structuredClone(item)], reports: [] } })));
  await page.reload(); await reopenExport(page);
  return { ...state, packages };
}
const exportedPackage = (page: Page, id: string) => page.getByRole('region', { name: `ACT inference package ${id}`, exact: true });

test('ACT exported packages show distinct identities and hand the exact second package to native quantization', async ({ page }, testInfo) => {
  const state = await exportedPackageFixture(page);
  for (const item of state.packages) {
    const card = exportedPackage(page, item.id);
    await expect(card).toContainText(`Export ${item.job_id.slice(0, 8)}`);
    await expect(card.getByRole('link', { name: 'Download ACT inference package' })).toHaveAttribute('href', `/api/v1/projects/${projectId}/artifacts/${encodeURIComponent(item.id)}/download`);
    await expect(card.getByRole('button', { name: 'Quantize this package' })).toBeEnabled();
  }
  await noOverflow(page);
  const exportSection = page.getByRole('region', { name: 'ACT inference export', exact: true });
  await exportSection.screenshot({ path: testInfo.outputPath('act-export-next-actions-light.png') });
  await page.getByRole('button', { name: 'Dark', exact: true }).click();
  await exportSection.screenshot({ path: testInfo.outputPath('act-export-next-actions-dark.png') });
  await page.getByRole('button', { name: 'Light', exact: true }).click();
  await exportedPackage(page, state.packages[1].id).getByRole('button', { name: 'Quantize this package' }).click();
  await expect(page.getByRole('button', { name: `Choose ACT inference export · ${state.packages[1].id}`, exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('group', { name: 'Policy', exact: true }).locator(`input[value="${state.packages[1].id}"]`)).toBeChecked();
  expect(state.submitted).toEqual([]);
  await noOverflow(page);
  await page.screenshot({ path: testInfo.outputPath('act-export-quantization-handoff.png'), fullPage: true });
  await page.route(`**/api/v1/projects/${projectId}/policy-jobs`, async route => {
    const request = route.request().postDataJSON(); state.submitted.push(request);
    await route.fulfill({ status: 202, json: { ...state.job, id: 'explicit-quantization', kind: 'policy.quantize', status: 'queued', request } });
  });
  await page.getByRole('button', { name: 'Create ACT quantized package', exact: true }).click();
  await expect.poll(() => state.submitted.length).toBe(1);
  expect(state.submitted).toEqual([{ operation: 'policy.quantize', runtime_id: 'native-act', artifact_id: state.packages[1].id, native_quantization: { format: 'firebird_quant', bits: 8, group_size: 64 }, timeout_seconds: 600 }]);
  expect(state.unexpected).toEqual([]);
});

test('ACT export handoff keeps a later manual source choice through a refresh', async ({ page }) => {
  const state = await exportedPackageFixture(page);
  await exportedPackage(page, state.packages[1].id).getByRole('button', { name: 'Quantize this package' }).click();
  const policies = page.getByRole('group', { name: 'Policy', exact: true });
  await expect(policies.locator(`input[value="${state.packages[1].id}"]`)).toBeChecked();
  await policies.locator(`input[value="${state.packages[0].id}"]`).check();
  state.artifacts.reverse();
  await page.getByRole('button', { name: 'Refresh ACT quantization jobs', exact: true }).click();
  await expect(policies.locator(`input[value="${state.packages[0].id}"]`)).toBeChecked();
  expect(state.submitted).toEqual([]); expect(state.unexpected).toEqual([]);
});

test('ACT exported package does not fall back to an engine when the native worker is unavailable', async ({ page }) => {
  const state = await exportedPackageFixture(page, false);
  const card = exportedPackage(page, state.packages[0].id);
  await expect(card.getByRole('button', { name: 'Quantize this package' })).toBeDisabled();
  await expect(card).toContainText('A local native ACT quantization worker is not available');
  await expect(card.getByRole('link', { name: 'Download ACT inference package' })).toBeVisible();
  expect(state.submitted).toEqual([]);
});

for (const fault of ['remote', 'incompatible', 'unrecorded', 'manifest', 'pending'] as const) test(`ACT exported package blocks ${fault} quantization handoff without hiding its download`, async ({ page }) => {
  const state = await exportedPackageFixture(page); const item = state.packages[0];
  const saved = state.jobs.find(job => job.id === item.job_id)!;
  if (fault === 'remote') Object.assign(item.metadata, { storage: 'gcs' });
  if (fault === 'incompatible') Object.assign(item.metadata, { use_vae: true });
  if (fault === 'unrecorded') saved.result.artifacts = [];
  if (fault === 'manifest') saved.result.artifacts[0].manifest_sha256 = 'f'.repeat(64);
  if (fault === 'pending') saved.status = 'running';
  await page.reload(); await reopenExport(page);
  const card = exportedPackage(page, item.id);
  await expect(card.getByRole('button', { name: 'Quantize this package' })).toBeDisabled();
  await expect(card.getByRole('status')).toContainText(fault === 'pending' ? 'Wait for this export' : fault === 'unrecorded' || fault === 'manifest' ? 'not recorded in the completed export result' : 'not a supported local ACT');
  await expect(card.getByRole('link', { name: 'Download ACT inference package' })).toBeVisible();
  expect(state.submitted).toEqual([]);
});

test('ACT export handoff waits for its completed history record without automatically navigating', async ({ page }) => {
  const state = await exportedPackageFixture(page); const item = state.packages[0];
  const original = structuredClone(state.jobs);
  await page.route(`**/api/v1/projects/${projectId}/jobs`, route => route.fulfill({ json: original.filter(job => job.id !== item.job_id) }));
  await page.reload(); await reopenExport(page);
  const action = exportedPackage(page, item.id).getByRole('button', { name: 'Quantize this package' });
  await expect(action).toBeDisabled();
  await page.route(`**/api/v1/projects/${projectId}/jobs`, route => route.fulfill({ json: original }));
  await page.getByRole('button', { name: 'Refresh export jobs', exact: true }).click();
  await expect(action).toBeEnabled();
  await expect(page.getByRole('article', { name: 'Training run monitor' })).toBeVisible();
  expect(state.submitted).toEqual([]);
});

test('ACT export preferred package stays within its owning project', async ({ page }) => {
  const state = await exportedPackageFixture(page);
  await page.route('**/api/v1/projects', route => route.fulfill({ json: [{ id: projectId, name: 'Training visibility', created_at: timestamp() }, { id: 'other-project', name: 'Other project', created_at: timestamp() }] }));
  await page.route('**/api/v1/projects/other-project/*', route => route.fulfill({ json: [] }));
  await page.reload(); await reopenExport(page);
  await exportedPackage(page, state.packages[1].id).getByRole('button', { name: 'Quantize this package' }).click();
  await expect(page.getByRole('group', { name: 'Policy', exact: true }).locator(`input[value="${state.packages[1].id}"]`)).toBeChecked();
  await selectProject(page, 'other-project');
  await expect(page.getByText('No saved models in this project yet')).toBeVisible();
  await expect(page.getByRole('group', { name: 'Policy', exact: true }).locator('input:checked')).toHaveCount(0);
  await expect(page.getByRole('group', { name: 'Policy', exact: true }).locator(`input[value="${state.packages[1].id}"]`)).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Create ACT quantized package', exact: true })).toHaveCount(0);
  expect(state.submitted).toEqual([]);
});


test('ACT export handoff preserves an unresolved quantization request without retrying it', async ({ page }) => {
  const state = await exportedPackageFixture(page);
  const key = `firebird:job-attempt:policy.quantize:${projectId}`;
  const stored = JSON.stringify({ state: 'uncertain', message: 'Earlier ACT request has an unverified outcome.' });
  await page.evaluate(({ key, stored }) => sessionStorage.setItem(key, stored), { key, stored });
  await exportedPackage(page, state.packages[1].id).getByRole('button', { name: 'Quantize this package' }).click();
  await expect(page.getByRole('group', { name: 'Policy', exact: true }).locator(`input[value="${state.packages[1].id}"]`)).toBeChecked();
  await expect(page.getByText('Earlier ACT request has an unverified outcome. Further submissions are paused.')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Create ACT quantized package', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'I checked the jobs; allow a new request' })).toBeDisabled();
  expect(await page.evaluate(key => sessionStorage.getItem(key), key)).toBe(stored);
  expect(state.submitted).toEqual([]);
});


// Generated completed exports only; these cases never run a model or provider.
async function teacherHandoffFixture(page: Page, configured = true) {
  const state = await exportedPackageFixture(page);
  if (configured) state.extraRuntimes.push({ id: 'student-cpu', label: 'Generated ACT distillation worker', provider: 'local', execution: 'native', device: 'cpu', enabled: true, launchable: true, native_distillation: true, native_distillation_only: true, training: false, simulation: false, engine_evaluation: false, run: false });
  const prepared = structuredClone(state.jobs.find(job => job.id === 'dataset')!);
  prepared.id = 'prepared-observations';
  Object.assign(prepared.result, { source: 'local', repo_id: null, inspection_scope: 'complete_snapshot', snapshot: { id: `sha256:${'b'.repeat(64)}`, manifest_sha256: 'b'.repeat(64), lineage_validated: true, total_episodes: 10 } });
  state.jobs.push(prepared);
  await page.reload(); await reopenExport(page);
  return state;
}
const teacherChoices = (page: Page) => page.getByRole('group', { name: 'Teacher', exact: true });
const useTeacher = (page: Page, id: string) => exportedPackage(page, id).getByRole('button', { name: 'Use as ACT teacher', exact: true });

test('training export continues to Distill with the exact second same-label package and explicit recipe', async ({ page }, testInfo) => {
  const state = await teacherHandoffFixture(page);
  for (const item of state.packages) {
    await expect(useTeacher(page, item.id)).toBeEnabled();
    await expect(exportedPackage(page, item.id).getByRole('button', { name: 'Quantize this package' })).toBeEnabled();
    await expect(exportedPackage(page, item.id).getByRole('link', { name: 'Download ACT inference package' })).toHaveAttribute('href', `/api/v1/projects/${projectId}/artifacts/${encodeURIComponent(item.id)}/download`);
  }
  await noOverflow(page);
  await page.getByRole('region', { name: 'ACT inference export', exact: true }).screenshot({ path: testInfo.outputPath('training-distill-continuation-light.png') });
  const chosen = state.packages[1];
  await useTeacher(page, chosen.id).click();
  const panel = page.getByRole('region', { name: 'ACT distillation', exact: true });
  await expect(panel).toBeVisible();
  await expect(teacherChoices(page).locator('input:checked')).toHaveValue(chosen.id);
  await expect(page.getByLabel('Selected teacher model', { exact: true })).toContainText(chosen.id);
  await expect(page.getByLabel('Selected teacher model', { exact: true })).toContainText(`Run ${chosen.job_id.slice(0, 8)}`);
  await expect(page.getByRole('group', { name: 'Dataset', exact: true }).locator('input:checked')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Train ACT256 student', exact: true })).toBeDisabled();
  expect(state.submitted).toEqual([]);
  await page.getByRole('button', { name: 'Dark', exact: true }).click();
  await noOverflow(page);
  await panel.screenshot({ path: testInfo.outputPath('training-distill-destination-dark.png') });
  await page.getByRole('button', { name: 'Light', exact: true }).click();
  await page.getByRole('group', { name: 'Dataset', exact: true }).locator('input[value="prepared-observations"]').check();
  const consent = page.getByRole('checkbox', { name: /I verified that the dataset/ });
  await expect(consent).not.toBeChecked();
  const units = ['degrees', 'degrees', 'degrees', 'degrees', 'degrees', 'recorded_gripper'];
  for (const [label, value] of [['Training episodes', '0, 1'], ['Validation episodes', '2, 3'], ['Final episodes', '4, 5'], ['Six coordinate units', units.join(', ')]]) await page.getByLabel(label, { exact: true }).fill(value);
  await page.getByRole('checkbox', { name: 'This snapshot contains generated test observations.' }).check();
  await expect(page.getByRole('button', { name: 'Train ACT256 student', exact: true })).toBeDisabled();
  await consent.check();
  await page.route(`**/api/v1/projects/${projectId}/policy-jobs`, async route => {
    const request = route.request().postDataJSON(); state.submitted.push(request);
    await route.fulfill({ status: 202, json: { ...state.job, id: 'explicit-student', kind: 'policy.distill', status: 'queued', request } });
  });
  await page.getByRole('button', { name: 'Train ACT256 student', exact: true }).click();
  await expect(page.getByRole('article', { name: 'Distillation job details' })).toHaveAttribute('data-job-id', 'explicit-student');
  expect(state.submitted).toEqual([{ operation: 'policy.distill', runtime_id: 'student-cpu', artifact_id: chosen.id, dataset_job_id: 'prepared-observations', timeout_seconds: 600, native_distillation: { adapter: 'act-act-v1', student: 'act-256', steps: 100, learning_rate: .0001, seed: 1729, frame_stride: 30, splits: { train: [0, 1], validation: [2, 3], final: [4, 5] }, units, coordinate_attestation: 'generated_fixture' } }]);
  expect(state.unexpected).toEqual([]);
});

test('training export explains unavailable distillation without changing quantization or download', async ({ page }) => {
  const state = await teacherHandoffFixture(page, false), item = state.packages[1];
  await expect(useTeacher(page, item.id)).toBeDisabled();
  await expect(exportedPackage(page, item.id)).toContainText('A local ACT distillation worker is not available');
  await expect(exportedPackage(page, item.id).getByRole('button', { name: 'Quantize this package' })).toBeEnabled();
  await expect(exportedPackage(page, item.id).getByRole('link', { name: 'Download ACT inference package' })).toBeVisible();
  expect(state.submitted).toEqual([]);
});

for (const fault of ['remote', 'architecture', 'unrecorded', 'manifest', 'foreign-output', 'pending', 'prediction-horizon'] as const) test(`training export refuses ${fault} teacher handoff`, async ({ page }) => {
  const state = await teacherHandoffFixture(page), item = state.packages[1];
  const saved = state.jobs.find(job => job.id === item.job_id)!;
  if (fault === 'remote') Object.assign(item.metadata, { storage: 'gcs' });
  if (fault === 'architecture') Object.assign(item.metadata, { architecture: 'smolvla' });
  if (fault === 'unrecorded') saved.result.artifacts = [];
  if (fault === 'manifest') saved.result.artifacts[0].manifest_sha256 = 'f'.repeat(64);
  if (fault === 'foreign-output') saved.result.artifacts[0].project_id = 'another-project';
  if (fault === 'pending') saved.status = 'running';
  if (fault === 'prediction-horizon') Object.assign(item.metadata, { prediction_horizon: 32 });
  await page.reload(); await reopenExport(page);
  await expect(useTeacher(page, item.id)).toBeDisabled();
  await expect(exportedPackage(page, item.id).getByRole('link', { name: 'Download ACT inference package' })).toBeVisible();
  if (fault === 'prediction-horizon') await expect(exportedPackage(page, item.id)).toContainText('model format, timing or simulator details are incomplete or unsupported');
  expect(state.submitted).toEqual([]);
});

test('manual Distill teacher choices reject partial timing metadata and preserve legacy timing metadata', async ({ page }) => {
  const state = await teacherHandoffFixture(page), changed = state.packages[1];
  Object.assign(changed.metadata, { prediction_horizon: 32 });
  await page.reload(); await reopenExport(page);
  await expect(useTeacher(page, changed.id)).toBeDisabled();
  await page.getByRole('button', { name: 'Distill', exact: true }).click();
  await page.getByRole('button', { name: `Choose ACT inference export · ${state.packages[0].id}`, exact: true }).click();
  await expect(teacherChoices(page).locator(`input[value="${changed.id}"]`)).toHaveCount(0);
  await expect(teacherChoices(page).locator(`input[value="${state.packages[0].id}"]`)).toHaveCount(1);
  await expect(teacherChoices(page).locator('input:checked')).toHaveValue(state.packages[0].id);
  await expect(page.getByRole('button', { name: 'Train ACT256 student', exact: true })).toBeDisabled();
  expect(state.submitted).toEqual([]);
});

for (const location of ['metadata', 'checkpoint'] as const) for (const field of ['control_contract', 'control_contract_sha256'] as const) test(`incomplete simulator ${location} ${field} preserves download but excludes both transformation paths`, async ({ page }) => {
  const state = await teacherHandoffFixture(page), guarded = state.packages[1], legacy = state.packages[0];
  const contract = { [field]: field === 'control_contract' ? { kind: 'simulator_joint_position', schema_version: 1 } : 'f'.repeat(64) };
  Object.assign(guarded.metadata, location === 'metadata' ? contract : { checkpoint: contract });
  // Optional legacy import fields carry no active contract claim.
  Object.assign(legacy.metadata, { control_contract: null, control_contract_sha256: null, checkpoint: { control_contract: null, control_contract_sha256: null } });
  await page.reload(); await reopenExport(page);
  const card = exportedPackage(page, guarded.id);
  await expect(useTeacher(page, guarded.id)).toBeDisabled();
  await expect(card.getByRole('button', { name: 'Quantize this package' })).toBeDisabled();
  await expect(card).toContainText('model format, timing or simulator details are incomplete or unsupported');
  await expect(card.getByRole('link', { name: 'Download ACT inference package' })).toHaveAttribute('href', `/api/v1/projects/${projectId}/artifacts/${encodeURIComponent(guarded.id)}/download`);
  await expect(useTeacher(page, legacy.id)).toBeEnabled();
  await expect(exportedPackage(page, legacy.id).getByRole('button', { name: 'Quantize this package' })).toBeEnabled();
  await page.getByRole('button', { name: 'Distill', exact: true }).click();
  await page.getByRole('button', { name: `Choose ACT inference export · ${state.packages[0].id}`, exact: true }).click();
  await expect(teacherChoices(page).locator(`input[value="${guarded.id}"]`)).toHaveCount(0);
  await expect(page.getByText('Some ACT packages are excluded because their model format, timing or simulator details are incomplete or unsupported. Refresh or export a complete inference package.', { exact: true })).toBeVisible();
  await expect(teacherChoices(page).locator(`input[value="${legacy.id}"]`)).toHaveCount(1);
  await expect(teacherChoices(page).locator('input:checked')).toHaveValue(legacy.id);
  await expect(page.getByRole('button', { name: 'Train ACT256 student', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Quantize', exact: true }).click();
  await page.getByRole('button', { name: `Choose ACT inference export · ${state.packages[0].id}`, exact: true }).click();
  const policies = page.getByRole('group', { name: 'Policy', exact: true });
  await expect(policies.locator(`input[value="${guarded.id}"]`)).toHaveCount(0);
  await expect(page.getByText('Some ACT packages are excluded because their model format, timing or simulator details are incomplete or unsupported. Refresh or export a complete inference package.', { exact: true })).toBeVisible();
  await expect(policies.locator(`input[value="${legacy.id}"]`)).toHaveCount(1);
  await expect(policies.locator('input:checked')).toHaveValue(legacy.id);
  await expect(page.getByRole('button', { name: 'Create ACT quantized package', exact: true })).toBeEnabled();
  expect(state.submitted).toEqual([]);
});

test('complete simulator-bound export keeps exact Distill and Quantize handoffs without dispatch', async ({ page }) => {
  const state = await teacherHandoffFixture(page), item = state.packages[1];
  Object.assign(item.metadata, { prediction_horizon: 32, execution_horizon: 8, temporal_contract_sha256: 'e'.repeat(64), control_contract_sha256: 'd'.repeat(64),
    control_contract: { schema_version: 1, kind: 'simulator_joint_position', controller: 'joint_position_targets', state_key: 'observation.state', action_key: 'action', state_units: 'radians', action_units: 'radians', timebase: 'simulation_seconds',
      joint_order: ['j0', 'j1', 'j2', 'j3', 'j4', 'gripper'], action_fps: 30, camera: { key: 'observation.images.front', width: 32, height: 32, prim: '/World/Camera' },
      source: { dataset_snapshot_id: `sha256:${'b'.repeat(64)}`, dataset_manifest_sha256: 'b'.repeat(64), demonstrations_sha256: 'b'.repeat(64), scene_sha256: ['c'.repeat(64)], scene_hash_scope: 'root USD bytes; referenced assets not inventoried', origins: ['synthetic'] }, physical_calibration_verified: false, task_success_verified: false } });
  const original = structuredClone(item.metadata);
  await page.reload(); await reopenExport(page);
  await expect(useTeacher(page, item.id)).toBeEnabled();
  await expect(exportedPackage(page, item.id).getByRole('button', { name: 'Quantize this package' })).toBeEnabled();
  await useTeacher(page, item.id).click();
  await expect(teacherChoices(page).locator('input:checked')).toHaveValue(item.id);
  await expect(page.getByText('Inherited action timing: plans 32 actions and applies 8 per update.', { exact: false })).toBeVisible();
  await expect(page.getByText(/Use the dataset this teacher was trained from and enter radians for all six joint units/)).toBeVisible();
  const datasetChoices = page.getByRole('group', { name: 'Dataset', exact: true });
  await expect(datasetChoices.locator('label').filter({ has: page.locator('input[value="prepared-observations"]') })).toContainText('Teacher source dataset');
  await expect(datasetChoices.locator('input:checked')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Train ACT256 student', exact: true })).toBeDisabled();
  await page.reload(); await reopenExport(page);
  await exportedPackage(page, item.id).getByRole('button', { name: 'Quantize this package' }).click();
  await expect(page.getByRole('group', { name: 'Policy', exact: true }).locator('input:checked')).toHaveValue(item.id);
  await expect(page.getByText('Inherited action timing: plans 32 actions and applies 8 per update.', { exact: false })).toBeVisible();
  await expect(page.getByText(/A compatible Run profile and separate rollout consent are still required/)).toBeVisible();
  expect(item.metadata).toEqual(original);
  expect(state.submitted).toEqual([]);
  expect(state.unexpected).toEqual([]);
});

test('training teacher continuation never replaces a manual selection when the package returns after a missing read', async ({ page }) => {
  const state = await teacherHandoffFixture(page), chosen = state.packages[1], manual = state.packages[0];
  await useTeacher(page, chosen.id).click();
  await expect(teacherChoices(page).locator('input:checked')).toHaveValue(chosen.id);
  const index = state.artifacts.findIndex(item => item.id === chosen.id); state.artifacts.splice(index, 1);
  await page.getByRole('button', { name: 'Refresh distillation jobs', exact: true }).click();
  await expect(teacherChoices(page).locator(`input[value="${chosen.id}"]`)).toHaveCount(0);
  await expect(teacherChoices(page).locator('input:checked')).toHaveCount(0);
  await teacherChoices(page).locator(`input[value="${manual.id}"]`).check();
  state.artifacts.push(chosen);
  await page.getByRole('button', { name: 'Refresh distillation jobs', exact: true }).click();
  await expect(teacherChoices(page).locator(`input[value="${chosen.id}"]`)).toHaveCount(1);
  await expect(teacherChoices(page).locator('input:checked')).toHaveValue(manual.id);
  await expect(page.getByLabel('Selected teacher model', { exact: true })).toHaveCount(0);
  expect(state.submitted).toEqual([]);
});

test('training teacher handoff preserves an unresolved distillation submission journal', async ({ page }) => {
  const state = await teacherHandoffFixture(page);
  const key = `firebird:job-attempt:policy.distill:${projectId}`;
  const stored = JSON.stringify({ state: 'uncertain', message: 'Earlier student request has an unverified outcome.' });
  await page.evaluate(({ key, stored }) => sessionStorage.setItem(key, stored), { key, stored });
  await useTeacher(page, state.packages[1].id).click();
  await expect(page.getByText('Earlier student request has an unverified outcome.', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Train ACT256 student', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'I checked recorded jobs; allow a new request', exact: true })).toBeDisabled();
  await expect(teacherChoices(page).locator('input:checked')).toHaveValue(state.packages[1].id);
  expect(await page.evaluate(key => sessionStorage.getItem(key), key)).toBe(stored);
  expect(state.submitted).toEqual([]);
});

test('training teacher handoff is cleared by a manual model choice or project switch', async ({ page }) => {
  const state = await teacherHandoffFixture(page);
  await page.route('**/api/v1/projects', route => route.fulfill({ json: [{ id: projectId, name: 'Training visibility', created_at: timestamp() }, { id: 'other-project', name: 'Other project', created_at: timestamp() }] }));
  // Deliberately return foreign records; the destination must apply project ownership.
  await page.route('**/api/v1/projects/other-project/jobs', route => route.fulfill({ json: state.jobs }));
  await page.route('**/api/v1/projects/other-project/artifacts', route => route.fulfill({ json: state.artifacts }));
  await page.reload(); await reopenExport(page);
  await useTeacher(page, state.packages[1].id).click();
  await expect(teacherChoices(page).locator('input:checked')).toHaveValue(state.packages[1].id);
  await page.getByRole('button', { name: 'Choose another teacher', exact: true }).click();
  await page.getByRole('button', { name: `Choose ACT inference export · ${state.packages[0].id}`, exact: true }).click();
  await expect(teacherChoices(page).locator('input:checked')).toHaveValue(state.packages[0].id);
  await reopenExport(page);
  await useTeacher(page, state.packages[1].id).click();
  await selectProject(page, 'other-project');
  await expect(page.getByText('No saved models in this project yet')).toBeVisible();
  await expect(teacherChoices(page)).toHaveCount(0);
  await expect(teacherChoices(page).locator(`input[value="${state.packages[1].id}"]`)).toHaveCount(0);
  await expect(page.getByLabel('Selected teacher model', { exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Train ACT256 student', exact: true })).toHaveCount(0);
  expect(state.submitted).toEqual([]);
});


// Generated browser fixtures exercise admission and exact requests; no training runs.
async function temporalTraining(page: Page, model = 'ACT', accumulationCatalog: AccumulationCatalog = 'supported', savedAccumulation?: number) {
  const state = await workspace(page, 'local-snapshot', false, undefined, accumulationCatalog);
  const dataset = state.jobs.find(item => item.id === 'dataset')!;
  Object.assign(dataset.result!, { source: 'huggingface', repo_id: 'fixture/robot', revision });
  state.extraRuntimes.push({ id: 'all-models-a100', label: 'A100', accelerator: 'A100', provider: 'gcp', execution: 'skypilot', device: 'cuda', enabled: true, training: true, training_model_ids: ['act', 'smolvla', 'pi05'], gpu_memory_mib: 40960 });
  if (savedAccumulation !== undefined) await page.evaluate(({ projectId, savedAccumulation }) => { const key = `firebird.workflow.${projectId}`; localStorage.setItem(key, JSON.stringify({ ...JSON.parse(localStorage.getItem(key) ?? '{}'), gradientAccumulation: savedAccumulation })); }, { projectId, savedAccumulation });
  await page.reload();
  await page.getByRole('button', { name: 'Fine-tune', exact: true }).click();
  await page.getByRole('button', { name: 'Start a new fine-tuning', exact: true }).click();
  const setup = page.getByRole('navigation', { name: 'Training setup' });
  await setup.getByRole('button', { name: 'Model', exact: true }).click();
  await chooseOnlyModel(page, model);
  await setup.getByRole('button', { name: 'Compute', exact: true }).click();
  await page.locator('summary').filter({ hasText: /^Advanced settings/ }).click();
  return { ...state, setup };
}

for (const model of ['ACT', 'SmolVLA']) {
  test(`training horizons submit independent ${model} action timing with exact dataset identity`, async ({ page }, testInfo) => {
    const state = await temporalTraining(page, model);
    const timing = page.getByRole('group', { name: 'Action timing', exact: true });
    await expect(timing.getByRole('spinbutton', { name: 'Prediction horizon', exact: true })).toHaveValue(model === 'ACT' ? '100' : '50');
    await timing.getByRole('spinbutton', { name: 'Prediction horizon', exact: true }).fill('24');
    await timing.getByRole('spinbutton', { name: 'Execution horizon', exact: true }).fill('8');
    await expect(page.getByRole('region', { name: 'Training recipe review' })).toContainText('Predict 24 · execute 8');
    await expect(page.getByRole('region', { name: 'Training recipe review' })).toContainText('fixture/robot');
    await noOverflow(page);
    await page.screenshot({ path: testInfo.outputPath(`training-timing-${model}.png`), fullPage: true });
    await page.getByRole('button', { name: 'Start fine-tuning', exact: true }).click();
    await expect.poll(() => state.submitted.length).toBe(1);
    expect(state.submitted[0]).toMatchObject({ operation: 'policy.finetune', dataset_job_id: 'dataset', runtime_id: 'all-models-a100', timeout_seconds: 86400,
      training: { prediction_horizon: 24, execution_horizon: 8, observation_history: 1, frame_stride: 1, model_id: model === 'ACT' ? 'code://lerobot/act' : 'lerobot/smolvla_base' } });
    expect(state.submitted[0].training).not.toHaveProperty('chunk_size');
    expect(state.unexpected).toEqual([]);
  });
}

test('training horizons reject invalid order and bounds before any POST', async ({ page }) => {
  const state = await temporalTraining(page);
  const timing = page.getByRole('group', { name: 'Action timing', exact: true });
  const prediction = timing.getByRole('spinbutton', { name: 'Prediction horizon', exact: true });
  const execution = timing.getByRole('spinbutton', { name: 'Execution horizon', exact: true });
  const submit = page.getByRole('button', { name: 'Start fine-tuning', exact: true });
  for (const [p, e] of [['8', '9'], ['0', '1'], ['1025', '1'], ['3.5', '1'], ['8', '0']]) {
    await prediction.fill(p); await execution.fill(e);
    await expect(timing.getByRole('alert')).toBeVisible();
    await expect(submit).toBeDisabled();
  }
  await prediction.fill('8'); await execution.fill('4');
  await expect(submit).toBeEnabled();
  expect(state.submitted).toEqual([]);
});

test('training horizons keep model drafts separate and omit unsupported family overrides', async ({ page }) => {
  const state = await temporalTraining(page);
  const timing = page.getByRole('group', { name: 'Action timing', exact: true });
  await timing.getByRole('spinbutton', { name: 'Prediction horizon', exact: true }).fill('12');
  await timing.getByRole('spinbutton', { name: 'Execution horizon', exact: true }).fill('4');
  async function choose(model: string) {
    await state.setup.getByRole('button', { name: 'Model', exact: true }).click();
    await chooseOnlyModel(page, model);
    await state.setup.getByRole('button', { name: 'Compute', exact: true }).click();
    await page.locator('summary').filter({ hasText: /^Advanced settings/ }).click();
  }
  await choose('SmolVLA');
  await expect(timing.getByRole('spinbutton', { name: 'Prediction horizon', exact: true })).toHaveValue('50');
  await choose('ACT');
  await expect(timing.getByRole('spinbutton', { name: 'Prediction horizon', exact: true })).toHaveValue('12');
  await expect(timing.getByRole('spinbutton', { name: 'Execution horizon', exact: true })).toHaveValue('4');
  await choose('π₀.₅');
  await expect(timing).toHaveCount(0);
  await expect(page.getByText('Action timing follows this model’s native configuration.', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Start fine-tuning', exact: true }).click();
  await expect.poll(() => state.submitted.length).toBe(1);
  for (const key of ['prediction_horizon', 'execution_horizon', 'observation_history', 'frame_stride', 'chunk_size']) expect(state.submitted[0].training).not.toHaveProperty(key);
});

test('training horizons leave checkpoint resume unchanged and show recorded timing only', async ({ page }) => {
  const state = await workspace(page, 'failed');
  state.job.request.training = { ...state.job.request.training, prediction_horizon: 24, execution_horizon: 6 };
  await state.monitor.getByRole('button', { name: 'Resume from checkpoint' }).click();
  await page.locator('summary').filter({ hasText: /^Advanced settings/ }).click();
  await expect(page.getByText('Original recipe preserved.', { exact: true })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Training recipe review' })).toContainText('Checkpoint-owned timing');
  await expect(page.getByRole('checkbox', { name: 'Customize action horizons' })).toHaveCount(0);
  await page.getByRole('button', { name: 'Resume fine-tuning', exact: true }).click();
  await expect.poll(() => state.submitted.length).toBe(1);
  expect(state.submitted[0]).toMatchObject({ artifact_id: 'checkpoint-artifact', training: null });
});

test('training horizons keep uncustomized recipes compatible with the prior form', async ({ page }) => {
  const state = await temporalTraining(page);
  await expect(page.getByRole('checkbox', { name: 'Customize action horizons' })).toHaveCount(0);
  await page.getByRole('spinbutton', { name: 'Prediction horizon', exact: true }).fill('24');
  await page.getByRole('spinbutton', { name: 'Execution horizon', exact: true }).fill('8');
  await page.getByRole('button', { name: 'Reset settings', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Training recipe review' })).toContainText('Predict 100 · execute 100 · default');
  await page.getByRole('button', { name: 'Start fine-tuning', exact: true }).click();
  await expect.poll(() => state.submitted.length).toBe(1);
  for (const key of ['prediction_horizon', 'execution_horizon', 'observation_history', 'frame_stride', 'chunk_size']) expect(state.submitted[0].training).not.toHaveProperty(key);
});


test('keyed training retry after reload preserves the original key and recipe with no automatic POST', async ({ page }) => {
  const state = await workspace(page, 'empty', false);
  const requests: { key: string; body: Record<string, any> }[] = [];
  await page.route(`**/api/v1/projects/${projectId}/policy-jobs`, async route => {
    const key = route.request().headers()['idempotency-key'], body = route.request().postDataJSON();
    requests.push({ key, body });
    if (requests.length === 1) { await route.abort(); return; }
    const job = { id: 'same-request-accepted', project_id: projectId, kind: 'policy.finetune', status: 'queued', request: body, created_at: timestamp(), updated_at: timestamp() };
    state.jobs.unshift(job); state.savedSubmissions.set(key, job);
    await route.fulfill({ status: 202, json: job, headers: { 'Idempotency-Key': key, 'Cache-Control': 'no-store' } });
  });
  await (await newTraining(page)).click();
  const recovery = page.getByRole('region', { name: 'Training submission recovery' });
  await expect(recovery).toBeVisible();
  // The journal appears before the support lookup; observe the lost POST before reloading.
  await expect.poll(() => requests.length).toBe(1);
  await expect(recovery.getByRole('button', { name: 'Check saved request' })).toBeEnabled();
  await page.evaluate(project => localStorage.setItem(`firebird.workflow.${project}`, JSON.stringify({ trainingDefaultsVersion: 3, trainingSteps: 777, trainingModelId: 'smolvla' })), projectId);
  await page.reload();
  await page.getByRole('button', { name: 'Fine-tune', exact: true }).click();
  await expect(recovery.getByRole('button', { name: 'Retry same request' })).toBeDisabled();
  await recovery.getByRole('button', { name: 'Check saved request' }).click();
  await expect(recovery).toContainText('new request remains blocked');
  await expect(recovery.getByRole('button', { name: 'Retry same request' })).toBeEnabled();
  expect(requests).toHaveLength(1);
  await recovery.getByRole('button', { name: 'Retry same request' }).click();
  await expect(page.getByRole('article', { name: 'Training run monitor' })).toHaveAttribute('data-run-id', 'same-request-accepted');
  expect(requests).toHaveLength(2);
  expect(requests[1]).toEqual(requests[0]);
  expect(requests[1].body.training.steps).toBe(20000);
  expect(state.submissionLookups).toEqual([requests[0].key, requests[0].key, requests[0].key]);
});

test('keyed training requires current server support before sending any POST', async ({ page }) => {
  const state = await workspace(page, 'empty', false);
  await page.route(`**/api/v1/projects/${projectId}/submissions/*`, route => route.fulfill({ status: 404, json: { detail: 'Not Found' } }));
  await (await newTraining(page)).click();
  const recovery = page.getByRole('region', { name: 'Training submission recovery' });
  await expect(recovery).toBeVisible();
  await expect(recovery.getByRole('button', { name: 'Retry same request' })).toBeDisabled();
  expect(state.submitted).toEqual([]);
});

test('keyed training retains the accepted receipt if storage reads fail after acknowledgement', async ({ page }) => {
  const state = await workspace(page, 'empty', false);
  await page.route(`**/api/v1/projects/${projectId}/policy-jobs`, async route => {
    const body = route.request().postDataJSON(), key = route.request().headers()['idempotency-key'];
    state.submitted.push(body);
    await page.evaluate(() => {
      const get = Storage.prototype.getItem;
      Storage.prototype.getItem = function (key) {
        if (key.startsWith('firebird:submission:v1:')) throw new Error('Denied after ACK');
        return get.call(this, key);
      };
    });
    await route.fulfill({ status: 202, json: { id: 'retained-after-read-denial', project_id: projectId, kind: 'policy.finetune', status: 'queued', request: body, created_at: timestamp(), updated_at: timestamp() }, headers: { 'Idempotency-Key': key, 'Cache-Control': 'no-store' } });
  });
  await (await newTraining(page)).click();
  await expect(page.getByRole('article', { name: 'Training run monitor' })).toHaveAttribute('data-run-id', 'retained-after-read-denial');
  const recovery = page.getByRole('region', { name: 'Training submission recovery' });
  await expect(recovery).toContainText('storage is unavailable');
  await expect(recovery.getByRole('button', { name: 'Check saved request' })).toBeDisabled();
  expect(state.submitted).toHaveLength(1);
});

test('keyed training storage read denial blocks initial submission', async ({ page }) => {
  await page.addInitScript(() => {
    const get = Storage.prototype.getItem;
    Storage.prototype.getItem = function (key) {
      if (key.startsWith('firebird:submission:v1:')) throw new Error('Read denied');
      return get.call(this, key);
    };
  });
  const state = await workspace(page, 'empty', false);
  const start = await newTraining(page);
  await expect(start).toBeDisabled();
  await expect(page.getByRole('region', { name: 'Training submission recovery' })).toContainText('storage is unavailable');
  expect(state.submitted).toEqual([]);
});

test('keyed training late acknowledgement retains history without replacing a manually selected job', async ({ page }) => {
  const state = await workspace(page, 'history', false);
  let release!: () => void;
  const gate = new Promise<void>(resolve => { release = resolve; });
  let posts = 0;
  await page.route(`**/api/v1/projects/${projectId}/policy-jobs`, async route => {
    posts += 1;
    const key = route.request().headers()['idempotency-key'], body = route.request().postDataJSON();
    await gate;
    const job = { id: 'late-accepted-training', project_id: projectId, kind: 'policy.finetune', status: 'queued', request: body, created_at: timestamp(), updated_at: timestamp() };
    state.jobs.unshift(job); state.savedSubmissions.set(key, job);
    await route.fulfill({ status: 202, json: job, headers: { 'Idempotency-Key': key, 'Cache-Control': 'no-store' } });
  });
  try {
    await (await newTraining(page)).click();
    await expect.poll(() => posts).toBe(1);
    await page.getByRole('button', { name: 'Back to jobs', exact: true }).click();
    await page.locator('.job-history-entry[data-job-id="older-run"]').click();
    await expect(state.monitor).toHaveAttribute('data-run-id', 'older-run');
    release();
    await expect(page.getByRole('region', { name: 'Training submission recovery' })).toHaveCount(0);
    await expect(state.monitor).toHaveAttribute('data-run-id', 'older-run');
    await page.getByRole('button', { name: 'Back to jobs', exact: true }).click();
    await expect(page.locator('.job-history-entry[data-job-id="late-accepted-training"]')).toBeVisible();
    expect(posts).toBe(1);
  } finally { release(); }
});

for (const fault of ['malformed', 'foreign'] as const) test(`legacy training recovery refuses ${fault} history before allowing a new request`, async ({ page }) => {
  const state = await workspace(page, 'empty', false);
  await page.evaluate(project => sessionStorage.setItem(`firebird:job-attempt:policy.finetune:${project}`, JSON.stringify({ state: 'uncertain', message: 'Legacy request outcome is unverified.' })), projectId);
  await page.reload();
  await page.getByRole('button', { name: 'Fine-tune', exact: true }).click();
  const recovery = page.getByRole('region', { name: 'Training submission recovery' });
  const acknowledge = recovery.getByRole('button', { name: 'I checked the jobs; allow a new request' });
  const path = `**/api/v1/projects/${projectId}/jobs`;
  await page.route(path, route => route.fulfill({ json: fault === 'malformed' ? { jobs: [] } : [{ ...state.jobs[0], project_id: 'another-project' }] }));
  await recovery.getByRole('button', { name: 'Refresh training jobs' }).click();
  await expect(recovery).toContainText('malformed or contains another project');
  await expect(acknowledge).toBeDisabled();
  expect(state.submitted).toEqual([]);
  await page.unroute(path);
  await recovery.getByRole('button', { name: 'Refresh training jobs' }).click();
  await expect(acknowledge).toBeEnabled();
  expect(state.submitted).toEqual([]);
});


// Generated catalog support proves browser admission only, not native/CUDA updates.
for (const [model, method] of [['ACT', 'full'], ['SmolVLA', 'lora'], ['SmolVLA', 'qlora']] as const) test(`advertised accumulation submits exact ${model} ${method} optimizer-update recipe`, async ({ page }) => {
  const state = await temporalTraining(page, model);
  if (method === 'qlora') {
    await state.setup.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByRole('group', { name: 'Training method', exact: true }).getByRole('radio', { name: 'QLoRA', exact: true }).check();
    await state.setup.getByRole('button', { name: 'Compute', exact: true }).click();
    await page.locator('summary').filter({ hasText: /^Advanced settings/ }).click();
  }
  await page.getByLabel('Steps', { exact: true }).fill('20');
  await page.getByLabel('Batch size', { exact: true }).fill('3');
  await page.getByLabel('Gradient accumulation', { exact: true }).fill('4');
  await page.getByRole('button', { name: 'Help for Gradient accumulation', exact: true }).focus();
  await expect(page.getByRole('tooltip').filter({ hasText: 'Nominal effective batch: 12' })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Training recipe review', exact: true })).toContainText('20 optimizer updates');
  await expect(page.getByRole('region', { name: 'Training recipe review', exact: true })).not.toContainText('checkpoint/validation cadence count completed optimizer updates');
  expect(state.submitted).toEqual([]);
  await page.getByRole('button', { name: 'Start fine-tuning', exact: true }).click();
  await expect.poll(() => state.submitted.length).toBe(1);
  expect(state.submitted[0]).toMatchObject({ operation: 'policy.finetune', runtime_id: 'all-models-a100', dataset_job_id: 'dataset', training_method: method, training: { steps: 20, batch_size: 3, gradient_accumulation_steps: 4 } });
});

for (const capability of ['missing', 'disabled', 'wrong-runtime', 'wrong-unit', 'world-size'] as const) test(`accumulation ${capability} capability preserves a saved request value and requires explicit reset`, async ({ page }) => {
  const state = await temporalTraining(page, 'ACT', capability, 4);
  const accumulation = page.getByLabel('Gradient accumulation', { exact: true });
  await expect(accumulation).toHaveValue('4'); await expect(accumulation).toBeDisabled();
  await expect(page.getByRole('region', { name: 'Training recipe review', exact: true }).getByRole('alert')).toContainText('Accumulation 4 is not supported');
  await expect(page.getByRole('button', { name: 'Start fine-tuning', exact: true })).toBeDisabled();
  expect(state.submitted).toEqual([]);
  await page.getByRole('button', { name: 'Use accumulation 1', exact: true }).click();
  await expect(accumulation).toHaveValue('1'); await expect(accumulation).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Start fine-tuning', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'Start fine-tuning', exact: true }).click();
  await expect.poll(() => state.submitted.length).toBe(1);
  expect(state.submitted[0]).toMatchObject({ runtime_id: 'all-models-a100', training_method: 'full', training: { gradient_accumulation_steps: 1 } });
});

test('switching to another model keeps accumulation visible and blocked until an explicit correction', async ({ page }) => {
  const state = await temporalTraining(page);
  await page.getByLabel('Gradient accumulation', { exact: true }).fill('3');
  await state.setup.getByRole('button', { name: 'Model', exact: true }).click();
  await chooseOnlyModel(page, 'π₀.₅');
  await state.setup.getByRole('button', { name: 'Compute', exact: true }).click();
  await page.locator('summary').filter({ hasText: /^Advanced settings/ }).click();
  await expect(page.getByLabel('Gradient accumulation', { exact: true })).toHaveValue('3');
  await expect(page.getByLabel('Gradient accumulation', { exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Start fine-tuning', exact: true })).toBeDisabled();
  await state.setup.getByRole('button', { name: 'Model', exact: true }).click();
  await chooseOnlyModel(page, 'ACT');
  await state.setup.getByRole('button', { name: 'Compute', exact: true }).click();
  await page.locator('summary').filter({ hasText: /^Advanced settings/ }).click();
  await expect(page.getByLabel('Gradient accumulation', { exact: true })).toHaveValue('3');
  await expect(page.getByLabel('Gradient accumulation', { exact: true })).toBeEnabled();
  expect(state.submitted).toEqual([]);
});

test('accumulation capability never rewrites an exact checkpoint resume recipe', async ({ page }) => {
  const state = await workspace(page, 'failed', true, undefined, 'missing');
  state.job.request.training.gradient_accumulation_steps = 4;
  await state.monitor.getByRole('button', { name: 'Resume from checkpoint' }).click();
  await page.locator('summary').filter({ hasText: /^Advanced settings/ }).click();
  await expect(page.getByText('Original recipe preserved.', { exact: true })).toBeVisible();
  await expect(page.getByLabel('Gradient accumulation', { exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Use accumulation 1', exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Resume fine-tuning', exact: true }).click();
  await expect.poll(() => state.submitted.length).toBe(1);
  expect(state.submitted[0]).toMatchObject({ artifact_id: 'checkpoint-artifact', training: null });
});

test('accumulation predicate requires exact model method runtime and complete catalog semantics', () => {
  const model: TrainingModel = { id: 'act', label: 'ACT', description: '', model_id: 'code://lerobot/act', model_revision: 'a'.repeat(40), methods: ['full'], gradient_accumulation_supported: true, gradient_accumulation_runtime_ids: ['worker'], training_step_unit: 'optimizer_updates', training_world_size: 1 };
  expect(gradientAccumulationAvailable(model, 'worker', 'full')).toBe(true);
  expect(gradientAccumulationAvailable(model, 'other', 'full')).toBe(false);
  expect(gradientAccumulationAvailable(model, 'worker', 'lora')).toBe(false);
  for (const key of ['gradient_accumulation_supported', 'gradient_accumulation_runtime_ids', 'training_step_unit', 'training_world_size'] as const) {
    const partial = { ...model }; delete partial[key]; expect(gradientAccumulationAvailable(partial, 'worker', 'full')).toBe(false);
  }
  for (const patch of [{ gradient_accumulation_supported: 'true' }, { gradient_accumulation_runtime_ids: 'worker' }, { training_step_unit: 'microbatches' }, { training_world_size: true }, { training_world_size: 2 }, { id: 'custom' }]) expect(gradientAccumulationAvailable({ ...model, ...patch } as TrainingModel, 'worker', 'full')).toBe(false);
  const smol = { ...model, id: 'smolvla', methods: ['lora', 'qlora', 'full'] };
  expect(gradientAccumulationAvailable(smol, 'worker', 'lora')).toBe(true);
  expect(gradientAccumulationAvailable(smol, 'worker', 'qlora')).toBe(true);
  expect(gradientAccumulationAvailable(smol, 'worker', 'full')).toBe(false);
});

async function chooseOnlyModel(page: Page, name: string) {
 const group = page.getByRole('group', { name: 'Base model', exact: true });
 const checked = group.locator('input:checked');
 for (const input of await checked.all()) if (await input.getAttribute('aria-label') !== name) await input.uncheck();
 await group.getByRole('checkbox', { name, exact: true }).check();
}

test('selects two datasets and two models as independent GPU jobs with pinned recipes', async ({page}, testInfo) => {
 const state = await temporalTraining(page, 'ACT');
 const dataset = state.jobs.find(item => item.id === 'dataset')!;
 const data = {...dataset, id:'dataset-two', result:{...dataset.result, repo_id:'fixture/second', revision:'c'.repeat(40)}};
 await page.route(`**/api/v1/projects/${projectId}/jobs`, route => route.fulfill({json:[dataset, data]}));
 await state.setup.getByRole('button', {name:'Dataset',exact:true}).click();
 await page.getByRole('checkbox',{name:'fixture/second',exact:true}).check();
 await state.setup.getByRole('button',{name:'Model',exact:true}).click();
 await page.getByRole('checkbox',{name:'SmolVLA',exact:true}).check();
 await state.setup.getByRole('button',{name:'Compute',exact:true}).click();
 await page.route(`**/api/v1/projects/${projectId}/training-plan`, route => {
   const requests=route.request().postDataJSON();
   expect(requests).toHaveLength(2);
   return route.fulfill({json:{valid:true,jobs:2,gpus_per_job:1}});
 });
 await expect(page.getByRole('region',{name:'Training recipe review'})).toContainText('≈');
 const review = page.getByRole('region',{name:'Training recipe review'});
 await expect(review.locator('.training-memory-estimate').filter({has:page.getByText('SmolVLA',{exact:true})})).toContainText('Predict 50 · execute 50');
 await expect(review.locator('.training-memory-estimate').filter({has:page.getByText('ACT',{exact:true})})).toContainText('Predict 100 · execute 100');
 await noOverflow(page);
 await page.screenshot({path:testInfo.outputPath('multiple-model-recipe.png'),fullPage:true});
 await page.getByRole('button',{name:'Start 2 training jobs',exact:true}).click();
 await expect.poll(() => state.submitted.length).toBe(2);
 expect(new Set(state.submitted.map(item => item.training.model_id)).size).toBe(2);
 for(const body of state.submitted) {
   expect(body.dataset_job_ids).toEqual(['dataset','dataset-two']);
   expect(body.dataset_camera_mappings['dataset-two']).toEqual({'observation.images.front':'observation.images.front'});
   expect(body.runtime_id).toBe('all-models-a100');
 }
});

test('an interrupted multi-model plan reloads without POST and retries its saved recipe and keys', async ({page}) => {
 const state = await temporalTraining(page, 'ACT');
 await state.setup.getByRole('button',{name:'Model',exact:true}).click();
 await page.getByRole('checkbox',{name:'SmolVLA',exact:true}).check();
 await state.setup.getByRole('button',{name:'Compute',exact:true}).click();
 await page.route(`**/api/v1/projects/${projectId}/training-plan`, route => route.fulfill({json:{valid:true,jobs:2,gpus_per_job:1}}));
 const sent: {key:string;body:Record<string,any>}[] = [];
 let failed = false;
 await page.route(`**/api/v1/projects/${projectId}/policy-jobs`, async route => {
   const key=route.request().headers()['idempotency-key'], body=route.request().postDataJSON();
   sent.push({key,body});
   if(!failed) { failed=true; await route.abort(); return; }
   await route.fallback();
 });
 await page.getByRole('button',{name:'Start 2 training jobs',exact:true}).click();
 await expect.poll(() => sent.length).toBe(1);
 await expect(page.getByRole('region',{name:'Multi-model submission recovery'})).toBeVisible();
 await page.reload();
 await page.getByRole('button',{name:'Fine-tune',exact:true}).click();
 expect(sent).toHaveLength(1);
 const recovery=page.getByRole('region',{name:'Multi-model submission recovery'});
 await recovery.getByRole('button',{name:'Check saved jobs'}).click();
 await expect(recovery.getByRole('button',{name:'Continue saved plan'})).toBeEnabled();
 await recovery.getByRole('button',{name:'Continue saved plan'}).click();
 await expect.poll(() => sent.length).toBe(3);
 expect(sent[1]).toEqual(sent[0]);
 expect(sent[2].key).not.toBe(sent[0].key);
 await expect(recovery).toHaveCount(0);
});
