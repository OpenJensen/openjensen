import { expect, test, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';

const projectId = 'training-monitor-fixture';
const revision = 'a'.repeat(40);
const timestamp = () => new Date().toISOString();

async function workspace(page: Page, mode: 'running' | 'preparing' | 'failed' | 'stale' | 'unavailable' | 'legacy' | 'validating' = 'running', exportMode?: 'local' | 'remote' | 'remote-complete' | 'unconfigured' | 'failed' | 'local-dataset') {
  const submitted: Record<string, any>[] = [];
  const unexpected: string[] = [];
  const cancelled: string[] = [];
  const telemetryRequests: string[] = [];
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
  const jobs = [job, dataset];
  const artifact: Record<string, any> = { id: 'checkpoint-artifact', project_id: projectId, job_id: job.id, format: 'training_checkpoint', label: 'Checkpoint step 20', file_bytes: 100, path: 'checkpoint', manifest_sha256: 'c'.repeat(64), parent_ids: [], metadata: {} };
  if (exportMode) artifact.metadata = { architecture: 'act', training_backend: 'lerobot', method: 'full',
    dataset: { source: exportMode === 'local-dataset' ? 'local' : 'huggingface' },
    ...(exportMode?.startsWith('remote') ? { storage: 'gcs', reload_verified: exportMode === 'remote-complete', step: 20 } : {}) };
  const artifacts: Record<string, any>[] = [artifact];
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
      await route.fulfill({ status: 202, json: created });
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
      const replies: Record<string, unknown> = {
        '/api/v1/health': { status: 'ok', version: 'training-fixture' }, '/api/v1/capabilities': [],
        '/api/v1/projects': [{ id: projectId, name: 'Training visibility', created_at: timestamp() }],
        [`/api/v1/projects/${projectId}/jobs`]: jobs,
        [`/api/v1/projects/${projectId}/artifacts`]: artifacts,
        '/api/v1/jobs/dataset/episodes': { episodes: [], total: 10, offset: 0, limit: 6 },
        '/api/v1/policy-options': {
          runtimes: [...(exportMode && exportMode !== 'unconfigured' ? [{ id: 'act-cpu', label: 'Local CPU export', provider: 'local', execution: 'native', device: 'cpu', enabled: true, training: false, act_export: true, export_only: true, simulation: false }] : []), { id: 'skypilot-gcp-A100', label: 'A100', accelerator: 'A100', execution: 'skypilot', provider: 'gcp', device: 'cuda', enabled: true, training: true, simulation: false, training_model_ids: ['smolvla'] }],
          compute: { local: { enabled: false, label: 'Local' }, gcp: { enabled: true, default_gpu: 'A100', disk_size_gb: 200, idle_minutes: 10 } },
          training_models: [], sources: [], training_methods: [{ id: 'lora', label: 'LoRA', description: 'Train adapters.' }], default_training_method: 'lora',
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
  await expect(monitor).toBeVisible();
  return { telemetry, job, monitor, submitted, unexpected, cancelled, telemetryRequests };
}

async function noOverflow(page: Page) {
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(await page.evaluate(() => document.documentElement.clientWidth) + 1);
}

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
  await page.locator('summary').filter({ hasText: /^Training settings/ }).click();
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
  await page.getByRole('navigation', { name: 'Training setup' }).getByRole('button', { name: 'Compute', exact: true }).click();
  await page.locator('summary').filter({ hasText: /^Training settings/ }).click();
  await page.getByLabel('Steps', { exact: true }).fill('200');
  await page.getByLabel('Batch size', { exact: true }).fill('8');
  await page.getByLabel('Random seed', { exact: false }).fill('7');
  await page.getByLabel('Learning rate', { exact: true }).fill('0.0002');
  await page.getByLabel('Gradient accumulation', { exact: false }).fill('2');
  await page.getByLabel('Validation fraction', { exact: false }).fill('0.3');
  await page.getByLabel('Validate every (steps)', { exact: true }).fill('20');
  await page.getByRole('button', { name: 'Start fine-tuning', exact: true }).click();
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0].training).toMatchObject({ steps: 200, batch_size: 8, learning_rate: 0.0002, gradient_accumulation_steps: 2, seed: 7, validation_fraction: 0.3, eval_every: 20 });
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
  const { monitor, submitted, unexpected } = await workspace(page, 'running', 'local');
  await expect(monitor.getByRole('button', { name: 'Quantize checkpoint' })).toHaveCount(0);
  await expect(monitor.getByText('This does not establish robot task success', { exact: false })).toBeVisible();
  await monitor.getByRole('button', { name: 'Export ACT inference package' }).click();
  await expect(monitor.getByRole('link', { name: 'Download ACT inference package' })).toHaveAttribute('href', /artifacts\/inference-export\/download$/);
  expect(submitted).toEqual([{ operation: 'policy.export', runtime_id: 'act-cpu', artifact_id: 'checkpoint-artifact', training_method: 'full', timeout_seconds: 600 }]);
  expect(unexpected).toEqual([]);
  await noOverflow(page);
});

for (const mode of ['remote', 'unconfigured'] as const) test(`ACT export refuses ${mode} checkpoint without a job`, async ({ page }) => {
  const { monitor, submitted } = await workspace(page, 'running', mode);
  await expect(monitor.getByRole('button', { name: mode === 'remote' ? 'Download checkpoint and export' : 'Export ACT inference package' })).toBeDisabled();
  await expect(monitor.getByText(mode === 'remote' ? 'Wait for the completed, reload-verified ACT checkpoint.' : 'Ask the app operator to configure the local ACT export worker.', { exact: false })).toBeVisible();
  expect(submitted).toEqual([]);
});

test('ACT export failure stays visible without a download or quality claim', async ({ page }) => {
  const { monitor, submitted } = await workspace(page, 'running', 'failed');
  await monitor.getByRole('button', { name: 'Export ACT inference package' }).click();
  await expect(monitor.getByRole('alert').filter({ hasText: 'Synthetic CPU parity did not match.' })).toBeVisible();
  await expect(monitor.getByRole('link', { name: 'Download ACT inference package' })).toHaveCount(0);
  expect(submitted).toHaveLength(1);
});


test('ACT export does not mislabel local snapshot lineage as Hugging Face', async ({ page }) => {
  const { monitor, submitted } = await workspace(page, 'running', 'local-dataset');
  await expect(monitor.getByRole('button', { name: 'Export ACT inference package' })).toBeDisabled();
  await expect(monitor.getByText('ACT export from local dataset snapshots is not supported yet.')).toBeVisible();
  expect(submitted).toEqual([]);
});


test('completed cloud ACT checkpoint explicitly downloads and exports with separate ancestry', async ({ page }) => {
  const { monitor, submitted, unexpected } = await workspace(page, 'running', 'remote-complete');
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
  await workspace(page, 'running', 'local');
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  await expect(page.getByRole('combobox', { name: 'Execution target', exact: true })).toBeDisabled();
  await expect(page.getByRole('combobox', { name: 'Execution target', exact: true })).not.toContainText('Local CPU export');
  await page.getByRole('button', { name: 'Quantize', exact: true }).click();
  await expect(page.getByRole('combobox', { name: 'Execution target', exact: true })).not.toContainText('Local CPU export');
});
