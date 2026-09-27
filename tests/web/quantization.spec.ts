import { expect, test, type Page } from '@playwright/test';

const projectId = 'quantization-fixture';

async function workspace(page: Page, cloudOnly = false, cloudChecks = false,
  initialize?: (data: { jobs: Record<string, any>[]; artifacts: Record<string, any>[] }) => void) {
  const submitted: Record<string, any>[] = [];
  const unexpected: string[] = [];
  const training = (id: string, date: string) => ({
    id, project_id: projectId, kind: 'policy.finetune', status: 'succeeded', stage: 'completed',
    created_at: date, updated_at: date,
    request: { operation: 'policy.finetune', runtime_id: 'gcp', training_method: 'lora', training: { steps: 100 } },
  });
  const jobs: Record<string, any>[] = [training('recent-run', '2026-09-26T10:00:00Z'), training('older-run', '2026-09-25T10:00:00Z')];
  const checkpoint = (id: string, jobId: string, step: number) => ({
    id, project_id: projectId, job_id: jobId, format: 'training_checkpoint', label: 'LoRA checkpoint',
    file_bytes: 100, path: `/descriptors/${id}`, manifest_sha256: 'a'.repeat(64), parent_ids: [],
    metadata: { architecture: 'smolvla', step, storage: 'gcs', remote_uri: `gs://fixture/jobs/${jobId}/checkpoint-${step}` },
  });
  // An older run may have a greater step count; "latest" means newest run first.
  const artifacts: Record<string, any>[] = [checkpoint('old-900', 'older-run', 900), checkpoint('recent-20', 'recent-run', 20), checkpoint('recent-100', 'recent-run', 100), checkpoint('duplicate-final', 'recent-run', 100)];
  const events = [{ sequence: 1, stage: 'quantize', message: 'Compressing the language model to Q4', timestamp: '2026-09-26T11:00:00Z', data: {} }];
  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() === 'POST' && path.endsWith('/policy-jobs')) {
      const body = request.postDataJSON();
      submitted.push(body);
      const created = { id: 'quantize-run', project_id: projectId, kind: body.operation, request: body, status: 'running', stage: 'quantize', created_at: new Date().toISOString(), updated_at: new Date().toISOString() };
      jobs.unshift(created);
      await route.fulfill({ status: 202, json: created });
      return;
    }
    if (request.method() === 'GET') {
      const replies: Record<string, unknown> = {
        '/api/v1/health': { status: 'ok', version: 'fixture' }, '/api/v1/capabilities': [],
        '/api/v1/projects': [{ id: projectId, name: 'Cloud checkpoints', created_at: new Date().toISOString() }],
        [`/api/v1/projects/${projectId}/jobs`]: jobs,
        [`/api/v1/projects/${projectId}/artifacts`]: artifacts,
        '/api/v1/policy-options': {
          runtimes: [
            ...(!cloudOnly ? [{ id: 'xbox', label: 'Xbox 360', device: 'cuda', training: true, simulation: false, provider: 'local', execution: 'native', training_model_ids: ['smolvla'] }] : []),
            { id: 'gcp', label: 'GCP L4', device: 'cuda', training: true, simulation: false, provider: 'gcp', execution: 'skypilot', engine_evaluation: cloudChecks, run: cloudChecks, training_model_ids: ['smolvla'] },
          ],
          sources: [], training_models: [], training_methods: [{ id: 'lora', label: 'LoRA', description: 'Adapters' }], default_training_method: 'lora',
          quantization_defaults: { cuda: { language: 'Q8_0', vision: null }, cpu: { language: 'Q8_0', vision: null }, note: '' },
        },
      };
      if (path in replies) { await route.fulfill({ json: replies[path] }); return; }
      if (path.endsWith('/events')) { await route.fulfill({ json: events }); return; }
      if (path.endsWith('/training')) {
        await route.fulfill({ json: { job_id: 'recent-run', status: 'succeeded', phase: 'completed', completed_steps: 100, total_steps: 100, percent: 100, metrics: [], checkpoints: [{ step: 100, name: 'checkpoint-100' }], logs: [], events: [], reproducibility: {} } });
        return;
      }
    }
    unexpected.push(`${request.method()} ${path}`);
    await route.fulfill({ status: 405, json: { detail: 'Unexpected fixture request.' } });
  });
  initialize?.({ jobs, artifacts });
  await page.goto('/');
  return { jobs, artifacts, submitted, unexpected };
}

test('defaults to the latest cloud checkpoint and quantizes it with visible precision and activity', async ({ page }) => {
  const { jobs, submitted, unexpected } = await workspace(page);
  await page.getByRole('button', { name: 'Quantize', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Quantization jobs', exact: true })).toBeVisible();
  await expect(page.getByLabel('Checkpoint or policy', { exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'New quantization', exact: true }).click();
  const picker = page.getByLabel('Checkpoint or policy', { exact: true });
  await expect(picker).toHaveValue('latest');
  await expect(picker.locator('option[value="latest"]')).toHaveText(/Latest checkpoint · Step 100/);
  await expect(picker.locator('optgroup[label="Trained checkpoints"] option')).toHaveCount(3);
  await expect(page.getByLabel('Execution target', { exact: true })).toHaveValue('gcp');
  await expect(page.getByText(/Stored on Google Cloud/)).toBeVisible();
  await page.getByLabel('Quantization precision', { exact: true }).selectOption('Q8_0');
  await page.getByRole('button', { name: 'Start quantization', exact: true }).click();
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0]).toMatchObject({ operation: 'policy.quantize', artifact_id: 'recent-100', runtime_id: 'gcp', precision: { language: 'Q8_0', vision: null } });
  expect(submitted[0]).not.toHaveProperty('candidates');
  expect(submitted[0]).not.toHaveProperty('evaluation');
  await expect(page.getByRole('region', { name: 'Current job stage' })).toContainText('Compressing the language model');
  await expect(page.getByRole('heading', { name: 'Quantizing weights', exact: true })).toBeVisible();
  await expect(page.locator('.workflow-job-technical')).toHaveJSProperty('open', false);
  await expect(page.getByRole('progressbar', { name: 'Run in progress' })).toBeVisible();
  const quantized = { id: 'q8-artifact', project_id: projectId, job_id: 'quantize-run', label: 'Q8 policy', format: 'gguf', path: 'gs://fixture/q8', file_bytes: 8 * 1024 * 1024, manifest_sha256: 'b'.repeat(64), metadata: { precision: 'Q8_0' } };
  Object.assign(jobs[0], { status: 'succeeded', stage: 'completed', result: { artifacts: [quantized], reports: [], decision: 'completed' } });
  await expect(page.getByText('Quantized policy ready.')).toBeVisible();
  await expect(page.getByRole('link', { name: 'Download Q8 policy' })).toHaveAttribute('href', `/api/v1/projects/${projectId}/artifacts/q8-artifact/download`);
  expect(unexpected).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(await page.evaluate(() => document.documentElement.clientWidth) + 1);
});

test('preserves an earlier checkpoint selected from a training run through quantization', async ({ page }) => {
  const { submitted, unexpected } = await workspace(page);
  await page.getByRole('button', { name: 'Fine-tune', exact: true }).click();
  await page.locator('.job-history-entry[data-job-id="recent-run"]').click();
  const checkpointActions = page.getByRole('region', { name: 'Use a trained checkpoint' });
  await expect(checkpointActions).toBeVisible();
  await checkpointActions.getByLabel('Checkpoint', { exact: true }).selectOption('recent-20');
  await checkpointActions.getByRole('button', { name: 'Quantize checkpoint' }).click();
  await expect(page.getByLabel('Checkpoint or policy', { exact: true })).toHaveValue('recent-20');
  await page.getByRole('button', { name: 'Start quantization', exact: true }).click();
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0]).toMatchObject({ artifact_id: 'recent-20', precision: { language: 'Q8_0', vision: null } });
  await page.getByRole('button', { name: 'Quantize', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Quantization jobs', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'New quantization', exact: true }).click();
  await expect(page.getByLabel('Checkpoint or policy', { exact: true })).toHaveValue('latest');
  expect(submitted).toHaveLength(1);
  expect(unexpected).toEqual([]);
});

test('cloud quantization keeps Q8 by default and accepts explicit experimental precision without a native Spatial protocol', async ({ page }) => {
  await page.addInitScript(id => localStorage.setItem(`firebird.workflow.${id}`, JSON.stringify({
    suite: 'libero_spatial', mode: 'engine', steps: 10, compareQ4: true,
  })), projectId);
  const { submitted, unexpected } = await workspace(page, true);
  await page.getByRole('button', { name: 'Quantize', exact: true }).click();
  await page.getByRole('button', { name: 'New quantization', exact: true }).click();
  const precision = page.getByLabel('Quantization precision', { exact: true });
  await expect(precision).toHaveValue('recommended');
  await expect(precision.locator('option[value="recommended"]')).toHaveText('Recommended · 8-bit (Q8)');
  await precision.selectOption('Q4_0');
  await page.getByText('Advanced quantization', { exact: true }).click();
  await page.getByLabel('Also quantize vision to Q8 (experimental)', { exact: true }).check();
  await page.getByRole('button', { name: 'Start quantization', exact: true }).click();
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0]).toMatchObject({ operation: 'policy.quantize', precision: { language: 'Q4_0', vision: 'Q8_0' } });
  expect(submitted[0]).not.toHaveProperty('evaluation');
  expect(submitted[0]).not.toHaveProperty('candidates');
  expect(unexpected).toEqual([]);
});

test('keeps unsupported checkpoints visible and explains the SmolVLA quantization limit before submission', async ({ page }) => {
  const { artifacts, submitted, unexpected } = await workspace(page);
  Object.assign(artifacts.find(item => item.id === 'recent-100')!.metadata, { architecture: 'psi0' });
  await page.getByRole('button', { name: 'Quantize', exact: true }).click();
  await page.getByRole('button', { name: 'New quantization', exact: true }).click();
  await expect(page.getByLabel('Checkpoint or policy', { exact: true })).toHaveValue('latest');
  await expect(page.getByText(/GGUF quantization currently supports SmolVLA checkpoints/)).toBeVisible();
  await expect(page.getByRole('button', { name: 'Start quantization', exact: true })).toBeDisabled();
  await page.getByLabel('Checkpoint or policy', { exact: true }).selectOption('recent-20');
  await expect(page.getByRole('button', { name: 'Start quantization', exact: true })).toBeEnabled();
  expect(submitted).toEqual([]);
  expect(unexpected).toEqual([]);
});

for (const stage of ['Evaluate', 'Run']) {
  test(`${stage} explains a cloud-only setup and cannot submit to an unsupported cloud runtime`, async ({ page }) => {
    const { artifacts, submitted, unexpected } = await workspace(page, true);
    artifacts.push({ ...artifacts[0], id: 'packed-policy', label: 'Compressed policy', format: 'gguf' });
    await page.getByRole('button', { name: stage, exact: true }).click();
    await page.getByRole('button', { name: stage === 'Evaluate' ? 'New evaluation' : 'New run', exact: true }).click();
    await expect(page.getByLabel('Execution target', { exact: true })).toBeDisabled();
    await expect(page.getByLabel('Execution target', { exact: true }).locator('option[value="gcp"]')).toHaveCount(0);
    await expect(page.getByText(/Cloud engine checks appear when supported by the connected backend/)).toBeVisible();
    await expect(page.getByRole('button', { name: stage === 'Evaluate' ? 'Start evaluation' : 'Reload and run', exact: true })).toBeDisabled();
    await page.getByRole('button', { name: 'View training metrics', exact: true }).click();
    await expect(page.getByRole('region', { name: 'Fine-tuning jobs' })).toBeVisible();
    expect(submitted).toEqual([]);
    expect(unexpected).toEqual([]);
  });
}

test('native evaluation targets remain usable while cloud training targets are excluded', async ({ page }) => {
  const { artifacts, unexpected } = await workspace(page);
  artifacts.push({ ...artifacts[0], id: 'packed-policy', label: 'Compressed policy', format: 'gguf', metadata: { architecture: 'smolvla', precision: 'Q8_0' } });
  await page.getByRole('button', { name: 'Evaluate', exact: true }).click();
  await page.getByRole('button', { name: 'New evaluation', exact: true }).click();
  await expect(page.getByLabel('Execution target', { exact: true })).toHaveValue('xbox');
  await expect(page.getByLabel('Execution target', { exact: true }).locator('option[value="gcp"]')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Start evaluation', exact: true })).toBeEnabled();
  expect(unexpected).toEqual([]);
});

test('model names distinguish checkpoint choices and training runs while legacy runs keep their fallback', async ({ page }) => {
  const { jobs, artifacts, unexpected } = await workspace(page);
  Object.assign(jobs[0].request.training, { model_id: 'lerobot/smolvla_base' });
  Object.assign(jobs[1].request, { training_method: 'full', training: { model_id: 'code://lerobot/act', steps: 900 } });
  Object.assign(artifacts.find(item => item.id === 'recent-100')!.metadata, { base_model: { repository: 'lerobot/smolvla_base' } });
  Object.assign(artifacts.find(item => item.id === 'old-900')!.metadata, { architecture: 'act' });
  jobs.push({ ...jobs[1], id: 'psi-run', request: { ...jobs[1].request, training: { model_id: 'USC-PSI-Lab/psi-model', steps: 30 } } });
  artifacts.push({ ...artifacts[0], id: 'psi-checkpoint', job_id: 'psi-run', metadata: { step: 30, storage: 'gcs', remote_uri: 'gs://fixture/psi-run/checkpoint-30', base_model: { repository: 'USC-PSI-Lab/psi-model' } } } as typeof artifacts[number]);
  jobs.push({ ...jobs[1], id: 'legacy-run', request: { ...jobs[1].request, training: null }, created_at: '2026-09-24T10:00:00Z' });
  await page.getByRole('button', { name: 'Quantize', exact: true }).click();
  await page.getByRole('button', { name: 'New quantization', exact: true }).click();
  const picker = page.getByLabel('Checkpoint or policy', { exact: true });
  await expect(picker.locator('option[value="recent-100"]')).toHaveText(/Step 100 · SmolVLA · .* · Run recent-r/);
  await expect(picker.locator('option[value="old-900"]')).toHaveText(/Step 900 · ACT · .* · Run older-ru/);
  await expect(picker.locator('option[value="psi-checkpoint"]')).toHaveText(/Step 30 · Psi-Zero · .* · Run psi-run/);
  await page.getByRole('button', { name: 'Fine-tune', exact: true }).click();
  const runs = page.locator('.job-history-entry');
  await expect(runs.filter({ hasText: /SmolVLA · LORA/ })).toHaveCount(1);
  await expect(runs.filter({ hasText: /ACT · FULL/ })).toHaveCount(1);
  await expect(runs.filter({ hasText: /Psi-Zero · FULL/ })).toHaveCount(1);
  await expect(runs.filter({ hasText: /FULL/ })).toHaveCount(3);
  await page.locator('.job-history-entry[data-job-id="recent-run"]').click();
  const checkpointActions = page.getByRole('region', { name: 'Use a trained checkpoint' });
  await expect(checkpointActions.getByLabel('Checkpoint', { exact: true }).locator('option[value="latest"]')).toHaveText(/Latest checkpoint · Step 100 · SmolVLA/);
  expect(unexpected).toEqual([]);
});


test('opens a previous quantization job from history without mixing its details with the new-job form', async ({ page }) => {
  const { submitted, unexpected } = await workspace(page, true, false, ({ jobs, artifacts }) => {
    const output = { ...artifacts[0], id: 'prior-q4', job_id: 'prior-quantization', label: 'Step 20 Q4 policy', format: 'gguf', file_bytes: 1024 ** 3,
      metadata: { architecture: 'smolvla', storage: 'gcs', precision: 'Q4_0' } };
    artifacts.push(output);
    jobs.unshift({ id: 'prior-quantization', project_id: projectId, kind: 'policy.quantize', status: 'succeeded', stage: 'completed',
      created_at: '2026-09-26T12:00:00Z', updated_at: '2026-09-26T13:00:00Z',
      request: { operation: 'policy.quantize', runtime_id: 'gcp', artifact_id: 'recent-20', precision: { language: 'Q4_0', vision: null } },
      result: { artifacts: [output], reports: [], decision: 'completed' } });
    jobs.unshift({ id: 'newer-quantization', project_id: projectId, kind: 'policy.quantize', status: 'running', stage: 'preparing',
      created_at: '2026-09-26T14:00:00Z', updated_at: '2026-09-26T14:00:00Z',
      request: { operation: 'policy.quantize', runtime_id: 'gcp', artifact_id: 'recent-100', precision: { language: 'Q8_0', vision: null } } });
  });
  await page.getByRole('button', { name: 'Quantize', exact: true }).click();
  const history = page.getByRole('region', { name: 'Quantization jobs', exact: true });
  await expect(history.locator('.job-history-entry')).toHaveCount(2);
  await expect(history.locator('.job-history-entry').first()).toContainText('Preparing worker');
  await expect(page.getByLabel('Checkpoint or policy')).toHaveCount(0);
  await history.locator('[data-job-id="prior-quantization"]').click();
  const detail = page.getByRole('article', { name: 'Quantize job details' });
  await expect(detail.getByRole('heading', { name: 'SmolVLA · Q4', exact: true })).toBeVisible();
  await expect(detail.getByText('Step 20', { exact: true })).toBeVisible();
  await expect(detail.getByRole('heading', { name: 'Completed', exact: true })).toBeVisible();
  await expect(detail.getByRole('link', { name: 'Download Step 20 Q4 policy' })).toHaveAttribute('href', `/api/v1/projects/${projectId}/artifacts/prior-q4/download`);
  await expect(detail.locator('.workflow-job-technical')).toHaveJSProperty('open', false);
  await expect(page.getByRole('button', { name: 'Start quantization', exact: true })).toHaveCount(0);
  await detail.getByText('Details and logs', { exact: true }).click();
  await expect(detail.getByText('prior-quantization', { exact: true })).toBeVisible();
  await detail.getByRole('button', { name: 'New quantization', exact: true }).click();
  await expect(page.getByLabel('Checkpoint or policy', { exact: true })).toHaveValue('latest');
  await expect(page.getByRole('article', { name: 'Quantize job details' })).toHaveCount(0);
  await expect(page.getByRole('link', { name: /^Download / })).toHaveCount(0);
  await page.getByRole('button', { name: '← All quantization jobs', exact: true }).click();
  await expect(history).toBeVisible();
  expect(submitted).toEqual([]);
  expect(unexpected).toEqual([]);
});

for (const stage of ['Evaluate', 'Run']) {
  test(`${stage} uses supported cloud engine checks and ignores saved simulator preferences`, async ({ page }) => {
    await page.addInitScript(id => localStorage.setItem(`firebird.workflow.${id}`, JSON.stringify({
      suite: 'libero_spatial', mode: 'libero', steps: 280, warmup: 1, repetitions: 2,
    })), projectId);
    const { jobs, submitted, unexpected } = await workspace(page, true, true, ({ artifacts }) => {
      artifacts.push({ ...artifacts[0], id: 'cloud-q8', label: 'Cloud Q8 policy', format: 'gguf',
        metadata: { architecture: 'smolvla', storage: 'gcs', precision: 'Q8_0' } });
    });
    await page.getByRole('button', { name: stage, exact: true }).click();
    await expect(page.getByRole('region', { name: stage === 'Evaluate' ? 'Evaluation jobs' : 'Run jobs', exact: true })).toBeVisible();
    await expect(page.getByLabel('Input policy', { exact: true })).toHaveCount(0);
    await page.getByRole('button', { name: stage === 'Evaluate' ? 'New evaluation' : 'New run', exact: true }).click();
    await expect(page.getByLabel('Execution target', { exact: true })).toHaveValue('gcp');
    await expect(page.getByText('Native loading, finite actions and timing on synthetic inputs. Robot task success is not measured.')).toBeVisible();
    await page.getByRole('button', { name: stage === 'Evaluate' ? 'Start evaluation' : 'Reload and run', exact: true }).click();
    await expect.poll(() => submitted.length).toBe(1);
    expect(submitted[0]).toMatchObject({ operation: stage === 'Evaluate' ? 'policy.evaluate' : 'policy.run', artifact_id: 'cloud-q8', runtime_id: 'gcp',
      evaluation: { mode: 'engine', suite: 'libero_object', warmups: 1, repetitions: 2 } });
    expect(submitted[0].evaluation).not.toHaveProperty('parity_limits');
    Object.assign(jobs[0], { status: 'succeeded', stage: 'completed', result: { artifacts: [], decision: 'diagnostics_only', reports: [
      { stage: 'evaluate', scope: 'engine_diagnostics', p50_ms: 50, p95_ms: 55, peak_device_mib: 1024, success_rate: null },
    ] } });
    const results = page.getByRole('region', { name: 'Measured results' });
    await expect(results).toContainText('50 ms');
    await expect(results).toContainText('Synthetic input checks. Robot task success was not measured.');
    await expect(results.getByText('Task success', { exact: true })).toHaveCount(0);
    expect(unexpected).toEqual([]);
  });
}
