import { chooseTransformationModel, openTransformationJob, transformationJobs } from './lifecycle-controls';
import { expect, test, type Page } from '@playwright/test';

const projectId = 'quantization-fixture';

function choice(page: Page, group: string, value: string) {
  return page.getByRole('group', { name: group, exact: true }).locator(`input[value="${value}"]`);
}

async function expectCompute(page:Page,id:string) {
 await expect.poll(async()=>{
  const picker=page.getByRole('combobox',{name:'Compute',exact:true});
  if(await picker.count()) return await picker.getAttribute('value')===id;
  const radio=choice(page,'Compute',id);
  return await radio.count() ? await radio.isChecked() : false;
 }).toBe(true);
}
async function selectCompute(page:Page,label:string) {
 await page.getByRole('combobox',{name:'Compute',exact:true}).click();
 await page.getByRole('listbox',{name:'Compute',exact:true}).getByRole('option',{name:label,exact:true}).click();
}

async function expectSelectedPolicy(page:Page,id:string,selected=true) {
 await expect.poll(async()=>{
  const checkpoint=page.getByLabel('Checkpoint',{exact:true});
  if (await checkpoint.count()) return await checkpoint.inputValue() === id;
  const radio=choice(page,'My model',id);
  return await radio.count() ? await radio.isChecked() : false;
 }).toBe(selected);
}
async function choosePolicy(page:Page,id:string) {
 const checkpoint=page.getByLabel('Checkpoint',{exact:true});
 if(await checkpoint.count()) await checkpoint.selectOption(id);else await choice(page,'My model',id).check();
}


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
        '/api/v1/datasets': [],
        '/api/v1/simulation-options': { profiles: [], scored_evaluation: false, max_archive_bytes: 4294967296 },
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
  await page.goto('/datasets/');
  return { jobs, artifacts, submitted, unexpected };
}

test('explicitly selects a cloud checkpoint and quantizes it with visible precision and activity', async ({ page }) => {
  const { jobs, submitted, unexpected } = await workspace(page);
  await page.getByRole('link', { name: 'Quantize', exact: true }).click();
  await chooseTransformationModel(page, 'quantization', 'Choose LoRA checkpoint · recent-100');
  await expect(page.getByRole('group', { name: 'My model', exact: true })).toBeVisible();
  const picker = page.getByRole('group', { name: 'My model', exact: true });
  await expectSelectedPolicy(page,'recent-100');
  await expect(picker.getByLabel('Checkpoint',{exact:true}).locator('option:checked')).toContainText('Latest checkpoint');
  await expect(picker.getByRole('radio')).toHaveCount(0);
  await expect(picker.getByLabel('Checkpoint',{exact:true}).locator('option[value="old-900"]')).toHaveCount(0);
  await expectCompute(page,'gcp');
  await expect(page.getByRole('group',{name:'My model',exact:true})).toContainText('Cloud storage');
  await expect(choice(page,'Language compression','Q8_0')).toBeChecked();
  await expect(page.getByRole('radio',{name:'Recommended',exact:true})).toHaveCount(0);
  await choice(page,'Language compression','Q4_0').focus();
  await page.keyboard.press('ArrowLeft');
  await expect(choice(page,'Language compression','Q8_0')).toBeChecked();
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
  await page.getByRole('link', { name: 'Fine-tune', exact: true }).click();
  await page.locator('.job-history-entry[data-job-id="recent-run"]').click();
  const checkpointActions = page.getByRole('region', { name: 'Use a trained checkpoint' });
  await expect(checkpointActions).toBeVisible();
  await checkpointActions.getByLabel('Checkpoint', { exact: true }).selectOption('recent-20');
  await checkpointActions.getByRole('button', { name: 'Quantize checkpoint' }).click();
  await expectSelectedPolicy(page,'recent-20');
  await page.getByRole('button', { name: 'Start quantization', exact: true }).click();
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0]).toMatchObject({ artifact_id: 'recent-20', precision: { language: 'Q8_0', vision: null } });
  await page.getByRole('link', { name: 'Quantize', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Quantization jobs', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Start a new quantization', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Your quantization models', exact: true }).locator('button[aria-pressed="true"]')).toHaveCount(0);
  expect(submitted).toHaveLength(1);
  expect(unexpected).toEqual([]);
});

test('visualizes paired action drift and matching model-file size without inventing validation loss', async ({ page }) => {
  const { unexpected, submitted } = await workspace(page, true, false, ({ jobs }) => {
    jobs.unshift({ id: 'paired-quantization', project_id: projectId, kind: 'policy.quantize', status: 'succeeded', stage: 'completed', created_at: '2026-10-08T12:00:00Z', updated_at: '2026-10-08T12:01:00Z', request: { operation: 'policy.quantize', runtime_id: 'gcp', artifact_id: 'recent-100', precision: { language: 'Q8_0', vision: null } }, result: { artifacts: [], reports: [{ source_artifact_id: 'recent-100', source_manifest_sha256: 'a'.repeat(64), comparison: { schema_version: 1, scope: 'paired_synthetic_native_actions', reference: 'floating_gguf_before_quantization', backend: 'cpu', samples: 1, coordinates: 300, action_chunk_size: 50, real_action_dim: 6, input_sha256: 'b'.repeat(64), executable_sha256: 'c'.repeat(64), source_model_sha256: 'd'.repeat(64), quantized_model_sha256: 'e'.repeat(64), action_rmse: .0042, action_mse: .00001764, action_mae: .003, action_max_abs_difference: .015, source_file_bytes: 1800 * 1024 ** 2, quantized_file_bytes: 900 * 1024 ** 2, validation_loss: null, task_success: null } }] } });
  });
  await page.getByRole('link', { name: 'Quantize', exact: true }).click();
  await page.locator('.job-history-entry[data-job-id="paired-quantization"]').click();
  const comparison = page.getByRole('region', { name: 'Comparison with original model', exact: true });
  await expect(comparison.getByRole('img', { name: 'Action difference from original model on fixed inputs' })).toBeVisible();
  await expect(comparison).toContainText('0.0042');
  await expect(comparison).toContainText('50% smaller');
  await expect(comparison).toContainText('1,800 MiB');
  await expect(comparison).toContainText('Validation loss and robot task success were not measured.');
  await comparison.getByRole('button', { name: 'Help for Quantization comparison' }).focus();
  await expect(comparison.getByRole('tooltip')).toContainText('Original is zero by definition');
  await page.getByRole('heading', { name: 'Compared with original', exact: true }).click();
  await page.screenshot({ path: test.info().outputPath('quantization-comparison.png'), fullPage: true });
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(await page.evaluate(() => document.documentElement.clientWidth) + 1);
  expect(submitted).toEqual([]);
  expect(unexpected).toEqual([]);
});

test('old quantization jobs explain missing paired measurements instead of drawing invented points', async ({ page }) => {
  const { submitted } = await workspace(page, true, false, ({ jobs }) => jobs.unshift({ id: 'legacy-quantization', project_id: projectId, kind: 'policy.quantize', status: 'succeeded', stage: 'completed', created_at: '2026-10-08T12:00:00Z', updated_at: '2026-10-08T12:01:00Z', request: { operation: 'policy.quantize', runtime_id: 'gcp', artifact_id: 'recent-100', precision: { language: 'Q8_0', vision: null } }, result: { artifacts: [], reports: [{ inference: { finite_action_values: 1600, wall_seconds: 2 } }] } }));
  await page.getByRole('link', { name: 'Quantize', exact: true }).click();
  await page.locator('.job-history-entry[data-job-id="legacy-quantization"]').click();
  const comparison = page.getByRole('region', { name: 'Comparison with original model', exact: true });
  await expect(comparison).toContainText('This job did not record a paired comparison');
  await expect(comparison.getByRole('img')).toHaveCount(0);
  expect(submitted).toEqual([]);
});

test('cloud quantization keeps Q8 by default and accepts explicit experimental precision without a native Spatial protocol', async ({ page }) => {
  await page.addInitScript(id => localStorage.setItem(`firebird.workflow.${id}`, JSON.stringify({
    suite: 'libero_spatial', mode: 'engine', steps: 10, compareQ4: true,
  })), projectId);
  const { submitted, unexpected } = await workspace(page, true);
  await page.getByRole('link', { name: 'Quantize', exact: true }).click();
  await chooseTransformationModel(page, 'quantization', 'Choose LoRA checkpoint · recent-100');
  await expect(choice(page,'Language compression','Q8_0')).toBeChecked();
  await expect(page.getByRole('radio', { name: '4-bit', exact: true })).toBeVisible();
  await choice(page, 'Language compression', 'Q4_0').check();
  await expect(page.getByText('Advanced quantization',{exact:true})).toHaveCount(0);
  await expect(choice(page,'Vision compression','source')).toBeChecked();
  await choice(page,'Vision compression','Q8_0').check();
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
  await page.getByRole('link', { name: 'Quantize', exact: true }).click();
  await page.getByRole('button', { name: 'Start a new quantization', exact: true }).click();
  await expect(page.locator('.model-overview-card[data-artifact-id="recent-100"]')).toBeEnabled();
  await chooseTransformationModel(page, 'quantization', 'Choose LoRA checkpoint · recent-100');
  await expect(page.getByLabel('Checkpoint',{exact:true}).locator('option[value="recent-100"]')).toHaveJSProperty('disabled',true);
  await expect(page.getByText(/GGUF quantization currently supports SmolVLA checkpoints/)).toBeVisible();
  await expect(page.getByRole('button', { name: 'Start quantization', exact: true })).toBeDisabled();
  await choosePolicy(page,'recent-20');
  await expect(page.getByRole('button', { name: 'Start quantization', exact: true })).toBeEnabled();
  expect(submitted).toEqual([]);
  expect(unexpected).toEqual([]);
});

for (const stage of ['Evaluate', 'Run']) {
  test(`${stage} explains a cloud-only setup and cannot submit to an unsupported cloud runtime`, async ({ page }) => {
    const { artifacts, submitted, unexpected } = await workspace(page, true);
    artifacts.push({ ...artifacts[0], id: 'packed-policy', label: 'Compressed policy', format: 'gguf' });
    await page.getByRole('link', { name: stage, exact: true }).click();
    if (stage === 'Run') await page.getByRole('group', { name: 'Run mode', exact: true }).getByRole('button', { name: 'Check inference', exact: true }).click();
    await page.getByRole('button', { name: stage === 'Evaluate' ? 'New evaluation' : 'New run', exact: true }).click();
    await expect(page.getByRole('group', { name: 'Compute', exact: true }).getByRole('radio')).toHaveCount(0);
    await expect(choice(page, 'Compute', 'gcp')).toHaveCount(0);
    await expect(page.getByText(stage === 'Evaluate' ? 'No evaluation target is available.' : 'No policy runner is available.', { exact: true })).toBeVisible();
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
  await page.getByRole('link', { name: 'Evaluate', exact: true }).click();
  await page.getByRole('button', { name: 'New evaluation', exact: true }).click();
  await expect(choice(page, 'Compute', 'xbox')).toBeChecked();
  await expect(choice(page, 'Compute', 'gcp')).toHaveCount(0);
  await choosePolicy(page,'packed-policy');
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
  await page.getByRole('link', { name: 'Quantize', exact: true }).click();
  await chooseTransformationModel(page, 'quantization', 'Choose LoRA checkpoint · recent-100');
  const picker = page.getByRole('group', { name: 'My model', exact: true });
  await expect(picker.getByLabel('Checkpoint',{exact:true})).toHaveValue('recent-100');
  await expect(page.getByRole('group',{name:'My model',exact:true})).toContainText('Run recent-r');
  await expect(picker.getByLabel('Checkpoint',{exact:true}).locator('option[value="old-900"]')).toHaveCount(0);
  await expect(picker.getByLabel('Checkpoint',{exact:true}).locator('option[value="psi-checkpoint"]')).toHaveCount(0);
  await page.getByRole('link', { name: 'Fine-tune', exact: true }).click();
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
  await page.getByRole('link', { name: 'Quantize', exact: true }).click();
  const history = page.getByRole('region', { name: 'Quantization jobs', exact: true });
  await expect(history.locator('.job-history-entry')).toHaveCount(2);
  await expect(history.locator('[data-job-id="prior-quantization"]')).toContainText('SmolVLA · Q4');
  await expect(history.locator('[data-job-id="prior-quantization"]')).toContainText('Step 20');
  await expect(history.locator('[data-job-id="newer-quantization"]')).toContainText('SmolVLA · Q8');
  await expect(history.locator('.job-history-entry').first()).toContainText('Preparing worker');
  await expect(page.getByRole('group', { name: 'My model', exact: true })).toHaveCount(0);
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
  await expectSelectedPolicy(page,'recent-100',false);
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
    await page.getByRole('link', { name: stage, exact: true }).click();
    if (stage === 'Run') await page.getByRole('group', { name: 'Run mode', exact: true }).getByRole('button', { name: 'Check inference', exact: true }).click();
    await expect(page.getByRole('region', { name: stage === 'Evaluate' ? 'Evaluation jobs' : 'Run jobs', exact: true })).toBeVisible();
    await expect(page.getByRole('group', { name: 'My model', exact: true })).toHaveCount(0);
    await page.getByRole('button', { name: stage === 'Evaluate' ? 'New evaluation' : 'New run', exact: true }).click();
    await expectCompute(page,'gcp');
    await expectSelectedPolicy(page,'cloud-q8',false);
    await choosePolicy(page,'cloud-q8');
    await expect(page.getByText('Synthetic inputs · no task success score')).toBeVisible();
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

test('shows native build progress as a setup stage without presenting it as overall job completion', async ({ page }) => {
  await workspace(page, true, true, ({ jobs }) => {
    jobs.unshift({ id: 'compiling-run', project_id: projectId, kind: 'policy.evaluate', status: 'running', stage: 'compiling',
      created_at: '2026-09-26T14:00:00Z', updated_at: '2026-09-26T14:25:00Z',
      request: { operation: 'policy.evaluate', runtime_id: 'gcp', artifact_id: 'recent-100', evaluation: { mode: 'engine' } } });
  });
  await page.route('**/api/v1/jobs/compiling-run/events', route => route.fulfill({ json: [{
    sequence: 1, stage: 'compiling', message: 'Compiling native engine · build progress 35%',
    timestamp: '2026-09-26T14:25:00Z', data: { scope: 'native_build', build_percent: 35 },
  }] }));
  await page.getByRole('link', { name: 'Evaluate', exact: true }).click();
  const job = page.locator('.job-history-entry[data-job-id="compiling-run"]');
  await expect(job).toContainText('Compiling native engine');
  await job.click();
  const stage = page.getByRole('region', { name: 'Current job stage' });
  await expect(stage.getByRole('heading', { name: 'Compiling native engine', exact: true })).toBeVisible();
  await expect(stage).toContainText('build progress 35%');
  await expect(stage.getByRole('progressbar')).not.toHaveAttribute('value');
  await expect(page.locator('.workflow-job-technical')).toHaveJSProperty('open', false);
});


for (const stage of ['Evaluate', 'Run']) {
  test(`${stage} explains how to prepare a trained checkpoint when no GGUF is available`, async ({ page }) => {
    const { submitted, unexpected } = await workspace(page, true, true);
    await page.getByRole('link', { name: stage, exact: true }).click();
    if (stage === 'Run') await page.getByRole('group', { name: 'Run mode', exact: true }).getByRole('button', { name: 'Check inference', exact: true }).click();
    await page.getByRole('button', { name: stage === 'Evaluate' ? 'New evaluation' : 'New run', exact: true }).click();
    await expectCompute(page,'gcp');
    await expect(page.getByText('Quantize a SmolVLA checkpoint first.', { exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: stage === 'Evaluate' ? 'Start evaluation' : 'Reload and run', exact: true })).toBeDisabled();
    await page.getByRole('button', { name: 'Go to quantization', exact: true }).click();
    await expect(page.getByRole('region', { name: 'Quantization jobs', exact: true })).toBeVisible();
    await chooseTransformationModel(page, 'quantization', 'Choose LoRA checkpoint · recent-100');
    await expectSelectedPolicy(page,'recent-100');
    await expect(page.getByRole('button', { name: 'Start quantization', exact: true })).toBeEnabled();
    expect(submitted).toEqual([]);
    expect(unexpected).toEqual([]);
  });
}


test('quantization uses one compute picker and preserves compression and checkpoint choices across targets',async({page},testInfo)=>{
 const {artifacts,submitted,unexpected}=await workspace(page,false,false,({artifacts})=>{
  for(const artifact of artifacts) artifact.metadata.base_model={repository:'lerobot/smolvla_base'};
 });
 await page.route('**/api/v1/policy-options',route=>route.fulfill({json:{runtimes:[
  {id:'gcp-l4',label:'L4',accelerator:'L4',device:'cuda',provider:'gcp',execution:'skypilot',enabled:true,gpu_memory_mib:24576},
  {id:'gcp-a100-80',label:'A100',accelerator:'A100-80GB',device:'cuda',provider:'gcp',execution:'skypilot',enabled:true,gpu_memory_mib:81920},
  {id:'local-gpu',label:'Local RTX 3070',device:'cuda',provider:'local',execution:'native',enabled:true,gpu_memory_mib:8192},
  {id:'local-cpu',label:'Local CPU',device:'cpu',provider:'local',execution:'native',enabled:true},
  {id:'unavailable',label:'Unavailable GPU',device:'cuda',provider:'local',execution:'native',enabled:false}],sources:[],training_models:[],training_methods:[],default_training_method:'lora',quantization_defaults:{cuda:{language:'Q8_0',vision:null},cpu:{language:'Q8_0',vision:null},note:''}}}));
 await page.getByRole('link',{name:'Quantize',exact:true}).click();
 await chooseTransformationModel(page,'quantization','Choose LoRA checkpoint · recent-100');
 await expect(page.getByRole('combobox',{name:'Compute',exact:true})).toBeVisible();
 await expect(page.getByRole('group',{name:'Compute',exact:true})).toHaveCount(0);
 const memory=page.getByRole('region',{name:'Estimated quantization memory',exact:true});
 await expect(memory).toContainText('≈ 4 GiB');await expect(memory).toContainText('≈ 9 GiB');
 await page.getByRole('button',{name:'Help for Quantization memory estimate',exact:true}).focus();
 await expect(page.getByRole('tooltip').filter({hasText:'450M-parameter'})).toBeVisible();
 await choosePolicy(page,'recent-20');await choice(page,'Language compression','Q4_0').check();await choice(page,'Vision compression','Q8_0').check();
 await selectCompute(page,'Local RTX 3070');await expectCompute(page,'local-gpu');
 await expect(page.getByRole('button',{name:'Run quantization workflow',exact:true})).toBeDisabled();
 await expect(page.getByText('This policy is stored on Google Cloud. Choose a cloud target.',{exact:true})).toBeVisible();
 await page.getByRole('combobox',{name:'Compute',exact:true}).click();
 const menu=page.getByRole('listbox',{name:'Compute',exact:true});await expect(menu.getByRole('option')).toHaveCount(4);await expect(menu.getByRole('option',{name:'Local CPU',exact:true})).toBeVisible();
 await expect(menu.getByRole('option',{name:'A100',exact:true})).toContainText('80 GB');await menu.getByRole('option',{name:'A100',exact:true}).click();
 await expectSelectedPolicy(page,'recent-20');await expect(choice(page,'Language compression','Q4_0')).toBeChecked();await expect(choice(page,'Vision compression','Q8_0')).toBeChecked();
 await page.screenshot({path:testInfo.outputPath('quantization-components-compute.png'),fullPage:true});
 await page.getByRole('button',{name:'Start quantization',exact:true}).click();await expect.poll(()=>submitted.length).toBe(1);
 expect(submitted[0]).toMatchObject({operation:'policy.quantize',runtime_id:'gcp-a100-80',artifact_id:'recent-20',precision:{language:'Q4_0',vision:'Q8_0'}});
 expect(unexpected).toEqual([]);
});

test('old recommended preference resolves to the single explicit 8-bit language choice',async({page})=>{
 await page.addInitScript(id=>localStorage.setItem(`firebird.workflow.${id}`,JSON.stringify({precision:'recommended',vision:false})),projectId);
 const {submitted}=await workspace(page);await page.getByRole('link',{name:'Quantize',exact:true}).click();
 await chooseTransformationModel(page,'quantization','Choose LoRA checkpoint · recent-100');
 await expect(choice(page,'Language compression','Q8_0')).toBeChecked();await expect(choice(page,'Vision compression','source')).toBeChecked();
 await expect(page.getByRole('radio',{name:'Recommended',exact:true})).toHaveCount(0);
 await page.getByRole('button',{name:'Start quantization',exact:true}).click();await expect.poll(()=>submitted.length).toBe(1);
 expect(submitted[0].precision).toEqual({language:'Q8_0',vision:null});
});
