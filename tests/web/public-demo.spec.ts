import { expect, test } from '@playwright/test';

test('public link opens saved training and loss metrics while owner controls stay disabled', async ({ page }) => {
  const time = '2026-09-27T04:00:00Z';
  const project = { id: 'demo-project', name: 'Real training examples', created_at: time };
  const job = { compute_target: { accelerator: 'L4', region: 'us-central1' }, id: 'trained-model', project_id: project.id, kind: 'policy.finetune', status: 'succeeded', stage: 'complete', created_at: time, updated_at: time,
    request: { operation: 'policy.finetune', runtime_id: 'gcp-l4', training_method: 'lora', training: { model_id: 'lerobot/smolvla_base', steps: 100 } },
    result: { decision: 'completed', reports: [{ steps: 100 }], artifacts: [] } };
  const metrics = [
    { step: 0, train_loss: 0.8, validation_loss: 0.9, learning_rate: 0.0001, elapsed_seconds: 0, timestamp: time },
    { step: 100, train_loss: 0.3, validation_loss: 0.1, learning_rate: 0.00001, elapsed_seconds: 120, timestamp: time },
  ];
  const requests: string[] = [];
  const failures: string[] = [];
  page.on('pageerror', error => failures.push(error.message));
  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname.replace('/firebird/api/v1', '');
    requests.push(`${request.method()} ${path}`);
    const replies: Record<string, unknown> = {
      '/health': { status: 'ok', version: 'test' }, '/projects': [project], '/capabilities': [],
      '/policy-options': { runtimes: [], sources: [], training_methods: [{ id: 'lora', label: 'LoRA' }], default_training_method: 'lora', quantization_defaults: { cpu: { language: 'Q8_0', vision: null }, cuda: { language: 'Q8_0', vision: null }, note: '' } },
      [`/projects/${project.id}/jobs`]: [job], [`/projects/${project.id}/artifacts`]: [],
      '/cloud-runs': { enabled: false, runs: [], errors: [], stale_after_seconds: 90 },
      '/jobs/trained-model/events': [{ sequence: 1, stage: 'training', message: 'Optimizer step 100 completed', timestamp: time, data: {} }],
      '/jobs/trained-model/training': { job_id: job.id, status: 'succeeded', phase: 'complete', completed_steps: 100, total_steps: 100, percent: 100, current_action: 'Training complete', metrics, latest: metrics[1], checkpoints: [{ step: 100, name: 'checkpoint-100', timestamp: time }], events: [], logs: [], elapsed_seconds: 120, wall_seconds: 150, reproducibility: {} },
    };
    await route.fulfill({ status: path in replies && request.method() === 'GET' ? 200 : 403, json: replies[path] ?? { detail: 'Owner access required' } });
  });
  await page.goto('/firebird/');
  await expect(page.getByRole('heading', { name: 'Fine-tune', exact: true })).toBeVisible();
  await expect(page.getByText('Inspect datasets and explore real training runs')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Start a new fine-tuning' })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Settings & diagnostics', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Create project', exact: true })).toBeHidden();
  await page.locator('[data-job-id="trained-model"]').click();
  await expect(page.getByRole('heading', { name: 'Loss over time' })).toBeVisible();
  await expect(page.getByText('100 / 100 optimizer steps')).toBeVisible();
  await expect(page.locator('.training-metrics')).toContainText('0.3');
  await expect(page.locator('.training-metrics')).toContainText('0.1');
  await expect(page.getByRole('button', { name: 'Diagnostics', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Quantize', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Quantization jobs', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'New quantization', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Cloud runs', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Application cloud jobs' })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Application job event log' })).toContainText('Optimizer step 100 completed');
  await page.getByRole('button', { name: 'Open training job' }).click();
  await expect(page.getByRole('heading', { name: 'Loss over time' })).toBeVisible();
  await expect(page.getByText('100 / 100 optimizer steps')).toBeVisible();
  expect(requests.every(request => request.startsWith('GET '))).toBe(true);
  expect(requests.some(request => /cloud-connections|compute-settings|huggingface-connection/.test(request))).toBe(false);
  expect(failures).toEqual([]);
});

test('public visitor can inspect a Hugging Face dataset and see its metadata', async ({ page }) => {
  const time = '2026-09-27T04:00:00Z';
  const project = { id: 'inspection-demo', name: 'Dataset demo', created_at: time };
  const jobs: unknown[] = [];
  const posts: unknown[] = [];
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname.replace('/firebird/api/v1', '');
    if (request.method() === 'POST' && path === `/projects/${project.id}/intakes`) {
      const body = request.postDataJSON(); posts.push(body);
      const job = { id: 'dataset-result', project_id: project.id, kind: 'dataset.inspect', status: 'succeeded', request: body, created_at: time, updated_at: time,
        result: { source: 'huggingface', repo_id: body.repo_id, revision: 'a'.repeat(40), format: 'lerobot_v3', inspection_scope: 'metadata_only', total_episodes: 30, total_frames: 4500, fps: 30, robot_type: 'so_follower', features: { action: { dtype: 'float32', shape: [6] }, 'observation.state': { dtype: 'float32', shape: [6] } }, warnings: [], metadata_sha256: 'b'.repeat(64), inspected_at: time } };
      jobs.unshift(job); await route.fulfill({ status: 202, json: job }); return;
    }
    const replies: Record<string, unknown> = {
      '/health': { status: 'ok' }, '/projects': [project], '/capabilities': [],
      '/policy-options': { runtimes: [], sources: [], training_methods: [] },
      [`/projects/${project.id}/jobs`]: jobs, [`/projects/${project.id}/artifacts`]: [],
      '/jobs/dataset-result/episodes': { repo_id: 'codywang/so101_pickup_test', revision: 'a'.repeat(40), total_episodes: 30, offset: 0, limit: 6, episodes: [], warnings: [] },
    };
    await route.fulfill({ status: path in replies ? 200 : 404, json: replies[path] ?? {} });
  });
  await page.goto('/firebird/');
  await page.getByRole('button', { name: 'Dataset', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Inspect dataset', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'Inspect dataset', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Dataset inspection', exact: true })).toBeVisible();
  await expect(page.locator('.dataset-facts')).toContainText('4,500');
  await expect(page.locator('.dataset-facts')).toContainText('so_follower');
  expect(posts).toHaveLength(1);
  expect(posts[0]).toMatchObject({ source: 'huggingface', repo_id: 'codywang/so101_pickup_test' });
  expect(errors).toEqual([]);
});
