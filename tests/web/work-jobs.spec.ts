import { expect, test, type Page } from '@playwright/test';
import { installedModules } from './module-fixtures';
import { fulfillApplicationUpdateRead } from './application-update-controls';

const created = '2026-01-01T12:00:00+00:00';
function job(id: string, fields: Record<string, unknown> = {}) {
  return {
    id: `core:${id}`,
    source_id: id,
    project_id: 'project',
    operation: 'policy.quantize',
    module: 'quantization',
    title: 'Packed policy',
    status: 'failed',
    source_status: 'failed',
    created_at: created,
    updated_at: created,
    stage: 'loading',
    message: 'Worker failed during model loading.',
    execution: { adapter: 'local', infrastructure: 'local', target_id: 'local', label: 'Local worker' },
    inputs: {},
    result: null,
    outputs: [],
    parent_ids: [],
    can_cancel: false,
    events: [{ timestamp: created, stage: 'loading', message: 'Loading the input model.' }],
    ...fields,
  };
}
async function fixture(page: Page) {
  const calls: string[] = [];
  const items = [
    job('failed'),
    job('scene-old', {
      id: 'scene:old',
      source_id: 'old',
      operation: 'scene.generate',
      module: 'real-to-sim',
      title: 'Earlier reconstruction',
      status: 'waiting',
      source_status: 'needs_review',
    }),
  ];
  await page.route('**/api/v1/**', async (route) => {
    if (await fulfillApplicationUpdateRead(route)) return;
    const path = decodeURIComponent(new URL(route.request().url()).pathname);
    calls.push(path);
    if (path.endsWith('/failure'))
      return route.fulfill({
        json: {
          code: 'WORKER_FAILED',
          stage: 'loading',
          message: 'The model could not be loaded.',
          cause: 'Model manifest validation failed.',
          exit_code: 1,
          retryable: false,
          cleanup: 'succeeded',
          attempt_id: 'attempt-1',
          timestamp: created,
        },
      });
    if (path.endsWith('/logs'))
      return route.fulfill({
        json: {
          records: [
            {
              sequence: 1,
              job_id: 'failed',
              attempt_id: 'attempt-1',
              timestamp: created,
              kind: 'log',
              component: 'worker',
              stage: 'loading',
              severity: 'error',
              stream: 'stderr',
              message: 'Model manifest validation failed.',
              data: {},
            },
          ],
          next_cursor: 1,
          has_more: false,
          retained_from: 1,
          truncated: false,
          available: true,
        },
      });
    if (path === '/api/v1/projects/project/work-jobs') {
      const query = new URL(route.request().url()).searchParams.get('q') ?? '';
      return route.fulfill({ json: items.filter((item) => item.title.toLowerCase().includes(query.toLowerCase())) });
    }
    const selected = items.find((item) => path === `/api/v1/projects/project/work-jobs/${item.id}`);
    if (selected) return route.fulfill({ json: selected });
    if (path === '/api/v1/projects/project/scenes')
      return route.fulfill({
        json: [
          {
            id: 'new',
            status: 'needs_review',
            stage: 'review',
            attempt: 1,
            message: 'Newer run',
            request: { episode_index: 9 },
            gates: {},
            events: [],
            artifacts: [],
            usage: {},
          },
          {
            id: 'old',
            status: 'needs_review',
            stage: 'review',
            attempt: 1,
            message: 'Earlier run',
            request: { episode_index: 2 },
            gates: {},
            events: [],
            artifacts: [],
            usage: {},
          },
        ],
      });
    const replies: Record<string, unknown> = {
      '/api/v1/health': { status: 'ok', version: 'test' },
      '/api/v1/capabilities': [],
      '/api/v1/modules': installedModules,
      '/api/v1/projects': [{ id: 'project', name: 'Workspace', created_at: created }],
      '/api/v1/projects/project/jobs': [],
      '/api/v1/job-operations': [
        { id: 'policy.quantize', title: 'Quantize', module: 'quantization', path: '/quantization/', submission: 'job' },
        { id: 'scene.generate', title: 'Real to sim', module: 'real-to-sim', path: '/real-to-sim/', submission: 'job' },
      ],
      '/api/v1/compute-targets': { schema_version: 1, targets: [], runtime_options: [], connection_refs: [] },
      '/api/v1/scene-generation/options': {
        models: [],
        default_model: '',
        configured: false,
        codex_available: false,
        builder_available: false,
        message: 'Configure a scene worker.',
        supported_repo: 'example/data',
        supported_revision: 'revision',
      },
      '/api/v1/datasets': [],
      '/api/v1/projects/project/artifacts': [],
      '/api/v1/policy-options': {
        runtimes: [],
        sources: [],
        training_models: [],
        training_methods: [],
        default_training_method: 'lora',
        quantization_defaults: {
          cpu: { language: 'Q8_0', vision: null },
          cuda: { language: 'Q8_0', vision: null },
          note: '',
        },
      },
    };
    return route.fulfill({ json: replies[path] ?? [] });
  });
  return { calls, items };
}

test('shared job detail explains failure and exposes correlated logs', async ({ page }) => {
  await fixture(page);
  await page.goto('/jobs/');
  await page.locator('[data-job-id="core:failed"]').click();
  await expect(page.getByRole('region', { name: 'Failure details' })).toContainText('The model could not be loaded.');
  await expect(page.getByRole('region', { name: 'Failure details' })).toContainText('succeeded');
  await page.getByRole('button', { name: 'Logs', exact: true }).click();
  await expect(page.getByRole('list', { name: 'Log messages' })).toContainText('Model manifest validation failed.');
  await expect(page.getByRole('list', { name: 'Log messages' })).toContainText('stderr');
  await expect(page.getByRole('link', { name: 'Download diagnostics' }).last()).toHaveAttribute(
    'href',
    /diagnostics\.zip$/,
  );
  await page.getByLabel('Follow live logs').uncheck();
  await page.getByRole('combobox', { name: 'Severity', exact: true }).selectOption('error');
  await expect(page.getByRole('list', { name: 'Log messages' })).toContainText('Model manifest validation failed.');
});

test('opening a historical scene selects that exact saved result', async ({ page }) => {
  await fixture(page);
  await page.goto('/jobs/');
  await page.locator('[data-job-id="scene:old"]').click();
  await page.getByRole('button', { name: 'Review scene', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Episode 2', exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Episode 9', exact: true })).toHaveCount(0);
});

test('modules with previously separate run systems open the shared history first', async ({ page }) => {
  await fixture(page);
  await page.goto('/real-to-sim/');
  await expect(page.getByRole('group', { name: 'Module view' })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Workspace', exact: true })).toHaveCount(0);
  await expect(page.getByRole('combobox', { name: 'Scene dataset', exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'New scene', exact: true }).click();
  await expect(page.getByRole('combobox', { name: 'Scene dataset', exact: true })).toBeVisible();
});

test('shared job filters remain usable at a narrow viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await fixture(page);
  await page.goto('/jobs/');
  await expect(page.getByRole('button', { name: 'Current project', exact: true })).toHaveAttribute(
    'data-project-id',
    'project',
  );
  await expect(page.locator('[data-job-id="core:failed"]')).toBeVisible();
  const filtered = page.waitForResponse((response) => {
    const url = new URL(response.url());
    return url.pathname === '/api/v1/projects/project/work-jobs' && url.searchParams.get('q') === 'reconstruction';
  });
  await page.getByLabel('Search jobs').fill('reconstruction');
  await filtered;
  await expect(page.getByLabel('Search jobs')).toHaveValue('reconstruction');
  await expect(page.locator('[data-job-id="core:failed"]')).toHaveCount(0);
  await expect(page.locator('[data-job-id="scene:old"]')).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1),
  ).toBe(true);
});

for (const [path, history, action, setup] of [
  ['/training/', 'Fine-tuning jobs', 'Start a new fine-tuning', 'Training setup'],
  ['/distillation/', 'Distillation jobs', 'Start a new distillation', 'Distillation setup'],
  ['/quantization/', 'Quantization jobs', 'Start a new quantization', 'Quantization setup'],
] as const) {
  test(path + ' opens jobs and its New action opens setup directly', async ({ page }) => {
    const { calls } = await fixture(page);
    await page.goto(path);
    await expect(page.getByRole('region', { name: history, exact: true })).toBeVisible();
    await expect(page.getByRole('group', { name: 'Module view' })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Workspace', exact: true })).toHaveCount(0);
    await page.getByRole('button', { name: action, exact: true }).click();
    if (setup === 'Training setup') await expect(page.getByRole('navigation', { name: setup })).toBeVisible();
    else await expect(page.getByRole('region', { name: setup, exact: true })).toBeVisible();
    await expect(page.getByRole('region', { name: history, exact: true })).toHaveCount(0);
    expect(calls.some((value) => value.endsWith('/policy-jobs'))).toBe(false);
  });
}

test('all-jobs New opens setup directly and compute is selected inside that setup', async ({ page }) => {
  const { calls } = await fixture(page);
  await page.route('**/api/v1/compute-targets', (route) =>
    route.fulfill({
      json: {
        schema_version: 1,
        connection_refs: [],
        runtime_options: [],
        targets: [
          {
            definition: {
              id: 'policy-pool',
              label: 'Policy pool',
              adapter: 'skypilot',
              infrastructure: 'ssh',
              enabled: true,
              runtime_id: 'portable-policy',
              connection_ref: null,
              location: 'team-pool',
              partition: null,
              account: null,
              qos: null,
              operations: ['policy.finetune'],
              resources: {
                cpus: 2,
                memory_gib: 4,
                accelerator: 'L4',
                gpu_count: 1,
                disk_gib: 100,
                timeout_seconds: 7200,
                concurrency: 1,
              },
            },
            operations: ['policy.finetune'],
            status: 'configured',
            message: 'Ready for a job.',
            launchable: true,
            checked_at: null,
            sha256: 'a'.repeat(64),
          },
        ],
      },
    }),
  );
  await page.goto('/jobs/');
  await page.getByRole('button', { name: 'New job', exact: true }).click();
  await page.getByRole('button', { name: 'Fine-tune', exact: true }).click();
  await expect(page.getByRole('navigation', { name: 'Training setup' })).toBeVisible();
  await page.getByRole('combobox', { name: 'Compute target', exact: true }).selectOption('policy-pool');
  await expect(page.getByRole('region', { name: 'Selected job compute' })).toContainText('Policy pool');
  await expect(page.getByRole('button', { name: 'Configure fine-tune', exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Workspace', exact: true })).toHaveCount(0);
  expect(calls.some((value) => value.endsWith('/policy-jobs'))).toBe(false);
});

async function sceneResultsFixture(page: Page, status = 'succeeded') {
  const state = await fixture(page);
  state.items.push(
    job('scene-result', {
      operation: 'scene.generate',
      module: 'real-to-sim',
      title: 'Dataset reconstruction',
      status,
      source_status: status,
      stage: status === 'running' ? 'simulating' : 'complete',
      message: 'Recorded episode simulation',
      outputs: [{ label: 'Source frame', url: '/api/v1/jobs/scene-result/scene/artifact?name=source-0.jpg' }],
    }),
  );
  await page.route('**/api/v1/projects/project/work-jobs/*/failure', (route) => route.fulfill({ json: null }));
  return state;
}

test('shared scene job shows registered environment images and replay inline', async ({ page }) => {
  await sceneResultsFixture(page);
  const names = ['simulation/front.png', 'simulation/side.png', 'simulation/overview.png'];
  const source = Array.from({ length: 20 }, (_, index) => ({
    role: 'source_image',
    path: `source-${index}.jpg`,
    label: `Source frame ${index}`,
  }));
  const media = [
    { role: 'environment_image', path: 'not-registered.png', label: 'Unregistered output' },
    { role: 'environment_image', path: '../private.png', label: 'Outside output' },
    ...names.map((path, index) => ({ role: 'environment_image', path, label: `Environment view ${index + 1}` })),
    { role: 'episode_replay', path: 'simulation/replay.mp4', label: 'Episode 7 replay', episode_index: 7 },
    ...source,
  ];
  await page.route('**/api/v1/jobs/scene-result/scene', (route) =>
    route.fulfill({
      json: {
        id: 'generated-scene',
        job_id: 'scene-result',
        project_id: 'project',
        artifacts: [
          ...names,
          'simulation/replay.mp4',
          'scene-bundle.zip',
          '../private.png',
          ...source.map((item) => item.path),
        ],
        media,
        gates: { isaac_runtime: true, calibrated_replay: false },
        source: { resource_cleanup: 'cluster_removed', replay_episode: 7 },
      },
    }),
  );
  const mediaRequests: string[] = [];
  await page.route('**/api/v1/jobs/scene-result/scene/artifact?*', (route) => {
    mediaRequests.push(new URL(route.request().url()).searchParams.get('name')!);
    return route.fulfill({
      contentType: 'image/png',
      body: Buffer.from(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
        'base64',
      ),
    });
  });
  await page.goto('/real-to-sim/');
  await page.locator('[data-job-id="core:scene-result"]').click();
  const images = page.getByRole('region', { name: 'Environment images', exact: true });
  await expect(images.getByRole('img')).toHaveCount(3);
  for (let index = 0; index < names.length; index++) {
    const url = await images
      .getByRole('img', { name: `Environment view ${index + 1}`, exact: true })
      .getAttribute('src');
    expect(new URL(url!, page.url()).pathname).toBe('/api/v1/jobs/scene-result/scene/artifact');
    expect(new URL(url!, page.url()).searchParams.get('name')).toBe(names[index]);
  }
  const replay = page.getByLabel('Episode 7 replay', { exact: true });
  await expect(replay).toHaveAttribute('src', /\/jobs\/scene-result\/scene\/artifact\?name=simulation%2Freplay.mp4$/);
  await expect(replay).toHaveAttribute('controls', '');
  await expect(replay).toHaveAttribute('preload', 'metadata');
  await expect(page.getByText('Calibration unverified', { exact: true })).toBeVisible();
  await expect(page.getByRole('img', { name: /Source frame/ })).toHaveCount(0);
  await expect(page.getByRole('img', { name: 'Unregistered output', exact: true })).toHaveCount(0);
  await expect(page.getByRole('link', { name: 'Source frame', exact: true })).not.toBeVisible();
  await page.getByText('Simulation details', { exact: true }).click();
  await expect(page.getByText('Released', { exact: true })).toBeVisible();
  expect(mediaRequests.every((name) => names.includes(name) || name === 'simulation/replay.mp4')).toBe(true);
});

test('a queued scene result stays pending and terminal transition fetches its final media', async ({ page }) => {
  const state = await sceneResultsFixture(page, 'running');
  let ready = false;
  await page.route('**/api/v1/jobs/scene-result/scene', (route) =>
    ready
      ? route.fulfill({
          json: {
            id: 'generated-scene',
            job_id: 'scene-result',
            project_id: 'project',
            artifacts: ['simulation/front.png'],
            media: [{ role: 'environment_image', path: 'simulation/front.png', label: 'Finished environment' }],
            gates: { isaac_runtime: true },
            source: { resource_cleanup: 'cluster_removed' },
          },
        })
      : route.fulfill({ status: 404, json: { detail: 'Worker queued' } }),
  );
  await page.goto('/jobs/');
  await page.locator('[data-job-id="core:scene-result"]').click();
  await expect(page.getByRole('status').filter({ hasText: 'Simulation results pending' })).toBeVisible();
  await expect(
    page.getByRole('region', { name: 'Job details', exact: true }).locator('.notice[role="alert"]'),
  ).toHaveCount(0);
  ready = true;
  const item = state.items.find((item) => item.id === 'core:scene-result')!;
  item.status = 'succeeded';
  item.source_status = 'succeeded';
  await expect(page.getByRole('img', { name: 'Finished environment', exact: true })).toBeVisible({ timeout: 10000 });
  await expect(page.getByText('Simulation results pending', { exact: true })).toHaveCount(0);
});

test('a scene result from another job cannot expose its media in shared results', async ({ page }) => {
  await sceneResultsFixture(page);
  await page.route('**/api/v1/jobs/scene-result/scene', (route) =>
    route.fulfill({
      json: {
        id: 'other-scene',
        job_id: 'different-job',
        project_id: 'project',
        artifacts: ['simulation/front.png'],
        media: [{ role: 'environment_image', path: 'simulation/front.png', label: 'Other job environment' }],
      },
    }),
  );
  await page.goto('/jobs/');
  await page.locator('[data-job-id="core:scene-result"]').click();
  await expect(
    page.getByRole('region', { name: 'Job details', exact: true }).locator('.notice[role="alert"]'),
  ).toContainText('Simulation results unavailable');
  await expect(page.getByRole('img', { name: 'Other job environment', exact: true })).toHaveCount(0);
});

test('job filters wait for project context and submit the query after it becomes ready', async ({ page }) => {
  await fixture(page);
  let release: () => void = () => {};
  const pending = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route('**/api/v1/projects', async (route) => {
    await pending;
    return route.fulfill({ json: [{ id: 'project', name: 'Workspace', created_at: created }] });
  });
  await page.goto('/jobs/');
  const search = page.getByRole('textbox', { name: 'Search jobs', exact: true });
  try {
    await expect(search).toBeDisabled();
    await expect(page.getByRole('combobox', { name: 'Module', exact: true })).toBeDisabled();
    await expect(page.getByRole('combobox', { name: 'Status', exact: true })).toBeDisabled();
    await expect(page.getByRole('button', { name: 'New job', exact: true })).toBeDisabled();
    await expect(search).toHaveValue('');
  } finally {
    release();
  }
  await expect(page.getByRole('button', { name: 'Current project', exact: true })).toHaveAttribute(
    'data-project-id',
    'project',
  );
  await expect(page.locator('[data-job-id="core:failed"]')).toBeVisible();
  await expect(search).toBeEnabled();
  await expect(page.getByRole('combobox', { name: 'Module', exact: true })).toBeEnabled();
  await expect(page.getByRole('combobox', { name: 'Status', exact: true })).toBeEnabled();
  const filtered = page.waitForResponse((response) => {
    const url = new URL(response.url());
    return url.pathname === '/api/v1/projects/project/work-jobs' && url.searchParams.get('q') === 'reconstruction';
  });
  await search.fill('reconstruction');
  await filtered;
  await expect(search).toHaveValue('reconstruction');
  await expect(page.locator('[data-job-id="core:failed"]')).toHaveCount(0);
  await expect(page.locator('[data-job-id="scene:old"]')).toBeVisible();
});
