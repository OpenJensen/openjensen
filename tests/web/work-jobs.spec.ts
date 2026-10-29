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
    };
    return route.fulfill({ json: replies[path] ?? [] });
  });
  return { calls };
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
  await page.getByLabel('Severity', { exact: true }).selectOption('error');
  await expect(page.getByRole('list', { name: 'Log messages' })).toContainText('Model manifest validation failed.');
});

test('opening a historical scene selects that exact run in the specialized workspace', async ({ page }) => {
  await fixture(page);
  await page.goto('/jobs/');
  await page.locator('[data-job-id="scene:old"]').click();
  await page.getByRole('button', { name: 'Open workspace', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Episode 2', exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Episode 9', exact: true })).toHaveCount(0);
});

test('modules with previously separate run systems open the shared history first', async ({ page }) => {
  await fixture(page);
  await page.goto('/real-to-sim/');
  await expect(
    page.getByRole('group', { name: 'Module view' }).getByRole('button', { name: 'Jobs', exact: true }),
  ).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('heading', { name: 'New scene', exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'New job', exact: true }).click();
  await page.getByRole('button', { name: 'Real to sim', exact: true }).click();
  await page.getByRole('button', { name: 'Configure real to sim', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'New scene', exact: true })).toBeVisible();
});

test('shared job filters remain usable at a narrow viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await fixture(page);
  await page.goto('/jobs/');
  await page.getByLabel('Search jobs').fill('reconstruction');
  await expect(page.locator('[data-job-id="core:failed"]')).toHaveCount(0);
  await expect(page.locator('[data-job-id="scene:old"]')).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1),
  ).toBe(true);
});
