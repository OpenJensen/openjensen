import { selectProject } from './project-controls';
import { expect, test, type Page } from '@playwright/test';

function run(id: string, changes: Record<string, unknown> = {}) {
  return {
    schema_version: 1, run_id: id, label: `Synthetic ${id}`, cluster: 'synthetic-group', job_id: '7',
    collected_at: '2026-09-27T00:00:00Z', status: 'RUNNING', collection_error: null,
    logs: { isaac: `Isaac ${id} fixture`, vla: `VLA ${id} fixture` },
    outcomes: { rollout_completed: null, pickup_success: null, calibration: 'unverified' },
    age_seconds: 1, stale: false, ...changes,
  };
}
function feed(runs: ReturnType<typeof run>[], errors: { run_id: string | null; message: string }[] = []) {
  return { enabled: true, server_time: '2026-09-27T00:00:01Z', stale_after_seconds: 90, runs, errors };
}
async function openCloud(page: Page) {
  await page.goto('/');
  await page.getByRole('button', { name: 'Cloud runs', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'External observations' })).toBeVisible();
}

test('cloud monitor is disabled by default and never launches a job', async ({ page, request }) => {
  // Read the real test application, not an invented enabled response.
  expect((await (await request.get('/api/v1/cloud-runs')).json()).enabled).toBe(false);
  const posts: string[] = [];
  page.on('request', req => { if (req.method() === 'POST') posts.push(req.url()); });
  await openCloud(page);
  await expect(page.getByText('External rollout monitoring is not configured.', { exact: false })).toBeVisible();
  await expect(page.getByLabel('Cloud run', { exact: true })).toHaveCount(0);
  expect(posts).toEqual([]);
});

test('cloud log switching preserves explicit outcomes and safely displays bounded text', async ({ page }) => {
  const literal = '</pre><script>window.cloudInjected=true</script>';
  await page.route('**/api/v1/cloud-runs', route => route.fulfill({ json: feed([
    run('first', { status: 'SUCCEEDED', logs: { isaac: literal, vla: 'VLA first fixture' },
      outcomes: { rollout_completed: true, pickup_success: null, calibration: 'unverified' } }),
    run('second', { status: 'FAILED_SETUP', stale: true, age_seconds: 120,
      collection_error: 'Synthetic remote collection failure', logs: { isaac: 'Isaac second fixture', vla: 'x'.repeat(48 * 1024) } }),
    run('recent-error', { collection_error: 'Latest remote refresh failed' }),
  ]) }));
  await openCloud(page);
  await expect(page.getByRole('region', { name: 'Isaac log tail' })).toHaveText(literal);
  expect(await page.evaluate(() => 'cloudInjected' in window)).toBe(false);
  await expect(page.getByText('Primary task status', { exact: true }).locator('..')).toContainText('SUCCEEDED');
  await expect(page.getByText('Pickup success', { exact: true }).locator('..')).toContainText('Not reported');
  await expect(page.getByText('Calibration', { exact: true }).locator('..')).toContainText('unverified');
  await page.getByRole('button', { name: 'VLA logs', exact: true }).click();
  await expect(page.getByRole('region', { name: 'VLA log tail' })).toHaveText('VLA first fixture');
  await page.getByLabel('Cloud run', { exact: true }).selectOption('second');
  await expect(page.getByRole('region', { name: 'VLA log tail' })).toHaveText('x'.repeat(48 * 1024));
  await expect(page.getByText('Stale or unavailable observation', { exact: true })).toBeVisible();
  await expect(page.locator('.cloud-runs').getByRole('alert')).toContainText('Synthetic remote collection failure');
  await page.getByRole('button', { name: 'Isaac logs', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Isaac log tail' })).toHaveText('Isaac second fixture');
  const width = await page.evaluate(() => ({ viewport: document.documentElement.clientWidth, content: document.documentElement.scrollWidth }));
  expect(width.content).toBeLessThanOrEqual(width.viewport + 1);
  await page.getByLabel('Cloud run', { exact: true }).selectOption('recent-error');
  await expect(page.getByText('Last observation (refresh failed)', { exact: true })).toBeVisible();
});

test('cloud monitor polls updates and marks retained observations unavailable on API failure', async ({ page }) => {
  let state: 'initial' | 'updated' | 'offline' | 'invalid' = 'initial';
  await page.route('**/api/v1/cloud-runs', route => {
    if (state === 'offline') return route.fulfill({ status: 503, json: { detail: 'Synthetic API outage' } });
    if (state === 'invalid') return route.fulfill({ json: feed([], [{ run_id: 'newest', message: 'Snapshot contains invalid versioned JSON data' }]) });
    return route.fulfill({ json: feed([run('polling', { logs: { isaac: state, vla: '' } })]) });
  });
  await openCloud(page);
  await expect(page.getByRole('region', { name: 'Isaac log tail' })).toHaveText('initial');
  state = 'updated';
  await expect(page.getByRole('region', { name: 'Isaac log tail' })).toHaveText('updated', { timeout: 10_000 });
  state = 'offline';
  await expect(page.locator('.cloud-runs').getByRole('alert')).toContainText('Cloud updates are unavailable.', { timeout: 10_000 });
  await expect(page.getByRole('region', { name: 'Isaac log tail' })).toHaveText('updated');
  await expect(page.getByText('Stale or unavailable observation', { exact: true })).toBeVisible();
  state = 'invalid';
  await page.getByRole('button', { name: 'Refresh cloud runs', exact: true }).click();
  await expect(page.locator('.cloud-runs').getByRole('alert')).toContainText('Snapshot read error for newest');
  await expect(page.getByText('No external rollout observations have been published yet.')).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Isaac log tail' })).toHaveCount(0);
});

test('cloud monitor distinguishes no collection from an empty configured feed', async ({ page }) => {
  let empty = false;
  await page.route('**/api/v1/cloud-runs', route => route.fulfill({ json: feed(empty ? [] : [run('unobserved', {
    collected_at: null, age_seconds: null, stale: true, status: 'UNKNOWN',
    collection_error: 'Initial remote collection failed', logs: { isaac: '', vla: '' },
  })]) }));
  await openCloud(page);
  await expect(page.getByText('No successful remote collection yet.')).toBeVisible();
  await expect(page.locator('.cloud-runs').getByRole('alert')).toContainText('Initial remote collection failed');
  await expect(page.getByRole('region', { name: 'Isaac log tail' })).toContainText('No log lines collected');
  empty = true;
  await page.getByRole('button', { name: 'Refresh cloud runs', exact: true }).click();
  await expect(page.getByText('No external rollout observations have been published yet.')).toBeVisible();
  await expect(page.locator('.cloud-runs').getByRole('alert')).toHaveCount(0);
});

async function managedWorkspace(page: Page) {
  const timestamp = new Date().toISOString();
  const cloud = { project_id: 'fixture-cloud', region: 'us-central1', accelerator: 'L4', gpu_count: 1, disk_size_gb: 200, idle_minutes: 10 };
  const job = (id: string, project: string, target: object | null, status = 'succeeded') => ({ id, project_id: project, kind: 'policy.finetune', status, stage: 'operation', compute_target: target, created_at: timestamp, updated_at: timestamp, error: null, result: null, request: { operation: 'policy.finetune', runtime_id: target ? 'renamed-old-runtime' : 'skypilot-name-is-not-proof', training_method: 'full', training: { model_id: 'act', steps: 100 } } });
  const jobs = [job('cloud-finished', 'alpha', cloud), job('cpu-export-source', 'alpha', null), job('cloud-old-failed', 'alpha', cloud, 'failed')];
  const state = { offline: false, eventsOffline: false };
  const posts: string[] = [];
  const eventRequests: string[] = [];
  await page.route('**/api/v1/**', async route => {
    const req = route.request(), path = new URL(req.url()).pathname;
    if (req.method() !== 'GET') { posts.push(path); return route.fulfill({ status: 405, json: { detail: 'No writes in observer test' } }); }
    if (path === '/api/v1/projects/alpha/jobs') return route.fulfill(state.offline ? { status: 503, json: { detail: 'Fixture jobs outage' } } : { json: jobs });
    if (/\/jobs\/[^/]+\/events$/.test(path)) { eventRequests.push(path); return route.fulfill(state.eventsOffline ? { status: 503, json: { detail: 'Fixture events outage' } } : { json: [{ sequence: 1, stage: 'completed', timestamp, message: `Recorded ${path.split('/')[4]}`, data: {} }] }); }
    if (path.endsWith('/training')) return route.fulfill({ status: 404, json: { detail: 'Legacy fixture' } });
    const replies: Record<string, unknown> = {
      '/api/v1/health': { status: 'ok', version: 'test' }, '/api/v1/capabilities': [], '/api/v1/datasets': [],
      '/api/v1/projects': ['alpha', 'beta'].map(id => ({ id, name: id, created_at: timestamp })),
      '/api/v1/projects/beta/jobs': [job('beta-cloud', 'beta', cloud)],
      '/api/v1/projects/alpha/artifacts': [], '/api/v1/projects/beta/artifacts': [],
      '/api/v1/policy-options': { runtimes: [], sources: [], training_methods: [], training_models: [] },
      '/api/v1/cloud-runs': { ...feed([]), enabled: false },
    };
    return route.fulfill({ json: replies[path] ?? {} });
  });
  await openCloud(page);
  return { state, posts, eventRequests };
}

test('managed cloud jobs use persisted targets and open the exact training run', async ({ page }) => {
  const { posts } = await managedWorkspace(page);
  await expect(page.getByRole('heading', { name: 'Jobs' })).toBeVisible();
  await expect(page.getByLabel('Application cloud job', { exact: true })).toHaveValue('cloud-finished');
  await expect(page.getByLabel('Application cloud job', { exact: true }).locator('option')).toHaveCount(2);
  await expect(page.getByRole('region', { name: 'Application job event log' })).toContainText('Recorded cloud-finished');
  await expect(page.getByText('External rollout monitoring is not configured.', { exact: false })).toBeVisible();
  await page.getByLabel('Application cloud job', { exact: true }).selectOption('cloud-old-failed');
  await expect(page.getByRole('region', { name: 'Application job event log' })).toContainText('Recorded cloud-old-failed');
  await page.getByRole('button', { name: 'Open training job', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Back to jobs', exact: true })).toBeVisible();
  await expect(page.getByRole('article', { name: 'Training run monitor' })).toHaveAttribute('data-run-id', 'cloud-old-failed');
  expect(posts).toEqual([]);
});

test('cloud observation failure and project switch cannot show another projects job', async ({ page }) => {
  const { state, posts } = await managedWorkspace(page);
  await expect(page.getByRole('region', { name: 'Application job event log' })).toContainText('Recorded cloud-finished');
  state.eventsOffline = true;
  await page.getByRole('button', { name: 'Refresh application jobs' }).click();
  await expect(page.getByText('Job events are unavailable.', { exact: false })).toBeVisible();
  state.offline = true;
  await page.getByRole('button', { name: 'Refresh application jobs' }).click();
  await expect(page.getByText('Application job updates are unavailable.', { exact: false })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Open training job' })).toBeDisabled();
  state.eventsOffline = false;
  await selectProject(page, 'beta');
  await expect(page.getByLabel('Application cloud job', { exact: true })).toHaveValue('beta-cloud');
  await expect(page.getByRole('region', { name: 'Application job event log' })).toContainText('Recorded beta-cloud');
  await expect(page.getByRole('region', { name: 'Application job event log' })).not.toContainText('cloud-finished');
  expect(posts).toEqual([]);
});
