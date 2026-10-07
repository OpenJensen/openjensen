import { expect, test, type Page } from '@playwright/test';

const projectId = 'augmentation-fixture';
const timestamp = '2026-09-26T12:00:00Z';
const revision = 'a'.repeat(40);
const camera = 'observation.images.front';

function inspectedDataset(id = 'inspected-video', repo = 'fixture/robot') {
  return {
    id, project_id: projectId, kind: 'dataset.inspect', status: 'succeeded',
    request: { source: 'huggingface', repo_id: repo, revision: 'main' },
    created_at: timestamp, updated_at: timestamp,
    result: {
      source: 'huggingface', repo_id: repo, revision, format: 'lerobot_v3',
      inspection_scope: 'metadata_only', total_episodes: 12, total_frames: 1800, fps: 30,
      features: {
        [camera]: { dtype: 'video', shape: [480, 640, 3] },
        'observation.images.side': { dtype: 'video', shape: [480, 640, 3] },
        'observation.images.wrist': { dtype: 'image', shape: [480, 640, 3] },
        action: { dtype: 'float32', shape: [6] }, 'observation.state': { dtype: 'float32', shape: [6] },
      },
      metadata_sha256: 'b'.repeat(64), inspected_at: timestamp, warnings: [],
    },
  };
}

function resultFor(body: Record<string, any>) {
  return {
    operation: 'dataset.augment', model: 'gemini-omni-1.1-flash', prompt: 'Preserve robot motion. Use soft side lighting.',
    source_job_id: body.source_job_id, repo_id: 'fixture/robot', revision, metadata_sha256: 'b'.repeat(64),
    review_required: true, warnings: ['Review visual changes against the original actions.'],
    clips: body.episode_indices.map((episode: number, index: number) => ({
      index, episode_index: episode, camera_key: body.camera_key,
      source_start_seconds: body.start_seconds, source_end_seconds: body.start_seconds + body.duration_seconds,
      input_sha256: 'c'.repeat(64), output_sha256: 'd'.repeat(64), interaction_id: 'mock-interaction',
    })),
  };
}

async function mockWorkspace(page: Page, config: { configured?: boolean; empty?: boolean; complete?: boolean; failFirst?: boolean; cloud?: boolean; projectsReady?: Promise<void>; jobsReady?: Promise<void>; failProjects?: () => boolean; submissionMode?: 'lost-ack' | 'lost-before-commit' | 'unsupported' | 'wrong-ack' | 'rejected' | 'delayed-ack'; priorRuns?: boolean } = {}) {
  let releaseAck!: () => void;
  const ackReady = new Promise<void>(resolve => { releaseAck = resolve; });
  const submitted: Record<string, any>[] = [];
  const keys: string[] = [];
  const bindings = new Map<string, Record<string, any>>();
  const lookups: string[] = [];
  const unexpected: string[] = [];
  const cancelled: string[] = [];
  const jobs: Record<string, any>[] = config.empty ? [] : [inspectedDataset(),
    { ...inspectedDataset('failed-inspection'), status: 'failed', result: null },
    { ...inspectedDataset('image-only'), result: { ...inspectedDataset().result, features: { 'observation.images.wrist': { dtype: 'image', shape: [480, 640, 3] } } } },
    { ...inspectedDataset('local-inspection'), result: { ...inspectedDataset().result, source: 'local' } },
  ];
  if (config.priorRuns) for (let i = 1; i <= 2; i++) {
    const request = { operation: 'dataset.augment', source_job_id: 'inspected-video', episode_indices: [i], camera_key: camera,
      preset: 'lighting', prompt: '', start_seconds: 0, duration_seconds: 5 };
    jobs.unshift({ id: `previous-${i}`, project_id: projectId, kind: 'dataset.augment', status: 'succeeded',
      request, created_at: `2026-09-26T1${i}:00:00Z`, updated_at: timestamp, result: resultFor(request) });
  }
  let optionsAttempts = 0;
  // Intercept all application requests, including every write. The browser
  // suite cannot invoke a model or upload media to a real provider.
  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const submissionPath = `/api/v1/projects/${projectId}/submissions/`;
    if (request.method() === 'GET' && path.startsWith(submissionPath)) {
      const key = decodeURIComponent(path.slice(submissionPath.length));
      lookups.push(key);
      const headers = config.submissionMode === 'unsupported' ? {} : { 'Idempotency-Key': key, 'Cache-Control': 'no-store' };
      await route.fulfill({ status: bindings.has(key) ? 200 : 404, headers,
        json: bindings.get(key) ?? { detail: 'No accepted submission found for this project, operation and key' } });
      return;
    }
    if (request.method() === 'POST' && path === `/api/v1/projects/${projectId}/augmentations`) {
      const body = request.postDataJSON();
      submitted.push(body);
      const key = request.headers()['idempotency-key'];
      keys.push(key);
      if (submitted.length === 1 && config.submissionMode === 'lost-before-commit') { await route.abort('connectionfailed'); return; }
      if (submitted.length === 1 && config.submissionMode === 'rejected') { await route.fulfill({ status: 422, json: { detail: 'Synthetic recipe rejection before admission.' } }); return; }
      const job = {
        id: `augmentation-${submitted.length}`, project_id: projectId, kind: 'dataset.augment',
        request: body, status: config.complete ? 'succeeded' : 'running', stage: 'generating_clips',
        created_at: '2026-09-26T14:00:00Z', updated_at: timestamp, error: null,
        result: config.complete ? resultFor(body) : null,
      };
      jobs.unshift(job);
      bindings.set(key, job);
      if (submitted.length === 1 && config.submissionMode === 'lost-ack') { await route.abort('connectionfailed'); return; }
      if (config.submissionMode === 'delayed-ack') await ackReady;
      await route.fulfill({ status: 202, headers: { 'Idempotency-Key': key, 'Cache-Control': 'no-store' }, json: config.submissionMode === 'wrong-ack' ? { ...job, project_id: 'another-project' } : job });
      return;
    }
    if (request.method() === 'POST' && /^\/api\/v1\/jobs\/[^/]+\/cancel$/.test(path)) {
      const id = path.split('/')[4];
      cancelled.push(id);
      const job = jobs.find(item => item.id === id)!;
      job.status = 'cancelled';
      await route.fulfill({ json: job });
      return;
    }
    if (request.method() === 'GET') {
      if (path === '/api/v1/projects') {
        await config.projectsReady;
        if (config.failProjects?.()) {
          await route.fulfill({ status: 503, json: { detail: 'Projects temporarily unavailable.' } });
          return;
        }
      }
      if (path === `/api/v1/projects/${projectId}/jobs`) await config.jobsReady;
      if (path === '/api/v1/augmentation-options' && ++optionsAttempts === 1 && config.failFirst) {
        await route.fulfill({ status: 503, json: { detail: 'Augmentation setup temporarily unavailable.' } });
        return;
      }
      const responses: Record<string, unknown> = {
        '/api/v1/health': { status: 'ok', version: 'browser-fixture' },
        '/api/v1/capabilities': [], '/api/v1/datasets': [],
        '/api/v1/projects': [{ id: projectId, name: 'Augmentation review', created_at: timestamp }],
        [`/api/v1/projects/${projectId}/jobs`]: jobs,
        '/api/v1/augmentation-options': {
          configured: config.configured ?? true, model: 'gemini-omni-1.1-flash', max_clips: 4, max_duration_seconds: 10,
          auth_mode: config.cloud ? 'google_cloud' : 'gemini_api_key',
          google_cloud_project: config.cloud ? 'robotics-demo' : null,
          auth_message: config.cloud ? 'Google Cloud · robotics-demo · uses your Google Cloud login, without an API key. Model access and billing are checked when a run starts.' : null,
          setup_message: config.configured === false ? 'Set GEMINI_API_KEY on the application server and install ffmpeg and ffprobe.' : null,
          presets: [
            { id: 'lighting', label: 'Lighting', prompt: 'Use soft side lighting.' },
            { id: 'texture', label: 'Textures', prompt: 'Change the tabletop texture.' },
            { id: 'custom', label: 'Custom', prompt: '' },
          ],
        },
      };
      if (path in responses) { await route.fulfill({ json: responses[path] }); return; }
      if (/\/augmentation\/clips\/\d+$/.test(path)) {
        // Empty media deliberately exercises the accessible fallback links.
        await route.fulfill({ status: 200, contentType: 'video/mp4', body: '' });
        return;
      }
    }
    unexpected.push(`${request.method()} ${path}`);
    await route.fulfill({ status: 405, json: { detail: 'Blocked by augmentation browser fixture.' } });
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Augmentation', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Augmentation', exact: true, level: 1 })).toBeVisible();
  await expect(page.getByRole('group', { name: 'Generator', exact: true })).toContainText('Gemini Omni');
  return { submitted, unexpected, jobs, cancelled, keys, lookups, releaseAck, optionsAttempts: () => optionsAttempts };
}

async function noOverflow(page: Page) {
  const dimensions = await page.evaluate(() => ({ width: document.documentElement.clientWidth, scroll: document.documentElement.scrollWidth }));
  expect(dimensions.scroll).toBeLessThanOrEqual(dimensions.width + 1);
}

test('submits selected clips and texture instructions, then compares and downloads the completed result', async ({ page }, testInfo) => {
  const { submitted, unexpected } = await mockWorkspace(page, { complete: true });
  await expect(page.getByRole('group', { name: 'Generator', exact: true })).toContainText('Gemini API');
  expect(submitted).toEqual([]);
  await expect(page.getByRole('radiogroup', { name: 'Dataset', exact: true }).getByRole('radio')).toBeChecked();
  await expect(page.getByRole('radiogroup', { name: 'Dataset', exact: true }).getByRole('radio')).toHaveCount(1);
  await expect(page.getByRole('radiogroup', { name: 'Camera', exact: true }).getByRole('radio')).toHaveCount(2);
  await page.getByRole('radio', { name: 'Side camera', exact: true }).locator('..').click();
  await page.getByLabel('Episode numbers', { exact: true }).fill('1, 3');
  await page.getByLabel('Start at (seconds)').fill('1.5');
  await page.getByLabel('Clip length (seconds)').fill('4');
  await page.getByRole('radio', { name: 'Change textures', exact: true }).locator('..').click();
  await page.getByLabel('Additional instructions (optional)').fill('Give the table a matte wood surface.');
  await expect(page.getByText(/Clips and prompts are sent to Google’s Gemini API. Charges may apply./)).toBeVisible();
  await noOverflow(page);
  await page.getByRole('button', { name: 'Generate augmented clips', exact: true }).click();
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0]).toEqual({
    operation: 'dataset.augment', source_job_id: 'inspected-video', episode_indices: [1, 3],
    camera_key: 'observation.images.side', preset: 'texture', prompt: 'Give the table a matte wood surface.',
    start_seconds: 1.5, duration_seconds: 4,
  });
  await expect(page.getByText('Generated clips may not match the original action labels. Review motion and timing, and validate labels before training.', { exact: true })).toBeVisible();
  await expect(page.getByLabel('Original episode 1', { exact: true })).toHaveAttribute('src', /augmentation-1\/augmentation\/clips\/0\?original=true$/);
  await expect(page.getByLabel('Augmented episode 1', { exact: true })).toHaveAttribute('src', /augmentation-1\/augmentation\/clips\/0\?original=false$/);
  await page.getByRole('button', { name: 'Preview episode 3', exact: true }).click();
  await expect(page.getByLabel('Augmented episode 3', { exact: true })).toHaveAttribute('src', /\/clips\/1\?original=false$/);
  await expect(page.getByRole('link', { name: 'Download review bundle' })).toHaveAttribute('href', /augmentation-1\/augmentation\/download$/);
  // The only write is the requested augmentation; no training job is submitted.
  expect(unexpected).toEqual([]);
  await page.getByText('Prompt & source provenance', { exact: true }).click();
  await expect(page.getByText('Preserve robot motion. Use soft side lighting.', { exact: true })).toBeVisible();
  await noOverflow(page);
  const screenshot = testInfo.outputPath('augmentation-review.png');
  await page.screenshot({ path: screenshot, fullPage: true });
  await testInfo.attach('augmentation-review', { path: screenshot, contentType: 'image/png' });
  expect(unexpected).toEqual([]);
});

test('validates episode selection, timing and required custom instructions before submitting', async ({ page }) => {
  const { submitted, unexpected } = await mockWorkspace(page);
  const start = page.getByRole('button', { name: 'Generate augmented clips', exact: true });
  await expect(start).toBeEnabled();
  for (const value of ['', '-1', '1.5', '0,0', '0,1,2,3,4', '12']) {
    await page.getByLabel('Episode numbers', { exact: true }).fill(value);
    await expect(start).toBeDisabled();
  }
  await page.getByLabel('Episode numbers', { exact: true }).fill('0, 2');
  await page.getByLabel('Start at (seconds)').fill('-1');
  await expect(start).toBeDisabled();
  await page.getByLabel('Start at (seconds)').fill('0');
  for (const value of ['0', '11', '']) {
    await page.getByLabel('Clip length (seconds)').fill(value);
    await expect(start).toBeDisabled();
  }
  await page.getByLabel('Clip length (seconds)').fill('5');
  await page.getByRole('radio', { name: 'Custom edit', exact: true }).focus();
  await page.keyboard.press('Space');
  await expect(start).toBeDisabled();
  await page.getByLabel('Edit instructions').fill('Use a blue tabletop and soft lighting.');
  await expect(start).toBeEnabled();
  await start.click();
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0]).toMatchObject({ episode_indices: [0, 2], preset: 'custom', prompt: 'Use a blue tabletop and soft lighting.' });
  expect(unexpected).toEqual([]);
});

test('running augmentation persists across tab changes, polls for completion and can be cancelled', async ({ page }) => {
  const { submitted, unexpected, jobs, cancelled } = await mockWorkspace(page);
  await page.getByRole('button', { name: 'Generate augmented clips', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Cancel augmentation', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Dataset', exact: true }).click();
  await page.getByRole('button', { name: 'Augmentation', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Cancel augmentation', exact: true })).toBeVisible();
  jobs[0].status = 'succeeded';
  jobs[0].result = resultFor(submitted[0]);
  await expect(page.getByText('Generated clips may not match the original action labels. Review motion and timing, and validate labels before training.', { exact: true })).toBeVisible({ timeout: 8000 });
  await page.getByRole('button', { name: 'Generate augmented clips', exact: true }).click();
  await expect.poll(() => submitted.length).toBe(2);
  await page.getByRole('button', { name: 'Cancel augmentation', exact: true }).click();
  await expect.poll(() => cancelled).toEqual(['augmentation-2']);
  await expect(page.getByText('Run did not complete.', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'View run 1', exact: true }).click();
  await expect(page.getByText('Generated clips may not match the original action labels. Review motion and timing, and validate labels before training.', { exact: true })).toBeVisible();
  expect(unexpected).toEqual([]);
});

test('requires server configuration, explains the setup, and keeps API keys out of the form', async ({ page }) => {
  const { submitted, unexpected } = await mockWorkspace(page, { configured: false });
  await expect(page.getByText('Connect Gemini to start generating', { exact: true })).toBeVisible();
  await page.getByText('Setup details', { exact: true }).click();
  await expect(page.getByText(/Set GEMINI_API_KEY/)).toBeVisible();
  await expect(page.getByText(/ffmpeg and ffprobe/).first()).toBeVisible();
  await expect(page.getByRole('button', { name: 'Generate augmented clips', exact: true })).toBeDisabled();
  await expect(page.locator('input[type=password]')).toHaveCount(0);
  await page.getByRole('button', { name: 'Check setup again', exact: true }).click();
  expect(submitted).toEqual([]);
  expect(unexpected).toEqual([]);
});

test('shows the Google Cloud login and billing project before augmentation', async ({ page }) => {
  const { submitted, unexpected } = await mockWorkspace(page, { cloud: true });
  await expect(page.getByRole('group', { name: 'Generator', exact: true })).toContainText('Google Cloud');
  await page.getByText('Connection details', { exact: true }).click();
  await expect(page.getByText(/uses your Google Cloud login, without an API key/)).toBeVisible();
  await expect(page.getByText('Clips and prompts are sent to Google Cloud. Charges apply to robotics-demo.', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Generate augmented clips', exact: true })).toBeEnabled();
  await expect(page.locator('input[type=password]')).toHaveCount(0);
  expect(submitted).toEqual([]);
  expect(unexpected).toEqual([]);
});

test('recovers a failed configuration request and offers source import when no video dataset exists', async ({ page }) => {
  const { submitted, unexpected } = await mockWorkspace(page, { empty: true, failFirst: true });
  await expect(page.getByText('Augmentation setup temporarily unavailable.', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Retry augmentation setup', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Retry augmentation setup', exact: true })).toHaveCount(0);
  await expect(page.getByText('No video datasets', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Generate augmented clips', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Import a dataset', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Import a dataset', exact: true })).toBeVisible();
  expect(submitted).toEqual([]);
  expect(unexpected).toEqual([]);
});

test('augmentation remains keyboard usable without horizontal overflow at 320px', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 740 });
  const { submitted, unexpected } = await mockWorkspace(page, { complete: true });
  await page.getByRole('button', { name: 'Dark', exact: true }).click();
  const custom = page.getByRole('radio', { name: 'Custom edit', exact: true });
  await custom.focus();
  await page.keyboard.press('Space');
  await expect(custom).toBeChecked();
  await page.getByLabel('Edit instructions').fill('Change only the lighting.');
  await noOverflow(page);
  await page.getByRole('button', { name: 'Generate augmented clips', exact: true }).click();
  await expect(page.getByText('Generated clips may not match the original action labels. Review motion and timing, and validate labels before training.', { exact: true })).toBeVisible();
  await page.getByText('Prompt & source provenance', { exact: true }).focus();
  await page.keyboard.press('Enter');
  await expect(page.getByText(revision, { exact: true }).last()).toBeVisible();
  await noOverflow(page);
  expect(submitted).toHaveLength(1);
  expect(unexpected).toEqual([]);
});


test('waits for confirmed project membership and preserves the first setup error until explicit retry', async ({ page }) => {
  let releaseProjects!: () => void;
  const projectsReady = new Promise<void>(resolve => { releaseProjects = resolve; });
  const fixture = await mockWorkspace(page, { projectsReady, failFirst: true });
  await expect(page.getByRole('region', { name: 'Project', exact: true }).getByRole('status')).toBeVisible();
  await expect(page.getByLabel('Additional instructions (optional)')).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Generate augmented clips', exact: true })).toBeDisabled();
  expect(fixture.optionsAttempts()).toBe(0);
  releaseProjects();
  await expect(page.getByRole('radiogroup', { name: 'Dataset', exact: true }).getByRole('radio')).toBeChecked();
  await expect(page.getByText('Augmentation setup temporarily unavailable.', { exact: true })).toBeVisible();
  expect(fixture.optionsAttempts()).toBe(1);
  await page.getByRole('button', { name: 'Retry augmentation setup', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Generate augmented clips', exact: true })).toBeEnabled();
  expect(fixture.optionsAttempts()).toBe(2);
  expect(fixture.submitted).toEqual([]);
  expect(fixture.unexpected).toEqual([]);
});

test('retains edit instructions when inspections arrive and disables cached controls after project lookup failure', async ({ page }) => {
  let releaseJobs!: () => void;
  const jobsReady = new Promise<void>(resolve => { releaseJobs = resolve; });
  let failProjects = false;
  const fixture = await mockWorkspace(page, { jobsReady, failProjects: () => failProjects });
  const prompt = page.getByLabel('Additional instructions (optional)');
  await expect(prompt).toBeEnabled();
  await prompt.fill('Preserve this lighting instruction.');
  releaseJobs();
  await expect(page.getByRole('radiogroup', { name: 'Dataset', exact: true }).getByRole('radio')).toBeChecked();
  await expect(prompt).toHaveValue('Preserve this lighting instruction.');
  // The application retries projects on window focus. Expire the real query's
  // five-second freshness; no provider/model request is allowed by the fixture.
  await page.waitForTimeout(5100);
  failProjects = true;
  await page.evaluate(() => window.dispatchEvent(new Event('visibilitychange')));
  await expect(page.getByText('Projects temporarily unavailable.', { exact: true })).toBeVisible();
  await expect(prompt).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Generate augmented clips', exact: true })).toBeDisabled();
  failProjects = false;
  await page.getByRole('button', { name: 'Retry projects', exact: true }).click();
  await expect(prompt).toBeEnabled();
  await expect(prompt).toHaveValue('Preserve this lighting instruction.');
  expect(fixture.submitted).toEqual([]);
  expect(fixture.unexpected).toEqual([]);
});


test('episode quick picks enforce the clip limit and keep manual selection in sync', async ({ page }) => {
  const { submitted, unexpected } = await mockWorkspace(page);
  const episode = (index: number) => page.getByRole('button', { name: `Episode ${index}`, exact: true });
  const numbers = page.getByLabel('Episode numbers', { exact: true });
  await expect(episode(0)).toHaveAttribute('aria-pressed', 'true');
  await episode(2).click();
  await episode(3).click();
  await episode(5).click();
  await expect(numbers).toHaveValue('0, 2, 3, 5');
  await expect(episode(1)).toBeDisabled();
  await episode(2).click();
  await expect(numbers).toHaveValue('0, 3, 5');
  await expect(episode(1)).toBeEnabled();
  await numbers.fill('6, 10');
  await expect(episode(6)).toHaveAttribute('aria-pressed', 'true');
  await expect(episode(0)).toHaveAttribute('aria-pressed', 'false');
  await page.getByRole('button', { name: 'Generate augmented clips', exact: true }).click();
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0].episode_indices).toEqual([6, 10]);
  expect(unexpected).toEqual([]);
});

test('dataset handoff only offers supported video sources and preserves the selected inspection', async ({ page }) => {
  const { jobs, submitted, unexpected } = await mockWorkspace(page);
  jobs.find(job => job.id === 'local-inspection')!.result.snapshot = { path: '/fixture/local', lineage_validated: true, total_episodes: 12 };
  jobs.push({ ...inspectedDataset('second-video', 'fixture/second'), created_at: '2026-09-25T12:00:00Z' });
  await page.route('**/api/v1/jobs/*/episodes**', route => route.fulfill({ json: { episodes: [], total_episodes: 12, offset: 0, limit: 6, warnings: [] } }));
  await page.reload();
  await page.getByRole('button', { name: /^Inspection/ }).click();
  const history = page.getByRole('combobox', { name: 'History', exact: true });
  const augment = page.getByRole('button', { name: 'Augment this dataset', exact: true });
  for (const id of ['local-inspection', 'image-only']) {
    await history.selectOption(id);
    await expect(augment).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Train on this dataset', exact: true })).toBeVisible();
  }
  await history.selectOption('failed-inspection');
  await expect(augment).toHaveCount(0);
  await history.selectOption('second-video');
  await expect(augment).toBeVisible();
  await augment.click();
  await expect(page.getByRole('radiogroup', { name: 'Dataset', exact: true }).getByRole('radio', { name: /^fixture\/second,/ })).toBeChecked();
  expect(submitted).toEqual([]);
  await page.getByRole('button', { name: 'Generate augmented clips', exact: true }).click();
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0].source_job_id).toBe('second-video');
  expect(unexpected).toEqual([]);
});


test('lost augmentation acknowledgement recovers its original job by saved key after reload without another POST', async ({ page }) => {
  const fixture = await mockWorkspace(page, { submissionMode: 'lost-ack', complete: true });
  await page.getByRole('button', { name: 'Generate augmented clips', exact: true }).click();
  const recovery = page.getByRole('complementary', { name: 'Request recovery' });
  await expect(recovery.getByRole('button', { name: 'Check saved request' })).toBeEnabled();
  expect(fixture.submitted).toHaveLength(1);
  expect(fixture.keys[0]).toMatch(/^[a-f0-9-]{36}$/);
  await page.reload();
  await page.getByRole('button', { name: 'Augmentation', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Generate augmented clips', exact: true })).toBeDisabled();
  await recovery.getByRole('button', { name: 'Check saved request' }).click();
  await expect(recovery).toHaveCount(0);
  await expect(page.getByLabel('Augmented episode 0', { exact: true })).toBeVisible();
  expect(fixture.submitted).toHaveLength(1);
  expect(fixture.lookups).toEqual([fixture.keys[0], fixture.keys[0]]);
  expect(fixture.unexpected).toEqual([]);
});

test('a missing binding keeps new augmentation blocked and explicit retry preserves the original key and recipe', async ({ page }) => {
  const fixture = await mockWorkspace(page, { submissionMode: 'lost-before-commit', complete: true });
  await page.getByLabel('Additional instructions (optional)').fill('Original saved lighting');
  await page.getByRole('button', { name: 'Generate augmented clips', exact: true }).click();
  const recovery = page.getByRole('complementary', { name: 'Request recovery' });
  await expect(recovery.getByRole('button', { name: 'Check saved request' })).toBeEnabled();
  await page.getByLabel('Additional instructions (optional)').fill('Edited form must not replace saved request');
  await recovery.getByRole('button', { name: 'Check saved request' }).click();
  await expect(recovery.getByRole('button', { name: 'Retry same request' })).toBeEnabled();
  await expect(page.getByRole('button', { name: 'Generate augmented clips', exact: true })).toBeDisabled();
  expect(fixture.submitted).toHaveLength(1);
  await recovery.getByRole('button', { name: 'Retry same request' }).click();
  await expect(recovery).toHaveCount(0);
  expect(fixture.keys).toEqual([fixture.keys[0], fixture.keys[0]]);
  expect(fixture.submitted).toEqual([fixture.submitted[0], fixture.submitted[0]]);
  expect(fixture.submitted[1].prompt).toBe('Original saved lighting');
  expect(fixture.unexpected).toEqual([]);
});

test('an older server cannot receive an unprotected augmentation POST', async ({ page }) => {
  const fixture = await mockWorkspace(page, { submissionMode: 'unsupported' });
  await page.getByRole('button', { name: 'Generate augmented clips', exact: true }).click();
  const recovery = page.getByRole('complementary', { name: 'Request recovery' });
  await expect(recovery).toContainText('did not verify durable submission support');
  await expect(page.getByRole('button', { name: 'Generate augmented clips', exact: true })).toBeDisabled();
  await expect(recovery.getByRole('button', { name: 'Retry same request' })).toHaveCount(0);
  expect(fixture.submitted).toEqual([]);
});

test('a wrong-project augmentation acknowledgement remains unresolved until scoped lookup verifies it', async ({ page }) => {
  const fixture = await mockWorkspace(page, { submissionMode: 'wrong-ack' });
  await page.getByRole('button', { name: 'Generate augmented clips', exact: true }).click();
  const recovery = page.getByRole('complementary', { name: 'Request recovery' });
  await expect(recovery.getByRole('button', { name: 'Check saved request' })).toBeEnabled();
  await expect(page.getByRole('button', { name: 'Generate augmented clips', exact: true })).toBeDisabled();
  await recovery.getByRole('button', { name: 'Check saved request' }).click();
  await expect(recovery).toHaveCount(0);
  expect(fixture.submitted).toHaveLength(1);
});

test('definitive initial recipe rejection allows correction with a new key', async ({ page }) => {
  const fixture = await mockWorkspace(page, { submissionMode: 'rejected' });
  const submit = page.getByRole('button', { name: 'Generate augmented clips', exact: true });
  await submit.click();
  await expect(page.getByText('Synthetic recipe rejection before admission.', { exact: true })).toBeVisible();
  await expect(submit).toBeEnabled();
  await page.getByLabel('Additional instructions (optional)').fill('Corrected recipe');
  await submit.click();
  await expect.poll(() => fixture.submitted.length).toBe(2);
  expect(fixture.keys[1]).not.toBe(fixture.keys[0]);
  expect(fixture.submitted[1].prompt).toBe('Corrected recipe');
});


test('late augmentation acknowledgement preserves a manually selected earlier result', async ({ page }) => {
  const state = await mockWorkspace(page, { submissionMode: 'delayed-ack', priorRuns: true, complete: true });
  await page.getByRole('button', { name: 'Generate augmented clips', exact: true }).click();
  await expect.poll(() => state.submitted.length).toBe(1);
  await page.getByRole('button', { name: 'View run 1', exact: true }).click();
  await expect(page.getByLabel('Augmented episode 1', { exact: true })).toBeVisible();
  state.releaseAck();
  await expect(page.getByRole('complementary', { name: 'Request recovery' })).toHaveCount(0);
  await expect(page.getByLabel('Augmented episode 1', { exact: true })).toBeVisible();
  expect(state.submitted).toHaveLength(1);
});
