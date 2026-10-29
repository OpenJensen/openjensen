import { fulfillWorkJobRead, openModuleTask } from './work-job-controls';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fulfillApplicationUpdateRead } from './application-update-controls';
import { expect, test, type Page } from '@playwright/test';
import { installedModules } from './module-fixtures';

const projectId = 'augmentation-fixture';
const timestamp = '2026-09-26T12:00:00Z';
const revision = 'a'.repeat(40);
const camera = 'observation.images.front';

function inspectedDataset(id = 'inspected-video', repo = 'fixture/robot') {
  return {
    id,
    project_id: projectId,
    kind: 'dataset.inspect',
    status: 'succeeded',
    request: { source: 'huggingface', repo_id: repo, revision: 'main' },
    created_at: timestamp,
    updated_at: timestamp,
    result: {
      source: 'huggingface',
      repo_id: repo,
      revision,
      format: 'lerobot_v3',
      inspection_scope: 'metadata_only',
      total_episodes: 12,
      total_frames: 1800,
      fps: 30,
      features: {
        [camera]: { dtype: 'video', shape: [480, 640, 3] },
        'observation.images.side': { dtype: 'video', shape: [480, 640, 3] },
        'observation.images.wrist': { dtype: 'image', shape: [480, 640, 3] },
        action: { dtype: 'float32', shape: [6] },
        'observation.state': { dtype: 'float32', shape: [6] },
      },
      metadata_sha256: 'b'.repeat(64),
      inspected_at: timestamp,
      warnings: [],
    },
  };
}

function resultFor(body: Record<string, any>) {
  return {
    operation: 'dataset.augment',
    model: 'gemini-omni-1.1-flash',
    prompt: 'Preserve robot motion. Use soft side lighting.',
    source_job_id: body.source_job_id,
    source_clip: body.source_clip ?? null,
    repo_id: 'fixture/robot',
    revision,
    metadata_sha256: 'b'.repeat(64),
    review_required: true,
    warnings: ['Review visual changes against the original actions.'],
    clips: body.episode_indices.map((episode: number, index: number) => ({
      index,
      episode_index: episode,
      camera_key: body.camera_key,
      source_start_seconds: body.start_seconds,
      source_end_seconds: body.start_seconds + body.duration_seconds,
      input_sha256: 'c'.repeat(64),
      output_sha256: 'd'.repeat(64),
      interaction_id: 'mock-interaction',
    })),
  };
}

async function mockWorkspace(
  page: Page,
  config: {
    configured?: boolean;
    empty?: boolean;
    complete?: boolean;
    failFirst?: boolean;
    cloud?: boolean;
    projectsReady?: Promise<void>;
    jobsReady?: Promise<void>;
    failProjects?: () => boolean;
    submissionMode?: 'lost-ack' | 'lost-before-commit' | 'unsupported' | 'wrong-ack' | 'rejected' | 'delayed-ack';
    priorRuns?: boolean;
    media?: { original: Buffer; augmented: Buffer };
  } = {},
) {
  let releaseAck!: () => void;
  const ackReady = new Promise<void>((resolve) => {
    releaseAck = resolve;
  });
  const submitted: Record<string, any>[] = [];
  const keys: string[] = [];
  const bindings = new Map<string, Record<string, any>>();
  const lookups: string[] = [];
  const unexpected: string[] = [];
  const cancelled: string[] = [];
  const plans: Record<string, any>[] = [];
  const jobs: Record<string, any>[] = config.empty
    ? []
    : [
        inspectedDataset(),
        { ...inspectedDataset('failed-inspection'), status: 'failed', result: null },
        {
          ...inspectedDataset('image-only'),
          result: {
            ...inspectedDataset().result,
            features: { 'observation.images.wrist': { dtype: 'image', shape: [480, 640, 3] } },
          },
        },
        { ...inspectedDataset('local-inspection'), result: { ...inspectedDataset().result, source: 'local' } },
      ];
  if (config.priorRuns)
    for (let i = 1; i <= 2; i++) {
      const request = {
        operation: 'dataset.augment',
        source_job_id: 'inspected-video',
        episode_indices: [i],
        camera_key: camera,
        preset: 'lighting',
        prompt: '',
        start_seconds: 0,
        duration_seconds: 5,
      };
      jobs.unshift({
        id: `previous-${i}`,
        project_id: projectId,
        kind: 'dataset.augment',
        status: 'succeeded',
        request,
        created_at: `2026-09-26T1${i}:00:00Z`,
        updated_at: timestamp,
        result: resultFor(request),
      });
    }
  let optionsAttempts = 0;
  // Intercept all application requests, including every write. The browser
  // suite cannot invoke a model or upload media to a real provider.
  await page.route('**/api/v1/**', async (route) => {
    if (await fulfillApplicationUpdateRead(route)) return;
    if (await fulfillWorkJobRead(route)) return;
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const submissionPath = `/api/v1/projects/${projectId}/submissions/`;
    if (request.method() === 'GET' && path.startsWith(submissionPath)) {
      const key = decodeURIComponent(path.slice(submissionPath.length));
      lookups.push(key);
      const headers =
        config.submissionMode === 'unsupported' ? {} : { 'Idempotency-Key': key, 'Cache-Control': 'no-store' };
      await route.fulfill({
        status: bindings.has(key) ? 200 : 404,
        headers,
        json: bindings.get(key) ?? { detail: 'No accepted submission found for this project, operation and key' },
      });
      return;
    }
    if (request.method() === 'POST' && path === `/api/v1/projects/${projectId}/augmentations/plan`) {
      const body = request.postDataJSON();
      plans.push(body);
      const count = body.dataset_scope.camera_keys.length;
      await route.fulfill({
        json: {
          total_episodes: 12,
          camera_keys: body.dataset_scope.camera_keys,
          total_source_seconds: 60 * count,
          total_clips: 12 * count,
          provider_requests: 12 * count,
          recipe_prompts: ['Preserve motion. Make the table metal.'],
          plan_sha256: 'f'.repeat(64),
          model: 'unpaid-fixture',
          warnings: ['Full episodes are covered in segments of at most 10 seconds.'],
        },
      });
      return;
    }
    if (request.method() === 'POST' && path === `/api/v1/projects/${projectId}/augmentations`) {
      const body = request.postDataJSON();
      submitted.push(body);
      const key = request.headers()['idempotency-key'];
      keys.push(key);
      if (submitted.length === 1 && config.submissionMode === 'lost-before-commit') {
        await route.abort('connectionfailed');
        return;
      }
      if (submitted.length === 1 && config.submissionMode === 'rejected') {
        await route.fulfill({ status: 422, json: { detail: 'Synthetic recipe rejection before admission.' } });
        return;
      }
      const job = {
        id: `augmentation-${submitted.length}`,
        project_id: projectId,
        kind: 'dataset.augment',
        request: body,
        status: config.complete ? 'succeeded' : 'running',
        stage: 'generating_clips',
        created_at: '2026-09-26T14:00:00Z',
        updated_at: timestamp,
        error: null,
        result: config.complete ? resultFor(body) : null,
      };
      jobs.unshift(job);
      bindings.set(key, job);
      if (submitted.length === 1 && config.submissionMode === 'lost-ack') {
        await route.abort('connectionfailed');
        return;
      }
      if (config.submissionMode === 'delayed-ack') await ackReady;
      await route.fulfill({
        status: 202,
        headers: { 'Idempotency-Key': key, 'Cache-Control': 'no-store' },
        json: config.submissionMode === 'wrong-ack' ? { ...job, project_id: 'another-project' } : job,
      });
      return;
    }
    if (request.method() === 'POST' && /^\/api\/v1\/jobs\/[^/]+\/cancel$/.test(path)) {
      const id = path.split('/')[4];
      cancelled.push(id);
      const job = jobs.find((item) => item.id === id)!;
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
        '/api/v1/capabilities': [],
        '/api/v1/modules': installedModules,
        '/api/v1/datasets': [],
        '/api/v1/projects': [{ id: projectId, name: 'Augmentation review', created_at: timestamp }],
        [`/api/v1/projects/${projectId}/jobs`]: jobs,
        '/api/v1/augmentation-options': {
          configured: config.configured ?? true,
          model: 'gemini-omni-1.1-flash',
          preservation_prompt: 'Preserve all unmentioned surfaces.\n',
          max_clips: 4,
          max_duration_seconds: 10,
          auth_mode: config.cloud ? 'google_cloud' : 'gemini_api_key',
          google_cloud_project: config.cloud ? 'robotics-demo' : null,
          auth_message: config.cloud
            ? 'Google Cloud · robotics-demo · uses your Google Cloud login, without an API key. Model access and billing are checked when a run starts.'
            : null,
          setup_message:
            config.configured === false
              ? 'Set GEMINI_API_KEY on the application server and install ffmpeg and ffprobe.'
              : null,
          presets: [
            { id: 'lighting', label: 'Lighting', prompt: 'Use soft side lighting.' },
            { id: 'texture', label: 'Textures', prompt: 'Change the tabletop texture.' },
            { id: 'custom', label: 'Custom', prompt: '' },
          ],
        },
      };
      if (path in responses) {
        await route.fulfill({ json: responses[path] });
        return;
      }
      if (/\/augmentation\/clips\/\d+$/.test(path)) {
        // Real playback fixtures support byte ranges, as the clip download endpoint does.
        const original = new URL(request.url()).searchParams.get('original') === 'true';
        const media = config.media?.[original ? 'original' : 'augmented'];
        const range = request.headers().range?.match(/^bytes=(\d+)-(\d*)$/);
        const start = Number(range?.[1] ?? 0);
        const end = Math.min(Number(range?.[2] || (media?.length ?? 1) - 1), (media?.length ?? 1) - 1);
        await route.fulfill({
          status: media && range ? 206 : 200,
          contentType: 'video/mp4',
          headers: media
            ? {
                'Accept-Ranges': 'bytes',
                ...(range ? { 'Content-Range': 'bytes ' + start + '-' + end + '/' + media.length } : {}),
              }
            : {},
          body: media ? media.subarray(start, end + 1) : '',
        });
        return;
      }
    }
    unexpected.push(`${request.method()} ${path}`);
    await route.fulfill({ status: 405, json: { detail: 'Blocked by augmentation browser fixture.' } });
  });
  await page.goto('/datasets/');
  await page.getByRole('link', { name: 'Augmentation', exact: true }).click();
  if (!config.projectsReady) {
    await openModuleTask(page);
    await expect(page.getByRole('group', { name: 'Generator', exact: true })).toContainText('Gemini Omni');
  }
  await expect(page.getByRole('heading', { name: 'Augmentation', exact: true, level: 1 })).toBeVisible();
  return {
    submitted,
    unexpected,
    jobs,
    cancelled,
    plans,
    keys,
    lookups,
    releaseAck,
    optionsAttempts: () => optionsAttempts,
  };
}

async function openStyle(page: Page) {
  const dialog = page.getByRole('dialog');
  if (await dialog.isVisible()) return dialog;
  const add = page.getByRole('button', { name: 'Add a style', exact: true });
  if (!(await add.isVisible())) await page.getByText('Choose preview clips', { exact: true }).click();
  await add.click();
  await expect(dialog).toBeVisible();
  return dialog;
}

async function fillStyle(page: Page, prompt: string) {
  const dialog = await openStyle(page);
  await dialog.getByLabel('Style prompt', { exact: true }).fill(prompt);
}

async function generateStyle(page: Page) {
  const dialog = await openStyle(page);
  if (!(await dialog.getByLabel('Style prompt', { exact: true }).inputValue()))
    await dialog.getByLabel('Style prompt', { exact: true }).fill('Use soft side lighting.');
  await dialog.getByRole('button', { name: 'Generate style', exact: true }).click();
}

async function noOverflow(page: Page) {
  const dimensions = await page.evaluate(() => ({
    width: document.documentElement.clientWidth,
    scroll: document.documentElement.scrollWidth,
  }));
  expect(dimensions.scroll).toBeLessThanOrEqual(dimensions.width + 1);
}

test('submits selected clips and texture instructions, then compares and downloads the completed result', async ({
  page,
}, testInfo) => {
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
  await fillStyle(page, 'Give the table a matte wood surface.');
  await expect(page.getByText(/Clips and prompts are sent to Google’s Gemini API. Charges may apply./)).toBeVisible();
  await noOverflow(page);
  await generateStyle(page);
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0]).toEqual({
    operation: 'dataset.augment',
    source_job_id: 'inspected-video',
    episode_indices: [1, 3],
    camera_key: 'observation.images.side',
    preset: 'custom',
    prompt: 'Give the table a matte wood surface.',
    start_seconds: 1.5,
    duration_seconds: 4,
  });
  await expect(
    page.getByText(
      'Generated clips may not match the original action labels. Review motion and timing, and validate labels before training.',
      { exact: true },
    ),
  ).toBeVisible();
  await expect(page.getByLabel('Original episode 1', { exact: true })).toHaveAttribute(
    'src',
    /augmentation-1\/augmentation\/clips\/0\?original=true$/,
  );
  await expect(page.getByLabel('Augmented episode 1', { exact: true })).toHaveAttribute(
    'src',
    /augmentation-1\/augmentation\/clips\/0\?original=false$/,
  );
  await page.getByRole('button', { name: 'Preview episode 3', exact: true }).click();
  await expect(page.getByLabel('Augmented episode 3', { exact: true })).toHaveAttribute(
    'src',
    /\/clips\/1\?original=false$/,
  );
  await expect(page.getByRole('link', { name: 'Download review bundle' })).toHaveAttribute(
    'href',
    /augmentation-1\/augmentation\/download$/,
  );
  // The only write is the requested augmentation; no training job is submitted.
  expect(unexpected).toEqual([]);
  await page.getByText('Prompt & source provenance', { exact: true }).click();
  await expect(page.getByText('Preserve robot motion. Use soft side lighting.', { exact: true }).last()).toBeVisible();
  await noOverflow(page);
  const screenshot = testInfo.outputPath('augmentation-review.png');
  await page.screenshot({ path: screenshot, fullPage: true });
  await testInfo.attach('augmentation-review', { path: screenshot, contentType: 'image/png' });
  expect(unexpected).toEqual([]);
});

test('validates source selection and requires a prompt in the style dialog', async ({ page }) => {
  const { submitted, unexpected } = await mockWorkspace(page);
  const add = page.getByRole('button', { name: 'Add a style', exact: true });
  await expect(add).toBeEnabled();
  for (const value of ['', '-1', '1.5', '0,0', '0,1,2,3,4', '12']) {
    await page.getByLabel('Episode numbers', { exact: true }).fill(value);
    await expect(add).toBeDisabled();
  }
  await page.getByLabel('Episode numbers', { exact: true }).fill('0, 2');
  await page.getByLabel('Start at (seconds)').fill('-1');
  await expect(add).toBeDisabled();
  await page.getByLabel('Start at (seconds)').fill('0');
  for (const value of ['0', '11', '']) {
    await page.getByLabel('Clip length (seconds)').fill(value);
    await expect(add).toBeDisabled();
  }
  await page.getByLabel('Clip length (seconds)').fill('5');
  const dialog = await openStyle(page);
  const start = dialog.getByRole('button', { name: 'Generate style', exact: true });
  await expect(start).toBeDisabled();
  await dialog.getByLabel('Style prompt').fill('   ');
  await expect(start).toBeDisabled();
  await dialog.getByLabel('Style prompt').fill('Use a blue tabletop and soft lighting.');
  await expect(start).toBeEnabled();
  await start.click();
  await expect(dialog).not.toBeVisible();
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0]).toMatchObject({
    episode_indices: [0, 2],
    preset: 'custom',
    prompt: 'Use a blue tabletop and soft lighting.',
  });
  expect(unexpected).toEqual([]);
});

test('running augmentation persists across tab changes, polls for completion and can be cancelled', async ({
  page,
}) => {
  const { submitted, unexpected, jobs, cancelled } = await mockWorkspace(page);
  await generateStyle(page);
  await expect(page.getByRole('button', { name: 'Cancel augmentation', exact: true })).toBeVisible();
  await page.getByRole('link', { name: 'Datasets', exact: true }).click();
  await page.getByRole('link', { name: 'Augmentation', exact: true }).click();
  await openModuleTask(page);
  await expect(page.getByRole('button', { name: 'Cancel augmentation', exact: true })).toBeVisible();
  jobs[0].status = 'succeeded';
  jobs[0].result = resultFor(submitted[0]);
  await expect(
    page.getByText(
      'Generated clips may not match the original action labels. Review motion and timing, and validate labels before training.',
      { exact: true },
    ),
  ).toBeVisible({ timeout: 8000 });
  await generateStyle(page);
  await expect.poll(() => submitted.length).toBe(2);
  await page.getByRole('button', { name: 'Cancel augmentation', exact: true }).click();
  await expect.poll(() => cancelled).toEqual(['augmentation-2']);
  await expect(page.getByText('Run did not complete.', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'View run 1', exact: true }).click();
  await expect(
    page.getByText(
      'Generated clips may not match the original action labels. Review motion and timing, and validate labels before training.',
      { exact: true },
    ),
  ).toBeVisible();
  expect(unexpected).toEqual([]);
});

test('requires server configuration, explains the setup, and keeps API keys out of the form', async ({ page }) => {
  const { submitted, unexpected } = await mockWorkspace(page, { configured: false });
  await expect(page.getByText('Connect Gemini to start generating', { exact: true })).toBeVisible();
  await page.getByText('Setup details', { exact: true }).click();
  await expect(page.getByText(/Set GEMINI_API_KEY/)).toBeVisible();
  await expect(page.getByText(/ffmpeg and ffprobe/).first()).toBeVisible();
  await expect(page.getByRole('button', { name: 'Add a style', exact: true })).toBeDisabled();
  await expect(page.locator('input[type=password]')).toHaveCount(0);
  await page.getByRole('button', { name: 'Check setup again', exact: true }).click();
  expect(submitted).toEqual([]);
  expect(unexpected).toEqual([]);
});

test('shows the Google Cloud login and billing project before augmentation', async ({ page }) => {
  const { submitted, unexpected } = await mockWorkspace(page, { cloud: true });
  await expect(page.getByRole('group', { name: 'Generator', exact: true })).toContainText('Google Cloud');
  await fillStyle(page, 'Use warm light.');
  await page.getByText('Generation details', { exact: true }).click();
  await expect(page.getByText(/uses your Google Cloud login, without an API key/)).toBeVisible();
  await expect(
    page.getByText('Clips and prompts are sent to Google Cloud. Charges apply to robotics-demo.', { exact: true }),
  ).toBeVisible();
  await expect(page.getByRole('button', { name: 'Generate style', exact: true })).toBeEnabled();
  await expect(page.locator('input[type=password]')).toHaveCount(0);
  expect(submitted).toEqual([]);
  expect(unexpected).toEqual([]);
});

test('recovers a failed configuration request and offers source import when no video dataset exists', async ({
  page,
}) => {
  const { submitted, unexpected } = await mockWorkspace(page, { empty: true, failFirst: true });
  await expect(page.getByText('Augmentation setup temporarily unavailable.', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Retry augmentation setup', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Retry augmentation setup', exact: true })).toHaveCount(0);
  await expect(page.getByText('No video datasets', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Add a style', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Import a dataset', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Import a dataset', exact: true })).toBeVisible();
  expect(submitted).toEqual([]);
  expect(unexpected).toEqual([]);
});

test('style dialog remains keyboard usable without horizontal overflow at 320px', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 740 });
  const { submitted, unexpected } = await mockWorkspace(page, { complete: true });
  await page.getByRole('button', { name: 'Dark', exact: true }).click();
  const add = page.getByRole('button', { name: 'Add a style', exact: true });
  await add.focus();
  await page.keyboard.press('Enter');
  const dialog = page.getByRole('dialog', { name: 'Add a style', exact: true });
  await expect(dialog.getByLabel('Style prompt')).toBeFocused();
  await fillStyle(page, 'Change only the lighting.');
  const bounds = await dialog.boundingBox();
  expect(bounds!.x).toBeGreaterThanOrEqual(0);
  expect(bounds!.width).toBeLessThanOrEqual(320);
  await noOverflow(page);
  await generateStyle(page);
  await expect(dialog).not.toBeVisible();
  await page.getByText('Prompt & source provenance', { exact: true }).focus();
  await page.keyboard.press('Enter');
  await expect(
    page.locator('details.augmentation-provenance[open]').getByText(revision, { exact: true }),
  ).toBeVisible();
  await noOverflow(page);
  expect(submitted).toHaveLength(1);
  expect(unexpected).toEqual([]);
});

test('waits for confirmed project membership and preserves the first setup error until explicit retry', async ({
  page,
}) => {
  let releaseProjects!: () => void;
  const projectsReady = new Promise<void>((resolve) => {
    releaseProjects = resolve;
  });
  const fixture = await mockWorkspace(page, { projectsReady, failFirst: true });
  await expect(page.getByRole('region', { name: 'Project', exact: true }).getByRole('status')).toBeVisible();
  await expect(page.getByLabel('Style prompt')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'New augmentation', exact: true })).toBeDisabled();
  expect(fixture.optionsAttempts()).toBe(0);
  releaseProjects();
  await openModuleTask(page);
  await expect(page.getByRole('radiogroup', { name: 'Dataset', exact: true }).getByRole('radio')).toBeChecked();
  await expect(page.getByText('Augmentation setup temporarily unavailable.', { exact: true })).toBeVisible();
  expect(fixture.optionsAttempts()).toBe(1);
  await page.getByRole('button', { name: 'Retry augmentation setup', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Add a style', exact: true })).toBeEnabled();
  expect(fixture.optionsAttempts()).toBe(2);
  expect(fixture.submitted).toEqual([]);
  expect(fixture.unexpected).toEqual([]);
});

test('retains a style draft and blocks generation while project membership cannot be refreshed', async ({ page }) => {
  let releaseJobs!: () => void;
  const jobsReady = new Promise<void>((resolve) => {
    releaseJobs = resolve;
  });
  let failProjects = false;
  const fixture = await mockWorkspace(page, { jobsReady, failProjects: () => failProjects });
  await expect(page.getByRole('button', { name: 'Add a style', exact: true })).toBeDisabled();
  releaseJobs();
  await fillStyle(page, 'Preserve this lighting instruction.');
  const prompt = page.getByLabel('Style prompt');
  await page.waitForTimeout(5100);
  failProjects = true;
  await page.evaluate(() => window.dispatchEvent(new Event('visibilitychange')));
  await expect(prompt).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Generate style', exact: true })).toBeDisabled();
  failProjects = false;
  await page.evaluate(() => window.dispatchEvent(new Event('visibilitychange')));
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
  await generateStyle(page);
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0].episode_indices).toEqual([6, 10]);
  expect(unexpected).toEqual([]);
});

test('dataset handoff only offers supported video sources and preserves the selected inspection', async ({ page }) => {
  const { jobs, submitted, unexpected } = await mockWorkspace(page);
  jobs.find((job) => job.id === 'local-inspection')!.result.snapshot = {
    path: '/fixture/local',
    lineage_validated: true,
    total_episodes: 12,
  };
  jobs.push({ ...inspectedDataset('second-video', 'fixture/second'), created_at: '2026-09-25T12:00:00Z' });
  await page.route('**/api/v1/jobs/*/episodes**', (route) =>
    route.fulfill({ json: { episodes: [], total_episodes: 12, offset: 0, limit: 6, warnings: [] } }),
  );
  await page.route('**/api/v1/jobs/*/cover', (route) =>
    route.fulfill({ status: 422, json: { detail: 'Cover media is unavailable in this handoff fixture' } }),
  );
  await page.route('**/api/v1/datasets**', (route) =>
    route.fulfill({
      json: jobs
        .filter((job) => job.kind === 'dataset.inspect' && job.result)
        .map((job) => ({
          id: `inspection:${job.id}`,
          job_id: job.id,
          project_id: projectId,
          name: job.id,
          source: job.result.source,
          status: 'ready',
          created_at: job.created_at,
          profile: job.result,
        })),
    }),
  );
  await page.reload();
  await page.getByRole('link', { name: 'Datasets', exact: true }).click();
  await page.getByRole('button', { name: 'Open dataset inspected-video', exact: true }).click();
  await page.getByText('Advanced', { exact: true }).click();
  const history = page.getByRole('combobox', { name: 'History', exact: true });
  const augment = page.getByRole('button', { name: 'Augment this dataset', exact: true });
  for (const id of ['local-inspection', 'image-only']) {
    if (!(await history.isVisible())) await page.getByText('Advanced', { exact: true }).click();
    await history.selectOption(id);
    await expect(augment).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Train on this dataset', exact: true })).toBeVisible();
  }
  if (!(await history.isVisible())) await page.getByText('Advanced', { exact: true }).click();
  await history.selectOption('failed-inspection');
  await expect(augment).toHaveCount(0);
  await page.getByRole('button', { name: '← Back to datasets', exact: true }).click();
  await page.getByRole('button', { name: 'Open dataset second-video', exact: true }).click();
  await expect(augment).toBeVisible();
  await augment.click();
  await expect(
    page.getByRole('radiogroup', { name: 'Dataset', exact: true }).getByRole('radio', { name: /^fixture\/second,/ }),
  ).toBeChecked();
  expect(submitted).toEqual([]);
  await generateStyle(page);
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0].source_job_id).toBe('second-video');
  expect(unexpected).toEqual([]);
});

test('lost augmentation acknowledgement recovers its original job by saved key after reload without another POST', async ({
  page,
}) => {
  const fixture = await mockWorkspace(page, { submissionMode: 'lost-ack', complete: true });
  await generateStyle(page);
  const recovery = page.getByRole('complementary', { name: 'Request recovery' });
  await expect(recovery.getByRole('button', { name: 'Check saved request' })).toBeEnabled();
  expect(fixture.submitted).toHaveLength(1);
  expect(fixture.keys[0]).toMatch(/^[a-f0-9-]{36}$/);
  await page.reload();
  await page.getByRole('link', { name: 'Augmentation', exact: true }).click();
  await openModuleTask(page);
  await expect(page.getByRole('button', { name: /Add one more/ })).toBeDisabled();
  await recovery.getByRole('button', { name: 'Check saved request' }).click();
  await expect(recovery).toHaveCount(0);
  await expect(page.getByLabel('Augmented episode 0', { exact: true })).toBeVisible();
  expect(fixture.submitted).toHaveLength(1);
  expect(fixture.lookups).toEqual([fixture.keys[0], fixture.keys[0]]);
  expect(fixture.unexpected).toEqual([]);
});

test('a missing binding keeps new augmentation blocked and explicit retry preserves the original key and recipe', async ({
  page,
}) => {
  const fixture = await mockWorkspace(page, { submissionMode: 'lost-before-commit', complete: true });
  await fillStyle(page, 'Original saved lighting');
  await generateStyle(page);
  const recovery = page.getByRole('complementary', { name: 'Request recovery' });
  await expect(recovery.getByRole('button', { name: 'Check saved request' })).toBeEnabled();
  await fillStyle(page, 'Edited form must not replace saved request');
  await recovery.getByRole('button', { name: 'Check saved request' }).click();
  await expect(recovery.getByRole('button', { name: 'Retry same request' })).toBeEnabled();
  await expect(page.getByRole('button', { name: 'Generate style', exact: true })).toBeDisabled();
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
  await generateStyle(page);
  const recovery = page.getByRole('complementary', { name: 'Request recovery' });
  await expect(recovery).toContainText('did not verify durable submission support');
  await expect(page.getByRole('button', { name: 'Generate style', exact: true })).toBeDisabled();
  await expect(recovery.getByRole('button', { name: 'Retry same request' })).toHaveCount(0);
  expect(fixture.submitted).toEqual([]);
});

test('a wrong-project augmentation acknowledgement remains unresolved until scoped lookup verifies it', async ({
  page,
}) => {
  const fixture = await mockWorkspace(page, { submissionMode: 'wrong-ack' });
  await generateStyle(page);
  const recovery = page.getByRole('complementary', { name: 'Request recovery' });
  await expect(recovery.getByRole('button', { name: 'Check saved request' })).toBeEnabled();
  await expect(page.getByRole('button', { name: 'Generate style', exact: true })).toBeDisabled();
  await recovery.getByRole('button', { name: 'Check saved request' }).click();
  await expect(recovery).toHaveCount(0);
  expect(fixture.submitted).toHaveLength(1);
});

test('definitive initial recipe rejection allows correction with a new key', async ({ page }) => {
  const fixture = await mockWorkspace(page, { submissionMode: 'rejected' });
  await fillStyle(page, 'Use soft light.');
  const submit = page.getByRole('button', { name: 'Generate style', exact: true });
  await submit.click();
  await expect(page.getByText('Synthetic recipe rejection before admission.', { exact: true })).toBeVisible();
  await expect(submit).toBeEnabled();
  await fillStyle(page, 'Corrected recipe');
  await submit.click();
  await expect.poll(() => fixture.submitted.length).toBe(2);
  expect(fixture.keys[1]).not.toBe(fixture.keys[0]);
  expect(fixture.submitted[1].prompt).toBe('Corrected recipe');
});

test('late augmentation acknowledgement preserves a manually selected earlier result', async ({ page }) => {
  const state = await mockWorkspace(page, { submissionMode: 'delayed-ack', priorRuns: true, complete: true });
  await generateStyle(page);
  await expect.poll(() => state.submitted.length).toBe(1);
  await page.getByRole('button', { name: 'Close style dialog', exact: true }).click();
  await page.getByRole('button', { name: 'View run 1', exact: true }).click();
  await expect(page.getByLabel('Augmented episode 1', { exact: true })).toBeVisible();
  state.releaseAck();
  await expect(page.getByRole('complementary', { name: 'Request recovery' })).toHaveCount(0);
  await expect(page.getByLabel('Augmented episode 1', { exact: true })).toBeVisible();
  expect(state.submitted).toHaveLength(1);
});

test('style creation uses only the typed prompt with no preset choices or duplicate preview', async ({ page }) => {
  const { submitted } = await mockWorkspace(page);
  for (const name of ['Change lighting', 'Change textures', 'Custom edit'])
    await expect(page.getByRole('radio', { name, exact: true })).toHaveCount(0);
  await expect(page.getByLabel('Style prompt')).toHaveCount(0);
  await fillStyle(page, 'Make only the table metal. Keep the wall unchanged.');
  await expect(page.getByRole('note', { name: 'Edit to send' })).toHaveCount(0);
  await page.getByText('Generation details', { exact: true }).click();
  await expect(
    page.getByText('Preserve all unmentioned surfaces.\nMake only the table metal. Keep the wall unchanged.', {
      exact: true,
    }),
  ).toBeVisible();
  expect(submitted).toEqual([]);
});

test('another variation reuses the selected input clip and waits for an explicit generation', async ({ page }) => {
  const { submitted, unexpected } = await mockWorkspace(page, { complete: true });
  await page.getByLabel('Episode numbers', { exact: true }).fill('1, 3');
  await page.getByLabel('Start at (seconds)').fill('1.5');
  await page.getByLabel('Clip length (seconds)').fill('4');
  await fillStyle(page, 'Make only the table metal.');
  await generateStyle(page);
  await page.getByRole('button', { name: 'Preview episode 3', exact: true }).click();
  await page.getByRole('button', { name: 'New variation from original', exact: true }).click();
  await expect(page.getByLabel('Selected source video')).toHaveAttribute(
    'src',
    /augmentation-1\/augmentation\/clips\/1\?original=true$/,
  );
  await expect(page.getByLabel('Style prompt')).toHaveValue('');
  await fillStyle(page, 'Make only the table metal.');
  await expect(page.getByRole('dialog').getByLabel('Episode numbers', { exact: true })).toHaveCount(0);
  expect(submitted).toHaveLength(1);
  await noOverflow(page);
  await generateStyle(page);
  await expect.poll(() => submitted.length).toBe(2);
  expect(submitted[1]).toEqual({
    ...submitted[0],
    episode_indices: [3],
    source_clip: { job_id: 'augmentation-1', clip_index: 1, variant: 'original' },
  });
  await expect(page.getByLabel('Original episode 3', { exact: true })).toBeVisible();
  await expect(page.getByText('Another variation from', { exact: false }).last()).toBeVisible();
  await page.getByRole('button', { name: 'Run 1', exact: true }).click();
  // Returning to an earlier run keeps that episode's experiment selected.
  await expect(page.getByLabel('Original episode 3', { exact: true })).toBeVisible();
  expect(unexpected).toEqual([]);
});

test('refinement starts from the augmented result and retains its source in history', async ({ page }) => {
  const { submitted, unexpected } = await mockWorkspace(page, { complete: true, priorRuns: true });
  await page.getByRole('button', { name: 'Refine this result', exact: true }).click();
  await expect(page.getByLabel('Selected source video')).toHaveAttribute(
    'src',
    /previous-2\/augmentation\/clips\/0\?original=false$/,
  );
  await expect(page.getByRole('dialog', { name: 'Refine style', exact: true })).toBeVisible();
  await expect(page.getByLabel('Style prompt', { exact: true })).toHaveValue('');
  await expect(page.getByRole('button', { name: 'Generate style', exact: true })).toBeDisabled();
  expect(submitted).toEqual([]);
  await fillStyle(page, 'Make only the tabletop brushed steel.');
  await noOverflow(page);
  await generateStyle(page);
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0]).toMatchObject({
    source_job_id: 'inspected-video',
    episode_indices: [2],
    camera_key: camera,
    preset: 'custom',
    prompt: 'Make only the tabletop brushed steel.',
    source_clip: { job_id: 'previous-2', clip_index: 0, variant: 'augmented' },
  });
  await expect(page.getByLabel('Original episode 2', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Run 2', exact: true })).toBeVisible();
  await noOverflow(page);
  await page.getByText('Choose preview clips', { exact: true }).click();
  await expect(page.getByLabel('Selected source video')).toHaveCount(0);
  await expect(page.getByLabel('Episode numbers', { exact: true })).toHaveValue('2');
  expect(submitted).toHaveLength(1);
  expect(unexpected).toEqual([]);
});

test('keeps the original and all variants together when generating or changing run history', async ({
  page,
}, testInfo) => {
  const fixture = await mockWorkspace(page, { complete: true });
  await fillStyle(page, 'Make the table oak.');
  await generateStyle(page);
  await page.getByRole('button', { name: 'New variation from original', exact: true }).click();
  await fillStyle(page, 'Make only the table metal.');
  await generateStyle(page);
  const gallery = page.getByRole('region', { name: 'Episode experiments', exact: true });
  await expect(gallery.locator('video')).toHaveCount(3);
  await expect(gallery.getByLabel('Original episode 0', { exact: true })).toHaveAttribute(
    'src',
    /augmentation-1.*original=true/,
  );
  await expect(gallery.getByLabel('Run 1 episode 0', { exact: true })).toHaveAttribute(
    'src',
    /augmentation-1.*original=false/,
  );
  await expect(gallery.getByLabel('Augmented episode 0', { exact: true })).toHaveAttribute(
    'src',
    /augmentation-2.*original=false/,
  );
  await page.getByRole('button', { name: 'View run 1', exact: true }).click();
  await expect(gallery.locator('video')).toHaveCount(3);
  await page.reload();
  await openModuleTask(page);
  await expect(page.getByRole('region', { name: 'Episode experiments' }).locator('video')).toHaveCount(3);
  await expect(page.getByRole('button', { name: 'Play all', exact: true })).toBeVisible();
  await noOverflow(page);
  expect(fixture.submitted).toHaveLength(2);
  expect(fixture.unexpected).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath('all-three-variants.png'), fullPage: true });
});

test('plans full episodes for chosen cameras and requires explicit generation after review', async ({ page }) => {
  const fixture = await mockWorkspace(page, { complete: true });
  await fillStyle(page, 'Make the table metal.');
  await generateStyle(page);
  await page.getByRole('button', { name: 'Apply to full dataset', exact: true }).click();
  const form = page.getByRole('region', { name: 'Apply augmentation to dataset', exact: true });
  await expect(form.getByRole('checkbox', { name: 'front', exact: true })).toBeChecked();
  await expect(form.getByRole('checkbox', { name: 'side', exact: true })).not.toBeChecked();
  await expect(form.getByRole('checkbox', { name: 'wrist', exact: true })).toHaveCount(0);
  await form.getByRole('checkbox', { name: 'side', exact: true }).check();
  await form.getByRole('button', { name: 'Review dataset plan', exact: true }).click();
  await expect(form).toContainText('12 full episodes · 2 cameras');
  await expect(form).toContainText('24 generation requests');
  expect(fixture.submitted).toHaveLength(1);
  expect(fixture.plans[0]).toMatchObject({
    episode_indices: [],
    dataset_scope: { camera_keys: [camera, 'observation.images.side'], recipe_job_id: 'augmentation-1' },
  });
  await form.getByRole('checkbox', { name: 'front', exact: true }).uncheck();
  await expect(form.getByRole('button', { name: 'Generate full-dataset augmentation', exact: true })).toHaveCount(0);
  await form.getByRole('button', { name: 'Review dataset plan', exact: true }).click();
  await expect(form).toContainText('12 generation requests');
  await noOverflow(page);
  await form.getByRole('button', { name: 'Generate full-dataset augmentation', exact: true }).click();
  await expect.poll(() => fixture.submitted.length).toBe(2);
  expect(fixture.submitted[1]).toMatchObject({
    source_job_id: 'inspected-video',
    episode_indices: [],
    camera_key: 'observation.images.side',
    start_seconds: 0,
    dataset_scope: {
      recipe_job_id: 'augmentation-1',
      camera_keys: ['observation.images.side'],
      plan_sha256: 'f'.repeat(64),
    },
  });
  expect(fixture.unexpected).toEqual([]);
});

test('collapses variant prompts and adds a fresh style from the same original clip', async ({ page }) => {
  const fixture = await mockWorkspace(page, { complete: true });
  await page.getByLabel('Episode numbers', { exact: true }).fill('1, 3');
  await fillStyle(page, 'Make only the table metal.');
  await generateStyle(page);
  await page.getByRole('button', { name: 'Preview episode 3', exact: true }).click();
  const gallery = page.getByRole('region', { name: 'Episode experiments', exact: true });
  const card = gallery.getByRole('article', { name: 'Run 1 video', exact: true });
  await expect(card.getByText('Make only the table metal.', { exact: true })).not.toBeVisible();
  await card.getByText('Prompt used', { exact: true }).click();
  await expect(card.getByText('Make only the table metal.', { exact: true })).toBeVisible();
  await card.getByText('Prompt used', { exact: true }).click();
  await gallery.getByRole('button', { name: /Add one more/ }).click();
  await expect(page.getByLabel('Style prompt', { exact: true })).toHaveValue('');
  await expect(page.getByRole('dialog', { name: 'Add a style', exact: true })).toBeVisible();
  await expect(page.getByLabel('Selected source video')).toHaveAttribute(
    'src',
    /augmentation-1.*clips\/1\?original=true$/,
  );
  expect(fixture.submitted).toHaveLength(1);
  await expect(page.getByRole('button', { name: 'Generate style', exact: true })).toBeDisabled();
  await fillStyle(page, 'Change only the cup appearance.');
  await generateStyle(page);
  await expect.poll(() => fixture.submitted.length).toBe(2);
  expect(fixture.submitted[1]).toMatchObject({
    episode_indices: [3],
    preset: 'custom',
    prompt: 'Change only the cup appearance.',
    source_clip: { job_id: 'augmentation-1', clip_index: 1, variant: 'original' },
  });
  await expect(gallery.locator('video')).toHaveCount(3);
  await noOverflow(page);
  expect(fixture.unexpected).toEqual([]);
});

test('augmentation opens the dedicated Cloud and APIs page for connection setup', async ({ page }) => {
  const { submitted } = await mockWorkspace(page, { configured: false });
  await page.getByText('Setup details', { exact: true }).click();
  await page.getByRole('button', { name: 'Cloud & APIs', exact: true }).click();
  await expect(page).toHaveURL(/\/cloud-apis\/$/);
  await expect(page.getByRole('heading', { name: 'Cloud & APIs', level: 1, exact: true })).toBeVisible();
  expect(submitted).toEqual([]);
});

test('add one more opens in place, makes the background inert and returns focus without submitting on dismissal', async ({
  page,
}, testInfo) => {
  const fixture = await mockWorkspace(page, { complete: true, priorRuns: true });
  const gallery = page.getByRole('region', { name: 'Episode experiments', exact: true });
  const add = gallery.getByRole('button', { name: /Add one more/ });
  await expect(page.getByLabel('Episode numbers', { exact: true })).not.toBeVisible();
  await add.scrollIntoViewIfNeeded();
  const before = await page.evaluate(() => window.scrollY);
  await add.click();
  const dialog = page.getByRole('dialog', { name: 'Add a style', exact: true });
  await expect(dialog.getByLabel('Style prompt')).toBeFocused();
  await expect(dialog.getByLabel('Selected source video')).toHaveAttribute('src', /previous-2.*original=true$/);
  expect(Math.abs((await page.evaluate(() => window.scrollY)) - before)).toBeLessThan(5);
  for (let i = 0; i < 12; i++) {
    await page.keyboard.press('Tab');
    // Native modal dialogs may move focus into browser chrome, but never into the background page.
    expect(
      await dialog.evaluate(
        (node) =>
          node.matches(':modal') && (document.activeElement === document.body || node.contains(document.activeElement)),
      ),
    ).toBe(true);
  }
  await dialog.getByLabel('Style prompt').fill('A draft that must not be submitted.');
  await page.keyboard.press('Escape');
  await expect(dialog).not.toBeVisible();
  await expect(add).toBeFocused();
  expect(Math.abs((await page.evaluate(() => window.scrollY)) - before)).toBeLessThan(5);
  await add.click();
  await expect(dialog.getByLabel('Style prompt')).toHaveValue('');
  await dialog.getByRole('button', { name: 'Close', exact: true }).click();
  await expect(dialog).not.toBeVisible();
  await expect(add).toBeFocused();
  await add.click();
  await page.mouse.click(2, 2);
  await expect(dialog).not.toBeVisible();
  expect(fixture.submitted).toEqual([]);
  expect(fixture.unexpected).toEqual([]);
  await add.click();
  await page.screenshot({ path: testInfo.outputPath('style-dialog.png') });
});

test('shared playback holds visible final frames, replays and seeks clips of different lengths without feedback', async ({
  page,
}, testInfo) => {
  function media(duration: number, name: string) {
    const path = testInfo.outputPath(name + '.mp4');
    execFileSync('ffmpeg', [
      '-v',
      'error',
      '-y',
      '-f',
      'lavfi',
      '-i',
      'color=c=red:s=96x64:r=30:d=' + duration,
      '-vf',
      "drawbox=color=blue:t=fill:enable='gte(t,0.7)'",
      '-an',
      '-c:v',
      'libx264',
      '-threads',
      '1',
      '-pix_fmt',
      'yuv420p',
      '-movflags',
      '+faststart',
      path,
    ]);
    return readFileSync(path);
  }
  const fixture = await mockWorkspace(page, {
    complete: true,
    media: { original: media(1, 'original'), augmented: media(1.2, 'variant') },
  });
  await fillStyle(page, 'Change only the object color.');
  await generateStyle(page);
  const gallery = page.getByRole('region', { name: 'Episode experiments', exact: true });
  await gallery.getByRole('button', { name: /Add one more/ }).click();
  await fillStyle(page, 'Try a second color.');
  await generateStyle(page);
  const videos = gallery.locator('video');
  await expect(videos).toHaveCount(3);
  await expect
    .poll(() =>
      videos.evaluateAll((nodes) =>
        nodes.every((node) => {
          const v = node as HTMLVideoElement;
          return v.readyState >= 2 && Number.isFinite(v.duration);
        }),
      ),
    )
    .toBe(true);
  await videos.evaluateAll((nodes) =>
    nodes.forEach((node) => {
      const v = node as HTMLVideoElement;
      v.dataset.seekCount = '0';
      v.addEventListener('seeked', () => {
        v.dataset.seekCount = String(Number(v.dataset.seekCount) + 1);
      });
    }),
  );
  async function expectFinalFrames() {
    await expect
      .poll(() =>
        videos.evaluateAll((nodes) =>
          nodes.map((node) => {
            const v = node as HTMLVideoElement;
            return v.paused && !v.seeking && v.currentTime >= v.duration - 0.1 && v.currentTime < v.duration
              ? 'held'
              : {
                  time: v.currentTime,
                  duration: v.duration,
                  paused: v.paused,
                  seeking: v.seeking,
                  seeks: v.dataset.seekCount,
                };
          }),
        ),
      )
      .toEqual(['held', 'held', 'held']);
    const pixels = await videos.evaluateAll((nodes) =>
      nodes.map((node) => {
        const canvas = document.createElement('canvas');
        canvas.width = canvas.height = 1;
        const context = canvas.getContext('2d')!;
        context.drawImage(node as HTMLVideoElement, 0, 0, 1, 1);
        return Array.from(context.getImageData(0, 0, 1, 1).data);
      }),
    );
    for (const [red, green, blue] of pixels) {
      expect(blue).toBeGreaterThan(200);
      expect(red).toBeLessThan(30);
      expect(green).toBeLessThan(30);
    }
  }
  await gallery.getByRole('button', { name: 'Play all', exact: true }).click();
  await expectFinalFrames();
  const seekCounts = await videos.evaluateAll((nodes) =>
    nodes.map((node) => Number((node as HTMLVideoElement).dataset.seekCount)),
  );
  for (const count of seekCounts) {
    expect(count).toBeGreaterThan(0);
    expect(count).toBeLessThanOrEqual(3);
  }
  await gallery.getByRole('button', { name: 'Play all', exact: true }).click();
  await expect
    .poll(() =>
      videos.evaluateAll((nodes) =>
        nodes.every((node) => {
          const v = node as HTMLVideoElement;
          return !v.paused && v.currentTime < 0.7;
        }),
      ),
    )
    .toBe(true);
  await expectFinalFrames();
  await gallery.getByRole('button', { name: 'Restart all', exact: true }).click();
  await expect
    .poll(() =>
      videos.evaluateAll((nodes) =>
        nodes.every((node) => {
          const v = node as HTMLVideoElement;
          return v.paused && !v.seeking && v.currentTime === 0;
        }),
      ),
    )
    .toBe(true);
  await videos.first().evaluate((node) => {
    (node as HTMLVideoElement).currentTime = 0.4;
  });
  await expect
    .poll(() =>
      videos.evaluateAll((nodes) =>
        nodes.every((node) => {
          const v = node as HTMLVideoElement;
          return !v.seeking && Math.abs(v.currentTime - 0.4) < 0.02;
        }),
      ),
    )
    .toBe(true);
  await videos.last().evaluate((node) => {
    const v = node as HTMLVideoElement;
    v.currentTime = v.duration;
  });
  await expectFinalFrames();
  await page.screenshot({ path: testInfo.outputPath('final-frames.png') });
  expect(fixture.submitted).toHaveLength(2);
  expect(fixture.unexpected).toEqual([]);
});
