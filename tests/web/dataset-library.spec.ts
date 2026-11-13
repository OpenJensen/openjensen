import { fulfillApplicationUpdateRead } from './application-update-controls';
import { waitForJob } from './job-waiter';
import { expect, test, type Page } from '@playwright/test';
import { spawnSync } from 'node:child_process';
import { resolve, sep } from 'node:path';
import { mkdir, readdir, readFile, stat, writeFile } from 'node:fs/promises';
import { selectProject } from './project-controls';

test.describe.configure({ timeout: 60000 });
async function ownProject(page: Page, name: string) {
  const response = await page.request.post('/api/v1/projects', { data: { name } });
  expect(response.status()).toBe(201);
  const project = await response.json();
  await page.addInitScript((id) => localStorage.setItem('firebird.project', id), project.id);
  await page.goto('/datasets/');
  await expect(page.getByLabel('Current project')).toHaveAttribute('data-project-id', project.id);
  return project;
}

test('open an imported two-view fixture, save labels, reopen it and reach the dashboard', async ({
  page,
}, testInfo) => {
  // Conversion and immutable-copy validation run real CPU subprocesses.
  test.setTimeout(120_000);
  const project = await ownProject(page, `Label fixture ${testInfo.project.name}`);
  await expect(page.getByRole('heading', { name: 'Saved datasets', exact: true })).toBeVisible();
  await expect(page.getByText('Example datasets', { exact: true })).toHaveCount(0);
  await expect(page.getByRole('status', { name: 'Application API connection' })).toHaveCount(0);
  await expect(page.getByText('LeRobot v2 / v3', { exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Create example', exact: true })).toHaveCount(0);
  await expect(page.getByText('Try the labeling playground', { exact: true })).toHaveCount(0);
  // The legacy API supplies an isolated test fixture; it has no product promotion.
  expect((await page.request.post(`/api/v1/projects/${project.id}/datasets/example`)).status()).toBe(202);
  // Saved but not yet prepared: the card says Imported.
  await expect(page.getByRole('button', { name: 'Open dataset Robot labeling playground' })).toContainText('Imported', {
    timeout: 30_000,
  });
  await page.getByRole('button', { name: 'Open dataset Robot labeling playground' }).click();
  const labels = page.getByRole('region', { name: 'Dataset labeling' });
  await expect(labels.getByRole('heading', { name: 'Robot labeling playground' })).toBeVisible();
  const image = labels.getByRole('img');
  await expect(image).toBeVisible({ timeout: 30_000 });
  await expect
    .poll(() =>
      image.evaluate((node) => (node as HTMLImageElement).complete && (node as HTMLImageElement).naturalWidth > 0),
    )
    .toBe(true);
  const views = labels.locator('input[aria-label^="View name"]');
  await expect(views).toHaveCount(2);
  await views.first().fill('Front camera');
  await labels.getByRole('textbox', { name: 'Image label', exact: true }).fill('Red block approaching the target');
  await expect(labels.getByRole('link', { name: 'Save labels to export' })).toHaveAttribute('aria-disabled', 'true');
  await labels.getByRole('button', { name: 'Save labels', exact: true }).click();
  await expect(labels.getByText('Labels saved', { exact: true })).toBeVisible();
  await labels.getByRole('button', { name: 'Next image', exact: true }).click();
  await expect(labels.getByRole('textbox', { name: 'Image label' })).toHaveValue('');
  await labels.getByRole('button', { name: 'Previous image', exact: true }).click();
  await expect(labels.getByRole('textbox', { name: 'Image label' })).toHaveValue('Red block approaching the target');
  await page.getByRole('button', { name: '← Back to datasets', exact: true }).click();
  await page.getByRole('button', { name: 'Open dataset Robot labeling playground' }).click();
  await expect(page.getByRole('textbox', { name: 'Image label' })).toHaveValue('Red block approaching the target');
  await expect(labels.getByRole('link', { name: 'Export dataset' })).toHaveAttribute(
    'href',
    /\/datasets\/[a-f0-9]{32}\/download$/,
  );
  await page.screenshot({ path: testInfo.outputPath('dataset-labeling.png'), fullPage: true });
  // One page: the dataset prepares itself for training where it is, with no trip back
  // through the import form.
  const readiness = page.getByRole('region', { name: 'Training readiness', exact: true });
  await expect(readiness.getByRole('heading', { name: 'Train on this dataset', exact: true })).toBeVisible();
  await expect(readiness).toContainText('2 episodes');
  const submitted = page.waitForResponse(
    (response) => response.url().endsWith(`/projects/${project.id}/intakes`) && response.request().method() === 'POST',
  );
  await readiness.getByRole('button', { name: 'Prepare for training', exact: true }).click();
  const response = await submitted;
  expect(response.status()).toBe(202);
  const history = await page.request.get(`/api/v1/projects/${project.id}/jobs`);
  expect(history.status()).toBe(200);
  const recorded = await history.json();
  expect(recorded).toHaveLength(1);
  expect(recorded[0].request).toMatchObject({
    library_id: response.request().postDataJSON().library_id,
    snapshot_for_training: true,
  });
  const completed = await waitForJob(page.request, recorded[0].id, 'succeeded');
  expect(completed.project_id).toBe(project.id);
  expect(completed.result.inspection_scope).toBe('complete_snapshot');
  await expect(readiness.getByRole('heading', { name: 'Ready to train', exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Import a dataset' })).toBeHidden();
  await expect(readiness.getByRole('button', { name: 'Train on this dataset' })).toBeEnabled();
  // The full inspection is one link away, and opening it does not inspect again.
  await readiness.getByRole('button', { name: 'Episodes and details', exact: true }).click();
  await expect(page.getByText(/Ready to train · \d+ files checked/)).toBeVisible();
  // One dataset, one name: its details are not titled "Local dataset".
  await expect(page.getByRole('heading', { name: 'Robot labeling playground', exact: true })).toBeVisible();
  await expect(page.getByText('Local dataset', { exact: true })).toHaveCount(0);
  expect(await (await page.request.get(`/api/v1/projects/${project.id}/jobs`)).json()).toHaveLength(1);
  await page.getByRole('button', { name: '← Back to datasets', exact: true }).click();
  // The list, the dataset's page and its details use the same words for the same state.
  const card = page.getByRole('button', { name: 'Open dataset Robot labeling playground' });
  await expect(card).toContainText('Ready to train', { timeout: 20_000 });
  await expect(page.getByText(/Training copy (saved|ready)|Files saved/)).toHaveCount(0);
  await card.click();
  await expect(readiness.getByRole('heading', { name: 'Ready to train', exact: true })).toBeVisible();
  await readiness.getByRole('button', { name: 'Train on this dataset' }).click();
  await expect(page.getByRole('navigation', { name: 'Training setup' })).toBeVisible();
  await expect(page.getByRole('checkbox', { name: 'Robot labeling playground', exact: true })).toBeChecked();
  // Every job this dataset caused is listed under the dataset's name.
  await page.getByRole('link', { name: 'Jobs', exact: true }).click();
  const jobRows = page.locator('[data-job-id]');
  await expect(jobRows.filter({ hasText: 'Inspect dataset' })).toContainText('Robot labeling playground');
  await expect(jobRows.filter({ hasText: 'Local dataset' })).toHaveCount(0);
  await page.getByRole('link', { name: 'Dashboard', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Resources & connections' })).toBeVisible();
  await expect(page.getByRole('link', { name: 'Models', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Datasets', exact: true })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('workspace-dashboard.png'), fullPage: true });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1),
  ).toBe(true);
});

test('native folder selection uploads files, detects records and converts for inspection', async ({
  page,
}, testInfo) => {
  await ownProject(page, `Folder import ${testInfo.project.name}`);
  const source = resolve(testInfo.outputPath('source-folder'));
  await mkdir(source, { recursive: true });
  const worker = resolve('packages/core/src/vla_platform/datasets/import_worker.py');
  const reader = resolve('workers/_cpu_readers/.venv/bin/python');
  const result = spawnSync(
    reader,
    [
      '-I',
      '-c',
      'import runpy,sys;from pathlib import Path;runpy.run_path(sys.argv[1],run_name="adapter")["demo"](Path(sys.argv[2]))',
      worker,
      source,
    ],
    { encoding: 'utf8' },
  );
  expect(result.status, result.stderr).toBe(0);
  await page.getByRole('radio', { name: 'Local files', exact: true }).check();
  await page.getByLabel('Choose dataset folder', { exact: true }).setInputFiles(source);
  const detected = page.getByRole('region', { name: 'Detected dataset' });
  await expect(detected.getByText('Images + records', { exact: true })).toBeVisible();
  await page.getByLabel('Frame rate (FPS)', { exact: true }).fill('6');
  await page.getByLabel('Task description', { exact: true }).fill('Move the block');
  await page.getByRole('button', { name: 'Convert to LeRobot' }).click();
  await expect(detected.getByText(/Imported ·/)).toBeVisible({ timeout: 30_000 });
  await page.getByRole('button', { name: 'Import dataset', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Dataset inspection', exact: true })).toBeVisible();
  // The inspection is titled with the name the folder was saved under.
  await expect(page.getByRole('heading', { name: 'source-folder', exact: true })).toBeVisible();
  await page.getByRole('button', { name: '← Back to datasets', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Open dataset source-folder' })).toBeVisible();
});

test('a one-episode dataset opens with its frames and says what training still needs', async ({ page }, testInfo) => {
  // Regression: the training-copy default turned this inspection into a bare "Failed".
  test.setTimeout(120_000);
  await ownProject(page, `One episode ${testInfo.project.name}`);
  const source = resolve(testInfo.outputPath('first-take'));
  await mkdir(source, { recursive: true });
  const worker = resolve('packages/core/src/vla_platform/datasets/import_worker.py');
  const reader = resolve('workers/_cpu_readers/.venv/bin/python');
  const result = spawnSync(
    reader,
    [
      '-I',
      '-c',
      'import runpy,sys;from pathlib import Path;runpy.run_path(sys.argv[1],run_name="adapter")["demo"](Path(sys.argv[2]),1)',
      worker,
      source,
    ],
    { encoding: 'utf8' },
  );
  expect(result.status, result.stderr).toBe(0);
  await page.getByRole('radio', { name: 'Local files', exact: true }).check();
  await page.getByLabel('Choose dataset folder', { exact: true }).setInputFiles(source);
  const detected = page.getByRole('region', { name: 'Detected dataset' });
  await page.getByLabel('Frame rate (FPS)', { exact: true }).fill('6');
  await page.getByLabel('Task description', { exact: true }).fill('Move the block');
  await page.getByRole('button', { name: 'Convert to LeRobot' }).click();
  await expect(detected.getByText(/Imported ·/)).toBeVisible({ timeout: 30_000 });
  // The default a saved dataset gets: ask for a training copy.
  await page.getByRole('checkbox', { name: 'Prepare for training', exact: true }).check();
  await page.getByRole('button', { name: 'Import dataset', exact: true }).click();
  const inspection = page.getByRole('region', { name: 'Dataset inspection', exact: true });
  await expect(inspection.getByRole('heading', { name: 'first-take', exact: true })).toBeVisible({
    timeout: 30_000,
  });
  await expect(inspection.getByText('Not ready for training yet.')).toBeVisible();
  await expect(inspection.getByText(/at least 2 finalized episodes.*this dataset has 1/)).toBeVisible();
  await expect(inspection.getByText('failed', { exact: true })).toHaveCount(0);
  await expect(inspection.getByRole('alert')).toHaveCount(0);
  await expect(inspection.getByLabel('Preview episode')).toBeVisible({ timeout: 30_000 });
  await expect(inspection.getByRole('img', { name: /episode 1 · frame/ }).first()).toBeVisible();
  await expect(page.getByRole('button', { name: /Train on this dataset/ })).toHaveCount(0);
  await page.screenshot({ path: testInfo.outputPath('one-episode-inspection.png'), fullPage: true });
  // The dataset's own page says the same thing, in the place people look first.
  await page.getByRole('button', { name: '← Back to datasets', exact: true }).click();
  await page.getByRole('button', { name: 'Open dataset first-take' }).click();
  const readiness = page.getByRole('region', { name: 'Training readiness', exact: true });
  await expect(readiness.getByRole('heading', { name: 'Not ready for training yet', exact: true })).toBeVisible();
  await expect(readiness).toContainText('this dataset has 1');
  await expect(readiness.getByRole('button', { name: 'Check again', exact: true })).toBeEnabled();
  await expect(readiness.getByRole('button', { name: /Train on this dataset/ })).toHaveCount(0);
});

test('recordings of one episode each combine into a dataset that is ready to train', async ({ page }, testInfo) => {
  // A supervised collection run saves one episode per dataset. Fifty of those are fifty
  // datasets nobody can train on, until they are combined.
  test.setTimeout(180_000);
  const project = await ownProject(page, `Combine ${testInfo.project.name}`);
  const worker = resolve('packages/core/src/vla_platform/datasets/import_worker.py');
  const reader = resolve('workers/_cpu_readers/.venv/bin/python');
  for (const run of [1, 2, 3]) {
    const source = resolve(testInfo.outputPath(`run-${run}`));
    await mkdir(source, { recursive: true });
    const made = spawnSync(
      reader,
      [
        '-I',
        '-c',
        'import runpy,sys;from pathlib import Path;runpy.run_path(sys.argv[1],run_name="adapter")["demo"](Path(sys.argv[2]),1)',
        worker,
        source,
      ],
      { encoding: 'utf8' },
    );
    expect(made.status, made.stderr).toBe(0);
    // Each run moved differently; identical uploads would be recognised as one dataset.
    const frames = (await readFile(resolve(source, 'frames.jsonl'), 'utf8'))
      .trim()
      .split('\n')
      .map((line) => {
        const row = JSON.parse(line);
        row.action[1] = run;
        return JSON.stringify(row);
      });
    await writeFile(resolve(source, 'frames.jsonl'), frames.join('\n') + '\n');
    const entry = await (
      await page.request.post(`/api/v1/projects/${project.id}/dataset-uploads`, { data: { name: `agent-run-${run}` } })
    ).json();
    for (const name of (await readdir(source, { recursive: true })) as string[]) {
      const path = resolve(source, name);
      if (!(await stat(path)).isFile()) continue;
      const uploaded = await page.request.put(
        `/api/v1/dataset-uploads/${entry.id}/file?path=${encodeURIComponent(name.split(sep).join('/'))}`,
        { data: await readFile(path), headers: { 'content-type': 'application/octet-stream' } },
      );
      expect(uploaded.status(), await uploaded.text()).toBe(200);
    }
    expect((await page.request.post(`/api/v1/dataset-uploads/${entry.id}/finish`)).status()).toBe(200);
    expect(
      (
        await page.request.post(`/api/v1/datasets/${entry.id}/convert`, {
          data: { fps: 6, task: 'Pick up the pen' },
        })
      ).status(),
    ).toBe(202);
    await expect
      .poll(async () => (await (await page.request.get(`/api/v1/datasets/${entry.id}`)).json()).status, {
        timeout: 30_000,
      })
      .toBe('ready');
  }
  await page.reload();
  const banner = page.getByRole('region', { name: 'Combine datasets', exact: true });
  await expect(banner.getByRole('heading', { name: '3 datasets hold one episode each', exact: true })).toBeVisible();
  await banner.getByRole('textbox', { name: 'Name of the combined dataset', exact: true }).fill('Pen pickup, all runs');
  await banner.getByRole('button', { name: 'Combine 3 datasets', exact: true }).click();
  // Straight to the new dataset's own page, where one more click prepares it.
  await expect(page.getByRole('heading', { name: 'Pen pickup, all runs', exact: true })).toBeVisible({
    timeout: 60_000,
  });
  const readiness = page.getByRole('region', { name: 'Training readiness', exact: true });
  await expect(readiness).toContainText('3 episodes');
  await readiness.getByRole('button', { name: 'Prepare for training', exact: true }).click();
  await expect(readiness.getByRole('heading', { name: 'Ready to train', exact: true })).toBeVisible({
    timeout: 60_000,
  });
  await expect(readiness.getByRole('button', { name: 'Train on this dataset' })).toBeEnabled();
  await page.screenshot({ path: testInfo.outputPath('combined-ready.png'), fullPage: true });
  // The originals are kept, and the offer to combine them does not come back.
  await page.getByRole('button', { name: '← Back to datasets', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Open dataset agent-run-1', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Open dataset Pen pickup, all runs', exact: true })).toBeVisible();
  await expect(banner).toHaveCount(0);
});

test('saved Hugging Face inspections reopen without a new inspection request', async ({ page }) => {
  const time = '2026-10-06T08:00:00Z',
    project = 'history-project';
  const writes: string[] = [];
  const profile = {
    source: 'huggingface',
    repo_id: 'our/robot-data',
    revision: 'a'.repeat(40),
    format: 'lerobot_v3',
    robot_type: 'test',
    total_episodes: 4,
    total_frames: 40,
    fps: 10,
    features: { action: { dtype: 'float32', shape: [2] } },
    license: null,
    metadata_sha256: 'b'.repeat(64),
    inspected_at: time,
    warnings: [],
    inspection_scope: 'metadata_only',
  };
  const job = {
    id: 'saved-inspection',
    project_id: project,
    kind: 'dataset.inspect',
    status: 'succeeded',
    created_at: time,
    updated_at: time,
    request: { source: 'huggingface', repo_id: 'our/robot-data', revision: 'a'.repeat(40) },
    result: profile,
  };
  await page.route('**/api/v1/**', async (route) => {
    if (await fulfillApplicationUpdateRead(route)) return;
    const request = route.request(),
      path = new URL(request.url()).pathname;
    if (request.method() !== 'GET') {
      writes.push(path);
      return route.fulfill({ status: 405, json: { detail: 'No writes during reuse' } });
    }
    if (path === '/api/v1/projects')
      return route.fulfill({ json: [{ id: project, name: 'History', created_at: time }] });
    if (path.endsWith('/jobs')) return route.fulfill({ json: [job] });
    if (path === '/api/v1/datasets')
      return route.fulfill({
        json: [
          {
            id: 'inspection:saved-inspection',
            job_id: job.id,
            project_id: project,
            name: profile.repo_id,
            source: 'huggingface',
            status: 'ready',
            created_at: time,
            profile,
          },
        ],
      });
    if (path.endsWith('/episodes'))
      return route.fulfill({ status: 422, json: { detail: 'Preview is offline in this test' } });
    return route.continue();
  });
  await page.goto('/datasets/');
  await page.getByRole('button', { name: 'Open dataset our/robot-data' }).click();
  await expect(page.getByRole('heading', { name: 'our/robot-data', exact: true })).toBeVisible();
  expect(writes).toEqual([]);
});

test('dashboard reports a failed local probe instead of successful resource checks', async ({ page }, testInfo) => {
  const settings = await (await page.request.get('/api/v1/compute-settings')).json();
  settings.local.enabled = false;
  settings.runtimes = [{ provider: 'local', training: true, gpu_name: 'Test GPU' }];
  await page.route('**/api/v1/compute-settings', (route) => route.fulfill({ json: settings }));
  await page.route('**/api/v1/compute-settings/gcp/check', (route) => route.fulfill({ json: settings }));
  await page.route('**/api/v1/compute-settings/local/check', (route) =>
    route.fulfill({ status: 503, json: { detail: 'Local GPU probe failed' } }),
  );
  await ownProject(page, `Resource failure ${testInfo.project.name}`);
  await page.getByRole('link', { name: 'Dashboard', exact: true }).click();
  const local = page
    .getByRole('article')
    .filter({ has: page.getByRole('heading', { name: 'Local compute', exact: true }) });
  await expect(local.getByText('Disabled', { exact: true })).toBeVisible();
  await expect(local.locator('.resource-state')).not.toHaveClass(/ready/);
  await page.getByRole('button', { name: 'Check resources', exact: true }).click();
  await expect(local.getByText('Check failed', { exact: true })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Resources & connections' }).getByRole('alert')).toContainText(
    'Local GPU probe failed',
  );
  await expect(page.getByText('Resources checked.', { exact: true })).toHaveCount(0);
});

for (const available of [true, false]) {
  test(`saved Hub cards ${available ? 'show a paused frame at the inspected episode boundary' : 'explain unavailable previews'}`, async ({
    page,
  }, testInfo) => {
    const project = 'preview-project',
      time = '2026-10-06T08:00:00Z',
      job = 'preview-inspection';
    const profile = {
      source: 'huggingface',
      repo_id: 'our/camera-data',
      revision: 'a'.repeat(40),
      format: 'lerobot_v3',
      total_episodes: 8,
      total_frames: 80,
      fps: 5,
      features: { 'observation.images.front': { dtype: 'video', shape: [64, 64, 3] } },
      warnings: [],
      inspection_scope: 'metadata_only',
    };
    const media = resolve(testInfo.outputPath('preview.mp4'));
    if (available) {
      await mkdir(resolve(testInfo.outputPath('.')), { recursive: true });
      const result = spawnSync(
        'ffmpeg',
        [
          '-v',
          'error',
          '-f',
          'lavfi',
          '-i',
          'color=c=blue:s=64x64:r=5:d=3',
          '-c:v',
          'libx264',
          '-pix_fmt',
          'yuv420p',
          '-movflags',
          '+faststart',
          media,
        ],
        { encoding: 'utf8' },
      );
      expect(result.status, result.stderr).toBe(0);
      const bytes = await readFile(media);
      await page.route('**/api/v1/fixture-preview.mp4', async (route) => {
        const range = /^bytes=(\d+)-(\d*)$/.exec(route.request().headers()['range'] ?? '');
        const start = range ? Number(range[1]) : 0;
        const end = range?.[2] ? Math.min(Number(range[2]), bytes.length - 1) : bytes.length - 1;
        await route.fulfill({
          status: range ? 206 : 200,
          contentType: 'video/mp4',
          body: bytes.subarray(start, end + 1),
          headers: {
            'Accept-Ranges': 'bytes',
            'Content-Length': String(end - start + 1),
            ...(range ? { 'Content-Range': `bytes ${start}-${end}/${bytes.length}` } : {}),
          },
        });
      });
    }
    const episode = { episode_index: 7, frame_count: 5, duration_seconds: 1, tasks: ['Pick up'] };
    const requests: string[] = [];
    await page.route('**/api/v1/**', async (route) => {
      if (await fulfillApplicationUpdateRead(route)) return;
      const path = new URL(route.request().url()).pathname;
      if (path === '/api/v1/fixture-preview.mp4') return route.fallback();
      if (path === '/api/v1/projects')
        return route.fulfill({ json: [{ id: project, name: 'Preview library', created_at: time }] });
      if (path.endsWith('/jobs')) return route.fulfill({ json: [] });
      if (path === '/api/v1/datasets')
        return route.fulfill({
          json: [
            {
              id: `inspection:${job}`,
              job_id: job,
              project_id: project,
              name: profile.repo_id,
              source: 'huggingface',
              status: 'ready',
              created_at: time,
              profile,
            },
          ],
        });
      if (path.endsWith('/cover')) {
        requests.push(route.request().url());
        return available
          ? route.fulfill({
              json: {
                ...episode,
                repo_id: profile.repo_id,
                revision: profile.revision,
                cameras: [
                  {
                    key: 'observation.images.front',
                    url: '/api/v1/fixture-preview.mp4',
                    start_seconds: 1,
                    end_seconds: 2,
                    width: 64,
                    height: 64,
                    fps: 5,
                  },
                ],
                samples: [],
                warnings: [],
                action_names: [],
                state_names: [],
              },
            })
          : route.fulfill({ status: 422, json: { detail: 'Preview unavailable' } });
      }
      return route.continue();
    });
    await page.goto('/datasets/');
    const card = page.getByRole('button', { name: 'Open dataset our/camera-data' });
    if (available) {
      const video = card.locator('video');
      await expect(video).toHaveClass('loaded');
      await expect.poll(() => video.evaluate((node) => (node as HTMLVideoElement).currentTime)).toBeCloseTo(1, 1);
      expect(await video.evaluate((node) => (node as HTMLVideoElement).paused)).toBe(true);
      await expect(card.getByText('Preview unavailable', { exact: true })).toHaveCount(0);
    } else await expect(card.getByText('Preview unavailable', { exact: true })).toBeVisible();
    expect(requests).toHaveLength(1);
    expect(new URL(requests[0]).pathname).toContain('/cover');
    if (available && testInfo.project.name === 'desktop') {
      const video = card.locator('video');
      await card.hover();
      await expect.poll(() => video.evaluate((node) => (node as HTMLVideoElement).paused)).toBe(false);
      await page.waitForTimeout(1200); // Exercise the exclusive end boundary across a full loop.
      expect(await video.evaluate((node) => (node as HTMLVideoElement).currentTime)).toBeLessThan(2);
      await page.getByRole('heading', { name: 'Datasets', exact: true }).hover();
      await expect.poll(() => video.evaluate((node) => (node as HTMLVideoElement).paused)).toBe(true);
    }
    await page.getByRole('link', { name: 'Fine-tune', exact: true }).click();
    await page.getByRole('link', { name: 'Datasets', exact: true }).click();
    if (available) {
      expect(requests).toHaveLength(1);
      await expect(card.locator('video')).toHaveClass('loaded');
      await expect
        .poll(() => card.locator('video').evaluate((node) => (node as HTMLVideoElement).currentTime))
        .toBeCloseTo(1, 1);
      await page.getByRole('searchbox', { name: 'Search my datasets', exact: true }).fill('no matching dataset');
      await expect(page.getByText('No matching datasets', { exact: true })).toBeVisible();
      await page.getByRole('searchbox', { name: 'Search my datasets', exact: true }).fill('');
      await expect(card.locator('video')).toHaveClass('loaded');
      await expect
        .poll(() => card.locator('video').evaluate((node) => (node as HTMLVideoElement).currentTime))
        .toBeCloseTo(1, 1);
      expect(requests).toHaveLength(1);
    } else await expect(card.getByText('Preview unavailable', { exact: true })).toBeVisible();
  });
}

test('dashboard uses the current project library scope, and inspection is reached only through its card', async ({
  page,
}) => {
  const time = '2026-10-07T00:00:00Z',
    project = 'alpha';
  const writes: string[] = [];
  const reads: string[] = [];
  const profile = {
    source: 'huggingface',
    repo_id: 'our/robot-data',
    revision: 'a'.repeat(40),
    format: 'lerobot_v3',
    robot_type: 'test',
    total_episodes: 4,
    total_frames: 40,
    fps: 10,
    features: { action: { dtype: 'float32', shape: [2] } },
    license: null,
    metadata_sha256: 'b'.repeat(64),
    inspected_at: time,
    warnings: [],
    inspection_scope: 'metadata_only',
  };
  const datasets = [0, 1, 2, 3, 4].map((index) => ({
    id: `inspection:data-${index}`,
    job_id: `data-${index}`,
    project_id: index < 3 ? project : 'beta',
    name: `our/data-${index}`,
    source: 'huggingface',
    status: 'ready',
    created_at: time,
    profile: { ...profile, repo_id: `our/data-${index}` },
  }));
  await page.route('**/api/v1/**', async (route) => {
    if (await fulfillApplicationUpdateRead(route)) return;
    const request = route.request(),
      url = new URL(request.url()),
      path = url.pathname;
    if (request.method() !== 'GET') {
      writes.push(path);
      return route.fulfill({ status: 405, json: { detail: 'Navigation must not create jobs' } });
    }
    if (path === '/api/v1/projects')
      return route.fulfill({
        json: [
          { id: project, name: 'Selected', created_at: time },
          { id: 'beta', name: 'Other', created_at: time },
        ],
      });
    if (path === '/api/v1/datasets') {
      reads.push(url.searchParams.get('project_id') ?? 'all');
      return route.fulfill({ json: datasets });
    } // Includes foreign rows to exercise the client boundary too.
    if (path.endsWith('/jobs'))
      return route.fulfill({
        json: datasets.map((entry) => ({
          id: entry.job_id,
          project_id: entry.project_id,
          kind: 'dataset.inspect',
          status: 'succeeded',
          created_at: time,
          updated_at: time,
          request: { source: 'huggingface', repo_id: entry.name },
          result: entry.profile,
        })),
      });
    if (path.endsWith('/artifacts')) return route.fulfill({ json: [] });
    if (path.endsWith('/episodes') || path.endsWith('/cover'))
      return route.fulfill({ status: 422, json: { detail: 'Preview offline in this navigation test' } });
    return route.continue();
  });
  await page.goto('/datasets/');
  await expect(page.getByRole('navigation', { name: 'Dataset views' })).toHaveCount(0);
  await expect(page.getByRole('button', { name: /Open dataset/ })).toHaveCount(3);
  await page.getByRole('link', { name: 'Dashboard', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Datasets', exact: true })).toContainText('3 saved datasets');
  await expect(page.getByRole('region', { name: 'Recent datasets' }).getByText('our/data-3')).toHaveCount(0);
  await page.getByRole('button', { name: 'Datasets', exact: true }).click();
  await page.getByRole('button', { name: 'Open dataset our/data-0', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Dataset inspection', exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Explore dataset', exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: '← Back to datasets', exact: true })).toHaveCount(1);
  await page.getByRole('button', { name: '← Back to datasets', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Saved datasets', exact: true })).toBeVisible();
  expect(reads).not.toContain('all');
  expect(reads).toContain(project);
  expect(writes).toEqual([]);
});

test('dashboard resource checks stay in the same position on a fresh visit and after loading dataset styles', async ({
  page,
}, testInfo) => {
  const project = await (
    await page.request.post('/api/v1/projects', { data: { name: `Stable dashboard ${testInfo.project.name}` } })
  ).json();
  await page.addInitScript((id) => localStorage.setItem('firebird.project', id), project.id);
  const settings = await (await page.request.get('/api/v1/compute-settings')).json();
  await page.route('**/api/v1/compute-settings/gcp/check', (route) => route.fulfill({ json: settings }));
  await page.route('**/api/v1/compute-settings/local/check', (route) =>
    route.fulfill({
      json: {
        host: { name: 'Test host', platform: 'test', architecture: 'test' },
        checked_at: new Date().toISOString(),
        status: 'unavailable',
        message: 'No GPU in this fixture',
        candidates: [],
        issues: [],
      },
    }),
  );
  await page.goto('/dashboard/');
  const button = page.getByRole('button', { name: 'Check resources', exact: true });
  await expect(button).toBeVisible();
  // Compare within the card: clicking an off-screen button normally scrolls mobile pages.
  const position = () =>
    button.evaluate((node) => {
      const button = node.getBoundingClientRect(),
        card = node.closest('.dashboard-resources')!.getBoundingClientRect();
      return { x: button.x - card.x, y: button.y - card.y };
    });
  const before = await position();
  const initial = (await button.boundingBox())!;
  const heading = (await page.getByRole('heading', { name: 'Resources & connections', exact: true }).boundingBox())!;
  if (testInfo.project.name === 'desktop') {
    expect(initial.x).toBeGreaterThan(heading.x + heading.width);
    expect(Math.abs(initial.y - heading.y)).toBeLessThan(2);
  }
  await button.click();
  await expect(page.getByText('Resources checked.', { exact: true })).toBeVisible();
  const after = await position();
  expect(Math.abs(after.x - before.x)).toBeLessThan(1);
  expect(Math.abs(after.y - before.y)).toBeLessThan(1);
  await page.getByRole('link', { name: 'Datasets', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Saved datasets', exact: true })).toBeVisible();
  await page.getByRole('link', { name: 'Dashboard', exact: true }).click();
  const revisited = await position();
  expect(Math.abs(revisited.x - before.x)).toBeLessThan(1);
  expect(Math.abs(revisited.y - before.y)).toBeLessThan(1);
  await page.screenshot({ path: testInfo.outputPath('stable-resource-check.png'), fullPage: true });
});

test('Prepare for training keeps its checkbox aligned and explains the copy and complete episodes on demand', async ({
  page,
}, testInfo) => {
  await ownProject(page, `Training copy help ${testInfo.project.name}`);
  await page.getByRole('radio', { name: 'Local files', exact: true }).check();
  const option = page.getByRole('checkbox', { name: 'Prepare for training', exact: true });
  const help = page.getByRole('button', { name: 'Help for Prepare for training', exact: true });
  const tooltip = page.getByRole('tooltip');
  await expect(tooltip).toBeHidden();
  await expect(option).not.toBeChecked();
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth))
    .toBeLessThanOrEqual(1);
  await help.focus();
  await expect(tooltip).toBeVisible();
  await expect(tooltip).toContainText('keeps a copy for training in the workspace’s dataset-snapshots/ folder');
  await expect(tooltip).toContainText('Training needs at least two complete episodes.');
  await expect(option).not.toBeChecked();
  const input = (await option.boundingBox())!,
    label = (await option.locator('..').boundingBox())!;
  expect(Math.abs(input.y + input.height / 2 - label.y - label.height / 2)).toBeLessThan(1);
  const bounds = (await tooltip.boundingBox())!;
  expect(bounds.x).toBeGreaterThanOrEqual(0);
  expect(bounds.x + bounds.width).toBeLessThanOrEqual(page.viewportSize()!.width);
  await page.screenshot({ path: testInfo.outputPath('training-copy-help.png'), fullPage: true });
  await page.getByRole('heading', { name: 'Datasets', exact: true }).click();
  await expect(tooltip).toBeHidden();
});

async function importRecoveryFixture(page: Page) {
  const project = 'label-drafts',
    time = '2026-10-07T00:00:00Z',
    id = 'a'.repeat(32);
  const entry: any = {
    id,
    project_id: project,
    name: 'Saved local data',
    source: 'local',
    status: 'ready',
    created_at: time,
    annotation_revision: 0,
    annotations: { views: {}, frames: {} },
    converted: { format: 'lerobot_v3', episodes: 2, frames: 4, fps: 10, cameras: ['front'] },
    detection: {
      format: 'lerobot_v3',
      convertible: true,
      cameras: ['front'],
      mapping: {},
      file_count: 2,
      total_bytes: 20,
    },
  };
  const state = { entries: [entry], revision: 0, deletes: [] as string[], saves: [] as any[] };
  await page.route('**/api/v1/**', async (route) => {
    if (await fulfillApplicationUpdateRead(route)) return;
    const request = route.request(),
      url = new URL(request.url()),
      path = url.pathname;
    if (path === '/api/v1/projects')
      return route.fulfill({
        json: [
          { id: project, name: 'Draft project', created_at: time },
          { id: 'other', name: 'Other project', created_at: time },
        ],
      });
    if (path === '/api/v1/datasets')
      return route.fulfill({
        json: state.entries.filter(
          (value) => value.project_id === null || value.project_id === url.searchParams.get('project_id'),
        ),
      });
    if (path === `/api/v1/datasets/${id}/annotations`) {
      const body = request.postDataJSON();
      state.saves.push(body);
      if (body.revision !== state.revision)
        return route.fulfill({
          status: 409,
          json: { detail: 'Labels changed in another window. Reload before saving.' },
        });
      entry.annotations = { views: body.views, frames: body.frames };
      entry.annotation_revision = ++state.revision;
      return route.fulfill({ json: entry });
    }
    if (path === `/api/v1/datasets/${id}/samples`) return route.fulfill({ json: [] });
    if (path.startsWith('/api/v1/datasets/'))
      return route.fulfill({ json: state.entries.find((value) => path.endsWith(value.id)) ?? entry });
    if (path.startsWith('/api/v1/dataset-uploads/') && request.method() === 'DELETE') {
      state.deletes.push(url.pathname + url.search);
      const upload = path.split('/').at(-1);
      state.entries = state.entries.filter((value) => value.id !== upload);
      return route.fulfill({ json: { discarded: true, id: upload } });
    }
    if (path.endsWith('/jobs') || path.endsWith('/artifacts')) return route.fulfill({ json: [] });
    return route.continue();
  });
  await page.addInitScript((id) => localStorage.setItem('firebird.project', id), project);
  await page.goto('/datasets/');
  return { state, entry, id, project };
}

test('unsaved labels survive library, workflow, project and reload navigation with their original revision', async ({
  page,
}) => {
  const { state, entry } = await importRecoveryFixture(page);
  page.on('dialog', (dialog) => void dialog.accept());
  const open = () => page.getByRole('button', { name: 'Open dataset Saved local data', exact: true }).click();
  const field = () => page.getByRole('textbox', { name: 'View name front', exact: true });
  await open();
  await field().fill('Unsaved front camera');
  await page.getByRole('button', { name: '← Back to datasets', exact: true }).click();
  await open();
  await expect(field()).toHaveValue('Unsaved front camera');
  await page.getByRole('link', { name: 'Fine-tune', exact: true }).click();
  await page.getByRole('link', { name: 'Datasets', exact: true }).click();
  await open();
  await expect(field()).toHaveValue('Unsaved front camera');
  await selectProject(page, 'other');
  await expect(page.getByRole('textbox', { name: 'View name front', exact: true })).toHaveCount(0);
  await selectProject(page, 'label-drafts');
  await open();
  await expect(field()).toHaveValue('Unsaved front camera');
  state.revision = 1;
  entry.annotation_revision = 1;
  entry.annotations = { views: { front: 'Changed in another window' }, frames: {} };
  await page.reload();
  await open();
  await expect(field()).toHaveValue('Unsaved front camera');
  await page.getByRole('button', { name: 'Save labels', exact: true }).click();
  await expect(page.getByRole('alert').filter({ hasText: 'Labels changed in another window' })).toBeVisible();
  expect(state.saves).toHaveLength(1);
  expect(state.saves[0].revision).toBe(0);
  await page.getByRole('button', { name: 'Reload saved labels', exact: true }).click();
  await expect(field()).toHaveValue('Changed in another window');
  await page.getByRole('button', { name: '← Back to datasets', exact: true }).click();
  await open();
  await expect(field()).toHaveValue('Changed in another window');
});

test('unavailable browser draft storage preserves labels during in-app navigation and gives an actionable warning', async ({
  page,
}) => {
  await page.addInitScript(() =>
    Object.defineProperty(window, 'sessionStorage', {
      value: {
        getItem() {
          throw new Error('denied');
        },
        setItem() {
          throw new Error('denied');
        },
        removeItem() {
          throw new Error('denied');
        },
      },
    }),
  );
  await importRecoveryFixture(page);
  await page.getByRole('button', { name: 'Open dataset Saved local data', exact: true }).click();
  const field = page.getByRole('textbox', { name: 'View name front', exact: true });
  await field.fill('Draft without storage');
  await expect(page.getByRole('alert').filter({ hasText: 'Browser storage is unavailable' })).toBeVisible();
  await page.getByRole('button', { name: '← Back to datasets', exact: true }).click();
  await page.getByRole('button', { name: 'Open dataset Saved local data', exact: true }).click();
  await expect(field).toHaveValue('Draft without storage');
  await page.getByRole('button', { name: 'Save labels', exact: true }).click();
  await expect(page.getByText('Labels saved', { exact: true })).toBeVisible();
});

test('unfinished uploads are visible and explicit discard targets only that project upload', async ({ page }) => {
  const { state, project } = await importRecoveryFixture(page);
  const id = 'b'.repeat(32);
  state.entries.push({
    id,
    project_id: project,
    name: 'Interrupted folder',
    source: 'local',
    status: 'uploading',
    created_at: new Date().toISOString(),
    uploaded_files: 3,
    total_bytes: 100,
    can_discard: true,
    error: 'Upload is unfinished.',
  });
  await page.getByRole('button', { name: 'Refresh datasets', exact: true }).click();
  const card = page.getByRole('button', { name: 'Open dataset Interrupted folder', exact: true });
  await expect(card).toContainText('3 files uploaded');
  await card.click();
  await expect(page.getByRole('region', { name: 'Detected dataset' })).toContainText('This upload is unfinished');
  await expect(page.getByRole('button', { name: 'Import dataset', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Discard unfinished upload', exact: true }).click();
  expect(state.deletes).toEqual([]);
  await page.getByRole('button', { name: 'Keep upload', exact: true }).click();
  expect(state.deletes).toEqual([]);
  await page.getByRole('button', { name: 'Discard unfinished upload', exact: true }).click();
  await page.getByRole('button', { name: 'Confirm discard upload', exact: true }).click();
  await expect(card).toHaveCount(0);
  expect(state.deletes).toEqual([`/api/v1/dataset-uploads/${id}?project_id=${project}`]);
  await expect(page.getByRole('button', { name: 'Open dataset Saved local data', exact: true })).toBeVisible();
  state.entries.push({
    id: 'c'.repeat(32),
    project_id: project,
    name: 'Damaged record',
    source: 'local',
    status: 'unavailable',
    created_at: '',
    can_discard: false,
    error: 'Record damaged.',
  });
  await page.getByRole('button', { name: 'Refresh datasets', exact: true }).click();
  await page.getByRole('button', { name: 'Open dataset Damaged record', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Discard unfinished upload', exact: true })).toHaveCount(0);
  await expect(page.getByRole('alert').filter({ hasText: 'files are preserved' })).toBeVisible();
});

test('unknown-owner records are workspace repair notices with no project or discard action', async ({ page }) => {
  const { state } = await importRecoveryFixture(page);
  state.entries.push({
    id: 'd'.repeat(32),
    project_id: null,
    name: 'Unavailable saved dataset',
    status: 'unavailable',
    created_at: '',
    can_discard: false,
    error: 'Record invalid; original files preserved.',
  });
  await page.getByRole('button', { name: 'Refresh datasets', exact: true }).click();
  const notice = page.getByRole('alert', { name: 'Workspace dataset repair notice' });
  await expect(notice).toContainText('project ownership could not be verified');
  await expect(notice).toContainText('original files preserved');
  await expect(notice.getByRole('button')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Open dataset Unavailable saved dataset', exact: true })).toHaveCount(
    0,
  );
  expect(state.deletes).toEqual([]);
  await page.getByRole('link', { name: 'Dashboard', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Datasets', exact: true })).toContainText(/1 saved dataset$/);
});

test('late label-save acknowledgement does not erase a newer draft after reopening', async ({ page }) => {
  const { entry, id } = await importRecoveryFixture(page);
  let release!: () => void;
  const waiting = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route(`**/api/v1/datasets/${id}/annotations`, async (route) => {
    const body = route.request().postDataJSON();
    await waiting;
    entry.annotations = { views: body.views, frames: body.frames };
    entry.annotation_revision = 1;
    await route.fulfill({ json: entry });
  });
  const open = () => page.getByRole('button', { name: 'Open dataset Saved local data', exact: true }).click();
  const field = () => page.getByRole('textbox', { name: 'View name front', exact: true });
  await open();
  await field().fill('First submitted draft');
  await page.getByRole('button', { name: 'Save labels', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Saving…', exact: true })).toBeVisible();
  await page.getByRole('button', { name: '← Back to datasets', exact: true }).click();
  await open();
  await field().fill('Newer unsaved draft');
  const response = page.waitForResponse((value) => value.url().endsWith(`/datasets/${id}/annotations`));
  release();
  await response;
  await page.getByRole('button', { name: '← Back to datasets', exact: true }).click();
  await open();
  await expect(field()).toHaveValue('Newer unsaved draft');
});

test('accepted conversion survives an earlier GET and keeps observing until ready without another POST', async ({
  page,
}) => {
  const { entry, id } = await importRecoveryFixture(page);
  entry.status = 'detected';
  entry.converted = undefined;
  entry.detection.format = 'image_records';
  const earlier = structuredClone(entry);
  let releaseEarlier!: () => void,
    reads = 0,
    posts = 0;
  const waiting = new Promise<void>((resolve) => {
    releaseEarlier = resolve;
  });
  await page.route(`**/api/v1/datasets/${id}`, async (route) => {
    reads++;
    if (reads === 1) {
      await waiting;
      await route.fulfill({ json: earlier }).catch(() => {});
      return;
    }
    // The first fresh read still reports detection; polling must not stop there.
    if (reads === 2) return route.fulfill({ json: earlier });
    return route.fulfill({
      json: {
        ...earlier,
        status: 'ready',
        converted: { format: 'lerobot_v3', episodes: 2, frames: 4, fps: 10, cameras: ['front'] },
      },
    });
  });
  await page.route(`**/api/v1/datasets/${id}/convert`, (route) => {
    posts++;
    return route.fulfill({ status: 202, json: { ...earlier, status: 'converting' } });
  });
  try {
    await page.getByRole('button', { name: 'Refresh datasets', exact: true }).click();
    await page.getByRole('button', { name: 'Open dataset Saved local data', exact: true }).click();
    await expect.poll(() => reads).toBe(1);
    await page.getByRole('button', { name: 'Convert to LeRobot', exact: true }).click();
    await expect.poll(() => reads).toBeGreaterThanOrEqual(2);
    releaseEarlier();
    await expect(page.getByRole('region', { name: 'Detected dataset' }).getByText(/Imported ·/)).toBeVisible();
    expect(reads).toBeGreaterThanOrEqual(3);
    expect(posts).toBe(1);
    await expect(page.getByRole('button', { name: 'Import dataset', exact: true })).toBeEnabled();
  } finally {
    releaseEarlier();
  }
});

test('upload detection initializes mapping and FPS once without replacing edits on a later read', async ({ page }) => {
  const { state, project } = await importRecoveryFixture(page),
    id = 'b'.repeat(32);
  state.entries = [];
  const uploading: any = {
    id,
    project_id: project,
    name: 'late-detection.hdf5',
    source: 'local',
    status: 'uploading',
    created_at: '2026-10-12T00:00:00Z',
    uploaded_files: 0,
    total_bytes: 0,
    can_discard: true,
  };
  const detected: any = {
    ...uploading,
    status: 'detected',
    can_discard: false,
    detection: {
      format: 'image_records',
      convertible: true,
      sha256: 'c'.repeat(64),
      file_count: 1,
      total_bytes: 4,
      fps: 12,
      cameras: ['front'],
      mapping: { action: 'controls', state: ['position.x', 'position.y'], cameras: ['front'], table: 'records.jsonl' },
    },
  };
  let saved = uploading,
    releaseFile!: () => void,
    refresh = false,
    refreshReads = 0;
  const fileWaiting = new Promise<void>((resolve) => {
    releaseFile = resolve;
  });
  const conversions: any[] = [];
  await page.route(`**/api/v1/projects/${project}/dataset-uploads`, (route) => {
    state.entries = [uploading];
    return route.fulfill({ status: 201, json: uploading });
  });
  await page.route(`**/api/v1/dataset-uploads/${id}/file?*`, async (route) => {
    await fileWaiting;
    return route.fulfill({ json: { uploaded: true } });
  });
  await page.route(`**/api/v1/dataset-uploads/${id}/finish`, (route) => {
    saved = detected;
    state.entries = [detected];
    return route.fulfill({ json: detected });
  });
  await page.route(`**/api/v1/datasets/${id}`, (route) => {
    if (refresh && saved.status === 'detected') {
      refreshReads++;
      return route.fulfill({
        json: {
          ...saved,
          detection: {
            ...detected.detection,
            fps: 24,
            mapping: { ...detected.detection.mapping, action: 'new_server_action', state: ['new_server_state'] },
          },
        },
      });
    }
    return route.fulfill({ json: saved });
  });
  await page.route(`**/api/v1/datasets/${id}/convert`, (route) => {
    conversions.push(route.request().postDataJSON());
    saved = {
      ...detected,
      status: 'ready',
      converted: { format: 'lerobot_v3', episodes: 2, frames: 4, fps: 9, cameras: ['front'] },
    };
    state.entries = [saved];
    return route.fulfill({ status: 202, json: { ...detected, status: 'converting' } });
  });
  try {
    await page.getByRole('button', { name: 'Refresh datasets', exact: true }).click();
    await page.getByRole('radio', { name: 'Local files', exact: true }).check();
    await page.getByLabel('Choose dataset archive or HDF5', { exact: true }).setInputFiles({
      name: 'late-detection.hdf5',
      mimeType: 'application/octet-stream',
      buffer: Buffer.from('data'),
    });
    const region = page.getByRole('region', { name: 'Detected dataset' });
    await expect(region.getByText(/This upload is unfinished/)).toBeVisible();
    await expect(page.getByLabel('Frame rate (FPS)', { exact: true })).toHaveCount(0);
    releaseFile();
    const fps = page.getByLabel('Frame rate (FPS)', { exact: true });
    await expect(fps).toHaveValue('12');
    await region.getByText('Field mapping & robot', { exact: true }).click();
    await expect(page.getByLabel('Action field', { exact: true })).toHaveValue('controls');
    await expect(page.getByLabel('State fields (in order, comma separated)', { exact: true })).toHaveValue(
      'position.x, position.y',
    );
    await fps.fill('9');
    await page.getByLabel('Task description', { exact: true }).fill('User task');
    refresh = true;
    await expect
      .poll(
        async () => {
          await page.evaluate(() => window.dispatchEvent(new Event('visibilitychange')));
          return refreshReads;
        },
        { timeout: 10_000 },
      )
      .toBeGreaterThan(0);
    await expect(fps).toHaveValue('9');
    await expect(page.getByLabel('Action field', { exact: true })).toHaveValue('controls');
    await expect(page.getByLabel('State fields (in order, comma separated)', { exact: true })).toHaveValue(
      'position.x, position.y',
    );
    await page.getByRole('button', { name: 'Convert to LeRobot', exact: true }).click();
    await expect(region.getByText(/Imported ·/)).toBeVisible();
    expect(conversions).toEqual([
      {
        fps: 9,
        task: 'User task',
        robot_type: 'unspecified',
        mapping: {
          action: 'controls',
          state: ['position.x', 'position.y'],
          cameras: ['front'],
          table: 'records.jsonl',
        },
      },
    ]);
  } finally {
    releaseFile();
  }
});

test('recorded local frames play and seek in inspection and training without Hub previews', async ({
  page,
}, testInfo) => {
  test.setTimeout(120000);
  const project = await ownProject(page, `Local playback ${testInfo.project.name}`);
  expect((await page.request.post(`/api/v1/projects/${project.id}/datasets/example`)).status()).toBe(202);
  let entry: any;
  await expect
    .poll(
      async () => {
        const rows = await (await page.request.get(`/api/v1/datasets?project_id=${project.id}`)).json();
        entry = rows.find((row: any) => row.status === 'ready');
        return !!entry;
      },
      { timeout: 30000 },
    )
    .toBe(true);
  const original = await (await page.request.get(`/api/v1/datasets/${entry.id}/samples`)).json();
  const frames = original.flatMap((sample: any) =>
    [0, 1, 2].map((frame) => ({
      ...sample,
      frame_index: frame,
      timestamp: frame / 6,
      state: [frame, 1],
      action: [frame + 1, 2],
    })),
  );
  await page.route(`**/api/v1/datasets/${entry.id}/samples`, (route) => route.fulfill({ json: frames }));
  const unexpected: string[] = [];
  await page.route('**/api/v1/jobs/*/episodes**', async (route) => {
    unexpected.push(route.request().url());
    await route.fulfill({ status: 422, json: { detail: 'Local playback must use saved frames' } });
  });
  await page.reload();
  await page.getByRole('button', { name: 'Open dataset Robot labeling playground' }).click();
  const readiness = page.getByRole('region', { name: 'Training readiness', exact: true });
  await readiness.getByRole('button', { name: 'Prepare for training', exact: true }).click();
  await expect(readiness.getByRole('heading', { name: 'Ready to train', exact: true })).toBeVisible({ timeout: 30000 });
  await readiness.getByRole('button', { name: 'Episodes and details', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Dataset inspection', exact: true })).toBeVisible();
  await expect(page.getByLabel('Preview episode')).toBeVisible({ timeout: 30000 });
  await page.getByLabel('Preview episode').selectOption('1');
  await page.getByRole('slider', { name: 'Episode playback position' }).press('End');
  const picture = page.getByRole('img', { name: 'observation.images.front · episode 2 · frame 2', exact: true });
  await expect(picture).toBeVisible();
  await expect
    .poll(() => picture.evaluate((node: HTMLImageElement) => node.complete && node.naturalWidth > 0))
    .toBe(true);
  await page.getByRole('button', { name: 'Play episode', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Play episode', exact: true })).toBeVisible();
  await expect(picture).toBeVisible();
  await page.getByRole('link', { name: 'Fine-tune', exact: true }).click();
  await page.getByRole('button', { name: 'Start a new fine-tuning', exact: true }).click();
  const training = page.getByRole('region', { name: 'Dataset', exact: true });
  await expect(training).toBeVisible();
  await training.getByLabel('Preview episode').selectOption('1');
  await training.getByRole('slider', { name: 'Episode playback position' }).press('End');
  await expect(
    training.getByRole('img', { name: 'observation.images.front · episode 2 · frame 2', exact: true }),
  ).toBeVisible();
  expect(unexpected).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath('local-recorded-playback.png'), fullPage: true });
});
