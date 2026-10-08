import { expect, test, type Page } from '@playwright/test';

async function chooseOnlyModel(page: Page, name: string) {
  const choices = page.getByRole('group', { name: 'Base model', exact: true });
  for (const choice of await choices.locator('input:checked').all()) await choice.uncheck();
  await choices.getByRole('checkbox', { name, exact: true }).check();
}

async function openGpuPicker(page: Page, local = false) {
  const timestamp = '2026-09-26T12:00:00Z';
  const writes: string[] = [];
  const unexpected: string[] = [];
  const runtimes = ['L4', 'T4', 'A100'].map(accelerator => ({
    id: `skypilot-gcp-${accelerator}`, accelerator, label: accelerator,
    execution: 'skypilot', provider: 'gcp', region: 'us-central1',
    enabled: true, device: 'cuda', training: true, simulation: false,
    training_model_ids: ['smolvla', 'pi05'],
  }));
  const localRuntime = {
    id: 'local-3070', label: 'Robotics lab RTX 3070', provider: 'local', execution: 'native',
    enabled: true, device: 'cuda', training: true, simulation: false, training_model_ids: ['smolvla', 'pi05'], gpu_memory_mib: 8192,
  };
  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() !== 'GET') {
      writes.push(`${request.method()} ${path}`);
      await route.fulfill({ status: 405, json: { detail: 'GPU picker tests never submit compute.' } });
      return;
    }
    const replies: Record<string, unknown> = {
      '/api/v1/health': { status: 'ok', version: 'gpu-picker-fixture' },
      '/api/v1/capabilities': [], '/api/v1/datasets': [],
      '/api/v1/projects': [{ id: 'picker', name: 'GPU selection', created_at: timestamp }],
      '/api/v1/projects/picker/jobs': [],
      '/api/v1/projects/picker/artifacts': [],
      '/api/v1/policy-options': {
        runtimes: local ? [...runtimes, localRuntime] : runtimes,
        sources: [], training_models: [{ id: 'pi05', label: 'π₀.₅', model_id: 'lerobot/pi05_base', model_revision: 'a'.repeat(40), description: 'Generalist policy', backend: 'lerobot', methods: ['full'], minimum_gpu_memory_gb: 40, runtime_ids: runtimes.map(item => item.id) }],
        training_methods: [{ id: 'lora', label: 'LoRA', description: 'Train adapters.' }, { id: 'full', label: 'Full training', description: 'Train policy weights.' }],
        default_training_method: 'lora',
        compute: { local: { enabled: local, label: 'Robotics lab' }, gcp: { enabled: true, default_gpu: 'A100', disk_size_gb: 200, idle_minutes: 10 } },
        quantization_defaults: { cuda: { language: 'Q8_0', vision: null }, cpu: { language: 'Q8_0', vision: null }, note: '' },
      },
    };
    if (path in replies) await route.fulfill({ json: replies[path] });
    else {
      unexpected.push(path);
      await route.fulfill({ status: 405, json: { detail: 'Unexpected picker fixture request.' } });
    }
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Fine-tune', exact: true }).click();
  await page.getByRole('button', { name: 'Start a new fine-tuning', exact: true }).click();
  await page.getByRole('navigation', { name: 'Training setup' }).getByRole('button', { name: 'Model', exact: true }).click();
  await chooseOnlyModel(page, 'SmolVLA');
  await page.getByRole('navigation', { name: 'Training setup' }).getByRole('button', { name: 'Compute', exact: true }).click();
  const picker = page.getByRole('combobox', { name: 'GPU', exact: true });
  await expect(picker).toBeVisible();
  await expect(picker).toHaveAttribute('value', 'A100');
  return { picker, writes, unexpected };
}

test('GPU picker supports keyboard navigation, explicit selection and Escape without submitting', async ({ page }) => {
  const { picker, writes, unexpected } = await openGpuPicker(page);
  await picker.focus();
  await page.keyboard.press('ArrowDown');
  const menu = page.getByRole('listbox', { name: 'GPU', exact: true });
  await expect(menu).toBeVisible();
  await expect(menu.getByRole('option')).toHaveCount(3);
  await expect(menu.getByRole('option', { name: 'A100', exact: true })).toHaveAttribute('aria-selected', 'true');
  await page.keyboard.press('Home');
  await expect(picker).toHaveAttribute('aria-activedescendant', (await menu.getByRole('option', { name: 'L4', exact: true }).getAttribute('id'))!);
  await expect(picker).toHaveAttribute('value', 'A100');
  await page.keyboard.press('ArrowDown');
  await expect(picker).toHaveAttribute('aria-activedescendant', (await menu.getByRole('option', { name: 'T4', exact: true }).getAttribute('id'))!);
  await page.keyboard.press('ArrowUp');
  await page.keyboard.press('End');
  await expect(picker).toHaveAttribute('aria-activedescendant', (await menu.getByRole('option', { name: 'A100', exact: true }).getAttribute('id'))!);
  await page.keyboard.press('Home');
  await page.keyboard.press('Enter');
  await expect(menu).toHaveCount(0);
  await expect(picker).toHaveAttribute('value', 'L4');
  await expect(picker).toBeFocused();
  await page.keyboard.press('Space');
  await page.keyboard.press('ArrowDown');
  await page.keyboard.press('Escape');
  await expect(menu).toHaveCount(0);
  await expect(picker).toHaveAttribute('value', 'L4');
  await expect(picker).toBeFocused();
  expect(writes).toEqual([]);
  expect(unexpected).toEqual([]);
});

test('changing to a large model removes undersized GPUs and clears the previous T4 selection', async ({ page }) => {
  const { picker, writes, unexpected } = await openGpuPicker(page, true);
  await picker.click();
  await page.getByRole('option', { name: 'T4', exact: true }).click();
  const setup = page.getByRole('navigation', { name: 'Training setup' });
  await setup.getByRole('button', { name: 'Model', exact: true }).click();
  await chooseOnlyModel(page, 'π₀.₅');
  await expect(page.getByRole('checkbox', { name: 'π₀.₅', exact: true }).locator('..').locator('.training-model-memory strong')).toHaveText('40 GB+');
  await setup.getByRole('button', { name: 'Compute', exact: true }).click();
  await expect(picker).toHaveAttribute('value', 'A100');
  await picker.click();
  const menu = page.getByRole('listbox', { name: 'GPU', exact: true });
  await expect(menu.getByRole('option')).toHaveCount(1);
  await expect(menu.getByRole('option', { name: 'A100', exact: true })).toBeVisible();
  await page.keyboard.press('Escape');
  await page.locator('summary').filter({ hasText: /^Advanced settings/ }).click();
  await expect(page.getByRole('spinbutton', { name: 'Batch size', exact: true })).toHaveValue('4');
  await page.getByRole('spinbutton', { name: 'Batch size', exact: true }).fill('3');
  await setup.getByRole('button', { name: 'Model', exact: true }).click();
  await chooseOnlyModel(page, 'SmolVLA');
  await setup.getByRole('button', { name: 'Compute', exact: true }).click();
  await page.locator('summary').filter({ hasText: /^Advanced settings/ }).click();
  await expect(page.getByRole('spinbutton', { name: 'Batch size', exact: true })).toHaveValue('64');
  await setup.getByRole('button', { name: 'Model', exact: true }).click();
  await chooseOnlyModel(page, 'π₀.₅');
  await setup.getByRole('button', { name: 'Compute', exact: true }).click();
  await page.locator('summary').filter({ hasText: /^Advanced settings/ }).click();
  await expect(page.getByRole('spinbutton', { name: 'Batch size', exact: true })).toHaveValue('3');
  expect(writes).toEqual([]);
  expect(unexpected).toEqual([]);
});

test('GPU picker supports pointer choice, typeahead, Tab and outside dismissal', async ({ page }) => {
  const { picker, writes, unexpected } = await openGpuPicker(page);
  await picker.click();
  await page.getByRole('option', { name: 'T4', exact: true }).click();
  await expect(picker).toHaveAttribute('value', 'T4');
  await expect(picker).toBeFocused();
  await page.keyboard.press('l');
  await page.keyboard.press('Enter');
  await expect(picker).toHaveAttribute('value', 'L4');
  await picker.click();
  await page.keyboard.press('End');
  await page.keyboard.press('Tab');
  await expect(page.getByRole('listbox')).toHaveCount(0);
  await expect(picker).toHaveAttribute('value', 'L4');
  await expect(picker).not.toBeFocused();
  await picker.click();
  // The menu can flip above and cover the section heading on mobile.
  await page.locator('body').click({ position: { x: 2, y: 2 } });
  await expect(page.getByRole('listbox')).toHaveCount(0);
  await expect(picker).toHaveAttribute('value', 'L4');
  expect(writes).toEqual([]);
  expect(unexpected).toEqual([]);
});

for (const theme of ['Light', 'Dark']) {
  test(`GPU picker menu fits 320px and shows memory badges in ${theme.toLowerCase()} mode`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width: 320, height: 740 });
    const { picker, writes, unexpected } = await openGpuPicker(page, true);
    await page.getByRole('button', { name: theme, exact: true }).click();
    await picker.click();
    const menu = page.getByRole('listbox', { name: 'GPU', exact: true });
    await expect(menu.getByRole('option')).toHaveCount(4);
    for (const [gpu, memory] of [['L4', '24 GB'], ['T4', '16 GB'], ['A100', '40 GB']]) {
      await expect(menu.getByRole('option', { name: gpu, exact: true }).locator('.gpu-picker-memory')).toHaveText(memory);
    }
    const bounds = await menu.boundingBox();
    expect(bounds).not.toBeNull();
    expect(bounds!.x).toBeGreaterThanOrEqual(0);
    expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(320);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(320);
    const screenshot = testInfo.outputPath(`gpu-picker-${theme.toLowerCase()}-320.png`);
    await page.screenshot({ path: screenshot, fullPage: true });
    await testInfo.attach(`gpu-picker-${theme.toLowerCase()}-320`, { path: screenshot, contentType: 'image/png' });
    await menu.getByRole('option', { name: 'Robotics lab RTX 3070', exact: true }).click();
    await expect(picker).toHaveAttribute('value', 'local-3070');
    await expect(picker).toContainText('Robotics lab RTX 3070');
    expect(writes).toEqual([]);
    expect(unexpected).toEqual([]);
  });
}

// These fixtures exercise admission and the exact submitted recipe only. They
// never invoke a provider or claim policy/model execution evidence.
async function trainingAdmission(page: Page, { format = 'lerobot_v3', dimensions = 6, cloudConnected = true } = {}) {
  const timestamp = '2026-09-26T12:00:00Z';
  const project = { id: 'admission', name: 'Admission fixture', created_at: timestamp };
  const state = { projectStatus: 200, projects: [project], projectReads: 0, submitted: [] as Record<string, any>[] };
  const profiles = [
    { id: 'act', label: 'ACT', model_id: 'code://lerobot/act', minimum_gpu_memory_gb: 16 },
    { id: 'gr00t_n17', label: 'GR00T N1.7', model_id: 'nvidia/GR00T-N1.7-LIBERO', minimum_gpu_memory_gb: 40 },
    { id: 'evo1', label: 'EVO-1', model_id: 'zuoxingdong/evo1_libero', minimum_gpu_memory_gb: 24 },
  ];
  const runtime = { id: 'skypilot-gcp-A100', label: 'A100', accelerator: 'A100', execution: 'skypilot', provider: 'gcp', enabled: true, device: 'cuda', training: true, simulation: false, gpu_memory_mib: 40960, training_model_ids: ['smolvla', ...profiles.map(model => model.id)] };
  const dataset = { id: 'admission-dataset', project_id: project.id, kind: 'dataset.inspect', status: 'succeeded', created_at: timestamp, updated_at: timestamp,
    request: { source: 'huggingface', repo_id: 'fixture/admission', revision: 'a'.repeat(40) },
    result: { source: 'huggingface', repo_id: 'fixture/admission', revision: 'a'.repeat(40), format, inspection_scope: 'metadata_only', total_episodes: 10, total_frames: 1000, fps: 30, metadata_sha256: 'b'.repeat(64), inspected_at: timestamp, warnings: [],
      features: { 'observation.images.front': { dtype: 'video', shape: [480, 640, 3] }, 'observation.state': { dtype: 'float32', shape: [dimensions] }, action: { dtype: 'float32', shape: [dimensions] } } } };
  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() === 'POST' && path === '/api/v1/projects/admission/policy-jobs') {
      state.submitted.push(request.postDataJSON());
      await route.fulfill({ status: 422, json: { detail: 'Recorded fixture request; no worker is launched.' }, headers: { 'Idempotency-Key': request.headers()['idempotency-key'], 'Cache-Control': 'no-store' } });
      return;
    }
    if (request.method() !== 'GET') throw new Error(`Unexpected write ${path}`);
    if (path.startsWith('/api/v1/projects/admission/submissions/')) {
      await route.fulfill({ status: 404, json: { detail: 'No saved fixture request.' }, headers: { 'Idempotency-Key': decodeURIComponent(path.split('/').at(-1)!), 'Cache-Control': 'no-store' } });
      return;
    }
    if (path === '/api/v1/projects') {
      state.projectReads += 1;
      await route.fulfill({ status: state.projectStatus, json: state.projectStatus === 200 ? state.projects : { detail: 'Project list temporarily unavailable' } });
      return;
    }
    const replies: Record<string, unknown> = {
      '/api/v1/health': { status: 'ok', version: 'admission-fixture' }, '/api/v1/capabilities': [], '/api/v1/datasets': [],
      '/api/v1/projects/admission/jobs': [dataset], '/api/v1/projects/admission/artifacts': [],
      '/api/v1/jobs/admission-dataset/episodes': { episodes: [], total: 10, offset: 0, limit: 6 },
      '/api/v1/policy-options': { runtimes: cloudConnected ? [runtime] : [], sources: [],
        training_models: [...profiles.map((model, index) => ({ ...model, model_revision: 'c'.repeat(40), description: 'Fixture profile', backend: 'lerobot', methods: ['full'], runtime_ids: cloudConnected ? [runtime.id] : [],
          // Include a stale ready record with no matching runtime: the client must
          // retain its own compute gate while accurately labeling the API state.
          ...(!cloudConnected ? { status: ['connect_account', 'setup_required', 'ready'][index] } : {}),
        })), ...(!cloudConnected ? [{ id: 'smolvla', label: 'SmolVLA', model_id: 'lerobot/smolvla_base', model_revision: 'a'.repeat(40), description: 'Fixture preflight-selectable adapter', backend: 'smolvla', methods: ['lora'], runtime_ids: [], status: 'connect_account' }] : [])],
        training_methods: [{ id: 'lora', label: 'LoRA', description: 'Adapter training' }, { id: 'full', label: 'Full training', description: 'Native policy training' }], default_training_method: 'lora',
        compute: { local: { enabled: false, label: 'Local' }, gcp: { enabled: true, default_gpu: 'A100', disk_size_gb: 200, idle_minutes: 10 } },
        quantization_defaults: { cuda: { language: 'Q8_0', vision: null }, cpu: { language: 'Q8_0', vision: null }, note: '' } },
    };
    if (!(path in replies)) throw new Error(`Unexpected admission request ${path}`);
    await route.fulfill({ json: replies[path] });
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Fine-tune', exact: true }).click();
  await page.getByRole('button', { name: 'Start a new fine-tuning', exact: true }).click();
  await expect(page.getByRole('checkbox', { name: 'fixture/admission', exact: true })).toBeVisible();
  const setup = page.getByRole('navigation', { name: 'Training setup' });
  async function chooseModel(label: string) {
    await setup.getByRole('button', { name: 'Model', exact: true }).click();
    await chooseOnlyModel(page, label);
    await setup.getByRole('button', { name: 'Compute', exact: true }).click();
  }
  return { state, setup, chooseModel, start: page.getByRole('button', { name: 'Start fine-tuning', exact: true }) };
}

test('native training explains the v3 requirement before submission while SmolVLA keeps its v2 path', async ({ page }) => {
  const fixture = await trainingAdmission(page, { format: 'lerobot_v2' });
  await fixture.chooseModel('ACT');
  await expect(fixture.start).toBeDisabled();
  await expect(page.getByText(/ACT currently needs a LeRobot v3 dataset/)).toBeVisible();
  expect(fixture.state.submitted).toEqual([]);
  await fixture.chooseModel('SmolVLA');
  await expect(fixture.start).toBeEnabled();
  await fixture.start.click();
  await expect.poll(() => fixture.state.submitted.length).toBe(1);
  expect(fixture.state.submitted[0].training.model_id).toBe('lerobot/smolvla_base');
});

test('native model dimension bounds allow GR00T vectors above 36 and reject narrower architectures', async ({ page }) => {
  const fixture = await trainingAdmission(page, { dimensions: 100 });
  await expect(page.getByRole('checkbox', { name: 'fixture/admission', exact: true })).toBeEnabled();
  await fixture.chooseModel('GR00T N1.7');
  await expect(fixture.start).toBeEnabled();
  await fixture.start.click();
  await expect.poll(() => fixture.state.submitted.length).toBe(1);
  expect(fixture.state.submitted[0]).toMatchObject({ dataset_job_id: 'admission-dataset', training_method: 'full', training: { model_id: 'nvidia/GR00T-N1.7-LIBERO', camera_keys: ['observation.images.front'] } });
  await fixture.chooseModel('EVO-1');
  await expect(fixture.start).toBeDisabled();
  await expect(page.getByText('EVO-1 supports at most 24 observation.state dimensions.')).toBeVisible();
  expect(fixture.state.submitted).toHaveLength(1);
});

test('training cannot submit using cached project data after a failed project refetch', async ({ page }) => {
  await page.clock.install();
  const fixture = await trainingAdmission(page);
  await fixture.chooseModel('ACT');
  await expect(fixture.start).toBeEnabled();
  const reads = fixture.state.projectReads;
  fixture.state.projectStatus = 503;
  await page.clock.fastForward(6_000);
  await page.evaluate(() => window.dispatchEvent(new Event('visibilitychange')));
  await expect.poll(() => fixture.state.projectReads).toBeGreaterThan(reads);
  await expect(fixture.start).toBeDisabled();
  expect(fixture.state.submitted).toEqual([]);
  expect(await page.evaluate(() => localStorage.getItem('firebird.workflow.'))).toBeNull();
  fixture.state.projectStatus = 200;
  await page.clock.fastForward(6_000);
  await page.evaluate(() => window.dispatchEvent(new Event('visibilitychange')));
  await expect(fixture.start).toBeEnabled();
  await fixture.start.click();
  await expect.poll(() => fixture.state.submitted.length).toBe(1);
  expect(fixture.state.submitted[0].training).toMatchObject({ model_id: 'code://lerobot/act', batch_size: 4 });
});


test('GR00T rejects vectors exceeding its architecture bound before any submission', async ({ page }) => {
  const fixture = await trainingAdmission(page, { dimensions: 133 });
  await fixture.chooseModel('GR00T N1.7');
  await expect(fixture.start).toBeDisabled();
  await expect(page.getByText('GR00T N1.7 supports at most 132 observation.state dimensions.')).toBeVisible();
  expect(fixture.state.submitted).toEqual([]);
});


test('disconnected cloud labels implemented trainers separately from planned adapters without enabling launch', async ({ page }) => {
  const fixture = await trainingAdmission(page, { cloudConnected: false });
  await fixture.setup.getByRole('button', { name: 'Model', exact: true }).click();
  for (const [model, label] of [
    ['ACT', 'Connect Google Cloud'],
    ['GR00T N1.7', 'Setup required'],
    ['EVO-1', 'Compute unavailable'],
    ['OpenVLA', 'Coming soon'],
  ]) {
    const radio = page.getByRole('checkbox', { name: model, exact: true });
    await expect(radio).toBeDisabled();
    await expect(radio.locator('..').locator('.training-model-status')).toHaveText(label);
  }
  // SmolVLA can still be selected before cloud preparation, but this cannot launch.
  const smol = page.getByRole('checkbox', { name: 'SmolVLA', exact: true });
  await expect(smol).toBeEnabled();
  await expect(smol.locator('..').locator('.training-model-status')).toHaveText('Connect Google Cloud');
  await chooseOnlyModel(page, 'SmolVLA');
  await fixture.setup.getByRole('button', { name: 'Compute', exact: true }).click();
  await expect(fixture.start).toBeDisabled();
  expect(fixture.state.submitted).toEqual([]);
});
