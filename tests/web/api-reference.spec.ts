import { expect, test, type Page } from '@playwright/test';

const methods = new Set(['get', 'post', 'put', 'patch', 'delete', 'options', 'head', 'trace']);

function endpoint(page: Page, method: string, path: string) {
  return page.locator(`details[id=${JSON.stringify(`endpoint-${method.toLowerCase()}-${encodeURIComponent(path)}`)}]`);
}

async function openReference(page: Page) {
  await page.goto('/docs/');
  await expect(page.getByRole('heading', { name: 'API reference', exact: true })).toBeVisible();
  await expect(endpoint(page, 'GET', '/api/v1/health')).toBeVisible();
}

async function expectNoPageOverflow(page: Page) {
  const sizes = await page.evaluate(() => ({
    viewport: document.documentElement.clientWidth,
    document: document.documentElement.scrollWidth,
    body: document.body.scrollWidth,
  }));
  expect(sizes.document).toBeLessThanOrEqual(sizes.viewport + 1);
  expect(sizes.body).toBeLessThanOrEqual(sizes.viewport + 1);
}

async function shellAppearance(page: Page) {
  return page.evaluate(() => {
    const properties: Record<string, string[]> = {
      body: ['font-family', 'font-size', 'line-height', 'color', 'background-color'],
      '.brand': ['font-family', 'font-size', 'font-weight', 'letter-spacing', 'color', 'gap'],
      '.brand-mark': ['background-color', 'color', 'border-radius', 'width', 'height'],
      '.sidebar': ['background-color', 'color', 'padding'],
      '.main-shell': ['background-color', 'border-color', 'border-width', 'border-radius'],
      '.topbar': ['height', 'padding', 'border-bottom', 'align-items', 'justify-content'],
      '.theme-toggle': ['padding', 'gap', 'background-color', 'border', 'border-radius'],
      '.theme-toggle button.selected': ['font-size', 'color', 'background-color', 'border', 'border-radius'],
      h1: ['font-family', 'font-size', 'font-weight', 'letter-spacing', 'line-height', 'color'],
    };
    const styles = Object.fromEntries(Object.entries(properties).map(([selector, names]) => {
      const element = document.querySelector(selector);
      if (!element) throw new Error(`Shared shell element disappeared: ${selector}`);
      const computed = getComputedStyle(element);
      return [selector, Object.fromEntries(names.map(name => [name, computed.getPropertyValue(name)]))];
    }));
    const root = getComputedStyle(document.documentElement);
    const tokens = Object.fromEntries(Array.from(root).filter(name => name.startsWith('--')).sort()
      .map(name => [name, root.getPropertyValue(name).trim()]));
    return { styles, tokens };
  });
}

for (const theme of ['Light', 'Dark']) {
  test(`workspace and API reference share rendered styling in ${theme.toLowerCase()} mode`, async ({ page }, testInfo) => {
    await page.goto('/');
    await page.getByRole('button', { name: theme, exact: true }).click();
    await expect(page.getByRole('button', { name: theme, exact: true })).toHaveAttribute('aria-pressed', 'true');
    const home = await shellAppearance(page);
    expect(Object.keys(home.tokens).length).toBeGreaterThan(10);

    await openReference(page);
    await expect(page.getByRole('button', { name: theme, exact: true })).toHaveAttribute('aria-pressed', 'true');
    // Compare against the current homepage, so a future shell or token change
    // cannot leave the API reference stuck on an independently copied design.
    expect(await shellAppearance(page)).toEqual(home);
    await expectNoPageOverflow(page);
    const screenshot = testInfo.outputPath(`api-reference-${theme.toLowerCase()}.png`);
    await page.screenshot({ path: screenshot, fullPage: true });
    await testInfo.attach(`api-reference-${theme.toLowerCase()}`, { path: screenshot, contentType: 'image/png' });
  });
}

test('theme selection survives home/reference navigation and direct reloads', async ({ page, context }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Dark', exact: true }).click();
  await page.locator('a[href="/docs/"]:visible').first().click();
  await expect(page).toHaveURL(/\/docs\/$/);
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await page.reload();
  await expect(page.getByRole('button', { name: 'Dark', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Light', exact: true }).click();
  await page.getByRole('link', { name: 'OPEN JENSEN workspace home' }).click();
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByRole('button', { name: 'Light', exact: true })).toHaveAttribute('aria-pressed', 'true');
  expect(await page.evaluate(() => localStorage.getItem('firebird.theme'))).toBe('light');
  expect(context.pages()).toHaveLength(1);
});

test('appearance controls still work when browser storage is unavailable', async ({ page }) => {
  await page.addInitScript(() => {
    Storage.prototype.getItem = () => { throw new DOMException('Storage unavailable', 'SecurityError'); };
    Storage.prototype.setItem = () => { throw new DOMException('Storage unavailable', 'SecurityError'); };
  });
  await openReference(page);
  await page.getByRole('button', { name: 'Dark', exact: true }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await expect(page.getByRole('button', { name: 'Dark', exact: true })).toHaveAttribute('aria-pressed', 'true');
});

test('documents every live route and model, with the raw schema still accessible', async ({ page, request }) => {
  const response = await request.get('/openapi.json');
  expect(response.ok()).toBeTruthy();
  const schema = await response.json();
  await openReference(page);
  const expectedEndpoints: string[] = [];
  for (const [path, item] of Object.entries(schema.paths)) {
    for (const method of Object.keys(item as object).filter(method => methods.has(method))) {
      expectedEndpoints.push(`endpoint-${method}-${encodeURIComponent(path)}`);
    }
  }
  // Compare complete inventories in two browser reads, rather than hundreds of round trips.
  await expect.poll(() => page.locator('details[id^="endpoint-"]').evaluateAll(nodes => nodes
    .filter(node => {
      const bounds = node.getBoundingClientRect();
      return bounds.width > 0 && bounds.height > 0 && getComputedStyle(node).visibility === 'visible';
    })
    .map(node => node.id).sort())).toEqual(expectedEndpoints.sort());
  const expectedSchemas = Object.keys(schema.components.schemas).map(name => `schema-${encodeURIComponent(name)}`).sort();
  await expect.poll(() => page.locator('details[id^="schema-"]').evaluateAll(nodes => nodes
    .map(node => node.id).sort())).toEqual(expectedSchemas);
  await expect(page.locator('a[href="/openapi.json"]').first()).toBeVisible();
  await expect(page.locator('.swagger-ui, .redoc-wrap')).toHaveCount(0);
});

test('filters endpoints by multiple search terms and HTTP method, then restores the list', async ({ page }) => {
  await openReference(page);
  const all = await page.locator('details[id^="endpoint-"]').count();
  const search = page.getByRole('searchbox', { name: 'Search endpoints' });
  const method = page.getByRole('combobox', { name: 'HTTP method' });
  await search.fill('  PROJECTS   intakes  ');
  await expect(page.locator('details[id^="endpoint-"]')).toHaveCount(1);
  await expect(endpoint(page, 'POST', '/api/v1/projects/{project_id}/intakes')).toBeVisible();
  await method.selectOption('GET');
  await expect(page.locator('details[id^="endpoint-"]')).toHaveCount(0);
  await expect(page.getByText(/No endpoints found/i)).toBeVisible();
  await search.fill('');
  const gets = page.locator('details[id^="endpoint-get-"]');
  expect(await gets.count()).toBeGreaterThan(0);
  await expect(page.locator('details[id^="endpoint-post-"]')).toHaveCount(0);
  await method.selectOption('ALL');
  await expect(page.locator('details[id^="endpoint-"]')).toHaveCount(all);
});

test('discovers newly introduced live endpoints and models without a web rebuild', async ({ page, request }) => {
  const schema = await (await request.get('/openapi.json')).json();
  schema.paths['/api/v1/runtime-discovery'] = {
    patch: {
      summary: 'New runtime operation', tags: ['Runtime discovery'],
      responses: { '200': { description: 'Fresh result', content: { 'application/json': { schema: { $ref: '#/components/schemas/RuntimeDiscovery' } } } } },
    },
  };
  schema.components.schemas.RuntimeDiscovery = {
    type: 'object', properties: { new_field: { type: 'string' } }, required: ['new_field'],
  };
  await page.route('**/openapi.json', route => route.fulfill({ json: schema }));
  await openReference(page);
  await page.getByRole('searchbox', { name: 'Search endpoints' }).fill('runtime discovery');
  await page.getByRole('combobox', { name: 'HTTP method' }).selectOption('PATCH');
  const operation = endpoint(page, 'PATCH', '/api/v1/runtime-discovery');
  await expect(operation).toBeVisible();
  await operation.locator('summary').click();
  await expect(operation).toContainText('Fresh result');
  await expect(operation).toContainText('RuntimeDiscovery');
  await expect(page.locator('[id="schema-RuntimeDiscovery"]')).toContainText('new_field');
});

test('shows path parameters, JSON request bodies, status codes, and referenced response models', async ({ page }) => {
  await openReference(page);
  const intake = endpoint(page, 'POST', '/api/v1/projects/{project_id}/intakes');
  await intake.locator('summary').click();
  await expect(intake).toContainText('project_id');
  await expect(intake).toContainText('path');
  await expect(intake).toContainText(/required/i);
  await expect(intake).toContainText('IntakeRequest');
  await expect(intake).toContainText('application/json');
  await expect(intake).toContainText('202');
  await expect(intake).toContainText('422');
  await expect(intake).toContainText('Job');
  await expect(intake.locator('pre').first()).toContainText(`${new URL(page.url()).origin}/api/v1/projects/{project_id}/intakes`);
  await expect(intake.locator('pre').first()).toContainText('--data @request.json');
  const requestModel = page.locator('[id="schema-IntakeRequest"]');
  await requestModel.locator('summary').first().click();
  await expect(requestModel).toContainText('huggingface');
  await expect(requestModel.getByRole('row').filter({ has: page.getByRole('rowheader', { name: /^repo_id/ }) }))
    .toContainText('maxLength: 200');
  await expect(requestModel.getByRole('row').filter({ has: page.getByRole('rowheader', { name: /^path/ }) }))
    .toContainText('maxLength: 4096');
  await expect(page.locator('[id="schema-Job"]')).toContainText('running');
  await expectNoPageOverflow(page);
});

test('endpoint disclosures are keyboard-operable and remain within the viewport', async ({ page }) => {
  await openReference(page);
  const health = endpoint(page, 'GET', '/api/v1/health');
  const summary = health.locator('summary');
  await expect(health).not.toHaveAttribute('open', '');
  await summary.focus();
  await page.keyboard.press('Enter');
  await expect(health).toHaveAttribute('open', '');
  await expect(health).toContainText('200');
  await page.keyboard.press('Space');
  await expect(health).not.toHaveAttribute('open', '');
  await expectNoPageOverflow(page);
});

test('endpoint and model permalinks open their targets after a direct load', async ({ page, request }) => {
  const schema = await (await request.get('/openapi.json')).json();
  const projectOperations = Object.entries(schema.paths)
    .filter(([path]) => path.includes('projects'))
    .reduce((count, [, operations]) => count + Object.keys(operations as object).filter(method => methods.has(method)).length, 0);
  const path = '/api/v1/projects/{project_id}/intakes';
  await page.goto(`/docs/#endpoint-post-${encodeURIComponent(path)}`);
  const intake = endpoint(page, 'POST', path);
  await expect(intake).toHaveAttribute('open', '');
  await intake.getByRole('link', { name: 'IntakeRequest', exact: true }).click();
  await expect(page).toHaveURL(/#schema-IntakeRequest$/);
  const model = page.locator('[id="schema-IntakeRequest"]');
  await expect(model).toHaveAttribute('open', '');
  await expect(model.getByRole('columnheader', { name: 'Type & constraints' })).toBeVisible();
  await model.locator('summary').first().click();
  await expect(model).not.toHaveAttribute('open', '');
  await intake.getByRole('link', { name: 'IntakeRequest', exact: true }).click();
  await expect(model).toHaveAttribute('open', '');
  await page.reload();
  await expect(model).toHaveAttribute('open', '');
  const search = page.getByRole('searchbox', { name: 'Search endpoints' });
  await search.scrollIntoViewIfNeeded();
  const beforeSearch = await page.evaluate(() => window.scrollY);
  await search.fill('projects');
  await expect(page.locator('details[id^="endpoint-"]')).toHaveCount(projectOperations);
  // Editing filters must not re-apply the old model permalink and scroll away
  // from the input on every keystroke.
  expect(await page.evaluate(() => window.scrollY)).toBeLessThanOrEqual(beforeSearch + 10);
});

test('copy examples uses the displayed active-origin command and reports clipboard failure', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: {
      writeText: async (value: string) => {
        if (value.includes('--request POST')) throw new DOMException('Denied', 'NotAllowedError');
        document.documentElement.setAttribute('data-copied-example', value);
      },
    } });
  });
  await openReference(page);
  const health = endpoint(page, 'GET', '/api/v1/health');
  await health.locator('summary').click();
  const command = await health.locator('pre').textContent();
  await health.getByRole('button', { name: 'Copy GET /api/v1/health example', exact: true }).click();
  await expect(health.getByRole('status')).toHaveText('Example copied to clipboard.');
  await expect(page.locator('html')).toHaveAttribute('data-copied-example', command!);
  const project = endpoint(page, 'POST', '/api/v1/projects');
  await project.locator('summary').click();
  await project.getByRole('button', { name: 'Copy POST /api/v1/projects example', exact: true }).click();
  await expect(project.getByRole('status')).toHaveText('Copy unavailable. Select the example text to copy it.');
  await expect(project.locator('pre')).toBeVisible();
});

test('long endpoint, cURL, and full model contracts stay usable at 320px', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 740 });
  await openReference(page);
  await endpoint(page, 'POST', '/api/v1/projects/{project_id}/intakes').locator('summary').click();
  const model = page.locator('[id="schema-Job"]');
  await model.locator('summary').first().click();
  await model.getByText('Full JSON schema', { exact: true }).click();
  await expectNoPageOverflow(page);
  await expect(model.locator('pre')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Dark', exact: true })).toBeVisible();
  const header = await page.locator('.topbar').evaluate(element => {
    const title = element.querySelector('.breadcrumb strong')!.getBoundingClientRect();
    const link = element.querySelector('.mobile-api-link')!.getBoundingClientRect();
    const toggle = element.querySelector('.theme-switcher')!.getBoundingClientRect();
    return { titleRight: title.right, linkLeft: link.left, linkRight: link.right, toggleLeft: toggle.left };
  });
  expect(header.titleRight).toBeLessThanOrEqual(header.linkLeft);
  expect(header.linkRight).toBeLessThanOrEqual(header.toggleLeft);
});

test('keeps shared navigation usable while the schema is loading', async ({ page, request }) => {
  const schema = await (await request.get('/openapi.json')).json();
  let release!: () => void;
  const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/openapi.json', async route => {
    await gate;
    await route.fulfill({ json: schema });
  });
  await page.goto('/docs/');
  await expect(page.getByText(/Loading API reference/i)).toBeVisible();
  await page.getByRole('button', { name: 'Dark', exact: true }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  release();
  await expect(endpoint(page, 'GET', '/api/v1/health')).toBeVisible();
  await expect(page.getByText(/Loading API reference/i)).toHaveCount(0);
});

test('recovers from an HTTP schema error through Retry', async ({ page, request }) => {
  const schema = await (await request.get('/openapi.json')).json();
  let attempts = 0;
  await page.route('**/openapi.json', route => ++attempts === 1
    ? route.fulfill({ status: 503, json: { detail: 'Temporary schema outage' } })
    : route.fulfill({ json: schema }));
  await page.goto('/docs/');
  const retry = page.getByRole('button', { name: 'Retry loading schema' });
  await expect(retry).toBeVisible();
  await retry.click();
  await expect(endpoint(page, 'GET', '/api/v1/health')).toBeVisible();
  await expect(retry).toHaveCount(0);
  expect(attempts).toBe(2);
});

test('reports malformed schemas without crashing the shared page', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/openapi.json', route => route.fulfill({ json: { openapi: '2.0' } }));
  await page.goto('/docs/');
  await expect(page.getByText(/invalid OpenAPI document/i)).toBeVisible();
  await expect(page.getByRole('button', { name: 'Retry loading schema' })).toBeVisible();
  await expect(page.getByRole('link', { name: 'OPEN JENSEN workspace home' })).toBeVisible();
  expect(errors).toEqual([]);
});

test('direct docs URLs and reloads render without hydration or runtime errors', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/docs');
  await expect(page).toHaveURL(/\/docs\/$/);
  await expect(endpoint(page, 'GET', '/api/v1/health')).toBeVisible();
  await page.reload();
  await expect(page.getByRole('heading', { name: 'API reference', exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});

test('shared workspace shell preserves training, defaults, and separate diagnostics', async ({ page }) => {
  // This view-only check requires an empty workspace even when real journey tests run.
  await page.route('**/api/v1/projects', route => route.fulfill({ json: [] }));
  await page.goto('/');
  await page.getByRole('button', { name: 'Fine-tune', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Fine-tuning jobs', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Start a new fine-tuning', exact: true })).toBeDisabled();
  await expect(page.getByRole('navigation', { name: 'Training setup' })).toHaveCount(0);
  await page.getByRole('button', { name: 'Settings & diagnostics', exact: true }).click();
  await page.getByRole('button', { name: 'Workflow settings', exact: true }).click();
  await expect(page.getByLabel('Quantization recipe')).toHaveValue('recommended');
  await expect(page.getByLabel('Also quantize vision to Q8 (experimental)')).not.toBeChecked();
  await page.getByRole('button', { name: 'Diagnostics', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Run diagnostics' })).toBeVisible();
  await expect(page.getByLabel('Quantization recipe')).toHaveCount(0);
  await expect(page.getByText('No policy runs in this project yet.', { exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Recorded benchmark comparison' })).toBeVisible();
  const reference = page.getByRole('table', { name: 'NVIDIA L4 · 8 vCPUs / 32 GiB · recorded reference results' });
  await expect(reference.getByRole('row', { name: /^native-bf16 / })).toContainText('342.59');
  await expect(reference.getByRole('row', { name: /^cpp-Q4_0 / })).toContainText('4/20');
  await page.getByLabel('Reference hardware').selectOption('rtx3070');
  const rtx = page.getByRole('table', { name: 'NVIDIA RTX 3070 · recorded reference results' });
  await expect(rtx.getByRole('row', { name: /^cpp-Q4_0 / })).toContainText('110.92');
  await expect(rtx.getByRole('row', { name: /^cpp-Q4_0 / })).toContainText('8/20');
  await expect(rtx.getByRole('columnheader', { name: /VRAM/ })).toHaveCount(1);
  await expect(rtx.getByRole('row', { name: /^cpp-Q4_0 / }).getByRole('cell').nth(2)).toHaveText('—');
  await expect(page.getByRole('link', { name: 'View source report' })).toHaveAttribute('href', /\/tree\/[0-9a-f]{40}\/workers\/benchmark_gpu\/evidence\/2026-09-26\/REPORT.md$/);
  await expectNoPageOverflow(page);
});

test('dataset inspection auto-loads camera previews only after opening and stays separate from policy jobs', async ({ page }) => {
  const timestamp = '2026-09-26T12:00:00Z';
  const datasetJob = {
    id: 'dataset-review', project_id: 'mixed-review', kind: 'dataset.inspect', status: 'succeeded',
    request: { source: 'huggingface', repo_id: 'fixture/robot', revision: 'main' },
    created_at: timestamp, updated_at: timestamp,
    result: {
      source: 'huggingface', repo_id: 'fixture/robot', revision: 'a'.repeat(40), format: 'lerobot_v3',
      inspection_scope: 'metadata_only', total_episodes: 2, total_frames: 12, fps: 5,
      features: { action: { dtype: 'float32', shape: [2] }, 'observation.state': { dtype: 'float32', shape: [2] }, 'observation.images.front': { dtype: 'video', shape: [480, 640, 3] }, 'observation.images.wrist': { dtype: 'video', shape: [480, 640, 3] } },
      metadata_sha256: 'b'.repeat(64), inspected_at: timestamp, warnings: [],
    },
  };
  const policyJob = {
    id: 'policy-review', project_id: 'mixed-review', kind: 'policy.import', status: 'succeeded',
    request: { operation: 'policy.import', runtime_id: 'fixture', source_id: 'source' },
    created_at: '2026-09-26T13:00:00Z', updated_at: '2026-09-26T13:00:00Z',
    result: { artifacts: [], reports: [], decision: 'completed' },
  };
  await page.route('**/api/v1/projects', route => route.fulfill({ json: [{ id: 'mixed-review', name: 'Mixed jobs', created_at: timestamp }] }));
  await page.route('**/api/v1/projects/mixed-review/jobs', route => route.fulfill({ json: [policyJob, datasetJob] }));
  const previews: string[] = [];
  const media: string[] = [];
  await page.route('**/api/v1/jobs/dataset-review/episodes**', route => {
    const url = new URL(route.request().url());
    const index = url.pathname.endsWith('/episodes') ? null : Number(url.pathname.split('/').pop());
    previews.push(index === null ? 'index' : String(index));
    const episode = (episode_index: number) => ({ episode_index, frame_count: 6, duration_seconds: 1.2, tasks: ['Pick up the cube'] });
    return route.fulfill({ json: index === null ? {
      repo_id: 'fixture/robot', revision: 'a'.repeat(40), total_episodes: 2, offset: 0, limit: 6,
      episodes: [episode(0), episode(1)], warnings: [],
    } : {
      ...episode(index), repo_id: 'fixture/robot', revision: 'a'.repeat(40),
      cameras: ['front', 'wrist'].map(key => ({
        key: `observation.images.${key}`, url: `/fixture-camera-${key}-${index}.mp4`,
        start_seconds: 0, end_seconds: 1.2, fps: 5, width: 640, height: 480,
      })),
      samples: [{ frame_index: 0, timestamp: 0, state: [.1, .2], action: [.3, .4] }],
      state_names: ['joint_1', 'joint_2'], action_names: ['motor_1', 'motor_2'], warnings: [],
    } });
  });
  await page.route('**/fixture-camera-*.mp4', route => {
    media.push(new URL(route.request().url()).pathname);
    return route.fulfill({ status: 200, contentType: 'video/mp4', body: '' });
  });
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await expect(page.getByRole('button', { name: /^Inspection/ })).toBeEnabled();
  await expect(page.getByRole('button', { name: 'Sources', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('heading', { name: 'Import a dataset', exact: true })).toBeVisible();
  expect(previews).toEqual([]);
  expect(media).toEqual([]);

  await page.getByRole('button', { name: /^Inspection/ }).click();
  await expect(page.getByRole('heading', { name: 'fixture/robot', exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Cameras', exact: true })).toBeVisible();
  await expect(page.locator('.dx-camera video')).toHaveCount(2);
  await expect(page.getByLabel(/^observation.images.front, episode 0/)).toHaveAttribute('src', '/fixture-camera-front-0.mp4');
  await expect.poll(() => previews).toEqual(['index', '0']);
  await expect.poll(() => media.length).toBe(2);
  await expect(page.locator('.dx-samples')).not.toHaveAttribute('open', '');
  await expect(page.locator('.dx-schema')).not.toHaveAttribute('open', '');
  await page.locator('.dx-samples > summary').click();
  await expect(page.getByRole('columnheader', { name: 'joint_1', exact: true })).toBeVisible();
  await expect(page.getByRole('cell', { name: '0.1', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Action', exact: true }).click();
  await expect(page.getByRole('columnheader', { name: 'motor_1', exact: true })).toBeVisible();
  await expect(page.getByRole('cell', { name: '0.3', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Episode 1', exact: true }).click();
  await expect(page.getByLabel(/^observation.images.wrist, episode 1/)).toHaveAttribute('src', '/fixture-camera-wrist-1.mp4');
  await expect.poll(() => previews).toEqual(['index', '0', '1']);
  await expect(page.getByText('policy-review', { exact: true })).toHaveCount(0);
  await expectNoPageOverflow(page);
  await page.getByRole('button', { name: 'Sources', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Import a dataset', exact: true })).toBeVisible();
  expect(previews).toEqual(['index', '0', '1']);
  expect(errors).toEqual([]);
});

const trainingProject = 'training-review';
const trainingTimestamp = '2026-09-26T12:00:00Z';
const modelRevision = 'c'.repeat(40);
const localTrainingRuntimes = [
  { id: 'cpu', label: 'CPU worker', device: 'cpu', training: false, simulation: false },
  { id: 'inference', label: 'Inference-only GPU', device: 'cuda', training: false, simulation: true },
  { id: 'gpu-3070', label: 'RTX 3070 workstation', device: 'cuda', training: true, simulation: true },
  { id: 'gpu-4090', label: 'RTX 4090 workstation', device: 'cuda', training: true, simulation: true },
];
const trainingRuntimes: Record<string, any>[] = [
  ...localTrainingRuntimes,
  ...['L4', 'T4', 'A100'].map(accelerator => ({
    id: `skypilot-gcp-${accelerator}`, label: `NVIDIA ${accelerator}`, execution: 'skypilot',
    accelerator, provider: 'gcp', region: 'us-central1', enabled: true,
    device: 'cuda', training: true, simulation: false, training_model_ids: ['smolvla'],
    gpu_name: `NVIDIA ${accelerator}`, gpu_memory_mib: { L4: 24576, T4: 16384, A100: 40960 }[accelerator],
  })),
];

async function chooseGpu(page: Page, value: string, label = value) {
  const picker = page.getByRole('combobox', { name: 'GPU', exact: true });
  await picker.click();
  await page.getByRole('listbox', { name: 'GPU', exact: true }).getByRole('option', { name: label, exact: true }).click();
  await expect(picker).toHaveAttribute('value', value);
}

async function expectGpuChoices(page: Page, labels: string[]) {
  await page.getByRole('combobox', { name: 'GPU', exact: true }).click();
  const list = page.getByRole('listbox', { name: 'GPU', exact: true });
  await expect(list.getByRole('option')).toHaveCount(labels.length);
  for (const label of labels) await expect(list.getByRole('option', { name: label, exact: true })).toBeVisible();
  await page.keyboard.press('Escape');
}

function trainingDataset(id: string, repo: string, camera = 'front', created = trainingTimestamp) {
  return {
    id, project_id: trainingProject, kind: 'dataset.inspect', status: 'succeeded',
    request: { source: 'huggingface', repo_id: repo, revision: 'main' },
    created_at: created, updated_at: created,
    result: {
      source: 'huggingface', repo_id: repo, revision: 'a'.repeat(40), format: 'lerobot_v3',
      inspection_scope: 'metadata_only', total_episodes: 12, total_frames: 720, fps: 30,
      features: {
        action: { dtype: 'float32', shape: [6] },
        'observation.state': { dtype: 'float32', shape: [6] },
        [`observation.images.${camera}`]: { dtype: 'video', shape: [480, 640, 3] },
        'observation.images.wrist': { dtype: 'video', shape: [480, 640, 3] },
      },
      metadata_sha256: 'b'.repeat(64), inspected_at: created, warnings: [],
    },
  };
}

async function mockTrainingWorkspace(page: Page, {
  runtimes = trainingRuntimes, previousRun = false, emptyPreview = false, configuredPi0 = false,
  configuredPsi = false, datasetFormat = 'lerobot_v3',
  local = { enabled: false, label: 'Local machine' },
  events = [] as { sequence: number; stage: string; message: string; timestamp: string }[],
  queuedRun = false, needsPreparation = false,
} = {}) {
  const submitted: Record<string, any>[] = [];
  const submissions = new Map<string, { body: unknown; job: Record<string, any> }>();
  const previews: string[] = [];
  const availableRuntimes = runtimes.map(runtime => ({
    ...runtime,
    ...(runtime.execution === 'skypilot' ? { launchable: true, needs_preparation: needsPreparation } : {}),
    ...(configuredPi0 && runtime.id === 'skypilot-gcp-A100' ? { training_model_ids: ['smolvla', 'pi0'] } : {}),
    ...(configuredPsi && runtime.id === 'skypilot-gcp-A100' ? { training_model_ids: ['smolvla', 'psi0'] } : {}),
  }));
  const gcp = { enabled: true, default_gpu: 'A100', disk_size_gb: 200, idle_minutes: 10 };
  const gcpStatus = { status: needsPreparation ? 'unchecked' : 'ready', configured: true, project_id: 'training-project-123', region: 'us-central1', skypilot_installed: !needsPreparation, checked_at: needsPreparation ? null : trainingTimestamp, message: needsPreparation ? 'Training setup will run automatically.' : 'SkyPilot setup verified.', setup_commands: [] };
  const unexpectedRequests: string[] = [];
  const artifacts: Record<string, any>[] = [];
  const jobs: Record<string, any>[] = [
    trainingDataset('dataset-new', 'fixture/pick-and-place', 'front', '2026-09-26T13:00:00Z'),
    // Repeated metadata inspections of the same pinned revision are one choice.
    trainingDataset('dataset-old', 'fixture/pick-and-place'),
    trainingDataset('dataset-stack', 'fixture/stack-blocks', 'side'),
    { ...trainingDataset('dataset-failed', 'fixture/failed'), status: 'failed', result: null },
    {
      ...trainingDataset('dataset-local', 'fixture/local'),
      request: { source: 'local', path: '/fixture/local' },
      result: { ...trainingDataset('dataset-local', 'fixture/local').result, source: 'local' },
    },
    {
      ...trainingDataset('dataset-unpinned', 'fixture/unpinned'),
      result: { ...trainingDataset('dataset-unpinned', 'fixture/unpinned').result, revision: 'main' },
    },
  ];
  if (datasetFormat !== 'lerobot_v3') {
    for (const job of jobs) if (job.result) job.result.format = datasetFormat;
  }
  if (previousRun) {
    // Resuming must retain the original inspection ID even when a newer
    // inspection of the identical repository/revision would normally win.
    jobs.unshift(trainingDataset('dataset-stack-newer', 'fixture/stack-blocks', 'side', '2026-09-26T14:15:00Z'));
    const interrupted = {
      id: 'interrupted-training', project_id: trainingProject, kind: 'policy.finetune', status: 'interrupted',
      request: {
        operation: 'policy.finetune', runtime_id: 'skypilot-gcp-A100', dataset_job_id: 'dataset-stack',
        training_method: 'qlora', timeout_seconds: 7200,
        training: { steps: 400, camera_key: 'observation.images.side', model_id: 'lerobot/smolvla_base', model_revision: modelRevision },
      },
      created_at: '2026-09-26T14:00:00Z', updated_at: '2026-09-26T14:10:00Z',
      result: null,
    };
    jobs.unshift(interrupted, {
      ...interrupted, id: 'completed-training', status: 'succeeded',
      created_at: '2026-09-26T14:20:00Z', updated_at: '2026-09-26T14:30:00Z',
    });
    artifacts.push({
      id: 'completed-checkpoint', job_id: 'completed-training', project_id: trainingProject,
      label: 'Completed training checkpoint', format: 'training_checkpoint',
      path: '/fixture/completed-checkpoint', file_bytes: 1024, manifest_sha256: 'd'.repeat(64),
      metadata: { training_method: 'qlora' },
    });
  }
  // Every application API call is intercepted, including all writes. These
  // tests cannot submit compute work even if a new request path is introduced.
  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    const submissionPath = `/api/v1/projects/${trainingProject}/submissions/`;
    if (request.method() === 'GET' && path.startsWith(submissionPath) && url.searchParams.get('operation') === 'policy.finetune') {
      const key = decodeURIComponent(path.slice(submissionPath.length));
      if (key && request.headers()['idempotency-key'] === key) {
        const saved = submissions.get(key);
        await route.fulfill({ status: saved ? 200 : 404,
          headers: { 'Idempotency-Key': key, 'Cache-Control': 'no-store' },
          json: saved?.job ?? { detail: 'No accepted submission found for this project, operation and key' } });
        return;
      }
    }
    if (request.method() === 'POST' && path === `/api/v1/projects/${trainingProject}/policy-jobs`) {
      const body = request.postDataJSON();
      const key = request.headers()['idempotency-key'];
      if (!key || body.operation !== 'policy.finetune') {
        unexpectedRequests.push(`${request.method()} ${path}`);
        await route.fulfill({ status: 405, json: { detail: 'Unkeyed or unsupported training request blocked by fixture.' } });
        return;
      }
      const headers = { 'Idempotency-Key': key, 'Cache-Control': 'no-store' };
      const saved = submissions.get(key);
      if (saved) {
        const same = JSON.stringify(saved.body) === JSON.stringify(body);
        await route.fulfill({ status: same ? 202 : 409, headers,
          json: same ? saved.job : { detail: 'Submission key already belongs to another request.' } });
        return;
      }
      submitted.push(body);
      const job: Record<string, any> = {
        id: `mock-training-${submitted.length}`, project_id: trainingProject, kind: 'policy.finetune',
        status: queuedRun ? 'queued' : 'succeeded', stage: queuedRun ? 'preparing' : null,
        request: body, created_at: '2026-09-26T15:00:00Z', updated_at: '2026-09-26T15:00:00Z',
        result: queuedRun ? null : { artifacts: [], reports: [], decision: 'completed' },
      };
      jobs.unshift(job);
      submissions.set(key, { body, job });
      await route.fulfill({ status: 202, headers, json: job });
      return;
    }
    if (request.method() === 'GET') {
      const episodePath = /^\/api\/v1\/jobs\/([^/]+)\/episodes(?:\/(\d+))?$/.exec(path);
      if (episodePath) {
        const dataset = jobs.find(job => job.id === episodePath[1]);
        if (dataset?.kind === 'dataset.inspect' && dataset.result) {
          previews.push(path);
          const index = episodePath[2] ? Number(episodePath[2]) : null;
          const episode = (episode_index: number) => ({ episode_index, frame_count: 60, duration_seconds: 2, tasks: ['Pick up the cube'] });
          const common = { repo_id: dataset.result.repo_id, revision: dataset.result.revision, warnings: [] };
          await route.fulfill({ json: index === null ? {
            ...common, episodes: [episode(0), episode(1)], total_episodes: 12, offset: 0, limit: 6,
          } : {
            ...common, ...episode(index), samples: [], state_names: [], action_names: [],
            cameras: emptyPreview ? [] : Object.keys(dataset.result.features)
              .filter(key => key.startsWith('observation.images.'))
              .map(key => ({ key, url: `/api/v1/mock-training-camera/${dataset.id}/${key}/${index}`,
                start_seconds: 0, end_seconds: 2, fps: 30, width: 640, height: 480 })),
          } });
          return;
        }
      }
      if (path.startsWith('/api/v1/mock-training-camera/')) {
        await route.fulfill({ status: 200, contentType: 'video/mp4', body: '' });
        return;
      }
      const responses: Record<string, unknown> = {
        '/api/v1/health': { status: 'ok', version: 'browser-fixture' },
        '/api/v1/capabilities': [],
        '/api/v1/projects': [{ id: trainingProject, name: 'Training review', created_at: trainingTimestamp }],
        [`/api/v1/projects/${trainingProject}/jobs`]: jobs,
        [`/api/v1/projects/${trainingProject}/artifacts`]: artifacts,
        '/api/v1/huggingface-connection': { configured: false, username: null, token_hint: null, checked_at: null, message: null },
        '/api/v1/cloud-connections': { providers: [{ provider: 'gcp', name: 'Google Cloud', status: 'connected', config: { project_id: 'training-project-123', region: 'us-central1' }, identity: { account: 'robotics@example.test' }, checked_at: trainingTimestamp, message: null, setup_commands: [] }] },
        '/api/v1/compute-settings': {
          local, gcp, runtimes: availableRuntimes, gcp_status: gcpStatus,
          gpu_options: ['L4', 'T4', 'A100'].map(id => ({ id, label: id, accelerator: id, gpu_count: 1, gpu_memory_mib: 16384, supported: true, available: true, unavailable_reason: null })),
        },
        '/api/v1/policy-options': {
          runtimes: availableRuntimes, sources: [], compute: { local, gcp },
          training_models: [{
            id: 'smolvla', label: 'SmolVLA', description: 'A compact vision-language-action model for robot learning.',
            model_id: 'lerobot/smolvla_base', model_revision: modelRevision, methods: ['lora', 'qlora'],
            // Older catalogs report no ready runtime IDs until preflight, even
            // while the cloud runtime already advertises its bundled adapter.
            ...(needsPreparation ? { runtime_ids: [], available: false, status: 'setup_required' } : {}),
          }, ...(configuredPi0 ? [{
            id: 'pi0', label: 'π₀', description: 'Flow-matching robot policy',
            model_id: 'lerobot/pi0_libero_finetuned_v044', model_revision: modelRevision,
            methods: ['lora'], runtime_ids: ['skypilot-gcp-A100'], available: true, status: 'ready',
          }] : []), ...(configuredPsi ? [{
            id: 'psi0', label: 'Psi-Zero', description: 'Humanoid action expert',
            model_id: 'USC-PSI-Lab/psi-model', model_revision: '4c6f9776fc5b18d87945254175e38bb74b9d7748',
            methods: ['full'], backend: 'psi0', minimum_gpu_memory_gb: 40, required_cameras: 1,
            runtime_ids: ['skypilot-gcp-A100'], available: true, status: 'ready',
          }] : [])],
          training_methods: [
            { id: 'lora', label: 'LoRA', description: 'Train adapters over a floating base.' },
            { id: 'qlora', label: 'QLoRA', description: 'Train adapters with a lower-memory base.' },
            { id: 'full', label: 'Full training', description: 'Train policy weights.' },
          ],
          default_training_method: 'lora',
          quantization_defaults: { cuda: { language: 'Q8_0', vision: null }, cpu: { language: 'Q8_0', vision: null }, note: '' },
        },
      };
      if (path in responses) {
        await route.fulfill({ json: responses[path] });
        return;
      }
      if (/^\/api\/v1\/jobs\/[^/]+\/events$/.test(path)) {
        await route.fulfill({ json: events });
        return;
      }
      const telemetryPath = /^\/api\/v1\/jobs\/([^/]+)\/training$/.exec(path);
      if (telemetryPath) {
        const run = jobs.find(job => job.id === telemetryPath[1])!;
        await route.fulfill({ json: { job_id: run.id, status: run.status, phase: run.stage ?? run.status,
          current_action: events.at(-1)?.message ?? '', updated_at: run.updated_at,
          completed_steps: null, total_steps: run.request.training?.steps ?? null,
          metrics: [], metrics_truncated: false, checkpoints: [], events, reproducibility: {} } });
        return;
      }
    }
    unexpectedRequests.push(`${request.method()} ${path}`);
    await route.fulfill({ status: 405, json: { detail: 'Request intentionally blocked by the training browser fixture.' } });
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Fine-tune', exact: true }).click();
  await page.getByRole('button', { name: 'Start a new fine-tuning', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Dataset', level: 2, exact: true })).toBeVisible();
  await expect(page.getByRole('radio', { name: 'fixture/pick-and-place', exact: true })).toBeVisible();
  await expect.poll(() => previews.length).toBeGreaterThanOrEqual(2);
  if (!emptyPreview) await expect(page.locator('.training-workspace .dx-camera video')).toHaveCount(2);
  const setup = page.getByRole('navigation', { name: 'Training setup' });
  await setup.getByRole('button', { name: 'Model', exact: true }).click();
  if (!await page.getByRole('group', { name: 'Base model', exact: true }).locator('input:checked').count()) {
    await page.getByRole('radio', { name: 'SmolVLA', exact: true }).locator('..').click();
  }
  await setup.getByRole('button', { name: 'Dataset', exact: true }).click();
  return { submitted, unexpectedRequests, previews, jobs };
}

test('training choices submit both selected cameras, pinned model, method and GPU', async ({ page }) => {
  const { submitted, unexpectedRequests } = await mockTrainingWorkspace(page);
  const datasetChoices = page.getByRole('group', { name: 'Inspected dataset', exact: true });
  await expect(datasetChoices.getByRole('radio')).toHaveCount(4);
  await expect(datasetChoices.getByRole('radio', { name: 'fixture/pick-and-place', exact: true })).toHaveCount(1);
  await expect(datasetChoices.getByRole('radio', { name: 'fixture/local', exact: true })).toBeDisabled();
  await expect(datasetChoices.getByRole('radio', { name: 'fixture/unpinned', exact: true })).toBeDisabled();
  await expect(datasetChoices.getByRole('radio', { name: 'fixture/failed', exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Start fine-tuning', exact: true })).toHaveCount(0);
  await expect(page.getByRole('checkbox', { name: 'observation.images.front', exact: true })).toBeChecked();
  await expect(page.getByRole('checkbox', { name: 'observation.images.wrist', exact: true })).toBeChecked();
  await page.getByRole('radio', { name: 'fixture/stack-blocks', exact: true }).locator('..').click();
  await expect(page.getByLabel(/^observation.images.side, episode 0/)).toBeVisible();
  const sideCamera = page.getByRole('checkbox', { name: 'observation.images.side', exact: true });
  const wristCamera = page.getByRole('checkbox', { name: 'observation.images.wrist', exact: true });
  await expect(sideCamera).toBeChecked();
  await expect(wristCamera).toBeChecked();
  await wristCamera.uncheck();
  await expect(wristCamera).not.toBeChecked();
  await expect(sideCamera).toBeChecked();
  await wristCamera.check();
  await page.getByRole('combobox', { name: 'Preview episode', exact: true }).selectOption('1');
  await expect(page.getByLabel(/^observation.images.side, episode 1/)).toHaveAttribute('src', /dataset-stack\/observation.images.side\/1$/);
  await expect(wristCamera).toBeChecked();
  await page.getByRole('button', { name: 'Next', exact: true }).click();
  await expect(page.getByRole('radio', { name: 'SmolVLA', exact: true })).toBeChecked();
  await expect(page.getByRole('radio', { name: 'LoRA', exact: true })).toBeChecked();
  await page.getByRole('radio', { name: 'QLoRA', exact: true }).locator('..').click();
  await page.getByRole('button', { name: 'Next', exact: true }).click();
  const compute = page.getByRole('combobox', { name: 'GPU', exact: true });
  await expectGpuChoices(page, ['L4', 'T4', 'A100']);
  await chooseGpu(page, 'L4');
  await page.locator('summary').filter({ hasText: /^Training settings/ }).click();
  await expect(page.getByRole('spinbutton', { name: 'Learning rate', exact: true })).toHaveValue('0.0001');
  await page.getByRole('spinbutton', { name: 'Steps', exact: true }).fill('123');
  await page.getByRole('spinbutton', { name: 'Batch size', exact: true }).fill('2');
  await page.getByRole('spinbutton', { name: 'Checkpoints', exact: true }).fill('0');
  await expect(page.getByRole('button', { name: 'Start fine-tuning', exact: true })).toBeDisabled();
  await page.getByRole('spinbutton', { name: 'Checkpoints', exact: true }).fill('5');
  await expectNoPageOverflow(page);
  await page.getByRole('button', { name: 'Start fine-tuning', exact: true }).click();
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0]).toMatchObject({
    operation: 'policy.finetune', dataset_job_id: 'dataset-stack', runtime_id: 'skypilot-gcp-L4', training_method: 'qlora',
    training: {
      steps: 123, batch_size: 2, learning_rate: 0.0001, save_every: 25, camera_key: 'observation.images.side',
      camera_keys: ['observation.images.side', 'observation.images.wrist'],
      model_id: 'lerobot/smolvla_base', model_revision: modelRevision,
    },
  });
  expect(unexpectedRequests).toEqual([]);
});

test('Psi-Zero explains incompatible v3 data before any training can be submitted', async ({ page }) => {
  const { submitted, unexpectedRequests } = await mockTrainingWorkspace(page, { configuredPsi: true });
  await page.getByRole('navigation', { name: 'Training setup' }).getByRole('button', { name: 'Model', exact: true }).click();
  await page.getByRole('radio', { name: 'Psi-Zero', exact: true }).locator('..').click();
  await expect(page.getByText(/Psi-Zero currently needs a LeRobot v2 dataset/)).toBeVisible();
  await expect(page.locator('.training-workspace').getByRole('button', { name: 'Next', exact: true })).toBeDisabled();
  await page.getByRole('navigation', { name: 'Training setup' }).getByRole('button', { name: 'Compute', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Start fine-tuning', exact: true })).toBeDisabled();
  expect(submitted).toEqual([]);
  expect(unexpectedRequests).toEqual([]);
});

test('Psi-Zero requires one camera and submits its frozen-backbone action-expert recipe on A100', async ({ page }) => {
  const { submitted, unexpectedRequests } = await mockTrainingWorkspace(page, { configuredPsi: true, datasetFormat: 'lerobot_v2' });
  const setup = page.getByRole('navigation', { name: 'Training setup' });
  await setup.getByRole('button', { name: 'Model', exact: true }).click();
  await page.getByRole('radio', { name: 'Psi-Zero', exact: true }).locator('..').click();
  await expect(page.getByText(/Psi-Zero requires 1 selected camera/)).toBeVisible();
  await setup.getByRole('button', { name: 'Dataset', exact: true }).click();
  await page.getByRole('checkbox', { name: 'observation.images.wrist', exact: true }).uncheck();
  await setup.getByRole('button', { name: 'Compute', exact: true }).click();
  await expectGpuChoices(page, ['A100']);
  await page.getByRole('button', { name: 'Start fine-tuning', exact: true }).click();
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0]).toMatchObject({ runtime_id: 'skypilot-gcp-A100', training_method: 'full', training: {
    model_id: 'USC-PSI-Lab/psi-model', model_revision: '4c6f9776fc5b18d87945254175e38bb74b9d7748', camera_keys: ['observation.images.front'], batch_size: 2,
  } });
  expect(unexpectedRequests).toEqual([]);
});

test('training requires at least one camera even when navigating directly to compute', async ({ page }) => {
  const { submitted, unexpectedRequests } = await mockTrainingWorkspace(page);
  const cameras = page.getByRole('checkbox', { name: /^observation\.images\./ });
  await expect(cameras).toHaveCount(2);
  for (const camera of await cameras.all()) await camera.uncheck();
  await expect(page.getByRole('button', { name: 'Next', exact: true })).toBeDisabled();
  await expect(page.getByText('Select at least one camera.', { exact: true })).toBeVisible();
  const steps = page.getByRole('navigation', { name: 'Training setup' });
  await steps.getByRole('button', { name: 'Compute', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Start fine-tuning', exact: true })).toBeDisabled();
  await expect(page.getByText('Select at least one camera.', { exact: true })).toBeVisible();
  await steps.getByRole('button', { name: 'Dataset', exact: true }).click();
  await page.getByRole('checkbox', { name: 'observation.images.front', exact: true }).check();
  await expect(page.getByRole('button', { name: 'Next', exact: true })).toBeEnabled();
  await steps.getByRole('button', { name: 'Compute', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Start fine-tuning', exact: true })).toBeEnabled();
  expect(submitted).toEqual([]);
  expect(unexpectedRequests).toEqual([]);
});

test('training retains one selector per camera when the preview returns no media', async ({ page }) => {
  const { submitted, unexpectedRequests } = await mockTrainingWorkspace(page, { emptyPreview: true });
  await expect(page.getByText('Preview unavailable', { exact: true })).toHaveCount(2);
  const cameras = page.getByRole('checkbox', { name: /^observation\.images\./ });
  await expect(cameras).toHaveCount(2);
  await expect(page.getByRole('checkbox', { name: 'observation.images.front', exact: true })).toBeChecked();
  await expect(page.getByRole('checkbox', { name: 'observation.images.wrist', exact: true })).toBeChecked();
  await page.getByRole('checkbox', { name: 'observation.images.front', exact: true }).uncheck();
  await expect(page.getByRole('checkbox', { name: 'observation.images.wrist', exact: true })).toBeChecked();
  await expect(page.getByRole('button', { name: 'Next', exact: true })).toBeEnabled();
  expect(submitted).toEqual([]);
  expect(unexpectedRequests).toEqual([]);
});

test('model picker offers SmolVLA on demand and marks unimplemented adapters Coming soon', async ({ page }) => {
  const { submitted, unexpectedRequests } = await mockTrainingWorkspace(page, { needsPreparation: true });
  await page.getByRole('button', { name: 'Next', exact: true }).click();
  const models = page.getByRole('group', { name: 'Base model', exact: true });
  await expect(models.getByRole('radio')).toHaveCount(6);
  await expect(models.getByRole('radio', { name: 'SmolVLA', exact: true })).toBeEnabled();
  await expect(models.getByRole('radio', { name: 'SmolVLA', exact: true })).toBeChecked();
  await expect(models.getByRole('radio', { name: 'SmolVLA', exact: true }).locator('..').getByText('Setup required', { exact: true })).toBeVisible();
  await expect(models.getByRole('radio', { name: 'SmolVLA', exact: true }).locator('..').getByText('Coming soon', { exact: true })).toHaveCount(0);
  for (const name of ['OpenVLA-OFT', 'OpenVLA', 'π₀', 'π₀.₅', 'GR00T N1.7']) {
    const choice = models.getByRole('radio', { name, exact: true });
    await expect(choice).toBeDisabled();
    await expect(choice.locator('..').getByText('Coming soon', { exact: true })).toBeVisible();
  }
  await expect(models.getByText('Setup required', { exact: true })).toHaveCount(1);
  await page.getByRole('button', { name: 'Next', exact: true }).click();
  await chooseGpu(page, 'L4');
  await expect(page.getByRole('button', { name: 'Start fine-tuning', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'Start fine-tuning', exact: true }).click();
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0]).toMatchObject({ operation: 'policy.finetune', runtime_id: 'skypilot-gcp-L4', training: { model_id: 'lerobot/smolvla_base', model_revision: modelRevision } });
  expect(unexpectedRequests).toEqual([]);
});

test('training honors the configured model recipe and its eligible runtime IDs', async ({ page }) => {
  const { submitted, unexpectedRequests } = await mockTrainingWorkspace(page, { configuredPi0: true });
  await page.getByRole('button', { name: 'Next', exact: true }).click();
  await expect(page.getByRole('radio', { name: 'π₀', exact: true })).toBeEnabled();
  await expect(page.getByRole('radio', { name: 'π₀', exact: true }).locator('..').getByText('Coming soon', { exact: true })).toHaveCount(0);
  await page.getByRole('radio', { name: 'π₀', exact: true }).locator('..').click();
  await expect(page.getByRole('radio', { name: 'LoRA', exact: true })).toBeChecked();
  await expect(page.getByRole('radio', { name: 'QLoRA', exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Next', exact: true }).click();
  const gpu = page.getByRole('combobox', { name: 'GPU', exact: true });
  await chooseGpu(page, 'L4');
  await expect(page.getByRole('button', { name: 'Start fine-tuning', exact: true })).toBeDisabled();
  await expect(gpu).toHaveAttribute('value', 'L4');
  await chooseGpu(page, 'A100');
  await expect(gpu).toHaveAttribute('value', 'A100');
  await page.getByRole('button', { name: 'Start fine-tuning', exact: true }).click();
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0]).toMatchObject({
    operation: 'policy.finetune', runtime_id: 'skypilot-gcp-A100', dataset_job_id: 'dataset-new', training_method: 'lora',
    training: {
      model_id: 'lerobot/pi0_libero_finetuned_v044', model_revision: modelRevision,
      camera_key: 'observation.images.front', camera_keys: ['observation.images.front', 'observation.images.wrist'],
    },
  });
  expect(unexpectedRequests).toEqual([]);
});

test('training cannot start when no CUDA training runtime is configured', async ({ page }) => {
  const { submitted, unexpectedRequests } = await mockTrainingWorkspace(page, { runtimes: trainingRuntimes.slice(0, 2) });
  await page.getByRole('button', { name: 'Next', exact: true }).click();
  await page.getByRole('button', { name: 'Next', exact: true }).click();
  await expectGpuChoices(page, ['L4', 'T4', 'A100']);
  await chooseGpu(page, 'T4');
  await expect(page.getByRole('combobox', { name: 'GPU', exact: true })).toHaveAttribute('value', 'T4');
  await expect(page.getByRole('button', { name: 'Start fine-tuning', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Connect account', exact: true })).toBeVisible();
  await expectNoPageOverflow(page);
  expect(submitted).toEqual([]);
  expect(unexpectedRequests).toEqual([]);
});

test('training cards and step navigation work from the keyboard at 320px', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 740 });
  const { submitted, unexpectedRequests } = await mockTrainingWorkspace(page);
  const dataset = page.getByRole('radio', { name: 'fixture/stack-blocks', exact: true });
  await dataset.focus();
  await page.keyboard.press('Space');
  await expect(dataset).toBeChecked();
  await expectNoPageOverflow(page);
  const steps = page.getByRole('navigation', { name: 'Training setup' });
  await steps.getByRole('button', { name: 'Model', exact: true }).focus();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('heading', { name: 'Model', level: 2, exact: true })).toBeFocused();
  const method = page.getByRole('radio', { name: 'QLoRA', exact: true });
  await method.focus();
  await page.keyboard.press('Space');
  await expect(method).toBeChecked();
  await expectNoPageOverflow(page);
  await steps.getByRole('button', { name: 'Compute', exact: true }).focus();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('heading', { name: 'Compute', level: 2, exact: true })).toBeFocused();
  const runtime = page.getByRole('combobox', { name: 'GPU', exact: true });
  await runtime.focus();
  await expect(runtime).toBeFocused();
  // Playwright's native picker API is portable across desktop/mobile OS menus.
  await chooseGpu(page, 'T4');
  await expect(runtime).toHaveAttribute('value', 'T4');
  await page.locator('summary').filter({ hasText: /^Training settings/ }).focus();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('spinbutton', { name: 'Steps', exact: true })).toBeVisible();
  await expectNoPageOverflow(page);
  expect(submitted).toEqual([]);
  expect(unexpectedRequests).toEqual([]);
});

test('resuming an interrupted run keeps its original dataset, method and training recipe', async ({ page }) => {
  const { submitted, unexpectedRequests } = await mockTrainingWorkspace(page, { previousRun: true });
  const steps = page.getByRole('navigation', { name: 'Training setup' });
  await steps.getByRole('button', { name: 'Compute', exact: true }).click();
  await page.getByText('Resume a previous run', { exact: true }).click();
  const resumeChoices = page.getByRole('group', { name: 'Resume checkpoint', exact: true });
  await expect(resumeChoices.getByRole('radio')).toHaveCount(2);
  await expect(resumeChoices.getByRole('radio', { name: 'Completed training checkpoint', exact: true })).toHaveCount(0);
  await page.getByRole('radio', { name: 'Last saved checkpoint · interrup', exact: true }).locator('..').click();
  await steps.getByRole('button', { name: 'Dataset', exact: true }).click();
  await expect(page.getByRole('radio', { name: 'fixture/stack-blocks', exact: true })).toBeChecked();
  await expect(page.getByRole('radio', { name: 'fixture/stack-blocks', exact: true })).toBeDisabled();
  await expect(page.getByRole('radio', { name: 'fixture/pick-and-place', exact: true })).not.toBeChecked();
  await expect(page.getByRole('checkbox', { name: 'observation.images.side', exact: true })).toBeChecked();
  await expect(page.getByRole('checkbox', { name: 'observation.images.side', exact: true })).toBeDisabled();
  await expect(page.getByRole('checkbox', { name: 'observation.images.wrist', exact: true })).not.toBeChecked();
  await steps.getByRole('button', { name: 'Model', exact: true }).click();
  await expect(page.getByRole('radio', { name: 'SmolVLA', exact: true })).toBeChecked();
  await expect(page.getByRole('radio', { name: 'SmolVLA', exact: true })).toBeDisabled();
  await expect(page.getByRole('radio', { name: 'QLoRA', exact: true })).toBeChecked();
  await expect(page.getByRole('radio', { name: 'QLoRA', exact: true })).toBeDisabled();
  await expect(page.getByRole('radio', { name: 'LoRA', exact: true })).not.toBeChecked();
  await steps.getByRole('button', { name: 'Compute', exact: true }).click();
  await chooseGpu(page, 'L4');
  await page.getByRole('button', { name: 'Resume fine-tuning', exact: true }).click();
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0]).toMatchObject({
    operation: 'policy.finetune', runtime_id: 'skypilot-gcp-L4', resume_job_id: 'interrupted-training',
    dataset_job_id: 'dataset-stack', training_method: 'qlora', training: null,
  });
  expect(unexpectedRequests).toEqual([]);
});


for (const gpu of ['L4', 'T4', 'A100']) {
  test(`simple GPU dropdown sends the exact ${gpu} selection with Kite UI defaults`, async ({ page }, testInfo) => {
    const { submitted, unexpectedRequests } = await mockTrainingWorkspace(page);
    await page.getByRole('navigation', { name: 'Training setup' }).getByRole('button', { name: 'Compute', exact: true }).click();
    const form = page.locator('.training-workspace');
    const select = form.getByRole('combobox', { name: 'GPU', exact: true });
    await expectGpuChoices(page, ['L4', 'T4', 'A100']);
    await chooseGpu(page, gpu);
    await expect(select).toHaveAttribute('value', gpu);
    await expect(form.getByRole('radio', { name: /Google Cloud|Local machine/ })).toHaveCount(0);
    await expect(form.getByText(/SkyPilot|GiB|1 GPU per run/)).toHaveCount(0);
    await expect(form.getByRole('button', { name: 'Refresh', exact: true })).toHaveCount(0);
    await expect(form.getByText('Worker setup', { exact: true })).toHaveCount(0);
    await form.locator('summary').filter({ hasText: /^Training settings/ }).click();
    await expect(form.getByRole('spinbutton', { name: 'Steps', exact: true })).toHaveValue('20000');
    await expect(form.getByRole('spinbutton', { name: 'Batch size', exact: true })).toHaveValue('64');
    await expect(form.getByRole('spinbutton', { name: 'Checkpoints', exact: true })).toHaveValue('5');
    await expect(form.getByRole('spinbutton', { name: 'Learning rate', exact: true })).toHaveValue('0.0001');
    await expectNoPageOverflow(page);
    if (gpu === 'A100') {
      await select.click();
      await expect(page.getByRole('listbox', { name: 'GPU', exact: true })).toBeVisible();
      const screenshot = testInfo.outputPath('training-compute-dropdown.png');
      await page.screenshot({ path: screenshot, fullPage: true });
      await testInfo.attach('training-compute-dropdown', { path: screenshot, contentType: 'image/png' });
      await page.keyboard.press('Escape');
    }
    await form.getByRole('button', { name: 'Start fine-tuning', exact: true }).click();
    await expect.poll(() => submitted.length).toBe(1);
    expect(submitted[0]).toMatchObject({
      operation: 'policy.finetune', runtime_id: `skypilot-gcp-${gpu}`, training_method: 'lora', timeout_seconds: 86400,
      training: { steps: 20000, batch_size: 64, save_every: 4000, gradient_accumulation_steps: 1 },
    });
    expect(unexpectedRequests).toEqual([]);
  });
}

test('a saved account starts training in one click while setup happens in the queued job', async ({ page }) => {
  const { submitted, unexpectedRequests, jobs } = await mockTrainingWorkspace(page, { needsPreparation: true, queuedRun: true });
  await page.getByRole('navigation', { name: 'Training setup' }).getByRole('button', { name: 'Compute', exact: true }).click();
  const form = page.locator('.training-workspace');
  const picker = form.getByRole('combobox', { name: 'GPU', exact: true });
  await chooseGpu(page, 'L4');
  await expect(form.getByRole('button', { name: /Prepare|Check SkyPilot|Connect account/ })).toHaveCount(0);
  await expect(form.getByText(/SkyPilot|setup commands|gcloud auth/i)).toHaveCount(0);
  await expect(form.getByRole('button', { name: 'Start fine-tuning', exact: true })).toBeEnabled();
  await form.getByRole('button', { name: 'Start fine-tuning', exact: true }).click();
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0].runtime_id).toBe('skypilot-gcp-L4');
  await expect(page.getByRole('region', { name: 'Fine-tuning job', exact: true })).toBeVisible();
  await expect(form).toHaveCount(0);
  await expect(page.getByRole('heading', { name: 'Preparing your GPU', exact: true })).toBeVisible();
  // Submission leaves the form, so a second Enter cannot submit a duplicate.
  await page.keyboard.press('Enter');
  expect(submitted).toHaveLength(1);
  const run = jobs.find(job => job.id === 'mock-training-1')!;
  run.status = 'running';
  run.stage = 'operation';
  await expect(page.getByRole('region', { name: 'Fine-tuning job', exact: true }).getByText('running', { exact: true })).toBeVisible();
  expect(submitted).toHaveLength(1);
  run.status = 'succeeded';
  run.stage = null;
  run.result = { artifacts: [], reports: [], decision: 'completed' };
  await expect(page.getByRole('region', { name: 'Fine-tuning job', exact: true }).getByText('succeeded', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Back to jobs', exact: true }).click();
  await page.getByRole('button', { name: 'Start a new fine-tuning', exact: true }).click();
  await page.getByRole('navigation', { name: 'Training setup' }).getByRole('button', { name: 'Compute', exact: true }).click();
  await expect(form.getByRole('button', { name: 'Start fine-tuning', exact: true })).toBeEnabled();
  await expect(picker).toBeEnabled();
  expect(submitted).toHaveLength(1);
  expect(unexpectedRequests).toEqual([]);
});

test('an unavailable selected GPU remains selected and never falls back to a ready GPU', async ({ page }) => {
  const { submitted, unexpectedRequests } = await mockTrainingWorkspace(page, {
    runtimes: trainingRuntimes.map(runtime => runtime.accelerator === 'T4' ? { ...runtime, enabled: false, unavailable_reason: 'Mock GPU setup required' } : runtime),
  });
  await page.getByRole('navigation', { name: 'Training setup' }).getByRole('button', { name: 'Compute', exact: true }).click();
  const gpu = page.getByRole('combobox', { name: 'GPU', exact: true });
  await expect(page.getByRole('button', { name: 'Start fine-tuning', exact: true })).toBeEnabled();
  await chooseGpu(page, 'T4');
  await expect(gpu).toHaveAttribute('value', 'T4');
  await expect(page.getByRole('button', { name: 'Start fine-tuning', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Connect account', exact: true })).toHaveCount(0);
  await expect(page.locator('.training-workspace').getByText('Mock GPU setup required', { exact: true })).toHaveCount(0);
  await chooseGpu(page, 'L4');
  await expect(gpu).toHaveAttribute('value', 'L4');
  await expect(page.getByRole('button', { name: 'Start fine-tuning', exact: true })).toBeEnabled();
  expect(submitted).toEqual([]);
  expect(unexpectedRequests).toEqual([]);
});

test('local GPUs stay out of the dropdown until local runs are explicitly enabled', async ({ page }) => {
  const { submitted, unexpectedRequests } = await mockTrainingWorkspace(page);
  await page.getByRole('navigation', { name: 'Training setup' }).getByRole('button', { name: 'Compute', exact: true }).click();
  const gpu = page.getByRole('combobox', { name: 'GPU', exact: true });
  await expectGpuChoices(page, ['L4', 'T4', 'A100']);
  await gpu.click();
  await expect(page.getByRole('listbox').getByRole('option', { name: /RTX/ })).toHaveCount(0);
  await page.keyboard.press('Escape');
  await expect(page.getByText('Local runs disabled', { exact: true })).toHaveCount(0);
  expect(submitted).toEqual([]);
  expect(unexpectedRequests).toEqual([]);
});

test('an explicitly enabled local GPU can be selected without adding provider controls', async ({ page }) => {
  const { submitted, unexpectedRequests } = await mockTrainingWorkspace(page, { local: { enabled: true, label: 'Robotics lab' } });
  await page.getByRole('navigation', { name: 'Training setup' }).getByRole('button', { name: 'Compute', exact: true }).click();
  const gpu = page.getByRole('combobox', { name: 'GPU', exact: true });
  await expectGpuChoices(page, ['L4', 'T4', 'A100', 'RTX 3070 workstation', 'RTX 4090 workstation']);
  await chooseGpu(page, 'gpu-4090', 'RTX 4090 workstation');
  await expect(gpu).toHaveAttribute('value', 'gpu-4090');
  await page.getByRole('button', { name: 'Start fine-tuning', exact: true }).click();
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0].runtime_id).toBe('gpu-4090');
  expect(unexpectedRequests).toEqual([]);
});

test('dataset intake waits for a confirmed project and preserves its draft during project recovery', async ({ page }) => {
  await page.clock.install();
  let releaseProjects!: () => void;
  const pendingProjects = new Promise<void>(resolve => { releaseProjects = resolve; });
  let projectError = false;
  const submitted: { projectId: string; body: unknown }[] = [];
  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    const lookup = /^\/api\/v1\/projects\/(first|second)\/submissions\/([^/]+)$/.exec(path);
    if (request.method() === 'GET' && lookup && url.searchParams.get('operation') === 'dataset.inspect') {
      const key = decodeURIComponent(lookup[2]);
      if (request.headers()['idempotency-key'] === key) {
        // This fixture rejects intake before allocation, so no key has an accepted binding.
        return route.fulfill({ status: 404, headers: { 'Idempotency-Key': key, 'Cache-Control': 'no-store' },
          json: { detail: 'No accepted submission found for this project, operation and key' } });
      }
    }
    if (request.method() === 'GET' && path === '/api/v1/projects') {
      await pendingProjects;
      return route.fulfill(projectError
        ? { status: 503, json: { detail: 'Project fixture unavailable' } }
        : { json: ['first', 'second'].map(id => ({ id, name: `Project ${id}`, created_at: trainingTimestamp })) });
    }
    if (request.method() === 'POST' && /^\/api\/v1\/projects\/(first|second)\/intakes$/.test(path) && request.headers()['idempotency-key']) {
      submitted.push({ projectId: path.split('/')[4], body: request.postDataJSON() });
      return route.fulfill({ status: 422, headers: { 'Idempotency-Key': request.headers()['idempotency-key'], 'Cache-Control': 'no-store' },
        json: { detail: 'Captured intake; no worker started' } });
    }
    if (request.method() !== 'GET' || path.includes('/submissions/')) return route.fulfill({ status: 405, json: { detail: 'Unexpected request blocked by intake fixture.' } });
    if (path === '/api/v1/health') return route.fulfill({ json: { status: 'ok', version: 'intake-fixture' } });
    return route.fulfill({ json: [] });
  });
  await page.goto('/');
  const repository = page.getByLabel('Dataset repository', { exact: true });
  const revision = page.getByLabel('Revision', { exact: true });
  const inspect = page.getByRole('button', { name: 'Inspect dataset', exact: true });
  await expect(repository).toBeDisabled();
  await page.locator('.intake-advanced > summary').click();
  await expect(revision).toBeDisabled();
  await expect(page.getByRole('radio', { name: 'Hugging Face', exact: true })).toBeDisabled();
  await expect(inspect).toBeDisabled();
  await expect(page.getByText('Loading projects before importing a dataset.', { exact: true })).toBeVisible();
  releaseProjects();
  await expect(repository).toBeEnabled();
  await repository.fill('fixture/operator-entry');
  // Confirm the mounted form reflects the actual operator input before refetch.
  await page.locator('.intake-advanced > summary').click();
  await revision.fill('operator-revision');
  projectError = true;
  await page.clock.fastForward(6_000);
  await page.evaluate(() => window.dispatchEvent(new Event('visibilitychange')));
  await expect(page.getByRole('button', { name: 'Retry projects', exact: true })).toBeVisible();
  await expect(repository).toBeDisabled();
  await expect(revision).toBeDisabled();
  await expect(inspect).toBeDisabled();
  await expect(page.getByText('Project list unavailable. Retry projects to continue.', { exact: true })).toBeVisible();
  await expect(repository).toHaveValue('fixture/operator-entry');
  await expect(revision).toHaveValue('operator-revision');
  projectError = false;
  await page.getByRole('button', { name: 'Retry projects', exact: true }).click();
  await expect(repository).toBeEnabled();
  await expect(repository).toHaveValue('fixture/operator-entry');
  await expect(revision).toHaveValue('operator-revision');
  await inspect.click();
  await expect.poll(() => submitted).toEqual([{ projectId: 'first', body: { source: 'huggingface', repo_id: 'fixture/operator-entry', revision: 'operator-revision' } }]);
  await page.getByLabel('Current project', { exact: true }).selectOption('second');
  await expect(repository).toHaveValue('codywang/so101_pickup_test');
  await repository.fill('fixture/second-project');
  await inspect.click();
  await expect.poll(() => submitted).toEqual([
    { projectId: 'first', body: { source: 'huggingface', repo_id: 'fixture/operator-entry', revision: 'operator-revision' } },
    { projectId: 'second', body: { source: 'huggingface', repo_id: 'fixture/second-project', revision: 'main' } },
  ]);
});

test('blank dataset revision uses latest and reused inspections keep one history entry and cached previews', async ({ page }) => {
  const projectId = 'inspection-cache';
  const stamp = '2026-09-26T12:00:00Z';
  const repoId = 'fixture/cached';
  const revision = 'a'.repeat(40);
  const job = {
    id: 'inspection-cached', project_id: projectId, kind: 'dataset.inspect', status: 'succeeded',
    request: { source: 'huggingface', repo_id: repoId, revision }, created_at: stamp, updated_at: stamp,
    result: {
      source: 'huggingface', repo_id: repoId, revision, format: 'lerobot_v3', inspection_scope: 'metadata_only',
      total_episodes: 1, total_frames: 10, fps: 10, metadata_sha256: 'b'.repeat(64), inspected_at: stamp, warnings: [],
      features: { action: { dtype: 'float32', shape: [2] }, 'observation.state': { dtype: 'float32', shape: [2] } },
    },
  };
  const submitted: unknown[] = [];
  const submissions = new Map<string, { body: unknown; job: typeof job }>();
  const previews: string[] = [];
  // Keep the existing live read-only bootstrap, but never let an unexpected write reach it.
  await page.route('**/api/v1/**', route => route.request().method() === 'GET' ? route.fallback()
    : route.fulfill({ status: 405, json: { detail: 'Unexpected mutation blocked by intake fixture.' } }));
  await page.route('**/api/v1/projects', route => route.fulfill({ json: [{ id: projectId, name: 'Saved inspections', created_at: stamp }] }));
  await page.route(`**/api/v1/projects/${projectId}/jobs`, route => route.fulfill({ json: [job] }));
  await page.route(`**/api/v1/projects/${projectId}/submissions/*`, route => {
    const request = route.request();
    const url = new URL(request.url());
    const key = decodeURIComponent(url.pathname.split('/').at(-1)!);
    if (request.method() !== 'GET' || url.searchParams.get('operation') !== 'dataset.inspect' || request.headers()['idempotency-key'] !== key) {
      return route.fulfill({ status: 405, json: { detail: 'Unsupported submission lookup blocked by fixture.' } });
    }
    const saved = submissions.get(key);
    return route.fulfill({ status: saved ? 200 : 404, headers: { 'Idempotency-Key': key, 'Cache-Control': 'no-store' },
      json: saved?.job ?? { detail: 'No accepted submission found for this project, operation and key' } });
  });
  await page.route(`**/api/v1/projects/${projectId}/intakes`, route => {
    const request = route.request();
    const key = request.headers()['idempotency-key'];
    if (request.method() !== 'POST' || !key) return route.fulfill({ status: 405, json: { detail: 'Unsupported intake request blocked by fixture.' } });
    const body = request.postDataJSON();
    const saved = submissions.get(key);
    const headers = { 'Idempotency-Key': key, 'Cache-Control': 'no-store' };
    if (saved) {
      const same = JSON.stringify(saved.body) === JSON.stringify(body);
      return route.fulfill({ status: same ? 202 : 409, headers,
        json: same ? saved.job : { detail: 'Submission key already belongs to another request.' } });
    }
    submitted.push(body);
    submissions.set(key, { body, job });
    return route.fulfill({ status: 202, headers, json: job });
  });
  await page.route('**/api/v1/jobs/inspection-cached/episodes**', route => {
    const isIndex = new URL(route.request().url()).pathname.endsWith('/episodes');
    previews.push(isIndex ? 'index' : 'episode');
    const episode = { episode_index: 0, frame_count: 10, duration_seconds: 1, tasks: [] };
    return route.fulfill({ json: isIndex ? {
      repo_id: repoId, revision, total_episodes: 1, offset: 0, limit: 6, episodes: [episode], warnings: [],
    } : { ...episode, repo_id: repoId, revision, cameras: [], samples: [], state_names: [], action_names: [], warnings: [] } });
  });
  await page.goto('/');
  await page.getByLabel('Dataset repository', { exact: true }).fill(repoId);
  await page.locator('.intake-advanced > summary').click();
  await expect(page.getByLabel('Revision', { exact: true })).toHaveValue('');
  await expect(page.getByLabel('Revision', { exact: true })).not.toHaveAttribute('required', '');
  await page.getByRole('button', { name: 'Inspect dataset', exact: true }).click();
  await expect(page.getByRole('heading', { name: repoId, exact: true })).toBeVisible();
  await expect.poll(() => previews).toEqual(['index', 'episode']);
  await page.getByRole('button', { name: 'Change source', exact: true }).click();
  await page.getByRole('button', { name: 'Inspect dataset', exact: true }).click();
  await expect(page.getByRole('heading', { name: repoId, exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: /^Inspection/ })).toHaveText('Inspection1');
  await expect(page.getByLabel('History', { exact: true })).toHaveCount(0);
  expect(submitted).toEqual([
    { source: 'huggingface', repo_id: repoId, revision: 'main' },
    { source: 'huggingface', repo_id: repoId, revision: 'main' },
  ]);
  expect(previews).toEqual(['index', 'episode']);
});


test('visiting compute settings preserves the draft through the jobs-first entry', async ({ page }) => {
  const { submitted, unexpectedRequests } = await mockTrainingWorkspace(page, {
    runtimes: trainingRuntimes.filter(runtime => runtime.accelerator !== 'T4'),
  });
  await page.getByRole('checkbox', { name: 'observation.images.wrist', exact: true }).uncheck();
  await page.getByRole('button', { name: 'Next', exact: true }).click();
  await page.getByRole('radio', { name: 'QLoRA', exact: true }).locator('..').click();
  await page.getByRole('button', { name: 'Next', exact: true }).click();
  await chooseGpu(page, 'T4');
  await page.locator('summary').filter({ hasText: /^Training settings/ }).click();
  await page.getByRole('spinbutton', { name: 'Steps', exact: true }).fill('123');
  await page.getByRole('spinbutton', { name: 'Batch size', exact: true }).fill('2');
  await page.getByRole('button', { name: 'Connect account', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Cloud providers', exact: true })).toBeVisible();
  await expect(page.getByRole('navigation', { name: 'Training setup', exact: true })).toBeHidden();
  await page.getByRole('button', { name: 'Fine-tune', exact: true }).click();
  await page.getByRole('button', { name: 'Start a new fine-tuning', exact: true }).click();
  await page.getByRole('navigation', { name: 'Training setup' }).getByRole('button', { name: 'Compute', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Compute', level: 2, exact: true })).toBeVisible();
  await expect(page.getByRole('combobox', { name: 'GPU', exact: true })).toHaveAttribute('value', 'T4');
  await page.locator('summary').filter({ hasText: /^Training settings/ }).click();
  await expect(page.getByRole('spinbutton', { name: 'Steps', exact: true })).toHaveValue('123');
  await expect(page.getByRole('spinbutton', { name: 'Batch size', exact: true })).toHaveValue('2');
  await expect(page.getByRole('button', { name: 'Start fine-tuning', exact: true })).toBeDisabled();
  await chooseGpu(page, 'L4');
  await page.getByRole('button', { name: 'Start fine-tuning', exact: true }).click();
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0]).toMatchObject({ runtime_id: 'skypilot-gcp-L4', training_method: 'qlora', training: { steps: 123, batch_size: 2, save_every: 25, camera_keys: ['observation.images.front'] } });
  expect(unexpectedRequests).toEqual([]);
});

for (const fixture of [
  { label: 'legacy starter defaults', saved: { trainingSteps: 1000, batchSize: 1 }, expected: { trainingSteps: 20000, batchSize: 64, checkpointCount: 5, checkpointIntervalOverride: null }, interval: 4000 },
  { label: 'shipped version 2 five-step cadence', saved: { trainingSteps: 20000, batchSize: 64, checkpointEvery: 5, trainingDefaultsVersion: 2 }, expected: { trainingSteps: 20000, batchSize: 64, checkpointCount: 5, checkpointIntervalOverride: null }, interval: 4000 },
  { label: 'custom legacy cadence', saved: { trainingSteps: 1234, batchSize: 3, checkpointEvery: 7 }, expected: { trainingSteps: 1234, batchSize: 3, checkpointCount: 177, checkpointIntervalOverride: 7 }, interval: 7 },
  { label: 'intentional version 2 small recipe', saved: { trainingSteps: 1000, batchSize: 1, trainingDefaultsVersion: 2 }, expected: { trainingSteps: 1000, batchSize: 1, checkpointCount: 5, checkpointIntervalOverride: null }, interval: 200 },
  { label: 'version 3 checkpoint count', saved: { trainingSteps: 900, batchSize: 4, checkpointCount: 3, trainingDefaultsVersion: 3 }, expected: { trainingSteps: 900, batchSize: 4, checkpointCount: 3, checkpointIntervalOverride: null }, interval: 300 },
]) {
  test(`training storage migrates ${fixture.label} without overwriting deliberate settings`, async ({ page }) => {
    await page.addInitScript(({ key, saved }) => localStorage.setItem(key, JSON.stringify(saved)), { key: `firebird.workflow.${trainingProject}`, saved: fixture.saved });
    const { submitted, unexpectedRequests } = await mockTrainingWorkspace(page);
    await page.getByRole('navigation', { name: 'Training setup' }).getByRole('button', { name: 'Compute', exact: true }).click();
    await page.locator('summary').filter({ hasText: /^Training settings/ }).click();
    await expect(page.getByRole('spinbutton', { name: 'Steps', exact: true })).toHaveValue(String(fixture.expected.trainingSteps));
    await expect(page.getByRole('spinbutton', { name: 'Batch size', exact: true })).toHaveValue(String(fixture.expected.batchSize));
    await expect(page.getByRole('spinbutton', { name: 'Checkpoints', exact: true })).toHaveValue(String(fixture.expected.checkpointCount));
    if (fixture.expected.checkpointIntervalOverride) await expect(page.getByText(`Saved preference: every ${fixture.interval} steps. Change this count to replace it.`, { exact: false })).toBeVisible();
    await expect.poll(() => page.evaluate(key => JSON.parse(localStorage.getItem(key) ?? '{}'), `firebird.workflow.${trainingProject}`)).toMatchObject({ ...fixture.expected, trainingDefaultsVersion: 3 });
    await page.getByRole('button', { name: 'Start fine-tuning', exact: true }).click();
    await expect.poll(() => submitted.length).toBe(1);
    expect(submitted[0].training.save_every).toBe(fixture.interval);
    expect(unexpectedRequests).toEqual([]);
  });
}

test('checkpoint count follows run length and replaces a clearly identified saved cadence', async ({ page }) => {
  await page.addInitScript(key => localStorage.setItem(key, JSON.stringify({ trainingSteps: 1234, batchSize: 3, checkpointEvery: 7, trainingDefaultsVersion: 2 })), `firebird.workflow.${trainingProject}`);
  const { submitted, unexpectedRequests } = await mockTrainingWorkspace(page);
  await page.getByRole('navigation', { name: 'Training setup' }).getByRole('button', { name: 'Compute', exact: true }).click();
  await page.locator('summary').filter({ hasText: /^Training settings/ }).click();
  await expect(page.getByText('Saved preference: every 7 steps. Change this count to replace it.', { exact: false })).toBeVisible();
  await page.getByRole('spinbutton', { name: 'Checkpoints', exact: true }).fill('5');
  await page.getByRole('spinbutton', { name: 'Steps', exact: true }).fill('20000');
  await expect(page.getByText('Every 4,000 steps. Final checkpoint included.', { exact: false })).toBeVisible();
  await expect(page.getByText(/Saved preference:/)).toHaveCount(0);
  await page.getByRole('button', { name: 'Start fine-tuning', exact: true }).click();
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0].training.save_every).toBe(4000);
  await expect.poll(() => page.evaluate(key => JSON.parse(localStorage.getItem(key) ?? '{}'), `firebird.workflow.${trainingProject}`)).toMatchObject({ checkpointCount: 5, checkpointIntervalOverride: null, trainingDefaultsVersion: 3 });
  expect(unexpectedRequests).toEqual([]);
});

test('training summarizes the latest checkpoint and keeps full activity behind details', async ({ page }) => {
  const { submitted, unexpectedRequests } = await mockTrainingWorkspace(page, {
    events: [
      { sequence: 1, stage: 'operation', message: 'Checkpoint 5 saved locally', timestamp: trainingTimestamp },
      { sequence: 2, stage: 'operation', message: 'Checkpoint transfer completed', timestamp: trainingTimestamp },
      { sequence: 3, stage: 'operation', message: 'Checkpoint 10 saved locally', timestamp: trainingTimestamp },
    ],
  });
  await page.getByRole('navigation', { name: 'Training setup' }).getByRole('button', { name: 'Compute', exact: true }).click();
  await page.getByRole('button', { name: 'Start fine-tuning', exact: true }).click();
  await expect.poll(() => submitted.length).toBe(1);
  const runs = page.getByRole('region', { name: 'Fine-tuning job', exact: true });
  await expect(runs.locator('.training-current-action')).toHaveText('Training complete');
  await expect(runs.getByText('Latest checkpoint: step 10', { exact: true })).toBeVisible();
  await expect(runs.getByLabel('Persisted training activity')).toHaveCount(0);
  await runs.locator('.training-details > summary').click();
  const activity = runs.getByLabel('Persisted training activity');
  await expect(activity.getByText('Checkpoint 10 saved locally', { exact: true })).toBeVisible();
  await expect(activity.getByText('Checkpoint 5 saved locally', { exact: true })).toBeVisible();
  await expect(activity.getByText('Checkpoint transfer completed', { exact: true })).toBeVisible();
  expect(unexpectedRequests).toEqual([]);
});

function savedWorkflowModel(projectId: string) {
  return { id: 'source', project_id: projectId, job_id: 'import-source', label: 'Synthetic policy', format: 'gguf', path: 'fixture/policy', manifest_sha256: 'a'.repeat(64), file_bytes: 100, parent_ids: [], metadata: { architecture: 'smolvla', precision: 'float' } };
}

test('quantization submits Q4 only after an explicit experimental choice', async ({ page }) => {
  const requests: { candidates: { language: string; vision: string | null }[] }[] = [];
  const timestamp = '2026-09-26T12:00:00Z';
  await page.route('**/api/v1/projects', route => route.fulfill({ json: [{ id: 'precision-review', name: 'Precision fixture', created_at: timestamp }] }));
  await page.route('**/api/v1/projects/precision-review/jobs', route => route.fulfill({ json: [] }));
  await page.route('**/api/v1/projects/precision-review/artifacts', route => route.fulfill({ json: [savedWorkflowModel('precision-review')] }));
  await page.route('**/api/v1/policy-options', route => route.fulfill({ json: {
    runtimes: [{ id: 'fixture', label: 'CPU fixture', device: 'cpu', training: false, simulation: false }],
    sources: [{ id: 'source', label: 'Synthetic policy', task: 'fixture' }],
    training_methods: [], default_training_method: 'lora',
    quantization_defaults: { cpu: { language: 'Q8_0', vision: null }, cuda: { language: 'Q8_0', vision: null }, note: 'Fixture only' },
  } }));
  await page.route('**/api/v1/projects/precision-review/policy-jobs', route => {
    requests.push(route.request().postDataJSON());
    return route.fulfill({ status: 422, json: { detail: 'Captured request; no worker started' } });
  });
  async function submitQuantization(count: number) {
    await page.getByRole('button', { name: 'Quantize', exact: true }).click();
    await page.getByRole('button', { name: 'Choose Synthetic policy · source', exact: true }).click();
    await page.getByRole('group', { name: 'My model', exact: true }).locator('input[value="source"]').check();
    await page.getByRole('button', { name: 'Run quantization workflow', exact: true }).click();
    await expect.poll(() => requests.length).toBe(count);
  }
  await page.goto('/');
  await submitQuantization(1);
  expect(requests[0].candidates).toEqual([{ language: 'Q8_0', vision: null }]);
  await page.getByRole('button', { name: 'Settings & diagnostics', exact: true }).click();
  await page.getByRole('button', { name: 'Workflow settings', exact: true }).click();
  const compare = page.getByLabel('Compare Q8 and Q4 (experimental)');
  await expect(compare).not.toBeChecked();
  await compare.check();
  await submitQuantization(2);
  expect(requests[1].candidates).toEqual([{ language: 'Q8_0', vision: null }, { language: 'Q4_0', vision: null }]);
  await page.getByRole('button', { name: 'Settings & diagnostics', exact: true }).click();
  await page.getByRole('button', { name: 'Workflow settings', exact: true }).click();
  const restoredCompare = page.getByLabel('Compare Q8 and Q4 (experimental)');
  // Settings remounts; wait until its saved preference has been restored.
  await expect(restoredCompare).toBeChecked();
  await restoredCompare.uncheck();
  await expect(restoredCompare).not.toBeChecked();
  await page.getByLabel('Quantization recipe').selectOption('Q4_0');
  await submitQuantization(3);
  expect(requests[2].candidates).toEqual([{ language: 'Q4_0', vision: null }]);
});

test('Spatial settings require explicit task and parity choices in the submitted request', async ({ page }) => {
  const submitted: { evaluation: Record<string, unknown> }[] = [];
  let releaseProjects!: () => void;
  const projectsReady = new Promise<void>(resolve => { releaseProjects = resolve; });
  await page.route('**/api/v1/projects', async route => {
    await projectsReady;
    await route.fulfill({ json: [{ id: 'spatial-review', name: 'Spatial fixture', created_at: '2026-09-26T12:00:00Z' }] });
  });
  await page.route('**/api/v1/projects/spatial-review/jobs', route => route.fulfill({ json: [] }));
  await page.route('**/api/v1/projects/spatial-review/artifacts', route => route.fulfill({ json: [savedWorkflowModel('spatial-review')] }));
  await page.route('**/api/v1/policy-options', route => route.fulfill({ json: {
    runtimes: [{ id: 'fixture', label: 'Synthetic L4 fixture', device: 'cuda', training: false, simulation: true }],
    sources: [{ id: 'source', label: 'Synthetic Spatial policy', task: 'libero_spatial' }],
    training_methods: [], default_training_method: 'lora',
    quantization_defaults: { cpu: { language: 'Q8_0', vision: null }, cuda: { language: 'Q8_0', vision: null }, note: 'Fixture only' },
  } }));
  await page.route('**/api/v1/projects/spatial-review/policy-jobs', route => {
    submitted.push(route.request().postDataJSON());
    return route.fulfill({ status: 422, json: { detail: 'Captured request; no worker started' } });
  });
  await page.goto('/');
  try {
    await page.getByRole('button', { name: 'Settings & diagnostics', exact: true }).click();
  await page.getByRole('button', { name: 'Workflow settings', exact: true }).click();
    await expect(page.getByLabel('Task suite')).toBeDisabled();
    await expect(page.getByLabel('Quantization recipe')).toBeDisabled();
    await page.getByRole('button', { name: 'Diagnostics', exact: true }).click();
    await expect(page.getByLabel('Diagnostic mode')).toBeDisabled();
    await expect(page.getByRole('heading', { name: 'Recorded benchmark comparison' })).toBeVisible();
    await page.getByRole('button', { name: 'Quantize', exact: true }).click();
    await expect(page.getByRole('region', { name: 'Your quantization models' }).getByRole('button', { name: 'Choose Synthetic policy · source', exact: true })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'New quantization', exact: true })).toHaveCount(0);
    await expect(page.getByRole('group', { name: 'My model', exact: true })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Run quantization workflow', exact: true })).toHaveCount(0);
    expect(await page.evaluate(() => localStorage.getItem('firebird.workflow.'))).toBeNull();
    expect(submitted).toEqual([]);
  } finally {
    releaseProjects();
  }
  await page.getByRole('button', { name: 'Settings & diagnostics', exact: true }).click();
  await page.getByRole('button', { name: 'Workflow settings', exact: true }).click();
  await page.getByLabel('Task suite').selectOption('libero_spatial');
  await expect(page.getByLabel('Spatial task IDs')).toHaveValue('0,1,2,3,4,5,6,7,8,9');
  await expect(page.getByLabel('Approved parity profile')).toHaveValue('');
  await expect(page.getByRole('combobox', { name: 'Protocol', exact: true })).toHaveValue('libero');
  await expect(page.getByRole('combobox', { name: 'Protocol', exact: true })).toBeDisabled();
  await expect(page.getByLabel('Episode step limit')).toHaveValue('280');
  await expect(page.getByLabel('Episode step limit')).toBeDisabled();
  await page.getByLabel('Spatial task IDs').fill('0,2');
  await page.getByLabel('Approved parity profile').fill('synthetic-test-only');
  await page.getByLabel('Maximum action RMSE').fill('0');
  await page.getByLabel('Maximum absolute action error').fill('0');
  await page.getByRole('button', { name: 'Quantize', exact: true }).click();
  await page.getByRole('button', { name: 'Choose Synthetic policy · source', exact: true }).click();
  await page.getByRole('group', { name: 'My model', exact: true }).locator('input[value="source"]').check();
  await page.getByRole('button', { name: 'Run quantization workflow', exact: true }).click();
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0].evaluation).toMatchObject({
    mode: 'libero', suite: 'libero_spatial', task_ids: [0, 2], steps: 280,
    parity_limits: { profile: 'synthetic-test-only', max_rmse: 0, max_abs_error: 0 },
  });
  // Old saved preferences must not revive the unsupported engine/short horizon.
  await page.evaluate(() => {
    const key = 'firebird.workflow.spatial-review';
    localStorage.setItem(key, JSON.stringify({ ...JSON.parse(localStorage.getItem(key) ?? '{}'), mode: 'engine', steps: 1 }));
  });
  await page.reload();
  await page.getByRole('button', { name: 'Settings & diagnostics', exact: true }).click();
  await page.getByRole('button', { name: 'Workflow settings', exact: true }).click();
  await expect(page.getByRole('combobox', { name: 'Protocol', exact: true })).toHaveValue('libero');
  await expect(page.getByLabel('Episode step limit')).toHaveValue('280');
  await page.getByRole('button', { name: 'Quantize', exact: true }).click();
  await page.getByRole('button', { name: 'Choose Synthetic policy · source', exact: true }).click();
  await page.getByRole('group', { name: 'My model', exact: true }).locator('input[value="source"]').check();
  await page.getByRole('button', { name: 'Run quantization workflow', exact: true }).click();
  await expect.poll(() => submitted.length).toBe(2);
  expect(submitted[1].evaluation).toEqual(submitted[0].evaluation);
});

// These tests inspect actual browser requests, with explicit fixture responses.
// They exercise preference ownership, not a worker or model-quality result.
async function workflowPreferenceFixture(page: Page) {
  const requests: { projectId: string; body: { evaluation: Record<string, unknown>; candidates: unknown[] } }[] = [];
  const projects = [
    { id: 'preferences-a', name: 'Project A', created_at: '2026-09-26T12:00:00Z' },
    { id: 'preferences-b', name: 'Project B', created_at: '2026-09-26T12:00:00Z' },
  ];
  await page.route('**/api/v1/projects/*/jobs', route => route.fulfill({ json: [] }));
  await page.route('**/api/v1/projects/*/artifacts', route => route.fulfill({ json: [savedWorkflowModel(new URL(route.request().url()).pathname.split('/').at(-2)!)] }));
  await page.route('**/api/v1/policy-options', route => route.fulfill({ json: {
    runtimes: [{ id: 'fixture', label: 'Synthetic target', device: 'cpu', training: false, simulation: true }],
    sources: [{ id: 'source', label: 'Synthetic policy', task: 'fixture' }],
    training_methods: [], default_training_method: 'lora',
    quantization_defaults: { cpu: { language: 'Q8_0', vision: null }, cuda: { language: 'Q8_0', vision: null }, note: 'Fixture only' },
  } }));
  await page.route('**/api/v1/projects/*/policy-jobs', route => {
    requests.push({ projectId: route.request().url().split('/').at(-2)!, body: route.request().postDataJSON() });
    return route.fulfill({ status: 422, json: { detail: 'Captured request; no worker started' } });
  });
  async function submit() {
    const count = requests.length;
    await page.getByRole('button', { name: 'Quantize', exact: true }).click();
    await page.getByRole('button', { name: 'Choose Synthetic policy · source', exact: true }).click();
    await page.getByRole('group', { name: 'My model', exact: true }).locator('input[value="source"]').check();
    await page.getByRole('button', { name: 'Run quantization workflow', exact: true }).click();
    await expect.poll(() => requests.length).toBe(count + 1);
    return requests.at(-1)!;
  }
  return { projects, requests, submit };
}

test('restored workflow preferences and submitted requests stay isolated when switching projects', async ({ page }) => {
  const fixture = await workflowPreferenceFixture(page);
  await page.route('**/api/v1/projects', route => route.fulfill({ json: fixture.projects }));
  await page.addInitScript(() => {
    // Seed once so reload verifies the user's changes, rather than resetting them.
    if (localStorage.getItem('firebird.project')) return;
    localStorage.setItem('firebird.project', 'preferences-a');
    localStorage.setItem('firebird.workflow.preferences-a', JSON.stringify({
      suite: 'libero_spatial', mode: 'engine', steps: 1, taskIds: '0,2',
      parityProfile: 'synthetic-project-a', parityRmse: '0', parityMaxError: '0', repetitions: 4,
    }));
    localStorage.setItem('firebird.workflow.preferences-b', JSON.stringify({
      suite: 'libero_object', mode: 'engine', steps: 500, repetitions: 6, precision: 'Q4_0',
    }));
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Settings & diagnostics', exact: true }).click();
  await page.getByRole('button', { name: 'Workflow settings', exact: true }).click();
  await expect(page.getByLabel('Spatial task IDs')).toHaveValue('0,2');
  await expect(page.getByLabel('Episode step limit')).toHaveValue('280');
  await page.getByLabel('Spatial task IDs').fill('1,3');
  await page.getByRole('combobox', { name: 'Current project', exact: true }).selectOption('preferences-b');
  await expect(page.getByLabel('Task suite')).toHaveValue('libero_object');
  await expect(page.getByLabel('Timed predictions')).toHaveValue('6');
  await page.getByLabel('Timed predictions').fill('9');
  const projectB = await fixture.submit();
  expect(projectB.projectId).toBe('preferences-b');
  expect(projectB.body.evaluation).toMatchObject({ mode: 'engine', suite: 'libero_object', steps: 500, repetitions: 9 });
  expect(projectB.body.evaluation).not.toHaveProperty('parity_limits');
  expect(projectB.body.candidates).toEqual([{ language: 'Q4_0', vision: null }]);
  await page.getByRole('combobox', { name: 'Current project', exact: true }).selectOption('preferences-a');
  const projectA = await fixture.submit();
  expect(projectA.projectId).toBe('preferences-a');
  expect(projectA.body.evaluation).toMatchObject({
    mode: 'libero', suite: 'libero_spatial', steps: 280, repetitions: 4, task_ids: [1, 3],
    parity_limits: { profile: 'synthetic-project-a', max_rmse: 0, max_abs_error: 0 },
  });
  expect(projectA.body.candidates).toEqual([{ language: 'Q8_0', vision: null }]);
  await page.reload();
  expect(await fixture.submit()).toEqual(projectA);
  const stored = await page.evaluate(() => ({
    empty: localStorage.getItem('firebird.workflow.'),
    a: JSON.parse(localStorage.getItem('firebird.workflow.preferences-a')!),
    b: JSON.parse(localStorage.getItem('firebird.workflow.preferences-b')!),
  }));
  expect(stored.empty).toBeNull();
  expect(stored.a.taskIds).toBe('1,3');
  expect(stored.b.repetitions).toBe(9);
});

for (const state of ['empty', 'error'] as const) {
  test(`workflow controls stay disabled with ${state} projects and recover after loading succeeds`, async ({ page }) => {
    const fixture = await workflowPreferenceFixture(page);
    let recovered = false;
    await page.route('**/api/v1/projects', route => route.fulfill(recovered
      ? { json: fixture.projects }
      : state === 'empty' ? { json: [] } : { status: 503, json: { detail: 'Project fixture unavailable' } }));
    await page.goto('/');
    await page.getByRole('button', { name: 'Settings & diagnostics', exact: true }).click();
  await page.getByRole('button', { name: 'Workflow settings', exact: true }).click();
    await expect(page.getByLabel('Task suite')).toBeDisabled();
    await page.getByRole('button', { name: 'Diagnostics', exact: true }).click();
    await expect(page.getByLabel('Diagnostic mode')).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Start diagnostics', exact: true })).toBeDisabled();
    await page.getByLabel('Reference hardware').selectOption('rtx3070');
    await expect(page.getByRole('table', { name: 'NVIDIA RTX 3070 · recorded reference results' })).toBeVisible();
    await page.getByRole('button', { name: 'Quantize', exact: true }).click();
    await expect(page.getByRole('region', { name: 'Your quantization models' }).getByRole('button', { name: 'Choose Synthetic policy · source', exact: true })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'New quantization', exact: true })).toHaveCount(0);
    await expect(page.getByRole('group', { name: 'My model', exact: true })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Run quantization workflow', exact: true })).toHaveCount(0);
    expect(fixture.requests).toEqual([]);
    expect(await page.evaluate(() => localStorage.getItem('firebird.workflow.'))).toBeNull();
    recovered = true;
    if (state === 'error') await page.getByRole('button', { name: 'Retry projects', exact: true }).click();
    else await page.reload();
    const submitted = await fixture.submit();
    expect(submitted.projectId).toBe('preferences-a');
    expect(submitted.body.evaluation).toMatchObject({ mode: 'engine', suite: 'libero_object', steps: 500 });
  });
}

for (const storage of ['invalid JSON', 'unavailable'] as const) {
  test(`workflow settings remain usable with ${storage} browser storage`, async ({ page }) => {
    const fixture = await workflowPreferenceFixture(page);
    await page.route('**/api/v1/projects', route => route.fulfill({ json: fixture.projects }));
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.addInitScript(mode => {
      if (mode === 'invalid JSON') localStorage.setItem('firebird.workflow.preferences-a', '{');
      else {
        Storage.prototype.getItem = () => { throw new DOMException('Storage unavailable', 'SecurityError'); };
        Storage.prototype.setItem = () => { throw new DOMException('Storage unavailable', 'SecurityError'); };
      }
    }, storage);
    await page.goto('/');
    await page.getByRole('button', { name: 'Settings & diagnostics', exact: true }).click();
  await page.getByRole('button', { name: 'Workflow settings', exact: true }).click();
    await expect(page.getByLabel('Quantization recipe')).toHaveValue('recommended');
    await page.getByLabel('Quantization recipe').selectOption('Q4_0');
    await expect(page.getByLabel('Quantization recipe')).toHaveValue('Q4_0');
    // Unavailable storage supports the current mount; persistence is not claimed.
    expect(errors).toEqual([]);
  });
}

test('a project removed during refetch cannot submit with its stale selection', async ({ page }) => {
  await page.clock.install();
  const fixture = await workflowPreferenceFixture(page);
  let removed = false;
  let emptyResponses = 0;
  await page.route('**/api/v1/projects', route => {
    if (removed) emptyResponses += 1;
    return route.fulfill({ json: removed ? [] : fixture.projects });
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Quantize', exact: true }).click();
  await page.getByRole('button', { name: 'Choose Synthetic policy · source', exact: true }).click();
  await page.getByRole('group', { name: 'My model', exact: true }).locator('input[value="source"]').check();
  await expect(page.getByRole('button', { name: 'Run quantization workflow', exact: true })).toBeEnabled();
  removed = true;
  // Advance beyond the configured 5s freshness period without a wall-clock sleep.
  await page.clock.fastForward(6_000);
  // React Query refetches stale project data when the browser regains visibility.
  await page.evaluate(() => window.dispatchEvent(new Event('visibilitychange')));
  await expect.poll(() => emptyResponses).toBeGreaterThan(0);
  await expect(page.getByRole('region', { name: 'Your quantization models' }).getByRole('button', { name: 'Choose Synthetic policy · source', exact: true })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'New quantization', exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Run quantization workflow', exact: true })).toHaveCount(0);
  await expect(page.getByRole('group', { name: 'My model', exact: true })).toHaveCount(0);
  expect(fixture.requests).toEqual([]);
  expect(await page.evaluate(() => localStorage.getItem('firebird.workflow.'))).toBeNull();
});
