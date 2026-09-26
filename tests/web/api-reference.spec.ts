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
  await page.getByRole('link', { name: 'Firebird workspace home' }).click();
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
  let expected = 0;
  for (const [path, item] of Object.entries(schema.paths)) {
    for (const method of Object.keys(item as object).filter(method => methods.has(method))) {
      expected += 1;
      await expect(endpoint(page, method, path)).toBeVisible();
    }
  }
  await expect(page.locator('details[id^="endpoint-"]')).toHaveCount(expected);
  for (const name of Object.keys(schema.components.schemas)) {
    await expect(page.locator(`[id=${JSON.stringify(`schema-${name}`)}]`)).toHaveCount(1);
  }
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
  await expect(intake.locator('pre').first()).toContainText('http://127.0.0.1:8765/api/v1/projects/{project_id}/intakes');
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
  await expect(page.getByRole('link', { name: 'Firebird workspace home' })).toBeVisible();
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
  await expect(page.getByRole('heading', { name: 'Fine-tune a policy' })).toBeVisible();
  const method = page.getByLabel('Fine-tuning method');
  await expect(method).toHaveValue('lora');
  await expect(method).toBeDisabled();
  await expect(method.locator('option[value=qlora]')).toHaveCount(1);
  await expect(page.getByRole('button', { name: 'Start fine-tuning' })).toBeDisabled();
  await page.getByRole('button', { name: 'Settings & diagnostics', exact: true }).click();
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

test('dataset inspection stays separate from newer policy jobs in the same project', async ({ page }) => {
  const timestamp = '2026-09-26T12:00:00Z';
  const datasetJob = {
    id: 'dataset-review', project_id: 'mixed-review', kind: 'dataset.inspect', status: 'succeeded',
    request: { source: 'huggingface', repo_id: 'fixture/robot', revision: 'main' },
    created_at: timestamp, updated_at: timestamp,
    result: {
      source: 'huggingface', repo_id: 'fixture/robot', revision: 'a'.repeat(40), format: 'lerobot_v3',
      inspection_scope: 'metadata_only', total_episodes: 1, total_frames: 6, fps: 5,
      features: { action: { dtype: 'float32', shape: [2] }, 'observation.state': { dtype: 'float32', shape: [2] } },
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
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await page.getByRole('button', { name: /^Inspection/ }).click();
  await expect(page.getByRole('heading', { name: 'fixture/robot', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Load visual preview', exact: true })).toBeVisible();
  await expect(page.getByText('policy-review', { exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Fine-tune', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Fine-tune a policy' })).toBeVisible();
  expect(errors).toEqual([]);
});

test('quantization submits Q4 only after an explicit experimental choice', async ({ page }) => {
  const requests: { candidates: { language: string; vision: string | null }[] }[] = [];
  const timestamp = '2026-09-26T12:00:00Z';
  await page.route('**/api/v1/projects', route => route.fulfill({ json: [{ id: 'precision-review', name: 'Precision fixture', created_at: timestamp }] }));
  await page.route('**/api/v1/projects/precision-review/jobs', route => route.fulfill({ json: [] }));
  await page.route('**/api/v1/projects/precision-review/artifacts', route => route.fulfill({ json: [] }));
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
    await page.getByLabel('Input policy').selectOption('source:source');
    await page.getByRole('button', { name: 'Run quantization workflow', exact: true }).click();
    await expect.poll(() => requests.length).toBe(count);
  }
  await page.goto('/');
  await submitQuantization(1);
  expect(requests[0].candidates).toEqual([{ language: 'Q8_0', vision: null }]);
  await page.getByRole('button', { name: 'Settings & diagnostics', exact: true }).click();
  const compare = page.getByLabel('Compare Q8 and Q4 (experimental)');
  await expect(compare).not.toBeChecked();
  await compare.check();
  await submitQuantization(2);
  expect(requests[1].candidates).toEqual([{ language: 'Q8_0', vision: null }, { language: 'Q4_0', vision: null }]);
  await page.getByRole('button', { name: 'Settings & diagnostics', exact: true }).click();
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
  await page.route('**/api/v1/projects/spatial-review/artifacts', route => route.fulfill({ json: [] }));
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
    await expect(page.getByLabel('Task suite')).toBeDisabled();
    await expect(page.getByLabel('Quantization recipe')).toBeDisabled();
    await page.getByRole('button', { name: 'Diagnostics', exact: true }).click();
    await expect(page.getByLabel('Diagnostic mode')).toBeDisabled();
    await expect(page.getByRole('heading', { name: 'Recorded benchmark comparison' })).toBeVisible();
    await page.getByRole('button', { name: 'Quantize', exact: true }).click();
    await expect(page.getByLabel('Input policy')).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Run quantization workflow', exact: true })).toBeDisabled();
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
  await page.getByLabel('Input policy').selectOption('source:source');
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
  await expect(page.getByRole('combobox', { name: 'Protocol', exact: true })).toHaveValue('libero');
  await expect(page.getByLabel('Episode step limit')).toHaveValue('280');
  await page.getByRole('button', { name: 'Quantize', exact: true }).click();
  await page.getByLabel('Input policy').selectOption('source:source');
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
  await page.route('**/api/v1/projects/*/artifacts', route => route.fulfill({ json: [] }));
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
    await page.getByLabel('Input policy').selectOption('source:source');
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
    await expect(page.getByLabel('Task suite')).toBeDisabled();
    await page.getByRole('button', { name: 'Diagnostics', exact: true }).click();
    await expect(page.getByLabel('Diagnostic mode')).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Start diagnostics', exact: true })).toBeDisabled();
    await page.getByLabel('Reference hardware').selectOption('rtx3070');
    await expect(page.getByRole('table', { name: 'NVIDIA RTX 3070 · recorded reference results' })).toBeVisible();
    await page.getByRole('button', { name: 'Quantize', exact: true }).click();
    await expect(page.getByLabel('Input policy')).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Run quantization workflow', exact: true })).toBeDisabled();
    // Programmatic submission must also respect readiness, not just disabled UI.
    await page.getByLabel('Input policy').evaluate(element => element.closest('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })));
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
  await page.getByLabel('Input policy').selectOption('source:source');
  await expect(page.getByRole('button', { name: 'Run quantization workflow', exact: true })).toBeEnabled();
  removed = true;
  // Advance beyond the configured 5s freshness period without a wall-clock sleep.
  await page.clock.fastForward(6_000);
  // React Query refetches stale project data when the browser regains visibility.
  await page.evaluate(() => window.dispatchEvent(new Event('visibilitychange')));
  await expect.poll(() => emptyResponses).toBeGreaterThan(0);
  await expect(page.getByRole('button', { name: 'Run quantization workflow', exact: true })).toBeDisabled();
  await expect(page.getByLabel('Input policy')).toBeDisabled();
  expect(fixture.requests).toEqual([]);
  expect(await page.evaluate(() => localStorage.getItem('firebird.workflow.'))).toBeNull();
});
