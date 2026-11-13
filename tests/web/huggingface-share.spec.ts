import { expect, test, type Page } from '@playwright/test';
import { waitForJob } from './job-waiter';
import { selectProject } from './project-controls';

// The real application, with "Hugging Face" kept in a folder by tests/hub_stand_in.py
// (see tests/web/serve.py). Two projects stand for the two machines.
test.describe.configure({ timeout: 120_000 });

async function project(page: Page, name: string) {
  const response = await page.request.post('/api/v1/projects', { data: { name } });
  expect(response.status()).toBe(201);
  return (await response.json()).id as string;
}

/** Open a page in one project. Later project changes made through the page are kept. */
async function enter(page: Page, id: string, path: string) {
  await page.addInitScript((value) => {
    if (sessionStorage.getItem('sharing-test-project')) return;
    sessionStorage.setItem('sharing-test-project', value);
    localStorage.setItem('firebird.project', value);
  }, id);
  await page.goto(path);
  await expect(page.getByRole('button', { name: 'Current project', exact: true })).toHaveAttribute(
    'data-project-id',
    id,
  );
}

async function shareJobs(page: Page, id: string) {
  const jobs = await (await page.request.get(`/api/v1/projects/${id}/work-jobs`)).json();
  return (jobs as { id: string; operation: string }[]).filter((job) => job.id.startsWith('share:'));
}

test('a saved dataset is published privately and added from its link in another project', async ({
  page,
}, testInfo) => {
  const run = testInfo.project.name;
  const laptop = await project(page, `Laptop ${run}`);
  const macbook = await project(page, `MacBook ${run}`);
  expect((await page.request.post(`/api/v1/projects/${laptop}/datasets/example`)).status()).toBe(202);
  await enter(page, laptop, '/datasets/');
  await page.getByRole('button', { name: 'Open dataset Robot labeling playground' }).click();
  const labels = page.getByRole('region', { name: 'Dataset labeling' });
  await expect(labels.getByRole('link', { name: 'Export dataset' })).toBeVisible({ timeout: 60_000 });

  // One button. It shows where the dataset will go, and that it will be private, first.
  await page.getByRole('button', { name: 'Publish to Hugging Face', exact: true }).click();
  const publish = page.getByRole('region', { name: 'Publish to Hugging Face' });
  await expect(publish.getByText('Signed in as')).toContainText('browser-fixture');
  const destination = publish.getByRole('textbox', { name: 'Destination repository' });
  await expect(destination).toHaveValue('browser-fixture/robot-labeling-playground');
  await expect(publish.getByText('Private.', { exact: true })).toBeVisible();
  const repository = `browser-fixture/playground-${run}`;
  await destination.fill(repository);
  const link = `https://huggingface.co/datasets/${repository}`;
  await expect(publish.getByText(link, { exact: true })).toBeVisible();
  expect(await shareJobs(page, laptop)).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath('publish-dataset.png'), fullPage: true });

  await publish.getByRole('button', { name: 'Publish privately', exact: true }).click();
  await expect(publish.getByText('Published privately.', { exact: true })).toBeVisible({ timeout: 45_000 });
  await expect(publish.getByRole('textbox', { name: 'Link to copy' })).toHaveValue(link);
  await expect(publish.getByRole('button', { name: 'Copy link', exact: true })).toBeVisible();
  const [published] = await shareJobs(page, laptop);
  expect(published.operation).toBe('dataset.publish');
  const sent = await (await page.request.get(`/api/v1/projects/${laptop}/work-jobs/${published.id}`)).json();
  expect(sent.result.private).toBe(true);
  expect(sent.inputs).toMatchObject({ repo_id: repository, private: true });
  expect(JSON.stringify(sent)).not.toContain('hf_');
  // Closing and opening the panel again still shows the link.
  await publish.getByRole('button', { name: 'Close', exact: true }).click();
  await page.getByRole('button', { name: 'Publish to Hugging Face', exact: true }).click();
  await expect(publish.getByRole('textbox', { name: 'Link to copy' })).toHaveValue(link);

  // The other machine: paste the link.
  await selectProject(page, macbook);
  await page.goto('/datasets/');
  const add = page.getByRole('form', { name: 'Add from a Hugging Face link' });
  await add.getByRole('textbox', { name: 'Add from a Hugging Face link' }).fill(link);
  await add.getByRole('button', { name: 'Add dataset', exact: true }).click();
  await expect(add.getByText('Added Robot labeling playground.')).toBeVisible({ timeout: 60_000 });
  await page.screenshot({ path: testInfo.outputPath('added-dataset.png'), fullPage: true });
  const saved = await (await page.request.get(`/api/v1/datasets?project_id=${macbook}`)).json();
  expect(saved).toHaveLength(1);
  expect(saved[0]).toMatchObject({ status: 'ready', name: 'Robot labeling playground', project_id: macbook });
  expect(saved[0].detection.sha256).toBe(sent.result.sha256);
  await add.getByRole('button', { name: 'Open dataset', exact: true }).click();
  await expect(
    page.getByRole('region', { name: 'Dataset labeling' }).getByRole('heading', { name: 'Robot labeling playground' }),
  ).toBeVisible();
  // The job is in the project's history with the others, with its link.
  await page.goto('/jobs/');
  await page
    .getByRole('button', { name: /Add dataset from Hugging Face/ })
    .first()
    .click();
  await expect(page.getByRole('link', { name: 'Open on Hugging Face', exact: true })).toHaveAttribute('href', link);
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1),
  ).toBe(true);
});

test('a model is published only after an explicit privacy choice and added from its link', async ({
  page,
}, testInfo) => {
  const run = testInfo.project.name;
  const laptop = await project(page, `Model laptop ${run}`);
  const macbook = await project(page, `Model MacBook ${run}`);
  const accepted = await page.request.post(`/api/v1/projects/${laptop}/policy-jobs`, {
    data: { operation: 'policy.import', runtime_id: 'browser-success', source_id: 'synthetic-source' },
  });
  expect(accepted.status()).toBe(202);
  await waitForJob(page.request, (await accepted.json()).id, 'succeeded');
  const [original] = await (await page.request.get(`/api/v1/projects/${laptop}/artifacts`)).json();
  await enter(page, laptop, '/models/');
  await page.getByRole('button', { name: `Open model ${original.label} · ${original.id}`, exact: true }).click();
  await page.getByRole('button', { name: 'Publish to Hugging Face', exact: true }).click();
  const publish = page.getByRole('region', { name: 'Publish to Hugging Face' });
  const destination = publish.getByRole('textbox', { name: 'Destination repository' });
  await expect(destination).toHaveValue(/^browser-fixture\/.+/);

  // Public is never the default; choosing it changes the wording and the button.
  const makePublic = publish.getByRole('checkbox', { name: 'Make this repository public' });
  await expect(makePublic).not.toBeChecked();
  await expect(publish.getByRole('button', { name: 'Publish privately', exact: true })).toBeVisible();
  await makePublic.check();
  await expect(publish.getByText('Public.', { exact: true })).toBeVisible();
  await expect(publish.getByRole('button', { name: 'Publish publicly', exact: true })).toBeVisible();
  await makePublic.uncheck();
  await destination.fill('not a repository name');
  await expect(publish.getByRole('button', { name: 'Publish privately', exact: true })).toBeDisabled();
  const repository = `browser-fixture/policy-${run}`;
  await destination.fill(repository);
  await publish.getByRole('button', { name: 'Publish privately', exact: true }).click();
  await expect(publish.getByText('Published privately.', { exact: true })).toBeVisible({ timeout: 45_000 });
  const link = `https://huggingface.co/${repository}`;
  await expect(publish.getByRole('textbox', { name: 'Link to copy' })).toHaveValue(link);
  await page.screenshot({ path: testInfo.outputPath('published-model.png'), fullPage: true });

  await selectProject(page, macbook);
  await page.goto('/models/');
  const add = page.getByRole('form', { name: 'Add from a Hugging Face link' });
  const field = add.getByRole('textbox', { name: 'Add from a Hugging Face link' });
  // A link to nothing says so and adds nothing.
  await field.fill('browser-fixture/no-such-model');
  await add.getByRole('button', { name: 'Add model', exact: true }).click();
  await expect(add.getByRole('alert')).toContainText('was not found', { timeout: 30_000 });
  expect(await (await page.request.get(`/api/v1/projects/${macbook}/artifacts`)).json()).toEqual([]);
  // Only the owner/name is needed.
  await field.fill(repository);
  await add.getByRole('button', { name: 'Add model', exact: true }).click();
  await expect(add.getByText(`Added ${original.label}.`)).toBeVisible({ timeout: 45_000 });
  const [copy] = await (await page.request.get(`/api/v1/projects/${macbook}/artifacts`)).json();
  expect(copy).toMatchObject({
    project_id: macbook,
    label: original.label,
    format: original.format,
    manifest_sha256: original.manifest_sha256,
    file_bytes: original.file_bytes,
    metadata: original.metadata,
  });
  await add.getByRole('button', { name: 'Open model', exact: true }).click();
  const details = page.getByRole('article', { name: 'Model details' });
  await expect(details).toHaveAttribute('data-model-id', copy.id);
  await expect(details.getByText('Added from Hugging Face', { exact: false }).first()).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1),
  ).toBe(true);
});

test('without a connected account the page says how to connect and offers nothing to send', async ({
  page,
}, testInfo) => {
  const laptop = await project(page, `No account ${testInfo.project.name}`);
  const accepted = await page.request.post(`/api/v1/projects/${laptop}/policy-jobs`, {
    data: { operation: 'policy.import', runtime_id: 'browser-success', source_id: 'synthetic-source' },
  });
  await waitForJob(page.request, (await accepted.json()).id, 'succeeded');
  const [original] = await (await page.request.get(`/api/v1/projects/${laptop}/artifacts`)).json();
  // This computer has neither a saved token nor an hf login.
  await page.route('**/huggingface-sharing/destination**', (route) =>
    route.fulfill({
      json: {
        kind: 'model',
        id: original.id,
        name: original.label,
        account: {
          connected: false,
          username: null,
          source: null,
          message:
            'Connect a Hugging Face account first: save a token in Settings, or run hf auth login on this computer.',
        },
        repo_id: null,
        url: null,
        private: true,
        active_job: null,
        published: null,
      },
    }),
  );
  await enter(page, laptop, '/models/');
  await page.getByRole('button', { name: `Open model ${original.label} · ${original.id}`, exact: true }).click();
  await page.getByRole('button', { name: 'Publish to Hugging Face', exact: true }).click();
  const publish = page.getByRole('region', { name: 'Publish to Hugging Face' });
  await expect(publish.getByRole('alert')).toContainText('run hf auth login on this computer');
  await expect(publish.getByRole('button', { name: /^Publish (privately|publicly)$/ })).toHaveCount(0);
  expect(await shareJobs(page, laptop)).toEqual([]);
});
