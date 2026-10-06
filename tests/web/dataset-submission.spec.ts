import { expect, test, type Page } from '@playwright/test';

const project = 'intake-recovery-fixture';
const timestamp = '2026-09-27T12:00:00Z';
async function fixture(page: Page, mode: 'normal' | 'lost-ack' | 'lost-before-commit' | 'unsupported' | 'wrong-ack' | 'delayed-ack' = 'normal') {
  let releaseAck!: () => void;
  const ackReady = new Promise<void>(resolve => { releaseAck = resolve; });
  const writes: { key: string; body: Record<string, unknown> }[] = [];
  const reads: string[] = [];
  const bindings = new Map<string, Record<string, unknown>>();
  const jobs: Record<string, unknown>[] = [];
  const unexpected: string[] = [];
  await page.route('**/api/v1/**', async route => {
    const request = route.request(), path = new URL(request.url()).pathname;
    const prefix = `/api/v1/projects/${project}/submissions/`;
    if (request.method() === 'GET' && path.startsWith(prefix)) {
      const key = decodeURIComponent(path.slice(prefix.length));
      reads.push(key);
      await route.fulfill({ status: bindings.has(key) ? 200 : 404,
        headers: mode === 'unsupported' ? {} : { 'Idempotency-Key': key, 'Cache-Control': 'no-store' },
        json: bindings.get(key) ?? { detail: 'No accepted submission found for this project, operation and key' } });
      return;
    }
    if (request.method() === 'POST' && path === `/api/v1/projects/${project}/intakes`) {
      const key = request.headers()['idempotency-key'], body = request.postDataJSON();
      writes.push({ key, body });
      if (mode === 'lost-before-commit' && writes.length === 1) { await route.abort('connectionfailed'); return; }
      const accepted = { id: `inspection-${writes.length}`, project_id: project, kind: 'dataset.inspect', status: 'queued',
        created_at: timestamp, updated_at: timestamp, request: body, result: null, stage: null, error: null };
      jobs.push(accepted); bindings.set(key, accepted);
      if (mode === 'lost-ack' && writes.length === 1) { await route.abort('connectionfailed'); return; }
      if (mode === 'delayed-ack') await ackReady;
      await route.fulfill({ status: 202, headers: { 'Idempotency-Key': key, 'Cache-Control': 'no-store' },
        json: mode === 'wrong-ack' ? { ...accepted, request: { ...body, repo_id: 'foreign/robot' } } : accepted });
      return;
    }
    if (request.method() === 'GET') {
      const responses: Record<string, unknown> = {
        '/api/v1/health': { status: 'ok', version: 'generated-browser-fixture' },
        '/api/v1/datasets': [],
        '/api/v1/capabilities': [{ operation: 'dataset.inspect.local', status: 'available', reason: null }],
        '/api/v1/projects': [{ id: project, name: 'Intake recovery', created_at: timestamp }],
        [`/api/v1/projects/${project}/jobs`]: jobs,
      };
      if (path in responses) { await route.fulfill({ json: responses[path] }); return; }
    }
    unexpected.push(`${request.method()} ${path}`);
    await route.fulfill({ status: 405, json: { detail: 'Blocked by generated intake fixture.' } });
  });
  await page.goto('/');
  await expect(page.getByLabel('Dataset repository')).toBeEnabled();
  await page.getByLabel('Dataset repository').fill('fixture/robot');
  await page.getByText('Revision (optional)', {exact:true}).click();
  await page.getByLabel('Revision', {exact:true}).fill('a'.repeat(40));
  return { writes, reads, jobs, unexpected, releaseAck };
}

test('intake checks durable support and submits one saved immutable source', async ({ page }) => {
  const state = await fixture(page);
  const source = await page.getByLabel('Dataset repository').inputValue();
  await page.getByRole('button', { name: 'Inspect dataset', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Waiting to inspect' })).toBeVisible();
  expect(state.writes).toHaveLength(1);
  expect(state.writes[0].key).toMatch(/^[a-f0-9-]{36}$/);
  expect(state.reads).toEqual([state.writes[0].key]);
  expect(state.writes[0].body.repo_id).toBe(source);
  expect(state.unexpected).toEqual([]);
});

test('lost intake response survives reload and resolves by key without a second POST', async ({ page }) => {
  const state = await fixture(page, 'lost-ack');
  await page.getByRole('button', { name: 'Inspect dataset', exact: true }).click();
  const recovery = page.getByRole('complementary', { name: 'Request recovery' });
  await expect(recovery.getByRole('button', { name: 'Check saved request' })).toBeEnabled();
  await page.reload();
  await expect(page.getByRole('button', { name: 'Inspect dataset', exact: true })).toBeDisabled();
  await recovery.getByRole('button', { name: 'Check saved request' }).click();
  await expect(page.getByRole('heading', { name: 'Waiting to inspect' })).toBeVisible();
  expect(state.writes).toHaveLength(1);
  expect(state.reads).toEqual([state.writes[0].key, state.writes[0].key]);
});

test('missing intake remains blocked until explicit retry of its original key and unchanged source', async ({ page }) => {
  const state = await fixture(page, 'lost-before-commit');
  await page.getByRole('button', { name: 'Inspect dataset', exact: true }).click();
  const recovery = page.getByRole('complementary', { name: 'Request recovery' });
  await expect(recovery.getByRole('button', { name: 'Check saved request' })).toBeEnabled();
  await page.getByLabel('Dataset repository').fill('edited/source');
  await recovery.getByRole('button', { name: 'Check saved request' }).click();
  await expect(page.getByRole('button', { name: 'Inspect dataset', exact: true })).toBeDisabled();
  await expect(recovery.getByRole('button', { name: 'Retry same request' })).toBeEnabled();
  expect(state.writes).toHaveLength(1);
  await recovery.getByRole('button', { name: 'Retry same request' }).click();
  await expect(page.getByRole('heading', { name: 'Waiting to inspect' })).toBeVisible();
  expect(state.writes).toHaveLength(2);
  expect(state.writes[1]).toEqual(state.writes[0]);
});

test('older intake backend receives no POST and cannot clear an unresolved request', async ({ page }) => {
  const state = await fixture(page, 'unsupported');
  await page.getByRole('button', { name: 'Inspect dataset', exact: true }).click();
  await expect(page.getByRole('complementary', { name: 'Request recovery' })).toContainText('did not verify durable submission support');
  await expect(page.getByRole('button', { name: 'Inspect dataset', exact: true })).toBeDisabled();
  expect(state.writes).toEqual([]);
});

test('wrong source acknowledgement cannot select another dataset and is resolved only by original scoped lookup', async ({ page }) => {
  const state = await fixture(page, 'wrong-ack');
  await page.getByRole('button', { name: 'Inspect dataset', exact: true }).click();
  const recovery = page.getByRole('complementary', { name: 'Request recovery' });
  await expect(recovery.getByRole('button', { name: 'Check saved request' })).toBeEnabled();
  await recovery.getByRole('button', { name: 'Check saved request' }).click();
  await expect(page.getByRole('heading', { name: 'Waiting to inspect' })).toBeVisible();
  expect(state.writes).toHaveLength(1);
});

test('denied recovery storage prevents any intake submission', async ({ page }) => {
  await page.addInitScript(() => {
    const original = Storage.prototype.setItem;
    Storage.prototype.setItem = function(key: string, value: string) {
      if (key.startsWith('firebird:submission:')) throw new DOMException('Fixture denied storage', 'QuotaExceededError');
      return original.call(this, key, value);
    };
  });
  const state = await fixture(page);
  await page.getByRole('button', { name: 'Inspect dataset', exact: true }).click();
  await expect(page.getByText(/Submission recovery storage is unavailable/)).toBeVisible();
  expect(state.reads).toEqual([]);
  expect(state.writes).toEqual([]);
});

test('intake acknowledgement allows host resolution but rejects changed immutable revision, scope and snapshot source', async () => {
  const { intakeAcknowledgement, augmentationAcknowledgement } = await import('../../apps/web/src/lib/dataset-submission');
  const base = { id: 'ack', project_id: project, kind: 'dataset.inspect', status: 'succeeded', created_at: timestamp, updated_at: timestamp };
  const body = { source: 'huggingface', repo_id: 'fixture/robot', revision: 'a'.repeat(40) };
  const value = { ...base, request: { ...body, path: null, snapshot_for_training: false } };
  expect(intakeAcknowledgement(value, project, body).id).toBe('ack');
  expect(() => intakeAcknowledgement({ ...value, request: { ...value.request, revision: 'b'.repeat(40) } }, project, body)).toThrow();
  expect(() => intakeAcknowledgement({ ...value, project_id: 'other' }, project, body)).toThrow();
  expect(() => intakeAcknowledgement({ ...value, kind: 'dataset.augment' }, project, body)).toThrow();
  const local = { source: 'local', path: 'generated', revision: 'main', snapshot_for_training: false };
  expect(intakeAcknowledgement({ ...base, request: { ...local, path: '/allowed/generated', revision: `metadata-sha256:${'c'.repeat(64)}` } }, project, local).id).toBe('ack');
  expect(() => intakeAcknowledgement({ ...base, request: { ...local, snapshot_for_training: true, path: 'another' } }, project, { ...local, snapshot_for_training: true })).toThrow();
  const augmentation = { operation: 'dataset.augment', source_job_id: 'source', episode_indices: [0, 1], camera_key: 'camera', preset: 'lighting', prompt: 'Saved prompt', start_seconds: 0, duration_seconds: 5 };
  const augmented = { ...base, kind: 'dataset.augment', request: augmentation };
  expect(augmentationAcknowledgement(augmented, project, augmentation).id).toBe('ack');
  expect(() => augmentationAcknowledgement({ ...augmented, request: { ...augmentation, episode_indices: [1, 0] } }, project, augmentation)).toThrow();
  expect(() => augmentationAcknowledgement({ ...augmented, request: { ...augmentation, prompt: 'Other prompt' } }, project, augmentation)).toThrow();
});


test('late intake acknowledgement records the job without replacing later stage navigation', async ({ page }) => {
  const state = await fixture(page, 'delayed-ack');
  await page.getByRole('button', { name: 'Inspect dataset', exact: true }).click();
  await expect.poll(() => state.writes.length).toBe(1);
  await page.getByRole('button', { name: 'Fine-tune', exact: true }).click();
  state.releaseAck();
  await expect.poll(() => page.evaluate(() => Object.keys(sessionStorage).filter(key => key.startsWith('firebird:submission:')).length)).toBe(0);
  await page.getByRole('button', { name: 'Dataset', exact: true }).click();
  await expect(page.getByLabel('Dataset repository')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Waiting to inspect' })).not.toBeVisible();
  expect(state.writes).toHaveLength(1);
});

for (const failure of ['unavailable', 'foreign'] as const) test(`legacy intake recovery withdraws prior approval after ${failure} history reread`, async ({ page }) => {
  const state = await fixture(page);
  await page.evaluate(projectId => sessionStorage.setItem(`firebird:job-attempt:dataset.inspect:${projectId}`, JSON.stringify({ state: 'uncertain', message: 'Legacy inspection outcome is unverified.' })), project);
  await page.reload();
  const recovery = page.getByRole('complementary', { name: 'Request recovery' });
  const refresh = recovery.getByRole('button', { name: 'Refresh job history', exact: true });
  const acknowledge = recovery.getByRole('button', { name: 'I reviewed the jobs; allow a new request', exact: true });
  await expect(acknowledge).toBeDisabled();
  await refresh.click();
  await expect(acknowledge).toBeEnabled();
  const historyPath = `**/api/v1/projects/${project}/jobs`;
  await page.route(historyPath, route => route.fulfill({
    status: failure === 'unavailable' ? 503 : 200,
    json: failure === 'unavailable' ? { detail: 'Generated history reread unavailable.' } : [{ id: 'foreign-inspection', project_id: 'another-project', kind: 'dataset.inspect', status: 'succeeded', created_at: timestamp, updated_at: timestamp, request: { source: 'huggingface', repo_id: 'foreign/dataset', revision: 'main' } }],
  }));
  await refresh.click();
  await expect(recovery.getByRole('alert')).toBeVisible();
  await expect(acknowledge).toBeDisabled();
  expect(state.writes).toEqual([]);
  await page.unroute(historyPath);
  await refresh.click();
  await expect(acknowledge).toBeEnabled();
  await acknowledge.click();
  await expect(recovery).toHaveCount(0);
  expect(state.writes).toEqual([]);
});
