import { expect, type Page, type Route } from '@playwright/test';

/** Open a category's new task through its visible primary action. */
export async function openModuleTask(page: Page) {
  const start = page.getByRole('button', {
    name: /^(New augmentation|New scene|New scoring|New teleoperation|New run)$/,
  });
  const back = page.getByRole('button', { name: 'Back to jobs', exact: true });
  await expect.poll(async () => (await start.isVisible()) || (await back.isVisible())).toBe(true);
  if (await start.isVisible()) await start.click();
}

export const jobOperationFixtures = [
  ['dataset.inspect', 'Inspect dataset', 'datasets'],
  ['dataset.import', 'Import dataset', 'datasets'],
  ['dataset.convert', 'Convert dataset', 'datasets'],
  ['dataset.augment', 'Augment dataset', 'augmentation'],
  ['scene.generate', 'Real to sim', 'real-to-sim'],
  ['policy.finetune', 'Fine-tune', 'training'],
  ['policy.distill', 'Distill', 'distillation'],
  ['policy.quantize', 'Quantize', 'quantization'],
  ['policy.evaluate', 'Evaluate', 'evaluation'],
  ['policy.run', 'Run policy', 'simulation'],
  ['policy.import', 'Import model', 'models'],
  ['policy.export', 'Export model', 'models'],
  ['policy.workflow', 'Policy workflow', 'evaluation'],
  ['teaching.capture', 'Record teaching', 'teaching'],
  ['agent.control', 'AI teleoperation', 'teaching'],
  ['decision.score', 'Decision scoring', 'decision-lab'],
].map(([id, title, module]) => ({ id, title, module, path: '/', submission: 'job' }));

/** Job navigation reads never start a worker in specialized browser fixtures. */
export async function fulfillWorkJobRead(route: Route): Promise<boolean> {
  const request = route.request();
  if (request.method() !== 'GET') return false;
  const path = new URL(request.url()).pathname;
  if (path === '/api/v1/job-operations') {
    await route.fulfill({ json: jobOperationFixtures });
    return true;
  }
  if (path.endsWith('/work-jobs')) {
    await route.fulfill({ json: [] });
    return true;
  }
  if (path === '/api/v1/compute-targets') {
    await route.fulfill({ json: { schema_version: 1, targets: [], runtime_options: [], connection_refs: [] } });
    return true;
  }
  return false;
}

/** Shared job-history reads for fixture-owned core jobs, without creating work. */
export async function fulfillCoreWorkJobRead(route: Route, records: Record<string, any>[]): Promise<boolean> {
  if (route.request().method() !== 'GET') return false;
  const path = decodeURIComponent(new URL(route.request().url()).pathname);
  const match = path.match(/\/projects\/([^/]+)\/work-jobs(?:\/([^/]+))?(?:\/(.*))?$/);
  if (!match) return false;
  const operations: Record<string, string> = {
    'policy.finetune': 'training',
    'policy.distill': 'distillation',
    'policy.quantize': 'quantization',
    'policy.evaluate': 'evaluation',
    'policy.run': 'simulation',
    'scene.generate': 'real-to-sim',
  };
  const items = records
    .filter((item) => item.project_id === match[1] && operations[item.kind])
    .map((item) => ({
      id: 'core:' + item.id,
      source_id: item.id,
      project_id: item.project_id,
      operation: item.kind,
      module: operations[item.kind],
      title: item.id,
      status: item.status,
      source_status: item.status,
      created_at: item.created_at,
      updated_at: item.updated_at,
      stage: item.stage ?? null,
      message: item.error ?? null,
      execution: null,
      inputs: item.request,
      result: item.result ?? null,
      outputs: [],
      parent_ids: [],
      can_cancel: ['queued', 'running'].includes(item.status),
      events: [],
    }));
  if (match[3] === 'failure') {
    await route.fulfill({ json: null });
    return true;
  }
  if (match[2]) {
    const selected = items.find((item) => item.id === match[2]);
    await route.fulfill(
      selected ? { json: selected } : { status: 404, json: { detail: 'Job not found in this project' } },
    );
  } else await route.fulfill({ json: items });
  return true;
}
