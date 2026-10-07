import { chooseTransformationModel, openTransformationJob, transformationJobs } from './lifecycle-controls';
import { expect, test, type Page } from '@playwright/test';
const time = '2026-09-30T12:00:00Z', hash = 'a'.repeat(64);
function artifact(id: string, project = 'alpha', extra: Record<string, unknown> = {}) {
  return { id, project_id: project, job_id: `run-${id}`, label: `Saved ${id}`, format: 'native_checkpoint', path: 'owned', manifest_sha256: hash, file_bytes: 4096, parent_ids: [], metadata: { architecture: 'act' }, ...extra };
}
function job(id: string, kind: string, request: Record<string, unknown>) { return { id, project_id: 'alpha', kind, status: 'succeeded', created_at: time, updated_at: time, request, result: null }; }
async function fixture(page: Page, models = [artifact('teacher'), artifact('student', 'alpha', { parent_ids: ['teacher'] }), artifact('other', 'beta')]) {
  const state = { jobs: [] as Record<string,any>[], models, fail: false, mutations: [] as unknown[] };
  const jobs = [job('dataset', 'dataset.inspect', { source: 'huggingface', repo_id: 'our/pickup' }), job('run-teacher', 'policy.finetune', { operation: 'policy.finetune', runtime_id: 'gcp-l4', dataset_job_id: 'dataset', training: { model_id: 'code://lerobot/act' } }), job('run-student', 'policy.distill', { operation: 'policy.distill', runtime_id: 'cpu-student', artifact_id: 'teacher', dataset_job_id: 'dataset', native_distillation: { adapter: 'act-act-v1', splits: { train: [0, 1], validation: [2], final: [3] } } })];
  state.jobs=jobs;
  Object.assign(jobs[0], { result: { source: 'huggingface', repo_id: 'our/pickup', revision: 'b'.repeat(40), metadata_sha256: hash, inspection_scope: 'metadata_only', format: 'lerobot_v3', total_episodes: 6, total_frames: 24, fps: 30, robot_type: 'synthetic_fixture', inspected_at: time, warnings: [], features: {} } });
  await page.route('**/api/v1/**', async route => {
    const request = route.request(), path = new URL(request.url()).pathname;
    if (request.method() !== 'GET') { state.mutations.push(request.postDataJSON()); return route.fulfill({ status: 405, json: { detail: 'No jobs during navigation.' } }); }
    if (path === '/api/v1/projects') return route.fulfill({ json: [{ id: 'alpha', name: 'Pickup robot', created_at: time }, { id: 'beta', name: 'Second robot', created_at: time }] });
    if (path.endsWith('/artifacts')) return route.fulfill(state.fail ? { status: 503, json: { detail: 'Model read failed' } } : { json: state.models });
    if (path.includes('/projects/') && path.endsWith('/jobs')) return route.fulfill({ json: jobs });
    if (path.endsWith('/training')) return route.fulfill({ json: { job_id: 'run-teacher', status: 'succeeded', phase: 'completed', events: [], metrics: [], reproducibility: { dataset: { repo_id: 'our/pickup', revision: 'b'.repeat(40), metadata_sha256: hash, total_episodes: 6 }, compute_target: { provider: 'gcp', accelerator: 'L4', region: 'us-central1' }, evidence: { splits: { train: [0, 1], validation: [2, 3] }, environment: { gpu_name: 'NVIDIA L4' } } } } });
    if (path === '/api/v1/policy-options') return route.fulfill({ json: { runtimes: [{ id: 'cpu-student', label: 'Local student worker', provider: 'local', execution: 'native', device: 'cpu', native_distillation: true, enabled: true }], sources: [{ id: 'abstract-source', label: 'Abstract base' }], training_models: [], training_methods: [], default_training_method: 'lora', quantization_defaults: { cpu: { language: 'Q8_0', vision: null }, cuda: { language: 'Q8_0', vision: null } } } });
    if (path === '/api/v1/simulation-options') return route.fulfill({ json: { profiles: [], max_archive_bytes: 4294967296, scored_evaluation: false } });
    if (path.endsWith('/events')) return route.fulfill({ json: [] });
    return route.continue();
  });
  await page.goto('/'); return state;
}
async function openLibrary(page: Page) { await page.getByRole('button', {name:'Dashboard',exact:true}).click(); await page.getByRole('button', {name:'My models',exact:true}).click(); }
test('collection shows model versions and their full recorded training and student history', async ({ page }, testInfo) => {
  const state = await fixture(page); await openLibrary(page);
  await expect(page.getByRole('button', { name: 'Open model Saved other · other' })).toBeVisible();
  await page.getByRole('button', { name: 'Open model Saved student · student' }).click();
  const detail = page.getByRole('article', { name: 'Model details' });
  await expect(detail).toHaveAttribute('data-model-id', 'student');
  await expect(detail.getByRole('region', { name: 'Model lineage' })).toContainText('Saved teacher');
  await expect(detail).toContainText('NVIDIA L4'); await expect(detail).toContainText('our/pickup');
  await expect(detail).toContainText('Outside recorded partitions'); await expect(detail).toContainText('2 · 4, 5');
  await expect(detail).toContainText('Distilled · Saved student');
  await expect(detail.getByRole('region', { name: 'Model lineage' }).getByText('our/pickup', { exact: true })).toHaveCount(2);
  await page.screenshot({ path: testInfo.outputPath('model-lineage.png'), fullPage: true });
  expect(state.mutations).toEqual([]);
  await page.setViewportSize({ width: 320, height: 900 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
});
test('Distill explicitly selects an owned teacher instead of an abstract architecture', async ({ page }) => {
  const state = await fixture(page); await page.getByRole('button', { name: 'Distill', exact: true }).click();
  await expect(page.getByRole('button', { name: 'ACT', exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Choose Saved other · other' })).toHaveCount(0);
  await page.getByRole('button', { name: 'Choose Saved teacher · teacher' }).click();
  await expect(page.getByRole('group', { name: 'Teacher', exact: true }).locator('input:checked')).toHaveValue('teacher');
  expect(state.mutations).toEqual([]);
});
test('empty collections cannot start work on nonexistent models', async ({ page }) => {
  const state = await fixture(page, []); await page.getByRole('button', { name: 'Distill', exact: true }).click();
  await expect(page.getByText('No saved models in this project yet')).toBeVisible();
  await expect(page.getByRole('button', { name: 'ACT', exact: true })).toHaveCount(0);
  await openLibrary(page); await expect(page.getByText('Your models will live here')).toBeVisible();
  expect(state.mutations).toEqual([]);
});
test('cross-project continuation selects the model owner and exact model', async ({ page }) => {
  const state = await fixture(page); await openLibrary(page); await page.getByRole('button', { name: 'Open model Saved other · other' }).click();
  await page.getByRole('region', { name: 'Continue with this model' }).getByRole('button', { name: 'Distill', exact: true }).click();
  await expect(page.getByLabel('Current project')).toHaveAttribute('data-project-id', 'beta');
  await expect(page.getByRole('group', { name: 'Teacher', exact: true }).locator('input:checked')).toHaveValue('other');
  expect(state.mutations).toEqual([]);
});
test('quantization selects the exact saved SmolVLA model and omits abstract sources', async ({ page }) => {
  const state = await fixture(page, ['smol', 'smol-second'].map(id => artifact(id, 'alpha', { format: 'training_checkpoint', metadata: { architecture: 'smolvla' } })));
  await page.getByRole('button', { name: 'Quantize', exact: true }).click(); await chooseTransformationModel(page, 'quantization', 'Choose Saved smol · smol');
  await expect(page.getByRole('group', { name: 'My model', exact: true }).locator('input:checked')).toHaveValue('smol');
  await expect(page.getByText('Abstract base', { exact: true })).toHaveCount(0); expect(state.mutations).toEqual([]);
  await chooseTransformationModel(page, 'quantization', 'Choose Saved smol-second · smol-second');
  await expect(page.getByRole('group', { name: 'My model', exact: true }).locator('input:checked')).toHaveValue('smol-second');
});
test('failed refresh keeps history visible and blocks actions until refreshed', async ({ page }) => {
  const state = await fixture(page); await openLibrary(page); await page.getByRole('button', { name: 'Open model Saved teacher · teacher' }).click();
  state.fail = true; await expect(page.getByRole('button', { name: 'Refresh records', exact: true })).toBeVisible({ timeout: 10000 });
  await expect(page.getByRole('region', { name: 'Continue with this model' }).getByRole('button', { name: 'Distill', exact: true })).toBeDisabled();
  state.fail = false; await page.getByRole('button', { name: 'Refresh records', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Continue with this model' }).getByRole('button', { name: 'Distill', exact: true })).toBeEnabled(); expect(state.mutations).toEqual([]);
});
test('opening training preserves the exact checkpoint and never replaces a missing one', async ({ page }) => {
  const earlier = artifact('teacher', 'alpha', { format: 'training_checkpoint', metadata: { architecture: 'act', step: 20 } });
  const later = artifact('later', 'alpha', { job_id: earlier.job_id, format: 'training_checkpoint', metadata: { architecture: 'act', step: 100 } });
  const state = await fixture(page, [earlier, later]);
  await openLibrary(page); await page.getByRole('button', { name: 'Open model Saved teacher · teacher' }).click();
  await page.getByRole('button', { name: 'Open this checkpoint’s training run and export →' }).click();
  await expect(page.getByLabel('Checkpoint', { exact: true })).toHaveValue('teacher');
  state.models = [later];
  await expect(page.getByText('The requested checkpoint is unavailable. Choose another saved checkpoint explicitly to continue.')).toBeVisible({ timeout: 10000 });
  await expect(page.getByRole('button', { name: 'Quantize checkpoint', exact: true })).toBeDisabled();
  expect(state.mutations).toEqual([]);
});
test('evaluation and simulation continue with the exact model without creating jobs', async ({ page }) => {
  const state = await fixture(page, [artifact('gguf', 'alpha', { format: 'gguf', metadata: { architecture: 'smolvla', precision: 'Q8_0' } }), artifact('teacher')]);
  await page.route('**/api/v1/simulation-options', route => route.fulfill({ json: { profiles: [{ id: 'cup', label: 'Cup simulator', architectures: ['act'], experimental: true, task_object: 'cup' }], max_archive_bytes: 4294967296, scored_evaluation: false } }));
  await openLibrary(page); await page.getByRole('button', { name: 'Open model Saved gguf · gguf' }).click();
  await page.getByRole('region', { name: 'Continue with this model' }).getByRole('button', { name: 'Evaluate', exact: true }).click();
  await expect(page.getByRole('group', { name: 'My model', exact: true }).locator('input:checked')).toHaveValue('gguf');
  state.models = state.models.filter(item => item.id !== 'gguf');
  await expect(page.getByText('The selected model is unavailable. Refresh or choose another saved model explicitly.')).toBeVisible({ timeout: 10000 });
  await expect(page.getByRole('button', { name: 'Start evaluation', exact: true })).toBeDisabled();
  await openLibrary(page); await page.getByRole('button', { name: 'Open model Saved teacher · teacher' }).click();
  await page.getByRole('region', { name: 'Continue with this model' }).getByRole('button', { name: 'Run in simulation', exact: true }).click();
  await expect(page.getByRole('radiogroup', { name: 'Native policy', exact: true }).locator('input:checked')).toHaveValue('teacher');
  await expect(page.getByRole('checkbox', { name: /experimental, paid cloud rollout/ })).not.toBeChecked();
  expect(state.mutations).toEqual([]);
});

test('distilled and quantized models share the model collection and reopen their exact producing jobs', async ({page})=>{
  const student=artifact('student','alpha',{parent_ids:['teacher'],metadata:{architecture:'act',recipe:'act-action-distillation-v1'}});
  const packed=artifact('packed','alpha',{format:'native_quantized',parent_ids:['student'],metadata:{architecture:'act',precision:'int8'}});
  const state=await fixture(page,[artifact('teacher'),student,packed,artifact('foreign','beta')]);
  state.jobs.find(job=>job.id==='run-student')!.result={artifacts:[student],reports:[]};
  const quant=job('run-packed','policy.quantize',{operation:'policy.quantize',runtime_id:'quant-cpu',artifact_id:'student',native_quantization:{format:'firebird_quant',bits:8,group_size:64}});
  quant.result={artifacts:[packed],reports:[]} as any;state.jobs.push(quant);
  await page.getByRole('navigation',{name:'Policy lifecycle'}).getByRole('button',{name:'Distill',exact:true}).click();
  await openTransformationJob(page,'distillation','run-student');
  await expect(page.getByRole('group',{name:'Distillation setup',exact:true})).toHaveCount(0);
  await page.getByRole('button',{name:'View Saved student in My models',exact:true}).click();
  await expect(page.getByRole('article',{name:'Model details',exact:true})).toHaveAttribute('data-model-id','student');
  await page.getByRole('button',{name:'Open this model’s distillation job →',exact:true}).click();
  await expect(page.getByRole('article',{name:'Distillation job details',exact:true})).toHaveAttribute('data-job-id','run-student');
  await page.getByRole('navigation',{name:'Policy lifecycle'}).getByRole('button',{name:'Quantize',exact:true}).click();
  await expect(page.getByRole('region',{name:'Quantization jobs',exact:true}).locator('[data-job-id="run-packed"]')).toBeVisible();
  await openTransformationJob(page,'quantization','run-packed');
  await expect(page.getByRole('group',{name:'ACT quantization setup',exact:true})).toHaveCount(0);
  await page.getByRole('button',{name:'View Saved packed in My models',exact:true}).click();
  await expect(page.getByRole('article',{name:'Model details',exact:true})).toHaveAttribute('data-model-id','packed');
  const lineage=page.getByRole('region',{name:'Model lineage',exact:true});
  await expect(lineage).toContainText('Saved teacher');await expect(lineage).toContainText('Saved student');await expect(lineage).toContainText('Saved packed');
  await page.getByRole('button',{name:'Open this model’s quantization job →',exact:true}).click();
  await expect(page.getByRole('article',{name:'ACT quantization job details',exact:true})).toHaveAttribute('data-job-id','run-packed');
  expect(state.mutations).toEqual([]);
});
