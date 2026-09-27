import { expect, test, type Page } from '@playwright/test';

// Generated API records exercise real workspace navigation; no model or cloud work runs.
const time = '2026-09-27T12:00:00Z', model = `sha256:${'f'.repeat(64)}`;
const units = ['degrees', 'degrees', 'degrees', 'degrees', 'degrees', 'recorded_gripper'];
const runtime = (id: string, capability: string) => ({ id, label: `Generated ${id}`, execution: 'native', provider: 'local', device: 'cpu', enabled: true, launchable: true, [capability]: true, [`${capability}_only`]: true, training: false, simulation: false, run: false, engine_evaluation: false });
const runtimes = [runtime('student-cpu', 'native_distillation'), runtime('quant-cpu', 'native_quantization'), runtime('replay-cpu', 'native_replay')];
function artifact(id: string, format = 'native_checkpoint', metadata: Record<string, unknown> = {}, project = 'alpha', jobId = 'source') {
  return { id, project_id: project, job_id: jobId, label: `Generated ${id}`, format, path: 'owned', manifest_sha256: 'a'.repeat(64), file_bytes: 4096, parent_ids: [], metadata: { architecture: 'act', inference_only: true, use_vae: false, ...metadata } };
}
const teacher = artifact('teacher'), student = artifact('distilled:policy', 'native_checkpoint', { recipe: 'act-action-distillation-v1' }, 'alpha', 'student-job');
const packed = artifact('quantized:policy', 'native_quantized', { format: 'firebird_quant', format_version: 1, model_id: model, precision: 'int8', fresh_reload_verified: true, cpu_reload_verified: true, source_artifact_id: student.id, source_artifact_manifest_sha256: student.manifest_sha256 }, 'alpha', 'quant-job');
function dataset(project = 'alpha') { return { id: `${project}-data`, project_id: project, kind: 'dataset.inspect', status: 'succeeded', created_at: time, updated_at: time, request: { source: 'local', path: 'generated' }, result: { source: 'local', format: 'lerobot_v3', repo_id: null, revision: `metadata-sha256:${'c'.repeat(64)}`, metadata_sha256: 'c'.repeat(64), fps: 30, robot_type: 'generated-so101', inspected_at: time, warnings: [], total_episodes: 6, total_frames: 24, inspection_scope: 'complete_snapshot', features: {}, snapshot: { id: `sha256:${'b'.repeat(64)}`, manifest_sha256: 'b'.repeat(64), lineage_validated: true, total_episodes: 6 } } }; }
function studentRequest() { return { operation: 'policy.distill', runtime_id: 'student-cpu', artifact_id: teacher.id, dataset_job_id: 'alpha-data', timeout_seconds: 600, native_distillation: { adapter: 'act-act-v1', student: 'act-256', steps: 100, learning_rate: .0001, seed: 1729, frame_stride: 30, splits: { train: [0, 1], validation: [2, 3], final: [4, 5] }, coordinate_attestation: 'generated_fixture', units } }; }
function quantRequest() { return { operation: 'policy.quantize', runtime_id: 'quant-cpu', artifact_id: student.id, timeout_seconds: 600, native_quantization: { format: 'firebird_quant', bits: 8, group_size: 64 } }; }
function makeJob(id: string, request: Record<string, any>) { return { id, project_id: 'alpha', kind: request.operation, status: 'running', stage: 'preparing', created_at: time, updated_at: time, request, error: null, result: null as Record<string, any> | null }; }
function completeStudent(job: ReturnType<typeof makeJob>) {
  job.status = 'succeeded'; job.stage = 'completed'; job.result = { artifacts: [student], reports: [{ operation: 'policy.distill', adapter: 'act-act-v1', teacher_artifact_id: teacher.id, dataset_job_id: 'alpha-data', steps: 100, fresh_reload_verified: true, quality_verified: false, calibration_verified: false, speedup_verified: false, task_success: null, dataset_kind: 'generated_fixture', student_weights_bytes: 55971416, teacher_inference_tensor_bytes: 136972568, trained_student: { validation: { teacher_normalized_l1: .33 }, final: { teacher_normalized_l1: .39 } } }] };
}
function completeQuant(job: ReturnType<typeof makeJob>) {
  job.status = 'succeeded'; job.stage = 'completed'; job.result = { artifacts: [packed], reports: [{ stage: 'operation', operation: 'policy.quantize', architecture: 'act', format: 'firebird_quant', format_version: 1, inference_only: true, precision: 'int8', model_id: model, source_artifact_id: student.id, source_artifact_manifest_sha256: student.manifest_sha256, source_weight_bytes: 55971416, packed_weight_bytes: 16000000, policy_package_bytes: 16010000, fresh_reload_verified: true, cpu_reload_verified: true, runtime_verified: false, isaac_runtime_verified: false, quality_verified: false, calibration_verified: false, speedup_verified: false, task_success: null, gpu_memory_bytes: null, inference_speedup: null, drift_from_fp32: [171, 902].map((seed, index) => ({ seed, input_sha256: (index ? 'd' : 'c').repeat(64), raw: { rmse: .003, maximum_absolute_difference: .01, coordinates: 600 }, postprocessed: { rmse: .09, maximum_absolute_difference: .14, coordinates: 600 } })) }] };
}
async function fixture(page: Page) {
  const state = { jobs: [dataset(), dataset('beta')] as Record<string, any>[], artifacts: [teacher, artifact('manual-float'), artifact('manual-packed', 'native_quantized'), artifact('beta-float', 'native_checkpoint', {}, 'beta'), artifact('beta-packed', 'native_quantized', {}, 'beta')], mutations: [] as { path: string; body: Record<string, any> }[] };
  await page.route('**/api/v1/**', async route => {
    const request = route.request(), path = new URL(request.url()).pathname;
    if (request.method() !== 'GET') {
      const body = request.postDataJSON(); state.mutations.push({ path, body });
      if (path === '/api/v1/projects/alpha/policy-jobs') {
        const id = body.operation === 'policy.distill' ? 'student-job' : body.operation === 'policy.quantize' ? 'quant-job' : 'replay-job';
        const job = makeJob(id, body); state.jobs.push(job); return route.fulfill({ status: 202, json: job });
      }
      return route.fulfill({ status: 405, json: { detail: 'Unexpected generated-fixture mutation' } });
    }
    if (path === '/api/v1/projects') return route.fulfill({ json: [{ id: 'alpha', name: 'Generated lifecycle journey', created_at: time }, { id: 'beta', name: 'Isolated second project', created_at: time }] });
    if (path === '/api/v1/policy-options') return route.fulfill({ json: { runtimes, sources: [], training_models: [], training_methods: [], default_training_method: 'lora', quantization_defaults: { cpu: { language: 'Q8_0', vision: null }, cuda: { language: 'Q8_0', vision: null }, note: '' } } });
    if (path.endsWith('/artifacts')) return route.fulfill({ json: state.artifacts }); // Deliberately includes foreign project records; clients must filter them.
    if (path.endsWith('/jobs') && path.includes('/projects/')) return route.fulfill({ json: state.jobs });
    if (path.endsWith('/events')) return route.fulfill({ json: [] });
    if (path.endsWith('/replay')) {
      const job = state.jobs.find(item => item.id === 'replay-job')!;
      return route.fulfill({ json: { artifact_id: 'replay:record', job_id: job.id, model_id: model, source_kind: 'generated_fixture', coordinate_names: ['j0', 'j1', 'j2', 'j3', 'j4', 'gripper'], units, records: job.request.native_replay.selection.map((row: object) => ({ ...row, reset_repeat_exact: true, actions: Array.from({ length: 100 }, (_, step) => Array.from({ length: 6 }, (_, joint) => joint + step / 100)) })) } });
    }
    if (path.startsWith('/api/v1/jobs/')) { const job = state.jobs.find(item => item.id === path.split('/').at(-1)); if (job) return route.fulfill({ json: job }); }
    return route.continue();
  });
  await page.goto('/'); await expect(page.getByLabel('Current project')).toHaveValue('alpha');
  return state;
}
async function openStudent(page: Page, state: Awaited<ReturnType<typeof fixture>>) {
  const job = makeJob('student-job', studentRequest()); completeStudent(job); state.jobs.push(job);
  await page.getByRole('button', { name: 'Distill', exact: true }).click();
  await page.getByLabel('Saved distillation job', { exact: true }).selectOption(job.id);
  await page.getByRole('button', { name: 'Open ACT quantization', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Native ACT quantization', exact: true })).toBeVisible();
}
async function openPacked(page: Page, state: Awaited<ReturnType<typeof fixture>>) {
  const job = makeJob('quant-job', quantRequest()); completeQuant(job); state.jobs.push(job);
  await page.getByRole('button', { name: 'Quantize', exact: true }).click();
  await page.getByRole('button', { name: 'Native ACT · INT8 / INT4', exact: true }).click();
  await page.getByLabel('Saved ACT quantization job', { exact: true }).selectOption(job.id);
  await page.getByRole('button', { name: 'Replay recorded observations', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Inspect the actions your policy predicts', exact: true })).toBeVisible();
}

test('explicit Distill to Quantize to Replay carries exact artifacts without submitting on navigation', async ({ page }) => {
  const state = await fixture(page);
  await page.getByRole('button', { name: 'Distill', exact: true }).click();
  await page.getByLabel('ACT teacher', { exact: true }).selectOption(teacher.id);
  await page.getByLabel('Verified dataset', { exact: true }).selectOption('alpha-data');
  for (const [label, value] of [['Training episodes', '0, 1'], ['Validation episodes', '2, 3'], ['Final episodes', '4, 5'], ['Six coordinate units', units.join(', ')]]) await page.getByLabel(label, { exact: true }).fill(value);
  await page.getByRole('checkbox', { name: 'This snapshot contains generated test observations.' }).check();
  await page.getByRole('checkbox', { name: /I verified that the dataset/ }).check();
  await page.getByRole('button', { name: 'Train ACT256 student', exact: true }).click();
  await expect(page.getByRole('article', { name: 'Distillation job details' })).toHaveAttribute('data-job-id', 'student-job');
  expect(state.mutations).toEqual([{ path: '/api/v1/projects/alpha/policy-jobs', body: studentRequest() }]);
  completeStudent(state.jobs.find(item => item.id === 'student-job') as ReturnType<typeof makeJob>); state.artifacts.push(student);
  await page.getByRole('button', { name: 'Refresh distillation jobs', exact: true }).click();
  await page.getByRole('button', { name: 'Open ACT quantization', exact: true }).click();
  await expect(page.getByLabel('ACT inference policy', { exact: true })).toHaveValue(student.id); expect(state.mutations).toHaveLength(1);
  await page.getByRole('button', { name: 'Create ACT quantized package', exact: true }).click();
  await expect(page.getByRole('article', { name: 'ACT quantization job details' })).toHaveAttribute('data-job-id', 'quant-job');
  expect(state.mutations[1]).toEqual({ path: '/api/v1/projects/alpha/policy-jobs', body: quantRequest() });
  completeQuant(state.jobs.find(item => item.id === 'quant-job') as ReturnType<typeof makeJob>); state.artifacts.push(packed);
  await page.getByRole('button', { name: 'Refresh ACT quantization jobs', exact: true }).click();
  await page.getByRole('button', { name: 'Replay recorded observations', exact: true }).click();
  await expect(page.getByLabel('Packed ACT policy', { exact: true })).toHaveValue(packed.id); expect(state.mutations).toHaveLength(2);
  const submit = page.getByRole('button', { name: 'Run CPU observation replay', exact: true }); await expect(submit).toBeDisabled();
  await expect(page.getByLabel('Observation dataset', { exact: true })).toHaveValue('');
  await page.getByLabel('Observation dataset', { exact: true }).selectOption('alpha-data');
  await page.getByLabel('Episode and frame pairs', { exact: true }).fill('0:3, 2:1');
  await page.getByLabel('Six replay coordinate units', { exact: true }).fill(units.join(', '));
  await page.getByRole('checkbox', { name: 'These are generated test observations.' }).check(); await expect(submit).toBeDisabled();
  await page.getByRole('checkbox', { name: /I verified that these/ }).check(); await submit.click();
  await expect(page.getByRole('article', { name: 'Observation replay details' })).toHaveAttribute('data-job-id', 'replay-job');
  expect(state.mutations).toHaveLength(3);
  expect(state.mutations[2]).toEqual({ path: '/api/v1/projects/alpha/policy-jobs', body: { operation: 'policy.run', runtime_id: 'replay-cpu', artifact_id: packed.id, dataset_job_id: 'alpha-data', timeout_seconds: 600, native_replay: { adapter: 'act-packed-observation-v1', selection: [{ episode_index: 0, frame_index: 3 }, { episode_index: 2, frame_index: 1 }], units, coordinate_attestation: 'generated_fixture' } } });
  const replay = state.jobs.find(item => item.id === 'replay-job')!; replay.status = 'succeeded'; replay.result = { artifacts: [artifact('replay:record', 'native_run_record', { recipe: 'native-observation-replay-v1', model_id: model }, 'alpha', replay.id)], reports: [{ operation: 'policy.run', stage: 'native_replay', mode: 'independent_observation_replay', device: 'cpu', source_artifact_id: packed.id, dataset_job_id: 'alpha-data', observation_source: { kind: 'generated_fixture' }, model_id: model, observations: 2, action_shape: [100, 6], reset_repeat_exact: true, server_closed: true, task_success: null, quality_verified: false, calibration_verified: false, speedup_verified: false, isaac_runtime_verified: false, elapsed_seconds: 2 }] };
  await page.getByRole('button', { name: 'Refresh replay jobs', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Predicted action chunks' }).getByRole('img')).toHaveCount(6);
  await expect(page.getByRole('link', { name: 'Download verified replay record', exact: true })).toHaveAttribute('href', /replay%3Arecord\/download$/);
  await expect(page.getByText(/does not measure task success/)).toBeVisible(); expect(state.mutations).toHaveLength(3);
});

for (const target of ['quantization', 'replay'] as const) test(`late ${target} handoff artifact cannot replace a manual selection`, async ({ page }) => {
  const state = await fixture(page);
  if (target === 'quantization') await openStudent(page, state); else await openPacked(page, state);
  const label = target === 'quantization' ? 'ACT inference policy' : 'Packed ACT policy', manual = target === 'quantization' ? 'manual-float' : 'manual-packed', incoming = target === 'quantization' ? student : packed;
  const picker = page.getByLabel(label, { exact: true }); await expect(picker.locator(`option[value="${manual}"]`)).toHaveCount(1); await expect(picker).toHaveValue('');
  await picker.selectOption(manual); state.artifacts.push(incoming);
  await page.getByRole('button', { name: target === 'quantization' ? 'Refresh ACT quantization jobs' : 'Refresh replay jobs', exact: true }).click();
  await expect(picker.locator(`option[value="${incoming.id}"]`)).toHaveCount(1); await expect(picker).toHaveValue(manual); expect(state.mutations).toHaveLength(0);
});

for (const target of ['quantization', 'replay'] as const) test(`${target} handoff does not leak into another project`, async ({ page }) => {
  const state = await fixture(page); state.artifacts.push(student, packed);
  if (target === 'quantization') await openStudent(page, state); else await openPacked(page, state);
  const label = target === 'quantization' ? 'ACT inference policy' : 'Packed ACT policy', expected = target === 'quantization' ? student.id : packed.id;
  await expect(page.getByLabel(label, { exact: true })).toHaveValue(expected);
  await page.getByLabel('Current project').selectOption('beta');
  const picker = page.getByLabel(label, { exact: true }); await expect(picker).toHaveValue(''); await expect(picker.locator(`option[value="${expected}"]`)).toHaveCount(0);
  await expect(page.getByRole('button', { name: target === 'quantization' ? 'Create ACT quantized package' : 'Run CPU observation replay', exact: true })).toBeDisabled();
  await expect(page.getByRole('article', { name: target === 'quantization' ? 'ACT quantization job details' : 'Observation replay details', exact: true })).toHaveCount(0); expect(state.mutations).toHaveLength(0);
});
