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
  const state = { jobs: [dataset(), dataset('beta')] as Record<string, any>[], artifacts: [teacher, artifact('manual-float'), artifact('manual-packed', 'native_quantized'), artifact('beta-float', 'native_checkpoint', {}, 'beta'), artifact('beta-packed', 'native_quantized', {}, 'beta')], runtimes: [...runtimes] as Record<string, any>[], failOptions: false, failSimulation: false, optionsGate: null as Promise<void> | null, profiles: [] as Record<string, any>[], mutations: [] as { path: string; body: Record<string, any> }[] };
  await page.route('**/api/v1/**', async route => {
    const request = route.request(), path = new URL(request.url()).pathname;
    if (request.method() !== 'GET') {
      const body = request.postDataJSON(); state.mutations.push({ path, body });
      if (path === '/api/v1/projects/alpha/policy-jobs') {
        const id = body.operation === 'policy.distill' ? 'student-job' : body.operation === 'policy.quantize' ? 'quant-job' : 'replay-job';
        const job = makeJob(id, body); state.jobs.push(job); return route.fulfill({ status: 202, json: job });
      }
      if (path.endsWith('/cancel')) { const job = state.jobs.find(item => item.id === path.split('/').at(-2))!; job.status = 'cancelled'; return route.fulfill({ json: job }); }
      return route.fulfill({ status: 405, json: { detail: 'Unexpected generated-fixture mutation' } });
    }
    if (path === '/api/v1/projects') return route.fulfill({ json: [{ id: 'alpha', name: 'Generated lifecycle journey', created_at: time }, { id: 'beta', name: 'Isolated second project', created_at: time }] });
    if (path === '/api/v1/simulation-options' && state.failSimulation) return route.fulfill({ status: 503, json: { detail: 'Generated profile outage' } });
    if (path === '/api/v1/simulation-options') return route.fulfill({ json: { profiles: state.profiles, scored_evaluation: false, max_archive_bytes: 4294967296 } });
    if (path === '/api/v1/policy-options' && state.optionsGate) await state.optionsGate;
    if (path === '/api/v1/policy-options' && state.failOptions) return route.fulfill({ status: 503, json: { detail: 'Generated options unavailable' } });
    if (path === '/api/v1/policy-options') return route.fulfill({ json: { runtimes: state.runtimes, sources: [], training_models: [], training_methods: [], default_training_method: 'lora', quantization_defaults: { cpu: { language: 'Q8_0', vision: null }, cuda: { language: 'Q8_0', vision: null }, note: '' } } });
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


test('initial native-only entry chooses ACT quantization and exact saved replay without mutation', async ({ page }) => {
  const state = await fixture(page);
  const quant = makeJob('quant-job', quantRequest()); completeQuant(quant); state.jobs.push(quant); state.artifacts.push(student, packed);
  const replay = savedReplay('replay-job');
  replay.result = { artifacts: [artifact('replay:record', 'native_run_record', { recipe: 'native-observation-replay-v1', model_id: model }, 'alpha', replay.id)], reports: [{ operation: 'policy.run', stage: 'native_replay', mode: 'independent_observation_replay', device: 'cpu', source_artifact_id: packed.id, dataset_job_id: 'alpha-data', observation_source: { kind: 'generated_fixture' }, model_id: model, observations: 1, action_shape: [100, 6], reset_repeat_exact: true, server_closed: true, task_success: null, quality_verified: false, calibration_verified: false, speedup_verified: false, isaac_runtime_verified: false, elapsed_seconds: 2 }] };
  state.jobs.push(replay);
  await page.reload(); await expect(page.getByLabel('Current project')).toHaveValue('alpha');
  await page.getByRole('button', { name: 'Quantize', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Native ACT · INT8 / INT4', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('article', { name: 'ACT quantization job details' })).toHaveAttribute('data-job-id', 'quant-job');
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Observation replay · ACT', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('article', { name: 'Observation replay details' })).toHaveAttribute('data-job-id', replay.id);
  await expect(page.getByRole('region', { name: 'Predicted action chunks' }).getByRole('img')).toHaveCount(6);
  await expect(page.getByRole('link', { name: 'Download verified replay record', exact: true })).toHaveAttribute('href', '/api/v1/projects/alpha/artifacts/replay%3Arecord/download');
  expect(state.mutations).toHaveLength(0);
});

test('engine history excludes replay while retaining an actual engine job', async ({ page }) => {
  const state = await fixture(page);
  const replay = makeJob('saved-replay', { operation: 'policy.run', runtime_id: 'replay-cpu', native_replay: { adapter: 'act-packed-observation-v1' } }); replay.status = 'succeeded';
  const engine = makeJob('saved-engine', { operation: 'policy.run', runtime_id: 'engine-cpu', artifact_id: 'gguf', evaluation: { mode: 'engine' } }); engine.status = 'succeeded'; state.jobs.push(replay, engine);
  await page.reload(); await expect(page.getByLabel('Current project')).toHaveValue('alpha');
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  await page.getByRole('button', { name: 'Engine checks · GGUF', exact: true }).click();
  await expect(page.locator('.job-history-entry[data-job-id="saved-engine"]')).toBeVisible();
  await expect(page.locator('.job-history-entry[data-job-id="saved-replay"]')).toHaveCount(0);
  expect(state.mutations).toHaveLength(0);
});

function savedReplay(id: string, status = 'succeeded', projectId = 'alpha') {
  return { ...makeJob(id, { operation: 'policy.run', runtime_id: 'replay-cpu', artifact_id: packed.id, dataset_job_id: 'alpha-data', timeout_seconds: 600,
    native_replay: { adapter: 'act-packed-observation-v1', selection: [{ episode_index: 0, frame_index: 3 }], units, coordinate_attestation: 'generated_fixture' } }), status, project_id: projectId };
}
async function reloadProject(page: Page) { await page.reload(); await expect(page.getByLabel('Current project')).toHaveValue('alpha'); }

test('capabilities alone choose the sole native modes without preparing or submitting a recipe', async ({ page }) => {
  const state = await fixture(page);
  await page.getByRole('button', { name: 'Quantize', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Native ACT · INT8 / INT4', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByLabel('ACT inference policy', { exact: true })).toHaveValue('');
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Observation replay · ACT', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByLabel('Packed ACT policy', { exact: true })).toHaveValue('');
  await expect(page.getByRole('button', { name: 'Run CPU observation replay', exact: true })).toBeDisabled();
  expect(state.mutations).toHaveLength(0);
});

for (const mode of ['quantize', 'replay'] as const) test(`automatically opened active ${mode} can cancel only that exact job`, async ({ page }) => {
  const state = await fixture(page), job = mode === 'quantize' ? makeJob('active-quant', quantRequest()) : savedReplay('active-replay', 'running');
  state.jobs.push(job); await reloadProject(page);
  await page.getByRole('button', { name: mode === 'quantize' ? 'Quantize' : 'Run', exact: true }).click();
  const details = page.getByRole('article', { name: mode === 'quantize' ? 'ACT quantization job details' : 'Observation replay details' });
  await expect(details).toHaveAttribute('data-job-id', job.id);
  await page.getByRole('button', { name: mode === 'quantize' ? 'Cancel selected ACT quantization' : 'Cancel selected replay', exact: true }).click();
  expect(state.mutations).toHaveLength(0);
  await page.getByRole('button', { name: 'Confirm cancellation', exact: true }).click();
  await expect(details).toContainText('cancelled');
  expect(state.mutations).toEqual([{ path: `/api/v1/jobs/${job.id}/cancel`, body: {} }]);
});

test('delayed capability reads and history polling cannot replace a manual mode or selected replay', async ({ page }) => {
  const state = await fixture(page); state.jobs.push(savedReplay('replay-a'), savedReplay('replay-b')); await reloadProject(page);
  let release!: () => void; state.optionsGate = new Promise<void>(resolve => { release = resolve; });
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  await page.getByRole('button', { name: 'Engine checks · GGUF', exact: true }).click();
  const response = page.waitForResponse('**/api/v1/policy-options'); release(); await response;
  await expect(page.getByRole('button', { name: 'Engine checks · GGUF', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Observation replay · ACT', exact: true }).click();
  await page.getByLabel('Saved replay', { exact: true }).selectOption('replay-b');
  state.jobs.push({ ...savedReplay('newer'), created_at: '2026-09-28T12:00:00Z' }); state.runtimes = [];
  await page.getByRole('button', { name: 'Refresh replay jobs', exact: true }).click();
  await expect(page.getByLabel('Saved replay', { exact: true }).locator('option[value="newer"]')).toHaveCount(1);
  await expect(page.getByRole('article', { name: 'Observation replay details' })).toHaveAttribute('data-job-id', 'replay-b');
  await page.getByRole('button', { name: 'Dataset', exact: true }).click(); await page.getByRole('button', { name: 'Run', exact: true }).click();
  await expect(page.getByRole('article', { name: 'Observation replay details' })).toHaveAttribute('data-job-id', 'replay-b');
  expect(state.mutations).toHaveLength(0);
});

test('manual mode is project scoped and saved foreign jobs never become entry context', async ({ page }) => {
  const state = await fixture(page); state.jobs.push(savedReplay('alpha-replay')); await reloadProject(page);
  await page.getByRole('button', { name: 'Run', exact: true }).click(); await page.getByRole('button', { name: 'Engine checks · GGUF', exact: true }).click();
  await page.getByLabel('Current project').selectOption('beta');
  await expect(page.getByRole('button', { name: 'Observation replay · ACT', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('article', { name: 'Observation replay details' })).toHaveCount(0);
  await page.getByLabel('Current project').selectOption('alpha');
  await expect(page.getByRole('button', { name: 'Engine checks · GGUF', exact: true })).toHaveAttribute('aria-pressed', 'true');
  expect(state.mutations).toHaveLength(0);
});

for (const mode of ['quantize', 'replay'] as const) test(`unknown ${mode} submission takes precedence over automatic history`, async ({ page }) => {
  const state = await fixture(page);
  const operation = mode === 'quantize' ? 'policy.quantize' : 'policy.run.replay';
  await page.evaluate(({ operation }) => sessionStorage.setItem(`firebird:job-attempt:${operation}:alpha`, JSON.stringify({ state: 'pending', message: 'Generated lost acknowledgement' })), { operation });
  state.runtimes = []; state.jobs.push(makeJob('other-mode', { operation: mode === 'quantize' ? 'policy.quantize' : 'policy.run', runtime_id: 'engine' })); await reloadProject(page);
  await page.getByRole('button', { name: mode === 'quantize' ? 'Quantize' : 'Run', exact: true }).click();
  await expect(page.getByRole('button', { name: mode === 'quantize' ? 'Native ACT · INT8 / INT4' : 'Observation replay · ACT', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByText(/An earlier request did not return a verified outcome/)).toBeVisible();
  await page.getByRole('button', { name: mode === 'quantize' ? 'Refresh ACT quantization jobs' : 'Refresh replay jobs', exact: true }).click();
  expect(state.mutations).toHaveLength(0);
  expect(await page.evaluate(operation => sessionStorage.getItem(`firebird:job-attempt:${operation}:alpha`), operation)).not.toBeNull();
});

test('unreadable request recovery pauses automatic selection without clearing stored evidence', async ({ page }) => {
  const state = await fixture(page);
  await page.addInitScript(() => { const original = Storage.prototype.getItem; Storage.prototype.getItem = function(key) { if (key.startsWith('firebird:job-attempt:')) throw new Error('Generated storage outage'); return original.call(this, key); }; });
  await reloadProject(page); await page.getByRole('button', { name: 'Run', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Workflow selection status' })).toContainText('Saved request recovery could not be read');
  await expect(page.getByRole('button', { name: 'Observation replay · ACT', exact: true })).toHaveAttribute('aria-pressed', 'false');
  await page.getByRole('button', { name: 'Observation replay · ACT', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Run CPU observation replay', exact: true })).toBeDisabled(); expect(state.mutations).toHaveLength(0);
});

test('Evaluate reports unavailable scoring and only offers configured observation replay', async ({ page }) => {
  const state = await fixture(page);
  await page.getByRole('button', { name: 'Evaluate', exact: true }).click();
  const purpose = page.getByRole('region', { name: 'Evaluation purpose' });
  await expect(purpose).toContainText('No engine evaluation target is configured.');
  await expect(purpose).toContainText('No LIBERO evaluation target is configured.');
  await expect(purpose).toContainText('scored ACT / Isaac evaluation is not configured');
  await expect(purpose.getByRole('button', { name: 'Open native Isaac Run' })).toHaveCount(0);
  await purpose.getByRole('button', { name: 'Open observation replay' }).click();
  await expect(page.getByRole('button', { name: 'Observation replay · ACT', exact: true })).toHaveAttribute('aria-pressed', 'true'); expect(state.mutations).toHaveLength(0);
});

test('query failures keep Evaluate availability unknown and do not choose an initial Run mode', async ({ page }) => {
  const state = await fixture(page); state.failOptions = true; state.failSimulation = true;
  await page.getByRole('button', { name: 'Evaluate', exact: true }).click();
  const purpose = page.getByRole('region', { name: 'Evaluation purpose' });
  await expect(purpose).toContainText('Availability is unknown.');
  await expect(purpose).not.toContainText('No LIBERO evaluation target is configured.');
  await expect(purpose.getByRole('button', { name: 'Open observation replay' })).toHaveCount(0);
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Workflow selection status' })).toContainText('could not be loaded');
  await expect(page.getByRole('button', { name: 'Engine checks · GGUF', exact: true })).toHaveAttribute('aria-pressed', 'false');
  state.failOptions = false; state.failSimulation = false; await page.getByRole('button', { name: 'Retry workflow context' }).click();
  await expect(page.getByRole('button', { name: 'Observation replay · ACT', exact: true })).toHaveAttribute('aria-pressed', 'true'); expect(state.mutations).toHaveLength(0);
});

test('the engine preparation link explicitly opens GGUF after a native quantization choice', async ({ page }) => {
  const state = await fixture(page);
  await page.getByRole('button', { name: 'Quantize', exact: true }).click(); await page.getByRole('button', { name: 'Native ACT · INT8 / INT4', exact: true }).click();
  await page.getByRole('button', { name: 'Evaluate', exact: true }).click(); await page.getByRole('button', { name: 'New evaluation', exact: true }).click();
  await page.getByRole('button', { name: 'Go to quantization', exact: true }).click();
  await expect(page.getByRole('button', { name: 'SmolVLA · GGUF', exact: true })).toHaveAttribute('aria-pressed', 'true'); expect(state.mutations).toHaveLength(0);
});

test('saved simulation history opens once and preserves the manually selected exact job', async ({ page }) => {
  const state = await fixture(page);
  state.profiles = [{ id: 'cup-fixture', label: 'Generated cup profile', architectures: ['act', 'smolvla'], experimental: true, task_object: 'cup' }];
  for (const id of ['isaac-a', 'isaac-b']) state.jobs.push({ ...makeJob(id, { operation: 'policy.run', runtime_id: 'cup-fixture', simulation: { profile_id: 'cup-fixture', experimental: true } }), status: 'succeeded' });
  await reloadProject(page); await page.getByRole('button', { name: 'Run', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Native Isaac · ACT / SmolVLA', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('article', { name: 'Native simulation job details' })).toHaveAttribute('data-job-id', 'isaac-a');
  await page.getByLabel('Saved simulation job', { exact: true }).selectOption('isaac-b');
  await page.getByRole('button', { name: 'Dataset', exact: true }).click(); await page.getByRole('button', { name: 'Run', exact: true }).click();
  await expect(page.getByRole('article', { name: 'Native simulation job details' })).toHaveAttribute('data-job-id', 'isaac-b');
  expect(state.mutations).toHaveLength(0);
});

test('mixed configured workflows retain the legacy entry and a deliberate native choice', async ({ page }) => {
  const state = await fixture(page);
  state.runtimes.push({ id: 'engine', label: 'Generated engine', execution: 'native', provider: 'local', device: 'cpu', enabled: true, launchable: true, run: true, engine_evaluation: true, training: false, simulation: true });
  state.profiles = [{ id: 'cup-fixture', label: 'Generated cup profile', architectures: ['act'], experimental: true, task_object: 'cup' }];
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Engine checks · GGUF', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Observation replay · ACT', exact: true }).click();
  await page.getByRole('button', { name: 'Evaluate', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Evaluation purpose' })).toContainText('A LIBERO target is configured');
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Observation replay · ACT', exact: true })).toHaveAttribute('aria-pressed', 'true'); expect(state.mutations).toHaveLength(0);
});
