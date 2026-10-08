import { chooseTransformationModel, openTransformationJob, transformationJobs, selectedQuantizationModel, quantizationModelOption, selectQuantizationModel } from './lifecycle-controls';
import { selectProject } from './project-controls';
import { expect, test, type Page } from '@playwright/test';
import { initialQuantizeEntry, initialRunEntry } from '../../apps/web/src/lib/workflow-entry';
import type { Job, PolicyArtifact } from '../../apps/web/src/lib/api';
import { studentTeacher } from '../../apps/web/src/lib/native-distillation';
import { hasSimulatorControlContract, nativeQuantizationInput } from '../../apps/web/src/lib/native-quantization';

// Generated API records exercise real workspace navigation; no model or cloud work runs.
const time = '2026-09-27T12:00:00Z', model = `sha256:${'f'.repeat(64)}`;
const policyChoices = (page: Page, label: string) => page.getByRole(label === 'ACT inference policy' ? 'group' : 'radiogroup', { name: label === 'ACT inference policy' ? 'Policy' : label, exact: true });
async function openSavedAct(page: Page) {
  await expect(page).toHaveURL(/\/(distillation|quantization)\//);
  const operation = page.url().includes('/distillation/') ? 'distillation' : 'quantization';
  await chooseTransformationModel(page, operation, 'Choose Generated teacher · teacher', 'source');
}

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
        const job = makeJob(body.simulation ? 'simulation-job' : id, body);
        if (body.simulation) Object.assign(job, { simulation_target: { profile_id: body.simulation.profile_id, profile_sha256: 'e'.repeat(64), policy_runtime: 'packed-act-cpu', provider: 'gcp', accelerators: ['L4'], source_manifest_sha256: state.artifacts.find(item => item.id === body.artifact_id)?.manifest_sha256 } });
        state.jobs.push(job); return route.fulfill({ status: 202, json: job });
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
  await page.goto('/datasets/'); await expect(page.getByLabel('Current project')).toHaveAttribute('data-project-id', 'alpha');
  return state;
}
async function openStudent(page: Page, state: Awaited<ReturnType<typeof fixture>>) {
  const job = makeJob('student-job', studentRequest()); completeStudent(job); state.jobs.push(job);
  await page.getByRole('link', { name: 'Distill', exact: true }).click();
  await openTransformationJob(page, 'distillation', job.id);
  await page.getByRole('button', { name: 'Open ACT quantization', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Native ACT quantization', exact: true })).toBeVisible();
}
async function openPacked(page: Page, state: Awaited<ReturnType<typeof fixture>>) {
  const job = makeJob('quant-job', quantRequest()); completeQuant(job); state.jobs.push(job);
  await page.getByRole('link', { name: 'Quantize', exact: true }).click();
  await openTransformationJob(page, 'quantization', job.id);
  await page.getByRole('button', { name: 'Replay recorded observations', exact: true }).click();
  await expect(page.getByRole('region', { name: 'CPU observation replay', exact: true })).toBeVisible();
}

const packedProfile = { id: 'packed-cpu', label: 'Generated packed CPU profile', architectures: ['act'], experimental: true, task_object: 'cup', policy_runtime: 'packed-act-cpu', policy_device: 'cpu', policy_formats: ['firebird_quant'], provider: 'gcp', accelerators: ['L4'] };
async function simulationContinuationFixture(page: Page) {
  const state = await fixture(page);
  state.profiles = [{ id: 'cuda-first', label: 'Generated CUDA profile', architectures: ['act', 'smolvla'], experimental: true, task_object: 'cup' }, packedProfile];
  const job = makeJob('quant-job', quantRequest()); completeQuant(job);
  const first = { ...structuredClone(packed), label: 'Same generated package' };
  const target = { ...structuredClone(packed), id: 'quantized:second', label: first.label, manifest_sha256: 'd'.repeat(64) };
  job.result!.artifacts = [first, target]; state.jobs.push(job); state.artifacts.push(structuredClone(first), structuredClone(target));
  await page.reload(); await expect(page.getByLabel('Current project')).toHaveAttribute('data-project-id', 'alpha');
  await page.getByRole('link', { name: 'Quantize', exact: true }).click();
  await openTransformationJob(page, 'quantization', job.id);
  const continueButton = page.locator(`[data-artifact-id="${target.id}"]`).getByRole('button', { name: 'Prepare simulation', exact: true });
  await expect(continueButton).toBeEnabled();
  return { ...state, target, first, continueButton };
}

test('exact second packed package prepares simulation with explicit profile and paid consent only', async ({ page }) => {
  const state = await simulationContinuationFixture(page);
  await state.continueButton.click();
  await expect(page.getByRole('region', { name: 'Package from quantization', exact: true })).toContainText(state.target.id);
  const profiles = page.getByRole('radiogroup', { name: 'Isaac profile', exact: true });
  const policies = page.getByRole('radiogroup', { name: 'Native policy', exact: true });
  await expect(profiles.locator('input:checked')).toHaveCount(0);
  await expect(policies.locator('input:checked')).toHaveCount(0);
  const consent = page.getByRole('checkbox', { name: /experimental, paid cloud rollout/ });
  await expect(consent).toBeDisabled();
  await profiles.getByRole('radio', { name: packedProfile.label }).check();
  await expect(policies.locator('input:checked')).toHaveValue(state.target.id);
  await expect(consent).not.toBeChecked();
  await expect(page.getByText('Paid L4 simulator + CPU policy worker. Timeout is not a spending cap.', { exact: true })).toBeVisible();
  await page.getByLabel('Simulation timeout (seconds)').fill('600');
  const launch = page.getByRole('button', { name: 'Start experimental simulation', exact: true });
  await expect(launch).toBeDisabled(); expect(state.mutations).toEqual([]);
  await consent.check(); await launch.dblclick();
  await expect(page.getByRole('article', { name: 'Native simulation job details' })).toHaveAttribute('data-job-id', 'simulation-job');
  expect(state.mutations).toEqual([{ path: '/api/v1/projects/alpha/policy-jobs', body: { operation: 'policy.run', runtime_id: packedProfile.id, artifact_id: state.target.id, simulation: { profile_id: packedProfile.id, experimental: true }, timeout_seconds: 600 } }]);
});

for (const fault of ['missing', 'manifest', 'model', 'foreign'] as const) test(`simulation continuation refuses ${fault} source and never chooses a replacement`, async ({ page }) => {
  const state = await simulationContinuationFixture(page);
  const index = state.artifacts.findIndex(item => item.id === state.target.id);
  if (fault === 'missing') state.artifacts.splice(index, 1);
  else if (fault === 'manifest') state.artifacts[index].manifest_sha256 = 'e'.repeat(64);
  else if (fault === 'model') state.artifacts[index].metadata = { ...state.artifacts[index].metadata, model_id: `sha256:${'e'.repeat(64)}` };
  else state.artifacts[index].project_id = 'beta';
  await state.continueButton.click(); await page.getByRole('button', { name: 'Refresh simulation jobs', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Package from quantization', exact: true })).toContainText('missing or its recorded identity changed');
  await page.getByRole('radio', { name: packedProfile.label, exact: true }).check();
  await expect(page.getByRole('radiogroup', { name: 'Native policy', exact: true }).locator('input:checked')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Start experimental simulation', exact: true })).toBeDisabled();
  expect(state.mutations).toEqual([]);
});

test('late simulation package cannot override a manual policy or restore paid consent', async ({ page }) => {
  const state = await simulationContinuationFixture(page);
  state.artifacts.splice(state.artifacts.findIndex(item => item.id === state.target.id), 1);
  await state.continueButton.click(); await page.getByRole('button', { name: 'Refresh simulation jobs', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Package from quantization', exact: true })).toContainText('missing or its recorded identity changed');
  await page.getByRole('radio', { name: packedProfile.label, exact: true }).check();
  const policies = page.getByRole('radiogroup', { name: 'Native policy', exact: true });
  await policies.locator(`input[value="${state.first.id}"]`).check();
  state.artifacts.push(structuredClone(state.target));
  await page.getByRole('button', { name: 'Refresh simulation jobs', exact: true }).click();
  await expect(policies.locator(`input[value="${state.target.id}"]`)).toHaveCount(1);
  await expect(policies.locator('input:checked')).toHaveValue(state.first.id);
  await expect(page.getByRole('region', { name: 'Package from quantization', exact: true })).toHaveCount(0);
  await expect(page.getByRole('checkbox', { name: /experimental, paid cloud rollout/ })).not.toBeChecked();
  expect(state.mutations).toEqual([]);
});

test('a changed handed-off manifest clears consent and blocks the previously selected policy', async ({ page }) => {
  const state = await simulationContinuationFixture(page);
  await state.continueButton.click();
  await page.getByRole('radio', { name: packedProfile.label, exact: true }).check();
  const consent = page.getByRole('checkbox', { name: /experimental, paid cloud rollout/ });
  await consent.check();
  await expect(page.getByRole('button', { name: 'Start experimental simulation', exact: true })).toBeEnabled();
  state.artifacts.find(item => item.id === state.target.id)!.manifest_sha256 = 'e'.repeat(64);
  await page.getByRole('button', { name: 'Refresh simulation jobs', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Package from quantization', exact: true })).toContainText('missing or its recorded identity changed');
  await expect(page.getByRole('radiogroup', { name: 'Native policy', exact: true }).locator('input:checked')).toHaveCount(0);
  await expect(consent).not.toBeChecked(); await expect(consent).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Start experimental simulation', exact: true })).toBeDisabled();
  expect(state.mutations).toEqual([]);
});

test('manual recording selection consumes simulation handoff and retains its exact video and download', async ({ page }) => {
  const state = await simulationContinuationFixture(page);
  const saved = makeJob('saved-cup-rollout', { operation: 'policy.run', runtime_id: packedProfile.id, artifact_id: state.first.id, simulation: { profile_id: packedProfile.id, experimental: true }, timeout_seconds: 600 });
  saved.status = 'succeeded'; saved.stage = 'simulation';
  saved.result = { artifacts: [artifact('saved-cup-record', 'simulation_record', {}, 'alpha', saved.id)], reports: [{ stage: 'simulation', artifacts: [{ path: 'artifacts/outputs/video.mp4' }] }] };
  state.jobs.push(saved);
  // Hold generated media transport so this checks the owned player route, not invented playback evidence.
  let release!: () => void; const mediaGate = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/api/v1/jobs/saved-cup-rollout/simulation-media/video', async route => { await mediaGate; await route.fulfill({ status: 200, contentType: 'video/mp4', body: Buffer.alloc(0) }); });
  try {
    await state.continueButton.click(); await page.getByRole('button', { name: 'Refresh simulation jobs', exact: true }).click();
    await page.getByLabel('Saved simulation job', { exact: true }).selectOption(saved.id);
    await expect(page.getByRole('article', { name: 'Native simulation job details', exact: true })).toHaveAttribute('data-job-id', saved.id);
    await expect(page.getByLabel('Recorded cup rollout', { exact: true })).toHaveAttribute('src', '/api/v1/jobs/saved-cup-rollout/simulation-media/video');
    await expect(page.getByRole('link', { name: 'Download simulation record', exact: true })).toHaveAttribute('href', '/api/v1/projects/alpha/artifacts/saved-cup-record/download');
    await page.getByText('Prepare another run', { exact: true }).click();
    await page.getByRole('button', { name: 'Refresh simulation jobs', exact: true }).click();
    await expect(page.getByRole('region', { name: 'Package from quantization', exact: true })).toHaveCount(0);
    await expect(page.getByRole('radiogroup', { name: 'Isaac profile', exact: true }).locator('input:checked')).toHaveCount(0);
    await expect(page.getByRole('article', { name: 'Native simulation job details', exact: true })).toHaveAttribute('data-job-id', saved.id);
    expect(state.mutations).toEqual([]);
  } finally { release(); }
});

test('simulation continuation preserves uncertainty and clears on explicit project or mode change', async ({ page }) => {
  const state = await simulationContinuationFixture(page);
  const key = 'firebird:job-attempt:policy.run.simulation:alpha';
  const saved = JSON.stringify({ state: 'uncertain', message: 'Earlier rollout outcome remains unknown.' });
  await page.evaluate(({ key, saved }) => sessionStorage.setItem(key, saved), { key, saved });
  await state.continueButton.click();
  await expect(page.getByRole('region', { name: 'Native simulation recovery', exact: true })).toContainText('Earlier rollout outcome remains unknown.');
  await expect(page.getByRole('radio', { name: packedProfile.label, exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'I checked the jobs; allow a new request', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Replay observations', exact: true }).click();
  await page.getByRole('button', { name: '3D simulation', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Package from quantization', exact: true })).toHaveCount(0);
  expect(await page.evaluate(key => sessionStorage.getItem(key), key)).toBe(saved);
  await selectProject(page, 'beta');
  await page.getByRole('button', { name: '3D simulation', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Package from quantization', exact: true })).toHaveCount(0);
  await expect(page.getByRole('radiogroup', { name: 'Native policy', exact: true }).locator('input:checked')).toHaveCount(0);
  expect(state.mutations).toEqual([]);
});

for (const fault of ['missing-profile', 'profile-error'] as const) test(`packed result keeps replay and download when ${fault} blocks simulation`, async ({ page }) => {
  const state = await simulationContinuationFixture(page);
  // Mutate the fixture's shared arrays/route state, not an application setting.
  if (fault === 'missing-profile') state.profiles.splice(1, 1);
  else await page.route('**/api/v1/simulation-options', route => route.fulfill({ status: 503, json: { detail: 'Generated profile outage' } }));
  await page.getByRole('link', { name: 'Run', exact: true }).click();
  await page.getByRole('button', { name: '3D simulation', exact: true }).click();
  await page.getByRole('button', { name: 'Refresh simulation jobs', exact: true }).click();
  await page.getByRole('link', { name: 'Quantize', exact: true }).click();
  await openTransformationJob(page, 'quantization', 'quant-job');
  const result = page.locator(`[data-artifact-id="${state.target.id}"]`);
  await expect(result.getByRole('button', { name: 'Prepare simulation', exact: true })).toBeDisabled();
  await expect(result).toContainText(fault === 'missing-profile' ? 'No packed ACT simulation profile is configured.' : 'Simulation profile availability is unknown.');
  await expect(result.getByRole('button', { name: 'Replay recorded observations', exact: true })).toBeEnabled();
  await expect(result.getByRole('link', { name: 'Download INT8 package', exact: true })).toHaveAttribute('href', `/api/v1/projects/alpha/artifacts/${encodeURIComponent(state.target.id)}/download`);
  expect(state.mutations).toEqual([]);
});

test('explicit Distill to Quantize to Replay carries exact artifacts without submitting on navigation', async ({ page }) => {
  const state = await fixture(page);
  await page.getByRole('link', { name: 'Distill', exact: true }).click();
  await openSavedAct(page);
  await page.getByRole('group', { name: 'Teacher', exact: true }).locator(`input[value="${teacher.id}"]`).check();
  await page.getByRole('group', { name: 'Dataset', exact: true }).locator('input[value="alpha-data"]').check();
  for (const [label, value] of [['Training episodes', '0, 1'], ['Validation episodes', '2, 3'], ['Final episodes', '4, 5'], ['Six coordinate units', units.join(', ')]]) await page.getByLabel(label, { exact: true }).fill(value);
  await page.getByRole('checkbox', { name: 'This snapshot contains generated test observations.' }).check();
  await page.getByRole('checkbox', { name: /I verified that the dataset/ }).check();
  await page.getByRole('button', { name: 'Train ACT256 student', exact: true }).click();
  await expect(page.getByRole('article', { name: 'Distillation job details' })).toHaveAttribute('data-job-id', 'student-job');
  expect(state.mutations).toEqual([{ path: '/api/v1/projects/alpha/policy-jobs', body: studentRequest() }]);
  completeStudent(state.jobs.find(item => item.id === 'student-job') as ReturnType<typeof makeJob>); state.artifacts.push(student);
  await page.getByRole('button', { name: 'Refresh distillation jobs', exact: true }).click();
  await page.getByRole('button', { name: 'Open ACT quantization', exact: true }).click();
  await expect(selectedQuantizationModel(page)).toHaveValue(student.id); expect(state.mutations).toHaveLength(1);
  await page.getByRole('button', { name: 'Create ACT quantized package', exact: true }).click();
  await expect(page.getByRole('article', { name: 'ACT quantization job details' })).toHaveAttribute('data-job-id', 'quant-job');
  expect(state.mutations[1]).toEqual({ path: '/api/v1/projects/alpha/policy-jobs', body: quantRequest() });
  completeQuant(state.jobs.find(item => item.id === 'quant-job') as ReturnType<typeof makeJob>); state.artifacts.push(packed);
  await page.getByRole('button', { name: 'Refresh ACT quantization jobs', exact: true }).click();
  await page.getByRole('button', { name: 'Replay recorded observations', exact: true }).click();
  await expect(policyChoices(page, 'Packed ACT policy').locator('input:checked')).toHaveValue(packed.id); expect(state.mutations).toHaveLength(2);
  const submit = page.getByRole('button', { name: 'Run CPU observation replay', exact: true }); await expect(submit).toBeDisabled();
  await expect(page.getByRole('radiogroup', { name: 'Observation dataset', exact: true }).locator('input:checked')).toHaveCount(0);
  await page.getByRole('radiogroup', { name: 'Observation dataset', exact: true }).locator('input[value="alpha-data"]').check();
  await page.getByLabel('Episode and frame pairs', { exact: true }).fill('0:3, 2:1');
  await page.getByLabel('Six replay coordinate units', { exact: true }).fill(units.join(', '));
  await page.getByRole('radio', { name: 'Generated test observations', exact: true }).check(); await expect(submit).toBeDisabled();
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
  const picker = policyChoices(page, label); await expect(picker.locator(`input[value="${manual}"]`)).toHaveCount(1); await expect(picker.locator('input:checked')).toHaveCount(0);
  await picker.locator(`input[value="${manual}"]`).check(); state.artifacts.push(incoming);
  await page.getByRole('button', { name: target === 'quantization' ? 'Refresh ACT quantization jobs' : 'Refresh replay jobs', exact: true }).click();
  await expect(picker.locator(`input[value="${incoming.id}"]`)).toHaveCount(1); await expect(picker.locator('input:checked')).toHaveValue(manual); expect(state.mutations).toHaveLength(0);
});

for (const target of ['quantization', 'replay'] as const) test(`${target} handoff does not leak into another project`, async ({ page }) => {
  const state = await fixture(page); state.artifacts.push(student, packed);
  if (target === 'quantization') await openStudent(page, state); else await openPacked(page, state);
  const label = target === 'quantization' ? 'ACT inference policy' : 'Packed ACT policy', expected = target === 'quantization' ? student.id : packed.id;
  await expect(target === 'quantization' ? selectedQuantizationModel(page) : policyChoices(page, label).locator('input:checked')).toHaveValue(expected);
  await selectProject(page, 'beta');
  if (target === 'quantization') {
    await expect(page.getByRole('region', { name: 'Your quantization models' }).locator('button[aria-pressed="true"]')).toHaveCount(0);
    await expect(page.getByRole('button', { name: `Choose Generated ${expected} · ${expected}`, exact: true })).toHaveCount(0);
    await chooseTransformationModel(page, 'quantization', 'Choose Generated beta-float · beta-float');
    await expect(selectedQuantizationModel(page)).toHaveValue('beta-float');
    await expect(quantizationModelOption(page, expected)).toHaveCount(0);
    expect(state.mutations).toHaveLength(0); return;
  }
  const choice = page.getByRole('button', { name: target === 'quantization' ? 'ACT' : 'Replay observations', exact: true });
  await expect(choice).toHaveAttribute('aria-pressed', 'false');
  await expect(page.getByRole('article', { name: target === 'quantization' ? 'ACT quantization job details' : 'Observation replay details', exact: true })).toHaveCount(0);
  await choice.click();
  const picker = policyChoices(page, label); await expect(picker.locator('input:checked')).toHaveCount(0); await expect(picker.locator(`input[value="${expected}"]`)).toHaveCount(0);
  await expect(page.getByRole('button', { name: target === 'quantization' ? 'Create ACT quantized package' : 'Run CPU observation replay', exact: true })).toBeDisabled();
  await expect(page.getByRole('article', { name: target === 'quantization' ? 'ACT quantization job details' : 'Observation replay details', exact: true })).toHaveCount(0); expect(state.mutations).toHaveLength(0);
});


test('owned saved history restores ACT quantization and exact replay without mutation', async ({ page }) => {
  const state = await fixture(page);
  const quant = makeJob('quant-job', quantRequest()); completeQuant(quant); state.jobs.push(quant); state.artifacts.push(student, packed);
  const replay = savedReplay('replay-job');
  replay.result = { artifacts: [artifact('replay:record', 'native_run_record', { recipe: 'native-observation-replay-v1', model_id: model }, 'alpha', replay.id)], reports: [{ operation: 'policy.run', stage: 'native_replay', mode: 'independent_observation_replay', device: 'cpu', source_artifact_id: packed.id, dataset_job_id: 'alpha-data', observation_source: { kind: 'generated_fixture' }, model_id: model, observations: 1, action_shape: [100, 6], reset_repeat_exact: true, server_closed: true, task_success: null, quality_verified: false, calibration_verified: false, speedup_verified: false, isaac_runtime_verified: false, elapsed_seconds: 2 }] };
  state.jobs.push(replay);
  await page.reload(); await expect(page.getByLabel('Current project')).toHaveAttribute('data-project-id', 'alpha');
  await page.getByRole('link', { name: 'Quantize', exact: true }).click();
  await openTransformationJob(page, 'quantization', 'quant-job');
  await expect(page.getByRole('article', { name: 'ACT quantization job details' })).toHaveAttribute('data-job-id', 'quant-job');
  await page.getByRole('link', { name: 'Run', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Replay observations', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('article', { name: 'Observation replay details' })).toHaveAttribute('data-job-id', replay.id);
  await expect(page.getByRole('region', { name: 'Predicted action chunks' }).getByRole('img')).toHaveCount(6);
  await expect(page.getByRole('link', { name: 'Download verified replay record', exact: true })).toHaveAttribute('href', '/api/v1/projects/alpha/artifacts/replay%3Arecord/download');
  expect(state.mutations).toHaveLength(0);
});

test('engine history excludes replay while retaining an actual engine job', async ({ page }) => {
  const state = await fixture(page);
  const replay = makeJob('saved-replay', { operation: 'policy.run', runtime_id: 'replay-cpu', native_replay: { adapter: 'act-packed-observation-v1' } }); replay.status = 'succeeded';
  const engine = makeJob('saved-engine', { operation: 'policy.run', runtime_id: 'engine-cpu', artifact_id: 'gguf', evaluation: { mode: 'engine' } }); engine.status = 'succeeded'; state.jobs.push(replay, engine);
  await page.reload(); await expect(page.getByLabel('Current project')).toHaveAttribute('data-project-id', 'alpha');
  await page.getByRole('link', { name: 'Run', exact: true }).click();
  await page.getByRole('button', { name: 'Check inference', exact: true }).click();
  await expect(page.locator('.job-history-entry[data-job-id="saved-engine"]')).toBeVisible();
  await expect(page.locator('.job-history-entry[data-job-id="saved-replay"]')).toHaveCount(0);
  expect(state.mutations).toHaveLength(0);
});

function savedReplay(id: string, status = 'succeeded', projectId = 'alpha') {
  return { ...makeJob(id, { operation: 'policy.run', runtime_id: 'replay-cpu', artifact_id: packed.id, dataset_job_id: 'alpha-data', timeout_seconds: 600,
    native_replay: { adapter: 'act-packed-observation-v1', selection: [{ episode_index: 0, frame_index: 3 }], units, coordinate_attestation: 'generated_fixture' } }), status, project_id: projectId };
}
async function reloadProject(page: Page) { await page.reload(); await expect(page.getByLabel('Current project')).toHaveAttribute('data-project-id', 'alpha'); }

for (const configured of ['none', 'native', 'engine', 'simulation'] as const) test(`${configured} capabilities do not choose a model or workflow for a project without saved policy jobs`, async ({ page }) => {
  const state = await fixture(page);
  state.runtimes = configured === 'native' ? [...runtimes] : configured === 'engine' ? [{ id: 'engine', label: 'Generated engine', execution: 'native', provider: 'local', device: 'cpu', enabled: true, launchable: true, run: true, engine_evaluation: true, training: false, simulation: true }] : [];
  state.profiles = configured === 'simulation' ? [{ id: 'cup-fixture', label: 'Generated cup profile', architectures: ['act'], experimental: true, task_object: 'cup' }] : [];
  await reloadProject(page);
  await page.getByRole('link', { name: 'Quantize', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Workflow selection status', exact: true })).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Your quantization models' }).locator('button[aria-pressed="true"]')).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Native ACT quantization', exact: true })).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Quantization jobs', exact: true })).toContainText('No quantization jobs yet.');
  await page.getByRole('link', { name: 'Run', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Workflow selection status', exact: true })).toHaveCount(0);
  for (const name of ['3D simulation', 'Replay observations', 'Check inference']) await expect(page.getByRole('button', { name, exact: true })).toHaveAttribute('aria-pressed', 'false');
  await expect(page.getByRole('region', { name: 'CPU observation replay', exact: true })).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Native Isaac simulation', exact: true })).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Run jobs', exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Replay observations', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Replay observations', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('button', { name: 'Run CPU observation replay', exact: true })).toBeDisabled();
  expect(state.mutations).toEqual([]);
});

for (const mode of ['quantize', 'replay'] as const) test(`explicitly opened active ${mode} can cancel only that exact job`, async ({ page }) => {
  const state = await fixture(page), job = mode === 'quantize' ? makeJob('active-quant', quantRequest()) : savedReplay('active-replay', 'running');
  state.jobs.push(job); await reloadProject(page);
  await page.getByRole('link', { name: mode === 'quantize' ? 'Quantize' : 'Run', exact: true }).click();
  if (mode === 'quantize') await openTransformationJob(page, 'quantization', job.id);
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
  await page.getByRole('link', { name: 'Run', exact: true }).click();
  await page.getByRole('button', { name: 'Check inference', exact: true }).click();
  const response = page.waitForResponse('**/api/v1/policy-options'); release(); await response;
  await expect(page.getByRole('button', { name: 'Check inference', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Replay observations', exact: true }).click();
  await page.getByLabel('Saved replay', { exact: true }).selectOption('replay-b');
  state.jobs.push({ ...savedReplay('newer'), created_at: '2026-09-28T12:00:00Z' }); state.runtimes = [];
  await page.getByRole('button', { name: 'Refresh replay jobs', exact: true }).click();
  await expect(page.getByLabel('Saved replay', { exact: true }).locator('option[value="newer"]')).toHaveCount(1);
  await expect(page.getByRole('article', { name: 'Observation replay details' })).toHaveAttribute('data-job-id', 'replay-b');
  await page.getByRole('link', { name: 'Dataset', exact: true }).click(); await page.getByRole('link', { name: 'Run', exact: true }).click();
  await expect(page.getByRole('article', { name: 'Observation replay details' })).toHaveAttribute('data-job-id', 'replay-b');
  expect(state.mutations).toHaveLength(0);
});

test('manual mode is project scoped and saved foreign jobs never become entry context', async ({ page }) => {
  const state = await fixture(page); state.jobs.push(savedReplay('alpha-replay')); await reloadProject(page);
  await page.getByRole('link', { name: 'Run', exact: true }).click(); await page.getByRole('button', { name: 'Check inference', exact: true }).click();
  await selectProject(page, 'beta');
  await expect(page.getByRole('region', { name: 'Workflow selection status', exact: true })).toHaveCount(0);
  for (const name of ['3D simulation', 'Replay observations', 'Check inference']) await expect(page.getByRole('button', { name, exact: true })).toHaveAttribute('aria-pressed', 'false');
  await expect(page.getByRole('article', { name: 'Observation replay details' })).toHaveCount(0);
  await selectProject(page, 'alpha');
  await expect(page.getByRole('button', { name: 'Check inference', exact: true })).toHaveAttribute('aria-pressed', 'true');
  expect(state.mutations).toHaveLength(0);
});

for (const mode of ['quantize', 'replay'] as const) test(`unknown ${mode} submission takes precedence over automatic history`, async ({ page }) => {
  const state = await fixture(page);
  const operation = mode === 'quantize' ? 'policy.quantize' : 'policy.run.replay';
  await page.evaluate(({ operation }) => sessionStorage.setItem(`firebird:job-attempt:${operation}:alpha`, JSON.stringify({ state: 'pending', message: 'Generated lost acknowledgement' })), { operation });
  state.runtimes = []; state.jobs.push(makeJob('other-mode', { operation: mode === 'quantize' ? 'policy.quantize' : 'policy.run', runtime_id: 'engine' })); await reloadProject(page);
  await page.getByRole('link', { name: mode === 'quantize' ? 'Quantize' : 'Run', exact: true }).click();
  if (mode === 'quantize') {
    await page.getByRole('button', { name: 'Review request', exact: true }).click();
    await expect(page.getByRole('region', { name: 'Native ACT quantization', exact: true })).toBeVisible();
  }
  else await expect(page.getByRole('button', { name: 'Replay observations', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByText(/An earlier request did not return a verified outcome/)).toBeVisible();
  await page.getByRole('button', { name: mode === 'quantize' ? 'Refresh ACT quantization jobs' : 'Refresh replay jobs', exact: true }).click();
  expect(state.mutations).toHaveLength(0);
  expect(await page.evaluate(operation => sessionStorage.getItem(`firebird:job-attempt:${operation}:alpha`), operation)).not.toBeNull();
});

test('unreadable request recovery pauses automatic selection without clearing stored evidence', async ({ page }) => {
  const state = await fixture(page);
  await page.addInitScript(() => { const original = Storage.prototype.getItem; Storage.prototype.getItem = function(key) { if (key.startsWith('firebird:job-attempt:')) throw new Error('Generated storage outage'); return original.call(this, key); }; });
  await reloadProject(page); await page.getByRole('link', { name: 'Run', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Workflow selection status' })).toContainText('Saved request recovery could not be read');
  await expect(page.getByRole('button', { name: 'Replay observations', exact: true })).toHaveAttribute('aria-pressed', 'false');
  await page.getByRole('button', { name: 'Replay observations', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Run CPU observation replay', exact: true })).toBeDisabled(); expect(state.mutations).toHaveLength(0);
});

test('Evaluate reports unavailable scoring and only offers configured observation replay', async ({ page }) => {
  const state = await fixture(page);
  await page.getByRole('link', { name: 'Evaluate', exact: true }).click();
  const purpose = page.getByRole('region', { name: 'Evaluation purpose' });
  await expect(purpose.locator('details')).not.toHaveAttribute('open');
  await expect(purpose.getByRole('status')).toBeVisible();
  await expect(purpose.getByRole('status')).toContainText('No engine evaluation target is configured.');
  await expect(purpose.getByRole('status')).toContainText('No LIBERO evaluation target is configured.');
  await expect(purpose).toContainText('scored ACT / Isaac evaluation is not configured');
  await expect(purpose.getByRole('button', { name: 'Open native Isaac Run' })).toHaveCount(0);
  await purpose.getByRole('button', { name: 'Open observation replay' }).click();
  await expect(page.getByRole('button', { name: 'Replay observations', exact: true })).toHaveAttribute('aria-pressed', 'true'); expect(state.mutations).toHaveLength(0);
});

test('query failures keep Evaluate availability unknown and do not choose an initial Run mode', async ({ page }) => {
  const state = await fixture(page); state.failOptions = true; state.failSimulation = true;
  await page.getByRole('link', { name: 'Evaluate', exact: true }).click();
  const purpose = page.getByRole('region', { name: 'Evaluation purpose' });
  await expect(purpose.locator('details')).not.toHaveAttribute('open');
  await expect(purpose.getByRole('alert').filter({ hasText: 'Availability is unknown.' })).toBeVisible();
  await expect(purpose.getByRole('alert').filter({ hasText: 'Isaac profile availability' })).toBeVisible();
  await expect(purpose).not.toContainText('No LIBERO evaluation target is configured.');
  await expect(purpose.getByRole('button', { name: 'Open observation replay' })).toHaveCount(0);
  await page.getByRole('link', { name: 'Run', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Workflow selection status' })).toContainText('could not be loaded');
  await expect(page.getByRole('button', { name: 'Check inference', exact: true })).toHaveAttribute('aria-pressed', 'false');
  state.failOptions = false; state.failSimulation = false; await page.getByRole('button', { name: 'Retry workflow context' }).click();
  await expect(page.getByRole('region', { name: 'Workflow selection status', exact: true })).toHaveCount(0);
  for (const name of ['3D simulation', 'Replay observations', 'Check inference']) await expect(page.getByRole('button', { name, exact: true })).toHaveAttribute('aria-pressed', 'false');
  expect(state.mutations).toHaveLength(0);
});

test('the engine preparation link returns to the saved model chooser', async ({ page }) => {
  const state = await fixture(page);
  await page.getByRole('link', { name: 'Quantize', exact: true }).click(); await openSavedAct(page);
  await page.getByRole('link', { name: 'Evaluate', exact: true }).click(); await page.getByRole('button', { name: 'New evaluation', exact: true }).click();
  await page.getByRole('button', { name: 'Go to quantization', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Quantization jobs', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Start a new quantization', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Your quantization models' })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Your quantization models' }).locator('button[aria-pressed="true"]')).toHaveCount(0); expect(state.mutations).toHaveLength(0);
});

test('saved simulation history opens once and preserves the manually selected exact job', async ({ page }) => {
  const state = await fixture(page);
  state.profiles = [{ id: 'cup-fixture', label: 'Generated cup profile', architectures: ['act', 'smolvla'], experimental: true, task_object: 'cup' }];
  for (const id of ['isaac-a', 'isaac-b']) state.jobs.push({ ...makeJob(id, { operation: 'policy.run', runtime_id: 'cup-fixture', simulation: { profile_id: 'cup-fixture', experimental: true } }), status: 'succeeded' });
  await reloadProject(page); await page.getByRole('link', { name: 'Run', exact: true }).click();
  await expect(page.getByRole('button', { name: '3D simulation', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('article', { name: 'Native simulation job details' })).toHaveAttribute('data-job-id', 'isaac-a');
  await page.getByLabel('Saved simulation job', { exact: true }).selectOption('isaac-b');
  await page.getByRole('link', { name: 'Dataset', exact: true }).click(); await page.getByRole('link', { name: 'Run', exact: true }).click();
  await expect(page.getByRole('article', { name: 'Native simulation job details' })).toHaveAttribute('data-job-id', 'isaac-b');
  expect(state.mutations).toHaveLength(0);
});

test('mixed configured workflows start unselected and preserve a deliberate native choice', async ({ page }) => {
  const state = await fixture(page);
  state.runtimes.push({ id: 'engine', label: 'Generated engine', execution: 'native', provider: 'local', device: 'cpu', enabled: true, launchable: true, run: true, engine_evaluation: true, training: false, simulation: true });
  state.profiles = [{ id: 'cup-fixture', label: 'Generated cup profile', architectures: ['act'], experimental: true, task_object: 'cup' }];
  await page.getByRole('link', { name: 'Run', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Workflow selection status', exact: true })).toHaveCount(0);
  for (const name of ['3D simulation', 'Replay observations', 'Check inference']) await expect(page.getByRole('button', { name, exact: true })).toHaveAttribute('aria-pressed', 'false');
  await page.getByRole('button', { name: 'Replay observations', exact: true }).click();
  await page.getByRole('link', { name: 'Evaluate', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Evaluation purpose' })).toContainText('A LIBERO target is configured');
  await page.getByRole('link', { name: 'Run', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Replay observations', exact: true })).toHaveAttribute('aria-pressed', 'true'); expect(state.mutations).toHaveLength(0);
});

test('a detected training worker is not offered for GGUF quantization or engine evaluation', async ({ page }) => {
  const state = await fixture(page);
  state.artifacts.push(artifact('smolvla-checkpoint', 'training_checkpoint', { architecture: 'smolvla' }, 'alpha', 'smol-source'));
  state.runtimes = [{ id: 'managed-local-smolvla-test', label: 'Detected GPU trainer', execution: 'native', provider: 'local', device: 'cuda', enabled: true, launchable: true, training: true, training_only: true, training_model_ids: ['smolvla'], simulation: false, run: false, engine_evaluation: false }];
  await reloadProject(page);
  await page.getByRole('link', { name: 'Quantize', exact: true }).click();
  await chooseTransformationModel(page, 'quantization', 'Choose Generated smolvla-checkpoint · smolvla-checkpoint');
  await expect(page.getByText('Connect a compatible worker in Compute settings.', { exact: true })).toBeVisible();
  await expect(page.getByRole('radio', { name: /Detected GPU trainer/ })).toHaveCount(0);
  await page.getByRole('link', { name: 'Evaluate', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Evaluation purpose' })).toContainText('No engine evaluation target is configured.');
  expect(state.mutations).toEqual([]);
});

for (const mode of ['distillation', 'quantization', 'replay'] as const) test(`${mode} keeps the remaining worker selectable after the chosen worker disappears`, async ({ page }) => {
  const state = await fixture(page);
  const primary = runtimes[mode === 'distillation' ? 0 : mode === 'quantization' ? 1 : 2];
  const alternate = { ...primary, id: `${primary.id}-alternate`, label: 'Alternate local worker' };
  state.runtimes.push(alternate);
  await page.getByRole('link', { name: mode === 'distillation' ? 'Distill' : mode === 'quantization' ? 'Quantize' : 'Run', exact: true }).click();
  if (mode === 'distillation') {
    await openSavedAct(page);
    await page.getByRole('group', { name: 'Teacher', exact: true }).locator(`input[value="${teacher.id}"]`).check();
    await page.getByRole('group', { name: 'Dataset', exact: true }).locator('input[value="alpha-data"]').check();
    for (const [label, value] of [['Training episodes', '0, 1'], ['Validation episodes', '2, 3'], ['Final episodes', '4, 5'], ['Six coordinate units', units.join(', ')]]) await page.getByLabel(label, { exact: true }).fill(value);
    await page.getByRole('checkbox', { name: /I verified that the dataset/ }).check();
  } else if (mode === 'quantization') {
    await openSavedAct(page);
    await selectQuantizationModel(page, teacher.id);
  } else {
    await page.getByRole('button', { name: 'Replay observations', exact: true }).click();
    await policyChoices(page, 'Packed ACT policy').locator('input[value="manual-packed"]').check();
    await page.getByRole('radiogroup', { name: 'Observation dataset', exact: true }).locator('input[value="alpha-data"]').check();
    await page.getByLabel('Episode and frame pairs', { exact: true }).fill('0:3');
    await page.getByLabel('Six replay coordinate units', { exact: true }).fill(units.join(', '));
    await page.getByRole('checkbox', { name: /I verified that these/ }).check();
  }
  const refreshButton = page.getByRole('button', { name: mode === 'distillation' ? 'Refresh distillation jobs' : mode === 'quantization' ? 'Refresh ACT quantization jobs' : 'Refresh replay jobs', exact: true });
  await refreshButton.click();
  const workers = mode === 'replay' ? page.getByRole('radiogroup', { name: 'Replay worker', exact: true }) : page.getByRole('group', { name: 'Compute', exact: true });
  const submit = page.getByRole('button', { name: mode === 'distillation' ? 'Train ACT256 student' : mode === 'quantization' ? 'Create ACT quantized package' : 'Run CPU observation replay', exact: true });
  await workers.getByRole('radio', { name: alternate.label, exact: true }).check();
  await expect(submit).toBeEnabled();

  state.runtimes = state.runtimes.filter(item => item.id !== alternate.id);
  await refreshButton.click();
  await expect(workers.getByRole('radio')).toHaveCount(1);
  const remaining = workers.getByRole('radio', { name: primary.label, exact: true });
  await expect(remaining).not.toBeChecked();
  await expect(remaining).toBeEnabled();
  await expect(submit).toBeDisabled();
  // The single-worker picker collapses again after this explicit choice.
  await remaining.click();
  await expect(submit).toBeEnabled();
  expect(state.mutations).toEqual([]);
});

test('initial entries ignore unrelated, foreign and unsupported specialized records', () => {
  const records = [
    dataset(),
    savedReplay('foreign-replay', 'running', 'beta'),
    { ...makeJob('foreign-quantization', quantRequest()), project_id: 'beta' },
    makeJob('unknown-replay', { operation: 'policy.run', runtime_id: 'future', native_replay: { adapter: 'unsupported' } }),
    makeJob('unknown-quantization', { operation: 'policy.quantize', runtime_id: 'future', native_quantization: { format: 'unsupported' } }),
  ] as unknown as Job[];
  expect(initialRunEntry('alpha', [])).toBeUndefined();
  expect(initialQuantizeEntry('alpha', [])).toBeUndefined();
  expect(initialRunEntry('alpha', records)).toBeUndefined();
  expect(initialQuantizeEntry('alpha', records)).toBeUndefined();
});

test('saved routing preserves active-job priority and exact native or engine identities', () => {
  const activeRun = { ...makeJob('owned-engine', { operation: 'policy.run', runtime_id: 'offline-engine' }), created_at: '2026-09-01T12:00:00Z' };
  const activeQuantize = { ...makeJob('owned-act', quantRequest()), created_at: '2026-09-01T12:00:00Z' };
  const latestRun = savedReplay('recent-replay');
  const latestQuantize = { ...makeJob('recent-gguf', { operation: 'policy.quantize', runtime_id: 'offline-engine' }), status: 'succeeded' };
  const records = [latestRun, latestQuantize, activeRun, activeQuantize, savedReplay('foreign-active', 'running', 'beta')] as unknown as Job[];
  expect(initialRunEntry('alpha', records)).toEqual({ mode: 'engine', jobId: activeRun.id, origin: 'automatic' });
  expect(initialQuantizeEntry('alpha', records)).toEqual({ mode: 'native', jobId: activeQuantize.id, origin: 'automatic' });
  expect(initialRunEntry('alpha', [latestRun] as unknown as Job[])).toEqual({ mode: 'replay', jobId: latestRun.id, origin: 'automatic' });
  expect(initialQuantizeEntry('alpha', [latestQuantize] as unknown as Job[])).toEqual({ mode: 'gguf', jobId: latestQuantize.id, origin: 'automatic' });
});

for (const mode of ['quantize', 'replay'] as const) test(`saved ${mode} history opens even when worker capability reads fail`, async ({ page }) => {
  const state = await fixture(page);
  const saved = mode === 'quantize' ? makeJob('offline-quantization', quantRequest()) : savedReplay('offline-replay', 'running');
  state.jobs.push(saved); state.failOptions = true; state.failSimulation = true;
  await reloadProject(page);
  await page.getByRole('link', { name: mode === 'quantize' ? 'Quantize' : 'Run', exact: true }).click();
  if (mode === 'quantize') {
    await openTransformationJob(page, 'quantization', saved.id);
    await expect(page.getByRole('region', { name: 'Native ACT quantization', exact: true })).toBeVisible();
  }
  else await expect(page.getByRole('button', { name: 'Replay observations', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('article', { name: mode === 'quantize' ? 'ACT quantization job details' : 'Observation replay details', exact: true })).toHaveAttribute('data-job-id', saved.id);
  expect(state.mutations).toEqual([]);
});

for (const failedRead of ['profiles', 'history'] as const) test(`Evaluate keeps ${failedRead} query failure visible while explanation stays collapsed`, async ({ page }) => {
  const state = await fixture(page);
  const path = failedRead === 'profiles' ? '**/api/v1/simulation-options' : '**/api/v1/projects/alpha/jobs';
  await page.route(path, route => route.fulfill({ status: 503, json: { detail: 'Generated review read failure' } }));
  await page.reload(); await expect(page.getByLabel('Current project')).toHaveAttribute('data-project-id', 'alpha');
  await page.getByRole('link', { name: 'Evaluate', exact: true }).click();
  const purpose = page.getByRole('region', { name: 'Evaluation purpose' });
  await expect(purpose.locator('details')).not.toHaveAttribute('open');
  await expect(purpose.getByRole('alert')).toHaveText(failedRead === 'profiles' ? 'Isaac profile availability could not be loaded.' : 'Saved workflow history could not be refreshed; previously received records may be stale.');
  await expect(purpose.getByRole('alert')).toBeVisible();
  expect(state.mutations).toEqual([]);
});


test('teacher eligibility preserves legacy timing and admits complete bounded horizon metadata', () => {
  const source = artifact('registered-teacher') as PolicyArtifact;
  expect(studentTeacher(source, 'alpha')).toBe(true);
  for (const prediction_horizon of [1, 32, 100, 1024]) expect(studentTeacher({ ...source, metadata: { ...source.metadata, prediction_horizon, execution_horizon: 1, temporal_contract_sha256: null } }, 'alpha')).toBe(true);
  for (const prediction_horizon of [32, 100, 101, 0, '100', null, [100], true]) {
    expect(studentTeacher({ ...source, metadata: { ...source.metadata, prediction_horizon } }, 'alpha')).toBe(false);
  }
  expect(studentTeacher(source, 'beta')).toBe(false);
  expect(studentTeacher({ ...source, metadata: { architecture: 'smolvla' } }, 'alpha')).toBe(false);
  expect(studentTeacher({ ...source, metadata: { ...source.metadata, storage: 'gcs' } }, 'alpha')).toBe(false);
});


test('ACT transform eligibility preserves null legacy imports and rejects incomplete or malformed simulator claims', () => {
  const source = artifact('contract-policy') as PolicyArtifact;
  for (const metadata of [{}, { control_contract: null }, { control_contract_sha256: null }, { control_contract: null, control_contract_sha256: null }, { checkpoint: { control_contract: null, control_contract_sha256: null } }]) {
    const legacy = { ...source, metadata: { ...source.metadata, ...metadata } };
    expect(hasSimulatorControlContract(legacy)).toBe(false);
    expect(studentTeacher(legacy, 'alpha')).toBe(true);
    expect(nativeQuantizationInput(legacy, 'alpha')).toBe(true);
  }
  for (const field of ['control_contract', 'control_contract_sha256']) for (const value of [{ kind: 'simulator_joint_position', schema_version: 1 }, 'f'.repeat(64), {}, [], '', 0, false]) {
    for (const claim of [{ [field]: value }, { control_contract: null, control_contract_sha256: null, checkpoint: { [field]: value } }]) {
      const guarded = { ...source, metadata: { ...source.metadata, ...claim } };
      expect(hasSimulatorControlContract(guarded)).toBe(true);
      expect(studentTeacher(guarded, 'alpha')).toBe(false);
      expect(nativeQuantizationInput(guarded, 'alpha')).toBe(false);
    }
  }
});
