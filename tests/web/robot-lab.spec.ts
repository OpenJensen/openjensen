import { expect, test, type Page } from '@playwright/test';
import { selectProject } from './project-controls';

const stamp = '2026-10-23T12:00:00Z';
const port = '/dev/cu.usbmodem-test';
const pixel = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jw1cAAAAASUVORK5CYII=', 'base64');
const joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'];
const values = (value: number) => Object.fromEntries(joints.map(name => [name, value]));
function initialStatus(calibrated = false): any {
  return { ok: true, connected: true, label: 'This Mac', version: '1', platform: 'Darwin', dependencies: { lerobot: { installed: false, version: null }, serial: { installed: true, version: '3.5' }, cv2: { installed: true, version: '4.12.0' }, scservo_sdk: { installed: true, version: '1.0.0' } }, capabilities: { assembled_setup: true, probe: true, camera: true, motor_setup: true, calibration: true, control: true, recording: true }, session: { mode: 'idle', robot_id: null, port: null, role: null, positions: {}, torque_enabled: false, watchdog_seconds: 2, error: null }, calibration: { phase: 'idle', mins: {}, maxes: {}, offsets: {}, path: null }, recording: { active: false, dataset_id: null, episode_active: false, frames: 0, episodes: 0, fps: 10, error: null }, assembled: { phase: 'idle', ids: [], selected_joint: null, detected_id: null, candidate_delta: null, mapping: {}, message: 'Start experimental assembled setup.', error: null, log: [], progress: null, calibration_required: false }, calibrations: calibrated ? [{ robot_id: 'my-so101', role: 'follower', path: 'private-calibration.json' }] : [], datasets: [] };
}
async function fixture(page: Page, calibrated = false) {
  const state = { status: initialStatus(calibrated), writes: [] as { path: string; body: any }[], errors: [] as string[], datasetReads: 0, cameraFailure: false, stopError: null as string | null };
  page.on('pageerror', error => state.errors.push(error.message));
  await page.route('**/api/v1/**', async route => {
    const request = route.request(), path = new URL(request.url()).pathname, body = request.method() === 'POST' ? request.postDataJSON() : undefined;
    if (request.method() === 'POST') state.writes.push({ path, body });
    if (path.startsWith('/api/v1/robot-lab/')) {
      const action = path.slice('/api/v1/robot-lab'.length);
      if (action === '/status') return route.fulfill({ json: state.status });
      if (action === '/devices') return route.fulfill({ json: { ports: [{ device: port, description: 'SO-101 controller', manufacturer: 'USB serial', serial_number: 'test-1' }, { device: '/dev/cu.usbmodem-leader', description: 'Leader controller', manufacturer: 'USB serial', serial_number: 'test-2' }], cameras: [{ index: 0, name: 'Robot camera', width: 640, height: 480 }], camera_error: null } });
      if (action === '/camera.jpg') return state.cameraFailure ? route.fulfill({ status: 503, json: { detail: 'Camera disconnected.' } }) : route.fulfill({ contentType: 'image/png', body: pixel });
      if (action === '/datasets') return route.fulfill({ json: { datasets: state.status.datasets } });
      if (action === '/probe') return route.fulfill({ json: { port, baudrate: 1000000, motors: [{ id: 1, model: 777, stable: true, responses: 3 }], ready: false, expected_ids: [1, 2, 3, 4, 5, 6], warning: 'A response at one ID cannot prove that only one physical motor is connected.' } });
      if (action === '/setup/motor') return route.fulfill({ json: { assigned: { joint: body.joint, id: joints.indexOf(body.joint) + 1, baudrate: 1000000 }, warning: 'Keep other motors disconnected.' } });
      if (action === '/setup/assembled/start') { state.status.session = { ...state.status.session, mode: 'setup', port: body.port, torque_enabled: false }; state.status.assembled = { ...state.status.assembled, phase: 'ready', ids: [11,12,13,14,15,16], message: 'Distinct addresses ready. Identify each joint by hand.' }; }
      else if (action === '/setup/assembled/observe') { state.status.assembled = { ...state.status.assembled, phase: 'observing', selected_joint: body.joint, detected_id: joints.indexOf(body.joint) + 11, candidate_delta: 180, message: 'Joint movement detected.' }; }
      else if (action === '/setup/assembled/accept') { state.status.assembled.mapping[body.joint] = body.motor_id; state.status.assembled.phase = Object.keys(state.status.assembled.mapping).length === 6 ? 'review' : 'ready'; state.status.assembled.selected_joint = null; state.status.assembled.detected_id = null; }
      else if (action === '/setup/assembled/finish') { state.status.assembled.phase = 'applying'; state.status.assembled.message = 'Writing and verifying canonical IDs.'; setTimeout(() => { state.status.assembled.phase = 'complete'; state.status.assembled.calibration_required = true; state.status.session.mode = 'idle'; }, 1000); }
      else if (action === '/setup/assembled/cancel') { state.status.assembled.phase = 'stopped'; state.status.session.mode = 'idle'; }
      else if (action === '/calibration/start') { state.status.session = { ...state.status.session, mode: 'calibration', port: body.port, robot_id: body.robot_id, role: body.role, positions: values(2048) }; state.status.calibration.phase = 'center'; }
      else if (action === '/calibration/center') { state.status.calibration = { ...state.status.calibration, phase: 'ranges', mins: values(700), maxes: values(3300), offsets: values(0) }; }
      else if (action === '/calibration/finish') { state.status.calibration.phase = 'saved'; state.status.assembled.calibration_required = false; state.status.session.mode = 'idle'; state.status.calibrations = [{ robot_id: 'my-so101', role: 'follower', path: 'private-calibration.json' }]; }
      else if (action === '/calibration/cancel' || action === '/stop') { if (action === '/stop') state.status.session.error = state.stopError; state.status.session.mode = 'idle'; state.status.session.torque_enabled = false; state.status.recording.episode_active = false; }
      else if (action === '/control/start') { state.status.session = { ...state.status.session, mode: body.mode, port: body.port, robot_id: body.robot_id, positions: values(0), torque_enabled: body.mode !== 'handguide' }; }
      else if (action === '/control/heartbeat') { /* Explicitly owned session only. */ }
      else if (action === '/control/jog') { state.status.session.positions[body.joint] += body.delta; }
      else if (action === '/recording/start') { state.status.recording = { active: true, dataset_id: body.dataset_id, episode_active: false, frames: 0, episodes: 0, fps: body.fps, error: null }; }
      else if (action === '/recording/episode/start') { state.status.recording.episode_active = true; state.status.recording.frames = 10; }
      else if (action === '/recording/episode/save') { state.status.recording.episode_active = false; state.status.recording.episodes++; state.status.recording.frames = 0; }
      else if (action === '/recording/episode/discard') { state.status.recording.episode_active = false; state.status.recording.frames = 0; }
      else if (action === '/recording/finish') { const current = state.status.recording; state.status.datasets = [{ id: current.dataset_id, path: 'private-local-dataset', episodes: current.episodes, frames: body.abandon_empty ? 0 : 10, fps: current.fps, task: 'Place the block in the tray.', format: 'lerobot', finalized: !body.abandon_empty, abandoned: !!body.abandon_empty, pending_frames: body.abandon_empty ? current.frames : 0 }]; state.status.recording = { ...current, active: false, episode_active: false, frames: 0 }; }
      else return route.fulfill({ status: 404, json: { detail: `Unexpected robot route ${action}` } });
      return route.fulfill({ json: state.status });
    }
    if (path.includes('/robot-lab/datasets/') && path.endsWith('/import')) return route.fulfill({ json: { id: 'library-entry', project_id: path.split('/')[4], status: 'ready' } });
    if (path === '/api/v1/health') return route.fulfill({ json: { status: 'ok', version: 'fixture' } });
    if (path === '/api/v1/projects') return route.fulfill({ json: ['alpha', 'beta'].map(id => ({ id, name: `Robot ${id}`, created_at: stamp })) });
    if (path === '/api/v1/policy-options') return route.fulfill({ json: { runtimes: [], sources: [], training_models: [], training_methods: [], default_training_method: 'full' } });
    if (path === '/api/v1/datasets') state.datasetReads++;
    return route.fulfill({ json: [] });
  });
  await page.goto('/robot-lab/');
  await expect(page.getByRole('heading', { level: 1, name: 'Robot Lab', exact: true })).toBeVisible();
  await expect(page.getByLabel('Robot USB port')).toHaveValue(port);
  return state;
}
const tab = (page: Page, title: string) => page.getByRole('navigation', { name: 'Robot setup steps' }).getByRole('button', { name: new RegExp(title) });
const writes = (state: Awaited<ReturnType<typeof fixture>>, path: string) => state.writes.filter(item => item.path.endsWith(path));
async function startManual(page: Page) {
  await tab(page, 'Teleoperate').click();
  await page.getByRole('checkbox', { name: /Arm secured; workspace clear/ }).check();
  await page.getByRole('button', { name: 'Enable on-screen controls', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Increase Shoulder pan', exact: true })).toBeEnabled();
}

test('Robot Lab navigation and device discovery issue no motor commands', async ({ page }, testInfo) => {
  const state = await fixture(page);
  await expect(page.getByText('This Mac connected', { exact: true })).toBeVisible();
  await expect(page.getByText('Some hardware tools need installation.', { exact: true })).toHaveCount(0);
  for (const title of ['Motor IDs', 'Calibrate', 'Teleoperate', 'Record', 'Connect']) await tab(page, title).click();
  await expect(page.getByRole('button', { name: 'Enable on-screen controls', exact: true })).not.toBeVisible();
  expect(state.writes).toEqual([]);
  await page.getByRole('button', { name: 'Start preview', exact: true }).click();
  await expect(page.getByAltText('Live view from the selected robot camera')).toBeVisible();
  state.cameraFailure = true;
  await page.getByRole('button', { name: 'Pause preview', exact: true }).click();
  await page.getByRole('button', { name: 'Start preview', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Retry camera', exact: true })).toBeVisible();
  await expect(page.locator('.robot-lab-camera-live')).toHaveCount(0);
  state.cameraFailure = false;
  await page.getByRole('button', { name: 'Retry camera', exact: true }).click();
  await expect(page.getByAltText('Live view from the selected robot camera')).toBeVisible();
  await page.evaluate(() => { if (document.activeElement instanceof HTMLElement) document.activeElement.blur(); window.scrollTo(0, 0); });
  await page.screenshot({ path: testInfo.outputPath('robot-lab-connect.png'), fullPage: true });
  expect(state.errors).toEqual([]);
});

test('motor IDs require a fresh isolation acknowledgement for every motor', async ({ page }) => {
  const state = await fixture(page);
  await tab(page, 'Motor IDs').click();
  await page.getByText('Advanced: configure one physically isolated motor', { exact: true }).click();
  const assign = page.getByRole('button', { name: 'Assign ID 6', exact: true });
  await expect(assign).toBeDisabled();
  await page.getByRole('checkbox', { name: /Only the gripper motor is connected/ }).check();
  await assign.click();
  await expect(page.getByRole('button', { name: 'Assign ID 5', exact: true })).toBeDisabled();
  expect(writes(state, '/setup/motor')).toEqual([{ path: '/api/v1/robot-lab/setup/motor', body: { port, joint: 'gripper', isolated: true, confirm: 'ASSIGN MOTOR ID' } }]);
  expect(writes(state, '/control/start')).toEqual([]);
});

test('calibration center and range save are explicit and use the official companion contract', async ({ page }) => {
  const state = await fixture(page);
  await tab(page, 'Calibrate').click();
  await page.getByRole('button', { name: 'Begin calibration', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Record middle position', exact: true })).toBeEnabled();
  await expect(page.getByRole('group', { name: 'Measured encoder values', exact: true })).toContainText('2048');
  await page.getByRole('button', { name: 'Record middle position', exact: true }).click();
  await expect(page.getByRole('columnheader', { name: 'Range', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Save calibration', exact: true }).click();
  await expect(page.getByText('my-so101 · follower', { exact: true })).toBeVisible();
  expect(writes(state, '/calibration/start')[0].body).toEqual({ port, robot_id: 'my-so101', role: 'follower', confirm: 'START CALIBRATION' });
  expect(writes(state, '/calibration/center')[0].body).toEqual({});
  expect(writes(state, '/calibration/finish')[0].body).toEqual({ confirm: 'SAVE CALIBRATION' });
  expect(writes(state, '/control/start')).toEqual([]);
});

test('manual control is explicit, jogs are bounded, and navigating away stops the arm', async ({ page }) => {
  const state = await fixture(page, true);
  await startManual(page);
  await page.getByLabel('Joint step size').selectOption('2');
  await page.getByRole('button', { name: 'Increase Shoulder pan', exact: true }).click();
  await expect(page.getByLabel('Shoulder pan position')).toHaveText('2.0°');
  expect(writes(state, '/control/start')[0].body).toEqual({ port, robot_id: 'my-so101', mode: 'manual', confirm: 'ENABLE MOTION' });
  expect(writes(state, '/control/jog')[0].body).toEqual({ joint: 'shoulder_pan', delta: 2 });
  await page.getByRole('navigation', { name: 'Policy lifecycle' }).getByRole('link', { name: 'Dashboard', exact: true }).click();
  await expect.poll(() => writes(state, '/stop').length).toBe(1);
});

test('episode recording saves and finalizes before importing into the selected project', async ({ page }, testInfo) => {
  const state = await fixture(page, true);
  await selectProject(page, 'beta');
  await startManual(page);
  await tab(page, 'Record').click();
  await page.getByLabel('Dataset name', { exact: true }).fill('block-demo');
  await page.getByLabel('Task instruction', { exact: true }).fill('Place the block in the tray.');
  await page.getByRole('button', { name: 'Create recording dataset', exact: true }).click();
  await page.getByRole('button', { name: 'Start episode', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Finish dataset', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Record increase Shoulder pan', exact: true }).click();
  await page.getByRole('button', { name: 'Save episode', exact: true }).click();
  await page.getByRole('button', { name: 'Finish dataset', exact: true }).click();
  await page.getByRole('button', { name: 'Add to project', exact: true }).click();
  await expect(page.getByText('block-demo is in this project’s dataset library.', { exact: true })).toBeVisible();
  expect(writes(state, '/recording/start')[0].body).toEqual({ dataset_id: 'block-demo', task: 'Place the block in the tray.', fps: 10, camera_index: 0 });
  expect(writes(state, '/import')[0].path).toBe('/api/v1/projects/beta/robot-lab/datasets/block-demo/import');
  expect(writes(state, '/recording/episode/save')).toHaveLength(1);
  await page.evaluate(() => { if (document.activeElement instanceof HTMLElement) document.activeElement.blur(); window.scrollTo(0, 0); });
  await page.screenshot({ path: testInfo.outputPath('robot-lab-record.png'), fullPage: true });
  expect(state.errors).toEqual([]);
});


test('unsaved recording can be explicitly closed after Stop while retained files stay unfinalized', async ({ page }) => {
  const state = await fixture(page, true);
  await startManual(page);
  await tab(page, 'Record').click();
  await page.getByLabel('Dataset name', { exact: true }).fill('empty-demo');
  await page.getByLabel('Task instruction', { exact: true }).fill('Place the block in the tray.');
  await page.getByRole('button', { name: 'Create recording dataset', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Finish dataset', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Start episode', exact: true }).click();
  await page.getByRole('button', { name: 'Stop arm and disable torque', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Close unsaved recording', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'Close unsaved recording', exact: true }).click();
  expect(writes(state, '/recording/finish')).toEqual([{ path: '/api/v1/robot-lab/recording/finish', body: { abandon_empty: true, confirm: 'ABANDON EMPTY RECORDING' } }]);
  const retained = page.getByRole('article').filter({ has: page.getByRole('heading', { name: 'empty-demo', exact: true }) });
  await expect(retained.getByRole('button', { name: 'Closed · files retained', exact: true })).toBeDisabled();
  await expect(retained.getByText('10 unsaved frames', { exact: true })).toBeVisible();
  expect(writes(state, '/import')).toEqual([]);
  await startManual(page);
  await tab(page, 'Record').click();
  await page.getByLabel('Dataset name', { exact: true }).fill('next-demo');
  await page.getByLabel('Recording frame rate', { exact: true }).selectOption('5');
  await page.getByRole('button', { name: 'Create recording dataset', exact: true }).click();
  expect(writes(state, '/recording/start').at(-1)?.body).toEqual({ dataset_id: 'next-demo', task: 'Place the block in the tray.', fps: 5, camera_index: 0 });
  expect(state.errors).toEqual([]);
});

test('offline companion cannot start hardware actions and does not retry mutations', async ({ page }) => {
  const state = await fixture(page, true);
  state.status = { connected: false, ok: false, label: 'This Mac', message: 'Companion is offline.' };
  await page.getByRole('button', { name: 'Check connection', exact: true }).click();
  await expect(page.getByText('Mac companion unavailable', { exact: true })).toBeVisible();
  await tab(page, 'Motor IDs').click();
  await page.getByText('Advanced: configure one physically isolated motor', { exact: true }).click();
  await page.getByRole('checkbox', { name: /Only the gripper motor is connected/ }).check();
  await expect(page.getByRole('button', { name: 'Assign ID 6', exact: true })).toBeDisabled();
  await tab(page, 'Teleoperate').click();
  await page.getByRole('checkbox', { name: /Arm secured; workspace clear/ }).check();
  await expect(page.getByRole('button', { name: 'Enable on-screen controls', exact: true })).toBeDisabled();
  expect(state.writes).toEqual([]);
});


test('hand guiding is an explicit torque-off session and leader mode needs a second calibrated arm', async ({ page }) => {
  const state = await fixture(page, true);
  await tab(page, 'Teleoperate').click();
  await page.getByRole('button', { name: 'Leader → follower', exact: false }).click();
  await page.getByRole('checkbox', { name: /Arm secured; workspace clear/ }).check();
  await expect(page.getByRole('button', { name: 'Start leader teleoperation', exact: true })).toBeDisabled();
  expect(writes(state, '/control/start')).toEqual([]);
  await page.getByRole('button', { name: 'Hand guiding', exact: false }).click();
  await page.getByRole('checkbox', { name: /Arm supported/ }).check();
  await page.getByRole('button', { name: 'Start hand guiding', exact: true }).click();
  await expect(page.getByText('Hand guiding is active. Torque is off.', { exact: true })).toBeVisible();
  expect(writes(state, '/control/start')[0].body).toEqual({ port, robot_id: 'my-so101', mode: 'handguide', confirm: 'ENABLE MOTION' });
  expect(state.status.session.torque_enabled).toBe(false);
  await expect.poll(() => writes(state, '/control/heartbeat').length).toBeGreaterThan(0);
  state.stopError = 'Stop attempted but motor acknowledgement failed. Keep supporting the arm.';
  await page.getByRole('button', { name: 'Stop arm and disable torque', exact: true }).click();
  await expect(page.locator('.robot-lab').getByRole('alert')).toContainText(state.stopError);
  await expect(page.getByText('Arm stopped. Torque is disabled.', { exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Start hand guiding', exact: true })).toBeVisible();
});


test('all six illustrated joints work by pointer and keyboard without idle hardware requests', async ({ page }, testInfo) => {
  const state = await fixture(page);
  await tab(page, 'Motor IDs').click();
  const guide = page.getByRole('region', { name: 'SO-101 joint illustration', exact: true });
  await expect(guide.getByRole('slider', { name: 'Illustrated joint angle', exact: true })).toBeVisible();
  await expect(guide).toHaveAttribute('data-motion', 'reduced');
  for (let index = 0; index < joints.length; index++) {
    await guide.getByRole('group', { name: 'Choose an illustrated joint', exact: true }).getByRole('button').nth(index).click();
    await expect(guide).toHaveAttribute('data-selected-joint', joints[index]);
    await expect(guide.getByRole('group', { name: 'Choose an illustrated joint', exact: true }).getByRole('button').nth(index)).toHaveAttribute('aria-pressed', 'true');
  }
  const gripper = guide.getByRole('button', { name: 'Joint 6: Gripper', exact: true });
  await gripper.focus(); await page.keyboard.press('ArrowLeft');
  await expect(guide).toHaveAttribute('data-selected-joint', 'wrist_roll');
  await page.keyboard.press('Home'); await expect(guide).toHaveAttribute('data-selected-joint', 'shoulder_pan');
  await page.keyboard.press('End'); await expect(guide).toHaveAttribute('data-selected-joint', 'gripper');
  const before = await guide.getAttribute('data-angle');
  await guide.getByRole('slider', { name: 'Illustrated joint angle', exact: true }).focus();
  await page.keyboard.press('End');
  await expect(guide).not.toHaveAttribute('data-angle', before!);
  await tab(page, 'Calibrate').click();
  await expect(page.getByRole('region', { name: 'SO-101 joint illustration', exact: true })).toHaveAttribute('data-selected-joint', 'gripper');
  await expect(page.getByRole('group', { name: 'Measured encoder values', exact: true })).toHaveCount(0);
  expect(state.writes).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
  await page.evaluate(() => { if (document.activeElement instanceof HTMLElement) document.activeElement.blur(); window.scrollTo(0,0); });
  await page.screenshot({ path: testInfo.outputPath('so101-joint-explorer.png'), fullPage: true });
});

test('offline joint illustration still selects and never invents measurements or sends mutations', async ({ page }) => {
  const state = await fixture(page);
  state.status = { ok: false, connected: false, message: 'Companion offline.' };
  await page.getByRole('button', { name: 'Check connection', exact: true }).click();
  await expect(page.getByText('Mac companion unavailable', { exact: true })).toBeVisible();
  await tab(page, 'Motor IDs').click();
  const guide = page.getByRole('region', { name: 'SO-101 joint illustration', exact: true });
  await guide.getByRole('button', { name: 'Joint 6: Gripper', exact: true }).click();
  await expect(guide.getByRole('heading', { name: 'Gripper', exact: true })).toBeVisible();
  await guide.getByRole('slider', { name: 'Illustrated joint angle', exact: true }).focus();
  await page.keyboard.press('End');
  await expect(page.getByRole('button', { name: 'Start assembled setup', exact: true })).toBeDisabled();
  expect(state.writes).toEqual([]);
});

test('motion preview is explicit, pauses, resets and ends without commanding hardware', async ({ page }) => {
  test.setTimeout(60_000);
  await page.clock.install({ time: new Date('2026-10-23T12:00:00Z') });
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  const state = await fixture(page);
  await tab(page, 'Motor IDs').click();
  const guide = page.getByRole('region', { name: 'SO-101 joint illustration', exact: true });
  await expect(guide).toHaveAttribute('data-motion', 'static');
  await expect(guide.locator('canvas')).not.toHaveAttribute('data-renderer', 'loading');
  await page.clock.pauseAt(new Date(await page.evaluate(() => Date.now() + 1_000)));
  await guide.getByRole('button', { name: 'Preview motion', exact: true }).click();
  await expect(guide).toHaveAttribute('data-motion', 'playing');
  await page.clock.runFor(64);
  await expect(guide).not.toHaveAttribute('data-angle', '0.00');
  await guide.getByRole('button', { name: 'Pause motion', exact: true }).click();
  await expect(guide).toHaveAttribute('data-motion', 'paused');
  const paused = await guide.getAttribute('data-angle');
  await page.clock.fastForward(120);
  await expect(guide).toHaveAttribute('data-angle', paused!);
  await guide.getByRole('button', { name: 'Reset', exact: true }).click();
  await expect(guide).toHaveAttribute('data-motion', 'static');
  await guide.getByRole('button', { name: 'Preview motion', exact: true }).click();
  await expect(guide).toHaveAttribute('data-motion', 'playing');
  await page.clock.runFor(32);
  await page.clock.fastForward(2_000);
  await expect(guide).toHaveAttribute('data-motion', 'static');
  await guide.getByRole('button', { name: 'Joint 1: Shoulder pan', exact: true }).click();
  await expect(guide).toHaveAttribute('data-motion', 'static');
  expect(state.writes).toEqual([]);
});

test('assembled setup prepares explicitly then picks start only encoder observation until mapping is confirmed', async ({ page }, testInfo) => {
  test.setTimeout(120_000);
  const state = await fixture(page, true);
  await tab(page, 'Motor IDs').click();
  const start = page.getByRole('button', { name: 'Start assembled setup', exact: true });
  await expect(start).toBeDisabled();
  const guide = page.getByRole('region', { name: 'SO-101 joint illustration', exact: true });
  await guide.getByRole('button', { name: 'Joint 6: Gripper', exact: true }).click();
  expect(state.writes).toEqual([]);
  const consent = page.getByRole('checkbox', { name: /Six STS3215 motors; arm supported/ });
  await consent.check();
  await page.getByLabel('Robot USB port').selectOption('/dev/cu.usbmodem-leader');
  await expect(consent).not.toBeChecked();
  await expect(start).toBeDisabled();
  await page.getByLabel('Robot USB port').selectOption(port);
  await consent.check();
  await start.click();
  expect(writes(state, '/setup/assembled/start')[0].body).toEqual({ port, supported: true, confirm: 'START ASSEMBLED SETUP' });
  for (let index = 5; index >= 0; index--) {
    const label = ['Shoulder pan','Shoulder lift','Elbow','Wrist flex','Wrist roll','Gripper'][index];
    const before = state.writes.length;
    await guide.getByRole('button', { name: `Joint ${index + 1}: ${label}`, exact: true }).click();
    await expect(page.getByText(`Movement detected at address ${index + 11}`, { exact: true })).toBeVisible();
    const selectedWrites = state.writes.slice(before).filter(item => !item.path.endsWith('/control/heartbeat'));
    expect(selectedWrites.map(item => item.path)).toEqual(['/api/v1/robot-lab/setup/assembled/observe']);
    expect(writes(state, '/setup/assembled/finish')).toEqual([]);
    if (index === 5) {
      await page.evaluate(() => { if (document.activeElement instanceof HTMLElement) document.activeElement.blur(); window.scrollTo(0,0); });
      await page.screenshot({ path: testInfo.outputPath('so101-assembled-detection.png'), fullPage: true });
    }
    await page.getByRole('button', { name: `Confirm ID ${index + 11} is the ${label.toLowerCase()}`, exact: true }).click();
    if (index === 5) {
      const observations = writes(state, '/setup/assembled/observe').length;
      await guide.getByRole('button', { name: 'Joint 6: Gripper', exact: true }).click();
      await expect(guide.getByText('Gripper is confirmed at address 16. Select the next joint.', { exact: true })).toBeVisible();
      expect(writes(state, '/setup/assembled/observe')).toHaveLength(observations);
    }
  }
  await expect(page.getByRole('button', { name: 'Save motor IDs 1–6', exact: true })).toBeEnabled();
  const heartbeatBeforeSave = writes(state, '/control/heartbeat').length;
  await page.getByRole('button', { name: 'Save motor IDs 1–6', exact: true }).click();
  await expect.poll(() => writes(state, '/control/heartbeat').length).toBeGreaterThan(heartbeatBeforeSave);
  expect(writes(state, '/setup/assembled/accept')).toHaveLength(6);
  expect(writes(state, '/setup/assembled/finish')[0].body).toEqual({ confirm: 'SAVE MOTOR IDS' });
  expect(writes(state, '/control/start')).toEqual([]);
  await expect(page.getByText('IDs saved', { exact: true })).toBeVisible();
  await tab(page, 'Teleoperate').click();
  await expect(page.getByRole('button', { name: 'Enable on-screen controls', exact: true })).toBeDisabled();
  state.status.assembled = { ...state.status.assembled, phase: 'stopped', ids: [3], mapping: {},
    message: 'Setup stopped. Partial motor IDs are preserved; restart setup to scan and resume.',
    log: ['Separating factory IDs: round 8/12', 'Setup stopped with responding ID 3.'] };
  await page.getByRole('button', { name: 'Check connection', exact: true }).click();
  await tab(page, 'Motor IDs').click();
  await expect(page.getByText(state.status.assembled.message, { exact: true })).toBeVisible();
  await page.getByText('Setup details', { exact: true }).click();
  await expect(page.getByText('Setup stopped with responding ID 3.', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Start assembled setup', exact: true })).toBeDisabled();
  expect(writes(state, '/setup/assembled/start')).toHaveLength(1);
});
