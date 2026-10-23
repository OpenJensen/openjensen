import { expect, test, type Page } from '@playwright/test';
import { selectProject } from './project-controls';

const stamp = '2026-10-23T12:00:00Z';
const port = '/dev/cu.usbmodem-test';
const pixel = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jw1cAAAAASUVORK5CYII=', 'base64');
const joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'];
const values = (value: number) => Object.fromEntries(joints.map(name => [name, value]));
function initialStatus(calibrated = false): any {
  return { ok: true, connected: true, label: 'This Mac', version: '1', platform: 'Darwin', dependencies: { lerobot: { installed: false, version: null }, serial: { installed: true, version: '3.5' }, cv2: { installed: true, version: '4.12.0' }, scservo_sdk: { installed: true, version: '1.0.0' } }, capabilities: { probe: true, camera: true, motor_setup: true, calibration: true, control: true, recording: true }, session: { mode: 'idle', robot_id: null, port: null, role: null, positions: {}, torque_enabled: false, watchdog_seconds: 2, error: null }, calibration: { phase: 'idle', mins: {}, maxes: {}, offsets: {}, path: null }, recording: { active: false, dataset_id: null, episode_active: false, frames: 0, episodes: 0, fps: 10, error: null }, calibrations: calibrated ? [{ robot_id: 'my-so101', role: 'follower', path: 'private-calibration.json' }] : [], datasets: [] };
}
async function fixture(page: Page, calibrated = false) {
  const state = { status: initialStatus(calibrated), writes: [] as { path: string; body: any }[], errors: [] as string[], datasetReads: 0 };
  page.on('pageerror', error => state.errors.push(error.message));
  await page.route('**/api/v1/**', async route => {
    const request = route.request(), path = new URL(request.url()).pathname, body = request.method() === 'POST' ? request.postDataJSON() : undefined;
    if (request.method() === 'POST') state.writes.push({ path, body });
    if (path.startsWith('/api/v1/robot-lab/')) {
      const action = path.slice('/api/v1/robot-lab'.length);
      if (action === '/status') return route.fulfill({ json: state.status });
      if (action === '/devices') return route.fulfill({ json: { ports: [{ device: port, description: 'SO-101 controller', manufacturer: 'USB serial', serial_number: 'test-1' }, { device: '/dev/cu.usbmodem-leader', description: 'Leader controller', manufacturer: 'USB serial', serial_number: 'test-2' }], cameras: [{ index: 0, name: 'Robot camera', width: 640, height: 480 }], camera_error: null } });
      if (action === '/camera.jpg') return route.fulfill({ contentType: 'image/png', body: pixel });
      if (action === '/datasets') return route.fulfill({ json: { datasets: state.status.datasets } });
      if (action === '/probe') return route.fulfill({ json: { port, baudrate: 1000000, motors: [{ id: 1, model: 777, stable: true, responses: 3 }], ready: false, expected_ids: [1, 2, 3, 4, 5, 6], warning: 'A response at one ID cannot prove that only one physical motor is connected.' } });
      if (action === '/setup/motor') return route.fulfill({ json: { assigned: { joint: body.joint, id: joints.indexOf(body.joint) + 1, baudrate: 1000000 }, warning: 'Keep other motors disconnected.' } });
      if (action === '/calibration/start') { state.status.session = { ...state.status.session, mode: 'calibration', port: body.port, robot_id: body.robot_id, role: body.role, positions: values(2048) }; state.status.calibration.phase = 'center'; }
      else if (action === '/calibration/center') { state.status.calibration = { ...state.status.calibration, phase: 'ranges', mins: values(700), maxes: values(3300), offsets: values(0) }; }
      else if (action === '/calibration/finish') { state.status.calibration.phase = 'saved'; state.status.session.mode = 'idle'; state.status.calibrations = [{ robot_id: 'my-so101', role: 'follower', path: 'private-calibration.json' }]; }
      else if (action === '/calibration/cancel' || action === '/stop') { state.status.session.mode = 'idle'; state.status.session.torque_enabled = false; state.status.recording.episode_active = false; }
      else if (action === '/control/start') { state.status.session = { ...state.status.session, mode: body.mode, port: body.port, robot_id: body.robot_id, positions: values(0), torque_enabled: body.mode !== 'handguide' }; }
      else if (action === '/control/heartbeat') { /* Explicitly owned session only. */ }
      else if (action === '/control/jog') { state.status.session.positions[body.joint] += body.delta; }
      else if (action === '/recording/start') { state.status.recording = { active: true, dataset_id: body.dataset_id, episode_active: false, frames: 0, episodes: 0, fps: body.fps, error: null }; }
      else if (action === '/recording/episode/start') { state.status.recording.episode_active = true; state.status.recording.frames = 10; }
      else if (action === '/recording/episode/save') { state.status.recording.episode_active = false; state.status.recording.episodes++; state.status.recording.frames = 0; }
      else if (action === '/recording/episode/discard') { state.status.recording.episode_active = false; state.status.recording.frames = 0; }
      else if (action === '/recording/finish') { state.status.recording.active = false; state.status.datasets = [{ id: state.status.recording.dataset_id, path: 'private-local-dataset', episodes: state.status.recording.episodes, frames: 10, fps: state.status.recording.fps, task: 'Place the block in the tray.', format: 'lerobot', finalized: true }]; }
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
  await page.getByRole('checkbox', { name: /The arm is secured/ }).check();
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
  await page.evaluate(() => { if (document.activeElement instanceof HTMLElement) document.activeElement.blur(); window.scrollTo(0, 0); });
  await page.screenshot({ path: testInfo.outputPath('robot-lab-connect.png'), fullPage: true });
  expect(state.errors).toEqual([]);
});

test('motor IDs require a fresh isolation acknowledgement for every motor', async ({ page }) => {
  const state = await fixture(page);
  await tab(page, 'Motor IDs').click();
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


test('offline companion cannot start hardware actions and does not retry mutations', async ({ page }) => {
  const state = await fixture(page, true);
  state.status = { connected: false, ok: false, label: 'This Mac', message: 'Companion is offline.' };
  await page.getByRole('button', { name: 'Check connection', exact: true }).click();
  await expect(page.getByText('Mac companion unavailable', { exact: true })).toBeVisible();
  await tab(page, 'Motor IDs').click();
  await page.getByRole('checkbox', { name: /Only the gripper motor is connected/ }).check();
  await expect(page.getByRole('button', { name: 'Assign ID 6', exact: true })).toBeDisabled();
  await tab(page, 'Teleoperate').click();
  await page.getByRole('checkbox', { name: /The arm is secured/ }).check();
  await expect(page.getByRole('button', { name: 'Enable on-screen controls', exact: true })).toBeDisabled();
  expect(state.writes).toEqual([]);
});


test('hand guiding is an explicit torque-off session and leader mode needs a second calibrated arm', async ({ page }) => {
  const state = await fixture(page, true);
  await tab(page, 'Teleoperate').click();
  await page.getByRole('button', { name: 'Leader → follower', exact: false }).click();
  await page.getByRole('checkbox', { name: /The arm is secured/ }).check();
  await expect(page.getByRole('button', { name: 'Start leader teleoperation', exact: true })).toBeDisabled();
  expect(writes(state, '/control/start')).toEqual([]);
  await page.getByRole('button', { name: 'Guide by hand', exact: false }).click();
  await page.getByRole('checkbox', { name: /I’m supporting the arm/ }).check();
  await page.getByRole('button', { name: 'Start hand guiding', exact: true }).click();
  await expect(page.getByText('Hand guiding is active. Torque is off.', { exact: true })).toBeVisible();
  expect(writes(state, '/control/start')[0].body).toEqual({ port, robot_id: 'my-so101', mode: 'handguide', confirm: 'ENABLE MOTION' });
  expect(state.status.session.torque_enabled).toBe(false);
  await expect.poll(() => writes(state, '/control/heartbeat').length).toBeGreaterThan(0);
  await page.getByRole('button', { name: 'Stop arm and disable torque', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Start hand guiding', exact: true })).toBeVisible();
});
