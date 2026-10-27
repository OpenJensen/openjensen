import { fulfillApplicationUpdateRead } from './application-update-controls';
import { expect, test, type Page } from '@playwright/test';
import { selectProject } from './project-controls';
import { installedModules } from './module-fixtures';

const stamp = '2026-10-23T12:00:00Z';
const port = '/dev/cu.usbmodem-test';
const pixel = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jw1cAAAAASUVORK5CYII=',
  'base64',
);
const joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'];
const values = (value: number) => Object.fromEntries(joints.map((name) => [name, value]));
function initialStatus(calibrated = false): any {
  return {
    ok: true,
    connected: true,
    label: 'This Mac',
    version: '1',
    platform: 'Darwin',
    dependencies: {
      lerobot: { installed: false, version: null },
      serial: { installed: true, version: '3.5' },
      cv2: { installed: true, version: '4.12.0' },
      scservo_sdk: { installed: true, version: '1.0.0' },
    },
    capabilities: {
      lerobot_position_control: true,
      large_joint_steps: true,
      arm_readiness: true,
      middle_tolerance: true,
      joint_sliders: true,
      calibration_persistence: true,
      partial_control: true,
      move_to_middle: true,
      mapping_verification: true,
      assembled_setup: true,
      probe: true,
      camera: true,
      motor_setup: true,
      calibration: true,
      control: true,
      recording: true,
    },
    session: {
      mode: 'idle',
      robot_id: null,
      port: null,
      role: null,
      positions: {},
      torque_enabled: false,
      watchdog_seconds: 2,
      error: null,
    },
    calibration: { phase: 'idle', mins: {}, maxes: {}, offsets: {}, path: null },
    recording: { active: false, dataset_id: null, episode_active: false, frames: 0, episodes: 0, fps: 10, error: null },
    assembled: {
      phase: 'idle',
      ids: [],
      selected_joint: null,
      detected_id: null,
      candidate_delta: null,
      mapping: {},
      message: 'Start experimental assembled setup.',
      error: null,
      log: [],
      progress: null,
      calibration_required: false,
    },
    calibrations: calibrated ? [{ robot_id: 'my-so101', role: 'follower', path: 'private-calibration.json' }] : [],
    datasets: [],
  };
}
async function fixture(page: Page, calibrated = false, hostPlatform = 'Darwin') {
  const state = {
    status: initialStatus(calibrated),
    writes: [] as { path: string; body: any }[],
    errors: [] as string[],
    datasetReads: 0,
    cameraFailure: false,
    stopError: null as string | null,
    centerError: null as string | null,
    portsConnected: true,
    middleInProgress: false,
    encoderLag: false,
    jogDelay: 0,
    jogStaleResponse: false,
    robotDatasetReads: 0,
    heartbeatFailures: 0,
    statusFailures: false,
    dropCalibrationReply: false,
  };
  if (hostPlatform === 'Linux') {
    state.status.platform = 'Linux';
    state.status.label = 'Linux robot computer';
  }
  page.on('pageerror', (error) => state.errors.push(error.message));
  await page.route('**/api/v1/**', async (route) => {
    if (await fulfillApplicationUpdateRead(route)) return;
    const request = route.request(),
      path = new URL(request.url()).pathname,
      body = request.method() === 'POST' ? request.postDataJSON() : undefined;
    if (request.method() === 'POST') state.writes.push({ path, body });
    if (path.startsWith('/api/v1/robot-lab/')) {
      const action = path.slice('/api/v1/robot-lab'.length);
      if (action === '/status')
        return state.statusFailures ? route.abort('failed') : route.fulfill({ json: state.status });
      if (action === '/devices')
        return route.fulfill({
          json: {
            ports: state.portsConnected
              ? [
                  {
                    device: hostPlatform === 'Linux' ? '/dev/ttyACM0' : port,
                    description: 'SO-101 controller',
                    manufacturer: 'USB serial',
                    serial_number: 'test-1',
                  },
                  {
                    device: '/dev/cu.usbmodem-leader',
                    description: 'Leader controller',
                    manufacturer: 'USB serial',
                    serial_number: 'test-2',
                  },
                ]
              : [],
            cameras: [{ index: 0, name: 'Robot camera', width: 640, height: 480 }],
            camera_error: null,
          },
        });
      if (action === '/camera.jpg')
        return state.cameraFailure
          ? route.fulfill({ status: 503, json: { detail: 'Camera disconnected.' } })
          : route.fulfill({ contentType: 'image/png', body: pixel });
      if (action === '/datasets') {
        state.robotDatasetReads++;
        return route.fulfill({ json: { datasets: state.status.datasets } });
      }
      if (action === '/probe')
        return route.fulfill({
          json: {
            port,
            baudrate: 1000000,
            motors: [{ id: 1, model: 777, stable: true, responses: 3 }],
            ready: false,
            expected_ids: [1, 2, 3, 4, 5, 6],
            warning: 'A response at one ID cannot prove that only one physical motor is connected.',
          },
        });
      if (action === '/setup/motor')
        return route.fulfill({
          json: {
            assigned: { joint: body.joint, id: joints.indexOf(body.joint) + 1, baudrate: 1000000 },
            warning: 'Keep other motors disconnected.',
          },
        });
      if (action === '/mapping/confirm') {
        const checks = state.status.mapping_checks?.checks ?? {};
        state.status.mapping_checks = {
          port,
          revision: 'fixture-revision',
          error: null,
          checks: {
            ...checks,
            [body.joint]: { run_id: body.run_id, correct: body.correct, checked_at: Date.now() / 1000 },
          },
        };
      } else if (action === '/mapping/wiggle') {
        const verification = {
          phase: 'checking',
          port: body.port,
          joint: body.joint,
          motor_id: joints.indexOf(body.joint) + 1,
          ticks: null,
          motion_active: false,
          error: null,
          run_id: String(state.writes.length),
        };
        state.status.verification = verification;
        state.status.session.mode = 'verification';
        setTimeout(() => {
          if (verification.phase !== 'stopped') {
            Object.assign(verification, { phase: 'done', ticks: 24 });
            state.status.session.mode = 'idle';
          }
        }, 800);
      } else if (action === '/setup/assembled/start') {
        state.status.session = { ...state.status.session, mode: 'setup', port: body.port, torque_enabled: false };
        state.status.assembled = {
          ...state.status.assembled,
          phase: 'ready',
          port: body.port,
          ids: [11, 12, 13, 14, 15, 16],
          message: 'Distinct addresses ready. Identify each joint by hand.',
        };
      } else if (action === '/setup/assembled/wiggle') {
        state.status.assembled = { ...state.status.assembled, phase: 'wiggling', motion_active: true, wiggle_id: null };
        setTimeout(() => {
          state.status.assembled.phase = 'ready';
          state.status.assembled.motion_active = false;
          state.status.assembled.wiggle_id = body.motor_id;
          state.status.assembled.wiggle_ticks = 24;
        }, 150);
      } else if (action === '/setup/assembled/observe') {
        state.status.assembled = {
          ...state.status.assembled,
          phase: 'observing',
          selected_joint: body.joint,
          detected_id: joints.indexOf(body.joint) + 11,
          candidate_delta: 180,
          message: 'Joint movement detected.',
        };
      } else if (action === '/setup/assembled/accept') {
        state.status.assembled.mapping[body.joint] = body.motor_id;
        state.status.assembled.phase = Object.keys(state.status.assembled.mapping).length === 6 ? 'review' : 'ready';
        state.status.assembled.selected_joint = null;
        state.status.assembled.detected_id = null;
        state.status.assembled.wiggle_id = null;
      } else if (action === '/setup/assembled/finish') {
        state.status.assembled.phase = 'applying';
        state.status.assembled.message = 'Writing and verifying canonical IDs.';
        setTimeout(() => {
          state.status.assembled.phase = 'complete';
          state.status.assembled.mapping = Object.fromEntries(joints.map((name, index) => [name, index + 1]));
          state.status.assembled.calibration_required = true;
          state.status.session.mode = 'idle';
        }, 1000);
      } else if (action === '/setup/assembled/cancel') {
        state.status.assembled.phase = 'stopped';
        state.status.session.mode = 'idle';
      } else if (action === '/calibration/start') {
        state.status.session = {
          ...state.status.session,
          mode: 'calibration',
          port: body.port,
          robot_id: body.robot_id,
          role: body.role,
          positions: values(2048),
        };
        state.status.calibration = {
          ...state.status.calibration,
          phase: body.resume ? (state.status.calibration.interrupted_stage ?? 'ranges') : 'center',
          recoverable: false,
          error: null,
          interrupted_stage: undefined,
        };
        state.status.session.error = null;
      } else if (action === '/calibration/center') {
        if (state.centerError) {
          state.status.session.mode = 'idle';
          state.status.session.error = state.centerError;
          state.status.calibration = {
            ...state.status.calibration,
            phase: 'interrupted',
            interrupted_stage: 'center',
            error: state.centerError,
          };
          return route.fulfill({ status: 409, json: { detail: state.centerError } });
        }
        state.status.calibration = {
          ...state.status.calibration,
          phase: 'ranges',
          mins: values(700),
          maxes: values(3300),
          offsets: values(0),
        };
      } else if (action === '/calibration/finish') {
        state.status.calibration.phase = 'saved';
        state.status.assembled.calibration_required = false;
        state.status.session.mode = 'idle';
        state.status.calibrations = [{ robot_id: 'my-so101', role: 'follower', path: 'private-calibration.json' }];
      } else if (action === '/calibration/cancel' || action === '/stop') {
        if (state.status.session.mode === 'setup') state.status.assembled.phase = 'stopped';
        if (state.status.session.mode === 'verification') state.status.verification.phase = 'stopped';
        if (action === '/stop') state.status.session.error = state.stopError;
        state.status.session.mode = 'idle';
        state.status.session.torque_enabled = false;
        state.status.recording.episode_active = false;
      } else if (action === '/control/start') {
        state.status.session = {
          ...state.status.session,
          mode: body.mode,
          active_joints: body.joints ?? joints,
          port: body.port,
          robot_id: body.robot_id,
          positions: values(0),
          targets: values(0),
          slider_targets: values(0),
          joint_limits: Object.fromEntries(
            (body.joints ?? joints).map((j: string) => [
              j,
              { min: j === 'gripper' ? 0.3 : -100, max: j === 'gripper' ? 99.7 : 100 },
            ]),
          ),
          torque_enabled: body.mode !== 'handguide',
          centering: {
            active: !!body.move_to_middle && state.middleInProgress,
            progress: 0.25,
            tolerance: body.middle_tolerance ?? 2,
          },
        };
      } else if (action === '/control/heartbeat') {
        if (state.heartbeatFailures > 0) {
          state.heartbeatFailures--;
          return route.abort('failed');
        }
      } else if (action === '/control/jog') {
        const before = structuredClone(state.status);
        if (state.jogDelay) await new Promise((resolve) => setTimeout(resolve, state.jogDelay));
        if (state.jogStaleResponse) return route.fulfill({ json: before });
        if (body.target !== undefined) {
          state.status.session.slider_targets[body.joint] = body.target;
        } else if (Math.abs(body.delta) > 2) {
          const limits = state.status.session.joint_limits[body.joint];
          state.status.session.slider_targets[body.joint] = Math.max(
            limits.min,
            Math.min(limits.max, state.status.session.slider_targets[body.joint] + body.delta),
          );
        } else {
          if (!state.encoderLag) state.status.session.positions[body.joint] += body.delta;
          state.status.session.targets[body.joint] += body.delta;
          state.status.session.slider_targets[body.joint] = state.status.session.targets[body.joint];
        }
      } else if (action === '/recording/start') {
        state.status.recording = {
          active: true,
          dataset_id: body.dataset_id,
          episode_active: false,
          frames: 0,
          episodes: 0,
          fps: body.fps,
          error: null,
        };
      } else if (action === '/recording/episode/start') {
        state.status.recording.episode_active = true;
        state.status.recording.frames = 10;
      } else if (action === '/recording/episode/save') {
        state.status.recording.episode_active = false;
        state.status.recording.episodes++;
        state.status.recording.frames = 0;
      } else if (action === '/recording/episode/discard') {
        state.status.recording.episode_active = false;
        state.status.recording.frames = 0;
      } else if (action === '/recording/finish') {
        const current = state.status.recording;
        state.status.datasets = [
          {
            id: current.dataset_id,
            path: 'private-local-dataset',
            episodes: current.episodes,
            frames: body.abandon_empty ? 0 : 10,
            fps: current.fps,
            task: 'Place the block in the tray.',
            format: 'lerobot',
            finalized: !body.abandon_empty,
            abandoned: !!body.abandon_empty,
            pending_frames: body.abandon_empty ? current.frames : 0,
          },
        ];
        state.status.recording = { ...current, active: false, episode_active: false, frames: 0 };
      } else return route.fulfill({ status: 404, json: { detail: `Unexpected robot route ${action}` } });
      if (state.dropCalibrationReply && ['/calibration/center', '/calibration/finish'].includes(action))
        return route.abort('failed');
      return route.fulfill({ json: state.status });
    }
    if (path.includes('/robot-lab/datasets/') && path.endsWith('/import'))
      return route.fulfill({ json: { id: 'library-entry', project_id: path.split('/')[4], status: 'ready' } });
    if (path === '/api/v1/health') return route.fulfill({ json: { status: 'ok', version: 'fixture' } });
    if (path === '/api/v1/modules')
      return route.fulfill({ json: installedModules.filter((item) => ['datasets', 'robot-lab'].includes(item.id)) });
    if (path === '/api/v1/projects')
      return route.fulfill({ json: ['alpha', 'beta'].map((id) => ({ id, name: `Robot ${id}`, created_at: stamp })) });
    if (path === '/api/v1/policy-options')
      return route.fulfill({
        json: { runtimes: [], sources: [], training_models: [], training_methods: [], default_training_method: 'full' },
      });
    if (path === '/api/v1/datasets') state.datasetReads++;
    return route.fulfill({ json: [] });
  });
  await page.goto('/robot-lab/');
  await expect(page.getByRole('heading', { level: 1, name: 'Robot Lab', exact: true })).toBeVisible();
  await expect(page.getByLabel('Robot USB port')).toHaveValue(hostPlatform === 'Linux' ? '/dev/ttyACM0' : port);
  return state;
}
const tab = (page: Page, title: string) =>
  page.getByRole('navigation', { name: 'Robot setup steps' }).getByRole('button', { name: new RegExp(title) });
const writes = (state: Awaited<ReturnType<typeof fixture>>, path: string) =>
  state.writes.filter((item) => item.path.endsWith(path));
async function startManual(page: Page) {
  await tab(page, 'Teleoperate').click();
  await page.getByRole('checkbox', { name: /Arm secured; workspace clear/ }).check();
  await page.getByRole('button', { name: 'Enable & move to recorded middle', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Increase Shoulder pan', exact: true })).toBeEnabled();
}

test('Robot Lab navigation and device discovery issue no motor commands', async ({ page }, testInfo) => {
  const state = await fixture(page);
  await expect(page.getByText('This Mac connected', { exact: true })).toBeVisible();
  await expect(page.getByText('Some hardware tools need installation.', { exact: true })).toHaveCount(0);
  for (const title of ['Motor IDs', 'Calibrate', 'Teleoperate', 'Record', 'Connect']) await tab(page, title).click();
  await expect(page.getByRole('button', { name: 'Enable & move to recorded middle', exact: true })).not.toBeVisible();
  expect(state.writes).toEqual([]);
  await page.getByRole('button', { name: 'Start preview', exact: true }).click();
  await expect(page.getByAltText('Live view from the selected robot camera')).toBeVisible();
  await page.getByRole('button', { name: 'Pause preview', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Start preview', exact: true })).toBeVisible();
  state.cameraFailure = true;
  await page.getByRole('button', { name: 'Start preview', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Retry camera', exact: true })).toBeVisible();
  await expect(page.locator('.robot-lab-camera-live')).toHaveCount(0);
  state.cameraFailure = false;
  await page.getByRole('button', { name: 'Retry camera', exact: true }).click();
  await expect(page.getByAltText('Live view from the selected robot camera')).toBeVisible();
  await page.evaluate(() => {
    if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
    window.scrollTo(0, 0);
  });
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
  expect(writes(state, '/setup/motor')).toEqual([
    {
      path: '/api/v1/robot-lab/setup/motor',
      body: { port, joint: 'gripper', isolated: true, confirm: 'ASSIGN MOTOR ID' },
    },
  ]);
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
  await expect(page.getByLabel('Selected arm calibration')).toContainText('my-so101 · follower');
  expect(writes(state, '/calibration/start')[0].body).toEqual({
    port,
    robot_id: 'my-so101',
    role: 'follower',
    confirm: 'START CALIBRATION',
  });
  expect(writes(state, '/calibration/center')[0].body).toEqual({});
  expect(writes(state, '/calibration/finish')[0].body).toEqual({ confirm: 'SAVE CALIBRATION' });
  expect(writes(state, '/control/start')).toEqual([]);
});

test('failed center capture stays at the interrupted stage and requires an explicit restart', async ({
  page,
}, testInfo) => {
  const state = await fixture(page);
  await tab(page, 'Calibrate').click();
  await page.getByRole('button', { name: 'Begin calibration', exact: true }).click();
  state.centerError = 'Motor 1 write Homing_Offset acknowledgement missing; read-back failed.';
  await page.getByRole('button', { name: 'Record middle position', exact: true }).click();
  await expect(page.getByRole('alert', { name: 'Calibration error' })).toHaveText(state.centerError);
  await expect(
    page.getByRole('region', { name: 'Calibrate', exact: true }).getByRole('alert', { name: 'Calibration error' }),
  ).toBeVisible();
  await expect(page.getByText('Calibration interrupted', { exact: true })).toBeVisible();
  await expect(
    page.getByText('Middle position was not saved. Restart calibration to continue.', { exact: true }),
  ).toBeVisible();
  await expect(page.locator('.robot-lab-calibration-steps .current')).toHaveText('2. Middle position');
  await expect(page.getByRole('button', { name: 'Begin calibration', exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Save calibration', exact: true })).toHaveCount(0);
  expect(writes(state, '/calibration/center')).toHaveLength(1);
  expect(writes(state, '/calibration/finish')).toHaveLength(0);
  await page.evaluate(() => {
    if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
    window.scrollTo(0, 0);
  });
  await page.screenshot({ path: testInfo.outputPath('calibration-interrupted.png'), fullPage: true });
  state.centerError = null;
  await page.getByRole('button', { name: 'Restart calibration', exact: true }).click();
  await expect(page.getByText('Calibration interrupted', { exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Record middle position', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Save calibration', exact: true })).toBeVisible();
  expect(writes(state, '/calibration/start')).toHaveLength(2);
  expect(writes(state, '/control/start')).toEqual([]);
  expect(state.errors).toEqual([]);
});

test('manual control is explicit, jogs are bounded, and navigating away stops the arm', async ({ page }, testInfo) => {
  const state = await fixture(page, true);
  await startManual(page);
  state.encoderLag = true;
  await page.getByLabel('Joint step size').selectOption('2');
  await page.getByRole('button', { name: 'Increase Shoulder pan', exact: true }).click();
  await expect(page.getByLabel('Shoulder pan position')).toHaveText('0.0°');
  await expect(page.getByLabel('Shoulder pan target', { exact: true })).toHaveText('→ 2.0°');
  await page.screenshot({ path: testInfo.outputPath('joint-target-readback.png'), fullPage: true });
  expect(writes(state, '/control/start')[0].body).toEqual({
    port,
    robot_id: 'my-so101',
    mode: 'manual',
    joints,
    move_to_middle: true,
    middle_tolerance: 2,
    confirm: 'ENABLE MOTION',
  });
  expect(writes(state, '/control/jog')[0].body).toEqual({ joint: 'shoulder_pan', delta: 2 });
  await page
    .getByRole('navigation', { name: 'Policy lifecycle' })
    .getByRole('link', { name: 'Dashboard', exact: true })
    .click();
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
  expect(writes(state, '/recording/start')[0].body).toEqual({
    dataset_id: 'block-demo',
    task: 'Place the block in the tray.',
    fps: 10,
    camera_index: 0,
  });
  expect(writes(state, '/import')[0].path).toBe('/api/v1/projects/beta/robot-lab/datasets/block-demo/import');
  expect(writes(state, '/recording/episode/save')).toHaveLength(1);
  await page.evaluate(() => {
    if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
    window.scrollTo(0, 0);
  });
  await page.screenshot({ path: testInfo.outputPath('robot-lab-record.png'), fullPage: true });
  expect(state.errors).toEqual([]);
});

test('unsaved recording can be explicitly closed after Stop while retained files stay unfinalized', async ({
  page,
}) => {
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
  expect(writes(state, '/recording/finish')).toEqual([
    { path: '/api/v1/robot-lab/recording/finish', body: { abandon_empty: true, confirm: 'ABANDON EMPTY RECORDING' } },
  ]);
  const retained = page
    .getByRole('article')
    .filter({ has: page.getByRole('heading', { name: 'empty-demo', exact: true }) });
  await expect(retained.getByRole('button', { name: 'Closed · files retained', exact: true })).toBeDisabled();
  await expect(retained.getByText('10 unsaved frames', { exact: true })).toBeVisible();
  expect(writes(state, '/import')).toEqual([]);
  await startManual(page);
  await tab(page, 'Record').click();
  await page.getByLabel('Dataset name', { exact: true }).fill('next-demo');
  await page.getByLabel('Recording frame rate', { exact: true }).selectOption('5');
  await page.getByRole('button', { name: 'Create recording dataset', exact: true }).click();
  expect(writes(state, '/recording/start').at(-1)?.body).toEqual({
    dataset_id: 'next-demo',
    task: 'Place the block in the tray.',
    fps: 5,
    camera_index: 0,
  });
  expect(state.errors).toEqual([]);
});

test('offline companion cannot start hardware actions and does not retry mutations', async ({ page }) => {
  const state = await fixture(page, true);
  state.status = { connected: false, ok: false, label: 'This Mac', message: 'Companion is offline.' };
  await page.getByRole('button', { name: 'Refresh devices', exact: true }).click();
  await expect(page.getByText('Robot companion unavailable', { exact: true })).toBeVisible();
  await tab(page, 'Motor IDs').click();
  await page.getByText('Advanced: configure one physically isolated motor', { exact: true }).click();
  await page.getByRole('checkbox', { name: /Only the gripper motor is connected/ }).check();
  await expect(page.getByRole('button', { name: 'Assign ID 6', exact: true })).toBeDisabled();
  await tab(page, 'Teleoperate').click();
  await page.getByRole('checkbox', { name: /Arm secured; workspace clear/ }).check();
  await expect(page.getByRole('button', { name: 'Enable & move to recorded middle', exact: true })).toBeDisabled();
  expect(state.writes).toEqual([]);
});

test('hand guiding is an explicit torque-off session and leader mode needs a second calibrated arm', async ({
  page,
}) => {
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
  expect(writes(state, '/control/start')[0].body).toEqual({
    port,
    robot_id: 'my-so101',
    mode: 'handguide',
    confirm: 'ENABLE MOTION',
  });
  expect(state.status.session.torque_enabled).toBe(false);
  await expect.poll(() => writes(state, '/control/heartbeat').length).toBeGreaterThan(0);
  state.stopError = 'Stop attempted but motor acknowledgement failed. Keep supporting the arm.';
  await page.getByRole('button', { name: 'Stop arm and disable torque', exact: true }).click();
  await expect(page.locator('.robot-lab').getByRole('alert')).toContainText(state.stopError);
  await expect(page.getByText('Arm stopped. Torque is disabled.', { exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Start hand guiding', exact: true })).toBeVisible();
});

test('all six illustrated joints work by pointer and keyboard without idle hardware requests', async ({
  page,
}, testInfo) => {
  const state = await fixture(page);
  await tab(page, 'Motor IDs').click();
  const guide = page.getByRole('region', { name: 'SO-101 joint illustration', exact: true });
  await expect(guide.getByRole('slider', { name: 'Illustrated joint angle', exact: true })).toBeVisible();
  await expect(guide).toHaveAttribute('data-motion', 'reduced');
  for (let index = 0; index < joints.length; index++) {
    await guide
      .getByRole('group', { name: 'Choose an illustrated joint', exact: true })
      .getByRole('button')
      .nth(index)
      .click();
    await expect(guide).toHaveAttribute('data-selected-joint', joints[index]);
    await expect(
      guide.getByRole('group', { name: 'Choose an illustrated joint', exact: true }).getByRole('button').nth(index),
    ).toHaveAttribute('aria-pressed', 'true');
  }
  const gripper = guide.getByRole('button', { name: 'Joint 6: Gripper', exact: true });
  await gripper.focus();
  await page.keyboard.press('ArrowLeft');
  await expect(guide).toHaveAttribute('data-selected-joint', 'wrist_roll');
  await page.keyboard.press('Home');
  await expect(guide).toHaveAttribute('data-selected-joint', 'shoulder_pan');
  await page.keyboard.press('End');
  await expect(guide).toHaveAttribute('data-selected-joint', 'gripper');
  const before = await guide.getAttribute('data-angle');
  await guide.getByRole('slider', { name: 'Illustrated joint angle', exact: true }).focus();
  await page.keyboard.press('End');
  await expect(guide).not.toHaveAttribute('data-angle', before!);
  await tab(page, 'Calibrate').click();
  await expect(page.getByRole('region', { name: 'SO-101 joint illustration', exact: true })).toHaveAttribute(
    'data-selected-joint',
    'gripper',
  );
  await expect(page.getByRole('group', { name: 'Measured encoder values', exact: true })).toHaveCount(0);
  expect(state.writes).toEqual([]);
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1),
  ).toBe(true);
  await page.evaluate(() => {
    if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
    window.scrollTo(0, 0);
  });
  await page.screenshot({ path: testInfo.outputPath('so101-joint-explorer.png'), fullPage: true });
});

test('offline joint illustration still selects and never invents measurements or sends mutations', async ({ page }) => {
  const state = await fixture(page);
  state.status = { ok: false, connected: false, message: 'Companion offline.' };
  await page.getByRole('button', { name: 'Refresh devices', exact: true }).click();
  await expect(page.getByText('Robot companion unavailable', { exact: true })).toBeVisible();
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

test('assembled setup prepares explicitly then picks start only encoder observation until mapping is confirmed', async ({
  page,
}, testInfo) => {
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
  expect(writes(state, '/setup/assembled/start')[0].body).toEqual({
    port,
    supported: true,
    confirm: 'START ASSEMBLED SETUP',
  });
  for (let index = 5; index >= 0; index--) {
    const label = ['Shoulder pan', 'Shoulder lift', 'Elbow', 'Wrist flex', 'Wrist roll', 'Gripper'][index];
    const before = state.writes.length;
    await guide.getByRole('button', { name: `Joint ${index + 1}: ${label}`, exact: true }).click();
    await expect(page.getByText(`Movement detected at address ${index + 11}`, { exact: true })).toBeVisible();
    const selectedWrites = state.writes.slice(before).filter((item) => !item.path.endsWith('/control/heartbeat'));
    expect(selectedWrites.map((item) => item.path)).toEqual(['/api/v1/robot-lab/setup/assembled/observe']);
    expect(writes(state, '/setup/assembled/finish')).toEqual([]);
    if (index === 5) {
      await page.evaluate(() => {
        if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
        window.scrollTo(0, 0);
      });
      await page.screenshot({ path: testInfo.outputPath('so101-assembled-detection.png'), fullPage: true });
    }
    await page
      .getByRole('button', { name: `Confirm ID ${index + 11} is the ${label.toLowerCase()}`, exact: true })
      .click();
    if (index === 5) {
      const observations = writes(state, '/setup/assembled/observe').length;
      await guide.getByRole('button', { name: 'Joint 6: Gripper', exact: true }).click();
      await expect(
        guide.getByText('Gripper is confirmed at address 16. Select the next joint.', { exact: true }),
      ).toBeVisible();
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
  await expect(page.getByRole('heading', { name: 'Saved IDs', exact: true })).toBeVisible();
  const saved = page.getByRole('region', { name: 'Saved motor mapping', exact: true });
  await expect(saved.getByRole('row')).toHaveCount(7);
  await expect(saved.getByRole('row').filter({ hasText: 'Gripper' })).toContainText('6');
  await expect(page.getByRole('button', { name: 'Start assembled setup', exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Stop arm and disable torque', exact: true })).toHaveCount(0);
  await tab(page, 'Teleoperate').click();
  await expect(page.getByRole('button', { name: 'Enable & move to recorded middle', exact: true })).toBeDisabled();
  state.status.assembled = {
    ...state.status.assembled,
    phase: 'stopped',
    ids: [3],
    mapping: {},
    message: 'Setup stopped. Partial motor IDs are preserved; restart setup to scan and resume.',
    log: ['Separating factory IDs: round 8/12', 'Setup stopped with responding ID 3.'],
  };
  await page.getByRole('button', { name: 'Refresh devices', exact: true }).click();
  await tab(page, 'Motor IDs').click();
  await expect(page.getByText(state.status.assembled.message, { exact: true })).toBeVisible();
  await page.getByText('Setup details', { exact: true }).click();
  await expect(page.getByText('Setup stopped with responding ID 3.', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Start assembled setup', exact: true })).toBeDisabled();
  expect(writes(state, '/setup/assembled/start')).toHaveLength(1);
});

test('wiggle identification requires explicit motion then observed-joint confirmation', async ({ page }, testInfo) => {
  test.setTimeout(60_000);
  const state = await fixture(page);
  await tab(page, 'Motor IDs').click();
  await page.getByRole('button', { name: 'Wiggle a motor', exact: true }).click();
  expect(state.writes).toEqual([]);
  await expect(page.getByRole('button', { name: 'Wiggle selected motor', exact: true })).toHaveCount(0);
  await page.getByRole('checkbox', { name: /Six STS3215 motors; arm supported/ }).check();
  await page.getByRole('button', { name: 'Start assembled setup', exact: true }).click();
  const pulse = page.getByRole('button', { name: 'Wiggle selected motor', exact: true });
  await expect(pulse).toBeDisabled();
  const guide = page.getByRole('region', { name: 'SO-101 joint illustration', exact: true });
  await guide.getByRole('button', { name: 'Joint 6: Gripper', exact: true }).click();
  expect(writes(state, '/setup/assembled/observe')).toHaveLength(0);
  expect(writes(state, '/setup/assembled/wiggle')).toHaveLength(0);
  await page.getByRole('checkbox', { name: /Arm supported; every joint clear of its stops/ }).check();
  await pulse.click();
  expect(writes(state, '/setup/assembled/wiggle')[0].body).toEqual({
    motor_id: 11,
    supported: true,
    confirm: 'WIGGLE MOTOR',
  });
  const assign = page.getByRole('button', { name: 'Assign address 11 to gripper', exact: true });
  await expect(assign).toBeVisible();
  expect(writes(state, '/setup/assembled/accept')).toHaveLength(0);
  await page.screenshot({ path: testInfo.outputPath('wiggle-identification.png'), fullPage: true });
  await assign.click();
  expect(writes(state, '/setup/assembled/accept')[0].body).toEqual({
    joint: 'gripper',
    motor_id: 11,
    method: 'wiggle',
    confirm: 'CONFIRM OBSERVED JOINT',
  });
  await expect(page.getByLabel('Motor address to identify')).toHaveValue('12');
  await expect(pulse).toBeDisabled();
  expect(writes(state, '/control/start')).toHaveLength(0);
  expect(state.errors).toEqual([]);
});

test('motor identification distinguishes preparation, listening and a stopped session', async ({ page }) => {
  const state = await fixture(page);
  state.status.assembled = { ...state.status.assembled, phase: 'stopped', ids: [2], progress: 1 };
  await page.getByRole('button', { name: 'Refresh devices', exact: true }).click();
  await tab(page, 'Motor IDs').click();
  const status = page.getByRole('status', { name: 'Motor identification status', exact: true });
  await expect(status).toContainText('Movement detection off');
  await expect(status).toContainText('1/6 addresses found');
  await page.getByRole('button', { name: 'Joint 6: Gripper', exact: true }).click();
  expect(writes(state, '/setup/assembled/observe')).toHaveLength(0);
  await page.getByRole('checkbox', { name: /Six STS3215 motors; arm supported/ }).check();
  await page.getByRole('button', { name: 'Start assembled setup', exact: true }).click();
  await expect(status).toContainText('Ready to identify joints');
  await expect(page.getByText('Experimental assembled setup started.', { exact: true })).toHaveCount(0);
  state.status.assembled = { ...state.status.assembled, phase: 'splitting', ids: [2], progress: 1 };
  await expect(status).toContainText('Preparing motor addresses');
  await expect(status).toContainText('Round 1/12 · 1/6 addresses found. No movement is commanded.');
  await page.getByRole('button', { name: 'Joint 6: Gripper', exact: true }).click();
  expect(writes(state, '/setup/assembled/observe')).toHaveLength(0);
  state.status.assembled = { ...state.status.assembled, phase: 'ready', ids: [11, 12, 13, 14, 15, 16] };
  await expect(status).toContainText('Ready to identify joints');
  await page.getByRole('button', { name: 'Start movement detection', exact: true }).click();
  await expect(status).toContainText('Listening for hand movement');
  expect(writes(state, '/setup/assembled/observe')).toHaveLength(1);
  await page.getByRole('button', { name: 'Stop arm and disable torque', exact: true }).click();
  await expect(status).toContainText('Movement detection off');
  await expect(page.getByRole('button', { name: 'Start movement detection', exact: true })).toHaveCount(0);
  expect(state.errors).toEqual([]);
});

test('hiding Robot Lab stops setup and visibly explains the cancellation', async ({ page }) => {
  const state = await fixture(page);
  await tab(page, 'Motor IDs').click();
  await page.getByRole('checkbox', { name: /Six STS3215 motors; arm supported/ }).check();
  await page.getByRole('button', { name: 'Start assembled setup', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Start movement detection', exact: true })).toBeEnabled();
  await page.evaluate(() => {
    Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => 'hidden' });
    document.dispatchEvent(new Event('visibilitychange'));
  });
  await expect.poll(() => writes(state, '/stop').length).toBe(1);
  await page.evaluate(() => {
    Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => 'visible' });
    document.dispatchEvent(new Event('visibilitychange'));
  });
  await expect(page.locator('.robot-lab').getByRole('alert')).toContainText('Robot Lab was hidden');
  await expect(page.getByRole('status', { name: 'Motor identification status', exact: true })).toContainText(
    'Movement detection off',
  );
  expect(writes(state, '/setup/assembled/observe')).toHaveLength(0);
  expect(writes(state, '/setup/assembled/wiggle')).toHaveLength(0);
});

test('background polling keeps device refresh quiet and idle hardware has no active Stop control', async ({ page }) => {
  const state = await fixture(page, true);
  const check = page.getByRole('button', { name: 'Refresh devices', exact: true });
  const stop = page.getByRole('button', { name: 'Stop arm and disable torque', exact: true });
  await expect(stop).toHaveCount(0);
  let unblock!: () => void,
    reads = 0;
  const gate = new Promise<void>((resolve) => {
    unblock = resolve;
  });
  await page.route('**/api/v1/robot-lab/status', async (route) => {
    reads++;
    await gate;
    await route.fulfill({ json: state.status });
  });
  await expect.poll(() => reads, { timeout: 10_000 }).toBeGreaterThan(0);
  await expect(check).toBeEnabled();
  await expect(page.getByRole('button', { name: 'Refreshing…', exact: true })).toHaveCount(0);
  await check.click();
  await expect(page.getByRole('button', { name: 'Refreshing…', exact: true })).toBeDisabled();
  unblock();
  await expect(check).toBeEnabled();
  await startManual(page);
  await expect(stop).toBeVisible();
  await stop.click();
  await expect(stop).toHaveCount(0);
  expect(state.errors).toEqual([]);
});

async function savedMapping(page: Page) {
  const state = await fixture(page);
  state.status.assembled = {
    ...state.status.assembled,
    port,
    phase: 'complete',
    ids: [1, 2, 3, 4, 5, 6],
    mapping: Object.fromEntries(joints.map((name, index) => [name, index + 1])),
    calibration_required: true,
  };
  await page.getByRole('button', { name: 'Refresh devices', exact: true }).click();
  await tab(page, 'Motor IDs').click();
  await expect(page.getByRole('heading', { name: 'Saved IDs', exact: true })).toBeVisible();
  return state;
}

test('saved mapping wiggles each selected ID only on click and requires observed confirmation', async ({
  page,
}, testInfo) => {
  const state = await savedMapping(page);
  const panel = page.getByRole('region', { name: 'Saved motor mapping', exact: true });
  await expect(panel.getByRole('button', { name: 'Wiggle gripper · ID 6', exact: true })).toBeDisabled();
  await panel.getByRole('checkbox', { name: /Arm supported; joints clear/ }).check();
  const labels = ['Shoulder pan', 'Shoulder lift', 'Elbow', 'Wrist flex', 'Wrist roll', 'Gripper'];
  for (let index = 0; index < 6; index++) {
    await panel.getByRole('button', { name: `Joint ${index + 1}: ${labels[index]}`, exact: true }).click();
    expect(writes(state, '/mapping/wiggle')).toHaveLength(index);
    const wiggle = panel.getByRole('button', {
      name: `Wiggle ${labels[index].toLowerCase()} · ID ${index + 1}`,
      exact: true,
    });
    await wiggle.click();
    expect(writes(state, '/mapping/wiggle')[index].body).toEqual({
      port,
      joint: joints[index],
      supported: true,
      confirm: 'WIGGLE SAVED MOTOR',
    });
    await expect(panel.getByRole('button', { name: 'Correct joint', exact: true })).toBeVisible();
    await expect(panel.getByText(`${index}/6 motion checks`, { exact: true })).toBeVisible();
    await panel.getByRole('button', { name: 'Correct joint', exact: true }).click();
    await expect(panel.getByText(`${index + 1}/6 motion checks`, { exact: true })).toBeVisible();
  }
  expect(writes(state, '/setup/assembled/start')).toEqual([]);
  expect(writes(state, '/setup/assembled/finish')).toEqual([]);
  expect(writes(state, '/calibration/start')).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath('saved-mapping-checks.png'), fullPage: true });
  expect(state.errors).toEqual([]);
});

test('mapping mismatch and cancellation never claim a successful check', async ({ page }) => {
  const state = await savedMapping(page);
  await page.getByRole('checkbox', { name: /Arm supported; joints clear/ }).check();
  const wiggle = page.getByRole('button', { name: 'Wiggle gripper · ID 6', exact: true });
  await wiggle.click();
  await page.getByRole('button', { name: 'Wrong joint', exact: true }).click();
  await expect(page.getByRole('cell', { name: 'Mismatch', exact: true })).toBeVisible();
  await expect(page.getByText('0/6 motion checks', { exact: true })).toBeVisible();
  await wiggle.click();
  await page.getByRole('button', { name: 'Stop arm and disable torque', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Correct joint', exact: true })).toHaveCount(0);
  await expect(wiggle).toBeDisabled();
  expect(state.status.assembled.mapping.gripper).toBe(6);
  expect(state.errors).toEqual([]);
});

test('orbiting the CAD model updates joint labels without hardware commands', async ({ page }) => {
  const state = await savedMapping(page);
  const view = page.getByRole('group', { name: '3D arm view', exact: true });
  const marker = page.getByRole('button', { name: 'Joint 6: Gripper', exact: true });
  const position = await marker.getAttribute('style');
  await view.focus();
  await page.keyboard.press('ArrowRight');
  await expect(view).toHaveAttribute('data-view-yaw', '15.0');
  expect(await marker.getAttribute('style')).not.toBe(position);
  await page.getByRole('button', { name: 'Reset view', exact: true }).click();
  const bounds = await view.boundingBox();
  if (!bounds) throw new Error('Missing CAD viewport');
  await page.mouse.move(bounds.x + 20, bounds.y + 20);
  await page.mouse.down();
  await page.mouse.move(bounds.x + 100, bounds.y + 55, { steps: 5 });
  await page.mouse.up();
  await expect(view).not.toHaveAttribute('data-view-yaw', '0.0');
  await page.getByRole('button', { name: 'Zoom in 3D view', exact: true }).click();
  await marker.click();
  expect(state.writes).toEqual([]);
  expect(state.errors).toEqual([]);
});

test('USB disconnect disables remembered motor controls and reconnect requires fresh motion acknowledgement', async ({
  page,
}) => {
  const state = await savedMapping(page);
  const wiggle = page.getByRole('button', { name: 'Wiggle gripper · ID 6', exact: true });
  const acknowledge = page.getByRole('checkbox', { name: /Arm supported; joints clear/ });
  await acknowledge.check();
  await expect(wiggle).toBeEnabled();
  state.portsConnected = false;
  await page.getByRole('button', { name: 'Refresh devices', exact: true }).click();
  await expect(page.getByText('Robot USB disconnected', { exact: true })).toBeVisible();
  await expect(wiggle).toBeDisabled();
  await expect(acknowledge).not.toBeChecked();
  await expect(acknowledge).toBeDisabled();
  await expect(page.getByRole('heading', { name: 'Saved IDs', exact: true })).toBeVisible();
  await expect(page.getByLabel('Robot USB port')).toHaveValue(port);
  await expect(page.getByLabel('Robot USB port').locator('option:checked')).toContainText('Disconnected');
  await tab(page, 'Calibrate').click();
  await expect(page.getByRole('button', { name: 'Begin calibration', exact: true })).toBeDisabled();
  state.status.calibrations = [{ robot_id: 'my-so101', role: 'follower', path: 'saved.json' }];
  state.status.assembled.calibration_required = false;
  await page.getByRole('button', { name: 'Refresh devices', exact: true }).click();
  // A saved calibration is not enough to teleoperate without the robot USB.
  await expect(tab(page, 'Teleoperate')).toContainText('Needs connection');
  await tab(page, 'Teleoperate').click();
  await page.getByRole('checkbox', { name: /Arm secured; workspace clear/ }).check();
  await expect(page.getByRole('button', { name: 'Enable & move to recorded middle', exact: true })).toBeDisabled();
  await tab(page, 'Connect').click();
  await expect(page.getByRole('button', { name: 'Scan motor IDs', exact: true })).toBeDisabled();
  await tab(page, 'Motor IDs').click();
  state.portsConnected = true;
  // Idle inventory polling notices USB return without refreshing or starting a session.
  await expect(page.getByText('Robot USB connected · Idle', { exact: true })).toBeVisible({ timeout: 8_000 });
  await expect(tab(page, 'Teleoperate')).toContainText('Ready');
  await expect(acknowledge).toBeEnabled();
  await expect(acknowledge).not.toBeChecked();
  await expect(wiggle).toBeDisabled();
  await acknowledge.check();
  await expect(wiggle).toBeEnabled();
  expect(state.writes).toEqual([]);
  expect(state.errors).toEqual([]);
});

test('moving to recorded middle blocks jogging and recording but keeps Stop available', async ({ page }) => {
  const state = await fixture(page, true);
  state.middleInProgress = true;
  await tab(page, 'Teleoperate').click();
  await page.getByRole('checkbox', { name: /Arm secured; workspace clear/ }).check();
  await page.getByRole('button', { name: 'Enable & move to recorded middle', exact: true }).click();
  await expect(page.getByText('Moving to recorded middle · 25%')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Increase Shoulder pan', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Create a dataset', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Stop & release motors', exact: true })).toBeEnabled();
  state.status.session.centering = { active: false, progress: 1 };
  await expect(page.getByRole('button', { name: 'Increase Shoulder pan', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'Stop & release motors', exact: true }).click();
  expect(writes(state, '/stop')).toHaveLength(1);
  expect(writes(state, '/control/jog')).toHaveLength(0);
});

test('selected motor control excludes disconnected motors and blocks dataset recording', async ({ page }, testInfo) => {
  const state = await fixture(page, true);
  await tab(page, 'Teleoperate').click();
  await page.getByRole('button', { name: 'Select connected motors', exact: true }).click();
  // Fixture probe reports only ID 1. Selection is explicit and never starts motion.
  const selection = page.getByRole('group', { name: 'Motors in this session' });
  await expect(selection.getByRole('checkbox', { name: '1 · Shoulder pan', exact: true })).toBeChecked();
  await expect(selection.getByRole('checkbox', { name: '6 · Gripper', exact: true })).not.toBeChecked();
  expect(writes(state, '/control/start')).toHaveLength(0);
  await expect(page.getByRole('button', { name: 'Enable selected motors', exact: true })).toBeDisabled();
  await page.getByRole('checkbox', { name: /Arm secured; workspace clear/ }).check();
  await page.getByRole('button', { name: 'Enable selected motors', exact: true }).click();
  expect(writes(state, '/control/start')[0].body).toMatchObject({ joints: ['shoulder_pan'], move_to_middle: false });
  await expect(page.getByRole('button', { name: 'Increase Shoulder pan', exact: true })).toBeEnabled();
  await expect(page.getByRole('button', { name: 'Increase Gripper', exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Create a dataset' })).toBeDisabled();
  await page.screenshot({ path: testInfo.outputPath('selected-motor-control.png'), fullPage: true });
  await tab(page, 'Record').click();
  await expect(page.getByText('Dataset recording requires all six joints.')).toBeVisible();
  await page.getByRole('button', { name: 'Stop arm and disable torque' }).click();
  await expect.poll(() => writes(state, '/stop').length).toBe(1);
});

for (const recording of [false, true]) {
  test(`joint buttons keep the page stable during ${recording ? 'recording' : 'teleoperation'}`, async ({
    page,
  }, testInfo) => {
    const state = await fixture(page, true);
    await startManual(page);
    if (recording) {
      await tab(page, 'Record').click();
      await page.getByLabel('Dataset name', { exact: true }).fill('stable-controls');
      await page.getByLabel('Task instruction', { exact: true }).fill('Move the elbow.');
      await page.getByRole('button', { name: 'Create recording dataset', exact: true }).click();
      await expect(page.getByRole('button', { name: 'Start episode', exact: true })).toBeEnabled();
    }
    await expect.poll(() => state.robotDatasetReads).toBeGreaterThan(recording ? 2 : 1);
    const reads = state.robotDatasetReads;
    const banner = page.locator('.robot-lab-event[role="status"]');
    const initialBanner = await banner.innerText();
    const plus = page.getByRole('button', {
      name: recording ? 'Record increase Elbow' : 'Increase Elbow',
      exact: true,
    });
    const minus = page.getByRole('button', {
      name: recording ? 'Record decrease Elbow' : 'Decrease Elbow',
      exact: true,
    });
    await plus.scrollIntoViewIfNeeded();
    const before = await plus.boundingBox();
    const scrollBefore = await page.evaluate(() => window.scrollY);
    let navigations = 0;
    page.on('framenavigated', (frame) => {
      if (frame === page.mainFrame()) navigations++;
    });
    state.jogDelay = 250;
    await plus.click();
    await expect.poll(() => writes(state, '/control/jog').length).toBe(1);
    await expect(banner).toHaveText(initialBanner);
    await expect(tab(page, recording ? 'Record' : 'Teleoperate')).toBeEnabled();
    await expect(page.getByRole('button', { name: 'Stop arm and disable torque' })).toBeEnabled();
    await expect.poll(async () => (await plus.boundingBox())?.y).toBeCloseTo(before!.y, 0);
    expect(await page.evaluate(() => window.scrollY)).toBe(scrollBefore);
    await expect(plus).toBeEnabled();
    await minus.click();
    await expect.poll(() => writes(state, '/control/jog').length).toBe(2);
    await expect(minus).toBeEnabled();
    await expect(banner).toHaveText(initialBanner);
    expect(state.robotDatasetReads).toBe(reads);
    expect(navigations).toBe(0);
    expect(state.errors).toEqual([]);
    await page.screenshot({ path: testInfo.outputPath('stable-joint-controls.png'), fullPage: true });
  });
}

test('Stop remains available during a joint request and a late reply cannot restore control', async ({ page }) => {
  const state = await fixture(page, true);
  await startManual(page);
  state.jogDelay = 500;
  state.jogStaleResponse = true;
  await page.getByRole('button', { name: 'Increase Elbow', exact: true }).click();
  await expect.poll(() => writes(state, '/control/jog').length).toBe(1);
  await page.getByRole('button', { name: 'Stop arm and disable torque' }).click();
  await expect(page.getByRole('button', { name: 'Enable & move to recorded middle', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Increase Elbow', exact: true })).toHaveCount(0);
  await expect.poll(() => state.status.session.mode).toBe('idle');
  // Wait for the deliberately stale manual-mode response to arrive.
  await page.waitForTimeout(650);
  await expect(page.getByRole('button', { name: 'Increase Elbow', exact: true })).toHaveCount(0);
  expect(writes(state, '/stop')).toHaveLength(1);
  expect(state.errors).toEqual([]);
});

test('calibration survives page reload, hidden tab and a temporary web disconnection', async ({ page }, testInfo) => {
  const state = await fixture(page);
  await tab(page, 'Calibrate').click();
  await page.getByRole('button', { name: 'Begin calibration', exact: true }).click();
  await page.getByRole('button', { name: 'Record middle position', exact: true }).click();
  state.status.calibration.mins = values(777);
  await page.reload();
  const save = page.getByRole('button', { name: 'Save calibration', exact: true });
  await expect(save).toBeEnabled();
  await expect(page.getByRole('region', { name: 'Calibrate', exact: true })).toContainText('777');
  expect(writes(state, '/calibration/start')).toHaveLength(1);
  expect(writes(state, '/calibration/center')).toHaveLength(1);
  await page.evaluate(() => {
    Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => 'hidden' });
    document.dispatchEvent(new Event('visibilitychange'));
  });
  expect(writes(state, '/stop')).toHaveLength(0);
  await page.evaluate(() => {
    Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => 'visible' });
    document.dispatchEvent(new Event('visibilitychange'));
  });
  state.statusFailures = true;
  await expect(
    page.getByText('Reconnecting… Calibration continues on the robot computer. Captured values are retained.', {
      exact: true,
    }),
  ).toBeVisible();
  await expect(save).toBeVisible();
  await expect(save).toBeDisabled();
  state.statusFailures = false;
  await expect(save).toBeEnabled();
  await page.getByRole('link', { name: 'Guide', exact: true }).filter({ visible: true }).first().click();
  await expect(page).toHaveURL(/\/guide\/(?:#robot-lab)?$/);
  await page.goBack();
  await expect(page).toHaveURL(/\/robot-lab\/?$/);
  await expect(save).toBeEnabled();
  expect(writes(state, '/stop')).toHaveLength(0);
  expect(writes(state, '/control/heartbeat')).toHaveLength(0);
  await page.screenshot({ path: testInfo.outputPath('calibration-reconnected.png'), fullPage: true });
});

test('calibration can resume a Mac checkpoint without repeating middle capture', async ({ page }) => {
  const state = await fixture(page);
  state.status.calibration = {
    ...state.status.calibration,
    phase: 'interrupted',
    interrupted_stage: 'ranges',
    recoverable: true,
    port,
    robot_id: 'my-so101',
    role: 'follower',
    mins: values(777),
    maxes: values(3000),
    offsets: values(10),
  };
  await tab(page, 'Calibrate').click();
  await page.getByRole('button', { name: 'Resume captured calibration', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Save calibration', exact: true })).toBeEnabled();
  expect(writes(state, '/calibration/start')[0].body).toEqual({ resume: true, confirm: 'START CALIBRATION' });
  expect(writes(state, '/calibration/center')).toHaveLength(0);
  await expect(page.getByRole('region', { name: 'Calibrate', exact: true })).toContainText('777');
});

test('one failed heartbeat is retried and does not abandon powered-session ownership', async ({ page }) => {
  const state = await fixture(page, true);
  await startManual(page);
  const before = writes(state, '/control/heartbeat').length;
  state.heartbeatFailures = 1;
  await expect.poll(() => state.heartbeatFailures).toBe(0);
  await expect.poll(() => writes(state, '/control/heartbeat').length).toBeGreaterThan(before + 1);
  await expect(page.getByRole('button', { name: 'Increase Elbow', exact: true })).toBeEnabled();
  expect(writes(state, '/stop')).toHaveLength(0);
  await expect(page.locator('.robot-lab').getByText(/Connection lost:/)).toHaveCount(0);
});

test('lost calibration replies reconcile the completed stage without replaying motor writes', async ({ page }) => {
  const state = await fixture(page);
  await tab(page, 'Calibrate').click();
  await page.getByRole('button', { name: 'Begin calibration', exact: true }).click();
  state.dropCalibrationReply = true;
  await page.getByRole('button', { name: 'Record middle position', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Save calibration', exact: true })).toBeEnabled();
  await expect(page.getByRole('alert', { name: 'Calibration error' })).toHaveCount(0);
  await page.getByRole('button', { name: 'Save calibration', exact: true }).click();
  await expect(page.getByLabel('Selected arm calibration')).toContainText('my-so101 · follower');
  await expect(page.getByRole('alert', { name: 'Calibration error' })).toHaveCount(0);
  expect(writes(state, '/calibration/center')).toHaveLength(1);
  expect(writes(state, '/calibration/finish')).toHaveLength(1);
  expect(writes(state, '/stop')).toHaveLength(0);
});

for (const recording of [false, true]) {
  test(`joint sliders commit a bounded target on release during ${recording ? 'recording' : 'teleoperation'}`, async ({
    page,
  }, testInfo) => {
    const state = await fixture(page, true);
    await startManual(page);
    if (recording) {
      await tab(page, 'Record').click();
      await page.getByLabel('Dataset name', { exact: true }).fill('slider-controls');
      await page.getByLabel('Task instruction', { exact: true }).fill('Move the arm.');
      await page.getByRole('button', { name: 'Create recording dataset', exact: true }).click();
      await page.getByRole('button', { name: 'Start episode', exact: true }).click();
    }
    const slider = page.getByRole('slider', { name: 'Elbow target slider', exact: true });
    await expect(slider).toBeEnabled();
    await expect(page.getByRole('slider', { name: /target slider/ })).toHaveCount(6);
    expect(await slider.getAttribute('min')).toBe('-100');
    expect(await page.getByRole('slider', { name: 'Gripper target slider' }).getAttribute('max')).toBe('99.7');
    await slider.scrollIntoViewIfNeeded();
    const box = (await slider.boundingBox())!;
    const reads = state.robotDatasetReads;
    const before = await page.evaluate(() => window.scrollY);
    let navigations = 0;
    page.on('framenavigated', (frame) => {
      if (frame === page.mainFrame()) navigations++;
    });
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.7, box.y + box.height / 2, { steps: 5 });
    expect(writes(state, '/control/jog')).toHaveLength(0);
    await page.mouse.up();
    await expect.poll(() => writes(state, '/control/jog').length).toBe(1);
    const command = writes(state, '/control/jog')[0].body;
    expect(command.joint).toBe('elbow_flex');
    expect(command.target).toBeGreaterThan(0);
    expect(command.target).toBeLessThanOrEqual(100);
    expect(command.delta).toBeUndefined();
    await expect(slider).toBeEnabled();
    await expect(slider).toHaveValue(String(command.target));
    await expect(page.getByLabel('Elbow position', { exact: true })).toHaveText('0.0°');
    await expect(page.getByLabel('Elbow target', { exact: true })).toHaveText(`→ ${command.target.toFixed(1)}°`);
    expect(await page.evaluate(() => window.scrollY)).toBe(before);
    expect(state.robotDatasetReads).toBe(reads);
    expect(navigations).toBe(0);
    await slider.focus();
    await page.keyboard.press('ArrowLeft');
    await expect.poll(() => writes(state, '/control/jog').length).toBe(2);
    expect(writes(state, '/control/jog')[1].body.target).toBeCloseTo(command.target - 0.1, 1);
    expect(state.errors).toEqual([]);
    await page.screenshot({ path: testInfo.outputPath('joint-sliders.png'), fullPage: true });
    await page.locator('.robot-lab-joints').screenshot({ path: testInfo.outputPath('joint-sliders-controls.png') });
  });
}

test('joint sliders respect centering and excluded joints and reject stale replies after Stop', async ({ page }) => {
  const state = await fixture(page, true);
  state.middleInProgress = true;
  await tab(page, 'Teleoperate').click();
  await page.getByRole('checkbox', { name: /Arm secured; workspace clear/ }).check();
  await page.getByRole('button', { name: 'Enable & move to recorded middle', exact: true }).click();
  await expect(page.getByRole('slider', { name: 'Elbow target slider' })).toBeDisabled();
  state.status.session.centering.active = false;
  await expect(page.getByRole('slider', { name: 'Elbow target slider' })).toBeEnabled();
  await page.getByRole('button', { name: 'Stop arm and disable torque' }).click();
  for (const name of ['Shoulder pan', 'Shoulder lift', 'Wrist flex', 'Wrist roll', 'Gripper']) {
    await page.getByRole('checkbox', { name: new RegExp(name) }).uncheck();
  }
  await page.getByRole('checkbox', { name: /Arm secured; workspace clear/ }).check();
  await page.getByRole('button', { name: 'Enable selected motors' }).click();
  const slider = page.getByRole('slider', { name: 'Elbow target slider', exact: true });
  await expect(slider).toBeEnabled();
  await expect(page.getByRole('slider', { name: /target slider/ })).toHaveCount(1);
  state.jogDelay = 500;
  state.jogStaleResponse = true;
  await slider.focus();
  await page.keyboard.press('ArrowRight');
  await expect.poll(() => writes(state, '/control/jog').length).toBe(1);
  await expect(page.getByRole('button', { name: 'Stop arm and disable torque' })).toBeEnabled();
  await page.getByRole('button', { name: 'Stop arm and disable torque' }).click();
  await page.waitForTimeout(650);
  await expect(page.getByRole('slider', { name: /target slider/ })).toHaveCount(0);
  expect(state.status.session.mode).toBe('idle');
  expect(state.errors).toEqual([]);
});

test('middle tolerance defaults to two and five is explicit, fixed during motion and remembered', async ({
  page,
}, testInfo) => {
  const state = await fixture(page, true);
  state.middleInProgress = true;
  await tab(page, 'Teleoperate').click();
  const tolerance = page.getByLabel('Middle tolerance', { exact: true });
  await expect(tolerance).toHaveValue('2');
  await tolerance.selectOption('5');
  expect(writes(state, '/control/start')).toHaveLength(0);
  await page.getByRole('checkbox', { name: /Arm secured; workspace clear/ }).check();
  await page.getByRole('button', { name: 'Enable & move to recorded middle', exact: true }).click();
  expect(writes(state, '/control/start')[0].body).toMatchObject({ move_to_middle: true, middle_tolerance: 5 });
  await expect(tolerance).toBeDisabled();
  await expect(tolerance).toHaveValue('5');
  await expect(page.getByRole('button', { name: 'Increase Shoulder lift', exact: true })).toBeDisabled();
  state.status.session.centering.active = false;
  await expect(page.getByRole('button', { name: 'Increase Shoulder lift', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'Stop arm and disable torque' }).click();
  await page.reload();
  await tab(page, 'Teleoperate').click();
  await expect(tolerance).toHaveValue('5');
  await expect(tolerance).toBeEnabled();
  expect(writes(state, '/control/start')).toHaveLength(1);
  expect(state.errors).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath('middle-tolerance.png'), fullPage: true });
});

test('readiness separates USB, motor responses, mapping and selected calibration without automatic motor work', async ({
  page,
}, testInfo) => {
  const state = await fixture(page, true);
  const readiness = page.getByRole('region', { name: 'Arm readiness', exact: true });
  await expect(readiness).toContainText('Not scanned');
  await expect(readiness).toContainText('Not identified');
  await expect(readiness).toContainText('Saved · not checked');
  await page.getByRole('button', { name: 'Refresh devices', exact: true }).click();
  expect(state.writes).toEqual([]);
  await readiness.getByRole('button', { name: 'Scan arm', exact: true }).click();
  await expect(readiness).toContainText('1/6 responding');
  const results = page.getByRole('region', { name: 'Motor scan results' });
  await expect(results.getByRole('row')).toHaveCount(7);
  await expect(results.getByText('No response', { exact: true })).toHaveCount(5);
  expect(writes(state, '/probe')[0].body).toEqual({ port, robot_id: 'my-so101', role: 'follower' });
  expect(writes(state, '/mapping/wiggle')).toEqual([]);
  await readiness.getByRole('button', { name: /Motor IDs/ }).click();
  await expect(page.getByRole('button', { name: 'Start assembled setup', exact: true })).toBeDisabled();
  await readiness.getByRole('button', { name: /Calibrate/ }).click();
  await expect(page.getByLabel('Selected arm calibration')).toContainText('Saved · not checked');
  await page.getByLabel('Robot name').fill('different-arm');
  await expect(page.getByLabel('Selected arm calibration')).toContainText('Not saved');
  expect(writes(state, '/calibration/start')).toEqual([]);
  expect(writes(state, '/control/start')).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath('readiness-selected-arm.png'), fullPage: true });
});

test('saved physical confirmations survive reload and guide the next unchecked joint without wiggling it', async ({
  page,
}) => {
  const state = await savedMapping(page);
  await page.getByRole('checkbox', { name: /Arm supported; joints clear/ }).check();
  await page.getByRole('button', { name: 'Wiggle gripper · ID 6', exact: true }).click();
  await page.getByRole('button', { name: 'Correct joint', exact: true }).click();
  await expect(page.getByText('1/6 motion checks', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Wiggle wrist roll · ID 5', exact: true })).toBeVisible();
  expect(writes(state, '/mapping/wiggle')).toHaveLength(1);
  expect(writes(state, '/mapping/confirm')[0].body).toMatchObject({
    port,
    joint: 'gripper',
    correct: true,
    confirm: 'CONFIRM PHYSICAL JOINT',
  });
  await page.reload();
  await tab(page, 'Motor IDs').click();
  await expect(page.getByText('1/6 motion checks', { exact: true })).toBeVisible();
  await expect(page.getByRole('checkbox', { name: /Arm supported; joints clear/ })).not.toBeChecked();
  expect(writes(state, '/mapping/wiggle')).toHaveLength(1);
});

test('readiness never turns an invalid or stale profile green and shows hardware mismatch', async ({
  page,
}, testInfo) => {
  const state = await fixture(page, true);
  const readiness = page.getByRole('region', { name: 'Arm readiness' });
  state.status.calibrations[0] = { ...state.status.calibrations[0], valid: false, error: 'Missing joint ranges' };
  await page.getByRole('button', { name: 'Refresh devices', exact: true }).click();
  await expect(readiness).toContainText('Invalid profile');
  await tab(page, 'Teleoperate').click();
  await page.getByRole('checkbox', { name: /Arm secured; workspace clear/ }).check();
  await expect(page.getByRole('button', { name: 'Enable & move to recorded middle', exact: true })).toBeDisabled();
  state.status.calibrations[0] = { ...state.status.calibrations[0], valid: true, setup_port: port, setup_fresh: false };
  await page.getByRole('button', { name: 'Refresh devices', exact: true }).click();
  await expect(readiness).toContainText('Recalibration needed');
  state.status.calibrations[0].setup_fresh = true;
  await page.getByRole('button', { name: 'Refresh devices', exact: true }).click();
  await page.route('**/api/v1/robot-lab/probe', (route) =>
    route.fulfill({
      json: {
        port,
        baudrate: 1000000,
        ready: true,
        expected_ids: [1, 2, 3, 4, 5, 6],
        motors: joints.map((_, i) => ({ id: i + 1, model: 777, stable: true, responses: 3 })),
        profile: { robot_id: 'my-so101', role: 'follower', state: 'mismatch', joints: { shoulder_lift: false } },
      },
    }),
  );
  await readiness.getByRole('button', { name: 'Scan arm', exact: true }).click();
  await expect(readiness).toContainText('Settings differ');
  await page.screenshot({ path: testInfo.outputPath('readiness-mismatch.png'), fullPage: true });
  state.status.calibrations = [];
  await page.getByRole('button', { name: 'Refresh devices', exact: true }).click();
  await expect(readiness).toContainText('Not saved');
  await expect(page.getByRole('button', { name: 'Enable & move to recorded middle', exact: true })).toBeDisabled();
  expect(writes(state, '/control/start')).toEqual([]);
});

test('calibration ranges indicate which joints still need movement before saving', async ({ page }) => {
  const state = await fixture(page);
  await tab(page, 'Calibrate').click();
  await page.getByRole('button', { name: 'Begin calibration', exact: true }).click();
  await page.getByRole('button', { name: 'Record middle position', exact: true }).click();
  state.status.calibration.mins = values(2047);
  state.status.calibration.maxes = values(2047);
  await page.getByRole('button', { name: 'Refresh devices', exact: true }).click();
  await expect(page.getByText('0/5 ranges detected · wrist roll automatic', { exact: true })).toBeVisible();
  await expect(page.getByRole('cell', { name: 'Move joint', exact: true })).toHaveCount(5);
  await expect(page.getByRole('button', { name: 'Save calibration', exact: true })).toBeDisabled();
  state.status.calibration.mins = values(700);
  state.status.calibration.maxes = values(3300);
  await page.getByRole('button', { name: 'Refresh devices', exact: true }).click();
  await expect(page.getByText('5/5 ranges detected · wrist roll automatic', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Save calibration', exact: true })).toBeEnabled();
  expect(writes(state, '/calibration/finish')).toEqual([]);
});

test('large joint steps offer five and ten and update only the requested destination without refreshing', async ({
  page,
}, testInfo) => {
  const state = await fixture(page, true);
  await startManual(page);
  const selector = page.getByLabel('Joint step size');
  await expect(selector.locator('option')).toHaveText(['0.5° / %', '1° / %', '2° / %', '5° / %', '10° / %']);
  await selector.selectOption('5');
  await page.getByRole('button', { name: 'Increase Elbow', exact: true }).click();
  await expect(page.getByLabel('Elbow target', { exact: true })).toHaveText('→ 5.0°');
  await expect(page.getByLabel('Elbow position', { exact: true })).toHaveText('0.0°');
  await selector.selectOption('10');
  await page.getByRole('button', { name: 'Increase Elbow', exact: true }).click();
  await expect(page.getByLabel('Elbow target', { exact: true })).toHaveText('→ 15.0°');
  await page.getByRole('button', { name: 'Decrease Elbow', exact: true }).click();
  await expect(page.getByLabel('Elbow target', { exact: true })).toHaveText('→ 5.0°');
  await page.getByRole('button', { name: 'Increase Gripper', exact: true }).click();
  await expect(page.getByLabel('Gripper target', { exact: true })).toHaveText('→ 10.0%');
  expect(writes(state, '/control/jog').map((write) => write.body)).toEqual([
    { joint: 'elbow_flex', delta: 5 },
    { joint: 'elbow_flex', delta: 10 },
    { joint: 'elbow_flex', delta: -10 },
    { joint: 'gripper', delta: 10 },
  ]);
  await expect(selector).toHaveValue('10');
  await page.screenshot({ path: testInfo.outputPath('large-joint-steps.png'), fullPage: true });
  await page.getByRole('button', { name: 'Stop & release motors', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Increase Elbow', exact: true })).toHaveCount(0);
  expect(state.errors).toEqual([]);
});

test('large joint steps are hidden when the connected companion does not support them', async ({ page }) => {
  const state = await fixture(page, true);
  state.status.capabilities.large_joint_steps = false;
  await page.getByRole('button', { name: 'Refresh devices', exact: true }).click();
  await startManual(page);
  await expect(page.getByLabel('Joint step size').locator('option')).toHaveText(['0.5° / %', '1° / %', '2° / %']);
  expect(writes(state, '/control/jog')).toEqual([]);
});

test('Linux companion discovers USB and camera without motor commands', async ({ page }) => {
  const state = await fixture(page, false, 'Linux');
  await expect(page.getByText('Linux robot computer connected', { exact: true })).toBeVisible();
  await expect(page.getByLabel('Robot USB port')).toHaveValue('/dev/ttyACM0');
  await tab(page, 'Record').click();
  await expect(page.getByRole('heading', { name: 'Datasets on the robot computer', exact: true })).toBeVisible();
  state.cameraFailure = true;
  await page.getByRole('button', { name: 'Start preview', exact: true }).click();
  await expect(
    page.getByText(
      'Camera could not open. Check camera permissions on the robot computer or close another app using it.',
      { exact: true },
    ),
  ).toBeVisible();
  expect(state.writes).toEqual([]);
  expect(state.errors).toEqual([]);
});

test('LeRobot control identifies direct goals and sends the full slider target', async ({ page }) => {
  const state = await fixture(page, true);
  await startManual(page);
  await expect(page.getByText('LeRobot SO101 control', { exact: true })).toBeVisible();
  await expect(
    page.getByText('Direct position targets; motors may move quickly. Keep the full path clear.', { exact: false }),
  ).toBeVisible();
  const slider = page.getByRole('slider', { name: 'Shoulder pan target slider' });
  await slider.press('End');
  await expect.poll(() => writes(state, '/control/jog').length).toBeGreaterThan(0);
  expect(writes(state, '/control/jog').at(-1)?.body.target).toBeGreaterThan(99);
});
