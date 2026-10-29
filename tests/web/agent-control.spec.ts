import { fulfillWorkJobRead, openModuleTask } from './work-job-controls';
import { expect, test, type Page } from '@playwright/test';
import { fulfillApplicationUpdateRead } from './application-update-controls';
import { installedModules } from './module-fixtures';

// These fixtures verify browser authority and evidence boundaries without model or motor calls.
async function fixture(
  page: Page,
  options: {
    planningMode?: 'joint' | 'task';
    physical?: boolean;
    deferManualHeartbeat?: boolean;
    holdAdmissionImages?: boolean;
    emptyPorts?: boolean;
    alternatePort?: boolean;
    staleCalibration?: boolean;
    cameraFailure?: boolean;
  } = {},
) {
  const requests: { path: string; body: Record<string, unknown> }[] = [];
  const reads: string[] = [];
  const stamp = new Date().toISOString();
  const observation = {
    id: 'obs-fixture',
    sha256: 'a'.repeat(64),
    target_session_id: 'replay-fixture',
    episode_id: null,
    authority_epoch: 1,
    revision: 0,
    step: 0,
    captured_at: stamp,
    expires_at: new Date(Date.now() + 120000).toISOString(),
    clock_policy: 'replay',
    environment_kind: 'replay',
    embodiment_manifest_id: 'fixture-arm',
    scene_id: 'fixture',
    joints: { shoulder: 0 } as Record<string, number>,
    units: 'radians',
    joint_units: {} as Record<string, string>,
    calibration_sha256: null as string | null,
    image_sha256: null as string | null,
    source: 'synthetic_fixture',
  };
  const saved = {
    id: 'agent-fixture',
    project_id: 'agent-project',
    provider_id: 'mock',
    provider_revision: 'mock-v1',
    task: 'Move shoulder by 0.03 radians',
    target_kind: 'replay',
    teaching_job_id: null,
    mode: 'supervised',
    clock_policy: 'replay',
    status: 'ready',
    authority_epoch: 1,
    created_at: stamp,
    updated_at: stamp,
    observation,
    pending_proposal: null as Record<string, unknown> | null,
    receipts: [] as Record<string, unknown>[],
    calls_used: 0,
    input_tokens: 0,
    output_tokens: 0,
    max_calls: 4,
    max_seconds: 120,
    max_output_tokens: 1024,
    max_total_tokens: 16000,
    message: 'Synthetic replay fixture; no motion',
    task_success_verified: false,
    planning_mode: options.planningMode ?? 'joint',
    max_steps: 12,
    step_count: 0,
    run_enabled: false,
    control_armed: false,
    supported_skills: ['joint_correction'],
    model_assessment: null as Record<string, unknown> | null,
    task_review: null as Record<string, unknown> | null,
    execution_observation: null as typeof observation | null,
  };
  const joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'];
  const jointUnits = Object.fromEntries(joints.map((joint) => [joint, joint === 'gripper' ? 'percent' : 'degrees']));
  const robot = {
    ok: true,
    connected: true,
    capabilities: { agent_control: !!options.physical, control: true, camera: true },
    agent_control: {
      schema_version: 1,
      available: !!options.physical,
      owner: null as string | null,
      authority_epoch: 0,
      target_session_id: null as string | null,
      active_execution_id: null,
      units: jointUnits,
      limits: {},
      lease_seconds: 2,
      cartesian_control: false,
      collision_avoidance: false,
      contact_force_control: false,
      image_sensor_timestamp_verified: false,
    },
    session: {
      mode: 'idle',
      generation: 7,
      robot_id: null as string | null,
      port: null as string | null,
      role: 'follower',
      positions: Object.fromEntries(joints.map((joint) => [joint, 0])),
      targets: {},
      active_joints: [] as string[],
      torque_enabled: false,
      watchdog_seconds: 2,
      error: null,
    },
    calibrations: [
      {
        robot_id: 'fixture-arm',
        role: 'follower',
        valid: true,
        setup_fresh: !options.staleCalibration,
        setup_port: '/dev/fixture',
        sha256: 'b'.repeat(64),
        path: 'fixture-only',
      },
    ],
  };
  let admissionCount = 0;
  let releaseManualHeartbeat: (() => void) | null = null;
  let acknowledgeManualHeartbeat: (() => void) | null = null;
  const manualHeartbeatHandled = new Promise<void>((resolve) => {
    acknowledgeManualHeartbeat = resolve;
  });
  let releaseAdmissionImages: () => void = () => {};
  const admissionImagesReady = new Promise<void>((resolve) => {
    releaseAdmissionImages = resolve;
  });
  const pixel = Buffer.from(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jw1cAAAAASUVORK5CYII=',
    'base64',
  );
  function propose(requestId: string) {
    saved.calls_used += 1;
    saved.status = 'waiting_for_acceptance';
    saved.pending_proposal = {
      id: 'proposal-exact',
      request_id: requestId,
      agent_session_id: saved.id,
      observation_id: observation.id,
      observation_sha256: observation.sha256,
      authority_epoch: saved.authority_epoch,
      expected_revision: 0,
      provider_revision: 'mock-v1',
      model: 'deterministic-mock',
      skill: saved.target_kind === 'robot_lab' ? 'joint_segment' : 'joint_correction',
      purpose: 'A bounded next action toward the stated goal',
      arguments:
        saved.target_kind === 'robot_lab'
          ? { targets: { shoulder_pan: 1 }, duration_seconds: 1 }
          : { joint: 'shoulder', delta_rad: 0.03 },
      status: 'proposed',
      created_at: stamp,
      expires_at: new Date(Date.now() + 120000).toISOString(),
    };
  }
  let sessions: (typeof saved)[] = [];
  let openaiKeySaved = false;
  const apiConnection = (id: string) => ({
    id,
    configured: id === 'openai' && openaiKeySaved,
    source: id === 'openai' && openaiKeySaved ? 'saved' : null,
    message: null,
  });
  let profiles = [
    {
      id: 'mock',
      provider: 'mock',
      model: 'deterministic-mock',
      reasoning_effort: 'low',
      revision: 'mock-v1',
      saved: false,
      checked: true,
      capabilities: ['named_joint_correction'],
      message: 'Built-in replay provider',
    },
  ];
  await page.route('**/api/v1/**', async (route) => {
    if (await fulfillApplicationUpdateRead(route)) return;
    if (await fulfillWorkJobRead(route)) return;
    const request = route.request(),
      path = new URL(request.url()).pathname.replace('/api/v1', '');
    if (request.method() !== 'GET') requests.push({ path, body: request.postData() ? request.postDataJSON() : {} });
    else reads.push(path);
    if (path === '/robot-lab/status') return route.fulfill({ json: robot });
    if (path === '/robot-lab/devices')
      return route.fulfill({
        json: {
          ports: options.emptyPorts
            ? []
            : [
                ...(options.alternatePort ? [{ device: '/dev/other-arm', description: 'Other synthetic USB' }] : []),
                { device: '/dev/fixture', description: 'Synthetic USB' },
              ],
          cameras: [{ index: 0, name: 'Synthetic local camera', width: 1, height: 1 }],
          camera_error: null,
        },
      });
    if (path === '/robot-lab/camera.jpg' || path.endsWith('/image')) {
      if (options.cameraFailure && path === '/robot-lab/camera.jpg') return route.fulfill({ status: 503, body: '' });
      if (options.holdAdmissionImages && path.includes('/observations/admission-')) await admissionImagesReady;
      return route.fulfill({ contentType: 'image/png', body: pixel });
    }
    if (path === '/robot-lab/control/start') {
      robot.session.mode = 'manual';
      robot.session.robot_id = 'fixture-arm';
      robot.session.port = '/dev/fixture';
      robot.session.active_joints = joints;
      robot.session.torque_enabled = true;
      return route.fulfill({ json: robot });
    }
    if (path === '/robot-lab/control/heartbeat') {
      if (options.deferManualHeartbeat && !releaseManualHeartbeat)
        await new Promise<void>((resolve) => {
          releaseManualHeartbeat = resolve;
        });
      if (robot.session.mode !== 'manual' || request.postDataJSON().expected_generation !== robot.session.generation) {
        await route.fulfill({ status: 422, json: { detail: 'Controller ownership changed' } });
        acknowledgeManualHeartbeat?.();
        return;
      }
      return route.fulfill({ json: robot });
    }
    if (path === '/robot-lab/stop') {
      const body = request.postDataJSON();
      if (body.expected_generation !== undefined && body.expected_generation !== robot.session.generation)
        return route.fulfill({ status: 422, json: { detail: 'Controller generation changed' } });
      if (body.expected_mode !== undefined && body.expected_mode !== robot.session.mode)
        return route.fulfill({ status: 422, json: { detail: 'Controller mode changed' } });
      robot.session.mode = 'idle';
      robot.session.torque_enabled = false;
      robot.session.generation += 1;
      return route.fulfill({ json: robot });
    }
    if (path === '/api-connections')
      return route.fulfill({ json: { providers: ['openai', 'gemini'].map(apiConnection) } });
    if (path === '/api-connections/openai' && request.method() === 'PUT') {
      openaiKeySaved = true;
      return route.fulfill({ json: apiConnection('openai') });
    }
    if (path === '/agent/providers') return route.fulfill({ json: profiles });
    if (path === '/agent/openai/models') return route.fulfill({ json: ['gpt-6-astra', 'gpt-6.1-sol'] });
    if (path === '/agent/providers/openai' && request.method() === 'PUT') {
      profiles = [
        ...profiles.filter((p) => p.id !== 'openai'),
        {
          id: 'openai',
          provider: 'openai_responses',
          model: request.postDataJSON().model,
          reasoning_effort: 'medium',
          revision: 'profile-v1',
          saved: true,
          checked: false,
          capabilities: [],
          message: 'Saved; access unverified',
        },
      ];
      return route.fulfill({ json: profiles.at(-1) });
    }
    if (path === '/projects/agent-project/agent-sessions') {
      if (request.method() === 'POST') {
        const body = request.postDataJSON();
        saved.task = body.task;
        saved.planning_mode = body.planning_mode;
        saved.mode = body.mode;
        saved.max_calls = body.max_calls;
        saved.max_steps = body.max_steps;
        if (body.target_kind === 'robot_lab') {
          saved.target_kind = 'robot_lab';
          saved.clock_policy = 'physical';
          saved.authority_epoch = 42;
          saved.control_armed = true;
          saved.supported_skills = ['joint_segment'];
          observation.environment_kind = 'physical';
          observation.units = 'mixed';
          observation.joints = robot.session.positions;
          observation.joint_units = jointUnits;
          observation.authority_epoch = 42;
          observation.image_sha256 = 'c'.repeat(64);
          observation.calibration_sha256 = 'b'.repeat(64);
          observation.expires_at = new Date(Date.now() - 1000).toISOString();
          robot.session.mode = 'agent';
          robot.agent_control.owner = saved.id;
          robot.agent_control.authority_epoch = 42;
          robot.agent_control.target_session_id = 'physical-fixture';
          if (options.deferManualHeartbeat) {
            releaseManualHeartbeat?.();
            await manualHeartbeatHandled;
            await new Promise((resolve) => setTimeout(resolve, 150));
          }
        }
        sessions = [saved];
      }
      return route.fulfill({ json: request.method() === 'POST' ? saved : sessions });
    }
    if (path === '/agent-sessions/agent-fixture/proposals') {
      propose(request.postDataJSON().request_id);
      return route.fulfill({ status: 202, json: saved });
    }
    if (path === '/agent-sessions/agent-fixture/run') {
      saved.run_enabled = true;
      propose('task-loop-fixture');
      return route.fulfill({ status: 202, json: saved });
    }
    if (path === '/agent-sessions/agent-fixture/heartbeat') return route.fulfill({ json: saved });
    if (path === '/agent-sessions/agent-fixture/execution-observation') {
      saved.execution_observation = {
        ...observation,
        id: `admission-${++admissionCount}`,
        sha256: 'd'.repeat(64),
        captured_at: new Date().toISOString(),
        expires_at: new Date(Date.now() + 1000).toISOString(),
      };
      return route.fulfill({ json: saved });
    }
    if (path === '/agent-sessions/agent-fixture/stop' || path === '/agent-sessions/agent-fixture/pause') {
      saved.status = 'stopped';
      saved.control_armed = false;
      saved.run_enabled = false;
      saved.authority_epoch += 1;
      saved.updated_at = new Date().toISOString();
      saved.pending_proposal = null;
      robot.session.mode = 'idle';
      robot.session.torque_enabled = false;
      robot.agent_control.owner = null;
      return route.fulfill({ json: saved });
    }
    if (path === '/agent-sessions/agent-fixture/review') {
      const reviewed = saved.execution_observation ?? saved.observation;
      if (request.postDataJSON().observation_id !== reviewed.id)
        return route.fulfill({
          status: 409,
          json: { detail: 'Review must identify the retained current observation' },
        });
      saved.task_review = { ...request.postDataJSON(), source: 'operator', reviewed_at: new Date().toISOString() };
      saved.task_success_verified = request.postDataJSON().outcome === 'success';
      return route.fulfill({ json: saved });
    }
    if (path === '/agent-sessions/agent-fixture/approve') {
      saved.status =
        saved.planning_mode === 'task' ? (saved.target_kind === 'robot_lab' ? 'paused' : 'complete') : 'ready';
      saved.run_enabled = false;
      saved.step_count += 1;
      if (saved.planning_mode === 'task')
        saved.model_assessment = {
          outcome: 'achieved',
          summary: 'The object appears lifted in this synthetic scenario.',
        };
      saved.receipts = [
        {
          id: 'receipt-fixture',
          proposal_id: 'proposal-exact',
          status: saved.target_kind === 'robot_lab' ? 'completed' : 'replayed',
          arguments: { joint: 'shoulder', delta_rad: 0.03 },
          applied_targets: {},
          pre_observation_id: observation.id,
          post_observation_id: null,
          elapsed_seconds: 0,
          reason: 'Recorded proposal replay; no actuator dispatch',
          task_success_verified: false,
        },
      ];
      saved.pending_proposal = null;
      return route.fulfill({ json: saved });
    }
    if (path === '/agent-sessions/agent-fixture') return route.fulfill({ json: saved });
    if (path === '/agent-sessions/agent-fixture/events') return route.fulfill({ json: [] });
    const replies: Record<string, unknown> = {
      '/health': { status: 'ok', version: 'test' },
      '/capabilities': [],
      '/modules': installedModules,
      '/projects': [{ id: 'agent-project', name: 'Agent project', created_at: stamp }],
      '/projects/agent-project/jobs': [],
      '/datasets': [],
      '/teaching/state': { connected: false, state: null, message: 'Fixture disconnected', voice_configured: false },
      '/teaching/voice/status': {
        broker_reachable: false,
        configuration_present: false,
        dependencies_present: false,
        message: 'Not configured',
      },
      '/teaching/intelligence/status': { broker_reachable: false, configured: false, busy: false },
      '/teaching/intelligence/settings': { saved: false, revision: null, voice_model: null, message: 'Not saved' },
      '/projects/agent-project/recordings/options': {
        configured: false,
        runtime_verified: false,
        configuration_sha256: null,
        max_episodes: 100,
        max_source_bytes: 8589934592,
        setup_message: 'Not configured',
      },
    };
    return route.fulfill(
      path in replies
        ? { json: replies[path] }
        : { status: 404, json: { detail: 'Not configured in this browser fixture' } },
    );
  });
  return { requests, reads, saved, robot, releaseAdmissionImages };
}

async function startReplay(page: Page) {
  await page.goto('/ai-teleoperation/');
  await openModuleTask(page);
  await page.getByRole('combobox', { name: 'Agent target', exact: true }).selectOption('replay');
  await expect(page.getByRole('combobox', { name: 'Agent target', exact: true })).toBeVisible();
  await page.getByLabel('Task goal', { exact: true }).fill('Move shoulder by 0.03 radians');
  await page.getByRole('combobox', { name: 'Planning style', exact: true }).selectOption('joint');
  await page.getByRole('button', { name: 'Create agent session', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Agent session evidence' })).toBeVisible();
  await page.getByRole('button', { name: 'Plan one correction', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Pending agent proposal' })).toContainText('"delta_rad": 0.03');
}

test('replay requires acceptance of the exact proposal and labels execution evidence truthfully', async ({ page }) => {
  const { requests } = await fixture(page);
  await startReplay(page);
  await expect(page.getByRole('button', { name: 'Accept this proposal', exact: true })).toBeEnabled();
  expect(requests.filter((request) => request.path.endsWith('/approve'))).toEqual([]);
  await page.getByRole('button', { name: 'Accept this proposal', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Latest execution receipt' })).toContainText('Replay receipt');
  await expect(page.getByRole('region', { name: 'Latest execution receipt' })).toContainText(
    'task success remains unverified',
  );
  expect(requests.find((request) => request.path.endsWith('/approve'))?.body).toEqual({
    proposal_id: 'proposal-exact',
    authority_epoch: 1,
  });
  expect(requests.filter((request) => request.path.endsWith('/proposals'))).toHaveLength(1);
  const dimensions = await page.evaluate(() => [
    document.documentElement.scrollWidth,
    document.documentElement.clientWidth,
  ]);
  expect(dimensions[0]).toBeLessThanOrEqual(dimensions[1] + 1);
});

test('a stale controller epoch disables acceptance even if the old proposal remains visible', async ({ page }) => {
  const { requests, saved } = await fixture(page);
  await startReplay(page);
  saved.authority_epoch = 2;
  saved.status = 'paused';
  await page.getByRole('button', { name: 'Refresh agent sessions', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Accept this proposal', exact: true })).toBeDisabled();
  await expect(page.getByRole('region', { name: 'Pending agent proposal' })).toContainText('Expired or superseded');
  expect(requests.filter((request) => request.path.endsWith('/approve'))).toEqual([]);
});

test('shared OpenAI key stays private and saving a provider profile does not imply a capability check', async ({
  page,
}) => {
  const { requests } = await fixture(page);
  await page.goto('/cloud-apis/');
  const card = page.getByRole('region', { name: 'OpenAI', exact: true });
  const key = card.getByLabel('OpenAI API key', { exact: true });
  const fakeKey = 'sk-BrowserFixtureOnlyNotARealKey12345678';
  await expect(key).toHaveAttribute('type', 'password');
  await key.fill(fakeKey);
  await card.getByRole('button', { name: 'Save API key', exact: true }).click();
  await expect(key).toHaveValue('');
  await card.getByText('Advanced OpenAI profiles', { exact: true }).click();
  const panel = card.getByRole('region', { name: 'Agent provider settings', exact: true });
  await panel.getByRole('button', { name: 'Save provider profile', exact: true }).click();
  await expect(panel.getByRole('region', { name: 'Provider openai', exact: true })).toContainText(
    'Capabilities unverified',
  );
  await expect(panel.getByRole('button', { name: 'Check openai', exact: true })).toBeDisabled();
  expect(requests.filter((request) => request.path.endsWith('/check'))).toEqual([]);
  expect(requests.find((request) => request.path === '/api-connections/openai')?.body.api_key).toBe(fakeKey);
  expect(requests.find((request) => request.path === '/agent/providers/openai')?.body).not.toHaveProperty('api_key');
  expect(
    await page.evaluate(() => JSON.stringify({ local: { ...localStorage }, session: { ...sessionStorage } })),
  ).not.toContain(fakeKey);
  await expect(panel).not.toContainText(fakeKey);
});

test('an abstract goal starts the bounded task loop and model completion needs separate operator review', async ({
  page,
}) => {
  const { requests } = await fixture(page, { planningMode: 'task' });
  await page.goto('/ai-teleoperation/');
  await openModuleTask(page);
  await expect(page.getByRole('combobox', { name: 'Agent target', exact: true })).toBeVisible();
  await page.getByRole('combobox', { name: 'Agent target', exact: true }).selectOption('replay');
  const goal = 'Pick up the object and place it in the tray';
  await page.getByLabel('Task goal', { exact: true }).fill(goal);
  await page.getByRole('button', { name: 'Create agent session', exact: true }).click();
  await page.getByRole('button', { name: 'Start task', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Pending agent proposal' })).toContainText('A bounded next action');
  await page.getByRole('button', { name: 'Accept this proposal', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Astra task assessment' })).toContainText('achieved');
  await expect(page.getByRole('region', { name: 'Astra task assessment' })).toContainText(
    'does not verify task success',
  );
  expect(requests.find((request) => request.path === '/projects/agent-project/agent-sessions')?.body).toMatchObject({
    task: goal,
    planning_mode: 'task',
    max_steps: 12,
  });
  expect(requests.filter((request) => request.path.endsWith('/run'))).toHaveLength(1);
  expect(requests.filter((request) => request.path.endsWith('/review'))).toEqual([]);
  await page.getByText('Review task outcome', { exact: true }).click();
  await page.getByRole('combobox', { name: 'Reviewed outcome', exact: true }).selectOption('success');
  const save = page.getByRole('button', { name: 'Save operator review', exact: true });
  await expect(save).toBeDisabled();
  await page.getByLabel('I reviewed this retained observation and am reporting the task outcome myself.').check();
  await save.click();
  await expect(page.getByRole('region', { name: 'Agent session evidence' })).toContainText('Operator reported success');
  expect(requests.find((request) => request.path.endsWith('/review'))?.body).toMatchObject({
    observation_id: 'obs-fixture',
    authority_epoch: 1,
    outcome: 'success',
  });
});

test('selecting physical control without a governed companion never enables motion or starts a heartbeat', async ({
  page,
}) => {
  const { requests } = await fixture(page);
  await page.goto('/ai-teleoperation/');
  await openModuleTask(page);
  await expect(page.getByRole('combobox', { name: 'Agent target', exact: true })).toBeVisible();
  await page.getByLabel('Task goal', { exact: true }).fill('Pick up the object');
  await page.getByRole('combobox', { name: 'Agent target', exact: true }).selectOption('replay');
  await page.getByRole('combobox', { name: 'Planning style', exact: true }).selectOption('joint');
  await page.getByRole('combobox', { name: 'Agent target', exact: true }).selectOption('robot_lab');
  await expect(page.getByRole('combobox', { name: 'Planning style', exact: true })).toHaveValue('task');
  await expect(page.getByRole('region', { name: 'Physical SO101 preflight' })).toContainText('has not advertised');
  await expect(page.getByRole('button', { name: 'Connect arm and hold current pose', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Arm AI session', exact: true })).toBeDisabled();
  expect(requests.filter((request) => request.path.startsWith('/robot-lab/'))).toEqual([]);
});

async function connectPhysical(page: Page) {
  await page.goto('/ai-teleoperation/');
  await openModuleTask(page);
  await expect(page.getByRole('combobox', { name: 'Agent target', exact: true })).toBeVisible();
  await page.getByLabel('Task goal', { exact: true }).fill('Pick up the object');
  await page.getByRole('combobox', { name: 'Agent target', exact: true }).selectOption('robot_lab');
  await page.getByRole('combobox', { name: 'Robot USB port', exact: true }).selectOption('/dev/fixture');
  await page.getByRole('combobox', { name: 'Saved arm', exact: true }).selectOption('fixture-arm');
  await page.getByRole('combobox', { name: 'Physical camera', exact: true }).selectOption('0');
  await page.getByLabel('Show the local camera preview', { exact: true }).check();
  await expect
    .poll(() =>
      page.getByAltText('Local SO101 camera preview').evaluate((node) => (node as HTMLImageElement).naturalWidth),
    )
    .toBe(1);
  await page.getByLabel('I am at the arm, supporting it where needed, with Stop within reach.').check();
  await page
    .getByLabel('The arm and its full movement path are clear; the selected camera shows the intended workspace.')
    .check();
  await page.getByRole('button', { name: 'Connect arm and hold current pose', exact: true }).click();
}

test('saved arm names choose their own connection and selecting a camera starts its visible preview', async ({
  page,
}) => {
  const { requests } = await fixture(page, { physical: true, alternatePort: true });
  await page.goto('/ai-teleoperation/');
  await openModuleTask(page);
  const camera = page.getByRole('region', { name: 'Workspace camera', exact: true });
  await expect(camera).toContainText('Select a camera to preview your workspace');
  await page.getByRole('combobox', { name: 'Saved arm', exact: true }).selectOption('fixture-arm');
  await expect(page.getByRole('combobox', { name: 'Robot USB port', exact: true })).toHaveValue('/dev/fixture');
  await page.getByRole('combobox', { name: 'Physical camera', exact: true }).selectOption('0');
  await expect(page.getByLabel('Show the local camera preview', { exact: true })).toBeChecked();
  await expect(camera.getByAltText('Local SO101 camera preview')).toBeVisible();
  expect(requests).toEqual([]);
});

test('a disconnected saved arm explains the missing connection without starting motion', async ({ page }) => {
  const { requests } = await fixture(page, { physical: true, emptyPorts: true });
  await page.goto('/ai-teleoperation/');
  await openModuleTask(page);
  await page.getByRole('combobox', { name: 'Saved arm', exact: true }).selectOption('fixture-arm');
  await expect(page.getByRole('combobox', { name: 'Robot USB port', exact: true })).toHaveValue('');
  await expect(page.getByText('The saved connection (/dev/fixture) is unavailable.', { exact: false })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Connect arm and hold current pose', exact: true })).toBeDisabled();
  expect(requests).toEqual([]);
});

test('a genuinely stale selected connection stays blocked and explains why', async ({ page }) => {
  const { requests } = await fixture(page, { physical: true, staleCalibration: true });
  await page.goto('/ai-teleoperation/');
  await openModuleTask(page);
  await page.getByRole('combobox', { name: 'Saved arm', exact: true }).selectOption('fixture-arm');
  await expect(page.getByText('Motor setup changed on this arm’s saved connection.', { exact: false })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Connect arm and hold current pose', exact: true })).toBeDisabled();
  expect(requests).toEqual([]);
});

test('a failed preview never admits a physical connection', async ({ page }) => {
  const { requests } = await fixture(page, { physical: true, cameraFailure: true });
  await page.goto('/ai-teleoperation/');
  await openModuleTask(page);
  await page.getByRole('combobox', { name: 'Saved arm', exact: true }).selectOption('fixture-arm');
  await page.getByRole('combobox', { name: 'Physical camera', exact: true }).selectOption('0');
  await expect(page.getByText('The physical camera preview is unavailable.', { exact: true })).toBeVisible();
  await page.getByLabel('I am at the arm, supporting it where needed, with Stop within reach.').check();
  await page
    .getByLabel('The arm and its full movement path are clear; the selected camera shows the intended workspace.')
    .check();
  await expect(page.getByRole('button', { name: 'Connect arm and hold current pose', exact: true })).toBeDisabled();
  expect(requests).toEqual([]);
});

test('OpenAI model names load explicitly and save their exact model IDs', async ({ page }) => {
  const { requests, reads } = await fixture(page);
  await page.goto('/cloud-apis/');
  await page.getByText('Advanced OpenAI profiles', { exact: true }).click();
  expect(reads).not.toContain('/agent/openai/models');
  await page.getByRole('button', { name: 'Load available OpenAI models', exact: true }).click();
  const models = page.getByRole('combobox', { name: 'OpenAI model', exact: true });
  await expect(models.getByRole('option', { name: 'GPT-6.1 Sol · gpt-6.1-sol', exact: true })).toBeAttached();
  await models.selectOption({ label: 'GPT-6.1 Sol · gpt-6.1-sol' });
  await page.getByRole('button', { name: 'Save provider profile', exact: true }).click();
  expect(requests.find((request) => request.path === '/agent/providers/openai')?.body.model).toBe('gpt-6.1-sol');
  expect(requests.filter((request) => request.path.endsWith('/check'))).toEqual([]);
  await page.goto('/ai-teleoperation/');
  await openModuleTask(page);
  await expect(page.getByRole('combobox', { name: 'Agent provider profile', exact: true })).toContainText(
    'GPT-6.1 Sol · openai',
  );
});

test('physical arming, current-image admission and focus loss retain explicit operator authority', async ({ page }) => {
  const { requests, releaseAdmissionImages } = await fixture(page, {
    planningMode: 'task',
    physical: true,
    holdAdmissionImages: true,
  });
  await connectPhysical(page);
  await expect(page.getByRole('button', { name: 'Arm AI session', exact: true })).toBeEnabled();
  expect(requests.find((request) => request.path === '/robot-lab/control/start')?.body).toMatchObject({
    mode: 'manual',
    move_to_middle: false,
    confirm: 'ENABLE MOTION',
  });
  await page.getByRole('button', { name: 'Arm AI session', exact: true }).click();
  await page.getByRole('button', { name: 'Start task', exact: true }).click();
  const accept = page.getByRole('button', { name: 'Accept this proposal', exact: true });
  await expect(accept).toBeDisabled();
  const sceneConsent = page.getByLabel(
    'I reviewed the current image and proposed targets; the object, workspace and path remain suitable for this exact action.',
  );
  await expect(page.getByRole('region', { name: 'Current physical admission observation' })).toContainText(
    'admission-',
  );
  await expect(sceneConsent).toBeDisabled();
  releaseAdmissionImages();
  await expect(page.getByRole('region', { name: 'Current physical admission observation' })).toContainText(
    'Current image loaded',
  );
  await page
    .getByLabel(
      'I reviewed the current image and proposed targets; the object, workspace and path remain suitable for this exact action.',
    )
    .check();
  await accept.click();
  const approval = requests.find((request) => request.path.endsWith('/approve'))?.body;
  expect(approval).toMatchObject({
    proposal_id: 'proposal-exact',
    authority_epoch: 42,
    scene_unchanged_confirmed: true,
    execution_observation_sha256: 'd'.repeat(64),
  });
  expect(String(approval?.execution_observation_id)).toMatch(/^admission-/);
  await expect
    .poll(() => requests.filter((request) => request.path === '/agent-sessions/agent-fixture/heartbeat').length)
    .toBeGreaterThan(0);
  expect(
    requests
      .filter((request) => request.path === '/agent-sessions/agent-fixture/heartbeat')
      .every((request) => request.body.authority_epoch === 42 && request.body.operator_present === true),
  ).toBe(true);
  await page.getByText('Review task outcome', { exact: true }).click();
  await expect(page.getByAltText('Task outcome review observation')).toBeVisible();
  await page.getByRole('combobox', { name: 'Reviewed outcome', exact: true }).selectOption('failure');
  await page.getByLabel('I reviewed this retained observation and am reporting the task outcome myself.').check();
  await page.getByRole('button', { name: 'Save operator review', exact: true }).click();
  expect(String(requests.find((request) => request.path.endsWith('/review'))?.body.observation_id)).toMatch(
    /^admission-/,
  );
  await page.evaluate(() => window.dispatchEvent(new Event('blur')));
  await expect
    .poll(() => requests.filter((request) => request.path === '/agent-sessions/agent-fixture/stop').length)
    .toBeGreaterThan(0);
  await expect(page.getByRole('region', { name: 'Physical SO101 preflight' })).toContainText('lost focus');
});

test('a rejected in-flight manual heartbeat cannot stop a legitimate AI ownership transfer', async ({ page }) => {
  const { requests } = await fixture(page, { physical: true, deferManualHeartbeat: true });
  await connectPhysical(page);
  await expect.poll(() => requests.some((request) => request.path === '/robot-lab/control/heartbeat')).toBe(true);
  await page.getByRole('button', { name: 'Arm AI session', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Agent session evidence' })).toContainText('agent-fixture');
  await expect
    .poll(() => requests.some((request) => request.path === '/agent-sessions/agent-fixture/heartbeat'))
    .toBe(true);
  expect(requests.filter((request) => request.path === '/robot-lab/stop')).toEqual([]);
  expect(requests.filter((request) => request.path === '/agent-sessions/agent-fixture/stop')).toEqual([]);
});

test('implicit manual cleanup is fenced against a newer controller', async ({ page }) => {
  const { requests, robot } = await fixture(page, { physical: true });
  await connectPhysical(page);
  robot.session.generation = 8;
  await page.evaluate(() => window.dispatchEvent(new Event('blur')));
  await expect.poll(() => requests.some((request) => request.path === '/robot-lab/stop')).toBe(true);
  expect(requests.find((request) => request.path === '/robot-lab/stop')?.body).toMatchObject({
    expected_generation: 7,
    expected_mode: 'manual',
  });
  expect(robot.session.generation).toBe(8);
  expect(robot.session.torque_enabled).toBe(true);
});

test('implicit preflight cleanup cannot stop an agent that took over the same generation', async ({ page }) => {
  const { requests, robot } = await fixture(page, { physical: true });
  await connectPhysical(page);
  robot.session.mode = 'agent';
  robot.agent_control.owner = 'different-controller';
  await page.evaluate(() => window.dispatchEvent(new Event('blur')));
  await expect.poll(() => requests.some((request) => request.path === '/robot-lab/stop')).toBe(true);
  expect(requests.find((request) => request.path === '/robot-lab/stop')?.body).toMatchObject({
    expected_generation: 7,
    expected_mode: 'manual',
  });
  expect(robot.session.mode).toBe('agent');
  expect(robot.session.torque_enabled).toBe(true);
});

test('AI teleoperation opens physical preparation without starting a simulator, camera or controller', async ({
  page,
}) => {
  const { requests, reads } = await fixture(page, { physical: true, emptyPorts: true });
  await page.goto('/ai-teleoperation/');
  await openModuleTask(page);
  await expect(page.getByRole('combobox', { name: 'Agent target', exact: true })).toHaveValue('robot_lab');
  await expect(page.getByLabel('Task goal', { exact: true })).toBeVisible();
  const preflight = page.getByRole('region', { name: 'Physical SO101 preflight' });
  await expect(preflight).toBeVisible();
  await expect(preflight).toContainText('Connect the follower USB adapter and refresh readiness');
  await expect(page.getByText('The mock provider cannot plan a live physical task.', { exact: false })).toBeVisible();
  await expect(page.getByRole('link', { name: 'Cloud & APIs → OpenAI', exact: true })).toHaveAttribute(
    'href',
    '/cloud-apis/',
  );
  await expect(page.getByRole('combobox', { name: 'Saved arm', exact: true })).toHaveValue('');
  await expect(page.getByLabel('Show the local camera preview', { exact: true })).not.toBeChecked();
  await expect(page.getByRole('region', { name: 'Simulation tools' })).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Recording dataset preparation', exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Connect arm and hold current pose', exact: true })).toBeDisabled();
  expect(reads.some((path) => path.startsWith('/teaching') || path.includes('camera.jpg'))).toBe(false);
  expect(requests).toEqual([]);
  await page.getByRole('link', { name: 'Cloud & APIs → OpenAI', exact: true }).click();
  await expect(page).toHaveURL(/\/cloud-apis\/$/);
  const openai = page.getByRole('region', { name: 'OpenAI', exact: true });
  await openai.getByText('Advanced OpenAI profiles', { exact: true }).click();
  await expect(page.getByRole('region', { name: 'Agent provider settings', exact: true })).toBeVisible();
  expect(requests).toEqual([]);
});

test('Agent Control stays visible while simulation and replay are selected', async ({ page }) => {
  const { requests } = await fixture(page);
  await page.goto('/ai-teleoperation/');
  await openModuleTask(page);
  const target = page.getByRole('combobox', { name: 'Agent target', exact: true });
  await target.selectOption('teaching');
  await expect(page.getByRole('region', { name: 'Simulation tools', exact: true })).toBeVisible();
  await expect(page.getByRole('region', { name: 'AI teleoperation connection', exact: true })).toBeVisible();
  const controls = page.getByRole('region', { name: 'Agent Control', exact: true });
  const heading = controls.getByRole('heading', { name: 'Agent Control', exact: true });
  await expect(page.getByRole('button', { name: 'Agent Control', exact: true })).toHaveCount(0);
  await page.getByRole('textbox', { name: 'Task goal', exact: true }).fill('Inspect the current scene');
  await heading.click();
  await expect(target).toBeVisible();
  await expect(target).toHaveValue('teaching');
  await expect(page.getByRole('region', { name: 'Simulation tools', exact: true })).toBeVisible();
  await target.selectOption('replay');
  await expect(page.getByRole('region', { name: 'Simulation tools', exact: true })).toHaveCount(0);
  await heading.click();
  await expect(target).toBeVisible();
  await expect(target).toHaveValue('replay');
  await expect(page.getByRole('textbox', { name: 'Task goal', exact: true })).toHaveValue('Inspect the current scene');
  expect(requests).toEqual([]);
});

test('switching targets or returning to physical preparation clears operator confirmations and preview', async ({
  page,
}) => {
  const { requests } = await fixture(page, { physical: true });
  await page.goto('/ai-teleoperation/');
  await openModuleTask(page);
  const present = page.getByLabel('I am at the arm, supporting it where needed, with Stop within reach.');
  const clear = page.getByLabel(
    'The arm and its full movement path are clear; the selected camera shows the intended workspace.',
  );
  await page.getByRole('combobox', { name: 'Physical camera', exact: true }).selectOption('0');
  await page.getByLabel('Show the local camera preview', { exact: true }).check();
  await expect(page.getByAltText('Local SO101 camera preview')).toBeVisible();
  await present.check();
  await clear.check();
  const target = page.getByRole('combobox', { name: 'Agent target', exact: true });
  await target.selectOption('teaching');
  await target.selectOption('robot_lab');
  await expect(present).not.toBeChecked();
  await expect(clear).not.toBeChecked();
  await expect(page.getByLabel('Show the local camera preview', { exact: true })).not.toBeChecked();
  await expect(page.getByAltText('Local SO101 camera preview')).toHaveCount(0);
  await present.check();
  await clear.check();
  await page.goto('/dashboard/');
  await page.goto('/ai-teleoperation/');
  await openModuleTask(page);
  await expect(page.getByRole('combobox', { name: 'Agent target', exact: true })).toHaveValue('robot_lab');
  await expect(present).not.toBeChecked();
  await expect(clear).not.toBeChecked();
  await expect(page.getByLabel('Show the local camera preview', { exact: true })).not.toBeChecked();
  expect(requests).toEqual([]);
});
