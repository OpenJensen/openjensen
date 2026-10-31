import { apiOrigin, ApiError } from './api';

export const robotJoints = [
  { name: 'shoulder_pan', id: 1, label: 'Shoulder pan', unit: 'degrees' },
  { name: 'shoulder_lift', id: 2, label: 'Shoulder lift', unit: 'degrees' },
  { name: 'elbow_flex', id: 3, label: 'Elbow', unit: 'degrees' },
  { name: 'wrist_flex', id: 4, label: 'Wrist flex', unit: 'degrees' },
  { name: 'wrist_roll', id: 5, label: 'Wrist roll', unit: 'degrees' },
  { name: 'gripper', id: 6, label: 'Gripper', unit: 'percent' },
] as const;
export type RobotJoint = (typeof robotJoints)[number]['name'];
export type RobotRole = 'follower' | 'leader';
export type ControlMode = 'manual' | 'leader' | 'handguide';
export type JointValues = Partial<Record<RobotJoint, number>>;
export type RobotDataset = {
  id: string;
  path: string;
  episodes: number;
  frames: number;
  fps: number;
  task: string;
  format: string;
  finalized: boolean;
  abandoned?: boolean;
  pending_frames?: number;
};
export type RobotDiagnosticEvent = {
  id: string;
  time: string;
  event: string;
  reason?: string;
  message?: string;
  torque_disable_acknowledged?: boolean;
  cleanup_errors?: string[];
  context?: { mode: string; robot_id: string | null; fault?: { source: string; message: string } | null };
};
export type RobotDiagnostics = {
  enabled: boolean;
  directory: string;
  error: string | null;
  dropped_entries: number;
  retention_bytes: number;
  last_stop: RobotDiagnosticEvent | null;
  events: RobotDiagnosticEvent[];
};
export type MotorCalibration = {
  id: number;
  drive_mode: number;
  homing_offset: number;
  range_min: number;
  range_max: number;
};
export type CalibrationDocument = {
  format: 'openjensen.so101.calibration';
  version: 1;
  robot_id: string;
  role: RobotRole;
  calibration: Record<RobotJoint, MotorCalibration>;
};
export type RobotPolicyState = {
  phase: 'idle' | 'loading' | 'ready' | 'checking' | 'running' | 'error';
  policy_id: string | null;
  name: string | null;
  error: string | null;
  fps: number | null;
  steps: number;
  inference_ms: number | null;
  preflight: {
    passed: boolean;
    camera_index: number;
    checked_at: number;
    fps: number;
    measured_fps: number;
    mean_latency_ms: number;
    max_latency_ms: number;
    fresh_frames: number;
    samples: number;
    message: string;
  } | null;
  device: string | null;
  limits: { max_speed: number; max_acceleration: number; max_run_seconds: number; operator_heartbeat_seconds: number };
};
export type RobotPolicyCatalog = {
  policies: {
    id: string;
    name: string;
    compatible: boolean;
    message: string;
    fps?: number;
    camera_key?: string;
    image_shape?: number[];
  }[];
  runtime: { configured: boolean; device: string | null; message: string };
  state: RobotPolicyState;
};
export type RobotStatus = {
  calibration_transfer?: { robot_id: string; role: RobotRole; verified: boolean; backup_created: boolean };
  diagnostics?: RobotDiagnostics;
  policy?: RobotPolicyState;
  controller?: {
    name: string;
    upstream: string;
    target_mode: string;
    max_relative_target: number | null;
    acceleration: number;
    maximum_acceleration: number;
    goal_velocity: number;
    encoder_range_tolerance_ticks?: number;
  };
  ok?: boolean;
  connected?: boolean;
  label?: string;
  message?: string;
  version?: string;
  platform?: string;
  dependencies?: Record<string, { installed: boolean; version: string | null }>;
  capabilities?: {
    persistent_diagnostics?: boolean;
    agent_control?: boolean;
    run_policy?: boolean;
    lerobot_position_control?: boolean;
    large_joint_steps?: boolean;
    arm_readiness?: boolean;
    joint_sliders?: boolean;
    calibration_persistence?: boolean;
    calibration_transfer?: boolean;
    device_registry?: boolean;
    device_removal?: boolean;
    partial_control?: boolean;
    rest_position?: boolean;
    move_to_middle?: boolean;
    middle_tolerance?: boolean;
    assembled_setup?: boolean;
    mapping_verification?: boolean;
    probe: boolean;
    camera: boolean;
    motor_setup: boolean;
    calibration: boolean;
    control: boolean;
    recording: boolean;
    auto_recording_fps?: boolean;
    guided_recording?: boolean;
  };
  session?: {
    mode: 'idle' | 'calibration' | 'setup' | 'verification' | 'agent' | 'policy' | ControlMode;
    generation?: number;
    robot_id: string | null;
    port: string | null;
    role: RobotRole | null;
    positions: JointValues;
    active_joints?: RobotJoint[];
    targets?: JointValues;
    slider_targets?: JointValues;
    joint_limits?: Partial<Record<RobotJoint, { min: number; max: number }>>;
    torque_enabled: boolean;
    rest_position?: {
      saved: boolean;
      phase: 'idle' | 'returning' | 'holding' | 'error';
      progress: number;
      targets: JointValues;
    };
    centering?: { active: boolean; progress: number; tolerance?: number };
    watchdog_seconds: number | null;
    error: string | null;
  };
  agent_control?: {
    schema_version: 1;
    available: boolean;
    owner: string | null;
    authority_epoch: number;
    target_session_id: string | null;
    active_execution_id: string | null;
    units: Partial<Record<RobotJoint, 'degrees' | 'percent'>>;
    limits: Record<string, number>;
    lease_seconds: number;
    cartesian_control: false;
    collision_avoidance: false;
    contact_force_control: false;
    image_sensor_timestamp_verified: false;
  };
  calibration?: {
    phase: 'idle' | 'center' | 'ranges' | 'saved' | 'interrupted';
    recoverable?: boolean;
    robot_id?: string;
    role?: RobotRole;
    port?: string;
    interrupted_stage?: 'center' | 'ranges';
    error?: string | null;
    mins: JointValues;
    maxes: JointValues;
    offsets: JointValues;
    path: string | null;
  };
  recording?: {
    active: boolean;
    dataset_id: string | null;
    episode_active: boolean;
    saving?: boolean;
    frames: number;
    episodes: number;
    fps: number;
    fps_mode?: 'auto' | 'manual';
    rate_status?: 'measuring' | 'ready' | 'error';
    rate_message?: string;
    actual_fps?: number | null;
    collection?: {
      enabled: boolean;
      saving?: boolean;
      phase: 'idle' | 'countdown' | 'recording' | 'saving' | 'reset' | 'paused' | 'complete';
      episode_time_s: number;
      reset_time_s: number;
      num_episodes: number;
      countdown_s: number;
      remaining_s: number;
      completed: number;
      cue_id: number;
      cue: string;
    } | null;
    error: string | null;
  };
  assembled?: {
    port?: string | null;
    phase:
      | 'idle'
      | 'scanning'
      | 'splitting'
      | 'ready'
      | 'observing'
      | 'wiggling'
      | 'review'
      | 'applying'
      | 'complete'
      | 'error'
      | 'stopped';
    motion_active?: boolean;
    wiggle_id?: number | null;
    wiggle_ticks?: number | null;
    ids: number[];
    selected_joint: RobotJoint | null;
    detected_id: number | null;
    candidate_delta: number | null;
    mapping: Partial<Record<RobotJoint, number>>;
    message: string;
    error: string | null;
    log: string[];
    progress: unknown;
    calibration_required: boolean;
  };
  verification?: {
    phase: 'idle' | 'checking' | 'wiggling' | 'done' | 'error' | 'stopped';
    port: string | null;
    joint: RobotJoint | null;
    motor_id: number | null;
    ticks: number | null;
    motion_active: boolean;
    error: string | null;
    run_id: string | null;
  };
  mapping_checks?: {
    port: string | null;
    revision: string | null;
    checks: Partial<Record<RobotJoint, { run_id: string; correct: boolean; checked_at: number }>>;
    error: string | null;
  };
  calibrations?: {
    robot_id: string;
    role: RobotRole;
    path: string;
    valid?: boolean;
    setup_port?: string | null;
    setup_fresh?: boolean;
    sha256?: string;
    error?: string | null;
  }[];
  datasets?: RobotDataset[];
};
export type RobotDevice = {
  id: string;
  port: string;
  serial_number: string | null;
  name: string;
  robot_id: string;
  role: RobotRole | null;
  profile_role: RobotRole | null;
  connected: boolean;
  description: string;
  motor_ids?: boolean;
  calibration?: 'saved' | 'needs_attention' | 'missing';
};
export type RobotDevices = {
  saved_devices?: RobotDevice[];
  removed_devices?: Pick<RobotDevice, 'id' | 'name' | 'port' | 'robot_id'>[];
  ports: { device: string; description: string; manufacturer: string | null; serial_number: string | null }[];
  cameras: { index: number; name: string; width: number; height: number }[];
  camera_error: string | null;
};
export type RobotProbe = {
  checked_at?: number;
  scan_scope?: 'expected_ids' | 'all_addresses';
  unexpected_ids?: number[];
  profile?: {
    robot_id: string;
    role: RobotRole;
    sha256?: string;
    state: 'missing' | 'invalid' | 'stale' | 'unavailable' | 'matched' | 'mismatch';
    joints: Partial<Record<RobotJoint, boolean>>;
    error?: string | null;
  };
  port: string;
  baudrate: number;
  motors: { id: number; model: number; stable: boolean; responses: number }[];
  ready: boolean;
  expected_ids: number[];
  warning: string | null;
};
export type MotorAssignment = { assigned: { joint: RobotJoint; id: number; baudrate: number }; warning: string | null };

async function request<T>(
  path: string,
  body?: unknown,
  options: { timeoutMs?: number; keepalive?: boolean } = {},
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), options.timeoutMs ?? 20_000);
  try {
    const response = await fetch(`${apiOrigin}/api/v1/robot-lab${path}`, {
      method: body === undefined ? 'GET' : 'POST',
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      cache: 'no-store',
      signal: controller.signal,
      keepalive: options.keepalive,
    });
    if (!response.ok) {
      let message = `Robot Lab request failed (${response.status}).`;
      try {
        const payload = await response.json();
        message =
          typeof payload.detail === 'string'
            ? payload.detail
            : typeof payload.error === 'string'
              ? payload.error
              : message;
      } catch {
        /* Preserve the response status. */
      }
      throw new ApiError(message, response.status);
    }
    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError')
      throw new Error('The robot companion did not respond in time. Check the connection before trying again.', {
        cause: error,
      });
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

export const robotApi = {
  status: () => request<RobotStatus>('/status'),
  policyCatalog: () => request<RobotPolicyCatalog>('/policy/catalog'),
  loadPolicy: (policy_id: string) => request<RobotStatus>('/policy/load', { policy_id }, { timeoutMs: 125_000 }),
  preflightPolicy: (camera_index: number) =>
    request<RobotStatus>('/policy/preflight', { camera_index }, { timeoutMs: 30_000 }),
  startPolicy: (input: { expected_generation: number; camera_index: number; duration_seconds: number }) =>
    request<RobotStatus>('/policy/start', {
      ...input,
      confirm: 'RUN POLICY',
      operator_present: true,
      workspace_clear: true,
    }),
  policyHeartbeat: (expected_generation: number) =>
    request<RobotStatus>('/policy/heartbeat', { expected_generation }, { timeoutMs: 2_000 }),
  diagnostics: () => request<RobotDiagnostics>('/diagnostics'),
  devices: () => request<RobotDevices>('/devices', undefined, { timeoutMs: 30_000 }),
  configureDevice: (input: { id: string; port: string; name: string; role: RobotRole }) =>
    request<RobotDevices>('/devices/configure', input),
  removeDevice: (id: string) => request<RobotDevices>('/devices/remove', { id }),
  restoreDevice: (id: string) => request<RobotDevices>('/devices/restore', { id }),
  datasets: () => request<{ datasets: RobotDataset[] }>('/datasets'),
  probe: (port: string, robot_id?: string, role: RobotRole = 'follower', discover = false) =>
    request<RobotProbe>(
      '/probe',
      { port, ...(robot_id ? { robot_id, role } : {}), ...(discover ? { discover: true } : {}) },
      { timeoutMs: 90_000 },
    ),
  assignMotor: (port: string, joint: RobotJoint) =>
    request<MotorAssignment>(
      '/setup/motor',
      { port, joint, isolated: true, confirm: 'ASSIGN MOTOR ID' },
      { timeoutMs: 90_000 },
    ),
  wiggleSaved: (port: string, joint: RobotJoint) =>
    request<RobotStatus>('/mapping/wiggle', { port, joint, supported: true, confirm: 'WIGGLE SAVED MOTOR' }),
  confirmMapping: (port: string, joint: RobotJoint, run_id: string, correct: boolean) =>
    request<RobotStatus>('/mapping/confirm', { port, joint, run_id, correct, confirm: 'CONFIRM PHYSICAL JOINT' }),
  startAssembled: (port: string) =>
    request<RobotStatus>('/setup/assembled/start', { port, supported: true, confirm: 'START ASSEMBLED SETUP' }),
  wiggleAssembled: (motor_id: number) =>
    request<RobotStatus>('/setup/assembled/wiggle', { motor_id, supported: true, confirm: 'WIGGLE MOTOR' }),
  acceptWiggle: (joint: RobotJoint, motor_id: number) =>
    request<RobotStatus>('/setup/assembled/accept', {
      joint,
      motor_id,
      method: 'wiggle',
      confirm: 'CONFIRM OBSERVED JOINT',
    }),
  observeAssembled: (joint: RobotJoint) => request<RobotStatus>('/setup/assembled/observe', { joint }),
  acceptAssembled: (joint: RobotJoint, motor_id: number) =>
    request<RobotStatus>('/setup/assembled/accept', { joint, motor_id }),
  finishAssembled: () => request<RobotStatus>('/setup/assembled/finish', { confirm: 'SAVE MOTOR IDS' }),
  cancelAssembled: () => request<RobotStatus>('/setup/assembled/cancel', {}),
  exportCalibration: (robot_id: string, role: RobotRole) =>
    request<CalibrationDocument>(`/calibration/export?${new URLSearchParams({ robot_id, role })}`),
  importCalibration: (input: {
    robot_id: string;
    role: RobotRole;
    document: unknown;
    port?: string;
    replace_existing: boolean;
  }) => request<RobotStatus>('/calibration/import', { ...input, confirm: 'IMPORT CALIBRATION' }),
  startCalibration: (port: string, robot_id: string, role: RobotRole) =>
    request<RobotStatus>(
      '/calibration/start',
      { port, robot_id, role, confirm: 'START CALIBRATION' },
      { timeoutMs: 45_000 },
    ),
  resumeCalibration: () =>
    request<RobotStatus>('/calibration/start', { resume: true, confirm: 'START CALIBRATION' }, { timeoutMs: 45_000 }),
  centerCalibration: () => request<RobotStatus>('/calibration/center', {}),
  finishCalibration: () => request<RobotStatus>('/calibration/finish', { confirm: 'SAVE CALIBRATION' }),
  cancelCalibration: () => request<RobotStatus>('/calibration/cancel', {}),
  startControl: (input: {
    port: string;
    robot_id: string;
    mode: ControlMode;
    joints?: RobotJoint[];
    move_to_middle?: boolean;
    middle_tolerance?: number;
    leader_port?: string;
    leader_id?: string;
  }) => request<RobotStatus>('/control/start', { ...input, confirm: 'ENABLE MOTION' }, { timeoutMs: 45_000 }),
  saveRestPosition: () => request<RobotStatus>('/control/rest/save', { confirm: 'SAVE REST POSITION' }),
  returnToRest: () => request<RobotStatus>('/control/rest/return', { confirm: 'RETURN TO REST' }),
  heartbeat: (expectedGeneration?: number) =>
    request<RobotStatus>(
      '/control/heartbeat',
      expectedGeneration === undefined ? {} : { expected_generation: expectedGeneration },
      { timeoutMs: 2_000 },
    ),
  jog: (joint: RobotJoint, delta: number) =>
    request<RobotStatus>('/control/jog', { joint, delta }, { timeoutMs: 4_000 }),
  targetJoint: (joint: RobotJoint, target: number) =>
    request<RobotStatus>('/control/jog', { joint, target }, { timeoutMs: 4_000 }),
  stop: (
    keepalive = false,
    expectedGeneration?: number,
    expectedMode?: 'manual',
    cause?: 'page_hidden' | 'page_closed',
  ) =>
    request<RobotStatus>(
      '/stop',
      {
        ...(expectedGeneration === undefined ? {} : { expected_generation: expectedGeneration }),
        ...(expectedMode === undefined ? {} : { expected_mode: expectedMode }),
        ...(cause === undefined ? {} : { cause }),
      },
      { timeoutMs: 10_000, keepalive },
    ),
  startRecording: (input: {
    dataset_id: string;
    task: string;
    fps: number | null;
    camera_index: number | null;
    collection?: { episode_time_s: number; reset_time_s: number; num_episodes: number; countdown_s: number };
  }) => request<RobotStatus>('/recording/start', input, { timeoutMs: 90_000 }),
  collectionAction: (action: 'start' | 'pause' | 'resume' | 'next' | 'rerecord') =>
    request<RobotStatus>(`/recording/collection/${action}`, {}, { timeoutMs: 90_000 }),
  startEpisode: () => request<RobotStatus>('/recording/episode/start', {}),
  saveEpisode: () => request<RobotStatus>('/recording/episode/save', {}, { timeoutMs: 90_000 }),
  discardEpisode: () => request<RobotStatus>('/recording/episode/discard', {}),
  finishRecording: () => request<RobotStatus>('/recording/finish', {}, { timeoutMs: 90_000 }),
  abandonRecording: () =>
    request<RobotStatus>(
      '/recording/finish',
      { abandon_empty: true, confirm: 'ABANDON EMPTY RECORDING' },
      { timeoutMs: 90_000 },
    ),
  cameraUrl: (index: number, stamp: number) => `${apiOrigin}/api/v1/robot-lab/camera.jpg?index=${index}&t=${stamp}`,
};

export async function importRobotDataset(
  projectId: string,
  datasetId: string,
): Promise<{ id: string; project_id: string }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 300_000);
  try {
    const response = await fetch(
      `${apiOrigin}/api/v1/projects/${encodeURIComponent(projectId)}/robot-lab/datasets/${encodeURIComponent(datasetId)}/import`,
      {
        method: 'POST',
        cache: 'no-store',
        headers: { 'Content-Type': 'application/json' },
        body: '{}',
        signal: controller.signal,
      },
    );
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new ApiError(
        typeof payload.detail === 'string' ? payload.detail : `Dataset import failed (${response.status}).`,
        response.status,
      );
    }
    return await response.json();
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError')
      throw new Error(
        'Dataset import is taking longer than expected. Check the project dataset library before retrying.',
        { cause: error },
      );
    throw error;
  } finally {
    clearTimeout(timer);
  }
}
