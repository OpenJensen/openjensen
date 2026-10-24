import { apiOrigin, ApiError } from './api';

export const robotJoints = [
  { name: 'shoulder_pan', id: 1, label: 'Shoulder pan', unit: 'degrees' },
  { name: 'shoulder_lift', id: 2, label: 'Shoulder lift', unit: 'degrees' },
  { name: 'elbow_flex', id: 3, label: 'Elbow', unit: 'degrees' },
  { name: 'wrist_flex', id: 4, label: 'Wrist flex', unit: 'degrees' },
  { name: 'wrist_roll', id: 5, label: 'Wrist roll', unit: 'degrees' },
  { name: 'gripper', id: 6, label: 'Gripper', unit: 'percent' },
] as const;
export type RobotJoint = typeof robotJoints[number]['name'];
export type RobotRole = 'follower' | 'leader';
export type ControlMode = 'manual' | 'leader' | 'handguide';
export type JointValues = Partial<Record<RobotJoint, number>>;
export type RobotDataset = { id: string; path: string; episodes: number; frames: number; fps: number; task: string; format: string; finalized: boolean; abandoned?: boolean; pending_frames?: number };
export type RobotStatus = {
  ok?: boolean; connected?: boolean; label?: string; message?: string; version?: string; platform?: string;
  dependencies?: Record<string, { installed: boolean; version: string | null }>;
  capabilities?: { joint_sliders?: boolean; calibration_persistence?: boolean; partial_control?: boolean; move_to_middle?: boolean; assembled_setup?: boolean; mapping_verification?: boolean; probe: boolean; camera: boolean; motor_setup: boolean; calibration: boolean; control: boolean; recording: boolean };
  session?: { mode: 'idle' | 'calibration' | 'setup' | 'verification' | ControlMode; robot_id: string | null; port: string | null; role: RobotRole | null; positions: JointValues; active_joints?: RobotJoint[]; targets?: JointValues; slider_targets?: JointValues; joint_limits?: Partial<Record<RobotJoint, { min: number; max: number }>>; torque_enabled: boolean; centering?: { active: boolean; progress: number }; watchdog_seconds: number | null; error: string | null };
  calibration?: { phase: 'idle' | 'center' | 'ranges' | 'saved' | 'interrupted'; recoverable?: boolean; robot_id?: string; role?: RobotRole; port?: string; interrupted_stage?: 'center' | 'ranges'; error?: string | null; mins: JointValues; maxes: JointValues; offsets: JointValues; path: string | null };
  recording?: { active: boolean; dataset_id: string | null; episode_active: boolean; frames: number; episodes: number; fps: number; error: string | null };
  assembled?: { port?: string | null; phase: 'idle' | 'scanning' | 'splitting' | 'ready' | 'observing' | 'wiggling' | 'review' | 'applying' | 'complete' | 'error' | 'stopped'; motion_active?: boolean; wiggle_id?: number | null; wiggle_ticks?: number | null; ids: number[]; selected_joint: RobotJoint | null; detected_id: number | null; candidate_delta: number | null; mapping: Partial<Record<RobotJoint, number>>; message: string; error: string | null; log: string[]; progress: unknown; calibration_required: boolean };
  verification?: { phase: 'idle' | 'checking' | 'wiggling' | 'done' | 'error' | 'stopped'; port: string | null; joint: RobotJoint | null; motor_id: number | null; ticks: number | null; motion_active: boolean; error: string | null; run_id: string | null };
  calibrations?: { robot_id: string; role: RobotRole; path: string }[];
  datasets?: RobotDataset[];
};
export type RobotDevices = { ports: { device: string; description: string; manufacturer: string | null; serial_number: string | null }[]; cameras: { index: number; name: string; width: number; height: number }[]; camera_error: string | null };
export type RobotProbe = { port: string; baudrate: number; motors: { id: number; model: number; stable: boolean; responses: number }[]; ready: boolean; expected_ids: number[]; warning: string | null };
export type MotorAssignment = { assigned: { joint: RobotJoint; id: number; baudrate: number }; warning: string | null };

async function request<T>(path: string, body?: unknown, options: { timeoutMs?: number; keepalive?: boolean } = {}): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), options.timeoutMs ?? 20_000);
  try {
    const response = await fetch(`${apiOrigin}/api/v1/robot-lab${path}`, {
      method: body === undefined ? 'GET' : 'POST',
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      cache: 'no-store', signal: controller.signal, keepalive: options.keepalive,
    });
    if (!response.ok) {
      let message = `Robot Lab request failed (${response.status}).`;
      try { const payload = await response.json(); message = typeof payload.detail === 'string' ? payload.detail : typeof payload.error === 'string' ? payload.error : message; } catch { /* Preserve the response status. */ }
      throw new ApiError(message, response.status);
    }
    return await response.json() as T;
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') throw new Error('The Mac companion did not respond in time. Check the connection before trying again.');
    throw error;
  } finally { clearTimeout(timer); }
}

export const robotApi = {
  status: () => request<RobotStatus>('/status'),
  devices: () => request<RobotDevices>('/devices', undefined, { timeoutMs: 30_000 }),
  datasets: () => request<{ datasets: RobotDataset[] }>('/datasets'),
  probe: (port: string) => request<RobotProbe>('/probe', { port }, { timeoutMs: 90_000 }),
  assignMotor: (port: string, joint: RobotJoint) => request<MotorAssignment>('/setup/motor', { port, joint, isolated: true, confirm: 'ASSIGN MOTOR ID' }, { timeoutMs: 90_000 }),
  wiggleSaved: (port: string, joint: RobotJoint) => request<RobotStatus>('/mapping/wiggle', { port, joint, supported: true, confirm: 'WIGGLE SAVED MOTOR' }),
  startAssembled: (port: string) => request<RobotStatus>('/setup/assembled/start', { port, supported: true, confirm: 'START ASSEMBLED SETUP' }),
  wiggleAssembled: (motor_id: number) => request<RobotStatus>('/setup/assembled/wiggle', { motor_id, supported: true, confirm: 'WIGGLE MOTOR' }),
  acceptWiggle: (joint: RobotJoint, motor_id: number) => request<RobotStatus>('/setup/assembled/accept', { joint, motor_id, method: 'wiggle', confirm: 'CONFIRM OBSERVED JOINT' }),
  observeAssembled: (joint: RobotJoint) => request<RobotStatus>('/setup/assembled/observe', { joint }),
  acceptAssembled: (joint: RobotJoint, motor_id: number) => request<RobotStatus>('/setup/assembled/accept', { joint, motor_id }),
  finishAssembled: () => request<RobotStatus>('/setup/assembled/finish', { confirm: 'SAVE MOTOR IDS' }),
  cancelAssembled: () => request<RobotStatus>('/setup/assembled/cancel', {}),
  startCalibration: (port: string, robot_id: string, role: RobotRole) => request<RobotStatus>('/calibration/start', { port, robot_id, role, confirm: 'START CALIBRATION' }, { timeoutMs: 45_000 }),
  resumeCalibration: () => request<RobotStatus>('/calibration/start', { resume: true, confirm: 'START CALIBRATION' }, { timeoutMs: 45_000 }),
  centerCalibration: () => request<RobotStatus>('/calibration/center', {}),
  finishCalibration: () => request<RobotStatus>('/calibration/finish', { confirm: 'SAVE CALIBRATION' }),
  cancelCalibration: () => request<RobotStatus>('/calibration/cancel', {}),
  startControl: (input: { port: string; robot_id: string; mode: ControlMode; joints?: RobotJoint[]; move_to_middle?: boolean; leader_port?: string; leader_id?: string }) => request<RobotStatus>('/control/start', { ...input, confirm: 'ENABLE MOTION' }, { timeoutMs: 45_000 }),
  heartbeat: () => request<RobotStatus>('/control/heartbeat', {}, { timeoutMs: 2_000 }),
  jog: (joint: RobotJoint, delta: number) => request<RobotStatus>('/control/jog', { joint, delta }, { timeoutMs: 4_000 }),
  targetJoint: (joint: RobotJoint, target: number) => request<RobotStatus>('/control/jog', { joint, target }, { timeoutMs: 4_000 }),
  stop: (keepalive = false) => request<RobotStatus>('/stop', {}, { timeoutMs: 10_000, keepalive }),
  startRecording: (input: { dataset_id: string; task: string; fps: number; camera_index: number | null }) => request<RobotStatus>('/recording/start', input, { timeoutMs: 90_000 }),
  startEpisode: () => request<RobotStatus>('/recording/episode/start', {}),
  saveEpisode: () => request<RobotStatus>('/recording/episode/save', {}, { timeoutMs: 90_000 }),
  discardEpisode: () => request<RobotStatus>('/recording/episode/discard', {}),
  finishRecording: () => request<RobotStatus>('/recording/finish', {}, { timeoutMs: 90_000 }),
  abandonRecording: () => request<RobotStatus>('/recording/finish', { abandon_empty: true, confirm: 'ABANDON EMPTY RECORDING' }, { timeoutMs: 90_000 }),
  cameraUrl: (index: number, stamp: number) => `${apiOrigin}/api/v1/robot-lab/camera.jpg?index=${index}&t=${stamp}`,
};

export async function importRobotDataset(projectId: string, datasetId: string): Promise<{ id: string; project_id: string }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 300_000);
  try {
    const response = await fetch(`${apiOrigin}/api/v1/projects/${encodeURIComponent(projectId)}/robot-lab/datasets/${encodeURIComponent(datasetId)}/import`, { method: 'POST', cache: 'no-store', headers: { 'Content-Type': 'application/json' }, body: '{}', signal: controller.signal });
    if (!response.ok) { const payload = await response.json().catch(() => ({})); throw new ApiError(typeof payload.detail === 'string' ? payload.detail : `Dataset import failed (${response.status}).`, response.status); }
    return await response.json();
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') throw new Error('Dataset import is taking longer than expected. Check the project dataset library before retrying.');
    throw error;
  } finally { clearTimeout(timer); }
}
