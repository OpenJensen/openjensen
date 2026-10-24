'use client';

import { useEffect, useRef, useState, type ReactNode } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Icon } from '@/components/icon';
import { useWorkspace } from '@/components/workspace-context';
import {
  importRobotDataset,
  robotApi,
  robotJoints,
  type ControlMode,
  type JointValues,
  type RobotDataset,
  type RobotJoint,
  type RobotProbe,
  type RobotRole,
  type RobotStatus,
} from '@/lib/robot-api';
import { So101JointGuide } from './so101-joint-guide';
import { Notice as SharedNotice } from './ui/notice';
import { describeHardwareError } from '@/lib/hardware-error';
import './robot-lab.css';

const steps = [
  { title: 'Connect' },
  { title: 'Motor IDs' },
  { title: 'Calibrate' },
  { title: 'Teleoperate' },
  { title: 'Record' },
];
const setupOrder: RobotJoint[] = ['gripper', 'wrist_roll', 'wrist_flex', 'elbow_flex', 'shoulder_lift', 'shoulder_pan'];
const statusKey = ['robot-lab', 'status'];
const number = (value: number | undefined, digits = 0) =>
  value === undefined || !Number.isFinite(value) ? '—' : value.toFixed(digits);
const readable = (error: unknown) => (error instanceof Error ? error.message : 'The operation could not be completed.');
const isStatus = (value: unknown): value is RobotStatus => !!value && typeof value === 'object' && 'session' in value;

function Heading({ title }: { title: string }) {
  return (
    <div className="robot-lab-panel-heading">
      <h2>{title}</h2>
    </div>
  );
}
function Notice({ children, kind = 'info' }: { children: ReactNode; kind?: 'info' | 'warning' | 'success' }) {
  return <SharedNotice tone={kind}>{children}</SharedNotice>;
}
function HardwareNotice({ message, label }: { message: string; label?: string }) {
  const described = describeHardwareError(message);
  return (
    <SharedNotice tone="danger" live="alert" label={label} title={described.title} details={described.details}>
      {described.next}
    </SharedNotice>
  );
}
function RangeTable({
  positions = {},
  mins = {},
  maxes = {},
  calibration = false,
}: {
  positions?: JointValues;
  mins?: JointValues;
  maxes?: JointValues;
  calibration?: boolean;
}) {
  return (
    <div className="robot-lab-table-wrap">
      <table className="robot-lab-table">
        <thead>
          <tr>
            <th>Joint</th>
            <th>{calibration ? 'Current · ticks' : 'Position'}</th>
            {calibration && (
              <>
                <th>Min</th>
                <th>Max</th>
                <th>Range</th>
              </>
            )}
          </tr>
        </thead>
        <tbody>
          {robotJoints.map((joint) => (
            <tr key={joint.name}>
              <td>{joint.label}</td>
              <td>
                {number(positions[joint.name], calibration ? 0 : 1)}
                {!calibration && positions[joint.name] !== undefined ? (joint.unit === 'percent' ? '%' : '°') : ''}
              </td>
              {calibration && (
                <>
                  <td>{number(mins[joint.name])}</td>
                  <td>{number(maxes[joint.name])}</td>
                  <td>
                    {joint.name === 'wrist_roll'
                      ? 'Full rotation'
                      : mins[joint.name] !== undefined && maxes[joint.name] !== undefined
                        ? maxes[joint.name]! - mins[joint.name]! < 100
                          ? 'Move joint'
                          : number(maxes[joint.name]! - mins[joint.name]!)
                        : '—'}
                  </td>
                </>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function JointControl({
  joint,
  position,
  target,
  sliderTarget,
  limits,
  sliders,
  disabled,
  step,
  recording = false,
  onJog,
  onTarget,
}: {
  joint: (typeof robotJoints)[number];
  position?: number;
  target?: number;
  sliderTarget?: number;
  limits?: { min: number; max: number };
  sliders: boolean;
  disabled: boolean;
  step: number;
  recording?: boolean;
  onJog: (joint: RobotJoint, delta: number) => Promise<void>;
  onTarget: (joint: RobotJoint, target: number) => Promise<void>;
}) {
  const [draft, setDraft] = useState<number | null>(null);
  const committing = useRef(false);
  const unit = joint.unit === 'percent' ? '%' : '°';
  const min = limits ? Math.ceil(limits.min * 10) / 10 : 0;
  const max = limits ? Math.floor(limits.max * 10) / 10 : 0;
  const desired = draft ?? sliderTarget ?? target ?? position ?? 0;
  async function commit(value: number) {
    if (disabled || draft === null || committing.current) return;
    committing.current = true;
    try {
      await onTarget(joint.name, value);
    } finally {
      committing.current = false;
      setDraft(null);
    }
  }
  return (
    <div className="robot-lab-joint">
      <strong>{joint.label}</strong>
      <span className="robot-lab-joint-readings">
        <output aria-label={`${joint.label} position`}>
          {number(position, 1)}
          {unit}
        </output>
        {target !== undefined && (
          <output aria-label={`${joint.label} target`}>
            → {number(desired, 1)}
            {unit}
          </output>
        )}
      </span>
      <button
        aria-label={`${recording ? 'Record decrease' : 'Decrease'} ${joint.label}`}
        disabled={disabled}
        onClick={() => void onJog(joint.name, -step)}
      >
        −
      </button>
      <button
        aria-label={`${recording ? 'Record increase' : 'Increase'} ${joint.label}`}
        disabled={disabled}
        onClick={() => void onJog(joint.name, step)}
      >
        +
      </button>
      {sliders && limits && max > min && (
        <div className="robot-lab-joint-slider">
          <span>
            {number(min, 1)}
            {unit}
          </span>
          <input
            type="range"
            aria-label={`${joint.label} target slider`}
            aria-valuetext={`${number(desired, 1)}${unit}`}
            min={min}
            max={max}
            step="0.1"
            value={Math.max(min, Math.min(max, desired))}
            disabled={disabled}
            title="Release to move"
            onChange={(event) => setDraft(Number(event.target.value))}
            onPointerUp={(event) => void commit(Number(event.currentTarget.value))}
            onPointerCancel={() => setDraft(null)}
            onBlur={() => {
              if (!committing.current) setDraft(null);
            }}
            onKeyUp={(event) => {
              if (
                ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', 'PageUp', 'PageDown'].includes(
                  event.key,
                )
              )
                void commit(Number(event.currentTarget.value));
            }}
          />
          <span>
            {number(max, 1)}
            {unit}
          </span>
        </div>
      )}
    </div>
  );
}

export function RobotLab() {
  const { projectId, navigateStage } = useWorkspace();
  const client = useQueryClient();
  const [step, setStep] = useState(0);
  const [port, setPort] = useState('');
  const [robotId, setRobotId] = useState('my-so101');
  const [role, setRole] = useState<RobotRole>('follower');
  const [leaderPort, setLeaderPort] = useState('');
  const [leaderId, setLeaderId] = useState('my-so101-leader');
  const [mode, setMode] = useState<ControlMode>('manual');
  const [isolated, setIsolated] = useState(false);
  const [assembledReady, setAssembledReady] = useState(false);
  const [reidentify, setReidentify] = useState(false);
  const [verificationReady, setVerificationReady] = useState(false);
  const [localMotionChecks, setMotionChecks] = useState<Partial<Record<RobotJoint, { run: string; correct: boolean }>>>(
    {},
  );
  const [checkingConnection, setCheckingConnection] = useState(false);
  const [identifyMethod, setIdentifyMethod] = useState<'hand' | 'wiggle'>('hand');
  const [wiggleReady, setWiggleReady] = useState(false);
  const [wiggleId, setWiggleId] = useState('');
  const [motorIndex, setMotorIndex] = useState(0);
  const [assigned, setAssigned] = useState<RobotJoint[]>([]);
  const [probe, setProbe] = useState<RobotProbe | null>(null);
  const [motionReady, setMotionReady] = useState(false);
  const [jogStep, setJogStep] = useState(1);
  const [middleTolerance, setMiddleTolerance] = useState(2);
  const [jogging, setJogging] = useState(false);
  const jogInFlight = useRef(false);
  const [controlJoints, setControlJoints] = useState<RobotJoint[]>(robotJoints.map((j) => j.name));
  const [cameraIndex, setCameraIndex] = useState(0);
  const [preview, setPreview] = useState(false);
  const [previewStamp, setPreviewStamp] = useState(0);
  const [cameraError, setCameraError] = useState(false);
  const [datasetId, setDatasetId] = useState('');
  const [task, setTask] = useState('');
  const [fps, setFps] = useState(10);
  const [recordCamera, setRecordCamera] = useState(true);
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [awaitingCalibration, setAwaitingCalibration] = useState<'ranges' | 'saved' | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [ownsSession, setOwnsSession] = useState(false);
  const [imported, setImported] = useState<{ dataset: string; projectId: string } | null>(null);
  const ownedRef = useRef(false);
  const localCalibrationRef = useRef(false);
  const lastConnectedState = useRef<RobotStatus | undefined>(undefined);
  const [reconnecting, setReconnecting] = useState(false);
  const epoch = useRef(0);
  const mounted = useRef(true);
  const status = useQuery({
    queryKey: statusKey,
    queryFn: robotApi.status,
    retry: false,
    refetchInterval: ownsSession ? 600 : 3_000,
    refetchOnWindowFocus: true,
  });
  if (status.data?.ok && status.data.connected !== false && !status.isError) lastConnectedState.current = status.data;
  const state =
    status.isError || status.data?.connected === false ? (lastConnectedState.current ?? status.data) : status.data;
  const online = !!status.data?.ok && status.data.connected !== false && !status.isError;
  const devices = useQuery({
    queryKey: ['robot-lab', 'devices'],
    queryFn: robotApi.devices,
    enabled: online,
    retry: false,
    staleTime: 30_000,
    refetchInterval: state?.session?.mode === 'idle' ? 5_000 : false,
  });
  const datasets = useQuery({
    queryKey: ['robot-lab', 'datasets'],
    queryFn: robotApi.datasets,
    enabled: online,
    retry: false,
    refetchInterval: 10_000,
  });
  const session = state?.session;
  const calibration = state?.calibration;
  localCalibrationRef.current = session?.mode === 'calibration' && !!state?.capabilities?.calibration_persistence;
  const calibrationStage = calibration?.phase === 'interrupted' ? calibration.interrupted_stage : calibration?.phase;
  const calibrationError = step === 2 ? error || calibration?.error || session?.error : null;
  const recording = state?.recording;
  const assembled = state?.assembled;
  const verification = state?.verification;
  const mappingComplete = assembled?.phase === 'complete' && assembled.port === port;
  const active = !!session && session.mode !== 'idle';
  const controlling = !!session && ['manual', 'leader', 'handguide'].includes(session.mode);
  const profileFor = (id: string, profileRole: RobotRole) =>
    state?.calibrations?.find((item) => item.robot_id === id && item.role === profileRole);
  const profileUsable = (profile: ReturnType<typeof profileFor>) =>
    !!profile &&
    profile.valid !== false &&
    (profile.setup_port === port
      ? profile.setup_fresh !== false
      : !mappingComplete || !assembled?.calibration_required);
  const selectedProfile = profileFor(robotId, role);
  const followerProfile = profileFor(robotId, 'follower');
  const followerMismatch =
    probe?.port === port &&
    probe.profile?.robot_id === robotId &&
    probe.profile.role === 'follower' &&
    probe.profile.state === 'mismatch' &&
    (!followerProfile?.sha256 || probe.profile.sha256 === followerProfile.sha256);
  const followerScanInvalid =
    probe?.port === port &&
    probe.profile?.robot_id === robotId &&
    probe.profile.role === 'follower' &&
    ['missing', 'invalid', 'stale'].includes(probe.profile.state) &&
    (!followerProfile?.sha256 || probe.profile.sha256 === followerProfile.sha256);
  const calibrated =
    session?.mode !== 'setup' && profileUsable(followerProfile) && !followerMismatch && !followerScanInvalid;
  const persistedMotionChecks = state?.capabilities?.arm_readiness
    ? state.mapping_checks?.port === port && mappingComplete
      ? (Object.fromEntries(
          Object.entries(state.mapping_checks.checks).map(([joint, check]) => [
            joint,
            { run: check!.run_id, correct: check!.correct },
          ]),
        ) as typeof localMotionChecks)
      : {}
    : localMotionChecks;
  const motionChecks = Object.fromEntries(
    Object.entries(persistedMotionChecks).filter(
      ([joint, check]) =>
        !(
          verification?.port === port &&
          verification.joint === joint &&
          ['checking', 'wiggling', 'done'].includes(verification.phase) &&
          verification.run_id !== check?.run
        ),
    ),
  ) as typeof localMotionChecks;
  const leaderCalibrated = profileFor(leaderId, 'leader')?.valid !== false && !!profileFor(leaderId, 'leader');
  const selectedMotor = robotJoints.find((item) => item.name === setupOrder[motorIndex])!;
  const availableCameras = devices.data?.cameras ?? [];
  const busy = !!pending;
  const canAct = online && !busy && !reconnecting;
  const portConnected =
    online && !!port && !devices.isError && !!devices.data?.ports.some((item) => item.device === port);
  const leaderConnected =
    online && !!leaderPort && !devices.isError && !!devices.data?.ports.some((item) => item.device === leaderPort);
  const canUseArm = canAct && portConnected;
  const usbStatus = devices.isPending
    ? 'Checking robot USB…'
    : devices.isError
      ? 'Robot USB status unavailable'
      : portConnected
        ? 'Robot USB connected · Idle'
        : port
          ? 'Robot USB disconnected'
          : 'Select a robot USB device';
  const datasetList = datasets.data?.datasets ?? state?.datasets ?? [];
  const missingHardwareTools = Object.entries(state?.dependencies ?? {}).filter(
    ([name, dependency]) => !dependency.installed && ['serial', 'scservo_sdk', 'cv2'].includes(name),
  );
  const unavailableFeatures = state?.capabilities && Object.values(state.capabilities).some((available) => !available);

  useEffect(() => {
    mounted.current = true;
    try {
      const saved = JSON.parse(localStorage.getItem('openjensen.robot-lab.preferences') ?? '{}');
      if (typeof saved.port === 'string') setPort(saved.port);
      if (typeof saved.robotId === 'string') setRobotId(saved.robotId);
      if (typeof saved.leaderPort === 'string') setLeaderPort(saved.leaderPort);
      if (typeof saved.leaderId === 'string') setLeaderId(saved.leaderId);
      if ([2, 5].includes(saved.middleTolerance)) setMiddleTolerance(saved.middleTolerance);
    } catch {
      /* Preferences are optional. */
    }
    return () => {
      mounted.current = false;
      if (ownedRef.current && !localCalibrationRef.current) void robotApi.stop(true).catch(() => undefined);
    };
  }, []);
  useEffect(() => {
    if (!devices.data) return;
    if (!port) setPort(devices.data.ports[0]?.device ?? '');
    const cameras = devices.data.cameras;
    if (!cameras.some((camera) => camera.index === cameraIndex)) setCameraIndex(cameras[0]?.index ?? 0);
  }, [devices.data, port, cameraIndex]);
  useEffect(() => {
    setIsolated(false);
    setAssembledReady(false);
    setReidentify(false);
    setVerificationReady(false);
    setMotionChecks({});
    setWiggleReady(false);
    setWiggleId('');
    setProbe(null);
    setAssigned([]);
    setMotionReady(false);
  }, [port]);
  useEffect(() => {
    setIsolated(false);
  }, [motorIndex]);
  useEffect(() => {
    setProbe(null);
  }, [robotId, role]);
  useEffect(() => {
    if (session?.mode === 'setup' || session?.mode === 'calibration') setProbe(null);
  }, [session?.mode]);
  useEffect(() => {
    if (!portConnected) {
      setVerificationReady(false);
      setMotionReady(false);
      setAssembledReady(false);
      setIsolated(false);
      setProbe(null);
    }
  }, [portConnected]);
  useEffect(() => {
    if (
      online &&
      session?.mode === 'calibration' &&
      state?.capabilities?.calibration_persistence &&
      !ownedRef.current
    ) {
      ownedRef.current = true;
      setOwnsSession(true);
      setStep(2);
      if (session.port) setPort(session.port);
      if (session.robot_id) setRobotId(session.robot_id);
      if (session.role) setRole(session.role);
      setError(null);
      setNotice(null);
      setReconnecting(false);
    }
  }, [
    online,
    session?.mode,
    session?.port,
    session?.robot_id,
    session?.role,
    state?.capabilities?.calibration_persistence,
  ]);
  useEffect(() => {
    if (
      awaitingCalibration &&
      online &&
      calibration?.phase === awaitingCalibration &&
      !calibration.error &&
      !session?.error
    ) {
      setError(null);
      setAwaitingCalibration(null);
      setNotice(
        awaitingCalibration === 'saved'
          ? 'Calibration saved.'
          : 'Center recorded. Move each joint through its comfortable range.',
      );
      if (awaitingCalibration === 'saved') {
        ownedRef.current = false;
        setOwnsSession(false);
      }
    }
  }, [awaitingCalibration, online, calibration?.phase, calibration?.error, session?.error]);
  useEffect(() => {
    if (!preview || !online) return;
    setCameraError(false);
    setPreviewStamp(Date.now());
    const timer = setInterval(() => setPreviewStamp(Date.now()), 700);
    return () => clearInterval(timer);
  }, [preview, online, cameraIndex]);
  useEffect(() => {
    if (!ownsSession) return;
    let stopped = false,
      inFlight = false;
    const heartbeat = async () => {
      if (localCalibrationRef.current || inFlight || stopped) return;
      inFlight = true;
      try {
        const next = await robotApi.heartbeat();
        if (!stopped && mounted.current) setReconnecting(false);
        if (!stopped && mounted.current && next.session?.mode === 'idle') {
          ownedRef.current = false;
          setOwnsSession(false);
          client.setQueryData(statusKey, next);
        }
      } catch {
        if (!stopped && mounted.current) {
          setReconnecting(true);
          void client.invalidateQueries({ queryKey: statusKey });
        }
      } finally {
        inFlight = false;
      }
    };
    const timer = setInterval(() => void heartbeat(), 500);
    const hide = () => {
      if (document.visibilityState === 'hidden' && ownedRef.current && !localCalibrationRef.current) {
        ownedRef.current = false;
        setOwnsSession(false);
        setNotice(null);
        setError('Robot Lab was hidden; stopping the session. Keep this page visible during setup or control.');
        void robotApi
          .stop(true)
          .then((next) => {
            if (mounted.current) client.setQueryData(statusKey, next);
          })
          .catch(() => undefined);
      }
    };
    document.addEventListener('visibilitychange', hide);
    return () => {
      stopped = true;
      clearInterval(timer);
      document.removeEventListener('visibilitychange', hide);
    };
  }, [ownsSession, client]);

  async function perform<T>(
    label: string,
    action: () => Promise<T>,
    done?: (result: T) => void,
    announce = true,
    calibrationTarget?: 'ranges' | 'saved',
  ) {
    const token = ++epoch.current;
    setPending(label);
    setError(null);
    setNotice(null);
    setAwaitingCalibration(null);
    try {
      const result = await action();
      if (!mounted.current || token !== epoch.current) return;
      if (isStatus(result)) {
        await client.cancelQueries({ queryKey: statusKey });
        client.setQueryData(statusKey, result);
      }
      done?.(result);
      setNotice(announce ? label : null);
      void client.invalidateQueries({ queryKey: ['robot-lab', 'datasets'] });
      void client.invalidateQueries({ queryKey: statusKey });
    } catch (failure) {
      if (mounted.current && token === epoch.current) {
        setAwaitingCalibration(calibrationTarget ?? null);
        setError(readable(failure));
        void client.invalidateQueries({ queryKey: statusKey });
      }
    } finally {
      if (mounted.current && token === epoch.current) setPending(null);
    }
  }
  function selectIllustratedJoint(joint: RobotJoint) {
    setMotorIndex(setupOrder.indexOf(joint));
    if (
      identifyMethod === 'hand' &&
      online &&
      ownsSession &&
      !busy &&
      session?.mode === 'setup' &&
      assembled &&
      assembled.mapping[joint] === undefined &&
      ['ready', 'observing', 'review'].includes(assembled.phase)
    )
      void perform(
        `Watching for ${robotJoints.find((item) => item.name === joint)!.label.toLowerCase()} hand movement.`,
        () => robotApi.observeAssembled(joint),
        undefined,
        false,
      );
  }
  const mappingValues = Object.values(assembled?.mapping ?? {});
  const allJointsMapped =
    robotJoints.every((joint) => typeof assembled?.mapping[joint.name] === 'number') &&
    new Set(mappingValues).size === 6;
  const selectedCandidate =
    assembled?.selected_joint === selectedMotor.name && typeof assembled.detected_id === 'number'
      ? assembled.detected_id
      : null;
  const detectionFeedback =
    identifyMethod === 'hand' &&
    online &&
    session?.mode === 'setup' &&
    assembled &&
    ['ready', 'observing', 'review'].includes(assembled.phase) ? (
      <div className="so101-detection-feedback" role="group" aria-label="Arm movement detection">
        {selectedCandidate !== null ? (
          <>
            <strong>Movement detected at address {selectedCandidate}</strong>
            <p>
              {assembled.candidate_delta === null
                ? ''
                : `${Math.abs(assembled.candidate_delta).toFixed(0)} encoder ticks changed. `}
              Hold the {selectedMotor.label.toLowerCase()} in its new position and confirm.
            </p>
            {selectedCandidate !== null && online && (
              <button
                className="secondary-button"
                disabled={!canAct || !ownsSession || assembled.phase !== 'observing'}
                onClick={() =>
                  void perform(
                    `${selectedMotor.label} address confirmed.`,
                    () => robotApi.acceptAssembled(selectedMotor.name, selectedCandidate),
                    () => {
                      const next = setupOrder.find(
                        (joint) => joint !== selectedMotor.name && assembled.mapping[joint] === undefined,
                      );
                      if (next) setMotorIndex(setupOrder.indexOf(next));
                    },
                  )
                }
              >
                Confirm ID {selectedCandidate} is the {selectedMotor.label.toLowerCase()}
              </button>
            )}
          </>
        ) : assembled.mapping[selectedMotor.name] !== undefined && assembled.selected_joint !== selectedMotor.name ? (
          <p>
            {selectedMotor.label} is confirmed at address {assembled.mapping[selectedMotor.name]}. Select the next
            joint.
          </p>
        ) : (
          <p>
            {assembled.selected_joint === selectedMotor.name && assembled.phase === 'observing'
              ? 'Move this joint gently, then hold its new position.'
              : 'Click the joint to start movement detection, then move it gently and hold its new position.'}
          </p>
        )}
      </div>
    ) : undefined;
  const jointGuide = (
    <So101JointGuide
      previewControls={!(step === 1 && mappingComplete && !reidentify)}
      feedback={detectionFeedback}
      selected={selectedMotor.name}
      onSelect={selectIllustratedJoint}
      setupActive={session?.mode === 'setup' && !['ready', 'observing', 'review'].includes(assembled?.phase ?? '')}
      detecting={
        identifyMethod === 'hand' &&
        online &&
        ownsSession &&
        assembled?.phase === 'observing' &&
        assembled.selected_joint === selectedMotor.name
      }
      calibrationActive={session?.mode === 'calibration' && !session.torque_enabled}
      measured={
        online &&
        session?.mode === 'calibration' &&
        (calibration?.phase === 'center' || calibration?.phase === 'ranges') &&
        Number.isFinite(session.positions[selectedMotor.name])
          ? {
              value: session.positions[selectedMotor.name]!,
              min: calibration.mins[selectedMotor.name],
              max: calibration.maxes[selectedMotor.name],
            }
          : undefined
      }
    />
  );
  const jointStep = state?.capabilities?.large_joint_steps ? jogStep : Math.min(jogStep, 2);
  const centering = !!session?.centering?.active;
  const partialSelection = controlJoints.length !== robotJoints.length;
  const sessionJoints = session?.active_joints ?? robotJoints.map((j) => j.name);
  const partialControl = controlling && sessionJoints.length !== robotJoints.length;
  const controlJointOptions = !active && mode === 'manual' && (
    <fieldset className="robot-lab-motor-selection" disabled={busy}>
      <legend>Motors in this session</legend>
      <div>
        {robotJoints.map((joint) => (
          <label key={joint.name}>
            <input
              type="checkbox"
              checked={controlJoints.includes(joint.name)}
              onChange={(event) => {
                setControlJoints((current) =>
                  event.target.checked ? [...current, joint.name] : current.filter((j) => j !== joint.name),
                );
                setMotionReady(false);
              }}
            />
            {joint.id} · {joint.label}
          </label>
        ))}
      </div>
      <button
        type="button"
        className="text-button"
        disabled={!canUseArm}
        onClick={() =>
          void perform(
            'Connected motors selected. Review the selection before enabling.',
            () => robotApi.probe(port),
            (result) => {
              setProbe(result);
              setControlJoints(
                robotJoints
                  .filter((j) => result.motors.some((m) => m.id === j.id && m.stable && m.model === 777))
                  .map((j) => j.name),
              );
              setMotionReady(false);
            },
          )
        }
      >
        Select connected motors
      </button>
      {partialSelection && <small>Selected motors hold their current pose. Recording requires all six.</small>}
    </fieldset>
  );
  function remember() {
    try {
      localStorage.setItem(
        'openjensen.robot-lab.preferences',
        JSON.stringify({ port, robotId, leaderPort, leaderId, middleTolerance }),
      );
    } catch {
      /* Preferences are optional. */
    }
  }
  async function jog(joint: RobotJoint, delta: number) {
    return commandJoint(() => robotApi.jog(joint, delta));
  }
  async function targetJoint(joint: RobotJoint, target: number) {
    return commandJoint(() => robotApi.targetJoint(joint, target));
  }
  async function commandJoint(action: () => Promise<RobotStatus>) {
    if (!canAct || !ownsSession || session?.mode !== 'manual' || centering || jogInFlight.current) return;
    const token = epoch.current;
    jogInFlight.current = true;
    setJogging(true);
    try {
      const next = await action();
      if (!mounted.current || token !== epoch.current) return;
      await client.cancelQueries({ queryKey: statusKey });
      if (!mounted.current || token !== epoch.current) return;
      client.setQueryData(statusKey, next);
      setError(null);
    } catch (failure) {
      if (mounted.current && token === epoch.current) {
        setError(readable(failure));
        void client.invalidateQueries({ queryKey: statusKey });
      }
    } finally {
      jogInFlight.current = false;
      if (mounted.current) setJogging(false);
    }
  }
  function ownSession() {
    setReconnecting(false);
    ownedRef.current = true;
    setOwnsSession(true);
    remember();
  }
  function stop() {
    ownedRef.current = false;
    setOwnsSession(false);
    setReconnecting(false);
    setMotionReady(false);
    setVerificationReady(false);
    void perform('Stop requested.', () => robotApi.stop());
  }
  async function refresh() {
    setCheckingConnection(true);
    setError(null);
    setNotice(null);
    try {
      await Promise.allSettled([status.refetch(), ...(online ? [devices.refetch(), datasets.refetch()] : [])]);
    } finally {
      if (mounted.current) setCheckingConnection(false);
    }
  }
  function updatePort(value: string) {
    setPort(value);
    setProbe(null);
  }
  const followerPort = (
    <label>
      Robot USB port
      <select
        aria-label="Robot USB port"
        value={port}
        disabled={active || busy || !online}
        onChange={(event) => updatePort(event.target.value)}
      >
        <option value="">Select a serial device</option>
        {port && !portConnected && (
          <option value={port} disabled>
            {port} · Disconnected
          </option>
        )}
        {devices.data?.ports.map((item) => (
          <option key={item.device} value={item.device}>
            {item.device} · {item.description}
          </option>
        ))}
      </select>
    </label>
  );
  const identity = (
    <div className="robot-lab-fields">
      {followerPort}
      <label>
        Robot name
        <input
          aria-label="Robot name"
          value={robotId}
          disabled={active || busy}
          onChange={(event) => {
            setRobotId(event.target.value);
            setMotionReady(false);
          }}
          maxLength={64}
          pattern="[A-Za-z0-9_-]+"
        />
      </label>
    </div>
  );

  const checkedCount = Object.values(motionChecks).filter((check) => check.correct).length;
  const mismatched = robotJoints.find((joint) => motionChecks[joint.name]?.correct === false);
  const nextUnchecked = setupOrder.find((joint) => !motionChecks[joint]?.correct);
  const scan = probe?.port === port && portConnected ? probe : null;
  const responding = scan?.motors.filter((motor) => motor.stable && motor.model === 777).length ?? 0;
  const profileCheck =
    selectedProfile &&
    selectedProfile.valid !== false &&
    scan?.profile?.robot_id === robotId &&
    scan.profile.role === role &&
    (!selectedProfile?.sha256 || scan.profile.sha256 === selectedProfile.sha256)
      ? scan.profile
      : undefined;
  const selectedCalibrated =
    profileUsable(selectedProfile) && !['missing', 'invalid', 'stale'].includes(profileCheck?.state ?? '');
  const calibrationLabel =
    session?.mode === 'calibration'
      ? 'Capturing…'
      : selectedProfile?.valid === false
        ? 'Invalid profile'
        : selectedProfile && !selectedCalibrated
          ? 'Recalibration needed'
          : profileCheck?.state === 'unavailable' && profileCheck.error
            ? 'Check failed'
            : profileCheck?.state === 'mismatch'
              ? 'Settings differ'
              : profileCheck?.state === 'matched'
                ? 'Matches arm'
                : selectedCalibrated
                  ? 'Saved · not checked'
                  : 'Not saved';
  const motorLabel = !portConnected
    ? 'USB unavailable'
    : active && session?.port === port
      ? `${session.active_joints?.length || 6} in session`
      : !scan
        ? 'Not scanned'
        : `${responding}/6 responding`;
  const mappingLabel = mismatched
    ? 'Mismatch'
    : !mappingComplete
      ? 'Not identified'
      : checkedCount === 6
        ? '6/6 verified'
        : `${checkedCount}/6 verified`;
  function openStep(index: number) {
    setStep(index);
    setNotice(null);
    setError(null);
  }
  function checkArm() {
    remember();
    void perform(
      'Arm scan completed.',
      () => robotApi.probe(port, robotId, role),
      (result) => setProbe(result),
      false,
    );
  }
  function verifyNext() {
    if (nextUnchecked) setMotorIndex(setupOrder.indexOf(nextUnchecked));
    openStep(1);
  }
  function confirmJoint(correct: boolean) {
    if (!verification?.run_id) return;
    const next = () => {
      if (correct) {
        const joint = setupOrder.find((j) => j !== selectedMotor.name && !motionChecks[j]?.correct);
        if (joint) setMotorIndex(setupOrder.indexOf(joint));
      }
    };
    if (state?.capabilities?.arm_readiness)
      void perform(
        'Motion observation saved.',
        () => robotApi.confirmMapping(port, selectedMotor.name, verification.run_id!, correct),
        next,
        false,
      );
    else {
      setMotionChecks((previous) => ({ ...previous, [selectedMotor.name]: { run: verification.run_id!, correct } }));
      next();
    }
  }
  const nextAction = !portConnected
    ? {
        text: online ? 'Connect the arm’s USB controller.' : 'Robot companion unavailable.',
        label: 'Open devices',
        target: 0,
        action: () => openStep(0),
      }
    : !scan
      ? { text: 'Check the connected arm without moving it.', label: 'Scan arm', action: checkArm }
      : !scan.ready
        ? {
            text: `${responding}/6 motors respond reliably. Review the missing IDs.`,
            label: 'Review motors',
            target: 0,
            action: () => openStep(0),
          }
        : mismatched
          ? {
              text: `ID ${mismatched.id} moved a different joint.`,
              label: 'Review mapping',
              target: 1,
              action: () => {
                setMotorIndex(setupOrder.indexOf(mismatched.name));
                openStep(1);
              },
            }
          : !mappingComplete
            ? {
                text: 'IDs respond; their physical joints still need identification.',
                label: 'Identify joints',
                target: 1,
                action: () => openStep(1),
              }
            : checkedCount < 6
              ? { text: 'Confirm which physical joint each ID moves.', label: 'Verify next joint', action: verifyNext }
              : !selectedCalibrated || profileCheck?.state === 'mismatch'
                ? {
                    text: selectedProfile
                      ? 'Calibration needs attention.'
                      : 'Record the arm’s middle and joint ranges.',
                    label: 'Open calibration',
                    target: 2,
                    action: () => openStep(2),
                  }
                : profileCheck?.state !== 'matched'
                  ? {
                      text: 'Calibration is saved. Check it against the arm.',
                      label: 'Check calibration',
                      action: checkArm,
                    }
                  : {
                      text: 'Motor responses, mapping and calibration checked.',
                      label: 'Open joint controls',
                      action: () => openStep(3),
                    };
  const stepStates = [
    { value: motorLabel, good: !!scan?.ready && !active, warn: !!scan && !scan.ready },
    { value: mappingLabel, good: mappingComplete && checkedCount === 6 && !mismatched, warn: !!mismatched },
    {
      value: calibrationLabel,
      good: profileCheck?.state === 'matched' && session?.mode !== 'calibration',
      warn:
        selectedProfile?.valid === false ||
        (!!selectedProfile && !selectedCalibrated) ||
        profileCheck?.state === 'mismatch',
    },
    {
      value: controlling ? 'Active' : selectedCalibrated ? 'Ready' : 'Needs calibration',
      good: controlling,
      warn: false,
    },
    {
      value: recording?.active ? 'Recording' : controlling && !partialControl ? 'Ready' : 'Needs teleoperation',
      good: !!recording?.active,
      warn: false,
    },
  ];
  const readiness = (
    <section className="robot-lab-readiness" aria-label="Arm readiness">
      <nav className="robot-lab-steps" aria-label="Robot setup steps">
        {steps.map((item, index) => {
          const itemState = stepStates[index];
          return (
            <button
              key={item.title}
              className={`robot-lab-step${step === index ? ' selected' : ''}${itemState.good ? ' verified' : itemState.warn ? ' attention' : ''}`}
              onClick={() => openStep(index)}
              aria-current={step === index ? 'step' : undefined}
              disabled={busy}
            >
              <span className="robot-lab-step-title">{item.title}</span>
              <span className="robot-lab-step-state">
                {itemState.good && <Icon name="check" size={12} />}
                {itemState.value}
              </span>
            </button>
          );
        })}
      </nav>
      {!active && (
        <div className="robot-lab-next">
          <span>{nextAction.text}</span>
          {nextAction.target !== step && (
            <button
              className="text-button"
              disabled={
                busy ||
                (nextAction.action === checkArm &&
                  (!canUseArm || !state?.capabilities?.probe || !/^[A-Za-z0-9_-]+$/.test(robotId)))
              }
              onClick={nextAction.action}
            >
              {pending === 'Arm scan completed.' ? 'Scanning…' : nextAction.label}
              <Icon name="arrow" size={13} />
            </button>
          )}
        </div>
      )}
    </section>
  );
  const capturedRanges = robotJoints.filter(
    (joint) =>
      joint.name !== 'wrist_roll' &&
      (calibration?.maxes[joint.name] ?? 0) - (calibration?.mins[joint.name] ?? 0) >= 100,
  ).length;
  const calibrationSummary = (
    <div className="robot-lab-profile" aria-label="Selected arm calibration">
      <div>
        <strong>{calibrationLabel}</strong>
        <span>
          {robotId} · {role}
        </span>
      </div>
      {selectedCalibrated && (
        <button className="text-button" disabled={active || busy} onClick={() => openStep(3)}>
          Open teleoperation <Icon name="arrow" size={13} />
        </button>
      )}
    </div>
  );

  return (
    <div className="robot-lab">
      <div className="robot-lab-statusbar">
        <div className="robot-lab-connection">
          <span className={`robot-lab-dot${online ? ' online' : ''}`} />
          <div>
            <strong>
              {status.isPending
                ? 'Connecting…'
                : online
                  ? `${state?.label ?? 'Robot companion'} connected`
                  : 'Robot companion unavailable'}
            </strong>
            <small>
              {online
                ? active
                  ? `${session?.mode === 'verification' ? 'Mapping check' : session?.mode === 'setup' ? 'Assembled motor setup' : session?.mode === 'calibration' ? 'Calibration' : session?.mode === 'handguide' ? 'Hand guiding' : session?.mode === 'leader' ? 'Leader teleoperation' : 'Manual control'} · ${session?.robot_id ?? robotId} · ${session?.mode === 'setup' ? 'Motor setup active' : session?.torque_enabled ? 'Torque command: on' : 'Torque command: off'}`
                  : usbStatus
                : (state?.message ?? status.error?.message ?? 'Check that the local companion is running.')}
            </small>
          </div>
        </div>
        <div className="robot-lab-status-actions">
          <button
            className="text-button"
            onClick={() => void refresh()}
            disabled={busy || checkingConnection || status.isPending}
          >
            {checkingConnection ? 'Refreshing…' : 'Refresh devices'}
          </button>
          {(active || session?.torque_enabled || session?.error || pending === 'Stop requested.') && (
            <button
              className="danger-button robot-lab-stop"
              onClick={stop}
              disabled={!state?.ok && !active}
              aria-label="Stop arm and disable torque"
            >
              <span />
              {pending === 'Stop requested.' ? 'Stopping…' : 'Stop arm'}
            </button>
          )}
        </div>
      </div>
      {readiness}
      {step !== 2 && (session?.error || error) && <HardwareNotice message={(session?.error || error)!} />}
      {(reconnecting || (!online && localCalibrationRef.current)) && (
        <p className="robot-lab-event robot-lab-warning" role="status">
          {localCalibrationRef.current
            ? 'Reconnecting… Calibration continues on the robot computer. Captured values are retained.'
            : 'Reconnecting… Motion controls are paused; the motor watchdog remains active.'}
        </p>
      )}
      {notice && (
        <p className="robot-lab-event" role="status">
          {notice}
        </p>
      )}
      {recording?.error && <HardwareNotice message={recording.error} />}
      <div className="robot-lab-grid">
        <section className="robot-lab-main" aria-label={steps[step].title}>
          {step === 0 && (
            <>
              <Heading title="Devices" />
              <div className="robot-lab-form">
                {!online && (
                  <Notice>
                    <strong>Companion offline</strong>Start the Robot companion, then check the connection.
                  </Notice>
                )}
                {devices.isPending && online && (
                  <p className="robot-lab-subtle" role="status">
                    Looking for serial devices and cameras…
                  </p>
                )}
                {devices.error && (
                  <Notice kind="warning">
                    <strong>Device discovery needs attention</strong>
                    {devices.error.message}
                    <button className="text-button" onClick={() => void devices.refetch()}>
                      Retry discovery
                    </button>
                  </Notice>
                )}
                {identity}

                {online && unavailableFeatures && missingHardwareTools.length > 0 && (
                  <Notice kind="warning">
                    <strong>Missing dependencies</strong>
                    {missingHardwareTools.map(([name]) => name).join(', ')}
                  </Notice>
                )}
                {online && devices.data?.camera_error && <Notice kind="warning">{devices.data.camera_error}</Notice>}
                <div className="robot-lab-actions">
                  <button
                    className="primary-button"
                    title="Read IDs and saved calibration settings; no motion"
                    disabled={!canUseArm || active || !state?.capabilities?.probe || !/^[A-Za-z0-9_-]+$/.test(robotId)}
                    onClick={checkArm}
                  >
                    {pending === 'Arm scan completed.' ? 'Scanning…' : 'Scan motor IDs'}
                  </button>
                  {scan?.ready && (
                    <button className="secondary-button" disabled={!canAct} onClick={verifyNext}>
                      Verify physical mapping
                      <Icon name="arrow" size={15} />
                    </button>
                  )}
                </div>
                {scan && (
                  <section className="robot-lab-form" aria-label="Motor scan results">
                    <div className="robot-lab-mapping-heading">
                      <h3>{responding}/6 motors responding</h3>
                      <span>
                        No motion ·{' '}
                        {scan.checked_at
                          ? new Date(scan.checked_at * 1000).toLocaleTimeString([], {
                              hour: '2-digit',
                              minute: '2-digit',
                            })
                          : 'Just scanned'}
                      </span>
                    </div>
                    <div className="robot-lab-table-wrap">
                      <table className="robot-lab-table">
                        <thead>
                          <tr>
                            <th>Expected joint</th>
                            <th>ID</th>
                            <th>Response</th>
                          </tr>
                        </thead>
                        <tbody>
                          {robotJoints.map((joint) => {
                            const motor = scan.motors.find((item) => item.id === joint.id);
                            return (
                              <tr key={joint.name}>
                                <td>{joint.label}</td>
                                <td>{joint.id}</td>
                                <td>
                                  <span
                                    className={
                                      motor?.stable && motor.model === 777 ? 'robot-lab-pass' : 'robot-lab-fail'
                                    }
                                  >
                                    {!motor
                                      ? 'No response'
                                      : motor.model !== 777
                                        ? `Unexpected model ${motor.model}`
                                        : motor.stable
                                          ? 'Responding'
                                          : 'Intermittent'}
                                  </span>
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                    {!scan.ready && (
                      <div className="robot-lab-actions">
                        <span className="robot-lab-subtle">Check motor power and IDs.</span>
                        <button className="text-button" onClick={() => openStep(1)}>
                          Open motor setup
                        </button>
                      </div>
                    )}
                    <details className="robot-lab-details">
                      <summary>What this scan checks</summary>
                      <p>Replies at IDs 1–6 and STS3215 model. It cannot tell which physical joint owns an ID.</p>
                      {scan.warning && <p>{scan.warning}</p>}
                    </details>
                  </section>
                )}
              </div>
            </>
          )}
          {step === 1 && (
            <>
              <Heading title="Motor IDs" />
              <div className="robot-lab-form">
                {mappingComplete && !reidentify && (
                  <section className="robot-lab-form" aria-label="Saved motor mapping">
                    <div className="robot-lab-mapping-heading">
                      <h3>Saved IDs</h3>
                      <span>{checkedCount}/6 motion checks</span>
                    </div>
                    {jointGuide}
                    <div className="robot-lab-form" role="group" aria-label="Verify saved motor mapping">
                      <label className="robot-lab-check">
                        <input
                          type="checkbox"
                          checked={verificationReady}
                          disabled={busy || active || !portConnected}
                          onChange={(event) => setVerificationReady(event.target.checked)}
                        />
                        <span>Arm supported; joints clear of stops. Brief motion, then torque off.</span>
                      </label>
                      <button
                        className="primary-button"
                        disabled={
                          !canUseArm || active || !verificationReady || !state?.capabilities?.mapping_verification
                        }
                        onClick={() => {
                          if (!state?.capabilities?.arm_readiness)
                            setMotionChecks((previous) => {
                              const next = { ...previous };
                              delete next[selectedMotor.name];
                              return next;
                            });
                          void perform(
                            'Mapping check requested.',
                            () => robotApi.wiggleSaved(port, selectedMotor.name),
                            ownSession,
                            false,
                          );
                        }}
                      >
                        {session?.mode === 'verification'
                          ? verification?.phase === 'wiggling'
                            ? `Wiggling ID ${verification.motor_id}…`
                            : 'Checking motor addresses…'
                          : `Wiggle ${selectedMotor.label.toLowerCase()} · ID ${assembled.mapping[selectedMotor.name]}`}
                      </button>
                      {verification?.port === port &&
                        verification.phase === 'done' &&
                        verification.joint === selectedMotor.name &&
                        verification.run_id &&
                        motionChecks[selectedMotor.name]?.run !== verification.run_id && (
                          <div className="so101-detection-feedback" role="group" aria-label="Confirm physical joint">
                            <strong>ID {verification.motor_id} finished · torque off</strong>
                            <p>Did only the {selectedMotor.label.toLowerCase()} move?</p>
                            <div className="robot-lab-actions">
                              <button
                                className="secondary-button"
                                disabled={active || busy}
                                onClick={() => confirmJoint(true)}
                              >
                                Correct joint
                              </button>
                              <button
                                className="secondary-button"
                                disabled={active || busy}
                                onClick={() => confirmJoint(false)}
                              >
                                Wrong joint
                              </button>
                            </div>
                          </div>
                        )}
                      {verification?.port === port && verification.error && (
                        <HardwareNotice message={verification.error} />
                      )}
                    </div>
                    <div className="robot-lab-table-wrap">
                      <table className="robot-lab-table">
                        <thead>
                          <tr>
                            <th>Joint</th>
                            <th>Motor ID</th>
                            <th>Motion check</th>
                          </tr>
                        </thead>
                        <tbody>
                          {robotJoints.map((joint) => (
                            <tr key={joint.name}>
                              <td>
                                <button
                                  className="text-button"
                                  aria-pressed={selectedMotor.name === joint.name}
                                  onClick={() => selectIllustratedJoint(joint.name)}
                                >
                                  {joint.label}
                                </button>
                              </td>
                              <td>{assembled.mapping[joint.name]}</td>
                              <td>
                                {motionChecks[joint.name]?.correct
                                  ? 'Correct'
                                  : motionChecks[joint.name]
                                    ? 'Mismatch'
                                    : 'Unchecked'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    {Object.values(motionChecks).some((check) => !check.correct) && (
                      <Notice kind="warning">Mapping mismatch. Reidentify the joints before calibration.</Notice>
                    )}
                    {state?.mapping_checks?.error && <HardwareNotice message={state.mapping_checks.error} />}
                    {checkedCount === 6 && !mismatched && (
                      <div className="robot-lab-actions">
                        <span className="robot-lab-pass">Physical mapping verified.</span>
                        <button
                          className="secondary-button"
                          disabled={active || busy}
                          onClick={() => openStep(selectedCalibrated ? 3 : 2)}
                        >
                          {selectedCalibrated ? 'Open joint controls' : 'Calibrate this arm'}
                          <Icon name="arrow" size={14} />
                        </button>
                      </div>
                    )}
                    <button
                      className="text-button"
                      disabled={busy || active}
                      onClick={() => {
                        setReidentify(true);
                        setVerificationReady(false);
                        setMotionChecks({});
                      }}
                    >
                      Reidentify joints
                    </button>
                  </section>
                )}
                {assembled?.phase !== 'complete' && (
                  <div className="so101-detection-feedback" role="status" aria-label="Motor identification status">
                    <strong>
                      {!online
                        ? 'Movement detection unavailable'
                        : session?.mode === 'setup' && ['scanning', 'splitting'].includes(assembled?.phase ?? '')
                          ? 'Preparing motor addresses'
                          : assembled?.phase === 'applying'
                            ? 'Saving motor IDs'
                            : assembled?.phase === 'observing' && identifyMethod === 'hand'
                              ? 'Listening for hand movement'
                              : session?.mode === 'setup' &&
                                  ['ready', 'observing', 'review'].includes(assembled?.phase ?? '')
                                ? 'Ready to identify joints'
                                : assembled?.phase === 'wiggling'
                                  ? 'Identifying motor by motion'
                                  : 'Movement detection off'}
                    </strong>
                    <p>
                      {!online
                        ? 'Connect the Robot companion first.'
                        : session?.mode === 'setup' && ['scanning', 'splitting'].includes(assembled?.phase ?? '')
                          ? `${assembled?.progress ? `Round ${assembled.progress}/12 · ` : ''}${assembled?.ids.length ?? 0}/6 addresses found. No movement is commanded. Keep this page visible until preparation finishes.`
                          : session?.mode === 'setup' &&
                              ['ready', 'observing', 'review', 'wiggling', 'applying'].includes(assembled?.phase ?? '')
                            ? assembled?.phase === 'applying'
                              ? 'Keep this page visible while the saved addresses are verified.'
                              : identifyMethod === 'hand'
                                ? assembled?.phase === 'observing'
                                  ? `Move only the ${selectedMotor.label.toLowerCase()}, then hold its new position.`
                                  : 'Click a joint on the model to start listening for its movement.'
                                : 'Select an address, request a wiggle, then select the joint that moved.'
                            : `${assembled?.ids.length ? `${assembled.ids.length}/6 addresses found. ` : ''}Start assembled setup below. Hand detection and wiggle require six separate motor addresses.`}
                    </p>
                  </div>
                )}
                {(!mappingComplete || reidentify) && jointGuide}
                {followerPort}
                {(!mappingComplete || reidentify) && session?.mode !== 'setup' && (
                  <fieldset className="robot-lab-identify-method">
                    <legend>Identification method</legend>
                    <button
                      type="button"
                      className={identifyMethod === 'hand' ? 'primary-button' : 'secondary-button'}
                      aria-pressed={identifyMethod === 'hand'}
                      onClick={() => {
                        setIdentifyMethod('hand');
                        setWiggleReady(false);
                      }}
                    >
                      Move by hand
                    </button>
                    <button
                      type="button"
                      className={identifyMethod === 'wiggle' ? 'primary-button' : 'secondary-button'}
                      aria-pressed={identifyMethod === 'wiggle'}
                      onClick={() => setIdentifyMethod('wiggle')}
                    >
                      Wiggle a motor
                    </button>
                  </fieldset>
                )}
                {(!mappingComplete || reidentify) && (
                  <section className="robot-lab-assembled" aria-label="Experimental assembled setup">
                    <div className="robot-lab-assembled-heading">
                      <h3>Assembled setup</h3>
                      <span className="robot-lab-experimental">Experimental</span>
                    </div>
                    {(!assembled || ['idle', 'stopped', 'error', 'complete'].includes(assembled.phase)) &&
                      session?.mode !== 'setup' && (
                        <div className="robot-lab-form">
                          <Notice kind="warning">
                            Support the arm. Keep power, USB and motor cables connected. Experimental setup reboots
                            motors and writes temporary IDs; torque can release.
                          </Notice>
                          {scan?.ready && (
                            <p className="robot-lab-subtle">
                              IDs 1–6 respond. Identification confirms which joint owns each ID.
                            </p>
                          )}
                          <label className="robot-lab-check">
                            <input
                              type="checkbox"
                              checked={assembledReady}
                              disabled={busy || active}
                              onChange={(event) => setAssembledReady(event.target.checked)}
                            />
                            <span>
                              Six STS3215 motors; arm supported. Allow motor reboots and persistent ID changes.
                            </span>
                          </label>
                          <div className="robot-lab-actions">
                            <button
                              className="primary-button"
                              disabled={
                                !canUseArm || active || !assembledReady || !state?.capabilities?.assembled_setup
                              }
                              onClick={() => {
                                setAssembledReady(false);
                                setReidentify(false);
                                void perform(
                                  'Experimental assembled setup started.',
                                  () => robotApi.startAssembled(port),
                                  ownSession,
                                  false,
                                );
                              }}
                            >
                              Start assembled setup
                            </button>
                            <button
                              className="secondary-button"
                              disabled={!canUseArm || active}
                              onClick={() =>
                                void perform(
                                  'Motor connection checked.',
                                  () => robotApi.probe(port),
                                  (result) => setProbe(result),
                                )
                              }
                            >
                              Verify current IDs
                            </button>
                          </div>
                        </div>
                      )}
                    {assembled && assembled.phase !== 'idle' && (
                      <div className="robot-lab-form">
                        <div className={`robot-lab-assembled-state phase-${assembled.phase}`} role="status">
                          <span className="robot-lab-mini-label">{assembled.phase.toUpperCase()}</span>
                          <p>{assembled.message}</p>
                          {assembled.ids.length > 0 && (
                            <small>
                              {assembled.ids.length} distinct {assembled.ids.length === 1 ? 'address' : 'addresses'}:{' '}
                              {assembled.ids.join(', ')}
                            </small>
                          )}
                        </div>
                        {assembled.error && <HardwareNotice message={assembled.error} />}
                        {session?.mode === 'setup' && ['ready', 'observing', 'review'].includes(assembled.phase) && (
                          <>
                            <fieldset className="robot-lab-identify-method" disabled={busy || assembled.motion_active}>
                              <legend>Identification method</legend>
                              <button
                                type="button"
                                className={identifyMethod === 'hand' ? 'primary-button' : 'secondary-button'}
                                aria-pressed={identifyMethod === 'hand'}
                                onClick={() => {
                                  setIdentifyMethod('hand');
                                  setWiggleReady(false);
                                }}
                              >
                                Move by hand
                              </button>
                              <button
                                type="button"
                                className={identifyMethod === 'wiggle' ? 'primary-button' : 'secondary-button'}
                                aria-pressed={identifyMethod === 'wiggle'}
                                onClick={() => setIdentifyMethod('wiggle')}
                              >
                                Wiggle a motor
                              </button>
                            </fieldset>
                            {identifyMethod === 'hand' ? (
                              <div className="robot-lab-actions">
                                <button
                                  className="primary-button"
                                  disabled={
                                    !canAct || !ownsSession || assembled.mapping[selectedMotor.name] !== undefined
                                  }
                                  onClick={() => selectIllustratedJoint(selectedMotor.name)}
                                >
                                  {assembled.phase === 'observing'
                                    ? 'Restart movement detection'
                                    : 'Start movement detection'}
                                </button>
                              </div>
                            ) : (
                              <div className="robot-lab-form">
                                <label>
                                  Motor address
                                  <select
                                    aria-label="Motor address to identify"
                                    value={
                                      wiggleId || String(assembled.ids.find((id) => !mappingValues.includes(id)) ?? '')
                                    }
                                    onChange={(event) => {
                                      setWiggleId(event.target.value);
                                      setWiggleReady(false);
                                    }}
                                  >
                                    {assembled.ids
                                      .filter((id) => !mappingValues.includes(id))
                                      .map((id) => (
                                        <option value={id} key={id}>
                                          Address {id}
                                        </option>
                                      ))}
                                  </select>
                                </label>
                                <label className="robot-lab-check">
                                  <input
                                    type="checkbox"
                                    checked={wiggleReady}
                                    onChange={(event) => setWiggleReady(event.target.checked)}
                                  />
                                  <span>
                                    Arm supported; every joint clear of its stops. Allow a brief ±2° motion at low
                                    torque.
                                  </span>
                                </label>
                                <button
                                  className="primary-button"
                                  disabled={!canAct || !ownsSession || !wiggleReady || allJointsMapped}
                                  onClick={() => {
                                    const id = Number(
                                      wiggleId || assembled.ids.find((id) => !mappingValues.includes(id)),
                                    );
                                    setWiggleReady(false);
                                    void perform(
                                      `Address ${id} identification requested.`,
                                      () => robotApi.wiggleAssembled(id),
                                      undefined,
                                      false,
                                    );
                                  }}
                                >
                                  Wiggle selected motor
                                </button>
                                {assembled.wiggle_id != null && (
                                  <div className="robot-lab-form">
                                    <p>Address {assembled.wiggle_id}: select the joint that moved on your arm.</p>
                                    <button
                                      className="secondary-button"
                                      disabled={
                                        !canAct || !ownsSession || assembled.mapping[selectedMotor.name] !== undefined
                                      }
                                      onClick={() =>
                                        void perform(
                                          'Joint address confirmed.',
                                          () => robotApi.acceptWiggle(selectedMotor.name, assembled.wiggle_id!),
                                          () => {
                                            setWiggleId('');
                                            const next = setupOrder.find(
                                              (joint) =>
                                                joint !== selectedMotor.name && assembled.mapping[joint] === undefined,
                                            );
                                            if (next) setMotorIndex(setupOrder.indexOf(next));
                                          },
                                        )
                                      }
                                    >
                                      Assign address {assembled.wiggle_id} to {selectedMotor.label.toLowerCase()}
                                    </button>
                                  </div>
                                )}
                              </div>
                            )}

                            <div className="robot-lab-table-wrap">
                              <table className="robot-lab-table">
                                <thead>
                                  <tr>
                                    <th>Joint to identify</th>
                                    <th>Detected address</th>
                                    <th>Final ID</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {robotJoints.map((joint) => (
                                    <tr key={joint.name}>
                                      <td>
                                        <button
                                          type="button"
                                          className="text-button"
                                          onClick={() => selectIllustratedJoint(joint.name)}
                                        >
                                          {joint.label}
                                        </button>
                                      </td>
                                      <td>{assembled.mapping[joint.name] ?? 'Not confirmed'}</td>
                                      <td>{joint.id}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                            {allJointsMapped && (
                              <>
                                <Notice kind="warning">
                                  Review all six matches. Saving writes IDs 1–6 and requires recalibration.
                                </Notice>
                                <button
                                  className="primary-button"
                                  disabled={!canAct || !ownsSession || assembled.phase !== 'review'}
                                  onClick={() =>
                                    void perform('Motor ID save requested.', robotApi.finishAssembled, (result) => {
                                      if (result.session?.mode === 'idle') {
                                        ownedRef.current = false;
                                        setOwnsSession(false);
                                      }
                                      setProbe(null);
                                    })
                                  }
                                >
                                  Save motor IDs 1–6
                                </button>
                              </>
                            )}
                          </>
                        )}
                        {session?.mode === 'setup' && (
                          <button
                            className="secondary-button"
                            disabled={busy}
                            onClick={() =>
                              void perform('Assembled setup stopped.', robotApi.cancelAssembled, () => {
                                ownedRef.current = false;
                                setOwnsSession(false);
                              })
                            }
                          >
                            Stop assembled setup
                          </button>
                        )}
                        {assembled.phase === 'complete' && (
                          <Notice kind="success">
                            <strong>IDs saved</strong>Calibration required.
                            <button className="text-button" disabled={busy} onClick={() => setStep(2)}>
                              Continue to calibration
                            </button>
                          </Notice>
                        )}
                        {assembled.log.length > 0 && (
                          <details className="robot-lab-setup-log">
                            <summary>Setup details</summary>
                            <ol>
                              {assembled.log.map((entry, index) => (
                                <li key={index}>{entry}</li>
                              ))}
                            </ol>
                          </details>
                        )}
                      </div>
                    )}
                  </section>
                )}
                <details className="robot-lab-advanced-ids">
                  <summary>Advanced: configure one physically isolated motor</summary>
                  <div className="robot-lab-form">
                    <div className="robot-lab-motor-list">
                      {setupOrder.map((name, index) => {
                        const joint = robotJoints.find((item) => item.name === name)!;
                        return (
                          <button
                            key={name}
                            className={`robot-lab-motor${motorIndex === index ? ' selected' : ''}${assigned.includes(name) ? ' assigned' : ''}`}
                            onClick={() => setMotorIndex(index)}
                            disabled={busy || active}
                            aria-pressed={motorIndex === index}
                          >
                            <b>{assigned.includes(name) ? <Icon name="check" size={12} /> : joint.id}</b>
                            <span>
                              <strong>{joint.label}</strong>
                              <small>
                                {assigned.includes(name) ? 'Assigned this session' : `Motor ID ${joint.id}`}
                              </small>
                            </span>
                          </button>
                        );
                      })}
                    </div>
                    <div className="robot-lab-motor-guide">
                      <span className="robot-lab-motor-number">{selectedMotor.id}</span>
                      <div>
                        <h3>{selectedMotor.label}</h3>
                        <p>
                          Power off the arm. Connect only this motor directly to the controller, with the rest of the
                          chain disconnected. Restore motor power and keep USB connected.
                        </p>
                      </div>
                    </div>
                    <label className="robot-lab-check">
                      <input
                        type="checkbox"
                        checked={isolated}
                        disabled={busy || active}
                        onChange={(event) => setIsolated(event.target.checked)}
                      />
                      <span>
                        <strong>Only the {selectedMotor.label.toLowerCase()} motor is connected.</strong>I have
                        physically disconnected every other motor from the controller.
                      </span>
                    </label>
                    <div className="robot-lab-actions">
                      <button
                        className="primary-button"
                        disabled={!canUseArm || active || !isolated || !state?.capabilities?.motor_setup}
                        onClick={() =>
                          void perform(
                            `${selectedMotor.label} assigned ID ${selectedMotor.id}.`,
                            () => robotApi.assignMotor(port, selectedMotor.name),
                            (result) => {
                              setAssigned((previous) => [...new Set([...previous, result.assigned.joint])]);
                              setIsolated(false);
                              if (motorIndex < 5) setMotorIndex(motorIndex + 1);
                            },
                          )
                        }
                      >
                        {busy ? 'Assigning motor ID…' : `Assign ID ${selectedMotor.id}`}
                      </button>
                      <button className="secondary-button" disabled={busy || active} onClick={() => setStep(2)}>
                        IDs already set? Calibrate
                        <Icon name="arrow" size={14} />
                      </button>
                    </div>
                    <Notice>
                      <strong>Reconnect the full arm when all six IDs are set.</strong>Switch motor power off before
                      changing cables. A scan cannot distinguish several motors sharing the same factory ID; this step
                      needs one physical motor at a time.
                    </Notice>
                  </div>
                </details>
              </div>
            </>
          )}
          {step === 2 && (
            <>
              <Heading title="Calibration" />
              <div className="robot-lab-form">
                {calibrationSummary}
                {localCalibrationRef.current && (
                  <p className="robot-lab-subtle">
                    Calibration runs locally with torque off. You can leave this page and return to continue.
                  </p>
                )}
                {jointGuide}
                {identity}
                <label>
                  Arm role
                  <select
                    value={role}
                    aria-label="Calibration arm role"
                    onChange={(event) => setRole(event.target.value as RobotRole)}
                    disabled={active || busy}
                  >
                    <option value="follower">Follower / single arm</option>
                    <option value="leader">Leader arm</option>
                  </select>
                </label>
                <div className="robot-lab-calibration-steps">
                  <span className={calibration?.phase === 'idle' || calibration?.phase === 'saved' ? 'current' : ''}>
                    1. Prepare
                  </span>
                  <Icon name="arrow" size={12} />
                  <span className={calibrationStage === 'center' ? 'current' : ''}>2. Middle position</span>
                  <Icon name="arrow" size={12} />
                  <span className={calibrationStage === 'ranges' ? 'current' : ''}>3. Joint ranges</span>
                </div>
                {calibrationError && <HardwareNotice label="Calibration error" message={calibrationError} />}
                {calibration?.recoverable ? (
                  <Notice kind="warning">
                    <strong>Calibration paused</strong>
                    <p>
                      {calibration.interrupted_stage === 'center' ||
                      (calibration.phase === 'idle' && !Object.keys(calibration.offsets).length)
                        ? 'Preparation is saved on the robot computer.'
                        : 'Captured middle and ranges are saved on the robot computer.'}
                    </p>
                    <button
                      className="primary-button"
                      disabled={!canUseArm || active}
                      onClick={() =>
                        void perform(
                          'Captured calibration resumed. Torque remains off.',
                          robotApi.resumeCalibration,
                          ownSession,
                        )
                      }
                    >
                      Resume captured calibration
                    </button>
                  </Notice>
                ) : (
                  calibration?.phase === 'interrupted' && (
                    <Notice kind="warning">
                      <strong>Calibration interrupted</strong>
                      <p>
                        {calibration.interrupted_stage === 'center'
                          ? 'Middle position was not saved.'
                          : 'Joint ranges were not saved.'}{' '}
                        Restart calibration to continue.
                      </p>
                    </Notice>
                  )
                )}
                {session?.mode !== 'calibration' && (
                  <>
                    <Notice>
                      Support the arm. Calibration releases torque; all six motors must be connected and assigned IDs.
                    </Notice>
                    <button
                      className="primary-button"
                      disabled={
                        !canUseArm || active || !/^[A-Za-z0-9_-]+$/.test(robotId) || !state?.capabilities?.calibration
                      }
                      onClick={() =>
                        void perform(
                          'Calibration started. Position each joint near its middle.',
                          () => robotApi.startCalibration(port, robotId, role),
                          ownSession,
                        )
                      }
                    >
                      {calibration?.recoverable
                        ? 'Start a new calibration'
                        : calibration?.phase === 'interrupted'
                          ? 'Restart calibration'
                          : selectedCalibrated
                            ? 'Recalibrate arm'
                            : 'Begin calibration'}
                    </button>
                  </>
                )}
                {session?.mode === 'calibration' && calibration?.phase === 'center' && (
                  <>
                    <Notice>
                      Position each joint at mid-travel, with the gripper half open. Support the arm and record the
                      center.
                    </Notice>
                    <RangeTable positions={session.positions} calibration />
                    <div className="robot-lab-actions">
                      <button
                        className="primary-button"
                        disabled={!canAct || !ownsSession}
                        onClick={() =>
                          void perform(
                            'Center recorded. Move each joint through its comfortable range.',
                            robotApi.centerCalibration,
                            undefined,
                            true,
                            'ranges',
                          )
                        }
                      >
                        {pending === 'Center recorded. Move each joint through its comfortable range.'
                          ? 'Recording middle position…'
                          : 'Record middle position'}
                      </button>
                      <button
                        className="secondary-button"
                        disabled={!canAct}
                        onClick={() =>
                          void perform('Calibration cancelled.', robotApi.cancelCalibration, () => {
                            ownedRef.current = false;
                            setOwnsSession(false);
                          })
                        }
                      >
                        Cancel calibration
                      </button>
                    </div>
                  </>
                )}
                {session?.mode === 'calibration' && calibration?.phase === 'ranges' && (
                  <>
                    <Notice>
                      Move each joint through its usable range without forcing it. Wrist roll uses its full encoder
                      range.
                    </Notice>
                    <div className="robot-lab-mapping-heading">
                      <h3>Joint ranges</h3>
                      <span>{capturedRanges}/5 ranges detected · wrist roll automatic</span>
                    </div>
                    <RangeTable
                      positions={session.positions}
                      mins={calibration.mins}
                      maxes={calibration.maxes}
                      calibration
                    />
                    <div className="robot-lab-actions">
                      <button
                        className="primary-button"
                        disabled={!canAct || !ownsSession || capturedRanges < 5}
                        onClick={() =>
                          void perform(
                            'Calibration saved.',
                            robotApi.finishCalibration,
                            () => {
                              ownedRef.current = false;
                              setOwnsSession(false);
                            },
                            true,
                            'saved',
                          )
                        }
                      >
                        Save calibration
                      </button>
                      <button
                        className="secondary-button"
                        disabled={!canAct}
                        onClick={() =>
                          void perform('Calibration cancelled.', robotApi.cancelCalibration, () => {
                            ownedRef.current = false;
                            setOwnsSession(false);
                          })
                        }
                      >
                        Cancel calibration
                      </button>
                    </div>
                  </>
                )}
                {selectedProfile && (
                  <details className="robot-lab-details">
                    <summary>Saved calibration details</summary>
                    <p>
                      {selectedProfile.valid === false
                        ? selectedProfile.error
                        : profileCheck?.state === 'matched'
                          ? 'Saved offsets and limits match all six connected motors.'
                          : profileCheck?.state === 'mismatch'
                            ? 'Motor offsets or limits differ from the saved profile. Recalibrate before control.'
                            : !selectedCalibrated
                              ? 'Motor setup changed after this calibration was saved.'
                              : 'Middle and ranges are saved. Scan the arm to compare its current settings.'}
                    </p>
                    {profileCheck?.error && <p>{profileCheck.error}</p>}
                    <code>{selectedProfile.path}</code>
                  </details>
                )}
              </div>
            </>
          )}
          {step === 3 && (
            <>
              <Heading title="Teleoperation" />
              <div className="robot-lab-form">
                {mode !== 'handguide' &&
                  (state?.capabilities?.lerobot_position_control ? (
                    <Notice>
                      <strong>LeRobot SO101 control</strong>Direct position targets; motors may move quickly. Keep the
                      full path clear.
                    </Notice>
                  ) : (
                    online && (
                      <Notice kind="warning">
                        <strong>Earlier control profile</strong>Update the robot companion to use LeRobot SO101 control.
                      </Notice>
                    )
                  ))}
                {identity}
                <div className="robot-lab-mode-grid" role="group" aria-label="Control mode">
                  {(
                    [
                      {
                        value: 'manual',
                        title: 'Joint control',
                        detail: 'Sliders and position steps',
                        icon: 'sliders',
                      },
                      { value: 'handguide', title: 'Hand guiding', detail: 'Torque off', icon: 'robot' },
                      { value: 'leader', title: 'Leader → follower', detail: 'Two calibrated arms', icon: 'layers' },
                    ] as const
                  ).map((item) => (
                    <button
                      key={item.value}
                      className={`robot-lab-mode${mode === item.value ? ' selected' : ''}`}
                      onClick={() => {
                        setMode(item.value);
                        setMotionReady(false);
                      }}
                      disabled={active || busy}
                      aria-pressed={mode === item.value}
                    >
                      <Icon name={item.icon} size={19} />
                      <strong>{item.title}</strong>
                      <small>{item.detail}</small>
                    </button>
                  ))}
                </div>
                {mode === 'leader' && (
                  <div className="robot-lab-fields">
                    <label>
                      Leader USB port
                      <select
                        value={leaderPort}
                        disabled={active || busy}
                        aria-label="Leader USB port"
                        onChange={(event) => setLeaderPort(event.target.value)}
                      >
                        <option value="">Select the second arm</option>
                        {devices.data?.ports
                          .filter((item) => item.device !== port)
                          .map((item) => (
                            <option key={item.device} value={item.device}>
                              {item.device}
                            </option>
                          ))}
                      </select>
                    </label>
                    <label>
                      Leader name
                      <input
                        value={leaderId}
                        disabled={active || busy}
                        aria-label="Leader name"
                        onChange={(event) => setLeaderId(event.target.value)}
                      />
                      <small>
                        {leaderCalibrated
                          ? 'Leader calibration found.'
                          : 'Calibrate this arm with the leader role first.'}
                      </small>
                    </label>
                  </div>
                )}
                {controlJointOptions}
                {mode === 'manual' && !partialSelection && state?.capabilities?.middle_tolerance && (
                  <label>
                    Middle tolerance
                    <select
                      aria-label="Middle tolerance"
                      value={active ? (session?.centering?.tolerance ?? middleTolerance) : middleTolerance}
                      onChange={(event) => {
                        setMiddleTolerance(Number(event.target.value));
                        setMotionReady(false);
                      }}
                      disabled={active || busy}
                    >
                      <option value={2}>2° / 2%</option>
                      <option value={5}>5° / 5%</option>
                    </select>
                  </label>
                )}
                {!active && (
                  <>
                    {!calibrated && (
                      <Notice kind="warning">
                        <strong>{followerMismatch ? 'Calibration settings differ' : 'Calibration required'}</strong>
                        Follower: {robotId}.
                        <button className="text-button" onClick={() => setStep(2)}>
                          Open calibration
                        </button>
                      </Notice>
                    )}
                    <label className="robot-lab-check">
                      <input
                        type="checkbox"
                        checked={motionReady}
                        disabled={busy}
                        onChange={(event) => setMotionReady(event.target.checked)}
                      />
                      <span>
                        <strong>{mode === 'handguide' ? 'Arm supported.' : 'Arm secured; workspace clear.'}</strong>
                        {mode === 'handguide'
                          ? 'Torque stays off.'
                          : mode === 'leader'
                            ? 'Enable torque; the follower starts tracking the leader immediately.'
                            : partialSelection
                              ? 'Enable selected motors and hold their current pose.'
                              : 'Move to the recorded middle. Keep the full path clear.'}
                      </span>
                    </label>
                    <button
                      className="primary-button"
                      disabled={
                        !canUseArm ||
                        !calibrated ||
                        !motionReady ||
                        !state?.capabilities?.control ||
                        (mode === 'manual' &&
                          (!controlJoints.length ||
                            !state?.capabilities?.move_to_middle ||
                            (partialSelection && !state?.capabilities?.partial_control))) ||
                        (mode === 'leader' && (!leaderConnected || !leaderCalibrated || leaderPort === port))
                      }
                      onClick={() =>
                        void perform(
                          mode === 'handguide'
                            ? 'Hand guiding started. Torque remains off.'
                            : 'Control session started. Stop releases the motors.',
                          () =>
                            robotApi.startControl({
                              port,
                              robot_id: robotId,
                              mode,
                              ...(mode === 'manual'
                                ? {
                                    joints: controlJoints,
                                    move_to_middle: !partialSelection,
                                    ...(!partialSelection && state?.capabilities?.middle_tolerance
                                      ? { middle_tolerance: middleTolerance }
                                      : {}),
                                  }
                                : {}),
                              ...(mode === 'leader' ? { leader_port: leaderPort, leader_id: leaderId } : {}),
                            }),
                          ownSession,
                        )
                      }
                    >
                      {mode === 'handguide'
                        ? 'Start hand guiding'
                        : mode === 'leader'
                          ? 'Start leader teleoperation'
                          : partialSelection
                            ? 'Enable selected motors'
                            : 'Enable & move to recorded middle'}
                    </button>
                  </>
                )}
                {controlling && (
                  <>
                    <Notice kind="success">
                      <strong>
                        {session?.mode === 'manual'
                          ? centering
                            ? `Moving to recorded middle · ${Math.round((session.centering?.progress ?? 0) * 100)}%`
                            : 'On-screen controls are active.'
                          : session?.mode === 'leader'
                            ? 'Leader teleoperation is active.'
                            : 'Hand guiding is active. Torque is off.'}
                      </strong>
                      {ownsSession
                        ? 'Tab hidden or disconnected → torque off.'
                        : 'Session owned by another client. Stop it to take control.'}
                    </Notice>
                    {session?.mode === 'manual' ? (
                      <>
                        {partialControl && (
                          <p className="robot-lab-subtle">Selected motors only · recording unavailable</p>
                        )}
                        <div className="robot-lab-control-heading">
                          <h3>Joint controls</h3>
                          <span className="robot-lab-subtle">Position → target · release slider to move</span>
                          <label>
                            Step
                            <select
                              value={jointStep}
                              aria-label="Joint step size"
                              onChange={(event) => setJogStep(Number(event.target.value))}
                            >
                              <option value={0.5}>0.5° / %</option>
                              <option value={1}>1° / %</option>
                              <option value={2}>2° / %</option>
                              {state?.capabilities?.large_joint_steps && (
                                <>
                                  <option value={5}>5° / %</option>
                                  <option value={10}>10° / %</option>
                                </>
                              )}
                            </select>
                          </label>
                        </div>
                        <div className="robot-lab-joints" aria-busy={jogging}>
                          {robotJoints
                            .filter((joint) => sessionJoints.includes(joint.name))
                            .map((joint) => (
                              <JointControl
                                key={joint.name}
                                joint={joint}
                                position={session.positions[joint.name]}
                                target={session.targets?.[joint.name]}
                                sliderTarget={session.slider_targets?.[joint.name]}
                                limits={session.joint_limits?.[joint.name]}
                                sliders={!!state?.capabilities?.joint_sliders}
                                disabled={
                                  !canAct || !ownsSession || jogging || centering || !!recording?.episode_active
                                }
                                step={jointStep}
                                onJog={jog}
                                onTarget={targetJoint}
                              />
                            ))}
                        </div>
                      </>
                    ) : (
                      <RangeTable positions={session?.positions} />
                    )}
                    <div className="robot-lab-actions">
                      <button className="danger-button robot-lab-stop" onClick={stop}>
                        <span />
                        Stop & release motors
                      </button>
                      <button
                        className="secondary-button"
                        disabled={busy || !ownsSession || centering || partialControl}
                        onClick={() => setStep(4)}
                      >
                        Create a dataset
                        <Icon name="arrow" size={14} />
                      </button>
                    </div>
                  </>
                )}
              </div>
            </>
          )}
          {step === 4 && (
            <>
              <Heading title="Recording" />
              <div className="robot-lab-form">
                {partialControl && <Notice>Dataset recording requires all six joints.</Notice>}
                {!recording?.active ? (
                  <>
                    {!controlling && (
                      <Notice>
                        <strong>No control session</strong>Start teleoperation before recording.
                        <button className="text-button" onClick={() => setStep(3)}>
                          Open teleoperation
                        </button>
                      </Notice>
                    )}
                    <div className="robot-lab-fields">
                      <label>
                        Dataset name
                        <input
                          value={datasetId}
                          aria-label="Dataset name"
                          placeholder="pick-up-block"
                          onChange={(event) =>
                            setDatasetId(event.target.value.toLowerCase().replace(/[^a-z0-9_-]/g, '-'))
                          }
                          maxLength={64}
                          disabled={busy}
                        />
                        {datasetList.some((dataset) => dataset.id === datasetId) && (
                          <small role="alert">Name already used.</small>
                        )}
                      </label>
                      <label>
                        Frames per second
                        <select
                          value={fps}
                          aria-label="Recording frame rate"
                          onChange={(event) => setFps(Number(event.target.value))}
                          disabled={busy}
                        >
                          {[5, 10, 15, 20].map((value) => (
                            <option key={value} value={value}>
                              {value} fps
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>
                    <label>
                      Task instruction
                      <textarea
                        value={task}
                        aria-label="Task instruction"
                        placeholder="Pick up the blue block and place it in the tray."
                        onChange={(event) => setTask(event.target.value)}
                        disabled={busy}
                        maxLength={500}
                      />
                    </label>
                    <label className="robot-lab-check">
                      <input
                        type="checkbox"
                        checked={recordCamera}
                        disabled={busy || !availableCameras.length}
                        onChange={(event) => setRecordCamera(event.target.checked)}
                      />
                      <span>
                        <strong>Record camera</strong>
                        {availableCameras.length ? '' : 'No camera available; state and actions only.'}
                      </span>
                    </label>
                    <button
                      className="primary-button"
                      disabled={
                        !canAct ||
                        !controlling ||
                        !ownsSession ||
                        centering ||
                        partialControl ||
                        !/^[a-z0-9][a-z0-9_-]{0,63}$/.test(datasetId) ||
                        !task.trim() ||
                        datasetList.some((dataset) => dataset.id === datasetId) ||
                        !state?.capabilities?.recording
                      }
                      onClick={() =>
                        void perform('Dataset created. Start an episode when you are ready.', () =>
                          robotApi.startRecording({
                            dataset_id: datasetId,
                            task: task.trim(),
                            fps,
                            camera_index: recordCamera && availableCameras.length ? cameraIndex : null,
                          }),
                        )
                      }
                    >
                      {busy ? 'Creating dataset…' : 'Create recording dataset'}
                    </button>
                  </>
                ) : (
                  <>
                    <div className={`robot-lab-recording-status${recording.episode_active ? ' recording' : ''}`}>
                      <span className="robot-lab-recording-light" />
                      <div>
                        <strong>{recording.dataset_id}</strong>
                        <small>
                          {recording.episode_active
                            ? 'Recording an episode'
                            : recording.frames > 0
                              ? 'Episode has pending frames'
                              : 'Ready for the next episode'}{' '}
                          · {recording.episodes} saved
                        </small>
                      </div>
                      <output>{recording.frames} frames</output>
                    </div>
                    {session?.mode === 'handguide' && (
                      <Notice>Action source: observed joint positions (hand guiding).</Notice>
                    )}
                    {session?.mode === 'manual' && <Notice>Action source: commanded joint targets.</Notice>}
                    <div className="robot-lab-actions">
                      <button
                        className="primary-button"
                        disabled={
                          !canAct ||
                          !controlling ||
                          !ownsSession ||
                          centering ||
                          partialControl ||
                          recording.episode_active ||
                          recording.frames > 0
                        }
                        onClick={() => void perform('Episode recording started.', robotApi.startEpisode)}
                      >
                        <span aria-hidden="true">●</span>Start episode
                      </button>
                      <button
                        className="secondary-button"
                        disabled={!canAct || !recording.frames}
                        onClick={() => void perform('Episode saved.', robotApi.saveEpisode)}
                      >
                        Save episode
                      </button>
                      <button
                        className="secondary-button"
                        disabled={!canAct || (!recording.episode_active && !recording.frames)}
                        onClick={() => void perform('Episode discarded.', robotApi.discardEpisode)}
                      >
                        Discard episode
                      </button>
                    </div>
                    {session?.mode === 'manual' && (
                      <div className="robot-lab-joints" aria-busy={jogging}>
                        {robotJoints
                          .filter((joint) => sessionJoints.includes(joint.name))
                          .map((joint) => (
                            <JointControl
                              key={joint.name}
                              joint={joint}
                              position={session.positions[joint.name]}
                              target={session.targets?.[joint.name]}
                              sliderTarget={session.slider_targets?.[joint.name]}
                              limits={session.joint_limits?.[joint.name]}
                              sliders={!!state?.capabilities?.joint_sliders}
                              disabled={!canAct || !ownsSession || jogging || centering}
                              step={1}
                              recording
                              onJog={jog}
                              onTarget={targetJoint}
                            />
                          ))}
                      </div>
                    )}
                    <div className="robot-lab-actions">
                      <button
                        className="secondary-button"
                        disabled={
                          !canAct || recording.episode_active || recording.frames > 0 || recording.episodes === 0
                        }
                        onClick={() => void perform('Dataset finalized.', robotApi.finishRecording)}
                      >
                        Finish dataset
                      </button>
                      {recording.episodes === 0 && (
                        <button
                          className="text-button"
                          disabled={!canAct}
                          onClick={() =>
                            void perform(
                              'Unsaved recording closed; files retained on the robot computer.',
                              robotApi.abandonRecording,
                            )
                          }
                        >
                          Close unsaved recording
                        </button>
                      )}
                    </div>
                    {recording.episodes === 0 && (
                      <p className="robot-lab-subtle">
                        Closing retains unsaved files. Create a new dataset to change the frame rate.
                      </p>
                    )}
                  </>
                )}
                <hr className="robot-lab-divider" />
                <div>
                  <h3>Datasets on the robot computer</h3>
                </div>
                {datasets.error && (
                  <p className="robot-lab-subtle" role="alert">
                    {datasets.error.message}
                  </p>
                )}
                {!datasetList.length ? (
                  <div className="robot-lab-empty">
                    <Icon name="database" size={26} />
                    No datasets
                  </div>
                ) : (
                  <div className="robot-lab-datasets">
                    {datasetList.map((dataset) => (
                      <DatasetRow
                        key={dataset.id}
                        dataset={dataset}
                        canImport={canAct && !!projectId}
                        pending={pending === `Imported ${dataset.id} into this project.`}
                        onImport={() => {
                          const owner = projectId;
                          void perform(
                            `Imported ${dataset.id} into this project.`,
                            () => importRobotDataset(owner, dataset.id),
                            () => {
                              setImported({ dataset: dataset.id, projectId: owner });
                              void client.invalidateQueries({ queryKey: ['datasets', owner] });
                              void client.invalidateQueries({ queryKey: ['jobs', owner] });
                            },
                          );
                        }}
                      />
                    ))}
                  </div>
                )}
                {imported && imported.projectId === projectId && (
                  <Notice kind="success">
                    <div className="robot-lab-imported">
                      <span>{imported.dataset} is in this project’s dataset library.</span>
                      <button className="text-button" onClick={() => navigateStage('datasets')}>
                        View datasets
                      </button>
                    </div>
                  </Notice>
                )}
              </div>
            </>
          )}
          {busy && (
            <p className="robot-lab-subtle" role="status" style={{ marginTop: 16 }}>
              {pending === 'Stop requested.'
                ? 'Stopping the arm… Keep supporting it while torque is released and motor cleanup finishes.'
                : 'Working… Keep this page open.'}
            </p>
          )}
        </section>
        <aside className="robot-lab-side">
          <section className="robot-lab-card" aria-label="Camera preview">
            <div className="robot-lab-card-header">
              <h3>Camera preview</h3>
            </div>
            <div className="robot-lab-camera">
              {preview && online && !cameraError ? (
                <>
                  <img
                    alt="Live view from the selected robot camera"
                    src={robotApi.cameraUrl(cameraIndex, previewStamp)}
                    onError={() => setCameraError(true)}
                  />
                  <span className="robot-lab-camera-live">LIVE</span>
                </>
              ) : (
                <div className="robot-lab-camera-empty">
                  <CameraIcon size={27} />
                  <span>
                    {cameraError
                      ? 'Camera could not open. Check camera permissions on the robot computer or close another app using it.'
                      : 'Preview stopped'}
                  </span>
                </div>
              )}
            </div>
            <div className="robot-lab-camera-controls">
              <label>
                Camera
                <select
                  value={cameraIndex}
                  aria-label="Preview camera"
                  onChange={(event) => {
                    setCameraIndex(Number(event.target.value));
                    setCameraError(false);
                  }}
                  disabled={!online || !availableCameras.length || recording?.active}
                >
                  {!availableCameras.length && <option value={0}>No camera available</option>}
                  {availableCameras.map((camera) => (
                    <option key={camera.index} value={camera.index}>
                      {camera.name} · {camera.width}×{camera.height}
                    </option>
                  ))}
                </select>
              </label>
              <button
                className="secondary-button"
                disabled={!online || !availableCameras.length || !state?.capabilities?.camera}
                onClick={() => {
                  if (cameraError) {
                    setCameraError(false);
                    setPreviewStamp(Date.now());
                    setPreview(true);
                  } else setPreview(!preview);
                }}
              >
                {cameraError ? 'Retry camera' : preview ? 'Pause preview' : 'Start preview'}
              </button>
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}

function CameraIcon({ size = 20 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M4 7h4l2-3h4l2 3h4a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1Z" />
      <circle cx="12" cy="13" r="4" />
    </svg>
  );
}
function DatasetRow({
  dataset,
  canImport,
  pending,
  onImport,
}: {
  dataset: RobotDataset;
  canImport: boolean;
  pending: boolean;
  onImport: () => void;
}) {
  return (
    <article className="robot-lab-dataset">
      <div className="robot-lab-dataset-heading">
        <div>
          <h3>{dataset.id}</h3>
          <p>{dataset.task}</p>
        </div>
        <button className="secondary-button" disabled={!canImport || !dataset.finalized || pending} onClick={onImport}>
          {pending
            ? 'Copying to project…'
            : dataset.finalized
              ? 'Add to project'
              : dataset.abandoned
                ? 'Closed · files retained'
                : 'Not finalized'}
        </button>
      </div>
      <div className="robot-lab-dataset-meta">
        <span>{dataset.episodes} episodes</span>
        <span>{dataset.frames} frames</span>
        <span>{dataset.fps} fps</span>
        <span>{dataset.finalized ? 'Finalized' : dataset.abandoned ? 'Closed' : 'Not finalized'}</span>
        {dataset.pending_frames ? <span>{dataset.pending_frames} unsaved frames</span> : null}
      </div>
    </article>
  );
}
