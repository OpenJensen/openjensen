'use client';

import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Icon } from '@/components/icon';
import { useWorkspace } from '@/components/workspace-context';
import {
  robotApi,
  robotJoints,
  type ControlMode,
  type RobotJoint,
  type RobotProbe,
  type RobotRole,
  type RobotStatus,
} from '@/lib/robot-api';
import { So101JointGuide } from './so101-joint-guide';
import { steps, setupOrder, statusKey, readable, isStatus } from './robot-lab-shared';

// Robot Lab session state and actions. RobotLab renders each setup step from this
// in its own component (robot-lab-connect.tsx, robot-lab-motor-ids.tsx, ...).
export function useRobotLab() {
  const { projectId, navigateStage } = useWorkspace();
  const client = useQueryClient();
  const [step, setStep] = useState(0);
  const [followerDevice, setFollowerDevice] = useState('');
  const [followerId, setFollowerId] = useState('my-so101');
  const [role, setRole] = useState<RobotRole>('follower');
  const [leaderPort, setLeaderPort] = useState('');
  const [leaderId, setLeaderId] = useState('my-so101-leader');
  const port = role === 'leader' ? leaderPort : followerDevice;
  const robotId = role === 'leader' ? leaderId : followerId;
  const setPort = role === 'leader' ? setLeaderPort : setFollowerDevice;
  const setRobotId = role === 'leader' ? setLeaderId : setFollowerId;
  const armLabel = role === 'leader' ? 'Leader' : 'Follower';
  const validRobotName = /^[A-Za-z0-9_-]{1,64}$/.test(robotId);
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
  const calibrationSession = session?.mode === 'calibration' || calibration?.phase === 'interrupted';
  // Older companions put this informational recovery message in the error field.
  const recoveredCheckpoint =
    calibration?.recoverable && calibration.error === 'Calibration checkpoint recovered on the robot computer';
  const calibrationError =
    step === 2
      ? error || (!recoveredCheckpoint && calibration?.error) || (calibrationSession && session?.error) || null
      : null;
  const previousSessionError = step === 2 && !calibrationSession && session?.error !== error ? session?.error : null;
  const recording = state?.recording;
  const assembled = state?.assembled?.port === port ? state.assembled : undefined;
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
  const motorDefinition = robotJoints.find((item) => item.name === setupOrder[motorIndex])!;
  const selectedMotor =
    role === 'leader' && motorDefinition.name === 'gripper'
      ? { ...motorDefinition, label: 'Trigger' }
      : motorDefinition;
  const availableCameras = devices.data?.cameras ?? [];
  const busy = !!pending;
  const canAct = online && !busy && !reconnecting;
  const portConnected =
    online && !!port && !devices.isError && !!devices.data?.ports.some((item) => item.device === port);
  const leaderConnected =
    online && !!leaderPort && !devices.isError && !!devices.data?.ports.some((item) => item.device === leaderPort);
  const canUseArm = canAct && portConnected;
  const usbStatus = devices.isPending
    ? `Checking ${role} USB…`
    : devices.isError
      ? `${armLabel} USB status unavailable`
      : portConnected
        ? `${armLabel} USB connected · Idle`
        : port
          ? `${armLabel} USB disconnected`
          : `Select the ${role} USB device`;
  const datasetList = datasets.data?.datasets ?? state?.datasets ?? [];
  const missingHardwareTools = Object.entries(state?.dependencies ?? {}).filter(
    ([name, dependency]) => !dependency.installed && ['serial', 'scservo_sdk', 'cv2'].includes(name),
  );
  const unavailableFeatures = state?.capabilities && Object.values(state.capabilities).some((available) => !available);

  useEffect(() => {
    mounted.current = true;
    try {
      const saved = JSON.parse(localStorage.getItem('openjensen.robot-lab.preferences') ?? '{}');
      if (typeof saved.port === 'string') setFollowerDevice(saved.port);
      if (typeof saved.robotId === 'string') setFollowerId(saved.robotId);
      if (saved.role === 'leader' || saved.role === 'follower') setRole(saved.role);
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
    if (role === 'follower' && !followerDevice)
      setFollowerDevice(devices.data.ports.find((item) => item.device !== leaderPort)?.device ?? '');
    const cameras = devices.data.cameras;
    if (!cameras.some((camera) => camera.index === cameraIndex)) setCameraIndex(cameras[0]?.index ?? 0);
  }, [devices.data, followerDevice, leaderPort, role, cameraIndex]);
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
    if (session?.error) setNotice(null);
  }, [session?.error]);
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
      if (session.role === 'leader') {
        if (session.port) setLeaderPort(session.port);
        if (session.robot_id) setLeaderId(session.robot_id);
      } else {
        if (session.port) setFollowerDevice(session.port);
        if (session.robot_id) setFollowerId(session.robot_id);
      }
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
      role={role}
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
  const centering = !!session?.centering?.active || session?.rest_position?.phase === 'returning';
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
        JSON.stringify({ port: followerDevice, robotId: followerId, role, leaderPort, leaderId, middleTolerance }),
      );
      return true;
    } catch {
      return false;
    }
  }
  function saveDeviceName() {
    if (active || busy || !validRobotName) return;
    if (!remember()) {
      setError('The browser could not save this name. Check that local storage is available.');
      return;
    }
    setError(null);
    setNotice(
      `${armLabel} name saved in this browser: ${robotId}. ${selectedProfile?.valid !== false && selectedProfile ? 'Saved calibration found; scan the arm to check it.' : 'Calibrate this arm to save motor settings under this name.'}`,
    );
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
      {armLabel} USB port
      <select
        aria-label={`${armLabel} USB port`}
        value={port}
        disabled={active || busy || !online}
        onChange={(event) => updatePort(event.target.value)}
      >
        <option value="">Select the {role} USB device</option>
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
      <div className="robot-lab-name-field">
        <label htmlFor="robot-profile-name">{armLabel} name</label>
        <div className="robot-lab-name-controls">
          <input
            id="robot-profile-name"
            aria-label={`${armLabel} name`}
            aria-describedby="robot-profile-name-help"
            aria-invalid={!validRobotName}
            value={robotId}
            disabled={active || busy}
            onChange={(event) => {
              setRobotId(event.target.value);
              setMotionReady(false);
              setNotice(null);
            }}
            maxLength={64}
            pattern="[A-Za-z0-9_-]+"
          />
          <button
            type="button"
            className="secondary-button"
            disabled={active || busy || !validRobotName}
            onClick={saveDeviceName}
          >
            Save name
          </button>
        </div>
        <small id="robot-profile-name-help">
          {validRobotName
            ? 'Used for this arm’s calibration profile.'
            : 'Use 1–64 letters, numbers, hyphens or underscores.'}
        </small>
      </div>
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
  function selectArmRole(nextRole: RobotRole) {
    if (active || busy || nextRole === role) return;
    setRole(nextRole);
    setMotionReady(false);
    setNotice(null);
    setError(null);
  }
  function openStep(index: number) {
    if (index >= 3 && role === 'leader') {
      if (active || busy) return;
      setRole('follower');
      setMode('leader');
      setMotionReady(false);
    }
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
        text: online ? `Connect the ${role} arm’s USB controller.` : 'Robot companion unavailable.',
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
                      label: role === 'leader' ? 'Set up follower control' : 'Open joint controls',
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
      value: session?.error
        ? 'Needs attention'
        : controlling
          ? 'Active'
          : !portConnected
            ? 'Needs connection'
            : role === 'leader'
              ? 'Needs follower'
              : selectedCalibrated
                ? 'Ready'
                : 'Needs calibration',
      good: controlling && !session?.error,
      warn: !!session?.error,
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
              disabled={busy || (active && role === 'leader' && index >= 3)}
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
          {role === 'leader' ? 'Set up follower control' : 'Open teleoperation'} <Icon name="arrow" size={13} />
        </button>
      )}
    </div>
  );

  return {
    active,
    armLabel,
    allJointsMapped,
    assembled,
    assembledReady,
    assigned,
    availableCameras,
    busy,
    calibrated,
    calibration,
    calibrationError,
    previousSessionError,
    calibrationStage,
    calibrationSummary,
    cameraError,
    cameraIndex,
    canAct,
    canUseArm,
    capturedRanges,
    centering,
    checkArm,
    checkedCount,
    checkingConnection,
    client,
    confirmJoint,
    controlJointOptions,
    controlJoints,
    controlling,
    datasetId,
    datasetList,
    datasets,
    devices,
    error,
    followerMismatch,
    followerPort,
    fps,
    identifyMethod,
    identity,
    imported,
    isolated,
    jog,
    jogging,
    jointGuide,
    jointStep,
    leaderCalibrated,
    leaderConnected,
    leaderId,
    leaderPort,
    localCalibrationRef,
    mappingComplete,
    mappingValues,
    middleTolerance,
    mismatched,
    missingHardwareTools,
    mode,
    motionChecks,
    motionReady,
    motorIndex,
    navigateStage,
    notice,
    online,
    openStep,
    ownSession,
    ownedRef,
    ownsSession,
    partialControl,
    partialSelection,
    pending,
    perform,
    port,
    portConnected,
    preview,
    previewStamp,
    profileCheck,
    projectId,
    readiness,
    reconnecting,
    recordCamera,
    recording,
    refresh,
    reidentify,
    responding,
    robotId,
    role,
    scan,
    selectIllustratedJoint,
    selectedCalibrated,
    selectedMotor,
    selectedProfile,
    selectArmRole,
    session,
    sessionJoints,
    setAssembledReady,
    setAssigned,
    setCameraError,
    setCameraIndex,
    setDatasetId,
    setFps,
    setIdentifyMethod,
    setImported,
    setIsolated,
    setJogStep,
    setLeaderId,
    setLeaderPort,
    setMiddleTolerance,
    setMode,
    setMotionChecks,
    setMotionReady,
    setMotorIndex,
    setOwnsSession,
    setPreview,
    setPreviewStamp,
    setProbe,
    setRecordCamera,
    setReidentify,
    setRole,
    setStep,
    setTask,
    setVerificationReady,
    setWiggleId,
    setWiggleReady,
    state,
    status,
    step,
    stop,
    targetJoint,
    task,
    unavailableFeatures,
    usbStatus,
    verification,
    verificationReady,
    verifyNext,
    wiggleId,
    wiggleReady,
  };
}

export type RobotLabState = ReturnType<typeof useRobotLab>;
