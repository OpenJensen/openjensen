'use client';

import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Icon } from '@/components/icon';
import { ApiError } from '@/lib/api';
import { useWorkspace } from '@/components/workspace-context';
import {
  robotApi,
  robotJoints,
  type ControlMode,
  type RobotJoint,
  type RobotDevice,
  type RobotProbe,
  type RobotRole,
  type RobotStatus,
} from '@/lib/robot-api';
import { So101JointGuide } from './so101-joint-guide';
import { setupOrder, statusKey, readable, isStatus } from './robot-lab-shared';

type CalibrationConfirmation = {
  phase: 'center' | 'ranges' | 'saved';
  port: string;
  robotId: string;
  role: RobotRole;
  generation?: number;
  newSession: boolean;
  label: string | null;
  failure: string;
  updatedAt: number;
  token: number;
};

function confirmsCalibration(value: RobotStatus | undefined, expected: CalibrationConfirmation) {
  const session = value?.session;
  return !!(
    value?.ok &&
    value.connected !== false &&
    session &&
    value.calibration?.phase === expected.phase &&
    !value.calibration.error &&
    !session.error &&
    session.mode === (expected.phase === 'saved' ? 'idle' : 'calibration') &&
    session.port === expected.port &&
    session.robot_id === expected.robotId &&
    session.role === expected.role &&
    (expected.generation === undefined ||
      (typeof session.generation === 'number' &&
        (expected.newSession
          ? session.generation !== expected.generation
          : session.generation === expected.generation)))
  );
}

// Robot Lab session state and actions. RobotLab renders each setup step from this
// in its own component (robot-lab-connect.tsx, robot-lab-motor-ids.tsx, ...).
export function useRobotLab() {
  const { projectId, navigateStage } = useWorkspace();
  const client = useQueryClient();
  const [step, setStep] = useState(0);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string | null>(null);
  const [deviceName, setDeviceName] = useState('');
  const [followerDevice, setFollowerDevice] = useState('');
  const [followerId, setFollowerId] = useState('my-so101');
  const [role, setRole] = useState<RobotRole>('follower');
  const [leaderPort, setLeaderPort] = useState('');
  const [leaderId, setLeaderId] = useState('my-so101-leader');
  const port = role === 'leader' ? leaderPort : followerDevice;
  const robotId = role === 'leader' ? leaderId : followerId;
  const setPort = role === 'leader' ? setLeaderPort : setFollowerDevice;
  const armLabel = role === 'leader' ? 'Leader' : 'Follower';
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
  const [fps, setFps] = useState<number | null>(null);
  const [recordCamera, setRecordCamera] = useState(true);
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [awaitingCalibration, setAwaitingCalibration] = useState<CalibrationConfirmation | null>(null);
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
  const catalog: RobotDevice[] =
    devices.data?.saved_devices ??
    (devices.data?.ports ?? []).map((entry, index) => {
      const profile =
        state?.calibrations?.find((p) => p.setup_port === entry.device) ??
        (index === 0 ? state?.calibrations?.find((p) => p.robot_id === followerId) : undefined);
      return {
        id: entry.device,
        port: entry.device,
        serial_number: entry.serial_number,
        name: profile?.robot_id ?? '',
        robot_id: profile?.robot_id ?? (index === 0 ? followerId : leaderId),
        role: profile?.role ?? null,
        profile_role: profile?.role ?? null,
        connected: true,
        description: entry.description,
        calibration: profile ? 'saved' : 'missing',
      };
    });
  const selectedDevice = catalog.find((d) => d.id === selectedDeviceId);
  function chooseDevice(device: RobotDevice) {
    if (active || busy) return;
    const deviceRole = device.role ?? 'follower';
    setSelectedDeviceId(device.id);
    setDeviceName(device.name);
    setRole(deviceRole);
    if (deviceRole === 'leader') {
      setLeaderPort(device.port);
      setLeaderId(device.robot_id);
    } else {
      setFollowerDevice(device.port);
      setFollowerId(device.robot_id);
    }
    setStep(0);
    setNotice(null);
    setError(null);
  }
  function selectControlDevice(id: string, slot: RobotRole) {
    if (active || busy || recording?.active) return;
    const device = catalog.find((d) => d.id === id);
    setRole('follower');
    if (slot === 'follower') {
      setFollowerDevice(device?.port ?? '');
      setFollowerId(device?.robot_id ?? '');
    } else {
      setLeaderPort(device?.port ?? '');
      setLeaderId(device?.robot_id ?? '');
    }
    setMotionReady(false);
    setNotice(null);
    setError(null);
  }
  async function saveDeviceConfiguration(nextRole: RobotRole = role) {
    if (!selectedDevice || active || busy) return;
    await perform(
      'Device configuration saved.',
      () =>
        robotApi.configureDevice({
          id: selectedDevice.id,
          port,
          name: deviceName,
          role: nextRole,
        }),
      (result) => {
        client.setQueryData(['robot-lab', 'devices'], result);
        setSelectedDeviceId(
          result.saved_devices?.find((d) => d.robot_id === selectedDevice.robot_id)?.id ?? selectedDevice.id,
        );
        setRole(nextRole);
        if (nextRole === 'leader') {
          setLeaderPort(port);
          setLeaderId(robotId);
        } else {
          setFollowerDevice(port);
          setFollowerId(robotId);
        }
      },
    );
  }
  async function removeDevice(id?: string) {
    if (!online || active || busy || recording?.active) return;
    const removed = catalog.find((device) => device.id === (id ?? selectedDevice?.id));
    if (!removed) return;
    await perform(
      'Device removed. Its calibration files are kept; you can restore it from Removed devices.',
      () => robotApi.removeDevice(removed.id),
      (result) => {
        client.setQueryData(['robot-lab', 'devices'], result);
        setSelectedDeviceId(null);
        if (followerId === removed.robot_id) {
          setFollowerDevice('');
          setFollowerId('');
        }
        if (leaderId === removed.robot_id) {
          setLeaderPort('');
          setLeaderId('');
        }
        try {
          const key = 'openjensen.robot-lab.preferences';
          const saved = JSON.parse(localStorage.getItem(key) ?? '{}');
          if (saved.robotId === removed.robot_id) Object.assign(saved, { port: '', robotId: '' });
          if (saved.leaderId === removed.robot_id) Object.assign(saved, { leaderPort: '', leaderId: '' });
          localStorage.setItem(key, JSON.stringify(saved));
        } catch {
          // Browser preference storage is optional; the registry removal succeeded.
        }
        setProbe(null);
        setMotionReady(false);
      },
    );
  }
  async function restoreDevice(id: string) {
    if (active || busy || recording?.active) return;
    await perform(
      'Device restored.',
      () => robotApi.restoreDevice(id),
      (result) => client.setQueryData(['robot-lab', 'devices'], result),
    );
  }
  const session = state?.session;
  const calibration =
    state?.calibration && (!state.calibration.port || state.calibration.port === port) ? state.calibration : undefined;
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
  const previousSessionError =
    session?.mode === 'idle' && (step !== 2 || !calibrationSession) && session.error !== error ? session.error : null;
  const recording = state?.recording;
  const assembled =
    state?.assembled?.port === port
      ? state.assembled
      : selectedDevice?.motor_ids && state?.assembled
        ? {
            ...state.assembled,
            port,
            phase: 'complete' as const,
            mapping: Object.fromEntries(robotJoints.map((j) => [j.name, j.id])),
            ids: [1, 2, 3, 4, 5, 6],
            calibration_required: false,
          }
        : undefined;
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
  const assignedFollower = catalog.find((d) => d.port === port && d.robot_id === robotId);
  const calibrated =
    (!devices.data?.saved_devices || (!!assignedFollower?.role && assignedFollower.connected)) &&
    session?.mode !== 'setup' &&
    profileUsable(followerProfile) &&
    !followerMismatch &&
    !followerScanInvalid;
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
  const leaderCalibrated =
    (!devices.data?.saved_devices ||
      !!catalog.find((d) => d.port === leaderPort && d.robot_id === leaderId && d.role && d.connected)) &&
    profileFor(leaderId, 'leader')?.valid !== false &&
    !!profileFor(leaderId, 'leader');
  const motorDefinition = robotJoints.find((item) => item.name === setupOrder[motorIndex])!;
  const selectedMotor =
    role === 'leader' && motorDefinition.name === 'gripper'
      ? { ...motorDefinition, label: 'Trigger' }
      : motorDefinition;
  const availableCameras = devices.data?.cameras ?? [];
  const busy = !!pending;
  const canAct = online && !busy && !reconnecting && !awaitingCalibration;
  const portConnected =
    online &&
    !!port &&
    !devices.isError &&
    !!devices.data?.ports.some((item) => item.device === port) &&
    (step >= 3 || !selectedDevice || selectedDevice.connected);
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
      if (ownedRef.current && !localCalibrationRef.current)
        void robotApi
          .stop(
            true,
            undefined,
            undefined,
            lastConnectedState.current?.capabilities?.persistent_diagnostics ? 'page_closed' : undefined,
          )
          .catch(() => undefined);
    };
  }, []);
  useEffect(() => {
    if (!devices.data) return;
    if (!devices.data.saved_devices && role === 'follower' && !followerDevice)
      setFollowerDevice(devices.data.ports.find((item) => item.device !== leaderPort)?.device ?? '');
    const cameras = devices.data.cameras;
    if (!cameras.some((camera) => camera.index === cameraIndex)) setCameraIndex(cameras[0]?.index ?? 0);
  }, [devices.data, followerDevice, leaderPort, role, cameraIndex]);
  useEffect(() => {
    if (active || busy || !selectedDevice || !selectedDevice.connected || step >= 3) return;
    if (selectedDevice.port !== port) {
      if (role === 'leader') setLeaderPort(selectedDevice.port);
      else setFollowerDevice(selectedDevice.port);
    }
  }, [selectedDevice?.port, selectedDevice?.connected, step, active, busy, port, role]);
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
      setSelectedDeviceId(null);
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
    if (session?.mode !== 'calibration' || !session.port || selectedDeviceId) return;
    const device = catalog.find((d) => d.port === session.port && d.robot_id === session.robot_id);
    if (device) {
      setSelectedDeviceId(device.id);
      setDeviceName(device.name);
    }
  }, [devices.data, session?.mode, session?.port, session?.robot_id, selectedDeviceId]);
  useEffect(() => {
    if (
      !awaitingCalibration ||
      awaitingCalibration.token !== epoch.current ||
      !online ||
      status.dataUpdatedAt <= awaitingCalibration.updatedAt
    )
      return;
    if (confirmsCalibration(status.data, awaitingCalibration)) {
      setError(null);
      setAwaitingCalibration(null);
      setNotice(awaitingCalibration.label);
      if (awaitingCalibration.phase === 'saved') {
        ownedRef.current = false;
        setOwnsSession(false);
      }
    } else {
      // A fresh read did not confirm this arm/stage. Keep a real refusal or fault visible.
      setError(status.data?.calibration?.error || status.data?.session?.error || awaitingCalibration.failure);
      setNotice(null);
      setAwaitingCalibration(null);
    }
  }, [awaitingCalibration, online, status.data, status.dataUpdatedAt]);
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
        const current = client.getQueryData<RobotStatus>(statusKey)?.session;
        const next =
          current?.mode === 'policy' && current.generation !== undefined
            ? await robotApi.policyHeartbeat(current.generation)
            : await robotApi.heartbeat(current?.generation);
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
      const mode = lastConnectedState.current?.session?.mode;
      if (mode === 'manual' || mode === 'leader' || mode === 'handguide') return;
      if (document.visibilityState === 'hidden' && ownedRef.current && !localCalibrationRef.current) {
        ownedRef.current = false;
        setOwnsSession(false);
        setNotice(null);
        setError('Robot Lab was hidden; stopping the session. Keep this page visible during setup.');
        void robotApi
          .stop(
            true,
            undefined,
            undefined,
            lastConnectedState.current?.capabilities?.persistent_diagnostics ? 'page_hidden' : undefined,
          )
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
    calibrationTarget?: 'start' | 'resume' | 'ranges' | 'saved',
  ) {
    const token = ++epoch.current;
    const calibrationExpected = calibrationTarget
      ? {
          phase:
            calibrationTarget === 'start'
              ? ('center' as const)
              : calibrationTarget === 'resume'
                ? (calibration?.interrupted_stage ??
                  (Object.keys(calibration?.offsets ?? {}).length ? ('ranges' as const) : ('center' as const)))
                : calibrationTarget,
          port:
            calibrationTarget === 'resume'
              ? (calibration?.port ?? port)
              : session?.mode === 'calibration'
                ? (session.port ?? port)
                : port,
          robotId:
            calibrationTarget === 'resume'
              ? (calibration?.robot_id ?? robotId)
              : session?.mode === 'calibration'
                ? (session.robot_id ?? robotId)
                : robotId,
          role:
            calibrationTarget === 'resume'
              ? (calibration?.role ?? role)
              : session?.mode === 'calibration'
                ? (session.role ?? role)
                : role,
          generation: session?.generation,
          newSession: calibrationTarget === 'start' || calibrationTarget === 'resume',
          label: announce ? label : null,
          token,
        }
      : null;
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
      setNotice(announce && !(isStatus(result) && result.session?.error) ? label : null);
      void client.invalidateQueries({ queryKey: ['robot-lab', 'datasets'] });
      void client.invalidateQueries({ queryKey: statusKey });
    } catch (failure) {
      if (mounted.current && token === epoch.current) {
        const uncertain = !(failure instanceof ApiError) || [502, 503, 504].includes(failure.status);
        if (calibrationExpected && uncertain) {
          setAwaitingCalibration({
            ...calibrationExpected,
            failure: readable(failure),
            updatedAt: client.getQueryState(statusKey)?.dataUpdatedAt ?? 0,
          });
          setNotice('Checking calibration progress on the robot computer…');
        } else {
          setError(readable(failure));
        }
        // Read status to confirm an uncertain outcome; never replay motor writes.
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
    void perform('Arm stopped.', () => robotApi.stop());
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
  const followerPort = <p className="robot-lab-subtle">USB · {port || 'Not assigned'}</p>;

  const identity = null;

  const checkedCount = Object.values(motionChecks).filter((check) => check.correct).length;
  const mismatched = robotJoints.find((joint) => motionChecks[joint.name]?.correct === false);
  const nextUnchecked = setupOrder.find((joint) => !motionChecks[joint]?.correct);
  const scan = probe?.port === port && portConnected ? probe : null;
  const responding =
    scan?.motors.filter((motor) => scan.expected_ids.includes(motor.id) && motor.stable && motor.model === 777)
      .length ?? 0;
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
        : scan.scan_scope === 'all_addresses'
          ? `${scan.motors.filter((motor) => motor.stable && motor.model === 777).length} addresses responding · ${responding}/6 expected IDs`
          : `${responding}/6 expected IDs responding`;
  const mappingLabel = mismatched
    ? 'Mismatch'
    : !mappingComplete
      ? 'Not identified'
      : checkedCount === 6
        ? '6/6 verified'
        : `${checkedCount}/6 verified`;
  function selectArmRole(nextRole: RobotRole) {
    if (active || busy || nextRole === role) return;
    if (nextRole === 'leader') {
      setLeaderPort(port);
      setLeaderId(robotId);
    } else {
      setFollowerDevice(port);
      setFollowerId(robotId);
    }
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
    if (index >= 3 && step < 3 && !active) {
      const follower =
        catalog.find((d) => d.port === followerDevice && d.role === 'follower') ??
        catalog.find((d) => d.role === 'follower' && d.connected);
      const leader = catalog.find((d) => d.role === 'leader' && d.connected && d.id !== follower?.id);
      if (follower) {
        setFollowerDevice(follower.port);
        setFollowerId(follower.robot_id);
      }
      if (leader) {
        setLeaderPort(leader.port);
        setLeaderId(leader.robot_id);
      }
      setRole('follower');
    }
    setStep(index);
    setNotice(null);
    setError(null);
  }
  function checkArm() {
    remember();
    void perform(
      'Arm scan completed.',
      async () => {
        const result = await robotApi.probe(port, robotId, role);
        return result.ready || result.scan_scope !== 'expected_ids'
          ? result
          : robotApi.probe(port, robotId, role, true);
      },
      (result) => setProbe(result),
      false,
    );
  }
  function discoverMotorIds() {
    remember();
    void perform(
      'Motor address search completed.',
      async () => {
        const result = await robotApi.probe(port, robotId, role, true);
        if (result.scan_scope !== 'all_addresses')
          throw new Error('Robot control service needs an update before searching all motor IDs.');
        return result;
      },
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
            text: `${responding}/6 expected IDs respond reliably. Review the motor addresses.`,
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
  const readiness = null;
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
    updatePort,
    catalog,
    selectedDevice,
    selectedDeviceId,
    setSelectedDeviceId,
    deviceName,
    setDeviceName,
    chooseDevice,
    selectControlDevice,
    saveDeviceConfiguration,
    removeDevice,
    restoreDevice,
    calibrationLabel,
    mappingLabel,
    stepStates,
    nextAction,
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
    discoverMotorIds,
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
