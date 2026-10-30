'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { agentApi, type AgentSession } from '@/lib/agent-api';
import { robotApi, robotJoints } from '@/lib/robot-api';

const statusKey = ['robot-lab', 'status'] as const;
const message = (error: unknown) =>
  error instanceof Error ? error.message : 'Physical control could not be verified.';
const present = () => document.visibilityState === 'visible' && document.hasFocus();

type Owner = { kind: 'manual'; generation: number } | { kind: 'agent'; id: string; epoch: number };
export type PhysicalArming = {
  confirm: 'ARM AI CONTROL';
  expected_generation: number;
  camera_index: number;
  operator_present: true;
  workspace_clear: true;
};

/** This page never adopts an existing controller merely because status reports one. */
export function useAiPhysicalConnection(enabled: boolean, onSession: (session: AgentSession) => void) {
  const client = useQueryClient();
  const [port, setPort] = useState('');
  const [robotId, setRobotId] = useState('');
  const [cameraIndex, setCameraIndex] = useState<number | null>(null);
  const [preview, setPreview] = useState(false);
  const [previewStamp, setPreviewStamp] = useState(0);
  const [imageLoadedAt, setImageLoadedAt] = useState(0);
  const [cameraError, setCameraError] = useState('');
  const [operatorPresent, setOperatorPresent] = useState(false);
  const [workspaceClear, setWorkspaceClear] = useState(false);
  const [owned, setOwned] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [error, setError] = useState('');
  const owner = useRef<Owner | null>(null);
  const transfer = useRef(false);
  const connectingRef = useRef(false);
  const pendingGeneration = useRef<number | null>(null);
  const mounted = useRef(true);
  const enabledRef = useRef(enabled);
  const operatorRef = useRef(operatorPresent);
  const clearRef = useRef(workspaceClear);
  const version = useRef(0);
  const callback = useRef(onSession);
  enabledRef.current = enabled;
  callback.current = onSession;
  operatorRef.current = operatorPresent;
  clearRef.current = workspaceClear;
  const status = useQuery({
    queryKey: statusKey,
    queryFn: robotApi.status,
    enabled,
    retry: false,
    refetchInterval: enabled ? (owned || connecting ? 500 : 3000) : false,
  });
  const state = status.isError || status.data?.connected === false ? null : status.data;
  const online = !!state?.ok && state.connected !== false;
  const devices = useQuery({
    queryKey: ['robot-lab', 'devices'],
    queryFn: robotApi.devices,
    enabled: enabled && online,
    retry: false,
    staleTime: 30000,
  });
  const capable = online && state?.capabilities?.agent_control === true && state.agent_control?.available === true;
  const profile = state?.calibrations?.find((item) => item.robot_id === robotId && item.role === 'follower');
  const profileReady =
    !!profile &&
    profile.valid !== false &&
    (profile.setup_port !== port || profile.setup_fresh !== false) &&
    !!profile.sha256;
  const portReady =
    !!port &&
    !devices.isError &&
    !!devices.data?.ports.some((item) => item.device === port) &&
    (!devices.data?.saved_devices ||
      !!devices.data.saved_devices.find((d) => d.robot_id === robotId && d.port === port && d.connected && d.role));
  const cameraReady =
    preview &&
    !cameraError &&
    cameraIndex !== null &&
    imageLoadedAt > 0 &&
    Math.max(0, previewStamp - imageLoadedAt) < 2000;
  const connectedHere =
    owner.current?.kind === 'manual' &&
    state?.session?.mode === 'manual' &&
    state.session.torque_enabled &&
    state.session.generation === owner.current.generation;
  const fullArm =
    !!state?.session?.active_joints && robotJoints.every((joint) => state.session!.active_joints!.includes(joint.name));
  const generation = state?.session?.generation;
  const canArm =
    capable &&
    !!connectedHere &&
    fullArm &&
    !state?.session?.error &&
    !transfer.current &&
    !connecting &&
    operatorPresent &&
    workspaceClear &&
    cameraReady &&
    typeof generation === 'number';
  const canConnect =
    capable &&
    state?.session?.mode === 'idle' &&
    profileReady &&
    portReady &&
    operatorPresent &&
    workspaceClear &&
    cameraReady &&
    !connecting &&
    !owned &&
    typeof generation === 'number';
  const connectionIssue = !profile
    ? 'Choose a saved arm by name, then review its USB connection.'
    : profile.valid === false
      ? 'This saved calibration is invalid. Review it in Initial setup.'
      : !portReady
        ? profile.setup_port
          ? `The saved connection (${profile.setup_port}) is unavailable. Connect the arm or select its current USB port.`
          : 'Select the current USB port for this saved arm.'
        : !profile.sha256
          ? 'The saved calibration identity is unavailable. Refresh readiness or update the companion.'
          : !profileReady
            ? 'Motor setup changed on this arm’s saved connection. Review its calibration in Initial setup.'
            : !cameraReady
              ? 'Start the workspace camera preview and wait for a current image.'
              : !operatorPresent || !workspaceClear
                ? 'Review the camera image and confirm operator presence and workspace clearance.'
                : state?.session?.mode !== 'idle' && !owned
                  ? 'Stop the existing controller before connecting from this page.'
                  : null;

  const stop = useCallback(
    async (reason = 'Physical control stopped.', force = false) => {
      const previous = owner.current;
      const uncertainStart = connectingRef.current || transfer.current;
      owner.current = null;
      transfer.current = false;
      connectingRef.current = false;
      operatorRef.current = false;
      clearRef.current = false;
      const token = ++version.current;
      if (mounted.current) {
        setOwned(false);
        setConnecting(false);
        setOperatorPresent(false);
        setWorkspaceClear(false);
        setError(reason);
      }
      try {
        if (force) {
          // The visible emergency button intentionally overrides any controller.
          const result = await robotApi.stop(true);
          if (mounted.current && token === version.current) client.setQueryData(statusKey, result);
          if (previous?.kind === 'agent') {
            try {
              const session = await agentApi.stop(previous.id, true);
              if (mounted.current && token === version.current) callback.current(session);
            } catch {
              /* The direct local Stop was acknowledged. */
            }
          }
        } else if (previous?.kind === 'agent') {
          // Stale agent cleanup must never stop a newer manual controller.
          const result = await agentApi.stop(previous.id, true);
          if (mounted.current && token === version.current) callback.current(result);
        } else if (previous?.kind === 'manual' || uncertainStart) {
          const expectedGeneration = previous?.kind === 'manual' ? previous.generation : pendingGeneration.current;
          if (expectedGeneration !== null) {
            const result = await robotApi.stop(true, expectedGeneration, 'manual');
            if (mounted.current && token === version.current) client.setQueryData(statusKey, result);
          }
        }
      } catch {
        if (mounted.current && token === version.current)
          setError(
            'Stop was not acknowledged. The local lease still expires independently; use the physical Stop path and support the arm.',
          );
      } finally {
        if (mounted.current) void client.invalidateQueries({ queryKey: statusKey });
      }
    },
    [client],
  );

  useEffect(() => {
    mounted.current = true;
    let inFlight = false;
    const heartbeat = async () => {
      const active = owner.current;
      if ((!active && !connectingRef.current) || transfer.current || inFlight) return;
      if (!enabledRef.current || !operatorRef.current || !clearRef.current) {
        void stop(
          'Physical control was disabled or operator confirmations were withdrawn. Physical Stop was requested.',
        );
        return;
      }
      if (!active) return;
      inFlight = true;
      try {
        if (active.kind === 'agent') {
          const result = await agentApi.heartbeat(active.id, active.epoch);
          if (owner.current === active && mounted.current) {
            callback.current(result);
            if ('control_armed' in result && result.control_armed === false) {
              owner.current = null;
              setOwned(false);
            } else {
              active.epoch = result.authority_epoch;
            }
          }
        } else {
          const result = await robotApi.heartbeat(active.generation);
          if (owner.current === active && mounted.current && !transfer.current) {
            if (result.session?.mode !== 'manual' || result.session.generation !== active.generation) {
              owner.current = null;
              setOwned(false);
              setError('The manual controller changed. Explicitly connect again before arming AI control.');
            }
            client.setQueryData(statusKey, result);
          }
        }
      } catch {
        if (owner.current === active && !(active.kind === 'manual' && transfer.current))
          void stop('The physical control heartbeat could not be verified. Stop was requested.');
      } finally {
        inFlight = false;
      }
    };
    const timer = setInterval(() => void heartbeat(), 500);
    return () => {
      mounted.current = false;
      clearInterval(timer);
      if (owner.current || connectingRef.current || transfer.current) void stop('AI teleoperation was closed.');
    };
  }, [client, stop]);

  useEffect(() => {
    if (!preview || !enabled) return;
    const timer = setInterval(() => setPreviewStamp(Date.now()), 500);
    return () => clearInterval(timer);
  }, [enabled, preview]);

  async function connect() {
    if (!canConnect || !present()) return;
    const token = ++version.current;
    connectingRef.current = true;
    pendingGeneration.current = generation ?? null;
    setConnecting(true);
    setError('');
    try {
      const result = await robotApi.startControl({
        port,
        robot_id: robotId,
        mode: 'manual',
        joints: robotJoints.map((joint) => joint.name),
        move_to_middle: false,
      });
      if (
        !mounted.current ||
        !enabledRef.current ||
        token !== version.current ||
        !operatorRef.current ||
        !clearRef.current
      ) {
        await robotApi.stop(true, result.session?.generation ?? pendingGeneration.current ?? undefined, 'manual');
        return;
      }
      if (
        result.session?.mode !== 'manual' ||
        !result.session.torque_enabled ||
        typeof result.session.generation !== 'number'
      )
        throw new Error('A full manual hold could not be verified.');
      owner.current = { kind: 'manual', generation: result.session.generation };
      setOwned(true);
      client.setQueryData(statusKey, result);
    } catch (problem) {
      await stop(message(problem));
    } finally {
      connectingRef.current = false;
      if (mounted.current && token === version.current) setConnecting(false);
    }
  }

  function beginTransfer() {
    if (!canArm || !present()) throw new Error('Review the connected arm, camera and operator confirmations again.');
    transfer.current = true;
  }
  function adopt(session: AgentSession) {
    if (!mounted.current || !enabledRef.current || !operatorRef.current || !clearRef.current) {
      void agentApi.stop(session.id, true);
      void stop('Operator presence changed during arming.');
      return;
    }
    owner.current = { kind: 'agent', id: session.id, epoch: session.authority_epoch };
    transfer.current = false;
    setOwned(true);
  }
  function updateLease(session: AgentSession) {
    const active = owner.current;
    if (active?.kind === 'agent' && active.id === session.id) active.epoch = session.authority_epoch;
  }
  function setPresent(value: boolean) {
    operatorRef.current = value;
    setOperatorPresent(value);
    if (!value && (owner.current || connectingRef.current || transfer.current))
      void stop('Operator presence was withdrawn.');
  }
  function setClear(value: boolean) {
    clearRef.current = value;
    setWorkspaceClear(value);
    if (!value && (owner.current || connectingRef.current || transfer.current))
      void stop('Workspace clearance was withdrawn.');
  }
  const imageUrl = preview && cameraIndex !== null ? robotApi.cameraUrl(cameraIndex, previewStamp) : null;
  return {
    state,
    status,
    devices,
    capable,
    owned,
    connecting,
    error,
    port,
    robotId,
    cameraIndex,
    preview,
    imageUrl,
    cameraError,
    cameraReady,
    operatorPresent,
    workspaceClear,
    canConnect,
    connectionIssue,
    canArm,
    generation,
    connect,
    stop,
    beginTransfer,
    adopt,
    updateLease,
    setPresent,
    setClear,
    arming: canArm
      ? ({
          confirm: 'ARM AI CONTROL',
          expected_generation: generation!,
          camera_index: cameraIndex!,
          operator_present: true,
          workspace_clear: true,
        } as PhysicalArming)
      : null,
    chooseDevice: (value: string) => {
      const device = devices.data?.saved_devices?.find((d) => d.id === value);
      setRobotId(device?.robot_id ?? '');
      setPort(device?.connected ? device.port : '');
      setOperatorPresent(false);
      setWorkspaceClear(false);
    },
    choosePort: (value: string) => {
      setPort(value);
      setOperatorPresent(false);
      setWorkspaceClear(false);
    },
    chooseRobot: (value: string) => {
      setRobotId(value);
      const saved = state?.calibrations?.find((item) => item.robot_id === value && item.role === 'follower');
      if (saved?.setup_port) {
        setPort(devices.data?.ports.some((item) => item.device === saved.setup_port) ? saved.setup_port : '');
      }
      setOperatorPresent(false);
      setWorkspaceClear(false);
    },
    chooseCamera: (value: number | null) => {
      setCameraIndex(value);
      setPreview(value !== null);
      setPreviewStamp(Date.now());
      setImageLoadedAt(0);
      setCameraError('');
      setOperatorPresent(false);
      setWorkspaceClear(false);
    },
    togglePreview: (value: boolean) => {
      setPreview(value);
      setPreviewStamp(Date.now());
      setImageLoadedAt(0);
      setCameraError('');
      if (!value && owner.current) void stop('The physical camera preview was closed.');
    },
    imageLoaded: () => {
      setImageLoadedAt(Date.now());
      setCameraError('');
    },
    imageFailed: () => {
      setImageLoadedAt(0);
      setCameraError('The physical camera preview is unavailable.');
      if (owner.current) void stop('The physical camera became unavailable.');
    },
  };
}

export function AiPhysicalConnection({ physical }: { physical: ReturnType<typeof useAiPhysicalConnection> }) {
  const { state, devices, capable, owned, connecting } = physical;
  return (
    <section className="agent-control-evidence" aria-label="Physical SO101 preflight">
      <h3>Physical SO101</h3>
      <p>
        Connect the arm here after completing calibration in Initial setup. Connecting enables torque and holds the
        measured current pose; it does not start an AI task. Keep the arm supported and Stop within reach.
      </p>
      {!capable && (
        <p role="status">
          The companion has not advertised the governed AI control capability. Update or connect the companion before
          arming physical control.
        </p>
      )}
      {devices.isSuccess && devices.data.ports.length === 0 && (
        <p role="status">
          No follower USB adapter was detected. Connect the follower USB adapter and refresh readiness.
        </p>
      )}
      {devices.data?.saved_devices ? (
        <div className="workbench-field-grid">
          <label className="workbench-field">
            Follower device
            <select
              aria-label="Follower device"
              value={devices.data.saved_devices.find((d) => d.robot_id === physical.robotId)?.id ?? ''}
              disabled={owned || connecting}
              onChange={(e) => physical.chooseDevice(e.target.value)}
            >
              <option value="">Choose a configured device</option>
              {devices.data.saved_devices.map((d) => (
                <option key={d.id} value={d.id} disabled={!d.role || !d.connected}>
                  {d.name || d.port || d.robot_id}
                  {!d.connected ? ' · Disconnected' : ''}
                </option>
              ))}
            </select>
            <small>{physical.port || 'Choose an arm configured in Initial setup → Devices.'}</small>
          </label>
        </div>
      ) : (
        <>
          {
            <div className="workbench-field-grid">
              <label className="workbench-field">
                Saved arm
                <select
                  value={physical.robotId}
                  disabled={owned || connecting}
                  onChange={(event) => physical.chooseRobot(event.target.value)}
                >
                  <option value="">Select a saved arm</option>
                  {state?.calibrations
                    ?.filter((profile) => profile.role === 'follower')
                    .map((profile) => (
                      <option key={profile.robot_id} value={profile.robot_id} disabled={profile.valid === false}>
                        {profile.robot_id}
                        {profile.valid === false ? ' · invalid' : profile.setup_fresh === false ? ' · stale' : ''}
                      </option>
                    ))}
                </select>
              </label>
              <label className="workbench-field">
                Robot USB port
                <select
                  value={physical.port}
                  disabled={owned || connecting}
                  onChange={(event) => physical.choosePort(event.target.value)}
                >
                  <option value="">Select a device</option>
                  {devices.data?.ports.map((port) => (
                    <option key={port.device} value={port.device}>
                      {port.device} · {port.description}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          }
        </>
      )}
      {physical.robotId && (
        <p>
          Configured devices remember their USB connection. If the port changed, select the current connection; the
          arm’s calibration is checked again before torque is enabled.
        </p>
      )}
      <section className="agent-workspace-camera" aria-label="Workspace camera">
        <h4>Workspace camera</h4>
        <label className="workbench-field">
          Physical camera
          <select
            value={physical.cameraIndex ?? ''}
            disabled={owned || connecting}
            onChange={(event) => physical.chooseCamera(event.target.value ? Number(event.target.value) : null)}
          >
            <option value="">Select a camera</option>
            {devices.data?.cameras.map((camera) => (
              <option key={camera.index} value={camera.index}>
                {camera.name}
              </option>
            ))}
          </select>
        </label>
        <label className="workbench-check">
          <input
            type="checkbox"
            checked={physical.preview}
            disabled={connecting || physical.cameraIndex === null}
            onChange={(event) => physical.togglePreview(event.target.checked)}
          />
          Show the local camera preview
        </label>
        {physical.imageUrl ? (
          <div className="agent-camera-preview">
            <img
              key={physical.cameraIndex}
              className="agent-control-camera"
              src={physical.imageUrl}
              alt="Local SO101 camera preview"
              onLoad={physical.imageLoaded}
              onError={physical.imageFailed}
            />
            {!physical.cameraReady && !physical.cameraError && <p role="status">Loading camera preview…</p>}
          </div>
        ) : (
          <div className="agent-camera-placeholder">
            <p>Select a camera to preview your workspace before connecting the arm.</p>
          </div>
        )}
        {physical.preview && <p>Local preview. AI planning sends images only after you enable cloud consent.</p>}
        {devices.data?.camera_error && <p role="alert">{devices.data.camera_error}</p>}
        {physical.cameraError && <p role="alert">{physical.cameraError}</p>}
      </section>
      <label className="workbench-check">
        <input
          type="checkbox"
          checked={physical.operatorPresent}
          onChange={(event) => physical.setPresent(event.target.checked)}
        />
        I am at the arm, supporting it where needed, with Stop within reach.
      </label>
      <label className="workbench-check">
        <input
          type="checkbox"
          checked={physical.workspaceClear}
          onChange={(event) => physical.setClear(event.target.checked)}
        />
        The arm and its full movement path are clear; the selected camera shows the intended workspace.
      </label>
      {!owned && physical.connectionIssue && <p role="status">{physical.connectionIssue}</p>}
      <div className="workbench-actions">
        <button
          type="button"
          className="primary-button"
          disabled={!physical.canConnect}
          onClick={() => void physical.connect()}
        >
          {connecting ? 'Connecting arm…' : 'Connect arm and hold current pose'}
        </button>
        <button
          type="button"
          className="danger-button"
          disabled={!owned && !connecting && !state?.session?.torque_enabled}
          onClick={() => void physical.stop('Stop requested. Support the arm while termination is verified.', true)}
        >
          Stop arm and disable torque
        </button>
        <button
          type="button"
          className="secondary-button"
          disabled={connecting}
          onClick={() => {
            void physical.status.refetch();
            void devices.refetch();
          }}
        >
          Refresh physical readiness
        </button>
      </div>
      <p>
        Physical control is supervised. This executor has joint limits and bounded segments; it does not provide
        collision avoidance, calibrated Cartesian grasping or contact-force control.
      </p>
      {physical.error && <p role="alert">{physical.error}</p>}
    </section>
  );
}
