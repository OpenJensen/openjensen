'use client';

import { useEffect, useRef, useState, type ReactNode } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Icon } from '@/components/icon';
import { useWorkspace } from '@/components/workspace-context';
import { importRobotDataset, robotApi, robotJoints, type ControlMode, type JointValues, type RobotDataset, type RobotJoint, type RobotProbe, type RobotRole, type RobotStatus } from '@/lib/robot-api';
import { So101JointGuide } from './so101-joint-guide';
import './robot-lab.css';

const steps = [{ title: 'Connect' }, { title: 'Motor IDs' }, { title: 'Calibrate' }, { title: 'Teleoperate' }, { title: 'Record' }];
const setupOrder: RobotJoint[] = ['gripper', 'wrist_roll', 'wrist_flex', 'elbow_flex', 'shoulder_lift', 'shoulder_pan'];
const statusKey = ['robot-lab', 'status'];
const number = (value: number | undefined, digits = 0) => value === undefined || !Number.isFinite(value) ? '—' : value.toFixed(digits);
const readable = (error: unknown) => error instanceof Error ? error.message : 'The operation could not be completed.';
const isStatus = (value: unknown): value is RobotStatus => !!value && typeof value === 'object' && 'session' in value;

function Heading({ title }: { title: string }) {
  return <div className="robot-lab-panel-heading"><h2>{title}</h2></div>;
}
function Notice({ children, kind = 'info' }: { children: ReactNode; kind?: 'info' | 'warning' | 'success' }) { return <div className={`robot-lab-${kind}`}>{children}</div>; }
function RangeTable({ positions = {}, mins = {}, maxes = {}, calibration = false }: { positions?: JointValues; mins?: JointValues; maxes?: JointValues; calibration?: boolean }) {
  return <div className="robot-lab-table-wrap"><table className="robot-lab-table"><thead><tr><th>Joint</th><th>{calibration ? 'Current · ticks' : 'Position'}</th>{calibration && <><th>Min</th><th>Max</th><th>Range</th></>}</tr></thead><tbody>{robotJoints.map(joint => <tr key={joint.name}><td>{joint.label}</td><td>{number(positions[joint.name], calibration ? 0 : 1)}{!calibration && positions[joint.name] !== undefined ? joint.unit === 'percent' ? '%' : '°' : ''}</td>{calibration && <><td>{number(mins[joint.name])}</td><td>{number(maxes[joint.name])}</td><td>{joint.name === 'wrist_roll' ? 'Full rotation' : mins[joint.name] !== undefined && maxes[joint.name] !== undefined ? number(maxes[joint.name]! - mins[joint.name]!) : '—'}</td></>}</tr>)}</tbody></table></div>;
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
  const [identifyMethod, setIdentifyMethod] = useState<'hand' | 'wiggle'>('hand');
  const [wiggleReady, setWiggleReady] = useState(false);
  const [wiggleId, setWiggleId] = useState('');
  const [motorIndex, setMotorIndex] = useState(0);
  const [assigned, setAssigned] = useState<RobotJoint[]>([]);
  const [probe, setProbe] = useState<RobotProbe | null>(null);
  const [motionReady, setMotionReady] = useState(false);
  const [jogStep, setJogStep] = useState(1);
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
  const [notice, setNotice] = useState<string | null>(null);
  const [ownsSession, setOwnsSession] = useState(false);
  const [imported, setImported] = useState<{ dataset: string; projectId: string } | null>(null);
  const ownedRef = useRef(false);
  const epoch = useRef(0);
  const mounted = useRef(true);
  const status = useQuery({ queryKey: statusKey, queryFn: robotApi.status, retry: false, refetchInterval: ownsSession ? 600 : 3_000, refetchOnWindowFocus: true });
  const state = status.data;
  const online = !!state?.ok && state.connected !== false && !status.isError;
  const devices = useQuery({ queryKey: ['robot-lab', 'devices'], queryFn: robotApi.devices, enabled: online, retry: false, staleTime: 30_000 });
  const datasets = useQuery({ queryKey: ['robot-lab', 'datasets'], queryFn: robotApi.datasets, enabled: online, retry: false, refetchInterval: 10_000 });
  const session = state?.session;
  const calibration = state?.calibration;
  const recording = state?.recording;
  const assembled = state?.assembled;
  const active = !!session && session.mode !== 'idle';
  const controlling = !!session && ['manual', 'leader', 'handguide'].includes(session.mode);
  const calibrated = session?.mode !== 'setup' && !assembled?.calibration_required && !!state?.calibrations?.some(item => item.robot_id === robotId && item.role === 'follower');
  const leaderCalibrated = !!state?.calibrations?.some(item => item.robot_id === leaderId && item.role === 'leader');
  const selectedMotor = robotJoints.find(item => item.name === setupOrder[motorIndex])!;
  const availableCameras = devices.data?.cameras ?? [];
  const busy = !!pending;
  const canAct = online && !busy;
  const datasetList = datasets.data?.datasets ?? state?.datasets ?? [];
  const missingHardwareTools = Object.entries(state?.dependencies ?? {}).filter(([name, dependency]) => !dependency.installed && ['serial', 'scservo_sdk', 'cv2'].includes(name));
  const unavailableFeatures = state?.capabilities && Object.values(state.capabilities).some(available => !available);

  useEffect(() => {
    mounted.current = true;
    try {
      const saved = JSON.parse(localStorage.getItem('openjensen.robot-lab.preferences') ?? '{}');
      if (typeof saved.port === 'string') setPort(saved.port);
      if (typeof saved.robotId === 'string') setRobotId(saved.robotId);
      if (typeof saved.leaderPort === 'string') setLeaderPort(saved.leaderPort);
      if (typeof saved.leaderId === 'string') setLeaderId(saved.leaderId);
    } catch { /* Preferences are optional. */ }
    return () => { mounted.current = false; if (ownedRef.current) void robotApi.stop(true).catch(() => undefined); };
  }, []);
  useEffect(() => {
    if (!devices.data) return;
    if (!port) setPort(devices.data.ports[0]?.device ?? '');
    if (!availableCameras.some(camera => camera.index === cameraIndex)) setCameraIndex(availableCameras[0]?.index ?? 0);
  }, [devices.data, port, cameraIndex, availableCameras]);
  useEffect(() => { setIsolated(false); setAssembledReady(false); setWiggleReady(false); setWiggleId(''); setProbe(null); setAssigned([]); setMotionReady(false); }, [port]);
  useEffect(() => { setIsolated(false); }, [motorIndex]);
  useEffect(() => {
    if (!preview || !online) return;
    setCameraError(false); setPreviewStamp(Date.now());
    const timer = setInterval(() => setPreviewStamp(Date.now()), 700);
    return () => clearInterval(timer);
  }, [preview, online, cameraIndex]);
  useEffect(() => {
    if (!ownsSession) return;
    let stopped = false, inFlight = false;
    const heartbeat = async () => {
      if (inFlight || stopped) return;
      inFlight = true;
      try {
        const next = await robotApi.heartbeat();
        if (!stopped && mounted.current && next.session?.mode === 'idle') { ownedRef.current = false; setOwnsSession(false); client.setQueryData(statusKey, next); }
      } catch (failure) {
        if (!stopped && mounted.current) { ownedRef.current = false; setOwnsSession(false); setError(`Connection lost: ${readable(failure)} The companion stops the arm when its heartbeat expires.`); void client.invalidateQueries({ queryKey: statusKey }); }
      } finally { inFlight = false; }
    };
    const timer = setInterval(() => void heartbeat(), 500);
    const hide = () => { if (document.visibilityState === 'hidden' && ownedRef.current) { ownedRef.current = false; setOwnsSession(false); void robotApi.stop(true).then(next => { if (mounted.current) client.setQueryData(statusKey, next); }).catch(() => undefined); } };
    document.addEventListener('visibilitychange', hide);
    return () => { stopped = true; clearInterval(timer); document.removeEventListener('visibilitychange', hide); };
  }, [ownsSession, client]);

  function selectIllustratedJoint(joint: RobotJoint) {
    setMotorIndex(setupOrder.indexOf(joint));
    if (identifyMethod === 'hand' && online && ownsSession && !busy && session?.mode === 'setup' && assembled && assembled.mapping[joint] === undefined && ['ready', 'observing', 'review'].includes(assembled.phase)) void perform(`Watching for ${robotJoints.find(item => item.name === joint)!.label.toLowerCase()} hand movement.`, () => robotApi.observeAssembled(joint), undefined, false);
  }
  const mappingValues = Object.values(assembled?.mapping ?? {});
  const allJointsMapped = robotJoints.every(joint => typeof assembled?.mapping[joint.name] === 'number') && new Set(mappingValues).size === 6;
  const selectedCandidate = assembled?.selected_joint === selectedMotor.name && typeof assembled.detected_id === 'number' ? assembled.detected_id : null;
  const detectionFeedback = identifyMethod === 'hand' && online && session?.mode === 'setup' && assembled && ['ready', 'observing', 'review'].includes(assembled.phase) ? <div className="so101-detection-feedback" role="group" aria-label="Arm movement detection">{selectedCandidate !== null ? <><strong>Movement detected at address {selectedCandidate}</strong><p>{assembled.candidate_delta === null ? '' : `${Math.abs(assembled.candidate_delta).toFixed(0)} encoder ticks changed. `}Hold the {selectedMotor.label.toLowerCase()} in its new position and confirm.</p>{selectedCandidate !== null && online && <button className="secondary-button" disabled={!canAct || !ownsSession || assembled.phase !== 'observing'} onClick={() => void perform(`${selectedMotor.label} address confirmed.`, () => robotApi.acceptAssembled(selectedMotor.name, selectedCandidate), () => { const next = setupOrder.find(joint => joint !== selectedMotor.name && assembled.mapping[joint] === undefined); if (next) setMotorIndex(setupOrder.indexOf(next)); })}>Confirm ID {selectedCandidate} is the {selectedMotor.label.toLowerCase()}</button>}</> : assembled.mapping[selectedMotor.name] !== undefined && assembled.selected_joint !== selectedMotor.name ? <p>{selectedMotor.label} is confirmed at address {assembled.mapping[selectedMotor.name]}. Select the next joint.</p> : <p>{assembled.selected_joint === selectedMotor.name && assembled.phase === 'observing' ? 'Move this joint gently, then hold its new position.' : 'Select a joint, move it gently, then hold its new position.'}</p>}</div> : undefined;
  const jointGuide = <So101JointGuide feedback={detectionFeedback} selected={selectedMotor.name} onSelect={selectIllustratedJoint} setupActive={session?.mode === 'setup' && !['ready', 'observing', 'review'].includes(assembled?.phase ?? '')} detecting={identifyMethod === 'hand' && online && session?.mode === 'setup' && !!assembled && ['ready', 'observing', 'review'].includes(assembled.phase)} calibrationActive={session?.mode === 'calibration' && !session.torque_enabled} measured={online && session?.mode === 'calibration' && (calibration?.phase === 'center' || calibration?.phase === 'ranges') && Number.isFinite(session.positions[selectedMotor.name]) ? { value: session.positions[selectedMotor.name]!, min: calibration.mins[selectedMotor.name], max: calibration.maxes[selectedMotor.name] } : undefined} />;
  function remember() { try { localStorage.setItem('openjensen.robot-lab.preferences', JSON.stringify({ port, robotId, leaderPort, leaderId })); } catch { /* Preferences are optional. */ } }
  async function perform<T>(label: string, action: () => Promise<T>, done?: (result: T) => void, announce = true) {
    const token = ++epoch.current;
    setPending(label); setError(null); setNotice(null);
    try {
      const result = await action();
      if (!mounted.current || token !== epoch.current) return;
      if (isStatus(result)) { await client.cancelQueries({ queryKey: statusKey }); client.setQueryData(statusKey, result); }
      done?.(result);
      setNotice(announce ? label : null);
      void client.invalidateQueries({ queryKey: ['robot-lab', 'datasets'] });
      void client.invalidateQueries({ queryKey: statusKey });
    } catch (failure) { if (mounted.current && token === epoch.current) setError(readable(failure)); }
    finally { if (mounted.current && token === epoch.current) setPending(null); }
  }
  function ownSession() { ownedRef.current = true; setOwnsSession(true); remember(); }
  function stop() {
    ownedRef.current = false; setOwnsSession(false); setMotionReady(false);
    void perform('Stop requested.', () => robotApi.stop());
  }
  function refresh() { setError(null); setNotice(null); void status.refetch(); if (online) { void devices.refetch(); void datasets.refetch(); } }
  function updatePort(value: string) { setPort(value); setProbe(null); }
  const followerPort = <label>Robot USB port<select aria-label="Robot USB port" value={port} disabled={active || busy || !online} onChange={event => updatePort(event.target.value)}><option value="">Select a serial device</option>{devices.data?.ports.map(item => <option key={item.device} value={item.device}>{item.device} · {item.description}</option>)}</select></label>;
  const identity = <div className="robot-lab-fields">{followerPort}<label>Robot name<input aria-label="Robot name" value={robotId} disabled={active || busy} onChange={event => { setRobotId(event.target.value); setMotionReady(false); }} maxLength={64} pattern="[A-Za-z0-9_-]+" /></label></div>;

  return <div className="robot-lab">
    <div className="robot-lab-statusbar"><div className="robot-lab-connection"><span className={`robot-lab-dot${online ? ' online' : ''}`} /><div><strong>{status.isPending ? 'Connecting…' : online ? `${state?.label ?? 'Mac companion'} connected` : 'Mac companion unavailable'}</strong><small>{online ? active ? `${session?.mode === 'setup' ? 'Assembled motor setup' : session?.mode === 'calibration' ? 'Calibration' : session?.mode === 'handguide' ? 'Hand guiding' : session?.mode === 'leader' ? 'Leader teleoperation' : 'Manual control'} · ${session?.robot_id ?? robotId} · ${session?.mode === 'setup' ? 'Motor setup active' : session?.torque_enabled ? 'Torque command: on' : 'Torque command: off'}` : 'Idle' : state?.message ?? status.error?.message ?? 'Check that the local companion is running.'}</small></div></div><div className="robot-lab-status-actions"><button className="text-button" onClick={refresh} disabled={busy || status.isFetching}>{status.isFetching ? 'Checking…' : 'Check connection'}</button><button className="robot-lab-stop" onClick={stop} disabled={!state?.ok && !active} aria-label="Stop arm and disable torque"><span />{pending === 'Stop requested.' ? 'Stopping…' : 'Stop arm'}</button></div></div>
    <nav className="robot-lab-steps" aria-label="Robot setup steps">{steps.map((item, index) => <button key={item.title} className={`robot-lab-step${step === index ? ' selected' : ''}`} onClick={() => { setStep(index); setNotice(null); setError(null); }} aria-current={step === index ? 'step' : undefined} disabled={busy}>{item.title}</button>)}</nav>
    {error && <p className="robot-lab-event robot-lab-error" role="alert">{error}</p>}
    {notice && <p className="robot-lab-event" role="status">{notice}</p>}
    {session?.error && <p className="robot-lab-event robot-lab-error" role="alert">{session.error}</p>}
    {recording?.error && <p className="robot-lab-event robot-lab-error" role="alert">{recording.error}</p>}
    <div className="robot-lab-grid"><section className="robot-lab-main" aria-label={steps[step].title}>
      {step === 0 && <><Heading title="Devices" /><div className="robot-lab-form">
        {!online && <Notice><strong>Companion offline</strong>Start the Mac companion, then check the connection.</Notice>}
        {devices.isPending && online && <p className="robot-lab-subtle" role="status">Looking for serial devices and cameras…</p>}{devices.error && <Notice kind="warning"><strong>Device discovery needs attention</strong>{devices.error.message}<button className="text-button" onClick={() => void devices.refetch()}>Retry discovery</button></Notice>}
        {identity}

        {online && unavailableFeatures && missingHardwareTools.length > 0 && <Notice kind="warning"><strong>Missing dependencies</strong>{missingHardwareTools.map(([name]) => name).join(', ')}</Notice>}
        {online && devices.data && <div className="robot-lab-device-summary"><span>{devices.data.ports.length} serial devices</span><span>{availableCameras.length} cameras</span>{devices.data.camera_error && <span role="alert">{devices.data.camera_error}</span>}</div>}
        <div className="robot-lab-actions"><button className="primary-button" disabled={!canAct || !port || active || !state?.capabilities?.probe} onClick={() => { remember(); void perform('Motor scan completed.', () => robotApi.probe(port), result => setProbe(result)); }}>{pending === 'Motor scan completed.' ? 'Checking motors…' : 'Check motor connection'}</button><button className="secondary-button" disabled={!online || busy} onClick={() => setStep(probe?.ready ? 2 : 1)}>{probe?.ready ? 'Continue to calibration' : 'Set up motor IDs'}<Icon name="arrow" size={15} /></button></div>
        {probe && <><Notice kind={probe.ready ? 'success' : 'warning'}><strong>{probe.ready ? 'All six motor IDs responded.' : 'Motor setup required'}</strong>{probe.ready ? 'Calibrate before enabling motion.' : `Responding IDs: ${probe.motors.map(motor => motor.id).join(', ') || 'none'}. Factory IDs may be duplicated. Continue to motor setup.`}</Notice>{probe.warning && <p className="robot-lab-subtle">{probe.warning}</p>}</>}
      </div></>}
      {step === 1 && <><Heading title="Motor IDs" /><div className="robot-lab-form">
        {jointGuide}
        {followerPort}
        {session?.mode !== 'setup' && <fieldset className="robot-lab-identify-method"><legend>Identification method</legend><button type="button" className={identifyMethod === 'hand' ? 'primary-button' : 'secondary-button'} aria-pressed={identifyMethod === 'hand'} onClick={() => { setIdentifyMethod('hand'); setWiggleReady(false); }}>Move by hand</button><button type="button" className={identifyMethod === 'wiggle' ? 'primary-button' : 'secondary-button'} aria-pressed={identifyMethod === 'wiggle'} onClick={() => setIdentifyMethod('wiggle')}>Wiggle a motor</button></fieldset>}
        <section className="robot-lab-assembled" aria-label="Experimental assembled setup"><div className="robot-lab-assembled-heading"><h3>Assembled setup</h3><span className="robot-lab-experimental">Experimental</span></div>
          {(!assembled || ['idle', 'stopped', 'error', 'complete'].includes(assembled.phase)) && session?.mode !== 'setup' && <div className="robot-lab-form">
            <Notice kind="warning">Support the arm. Keep power, USB and motor cables connected. Experimental setup reboots motors and writes temporary IDs; torque can release.</Notice>
            {probe?.ready && <Notice kind="success">All six expected IDs already respond. You can go straight to calibration.</Notice>}
            <label className="robot-lab-check"><input type="checkbox" checked={assembledReady} disabled={busy || active} onChange={event => setAssembledReady(event.target.checked)} /><span>Six STS3215 motors; arm supported. Allow motor reboots and persistent ID changes.</span></label>
            <div className="robot-lab-actions"><button className="primary-button" disabled={!canAct || active || !port || !assembledReady || !state?.capabilities?.assembled_setup} onClick={() => { setAssembledReady(false); void perform('Experimental assembled setup started.', () => robotApi.startAssembled(port), ownSession); }}>Start assembled setup</button><button className="secondary-button" disabled={!canAct || active || !port} onClick={() => void perform('Motor connection checked.', () => robotApi.probe(port), result => setProbe(result))}>Verify current IDs</button></div>
          </div>}
          {assembled && assembled.phase !== 'idle' && <div className="robot-lab-form">
            <div className={`robot-lab-assembled-state phase-${assembled.phase}`} role="status"><span className="robot-lab-mini-label">{assembled.phase.toUpperCase()}</span><p>{assembled.message}</p>{assembled.ids.length > 0 && <small>{assembled.ids.length} distinct {assembled.ids.length === 1 ? 'address' : 'addresses'}: {assembled.ids.join(', ')}</small>}</div>
            {assembled.error && <p className="robot-lab-event robot-lab-error" role="alert">{assembled.error}</p>}
            {session?.mode === 'setup' && ['ready', 'observing', 'review'].includes(assembled.phase) && <>
              <fieldset className="robot-lab-identify-method" disabled={busy || assembled.motion_active}><legend>Identification method</legend><button type="button" className={identifyMethod === 'hand' ? 'primary-button' : 'secondary-button'} aria-pressed={identifyMethod === 'hand'} onClick={() => { setIdentifyMethod('hand'); setWiggleReady(false); }}>Move by hand</button><button type="button" className={identifyMethod === 'wiggle' ? 'primary-button' : 'secondary-button'} aria-pressed={identifyMethod === 'wiggle'} onClick={() => setIdentifyMethod('wiggle')}>Wiggle a motor</button></fieldset>
              {identifyMethod === 'hand' ? <div className="robot-lab-actions"><button className="primary-button" disabled={!canAct || !ownsSession || assembled.mapping[selectedMotor.name] !== undefined} onClick={() => selectIllustratedJoint(selectedMotor.name)}>Restart movement detection</button></div> : <div className="robot-lab-form">
                <label>Motor address<select aria-label="Motor address to identify" value={wiggleId || String(assembled.ids.find(id => !mappingValues.includes(id)) ?? '')} onChange={event => { setWiggleId(event.target.value); setWiggleReady(false); }}>{assembled.ids.filter(id => !mappingValues.includes(id)).map(id => <option value={id} key={id}>Address {id}</option>)}</select></label>
                <label className="robot-lab-check"><input type="checkbox" checked={wiggleReady} onChange={event => setWiggleReady(event.target.checked)} /><span>Arm supported; every joint clear of its stops. Allow a brief ±2° motion at low torque.</span></label>
                <button className="primary-button" disabled={!canAct || !ownsSession || !wiggleReady || allJointsMapped} onClick={() => { const id = Number(wiggleId || assembled.ids.find(id => !mappingValues.includes(id))); setWiggleReady(false); void perform(`Address ${id} identification requested.`, () => robotApi.wiggleAssembled(id), undefined, false); }}>Wiggle selected motor</button>
                {assembled.wiggle_id != null && <div className="robot-lab-form"><p>Address {assembled.wiggle_id}: select the joint that moved on your arm.</p><button className="secondary-button" disabled={!canAct || !ownsSession || assembled.mapping[selectedMotor.name] !== undefined} onClick={() => void perform('Joint address confirmed.', () => robotApi.acceptWiggle(selectedMotor.name, assembled.wiggle_id!), () => { setWiggleId(''); const next = setupOrder.find(joint => joint !== selectedMotor.name && assembled.mapping[joint] === undefined); if (next) setMotorIndex(setupOrder.indexOf(next)); })}>Assign address {assembled.wiggle_id} to {selectedMotor.label.toLowerCase()}</button></div>}
              </div>}

              <div className="robot-lab-table-wrap"><table className="robot-lab-table"><thead><tr><th>Joint to identify</th><th>Detected address</th><th>Final ID</th></tr></thead><tbody>{robotJoints.map(joint => <tr key={joint.name}><td><button type="button" className="text-button" onClick={() => selectIllustratedJoint(joint.name)}>{joint.label}</button></td><td>{assembled.mapping[joint.name] ?? 'Not confirmed'}</td><td>{joint.id}</td></tr>)}</tbody></table></div>
              {allJointsMapped && <><Notice kind="warning">Review all six matches. Saving writes IDs 1–6 and requires recalibration.</Notice><button className="primary-button" disabled={!canAct || !ownsSession || assembled.phase !== 'review'} onClick={() => void perform('Motor ID save requested.', robotApi.finishAssembled, result => { if (result.session?.mode === 'idle') { ownedRef.current = false; setOwnsSession(false); } setProbe(null); })}>Save motor IDs 1–6</button></>}
            </>}
            {session?.mode === 'setup' && <button className="secondary-button" disabled={busy} onClick={() => void perform('Assembled setup stopped.', robotApi.cancelAssembled, () => { ownedRef.current = false; setOwnsSession(false); })}>Stop assembled setup</button>}
            {assembled.phase === 'complete' && <Notice kind="success"><strong>IDs saved</strong>Calibration required.<button className="text-button" disabled={busy} onClick={() => setStep(2)}>Continue to calibration →</button></Notice>}
            {assembled.log.length > 0 && <details className="robot-lab-setup-log"><summary>Setup details</summary><ol>{assembled.log.map((entry, index) => <li key={index}>{entry}</li>)}</ol></details>}
          </div>}
        </section>
        <details className="robot-lab-advanced-ids"><summary>Advanced: configure one physically isolated motor</summary><div className="robot-lab-form">
<div className="robot-lab-motor-list">{setupOrder.map((name, index) => { const joint = robotJoints.find(item => item.name === name)!; return <button key={name} className={`robot-lab-motor${motorIndex === index ? ' selected' : ''}${assigned.includes(name) ? ' assigned' : ''}`} onClick={() => setMotorIndex(index)} disabled={busy || active} aria-pressed={motorIndex === index}><b>{assigned.includes(name) ? <Icon name="check" size={12} /> : joint.id}</b><span><strong>{joint.label}</strong><small>{assigned.includes(name) ? 'Assigned this session' : `Motor ID ${joint.id}`}</small></span></button>; })}</div>
        <div className="robot-lab-motor-guide"><span className="robot-lab-motor-number">{selectedMotor.id}</span><div><h3>{selectedMotor.label}</h3><p>Power off the arm. Connect only this motor directly to the controller, with the rest of the chain disconnected. Restore motor power and keep USB connected.</p></div></div>
        <label className="robot-lab-check"><input type="checkbox" checked={isolated} disabled={busy || active} onChange={event => setIsolated(event.target.checked)} /><span><strong>Only the {selectedMotor.label.toLowerCase()} motor is connected.</strong>I have physically disconnected every other motor from the controller.</span></label>
        <div className="robot-lab-actions"><button className="primary-button" disabled={!canAct || !port || active || !isolated || !state?.capabilities?.motor_setup} onClick={() => void perform(`${selectedMotor.label} assigned ID ${selectedMotor.id}.`, () => robotApi.assignMotor(port, selectedMotor.name), result => { setAssigned(previous => [...new Set([...previous, result.assigned.joint])]); setIsolated(false); if (motorIndex < 5) setMotorIndex(motorIndex + 1); })}>{busy ? 'Assigning motor ID…' : `Assign ID ${selectedMotor.id}`}</button><button className="secondary-button" disabled={busy || active} onClick={() => setStep(2)}>IDs already set? Calibrate<Icon name="arrow" size={14} /></button></div>
        <Notice><strong>Reconnect the full arm when all six IDs are set.</strong>Switch motor power off before changing cables. A scan cannot distinguish several motors sharing the same factory ID; this step needs one physical motor at a time.</Notice>
        </div></details>
      </div></>}
      {step === 2 && <><Heading title="Calibration" /><div className="robot-lab-form">
        {jointGuide}
        {identity}<label>Arm role<select value={role} aria-label="Calibration arm role" onChange={event => setRole(event.target.value as RobotRole)} disabled={active || busy}><option value="follower">Follower / single arm</option><option value="leader">Leader arm</option></select></label>
        <div className="robot-lab-calibration-steps"><span className={calibration?.phase === 'idle' || calibration?.phase === 'saved' ? 'current' : ''}>1. Prepare</span><Icon name="arrow" size={12} /><span className={calibration?.phase === 'center' ? 'current' : ''}>2. Middle position</span><Icon name="arrow" size={12} /><span className={calibration?.phase === 'ranges' ? 'current' : ''}>3. Joint ranges</span></div>
        {session?.mode !== 'calibration' && <><Notice>Support the arm. Calibration releases torque; all six motors must be connected and assigned IDs.</Notice><button className="primary-button" disabled={!canAct || active || !port || !/^[A-Za-z0-9_-]+$/.test(robotId) || !state?.capabilities?.calibration} onClick={() => void perform('Calibration started. Position each joint near its middle.', () => robotApi.startCalibration(port, robotId, role), ownSession)}>Begin calibration</button></>}
        {session?.mode === 'calibration' && calibration?.phase === 'center' && <><Notice>Position each joint at mid-travel, with the gripper half open. Support the arm and record the center.</Notice><RangeTable positions={session.positions} calibration /><div className="robot-lab-actions"><button className="primary-button" disabled={!canAct || !ownsSession} onClick={() => void perform('Center recorded. Move each joint through its comfortable range.', robotApi.centerCalibration)}>Record middle position</button><button className="secondary-button" disabled={!canAct} onClick={() => void perform('Calibration cancelled.', robotApi.cancelCalibration, () => { ownedRef.current = false; setOwnsSession(false); })}>Cancel calibration</button></div></>}
        {session?.mode === 'calibration' && calibration?.phase === 'ranges' && <><Notice>Move each joint through its usable range without forcing it. Wrist roll uses its full encoder range.</Notice><RangeTable positions={session.positions} mins={calibration.mins} maxes={calibration.maxes} calibration /><div className="robot-lab-actions"><button className="primary-button" disabled={!canAct || !ownsSession} onClick={() => void perform('Calibration saved. The arm is ready for controlled motion.', robotApi.finishCalibration, () => { ownedRef.current = false; setOwnsSession(false); })}>Save calibration</button><button className="secondary-button" disabled={!canAct} onClick={() => void perform('Calibration cancelled.', robotApi.cancelCalibration, () => { ownedRef.current = false; setOwnsSession(false); })}>Cancel calibration</button></div></>}
        {state?.calibrations?.length ? <Notice kind="success"><strong>Saved calibrations</strong>{state.calibrations.map(item => <div key={`${item.role}:${item.robot_id}`}>{item.robot_id} · {item.role}</div>)}<button className="text-button" onClick={() => setStep(3)} disabled={active || busy}>Continue to teleoperation →</button></Notice> : null}
      </div></>}
      {step === 3 && <><Heading title="Teleoperation" /><div className="robot-lab-form">
        {identity}<div className="robot-lab-mode-grid" role="group" aria-label="Control mode">{([{ value: 'manual', title: 'Joint control', detail: 'Bounded position steps', icon: 'sliders' }, { value: 'handguide', title: 'Hand guiding', detail: 'Torque off', icon: 'robot' }, { value: 'leader', title: 'Leader → follower', detail: 'Two calibrated arms', icon: 'layers' }] as const).map(item => <button key={item.value} className={`robot-lab-mode${mode === item.value ? ' selected' : ''}`} onClick={() => { setMode(item.value); setMotionReady(false); }} disabled={active || busy} aria-pressed={mode === item.value}><Icon name={item.icon} size={19} /><strong>{item.title}</strong><small>{item.detail}</small></button>)}</div>
        {mode === 'leader' && <div className="robot-lab-fields"><label>Leader USB port<select value={leaderPort} disabled={active || busy} aria-label="Leader USB port" onChange={event => setLeaderPort(event.target.value)}><option value="">Select the second arm</option>{devices.data?.ports.filter(item => item.device !== port).map(item => <option key={item.device} value={item.device}>{item.device}</option>)}</select></label><label>Leader name<input value={leaderId} disabled={active || busy} aria-label="Leader name" onChange={event => setLeaderId(event.target.value)} /><small>{leaderCalibrated ? 'Leader calibration found.' : 'Calibrate this arm with the leader role first.'}</small></label></div>}
        {!active && <>{!calibrated && <Notice kind="warning"><strong>Calibration required</strong>Follower: {robotId}.<button className="text-button" onClick={() => setStep(2)}>Open calibration →</button></Notice>}<label className="robot-lab-check"><input type="checkbox" checked={motionReady} disabled={busy} onChange={event => setMotionReady(event.target.checked)} /><span><strong>{mode === 'handguide' ? 'Arm supported.' : 'Arm secured; workspace clear.'}</strong>{mode === 'handguide' ? 'Torque stays off.' : mode === 'leader' ? 'Enable torque; the follower starts tracking the leader immediately.' : 'Enable torque and hold the current pose.'}</span></label><button className="primary-button" disabled={!canAct || !port || !calibrated || !motionReady || !state?.capabilities?.control || (mode === 'leader' && (!leaderPort || !leaderCalibrated || leaderPort === port))} onClick={() => void perform(mode === 'handguide' ? 'Hand guiding started. Torque remains off.' : 'Controls enabled. Use Stop arm to release the motors.', () => robotApi.startControl({ port, robot_id: robotId, mode, ...(mode === 'leader' ? { leader_port: leaderPort, leader_id: leaderId } : {}) }), ownSession)}>{mode === 'handguide' ? 'Start hand guiding' : mode === 'leader' ? 'Start leader teleoperation' : 'Enable on-screen controls'}</button></>}
        {controlling && <><Notice kind="success"><strong>{session?.mode === 'manual' ? 'On-screen controls are active.' : session?.mode === 'leader' ? 'Leader teleoperation is active.' : 'Hand guiding is active. Torque is off.'}</strong>{ownsSession ? 'Tab hidden or disconnected → torque off.' : 'Session owned by another client. Stop it to take control.'}</Notice>{session?.mode === 'manual' ? <><div className="robot-lab-control-heading"><h3>Joint controls</h3><label>Step<select value={jogStep} aria-label="Joint step size" onChange={event => setJogStep(Number(event.target.value))}><option value={.5}>0.5° / %</option><option value={1}>1° / %</option><option value={2}>2° / %</option></select></label></div><div className="robot-lab-joints">{robotJoints.map(joint => <div key={joint.name} className="robot-lab-joint"><strong>{joint.label}</strong><output aria-label={`${joint.label} position`}>{number(session.positions[joint.name], 1)}{joint.unit === 'percent' ? '%' : '°'}</output><button aria-label={`Decrease ${joint.label}`} disabled={!canAct || !ownsSession || recording?.episode_active} onClick={() => void perform(`${joint.label} target requested: −${jogStep}${joint.unit === 'percent' ? '%' : '°'}.`, () => robotApi.jog(joint.name, -jogStep))}>−</button><button aria-label={`Increase ${joint.label}`} disabled={!canAct || !ownsSession || recording?.episode_active} onClick={() => void perform(`${joint.label} target requested: +${jogStep}${joint.unit === 'percent' ? '%' : '°'}.`, () => robotApi.jog(joint.name, jogStep))}>+</button></div>)}</div></> : <RangeTable positions={session?.positions} />}<div className="robot-lab-actions"><button className="robot-lab-stop" onClick={stop}><span />Stop & release motors</button><button className="secondary-button" disabled={busy || !ownsSession} onClick={() => setStep(4)}>Create a dataset<Icon name="arrow" size={14} /></button></div></>}
      </div></>}
      {step === 4 && <><Heading title="Recording" /><div className="robot-lab-form">
        {!recording?.active ? <>{!controlling && <Notice><strong>No control session</strong>Start teleoperation before recording.<button className="text-button" onClick={() => setStep(3)}>Open teleoperation →</button></Notice>}<div className="robot-lab-fields"><label>Dataset name<input value={datasetId} aria-label="Dataset name" placeholder="pick-up-block" onChange={event => setDatasetId(event.target.value.toLowerCase().replace(/[^a-z0-9_-]/g, '-'))} maxLength={64} disabled={busy} />{datasetList.some(dataset => dataset.id === datasetId) && <small role="alert">Name already used.</small>}</label><label>Frames per second<select value={fps} aria-label="Recording frame rate" onChange={event => setFps(Number(event.target.value))} disabled={busy}>{[5, 10, 15, 20].map(value => <option key={value} value={value}>{value} fps</option>)}</select></label></div><label>Task instruction<textarea value={task} aria-label="Task instruction" placeholder="Pick up the blue block and place it in the tray." onChange={event => setTask(event.target.value)} disabled={busy} maxLength={500} /></label><label className="robot-lab-check"><input type="checkbox" checked={recordCamera} disabled={busy || !availableCameras.length} onChange={event => setRecordCamera(event.target.checked)} /><span><strong>Record camera</strong>{availableCameras.length ? '' : 'No camera available; state and actions only.'}</span></label><button className="primary-button" disabled={!canAct || !controlling || !ownsSession || !/^[a-z0-9][a-z0-9_-]{0,63}$/.test(datasetId) || !task.trim() || datasetList.some(dataset => dataset.id === datasetId) || !state?.capabilities?.recording} onClick={() => void perform('Dataset created. Start an episode when you are ready.', () => robotApi.startRecording({ dataset_id: datasetId, task: task.trim(), fps, camera_index: recordCamera && availableCameras.length ? cameraIndex : null }))}>{busy ? 'Creating dataset…' : 'Create recording dataset'}</button></> : <><div className={`robot-lab-recording-status${recording.episode_active ? ' recording' : ''}`}><span className="robot-lab-recording-light" /><div><strong>{recording.dataset_id}</strong><small>{recording.episode_active ? 'Recording an episode' : recording.frames > 0 ? 'Episode has pending frames' : 'Ready for the next episode'} · {recording.episodes} saved</small></div><output>{recording.frames} frames</output></div>{session?.mode === 'handguide' && <Notice>Action source: observed joint positions (hand guiding).</Notice>}{session?.mode === 'manual' && <Notice>Action source: commanded joint targets.</Notice>}<div className="robot-lab-actions"><button className="primary-button" disabled={!canAct || !controlling || !ownsSession || recording.episode_active || recording.frames > 0} onClick={() => void perform('Episode recording started.', robotApi.startEpisode)}><span aria-hidden="true">●</span>Start episode</button><button className="secondary-button" disabled={!canAct || !recording.frames} onClick={() => void perform('Episode saved.', robotApi.saveEpisode)}>Save episode</button><button className="secondary-button" disabled={!canAct || (!recording.episode_active && !recording.frames)} onClick={() => void perform('Episode discarded.', robotApi.discardEpisode)}>Discard episode</button></div>{session?.mode === 'manual' && <div className="robot-lab-joints">{robotJoints.map(joint => <div key={joint.name} className="robot-lab-joint"><strong>{joint.label}</strong><output>{number(session.positions[joint.name], 1)}{joint.unit === 'percent' ? '%' : '°'}</output><button aria-label={`Record decrease ${joint.label}`} disabled={!canAct || !ownsSession} onClick={() => void perform('Joint target updated.', () => robotApi.jog(joint.name, -1))}>−</button><button aria-label={`Record increase ${joint.label}`} disabled={!canAct || !ownsSession} onClick={() => void perform('Joint target updated.', () => robotApi.jog(joint.name, 1))}>+</button></div>)}</div>}<div className="robot-lab-actions"><button className="secondary-button" disabled={!canAct || recording.episode_active || recording.frames > 0 || recording.episodes === 0} onClick={() => void perform('Dataset finalized.', robotApi.finishRecording)}>Finish dataset</button>{recording.episodes === 0 && <button className="text-button" disabled={!canAct} onClick={() => void perform('Unsaved recording closed; files retained on this Mac.', robotApi.abandonRecording)}>Close unsaved recording</button>}</div>{recording.episodes === 0 && <p className="robot-lab-subtle">Closing retains unsaved files. Create a new dataset to change the frame rate.</p>}</>}
        <hr className="robot-lab-divider" /><div><h3>Datasets on this Mac</h3></div>{datasets.error && <p className="robot-lab-subtle" role="alert">{datasets.error.message}</p>}{!datasetList.length ? <div className="robot-lab-empty"><Icon name="database" size={26} />No datasets</div> : <div className="robot-lab-datasets">{datasetList.map(dataset => <DatasetRow key={dataset.id} dataset={dataset} canImport={canAct && !!projectId} pending={pending === `Imported ${dataset.id} into this project.`} onImport={() => { const owner = projectId; void perform(`Imported ${dataset.id} into this project.`, () => importRobotDataset(owner, dataset.id), () => { setImported({ dataset: dataset.id, projectId: owner }); void client.invalidateQueries({ queryKey: ['datasets', owner] }); void client.invalidateQueries({ queryKey: ['jobs', owner] }); }); }} />)}</div>}{imported && imported.projectId === projectId && <Notice kind="success"><div className="robot-lab-imported"><span>{imported.dataset} is in this project’s dataset library.</span><button className="text-button" onClick={() => navigateStage(0)}>View datasets →</button></div></Notice>}
      </div></>}
      {busy && <p className="robot-lab-subtle" role="status" style={{ marginTop: 16 }}>{pending === 'Stop requested.' ? 'Stopping the arm… Keep supporting it while torque is released and motor cleanup finishes.' : 'Working… Keep this page open.'}</p>}
    </section><aside className="robot-lab-side">
      <section className="robot-lab-card" aria-label="Camera preview"><div className="robot-lab-card-header"><h3>Camera preview</h3></div><div className="robot-lab-camera">{preview && online && !cameraError ? <><img alt="Live view from the selected robot camera" src={robotApi.cameraUrl(cameraIndex, previewStamp)} onError={() => setCameraError(true)} /><span className="robot-lab-camera-live">LIVE</span></> : <div className="robot-lab-camera-empty"><CameraIcon size={27} /><span>{cameraError ? 'Camera could not open. Check macOS permissions or close another app using it.' : 'Preview stopped'}</span></div>}</div><div className="robot-lab-camera-controls"><label>Camera<select value={cameraIndex} aria-label="Preview camera" onChange={event => { setCameraIndex(Number(event.target.value)); setCameraError(false); }} disabled={!online || !availableCameras.length || recording?.active}>{!availableCameras.length && <option value={0}>No camera available</option>}{availableCameras.map(camera => <option key={camera.index} value={camera.index}>{camera.name} · {camera.width}×{camera.height}</option>)}</select></label><button className="secondary-button" disabled={!online || !availableCameras.length || !state?.capabilities?.camera} onClick={() => { if (cameraError) { setCameraError(false); setPreviewStamp(Date.now()); setPreview(true); } else setPreview(!preview); }}>{cameraError ? 'Retry camera' : preview ? 'Pause preview' : 'Start preview'}</button></div></section>
    </aside></div>
  </div>;
}

function CameraIcon({ size = 20 }: { size?: number }) { return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M4 7h4l2-3h4l2 3h4a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1Z" /><circle cx="12" cy="13" r="4" /></svg>; }
function DatasetRow({ dataset, canImport, pending, onImport }: { dataset: RobotDataset; canImport: boolean; pending: boolean; onImport: () => void }) { return <article className="robot-lab-dataset"><div className="robot-lab-dataset-heading"><div><h3>{dataset.id}</h3><p>{dataset.task}</p></div><button className="secondary-button" disabled={!canImport || !dataset.finalized || pending} onClick={onImport}>{pending ? 'Copying to project…' : dataset.finalized ? 'Add to project' : dataset.abandoned ? 'Closed · files retained' : 'Not finalized'}</button></div><div className="robot-lab-dataset-meta"><span>{dataset.episodes} episodes</span><span>{dataset.frames} frames</span><span>{dataset.fps} fps</span><span>{dataset.finalized ? 'Finalized' : dataset.abandoned ? 'Closed' : 'Not finalized'}</span>{dataset.pending_frames ? <span>{dataset.pending_frames} unsaved frames</span> : null}</div></article>; }
