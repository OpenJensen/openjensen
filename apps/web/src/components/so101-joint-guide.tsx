'use client';

import { useEffect, useId, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import { robotJoints, type RobotJoint } from '@/lib/robot-api';
import { projectSo101, schematicAxis, so101Schematic, type Point3 } from '@/lib/so101-schematic';
import { Icon } from './icon';
import './so101-joint-guide.css';

const movement: Record<RobotJoint, { title: string; axis: string; direction: string; instruction: string }> = {
  shoulder_pan: { title: 'Turn the whole arm', axis: 'Vertical axis', direction: 'Left ↔ right around the base', instruction: 'Hold the shoulder assembly and turn it gently left and right above the fixed base.' },
  shoulder_lift: { title: 'Raise the upper arm', axis: 'Shoulder hinge', direction: 'Upper arm up ↔ down', instruction: 'Support the upper arm. Raise or lower it at the shoulder hinge just above the base; the elbow and wrist move with it.' },
  elbow_flex: { title: 'Bend the elbow', axis: 'Elbow hinge', direction: 'Forearm bend ↔ straighten', instruction: 'Support the forearm and hold the shoulder still. Bend and straighten the hinge between the upper arm and forearm.' },
  wrist_flex: { title: 'Tilt the wrist', axis: 'Wrist hinge', direction: 'Gripper tilt up ↔ down', instruction: 'Hold the forearm steady. Tilt the gripper up and down at the wrist hinge, without twisting it.' },
  wrist_roll: { title: 'Twist the gripper', axis: 'Along the tool', direction: 'Gripper rotate ↔ rotate back', instruction: 'Hold the wrist body. Twist the gripper around the length of the tool, without bending the wrist hinge.' },
  gripper: { title: 'Open and close the gripper', axis: 'Moving jaw hinge', direction: 'Finger open ↔ close', instruction: 'Hold the gripper body. Gently open and close the moving finger; the other finger stays fixed.' },
};
const jointOrder = robotJoints.map(joint => joint.name);
const pointList = (points: Point3[]) => points.map(point => { const projected = projectSo101(point); return `${projected.x},${projected.y}`; }).join(' ');

export function So101JointGuide({ selected, onSelect, calibrationActive = false, detecting = false, setupActive = false, measured, feedback }: {
  selected: RobotJoint;
  onSelect: (joint: RobotJoint) => void;
  calibrationActive?: boolean;
  detecting?: boolean;
  setupActive?: boolean;
  measured?: { value: number; min?: number; max?: number };
  feedback?: ReactNode;
}) {
  const markerId = useId().replaceAll(':', '');
  const [phase, setPhase] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [replay, setReplay] = useState(0);
  const [reduced, setReduced] = useState(true);
  const buttons = useRef<Partial<Record<RobotJoint, HTMLButtonElement | null>>>({});
  const joint = robotJoints.find(item => item.name === selected)!;
  const description = movement[selected];
  const pose = so101Schematic(selected, phase);
  const selectedPoint = projectSo101(pose.joints[selected]);
  const [axisStart, axisEnd] = schematicAxis(pose, selected);
  const armPoints = jointOrder.map(name => pose.joints[name]);
  const toolSegment = selected === 'wrist_roll' || selected === 'gripper';

  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => setReduced(media.matches);
    update(); media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, []);
  useEffect(() => { setPhase(0); setPlaying(true); }, [selected]);
  useEffect(() => {
    if (!playing || reduced) return;
    let frame = 0, previous = 0, elapsed = phase / (Math.PI * 2 / 3.6) * 1000;
    const animate = (now: number) => {
      if (previous && now - previous >= 32) { elapsed += Math.min(now - previous, 80); previous = now; setPhase(elapsed / 1000 * Math.PI * 2 / 3.6); }
      else if (!previous) previous = now;
      frame = requestAnimationFrame(animate);
    };
    frame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frame);
  }, [selected, playing, reduced, replay]);
  function keyboard(event: KeyboardEvent<HTMLButtonElement>, name: RobotJoint) {
    const index = jointOrder.indexOf(name);
    const next = event.key === 'ArrowRight' || event.key === 'ArrowDown' ? (index + 1) % 6 : event.key === 'ArrowLeft' || event.key === 'ArrowUp' ? (index + 5) % 6 : event.key === 'Home' ? 0 : event.key === 'End' ? 5 : -1;
    if (next < 0) return;
    event.preventDefault(); onSelect(jointOrder[next]); buttons.current[jointOrder[next]]?.focus();
  }
  return <section className="so101-guide" aria-label="SO-101 joint illustration" data-selected-joint={selected} data-motion={reduced ? 'reduced' : playing ? 'playing' : 'paused'}>
    <div className="so101-guide-heading"><div><span className="so101-guide-eyebrow">MEET YOUR ASSEMBLED ARM</span><h3>Pick a joint. See its movement.</h3></div><span className="so101-guide-schematic-label">SCHEMATIC</span></div>
    <p className="so101-guide-caption">{detecting ? 'Illustration only — the pose is not live. Encoder detection is shown separately; selection never commands movement.' : 'Illustration only — does not move or read your arm.'}</p>
    <div className="so101-guide-visual">
      <svg viewBox="0 0 520 380" aria-hidden="true"><defs><pattern id={`${markerId}-grid`} width="26" height="26" patternUnits="userSpaceOnUse"><path d="M26 0H0V26" fill="none" stroke="currentColor" strokeWidth=".5" /></pattern><marker id={`${markerId}-arrow`} markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto-start-reverse"><path d="M0 0L6 3L0 6" fill="none" stroke="currentColor" strokeWidth="1" /></marker></defs><rect x="20" y="12" width="480" height="350" fill={`url(#${markerId}-grid)`} className="so101-guide-grid" /><ellipse cx="96" cy="335" rx="69" ry="15" className="so101-guide-shadow" /><path d="M35 318h114v19H35zM49 310h86v9H49z" className="so101-guide-base" /><polyline points={pointList(armPoints)} className="so101-guide-arm-outline" /><polyline points={pointList(armPoints)} className="so101-guide-arm" />
        {robotJoints.map((item, index) => { if (index === 5) return null; const p = projectSo101(pose.joints[item.name]); return <g key={item.name}><circle cx={p.x} cy={p.y} r={item.name === selected ? 21 : 17} className={item.name === selected ? 'so101-guide-motor selected' : 'so101-guide-motor'} /><circle cx={p.x} cy={p.y} r="5" className="so101-guide-hub" /></g>; })}
        <polyline points={pointList(pose.palm)} className={`so101-guide-palm${toolSegment ? ' selected' : ''}`} /><polyline points={pointList(pose.fixedFinger)} className={`so101-guide-finger${toolSegment ? ' selected' : ''}`} /><polyline points={pointList(pose.movingFinger)} className={`so101-guide-finger moving${toolSegment ? ' selected' : ''}`} />
        <line x1={axisStart.x} y1={axisStart.y} x2={axisEnd.x} y2={axisEnd.y} className="so101-guide-axis" markerStart={`url(#${markerId}-arrow)`} markerEnd={`url(#${markerId}-arrow)`} /><circle cx={selectedPoint.x} cy={selectedPoint.y} r="32" className="so101-guide-focus-ring" /><text x="30" y="36" className="so101-guide-view-label">SO-101 · SIMPLIFIED VIEW</text><text x="30" y="360" className="so101-guide-view-label">ILLUSTRATED POSE · NOT TELEMETRY</text>
      </svg>
      <div className="so101-guide-markers" role="group" aria-label="Choose an illustrated joint">{robotJoints.map(item => { const p = projectSo101(pose.joints[item.name]); const offset = item.name === 'gripper' ? { x: 18, y: -24 } : item.name === 'wrist_roll' ? { x: 0, y: -28 } : item.name === 'wrist_flex' ? { x: 0, y: 27 } : { x: -28, y: 0 }; return <button key={item.name} ref={element => { buttons.current[item.name] = element; }} type="button" style={{ left: `${(p.x + offset.x) / 520 * 100}%`, top: `${(p.y + offset.y) / 380 * 100}%` }} className={`so101-guide-marker${selected === item.name ? ' selected' : ''}`} aria-label={`Joint ${item.id}: ${item.label}`} aria-pressed={selected === item.name} onClick={() => onSelect(item.name)} onKeyDown={event => keyboard(event, item.name)}>{item.id}</button>; })}</div>
    </div>
    <div className="so101-guide-controls"><span><i className="so101-guide-axis-key" />{description.axis}</span><div>{reduced ? <button type="button" onClick={() => { setPlaying(false); setPhase(previous => previous + Math.PI / 2); }}><Icon name="arrow" size={13} />Show next pose</button> : <><button type="button" onClick={() => setPlaying(!playing)} aria-label={playing ? 'Pause illustration' : 'Play illustration'}><span aria-hidden="true">{playing ? 'Ⅱ' : '▷'}</span>{playing ? 'Pause' : 'Play'}</button><button type="button" onClick={() => { setPhase(0); setPlaying(true); setReplay(value => value + 1); }}><span aria-hidden="true">↻</span>Replay</button></>}</div></div>
    {reduced && <p className="so101-guide-reduced">Reduced motion is on. Use “Show next pose” to compare still positions.</p>}
    {feedback}
    <div className="so101-guide-joint-options" role="group" aria-label="Joint names">{robotJoints.map(item => <button key={item.name} type="button" aria-pressed={selected === item.name} onClick={() => onSelect(item.name)}><b>{item.id}</b>{item.label}</button>)}</div>
    <div className="so101-guide-selected" aria-live="polite" aria-atomic="true"><span className="so101-guide-selected-number">{joint.id}</span><div><span className="so101-guide-selected-name">{joint.label}</span><h4>{description.title}</h4><p>{description.instruction}</p><strong>{description.direction}</strong></div></div>
    <p className="so101-guide-hand-note">{calibrationActive ? 'During active calibration: keep motor power on and all cables connected. Torque is released; support the arm while moving this joint.' : detecting ? 'During joint identification: keep motor power on and all cables connected. Torque is released. Support the arm, gently move only the highlighted joint, and hold its new position until you confirm.' : setupActive ? 'During assembled setup: keep motor power on and all cables connected. Support the arm; wait for joint identification before moving a joint.' : 'To explore by hand: switch motor power off, support the arm, and move gently. Never force a stiff joint.'}</p>
    {measured && <div className="so101-guide-measured" role="group" aria-label="Measured encoder values"><span>MEASURED FROM ARM · ENCODER TICKS</span><strong>{Number.isFinite(measured.value) ? measured.value.toFixed(0) : '—'}</strong>{measured.min !== undefined && measured.max !== undefined && <small>Min {measured.min.toFixed(0)} · Max {measured.max.toFixed(0)}</small>}<p>Separate sensor readings; the illustration does not follow this value.</p></div>}
  </section>;
}
