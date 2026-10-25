'use client';

import { useRef, useState, type ReactNode } from 'react';
import { robotJoints, type JointValues, type RobotDataset, type RobotJoint, type RobotStatus } from '@/lib/robot-api';
import { Notice as SharedNotice } from './ui/notice';
import { describeHardwareError } from '@/lib/hardware-error';

export const steps = [
  { title: 'Connect' },
  { title: 'Motor IDs' },
  { title: 'Calibrate' },
  { title: 'Teleoperate' },
  { title: 'Record' },
];
export const setupOrder: RobotJoint[] = [
  'gripper',
  'wrist_roll',
  'wrist_flex',
  'elbow_flex',
  'shoulder_lift',
  'shoulder_pan',
];
export const statusKey = ['robot-lab', 'status'];
export const number = (value: number | undefined, digits = 0) =>
  value === undefined || !Number.isFinite(value) ? '—' : value.toFixed(digits);
export const readable = (error: unknown) =>
  error instanceof Error ? error.message : 'The operation could not be completed.';
export const isStatus = (value: unknown): value is RobotStatus =>
  !!value && typeof value === 'object' && 'session' in value;

export function Heading({ title }: { title: string }) {
  return (
    <div className="robot-lab-panel-heading">
      <h2>{title}</h2>
    </div>
  );
}
export function Notice({ children, kind = 'info' }: { children: ReactNode; kind?: 'info' | 'warning' | 'success' }) {
  return <SharedNotice tone={kind}>{children}</SharedNotice>;
}
export function HardwareNotice({ message, label }: { message: string; label?: string }) {
  const described = describeHardwareError(message);
  return (
    <SharedNotice tone="danger" live="alert" label={label} title={described.title} details={described.details}>
      {described.next}
    </SharedNotice>
  );
}
export function RangeTable({
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

export function JointControl({
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

export function CameraIcon({ size = 20 }: { size?: number }) {
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
export function DatasetRow({
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
