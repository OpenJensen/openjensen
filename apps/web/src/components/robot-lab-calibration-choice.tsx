'use client';

import { useId, useState } from 'react';
import { robotApi, type RobotRole } from '@/lib/robot-api';
import type { RobotLabState } from './robot-lab-state';

type Library = Awaited<ReturnType<typeof robotApi.calibrationProfiles>>;

export function CalibrationChoice({
  lab,
  robotId,
  role,
  port,
  library,
  onDone,
}: {
  lab: RobotLabState;
  robotId: string;
  role: RobotRole;
  port: string;
  library: Library;
  onDone: () => void;
}) {
  const group = useId();
  const { profiles, conflict } = library;
  const recommended =
    conflict?.recommended ?? (library.remembered_source === 'saved' && profiles.length ? 'saved' : 'motors');
  const [choice, setChoice] = useState<'saved' | 'motors' | null>(null);
  const source = choice ?? recommended;
  const [chosenKey, setChosenKey] = useState('');
  const [remember, setRemember] = useState(true);
  const key = chosenKey || profiles.find((p) => p.current)?.key || profiles[0]?.key || '';
  const busy = lab.busy || lab.active || !lab.online;
  return (
    <fieldset className="robot-lab-source-choice" disabled={busy}>
      <legend>Choose calibration for {robotId}</legend>
      {conflict && (
        <p className="robot-lab-event robot-lab-warning" role="status">
          The arm and this machine have different settings for{' '}
          {conflict.joints.map((j) => j.replaceAll('_', ' ')).join(', ')}.
          {conflict.motors_available === false ? ' The arm has factory settings.' : ''}
        </p>
      )}
      <label className="robot-lab-check">
        <input
          type="radio"
          name={group}
          checked={source === 'motors'}
          disabled={conflict?.motors_available === false}
          onChange={() => setChoice('motors')}
        />
        <span>
          <strong>On the arm{recommended === 'motors' ? ' · Recommended' : ''}</strong>
          Use the offsets and limits stored in this arm’s motors. The machine keeps a copy.
        </span>
      </label>
      <label className="robot-lab-check">
        <input
          type="radio"
          name={group}
          checked={source === 'saved'}
          disabled={!profiles.length}
          onChange={() => setChoice('saved')}
        />
        <span>
          <strong>On this machine{recommended === 'saved' ? ' · Recommended' : ''}</strong>
          Apply a saved profile to this arm’s motors with torque off. Its previous settings are backed up.
        </span>
      </label>
      {source === 'saved' && (
        <label>
          Machine profile
          <select value={key} onChange={(event) => setChosenKey(event.target.value)}>
            {profiles.map((entry) => (
              <option key={entry.key} value={entry.key}>
                {entry.robot_id} · {entry.current ? 'Current' : entry.source} ·{' '}
                {new Date(entry.saved_at * 1000).toLocaleString()}
              </option>
            ))}
          </select>
        </label>
      )}
      <label className="robot-lab-check">
        <input type="checkbox" checked={remember} onChange={(event) => setRemember(event.target.checked)} />
        Remember this choice for this arm
      </label>
      <button
        type="button"
        className="primary-button"
        disabled={
          !port || (source === 'saved' ? !profiles.some((p) => p.key === key) : conflict?.motors_available === false)
        }
        onClick={() =>
          void lab.perform(
            'Calibration verified. Ready to connect.',
            () =>
              robotApi.reuseCalibration({
                robot_id: robotId,
                role,
                port,
                source,
                remember,
                same_arm: true,
                connection_key: library.connection_key ?? undefined,
                ...(source === 'saved' ? { profile_key: key, apply_to_arm: true } : {}),
              }),
            () => {
              lab.setProbe(null);
              onDone();
            },
          )
        }
      >
        {source === 'saved' ? 'Apply machine profile to arm' : 'Use arm calibration'}
      </button>
    </fieldset>
  );
}
