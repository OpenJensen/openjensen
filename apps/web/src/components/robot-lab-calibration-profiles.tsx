'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { robotApi } from '@/lib/robot-api';
import type { RobotLabState } from './robot-lab-state';

export function RobotLabCalibrationProfiles({ lab }: { lab: RobotLabState }) {
  const { active, busy, online, perform, port, portConnected, robotId, role, setProbe, state } = lab;
  const [chosen, setChosen] = useState<string | null>(null);
  const [sameArm, setSameArm] = useState(false);
  const supported = !!state?.capabilities?.calibration_profiles;
  const profiles = useQuery({
    queryKey: ['robot-lab', 'calibration-profiles', robotId, role, port],
    queryFn: () => robotApi.calibrationProfiles(robotId, role, port),
    enabled: online && supported && !!robotId && !active,
    refetchOnMount: 'always',
  });
  const entries = profiles.data?.profiles ?? [];
  const selected = chosen ?? entries.find((entry) => entry.current)?.key ?? entries[0]?.key ?? '';
  const unavailable = active || busy || !online || !supported || !portConnected || !sameArm;
  if (!supported) return null;

  function reuse(source: 'saved' | 'motors') {
    void perform(
      'Calibration saved locally and verified against all six motors. Ready to connect; no motor settings changed.',
      () =>
        robotApi.reuseCalibration({
          robot_id: robotId,
          role,
          port,
          source,
          same_arm: sameArm,
          ...(source === 'saved' ? { profile_key: selected } : {}),
        }),
      () => {
        setProbe(null);
        setSameArm(false);
        void profiles.refetch();
      },
    );
  }

  return (
    <section className="robot-lab-transfer" aria-label="Reuse saved calibration">
      <h3>Reuse saved calibration</h3>
      <p className="robot-lab-subtle">
        Calibrations and their previous versions stay on the robot computer, outside the application source. They
        survive updates and are never uploaded to GitHub.
      </p>
      {profiles.data?.needs_review && (
        <p className="robot-lab-event robot-lab-warning" role="status">
          Calibration needs review. If this is the same calibrated arm, reuse a matching file or read its motor
          settings. Manual calibration is needed when the arm mechanics or joint-to-motor IDs have changed.
        </p>
      )}
      <label>
        Local calibration profile
        <select
          value={selected}
          disabled={active || busy || profiles.isPending}
          onChange={(event) => {
            setChosen(event.target.value);
            setSameArm(false);
          }}
        >
          {!entries.length && <option value="">No saved profiles yet</option>}
          {entries.map((entry) => (
            <option key={entry.key} value={entry.key}>
              {entry.robot_id} · {entry.current ? 'Current' : entry.source} ·{' '}
              {new Date(entry.saved_at * 1000).toLocaleString()}
            </option>
          ))}
        </select>
      </label>
      {profiles.error && <p role="alert">Could not load local profiles: {profiles.error.message}</p>}
      <label className="robot-lab-check">
        <input
          type="checkbox"
          checked={sameArm}
          disabled={active || busy}
          onChange={(event) => setSameArm(event.target.checked)}
        />
        This is the same calibrated arm. Its mechanics and joint-to-motor IDs are unchanged.
      </label>
      <div className="robot-lab-actions">
        <button
          className="primary-button"
          disabled={unavailable || !entries.some((entry) => entry.key === selected)}
          onClick={() => reuse('saved')}
        >
          Use saved profile
        </button>
        <button className="secondary-button" disabled={unavailable} onClick={() => reuse('motors')}>
          Use calibration on motors
        </button>
        <button
          className="text-button"
          disabled={active || busy || profiles.isFetching}
          onClick={() => void profiles.refetch()}
        >
          Refresh profiles
        </button>
      </div>
      <p className="robot-lab-subtle">
        {active ? 'Stop the current session first. ' : !portConnected ? 'Connect this arm first. ' : ''}
        Both options check all six motors with torque off and keep a backup of the current file. A saved file must match
        the motor offsets and limits. Reading from motors saves those values locally. Neither option moves the arm or
        rewrites the motors.
      </p>
    </section>
  );
}
