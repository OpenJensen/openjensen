'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { robotApi } from '@/lib/robot-api';
import type { RobotLabState } from './robot-lab-state';

export function RobotLabCalibrationProfiles({ lab }: { lab: RobotLabState }) {
  const { active, busy, online, perform, port, portConnected, robotId, role, setProbe, state } = lab;
  const [chosen, setChosen] = useState<string | null>(null);
  const [sameArm, setSameArm] = useState(false);
  const [editing, setEditing] = useState(false);
  const [source, setSource] = useState<'saved' | 'motors'>('saved');
  const supported = !!state?.capabilities?.calibration_profiles;
  const profiles = useQuery({
    queryKey: ['robot-lab', 'calibration-profiles', robotId, role, port],
    queryFn: () => robotApi.calibrationProfiles(robotId, role, port),
    enabled: online && supported && !!robotId && !active,
    refetchOnMount: 'always',
  });
  const entries = profiles.data?.profiles ?? [];
  const current = entries.find((entry) => entry.current);
  const selected = chosen ?? current?.key ?? entries[0]?.key ?? '';
  const needsReview = profiles.data?.needs_review;
  const ready = profiles.data && !needsReview;
  const showChoices = editing || needsReview;
  const unavailable = active || busy || !online || !supported || !portConnected || !sameArm;
  if (!supported) return null;

  function reuse() {
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
        setEditing(false);
        setChosen(null);
        void profiles.refetch();
      },
    );
  }

  return (
    <section className="robot-lab-transfer robot-lab-calibration-profile" aria-label="Calibration profile">
      <div className="robot-lab-calibration-heading">
        <h3>Calibration profile</h3>
        <span className={needsReview ? 'robot-lab-calibration-attention' : ''}>
          {needsReview ? 'Review needed' : ready ? 'Saved and ready' : 'Checking…'}
        </span>
      </div>
      {current && (
        <div className="robot-lab-calibration-current">
          <strong>{current.robot_id}</strong>
          <span>Current profile · {new Date(current.saved_at * 1000).toLocaleString()}</span>
        </div>
      )}
      {ready && !editing && (
        <>
          <p>Your calibration is saved. You do not need to save it again.</p>
          <div className="robot-lab-actions">
            <button className="primary-button" disabled={busy || active} onClick={() => lab.openStep(3)}>
              Continue to teleoperation
            </button>
            <button
              className="secondary-button"
              disabled={busy || active}
              onClick={() => {
                setEditing(true);
                setSameArm(false);
              }}
            >
              Change profile
            </button>
          </div>
        </>
      )}
      {needsReview && (
        <p className="robot-lab-event robot-lab-warning" role="status">
          This arm needs a calibration check. If its mechanics and motor IDs are unchanged, verify the saved file or
          read its motor settings. You do not need to repeat manual calibration.
        </p>
      )}
      {profiles.error && <p role="alert">Could not load local profiles: {profiles.error.message}</p>}
      {showChoices && (
        <>
          <div className="robot-lab-actions" role="group" aria-label="Calibration source">
            <button
              className="secondary-button"
              aria-pressed={source === 'saved'}
              disabled={active || busy}
              onClick={() => {
                setSource('saved');
                setSameArm(false);
              }}
            >
              Saved file
            </button>
            <button
              className="secondary-button"
              aria-pressed={source === 'motors'}
              disabled={active || busy}
              onClick={() => {
                setSource('motors');
                setSameArm(false);
              }}
            >
              Settings on motors
            </button>
          </div>
          {source === 'saved' ? (
            <label>
              Saved calibration
              <select
                value={selected}
                disabled={active || busy || profiles.isPending}
                onChange={(event) => {
                  setChosen(event.target.value);
                  setSameArm(false);
                }}
              >
                {!entries.length && <option value="">No saved files yet</option>}
                {entries.map((entry) => (
                  <option key={entry.key} value={entry.key}>
                    {entry.current ? 'Current' : entry.source === 'backup' ? 'Backup' : 'Saved'} · {entry.robot_id} ·{' '}
                    {new Date(entry.saved_at * 1000).toLocaleString()}
                  </option>
                ))}
              </select>
            </label>
          ) : (
            <p>
              Read the offsets and joint limits already stored on this arm, then save them as its current local profile.
            </p>
          )}
          <label className="robot-lab-check">
            <input
              type="checkbox"
              checked={sameArm}
              disabled={active || busy}
              onChange={(event) => setSameArm(event.target.checked)}
            />
            Same arm — mechanics and joint-to-motor IDs have not changed.
          </label>
          <div className="robot-lab-actions">
            <button
              className="primary-button"
              disabled={unavailable || (source === 'saved' && !entries.some((entry) => entry.key === selected))}
              onClick={reuse}
            >
              {busy
                ? 'Checking calibration…'
                : source === 'motors'
                  ? 'Read and save motor settings'
                  : needsReview && selected === current?.key
                    ? 'Verify saved calibration'
                    : 'Use this profile'}
            </button>
            {editing && !needsReview && (
              <button
                className="text-button"
                disabled={busy}
                onClick={() => {
                  setEditing(false);
                  setSameArm(false);
                }}
              >
                Cancel
              </button>
            )}
          </div>
          <p className="robot-lab-subtle">
            {active ? 'Stop the current session first. ' : !portConnected ? 'Connect this arm first. ' : ''}
            Checks all six motors with torque off. A saved file must match their settings. Neither option moves the arm
            or writes to the motors. Your previous file is kept as a backup.
          </p>
        </>
      )}
      <div className="robot-lab-calibration-footer">
        <small>Saved only on this robot computer. Kept through updates. Never uploaded to GitHub.</small>
        <button
          className="text-button"
          disabled={active || busy || profiles.isFetching}
          onClick={() => void profiles.refetch()}
        >
          Refresh
        </button>
      </div>
    </section>
  );
}
