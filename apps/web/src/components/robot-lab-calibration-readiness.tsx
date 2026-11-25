'use client';

import { CalibrationChoice } from './robot-lab-calibration-choice';
import { useState } from 'react';
import { useQueries } from '@tanstack/react-query';
import { robotApi } from '@/lib/robot-api';
import type { RobotLabState } from './robot-lab-state';

// The readiness card provides the next action for these setup conditions.
export const isCalibrationSetupError = (message?: string | null) =>
  !!message &&
  /^(Motor setup changed on this port|Motor IDs changed on this port|No saved calibration for )/.test(message);

export function RobotLabCalibrationReadiness({ lab }: { lab: RobotLabState }) {
  const [editing, setEditing] = useState<string | null>(null);
  const modern = !!lab.state?.capabilities?.calibration_choice;
  const slots = [
    { role: 'follower' as const, robotId: lab.robotId, port: lab.port },
    ...(lab.mode === 'leader' ? [{ role: 'leader' as const, robotId: lab.leaderId, port: lab.leaderPort }] : []),
  ];
  const devices = slots.flatMap((slot) => {
    const device = lab.catalog.find((d) => d.connected && d.robot_id === slot.robotId && d.port === slot.port);
    return device ? [{ ...device, role: slot.role }] : [];
  });
  const checks = useQueries({
    queries: devices.map((device) => ({
      queryKey: ['robot-lab', 'calibration-profiles', device.robot_id, device.role ?? 'follower', device.port],
      queryFn: () => robotApi.calibrationProfiles(device.robot_id, device.role ?? 'follower', device.port),
      enabled: lab.online && !lab.active,
      refetchInterval: lab.active ? (false as const) : 5000,
      refetchOnMount: 'always' as const,
    })),
  });
  if (lab.active || !devices.length) return null;
  const needsReview = checks.filter((check) => check.data?.needs_review).length;
  return (
    <section className="robot-lab-calibration-readiness" aria-label="Arm calibration status">
      <strong>{needsReview ? 'Calibration needs attention' : 'Saved calibrations'}</strong>
      <div className="robot-lab-calibration-arms">
        {devices.map((device, index) => {
          const check = checks[index];
          const review = check.data?.needs_review;
          const id = device.id + '/' + device.role;
          const library = check.data;
          const showChoice = modern && library && (library.needs_review || editing === id);
          return (
            <div key={id}>
              <div className="robot-lab-calibration-arm" aria-label={device.name || device.robot_id}>
                <div>
                  <strong>
                    {device.role === 'leader' ? 'Leader' : 'Follower'} · {device.name || device.robot_id}
                  </strong>
                  <span
                    className={
                      library?.conflict
                        ? 'robot-lab-choice-warning'
                        : review || check.isError
                          ? 'robot-lab-calibration-attention'
                          : ''
                    }
                  >
                    {check.isError
                      ? 'Could not check saved calibration'
                      : !check.data
                        ? 'Checking saved calibration…'
                        : modern && library?.conflict
                          ? 'Choose which calibration to use'
                          : modern && !review
                            ? library?.remembered_source
                              ? `Remembered · ${library.remembered_source === 'motors' ? 'On the arm' : 'On this machine'}`
                              : 'Profile saved on this machine'
                            : review
                              ? 'Needs review — saved files are available'
                              : 'Saved — no action needed'}
                  </span>
                </div>
                {(!modern || !review) && (
                  <button
                    className={review ? 'secondary-button' : 'text-button'}
                    disabled={lab.busy}
                    onClick={() => {
                      if (modern) {
                        setEditing(editing === id ? null : id);
                        return;
                      }
                      lab.chooseDevice(device);
                      lab.openStep(2);
                    }}
                  >
                    {modern && editing === id ? 'Close' : review ? 'Review calibration' : 'Manage profile'}
                  </button>
                )}
              </div>
              {showChoice && (
                <CalibrationChoice
                  lab={lab}
                  port={device.port}
                  robotId={device.robot_id}
                  role={device.role}
                  library={library}
                  onDone={() => setEditing(null)}
                />
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
