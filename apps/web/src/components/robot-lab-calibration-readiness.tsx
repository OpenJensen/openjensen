'use client';

import { useQueries } from '@tanstack/react-query';
import { robotApi } from '@/lib/robot-api';
import type { RobotLabState } from './robot-lab-state';

// These admission errors are replaced by current, per-arm status. Other hardware
// errors remain visible, including mismatched values and failed motor reads.
export const isCalibrationSetupError = (message?: string | null) =>
  !!message &&
  /^(Motor setup changed on this port|Motor IDs changed on this port|No saved calibration for )/.test(message);

export function RobotLabCalibrationReadiness({ lab }: { lab: RobotLabState }) {
  const devices = lab.catalog.filter(
    (device) =>
      device.connected &&
      ((device.robot_id === lab.robotId && device.port === lab.port) ||
        (lab.mode === 'leader' && device.robot_id === lab.leaderId && device.port === lab.leaderPort)),
  );
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
          return (
            <div className="robot-lab-calibration-arm" key={device.id} aria-label={device.name || device.robot_id}>
              <div>
                <strong>
                  {device.role === 'leader' ? 'Leader' : 'Follower'} · {device.name || device.robot_id}
                </strong>
                <span className={review || check.isError ? 'robot-lab-calibration-attention' : ''}>
                  {check.isError
                    ? 'Could not check saved calibration'
                    : !check.data
                      ? 'Checking saved calibration…'
                      : review
                        ? 'Needs review — saved files are available'
                        : 'Saved — no action needed'}
                </span>
              </div>
              <button
                className={review ? 'secondary-button' : 'text-button'}
                disabled={lab.busy}
                onClick={() => {
                  lab.chooseDevice(device);
                  lab.openStep(2);
                }}
              >
                {review ? 'Review calibration' : 'Manage profile'}
              </button>
            </div>
          );
        })}
      </div>
    </section>
  );
}
