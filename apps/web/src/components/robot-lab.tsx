'use client';

import { steps, HardwareNotice } from './robot-lab-shared';
import { useRobotLab } from './robot-lab-state';
import { RobotLabConnect } from './robot-lab-connect';
import { RobotLabMotorIds } from './robot-lab-motor-ids';
import { RobotLabCalibrate } from './robot-lab-calibrate';
import { RobotLabTeleoperate } from './robot-lab-teleoperate';
import { RobotLabRecord } from './robot-lab-record';
import { RobotLabCamera } from './robot-lab-camera';
import './robot-lab.css';

export function RobotLab() {
  const lab = useRobotLab();
  const {
    active,
    busy,
    checkingConnection,
    error,
    localCalibrationRef,
    notice,
    online,
    pending,
    readiness,
    reconnecting,
    recording,
    refresh,
    robotId,
    session,
    state,
    status,
    step,
    stop,
    usbStatus,
  } = lab;
  return (
    <div className="robot-lab">
      <div className="robot-lab-statusbar">
        <div className="robot-lab-connection">
          <span className={`robot-lab-dot${online ? ' online' : ''}`} />
          <div>
            <strong>
              {status.isPending
                ? 'Connecting…'
                : online
                  ? `${state?.label ?? 'Robot companion'} connected`
                  : 'Robot companion unavailable'}
            </strong>
            <small>
              {online
                ? active
                  ? `${session?.mode === 'verification' ? 'Mapping check' : session?.mode === 'setup' ? 'Assembled motor setup' : session?.mode === 'calibration' ? 'Calibration' : session?.mode === 'handguide' ? 'Hand guiding' : session?.mode === 'leader' ? 'Leader teleoperation' : session?.mode === 'agent' ? 'AI teleoperation' : 'Manual control'} · ${session?.robot_id ?? robotId} · ${session?.mode === 'setup' ? 'Motor setup active' : session?.torque_enabled ? 'Torque command: on' : 'Torque command: off'}`
                  : usbStatus
                : (state?.message ?? status.error?.message ?? 'Check that the local companion is running.')}
            </small>
          </div>
        </div>
        <div className="robot-lab-status-actions">
          <button
            className="text-button"
            onClick={() => void refresh()}
            disabled={busy || checkingConnection || status.isPending}
          >
            {checkingConnection ? 'Refreshing…' : 'Refresh devices'}
          </button>
          {(active || session?.torque_enabled || session?.error || pending === 'Stop requested.') && (
            <button
              className="danger-button robot-lab-stop"
              onClick={stop}
              disabled={!state?.ok && !active}
              aria-label="Stop arm and disable torque"
            >
              <span />
              {pending === 'Stop requested.' ? 'Stopping…' : 'Stop arm'}
            </button>
          )}
        </div>
      </div>
      {readiness}
      {step !== 2 && (session?.error || error) && <HardwareNotice message={(session?.error || error)!} />}
      {(reconnecting || (!online && localCalibrationRef.current)) && (
        <p className="robot-lab-event robot-lab-warning" role="status">
          {localCalibrationRef.current
            ? 'Reconnecting… Calibration continues on the robot computer. Captured values are retained.'
            : 'Reconnecting… Motion controls are paused; the motor watchdog remains active.'}
        </p>
      )}
      {notice && (
        <p className="robot-lab-event" role="status">
          {notice}
        </p>
      )}
      {recording?.error && <HardwareNotice message={recording.error} />}
      <div className="robot-lab-grid">
        <section className="robot-lab-main" aria-label={steps[step].title}>
          {step === 0 && <RobotLabConnect lab={lab} />}
          {step === 1 && <RobotLabMotorIds lab={lab} />}
          {step === 2 && <RobotLabCalibrate lab={lab} />}
          {step === 3 && <RobotLabTeleoperate lab={lab} />}
          {step === 4 && <RobotLabRecord lab={lab} />}
          {busy && (
            <p className="robot-lab-subtle" role="status" style={{ marginTop: 16 }}>
              {pending === 'Stop requested.'
                ? 'Stopping the arm… Keep supporting it while torque is released and motor cleanup finishes.'
                : 'Working… Keep this page open.'}
            </p>
          )}
        </section>
        <RobotLabCamera lab={lab} />
      </div>
    </div>
  );
}
