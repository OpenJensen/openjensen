'use client';

import { HardwareNotice } from './robot-lab-shared';
import { useRobotLab } from './robot-lab-state';
import { RobotLabDevices, RobotLabDeviceDetail } from './robot-lab-devices';
import { RobotLabConnect } from './robot-lab-connect';
import { RobotLabMotorIds } from './robot-lab-motor-ids';
import { RobotLabCalibrate } from './robot-lab-calibrate';
import { RobotLabTeleoperate } from './robot-lab-teleoperate';
import { RobotLabRecord } from './robot-lab-record';
import { RobotLabCamera } from './robot-lab-camera';
import { RobotLabDiagnostics } from './robot-lab-diagnostics';
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
    previousSessionError,
    reconnecting,
    recording,
    refresh,
    robotId,
    session,
    state,
    status,
    step,
    stop,
  } = lab;
  return (
    <div className="robot-lab">
      <div className="robot-lab-statusbar">
        <div className="robot-lab-connection">
          <span className={`robot-lab-dot${online ? ' online' : ''}`} />
          <div>
            <strong>
              {status.isPending
                ? 'Looking for devices…'
                : online
                  ? `${lab.catalog.filter((d) => d.connected).length} USB devices connected`
                  : 'Device connection service unavailable'}
            </strong>
            <small>
              {online
                ? active
                  ? `${session?.mode} · ${session?.robot_id ?? robotId} · Torque ${session?.torque_enabled ? 'on' : 'off'}`
                  : 'Open a device to configure its motors and calibration.'
                : (state?.message ??
                  status.error?.message ??
                  'Start the robot connection service on the computer with the USB cables.')}
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
          {(active || session?.torque_enabled || session?.error || pending === 'Arm stopped.') && (
            <button
              className="danger-button robot-lab-stop"
              onClick={stop}
              disabled={!state?.ok && !active}
              aria-label="Stop arm and disable torque"
            >
              <span />
              {pending === 'Arm stopped.' ? 'Stopping…' : 'Stop arm'}
            </button>
          )}
        </div>
      </div>
      <nav className="robot-lab-main-tabs" aria-label="Robot Lab tabs">
        {[
          { title: 'Devices', step: 0 },
          { title: 'Teleoperation', step: 3 },
          { title: 'Record', step: 4 },
        ].map((item) => (
          <button
            key={item.title}
            aria-current={(step < 3 ? 0 : step) === item.step ? 'page' : undefined}
            className={(step < 3 ? 0 : step) === item.step ? 'selected' : ''}
            disabled={
              busy ||
              (active &&
                (item.step === 0 ||
                  session?.mode === 'calibration' ||
                  session?.mode === 'setup' ||
                  session?.mode === 'verification'))
            }
            onClick={() => {
              if (item.step === 0) lab.setSelectedDeviceId(null);
              lab.openStep(item.step);
            }}
          >
            {item.title}
          </button>
        ))}
      </nav>
      {step !== 2 && error && <HardwareNotice label="Latest operation error" message={error} />}
      {step !== 2 && session?.error && session.error !== error && !previousSessionError && (
        <HardwareNotice label="Hardware session error" message={session.error} />
      )}
      {previousSessionError && (
        <HardwareNotice
          label="Previous session error"
          context="Previous hardware session stopped"
          message={previousSessionError}
        />
      )}
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
      <RobotLabDiagnostics diagnostics={state?.diagnostics} />
      {step < 3 && !lab.selectedDeviceId ? (
        <RobotLabDevices lab={lab} />
      ) : (
        <div className="robot-lab-grid">
          <section
            className="robot-lab-main"
            aria-label={step < 3 ? 'Device configuration' : step === 3 ? 'Teleoperation' : 'Record'}
          >
            {step < 3 && (
              <RobotLabDeviceDetail lab={lab}>
                {step === 0 && <RobotLabConnect lab={lab} />}
                {step === 1 && <RobotLabMotorIds lab={lab} />}
                {step === 2 && <RobotLabCalibrate lab={lab} />}
              </RobotLabDeviceDetail>
            )}
            {step === 3 && <RobotLabTeleoperate lab={lab} />}
            {step === 4 && <RobotLabRecord lab={lab} />}
            {busy && (
              <p className="robot-lab-subtle" role="status" style={{ marginTop: 16 }}>
                {pending === 'Arm stopped.'
                  ? 'Stopping the arm… Keep supporting it while torque is released and motor cleanup finishes.'
                  : 'Working… Keep this page open.'}
              </p>
            )}
          </section>
          <RobotLabCamera lab={lab} />
        </div>
      )}
    </div>
  );
}
