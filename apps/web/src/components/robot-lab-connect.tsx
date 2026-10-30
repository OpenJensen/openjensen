'use client';

import { Icon } from '@/components/icon';
import { RobotLabMotorScan } from './robot-lab-motor-scan';
import { Notice } from './robot-lab-shared';
import type { RobotLabState } from './robot-lab-state';

export function RobotLabConnect({ lab }: { lab: RobotLabState }) {
  const {
    active,
    canAct,
    canUseArm,
    checkArm,
    devices,
    identity,
    missingHardwareTools,
    online,
    pending,
    robotId,
    scan,
    state,
    unavailableFeatures,
    verifyNext,
  } = lab;
  return (
    <>
      <div className="robot-lab-form">
        {!online && (
          <Notice>
            <strong>Companion offline</strong>Start the Robot companion, then check the connection.
          </Notice>
        )}
        {devices.isPending && online && (
          <p className="robot-lab-subtle" role="status">
            Looking for serial devices and cameras…
          </p>
        )}
        {devices.error && (
          <Notice kind="warning">
            <strong>Device discovery needs attention</strong>
            {devices.error.message}
            <button className="text-button" onClick={() => void devices.refetch()}>
              Retry discovery
            </button>
          </Notice>
        )}
        {identity}

        {online && unavailableFeatures && missingHardwareTools.length > 0 && (
          <Notice kind="warning">
            <strong>Missing dependencies</strong>
            {missingHardwareTools.map(([name]) => name).join(', ')}
          </Notice>
        )}
        {online && devices.data?.camera_error && <Notice kind="warning">{devices.data.camera_error}</Notice>}
        <div className="robot-lab-actions">
          <button
            className="primary-button"
            title="Read IDs and saved calibration settings; no motion"
            disabled={!canUseArm || active || !state?.capabilities?.probe || !/^[A-Za-z0-9_-]+$/.test(robotId)}
            onClick={checkArm}
          >
            {pending === 'Arm scan completed.' ? 'Scanning…' : 'Scan motor IDs'}
          </button>
          {scan?.ready && (
            <button className="secondary-button" disabled={!canAct} onClick={verifyNext}>
              Verify physical mapping
              <Icon name="arrow" size={15} />
            </button>
          )}
        </div>
        <RobotLabMotorScan lab={lab} />
      </div>
    </>
  );
}
