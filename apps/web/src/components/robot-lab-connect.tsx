'use client';

import { Icon } from '@/components/icon';
import { robotJoints } from '@/lib/robot-api';
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
    openStep,
    pending,
    responding,
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
        {scan && (
          <section className="robot-lab-form" aria-label="Motor scan results">
            <div className="robot-lab-mapping-heading">
              <h3>{responding}/6 motors responding</h3>
              <span>
                No motion ·{' '}
                {scan.checked_at
                  ? new Date(scan.checked_at * 1000).toLocaleTimeString([], {
                      hour: '2-digit',
                      minute: '2-digit',
                    })
                  : 'Just scanned'}
              </span>
            </div>
            <div className="robot-lab-table-wrap">
              <table className="robot-lab-table">
                <thead>
                  <tr>
                    <th>Expected joint</th>
                    <th>ID</th>
                    <th>Response</th>
                  </tr>
                </thead>
                <tbody>
                  {robotJoints.map((joint) => {
                    const motor = scan.motors.find((item) => item.id === joint.id);
                    return (
                      <tr key={joint.name}>
                        <td>{joint.label}</td>
                        <td>{joint.id}</td>
                        <td>
                          <span className={motor?.stable && motor.model === 777 ? 'robot-lab-pass' : 'robot-lab-fail'}>
                            {!motor
                              ? 'No response'
                              : motor.model !== 777
                                ? `Unexpected model ${motor.model}`
                                : motor.stable
                                  ? 'Responding'
                                  : 'Intermittent'}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {!scan.ready && (
              <div className="robot-lab-actions">
                <span className="robot-lab-subtle">Check motor power and IDs.</span>
                <button className="text-button" onClick={() => openStep(1)}>
                  Open motor setup
                </button>
              </div>
            )}
            <details className="robot-lab-details">
              <summary>What this scan checks</summary>
              <p>Replies at IDs 1–6 and STS3215 model. It cannot tell which physical joint owns an ID.</p>
              {scan.warning && <p>{scan.warning}</p>}
            </details>
          </section>
        )}
      </div>
    </>
  );
}
