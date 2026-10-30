'use client';

import { robotJoints } from '@/lib/robot-api';
import { Notice } from './robot-lab-shared';
import type { RobotLabState } from './robot-lab-state';

export function RobotLabMotorScan({ lab }: { lab: RobotLabState }) {
  const { active, canUseArm, discoverMotorIds, openStep, pending, responding, scan, step } = lab;
  if (!scan) return null;
  const unexpected = scan.motors.filter((motor) => !scan.expected_ids.includes(motor.id));
  const fullScan = scan.scan_scope === 'all_addresses';
  return (
    <section className="robot-lab-form" aria-label="Motor scan results">
      <div className="robot-lab-mapping-heading">
        <h3>
          {fullScan
            ? `${scan.motors.filter((motor) => motor.stable && motor.model === 777).length} motor addresses responding`
            : `${responding}/6 expected IDs responding`}
        </h3>
        <span>
          No motion ·{' '}
          {scan.checked_at
            ? new Date(scan.checked_at * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            : 'Just scanned'}
        </span>
      </div>
      {fullScan && (
        <p className="robot-lab-subtle">
          {responding}/6 expected IDs (1–6) respond. Other addresses below also count as connected motors; reconnecting
          USB does not reset their IDs.
        </p>
      )}
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
                        ? 'No reply at this ID'
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
      {unexpected.length > 0 && (
        <Notice kind="warning">
          <strong>Other motor IDs found: {unexpected.map((motor) => motor.id).join(', ')}</strong>
          <p>
            Motors replied outside IDs 1–6. Interrupted motor setup can leave temporary IDs. Identify which physical
            joints own these addresses before saving IDs; their joint names cannot be inferred from this scan.
          </p>
          <ul>
            {unexpected.map((motor) => (
              <li key={motor.id}>
                ID {motor.id} · {motor.model === 777 ? 'STS3215' : `Model ${motor.model}`} ·{' '}
                {motor.stable ? 'Responding' : 'Intermittent'}
              </li>
            ))}
          </ul>
        </Notice>
      )}
      {!scan.ready && (
        <>
          <p className="robot-lab-subtle">
            {fullScan
              ? `Searched addresses 0–253 at ${scan.baudrate.toLocaleString()} baud. ${scan.motors.length} addresses replied. Check motor power, cables and configured baud rate if expected addresses are still absent.`
              : 'A missing reply at an expected ID does not prove that the physical motor is disconnected. Search other addresses before restarting motor setup.'}
          </p>
          <div className="robot-lab-actions">
            <button className="secondary-button" disabled={!canUseArm || active} onClick={discoverMotorIds}>
              {pending === 'Motor address search completed.' ? 'Searching motor IDs…' : 'Find other motor IDs'}
            </button>
            {step !== 1 && (
              <button className="text-button" onClick={() => openStep(1)}>
                Open motor setup
              </button>
            )}
          </div>
          <p className="robot-lab-subtle">
            This search only reads motor replies. It does not change IDs, calibration or torque. Assembled setup is a
            separate action that can rewrite motor IDs.
          </p>
        </>
      )}
      <details className="robot-lab-details">
        <summary>What this scan checks</summary>
        <p>
          {fullScan ? 'Replies at addresses 0–253' : 'Replies at IDs 1–6'} and STS3215 model. It cannot tell which
          physical joint owns an ID. Motors using a different baud rate will not appear.
        </p>
        {scan.warning && <p>{scan.warning}</p>}
      </details>
    </section>
  );
}
