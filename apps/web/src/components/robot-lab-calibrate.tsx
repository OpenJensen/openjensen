'use client';

import { Icon } from '@/components/icon';
import { robotApi } from '@/lib/robot-api';
import { Heading, Notice, HardwareNotice, RangeTable } from './robot-lab-shared';
import type { RobotLabState } from './robot-lab-state';

export function RobotLabCalibrate({ lab }: { lab: RobotLabState }) {
  const {
    active,
    armLabel,
    calibration,
    calibrationError,
    calibrationStage,
    calibrationSummary,
    canAct,
    canUseArm,
    capturedRanges,
    identity,
    jointGuide,
    localCalibrationRef,
    ownSession,
    ownedRef,
    ownsSession,
    pending,
    perform,
    port,
    profileCheck,
    robotId,
    role,
    selectedCalibrated,
    selectedProfile,
    session,
    setOwnsSession,
    state,
  } = lab;
  return (
    <>
      <Heading title={`${armLabel} calibration`} />
      <div className="robot-lab-form">
        {calibrationSummary}
        {localCalibrationRef.current && (
          <p className="robot-lab-subtle">
            Calibration runs locally with torque off. You can leave this page and return to continue.
          </p>
        )}
        {jointGuide}
        {identity}
        <div className="robot-lab-calibration-steps">
          <span className={calibration?.phase === 'idle' || calibration?.phase === 'saved' ? 'current' : ''}>
            1. Prepare
          </span>
          <Icon name="arrow" size={12} />
          <span className={calibrationStage === 'center' ? 'current' : ''}>2. Middle position</span>
          <Icon name="arrow" size={12} />
          <span className={calibrationStage === 'ranges' ? 'current' : ''}>3. Joint ranges</span>
        </div>
        {calibrationError && <HardwareNotice label="Calibration error" message={calibrationError} />}
        {calibration?.recoverable ? (
          <Notice kind="warning">
            <strong>Calibration paused</strong>
            <p>
              {calibration.interrupted_stage === 'center' ||
              (calibration.phase === 'idle' && !Object.keys(calibration.offsets).length)
                ? 'Preparation is saved on the robot computer.'
                : 'Captured middle and ranges are saved on the robot computer.'}
            </p>
            <button
              className="primary-button"
              disabled={!canUseArm || active}
              onClick={() =>
                void perform(
                  'Captured calibration resumed. Torque remains off.',
                  robotApi.resumeCalibration,
                  ownSession,
                )
              }
            >
              Resume captured calibration
            </button>
          </Notice>
        ) : (
          calibration?.phase === 'interrupted' && (
            <Notice kind="warning">
              <strong>Calibration interrupted</strong>
              <p>
                {calibration.interrupted_stage === 'center'
                  ? 'Middle position was not saved.'
                  : 'Joint ranges were not saved.'}{' '}
                Restart calibration to continue.
              </p>
            </Notice>
          )
        )}
        {session?.mode !== 'calibration' && (
          <>
            <Notice>
              Support the arm. Calibration releases torque; all six motors must be connected and assigned IDs.
            </Notice>
            <button
              className="primary-button"
              disabled={!canUseArm || active || !/^[A-Za-z0-9_-]+$/.test(robotId) || !state?.capabilities?.calibration}
              onClick={() =>
                void perform(
                  'Calibration started. Position each joint near its middle.',
                  () => robotApi.startCalibration(port, robotId, role),
                  ownSession,
                )
              }
            >
              {calibration?.recoverable
                ? 'Start a new calibration'
                : calibration?.phase === 'interrupted'
                  ? 'Restart calibration'
                  : selectedCalibrated
                    ? 'Recalibrate arm'
                    : 'Begin calibration'}
            </button>
          </>
        )}
        {session?.mode === 'calibration' && calibration?.phase === 'center' && (
          <>
            <Notice>
              Position each joint at mid-travel, with the gripper half open. Support the arm and record the center.
            </Notice>
            <RangeTable positions={session.positions} calibration />
            <div className="robot-lab-actions">
              <button
                className="primary-button"
                disabled={!canAct || !ownsSession}
                onClick={() =>
                  void perform(
                    'Center recorded. Move each joint through its comfortable range.',
                    robotApi.centerCalibration,
                    undefined,
                    true,
                    'ranges',
                  )
                }
              >
                {pending === 'Center recorded. Move each joint through its comfortable range.'
                  ? 'Recording middle position…'
                  : 'Record middle position'}
              </button>
              <button
                className="secondary-button"
                disabled={!canAct}
                onClick={() =>
                  void perform('Calibration cancelled.', robotApi.cancelCalibration, () => {
                    ownedRef.current = false;
                    setOwnsSession(false);
                  })
                }
              >
                Cancel calibration
              </button>
            </div>
          </>
        )}
        {session?.mode === 'calibration' && calibration?.phase === 'ranges' && (
          <>
            <Notice>
              Move each joint through its usable range without forcing it. Wrist roll uses its full encoder range.
            </Notice>
            <div className="robot-lab-mapping-heading">
              <h3>Joint ranges</h3>
              <span>{capturedRanges}/5 ranges detected · wrist roll automatic</span>
            </div>
            <RangeTable positions={session.positions} mins={calibration.mins} maxes={calibration.maxes} calibration />
            <div className="robot-lab-actions">
              <button
                className="primary-button"
                disabled={!canAct || !ownsSession || capturedRanges < 5}
                onClick={() =>
                  void perform(
                    'Calibration saved.',
                    robotApi.finishCalibration,
                    () => {
                      ownedRef.current = false;
                      setOwnsSession(false);
                    },
                    true,
                    'saved',
                  )
                }
              >
                Save calibration
              </button>
              <button
                className="secondary-button"
                disabled={!canAct}
                onClick={() =>
                  void perform('Calibration cancelled.', robotApi.cancelCalibration, () => {
                    ownedRef.current = false;
                    setOwnsSession(false);
                  })
                }
              >
                Cancel calibration
              </button>
            </div>
          </>
        )}
        {selectedProfile && (
          <details className="robot-lab-details">
            <summary>Saved calibration details</summary>
            <p>
              {selectedProfile.valid === false
                ? selectedProfile.error
                : profileCheck?.state === 'matched'
                  ? 'Saved offsets and limits match all six connected motors.'
                  : profileCheck?.state === 'mismatch'
                    ? 'Motor offsets or limits differ from the saved profile. Recalibrate before control.'
                    : !selectedCalibrated
                      ? 'Motor setup changed after this calibration was saved.'
                      : 'Middle and ranges are saved. Scan the arm to compare its current settings.'}
            </p>
            {profileCheck?.error && <p>{profileCheck.error}</p>}
            <code>{selectedProfile.path}</code>
          </details>
        )}
      </div>
    </>
  );
}
