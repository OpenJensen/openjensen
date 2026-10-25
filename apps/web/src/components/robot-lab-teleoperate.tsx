'use client';

import { Icon } from '@/components/icon';
import { robotApi, robotJoints } from '@/lib/robot-api';
import { Heading, Notice, RangeTable, JointControl } from './robot-lab-shared';
import type { RobotLabState } from './robot-lab-state';

export function RobotLabTeleoperate({ lab }: { lab: RobotLabState }) {
  const {
    active,
    busy,
    calibrated,
    canAct,
    canUseArm,
    centering,
    controlJointOptions,
    controlJoints,
    controlling,
    devices,
    followerMismatch,
    identity,
    jog,
    jogging,
    jointStep,
    leaderCalibrated,
    leaderConnected,
    leaderId,
    leaderPort,
    middleTolerance,
    mode,
    motionReady,
    online,
    ownSession,
    ownsSession,
    partialControl,
    partialSelection,
    perform,
    port,
    recording,
    robotId,
    session,
    sessionJoints,
    setJogStep,
    setLeaderId,
    setLeaderPort,
    setMiddleTolerance,
    setMode,
    setMotionReady,
    setStep,
    state,
    stop,
    targetJoint,
  } = lab;
  return (
    <>
      <Heading title="Teleoperation" />
      <div className="robot-lab-form">
        {mode !== 'handguide' &&
          (state?.capabilities?.lerobot_position_control ? (
            <Notice>
              <strong>LeRobot SO101 control</strong>Direct position targets; motors may move quickly. Keep the full path
              clear.
            </Notice>
          ) : (
            online && (
              <Notice kind="warning">
                <strong>Earlier control profile</strong>Update the robot companion to use LeRobot SO101 control.
              </Notice>
            )
          ))}
        {identity}
        <div className="robot-lab-mode-grid" role="group" aria-label="Control mode">
          {(
            [
              {
                value: 'manual',
                title: 'Joint control',
                detail: 'Sliders and position steps',
                icon: 'sliders',
              },
              { value: 'handguide', title: 'Hand guiding', detail: 'Torque off', icon: 'robot' },
              { value: 'leader', title: 'Leader → follower', detail: 'Two calibrated arms', icon: 'layers' },
            ] as const
          ).map((item) => (
            <button
              key={item.value}
              className={`robot-lab-mode${mode === item.value ? ' selected' : ''}`}
              onClick={() => {
                setMode(item.value);
                setMotionReady(false);
              }}
              disabled={active || busy}
              aria-pressed={mode === item.value}
            >
              <Icon name={item.icon} size={19} />
              <strong>{item.title}</strong>
              <small>{item.detail}</small>
            </button>
          ))}
        </div>
        {mode === 'leader' && (
          <div className="robot-lab-fields">
            <label>
              Leader USB port
              <select
                value={leaderPort}
                disabled={active || busy}
                aria-label="Leader USB port"
                onChange={(event) => setLeaderPort(event.target.value)}
              >
                <option value="">Select the second arm</option>
                {devices.data?.ports
                  .filter((item) => item.device !== port)
                  .map((item) => (
                    <option key={item.device} value={item.device}>
                      {item.device}
                    </option>
                  ))}
              </select>
            </label>
            <label>
              Leader name
              <input
                value={leaderId}
                disabled={active || busy}
                aria-label="Leader name"
                onChange={(event) => setLeaderId(event.target.value)}
              />
              <small>
                {leaderCalibrated ? 'Leader calibration found.' : 'Calibrate this arm with the leader role first.'}
              </small>
            </label>
          </div>
        )}
        {controlJointOptions}
        {mode === 'manual' && !partialSelection && state?.capabilities?.middle_tolerance && (
          <label>
            Middle tolerance
            <select
              aria-label="Middle tolerance"
              value={active ? (session?.centering?.tolerance ?? middleTolerance) : middleTolerance}
              onChange={(event) => {
                setMiddleTolerance(Number(event.target.value));
                setMotionReady(false);
              }}
              disabled={active || busy}
            >
              <option value={2}>2° / 2%</option>
              <option value={5}>5° / 5%</option>
            </select>
          </label>
        )}
        {!active && (
          <>
            {!calibrated && (
              <Notice kind="warning">
                <strong>{followerMismatch ? 'Calibration settings differ' : 'Calibration required'}</strong>
                Follower: {robotId}.
                <button className="text-button" onClick={() => setStep(2)}>
                  Open calibration
                </button>
              </Notice>
            )}
            <label className="robot-lab-check">
              <input
                type="checkbox"
                checked={motionReady}
                disabled={busy}
                onChange={(event) => setMotionReady(event.target.checked)}
              />
              <span>
                <strong>{mode === 'handguide' ? 'Arm supported.' : 'Arm secured; workspace clear.'}</strong>
                {mode === 'handguide'
                  ? 'Torque stays off.'
                  : mode === 'leader'
                    ? 'Enable torque; the follower starts tracking the leader immediately.'
                    : partialSelection
                      ? 'Enable selected motors and hold their current pose.'
                      : 'Move to the recorded middle. Keep the full path clear.'}
              </span>
            </label>
            <button
              className="primary-button"
              disabled={
                !canUseArm ||
                !calibrated ||
                !motionReady ||
                !state?.capabilities?.control ||
                (mode === 'manual' &&
                  (!controlJoints.length ||
                    !state?.capabilities?.move_to_middle ||
                    (partialSelection && !state?.capabilities?.partial_control))) ||
                (mode === 'leader' && (!leaderConnected || !leaderCalibrated || leaderPort === port))
              }
              onClick={() =>
                void perform(
                  mode === 'handguide'
                    ? 'Hand guiding started. Torque remains off.'
                    : 'Control session started. Stop releases the motors.',
                  () =>
                    robotApi.startControl({
                      port,
                      robot_id: robotId,
                      mode,
                      ...(mode === 'manual'
                        ? {
                            joints: controlJoints,
                            move_to_middle: !partialSelection,
                            ...(!partialSelection && state?.capabilities?.middle_tolerance
                              ? { middle_tolerance: middleTolerance }
                              : {}),
                          }
                        : {}),
                      ...(mode === 'leader' ? { leader_port: leaderPort, leader_id: leaderId } : {}),
                    }),
                  ownSession,
                )
              }
            >
              {mode === 'handguide'
                ? 'Start hand guiding'
                : mode === 'leader'
                  ? 'Start leader teleoperation'
                  : partialSelection
                    ? 'Enable selected motors'
                    : 'Enable & move to recorded middle'}
            </button>
          </>
        )}
        {controlling && (
          <>
            <Notice kind="success">
              <strong>
                {session?.mode === 'manual'
                  ? centering
                    ? `Moving to recorded middle · ${Math.round((session.centering?.progress ?? 0) * 100)}%`
                    : 'On-screen controls are active.'
                  : session?.mode === 'leader'
                    ? 'Leader teleoperation is active.'
                    : 'Hand guiding is active. Torque is off.'}
              </strong>
              {ownsSession
                ? 'Tab hidden or disconnected → torque off.'
                : 'Session owned by another client. Stop it to take control.'}
            </Notice>
            {session?.mode === 'manual' ? (
              <>
                {partialControl && <p className="robot-lab-subtle">Selected motors only · recording unavailable</p>}
                <div className="robot-lab-control-heading">
                  <h3>Joint controls</h3>
                  <span className="robot-lab-subtle">Position → target · release slider to move</span>
                  <label>
                    Step
                    <select
                      value={jointStep}
                      aria-label="Joint step size"
                      onChange={(event) => setJogStep(Number(event.target.value))}
                    >
                      <option value={0.5}>0.5° / %</option>
                      <option value={1}>1° / %</option>
                      <option value={2}>2° / %</option>
                      {state?.capabilities?.large_joint_steps && (
                        <>
                          <option value={5}>5° / %</option>
                          <option value={10}>10° / %</option>
                        </>
                      )}
                    </select>
                  </label>
                </div>
                <div className="robot-lab-joints" aria-busy={jogging}>
                  {robotJoints
                    .filter((joint) => sessionJoints.includes(joint.name))
                    .map((joint) => (
                      <JointControl
                        key={joint.name}
                        joint={joint}
                        position={session.positions[joint.name]}
                        target={session.targets?.[joint.name]}
                        sliderTarget={session.slider_targets?.[joint.name]}
                        limits={session.joint_limits?.[joint.name]}
                        sliders={!!state?.capabilities?.joint_sliders}
                        disabled={!canAct || !ownsSession || jogging || centering || !!recording?.episode_active}
                        step={jointStep}
                        onJog={jog}
                        onTarget={targetJoint}
                      />
                    ))}
                </div>
              </>
            ) : (
              <RangeTable positions={session?.positions} />
            )}
            <div className="robot-lab-actions">
              <button className="danger-button robot-lab-stop" onClick={stop}>
                <span />
                Stop & release motors
              </button>
              <button
                className="secondary-button"
                disabled={busy || !ownsSession || centering || partialControl}
                onClick={() => setStep(4)}
              >
                Create a dataset
                <Icon name="arrow" size={14} />
              </button>
            </div>
          </>
        )}
      </div>
    </>
  );
}
