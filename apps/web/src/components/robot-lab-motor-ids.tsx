'use client';

import { Icon } from '@/components/icon';
import { robotApi, robotJoints } from '@/lib/robot-api';
import { setupOrder, Heading, Notice, HardwareNotice } from './robot-lab-shared';
import type { RobotLabState } from './robot-lab-state';

export function RobotLabMotorIds({ lab }: { lab: RobotLabState }) {
  const {
    active,
    allJointsMapped,
    assembled,
    assembledReady,
    assigned,
    busy,
    canAct,
    canUseArm,
    checkedCount,
    confirmJoint,
    followerPort,
    identifyMethod,
    isolated,
    jointGuide,
    mappingComplete,
    mappingValues,
    mismatched,
    motionChecks,
    motorIndex,
    online,
    openStep,
    ownSession,
    ownedRef,
    ownsSession,
    perform,
    port,
    portConnected,
    reidentify,
    scan,
    selectIllustratedJoint,
    selectedCalibrated,
    selectedMotor,
    session,
    setAssembledReady,
    setAssigned,
    setIdentifyMethod,
    setIsolated,
    setMotionChecks,
    setMotorIndex,
    setOwnsSession,
    setProbe,
    setReidentify,
    setStep,
    setVerificationReady,
    setWiggleId,
    setWiggleReady,
    state,
    verification,
    verificationReady,
    wiggleId,
    wiggleReady,
  } = lab;
  return (
    <>
      <Heading title="Motor IDs" />
      <div className="robot-lab-form">
        {assembled && mappingComplete && !reidentify && (
          <section className="robot-lab-form" aria-label="Saved motor mapping">
            <div className="robot-lab-mapping-heading">
              <h3>Saved IDs</h3>
              <span>{checkedCount}/6 motion checks</span>
            </div>
            {jointGuide}
            <div className="robot-lab-form" role="group" aria-label="Verify saved motor mapping">
              <label className="robot-lab-check">
                <input
                  type="checkbox"
                  checked={verificationReady}
                  disabled={busy || active || !portConnected}
                  onChange={(event) => setVerificationReady(event.target.checked)}
                />
                <span>Arm supported; joints clear of stops. Brief motion, then torque off.</span>
              </label>
              <button
                className="primary-button"
                disabled={!canUseArm || active || !verificationReady || !state?.capabilities?.mapping_verification}
                onClick={() => {
                  if (!state?.capabilities?.arm_readiness)
                    setMotionChecks((previous) => {
                      const next = { ...previous };
                      delete next[selectedMotor.name];
                      return next;
                    });
                  void perform(
                    'Mapping check requested.',
                    () => robotApi.wiggleSaved(port, selectedMotor.name),
                    ownSession,
                    false,
                  );
                }}
              >
                {session?.mode === 'verification'
                  ? verification?.phase === 'wiggling'
                    ? `Wiggling ID ${verification.motor_id}…`
                    : 'Checking motor addresses…'
                  : `Wiggle ${selectedMotor.label.toLowerCase()} · ID ${assembled.mapping[selectedMotor.name]}`}
              </button>
              {verification?.port === port &&
                verification.phase === 'done' &&
                verification.joint === selectedMotor.name &&
                verification.run_id &&
                motionChecks[selectedMotor.name]?.run !== verification.run_id && (
                  <div className="so101-detection-feedback" role="group" aria-label="Confirm physical joint">
                    <strong>ID {verification.motor_id} finished · torque off</strong>
                    <p>Did only the {selectedMotor.label.toLowerCase()} move?</p>
                    <div className="robot-lab-actions">
                      <button className="secondary-button" disabled={active || busy} onClick={() => confirmJoint(true)}>
                        Correct joint
                      </button>
                      <button
                        className="secondary-button"
                        disabled={active || busy}
                        onClick={() => confirmJoint(false)}
                      >
                        Wrong joint
                      </button>
                    </div>
                  </div>
                )}
              {verification?.port === port && verification.error && <HardwareNotice message={verification.error} />}
            </div>
            <div className="robot-lab-table-wrap">
              <table className="robot-lab-table">
                <thead>
                  <tr>
                    <th>Joint</th>
                    <th>Motor ID</th>
                    <th>Motion check</th>
                  </tr>
                </thead>
                <tbody>
                  {robotJoints.map((joint) => (
                    <tr key={joint.name}>
                      <td>
                        <button
                          className="text-button"
                          aria-pressed={selectedMotor.name === joint.name}
                          onClick={() => selectIllustratedJoint(joint.name)}
                        >
                          {joint.label}
                        </button>
                      </td>
                      <td>{assembled.mapping[joint.name]}</td>
                      <td>
                        {motionChecks[joint.name]?.correct
                          ? 'Correct'
                          : motionChecks[joint.name]
                            ? 'Mismatch'
                            : 'Unchecked'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {Object.values(motionChecks).some((check) => !check.correct) && (
              <Notice kind="warning">Mapping mismatch. Reidentify the joints before calibration.</Notice>
            )}
            {state?.mapping_checks?.error && <HardwareNotice message={state.mapping_checks.error} />}
            {checkedCount === 6 && !mismatched && (
              <div className="robot-lab-actions">
                <span className="robot-lab-pass">Physical mapping verified.</span>
                <button
                  className="secondary-button"
                  disabled={active || busy}
                  onClick={() => openStep(selectedCalibrated ? 3 : 2)}
                >
                  {selectedCalibrated ? 'Open joint controls' : 'Calibrate this arm'}
                  <Icon name="arrow" size={14} />
                </button>
              </div>
            )}
            <button
              className="text-button"
              disabled={busy || active}
              onClick={() => {
                setReidentify(true);
                setVerificationReady(false);
                setMotionChecks({});
              }}
            >
              Reidentify joints
            </button>
          </section>
        )}
        {assembled?.phase !== 'complete' && (
          <div className="so101-detection-feedback" role="status" aria-label="Motor identification status">
            <strong>
              {!online
                ? 'Movement detection unavailable'
                : session?.mode === 'setup' && ['scanning', 'splitting'].includes(assembled?.phase ?? '')
                  ? 'Preparing motor addresses'
                  : assembled?.phase === 'applying'
                    ? 'Saving motor IDs'
                    : assembled?.phase === 'observing' && identifyMethod === 'hand'
                      ? 'Listening for hand movement'
                      : session?.mode === 'setup' && ['ready', 'observing', 'review'].includes(assembled?.phase ?? '')
                        ? 'Ready to identify joints'
                        : assembled?.phase === 'wiggling'
                          ? 'Identifying motor by motion'
                          : 'Movement detection off'}
            </strong>
            <p>
              {!online
                ? 'Connect the Robot companion first.'
                : session?.mode === 'setup' && ['scanning', 'splitting'].includes(assembled?.phase ?? '')
                  ? `${assembled?.progress ? `Round ${assembled.progress}/12 · ` : ''}${assembled?.ids.length ?? 0}/6 addresses found. No movement is commanded. Keep this page visible until preparation finishes.`
                  : session?.mode === 'setup' &&
                      ['ready', 'observing', 'review', 'wiggling', 'applying'].includes(assembled?.phase ?? '')
                    ? assembled?.phase === 'applying'
                      ? 'Keep this page visible while the saved addresses are verified.'
                      : identifyMethod === 'hand'
                        ? assembled?.phase === 'observing'
                          ? `Move only the ${selectedMotor.label.toLowerCase()}, then hold its new position.`
                          : 'Click a joint on the model to start listening for its movement.'
                        : 'Select an address, request a wiggle, then select the joint that moved.'
                    : `${assembled?.ids.length ? `${assembled.ids.length}/6 addresses found. ` : ''}Start assembled setup below. Hand detection and wiggle require six separate motor addresses.`}
            </p>
          </div>
        )}
        {(!mappingComplete || reidentify) && jointGuide}
        {followerPort}
        {(!mappingComplete || reidentify) && session?.mode !== 'setup' && (
          <fieldset className="robot-lab-identify-method">
            <legend>Identification method</legend>
            <button
              type="button"
              className={identifyMethod === 'hand' ? 'primary-button' : 'secondary-button'}
              aria-pressed={identifyMethod === 'hand'}
              onClick={() => {
                setIdentifyMethod('hand');
                setWiggleReady(false);
              }}
            >
              Move by hand
            </button>
            <button
              type="button"
              className={identifyMethod === 'wiggle' ? 'primary-button' : 'secondary-button'}
              aria-pressed={identifyMethod === 'wiggle'}
              onClick={() => setIdentifyMethod('wiggle')}
            >
              Wiggle a motor
            </button>
          </fieldset>
        )}
        {(!mappingComplete || reidentify) && (
          <section className="robot-lab-assembled" aria-label="Experimental assembled setup">
            <div className="robot-lab-assembled-heading">
              <h3>Assembled setup</h3>
              <span className="robot-lab-experimental">Experimental</span>
            </div>
            {(!assembled || ['idle', 'stopped', 'error', 'complete'].includes(assembled.phase)) &&
              session?.mode !== 'setup' && (
                <div className="robot-lab-form">
                  <Notice kind="warning">
                    Support the arm. Keep power, USB and motor cables connected. Experimental setup reboots motors and
                    writes temporary IDs; torque can release.
                  </Notice>
                  {scan?.ready && (
                    <p className="robot-lab-subtle">
                      IDs 1–6 respond. Identification confirms which joint owns each ID.
                    </p>
                  )}
                  <label className="robot-lab-check">
                    <input
                      type="checkbox"
                      checked={assembledReady}
                      disabled={busy || active}
                      onChange={(event) => setAssembledReady(event.target.checked)}
                    />
                    <span>Six STS3215 motors; arm supported. Allow motor reboots and persistent ID changes.</span>
                  </label>
                  <div className="robot-lab-actions">
                    <button
                      className="primary-button"
                      disabled={!canUseArm || active || !assembledReady || !state?.capabilities?.assembled_setup}
                      onClick={() => {
                        setAssembledReady(false);
                        setReidentify(false);
                        void perform(
                          'Experimental assembled setup started.',
                          () => robotApi.startAssembled(port),
                          ownSession,
                          false,
                        );
                      }}
                    >
                      Start assembled setup
                    </button>
                    <button
                      className="secondary-button"
                      disabled={!canUseArm || active}
                      onClick={() =>
                        void perform(
                          'Motor connection checked.',
                          () => robotApi.probe(port),
                          (result) => setProbe(result),
                        )
                      }
                    >
                      Verify current IDs
                    </button>
                  </div>
                </div>
              )}
            {assembled && assembled.phase !== 'idle' && (
              <div className="robot-lab-form">
                <div className={`robot-lab-assembled-state phase-${assembled.phase}`} role="status">
                  <span className="robot-lab-mini-label">{assembled.phase.toUpperCase()}</span>
                  <p>{assembled.message}</p>
                  {assembled.ids.length > 0 && (
                    <small>
                      {assembled.ids.length} distinct {assembled.ids.length === 1 ? 'address' : 'addresses'}:{' '}
                      {assembled.ids.join(', ')}
                    </small>
                  )}
                </div>
                {assembled.error && <HardwareNotice message={assembled.error} />}
                {session?.mode === 'setup' && ['ready', 'observing', 'review'].includes(assembled.phase) && (
                  <>
                    <fieldset className="robot-lab-identify-method" disabled={busy || assembled.motion_active}>
                      <legend>Identification method</legend>
                      <button
                        type="button"
                        className={identifyMethod === 'hand' ? 'primary-button' : 'secondary-button'}
                        aria-pressed={identifyMethod === 'hand'}
                        onClick={() => {
                          setIdentifyMethod('hand');
                          setWiggleReady(false);
                        }}
                      >
                        Move by hand
                      </button>
                      <button
                        type="button"
                        className={identifyMethod === 'wiggle' ? 'primary-button' : 'secondary-button'}
                        aria-pressed={identifyMethod === 'wiggle'}
                        onClick={() => setIdentifyMethod('wiggle')}
                      >
                        Wiggle a motor
                      </button>
                    </fieldset>
                    {identifyMethod === 'hand' ? (
                      <div className="robot-lab-actions">
                        <button
                          className="primary-button"
                          disabled={!canAct || !ownsSession || assembled.mapping[selectedMotor.name] !== undefined}
                          onClick={() => selectIllustratedJoint(selectedMotor.name)}
                        >
                          {assembled.phase === 'observing' ? 'Restart movement detection' : 'Start movement detection'}
                        </button>
                      </div>
                    ) : (
                      <div className="robot-lab-form">
                        <label>
                          Motor address
                          <select
                            aria-label="Motor address to identify"
                            value={wiggleId || String(assembled.ids.find((id) => !mappingValues.includes(id)) ?? '')}
                            onChange={(event) => {
                              setWiggleId(event.target.value);
                              setWiggleReady(false);
                            }}
                          >
                            {assembled.ids
                              .filter((id) => !mappingValues.includes(id))
                              .map((id) => (
                                <option value={id} key={id}>
                                  Address {id}
                                </option>
                              ))}
                          </select>
                        </label>
                        <label className="robot-lab-check">
                          <input
                            type="checkbox"
                            checked={wiggleReady}
                            onChange={(event) => setWiggleReady(event.target.checked)}
                          />
                          <span>
                            Arm supported; every joint clear of its stops. Allow a brief ±2° motion at low torque.
                          </span>
                        </label>
                        <button
                          className="primary-button"
                          disabled={!canAct || !ownsSession || !wiggleReady || allJointsMapped}
                          onClick={() => {
                            const id = Number(wiggleId || assembled.ids.find((id) => !mappingValues.includes(id)));
                            setWiggleReady(false);
                            void perform(
                              `Address ${id} identification requested.`,
                              () => robotApi.wiggleAssembled(id),
                              undefined,
                              false,
                            );
                          }}
                        >
                          Wiggle selected motor
                        </button>
                        {assembled.wiggle_id != null && (
                          <div className="robot-lab-form">
                            <p>Address {assembled.wiggle_id}: select the joint that moved on your arm.</p>
                            <button
                              className="secondary-button"
                              disabled={!canAct || !ownsSession || assembled.mapping[selectedMotor.name] !== undefined}
                              onClick={() =>
                                void perform(
                                  'Joint address confirmed.',
                                  () => robotApi.acceptWiggle(selectedMotor.name, assembled.wiggle_id!),
                                  () => {
                                    setWiggleId('');
                                    const next = setupOrder.find(
                                      (joint) => joint !== selectedMotor.name && assembled.mapping[joint] === undefined,
                                    );
                                    if (next) setMotorIndex(setupOrder.indexOf(next));
                                  },
                                )
                              }
                            >
                              Assign address {assembled.wiggle_id} to {selectedMotor.label.toLowerCase()}
                            </button>
                          </div>
                        )}
                      </div>
                    )}

                    <div className="robot-lab-table-wrap">
                      <table className="robot-lab-table">
                        <thead>
                          <tr>
                            <th>Joint to identify</th>
                            <th>Detected address</th>
                            <th>Final ID</th>
                          </tr>
                        </thead>
                        <tbody>
                          {robotJoints.map((joint) => (
                            <tr key={joint.name}>
                              <td>
                                <button
                                  type="button"
                                  className="text-button"
                                  onClick={() => selectIllustratedJoint(joint.name)}
                                >
                                  {joint.label}
                                </button>
                              </td>
                              <td>{assembled.mapping[joint.name] ?? 'Not confirmed'}</td>
                              <td>{joint.id}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    {allJointsMapped && (
                      <>
                        <Notice kind="warning">
                          Review all six matches. Saving writes IDs 1–6 and requires recalibration.
                        </Notice>
                        <button
                          className="primary-button"
                          disabled={!canAct || !ownsSession || assembled.phase !== 'review'}
                          onClick={() =>
                            void perform('Motor ID save requested.', robotApi.finishAssembled, (result) => {
                              if (result.session?.mode === 'idle') {
                                ownedRef.current = false;
                                setOwnsSession(false);
                              }
                              setProbe(null);
                            })
                          }
                        >
                          Save motor IDs 1–6
                        </button>
                      </>
                    )}
                  </>
                )}
                {session?.mode === 'setup' && (
                  <button
                    className="secondary-button"
                    disabled={busy}
                    onClick={() =>
                      void perform('Assembled setup stopped.', robotApi.cancelAssembled, () => {
                        ownedRef.current = false;
                        setOwnsSession(false);
                      })
                    }
                  >
                    Stop assembled setup
                  </button>
                )}
                {assembled.phase === 'complete' && (
                  <Notice kind="success">
                    <strong>IDs saved</strong>Calibration required.
                    <button className="text-button" disabled={busy} onClick={() => setStep(2)}>
                      Continue to calibration
                    </button>
                  </Notice>
                )}
                {assembled.log.length > 0 && (
                  <details className="robot-lab-setup-log">
                    <summary>Setup details</summary>
                    <ol>
                      {assembled.log.map((entry, index) => (
                        <li key={index}>{entry}</li>
                      ))}
                    </ol>
                  </details>
                )}
              </div>
            )}
          </section>
        )}
        <details className="robot-lab-advanced-ids">
          <summary>Advanced: configure one physically isolated motor</summary>
          <div className="robot-lab-form">
            <div className="robot-lab-motor-list">
              {setupOrder.map((name, index) => {
                const joint = robotJoints.find((item) => item.name === name)!;
                return (
                  <button
                    key={name}
                    className={`robot-lab-motor${motorIndex === index ? ' selected' : ''}${assigned.includes(name) ? ' assigned' : ''}`}
                    onClick={() => setMotorIndex(index)}
                    disabled={busy || active}
                    aria-pressed={motorIndex === index}
                  >
                    <b>{assigned.includes(name) ? <Icon name="check" size={12} /> : joint.id}</b>
                    <span>
                      <strong>{joint.label}</strong>
                      <small>{assigned.includes(name) ? 'Assigned this session' : `Motor ID ${joint.id}`}</small>
                    </span>
                  </button>
                );
              })}
            </div>
            <div className="robot-lab-motor-guide">
              <span className="robot-lab-motor-number">{selectedMotor.id}</span>
              <div>
                <h3>{selectedMotor.label}</h3>
                <p>
                  Power off the arm. Connect only this motor directly to the controller, with the rest of the chain
                  disconnected. Restore motor power and keep USB connected.
                </p>
              </div>
            </div>
            <label className="robot-lab-check">
              <input
                type="checkbox"
                checked={isolated}
                disabled={busy || active}
                onChange={(event) => setIsolated(event.target.checked)}
              />
              <span>
                <strong>Only the {selectedMotor.label.toLowerCase()} motor is connected.</strong>I have physically
                disconnected every other motor from the controller.
              </span>
            </label>
            <div className="robot-lab-actions">
              <button
                className="primary-button"
                disabled={!canUseArm || active || !isolated || !state?.capabilities?.motor_setup}
                onClick={() =>
                  void perform(
                    `${selectedMotor.label} assigned ID ${selectedMotor.id}.`,
                    () => robotApi.assignMotor(port, selectedMotor.name),
                    (result) => {
                      setAssigned((previous) => [...new Set([...previous, result.assigned.joint])]);
                      setIsolated(false);
                      if (motorIndex < 5) setMotorIndex(motorIndex + 1);
                    },
                  )
                }
              >
                {busy ? 'Assigning motor ID…' : `Assign ID ${selectedMotor.id}`}
              </button>
              <button className="secondary-button" disabled={busy || active} onClick={() => setStep(2)}>
                IDs already set? Calibrate
                <Icon name="arrow" size={14} />
              </button>
            </div>
            <Notice>
              <strong>Reconnect the full arm when all six IDs are set.</strong>Switch motor power off before changing
              cables. A scan cannot distinguish several motors sharing the same factory ID; this step needs one physical
              motor at a time.
            </Notice>
          </div>
        </details>
      </div>
    </>
  );
}
