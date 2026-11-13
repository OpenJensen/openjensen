'use client';

import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { robotApi, robotJoints } from '@/lib/robot-api';
import { policyCameras, sameCameras, viewName } from '@/lib/policy-cameras';
import { RobotLabPolicyPreview } from './robot-lab-policy-preview';
import { RobotLabPolicyInstall } from './robot-lab-policy-install';
import { Heading, Notice } from './robot-lab-shared';
import type { RobotLabState } from './robot-lab-state';

export function RobotLabPolicy({ lab }: { lab: RobotLabState }) {
  const [selected, setSelected] = useState('');
  const [duration, setDuration] = useState(10);
  const [attending, setAttending] = useState(false);
  const [viewCameras, setViewCameras] = useState<Record<string, number>>({});
  const [planningPause, setPlanningPause] = useState(false);
  const [localCheck, setLocalCheck] = useState<number>();
  const [expiredCheck, setExpiredCheck] = useState<number>();
  const catalog = useQuery({
    queryKey: ['robot-lab', 'policy-catalog'],
    queryFn: robotApi.policyCatalog,
    enabled: lab.online && !!lab.state?.capabilities?.run_policy,
    refetchInterval: 10_000,
  });
  const policy = lab.state?.policy ?? catalog.data?.state;
  const installed = catalog.data?.policies ?? [];
  const selectedId =
    selected ||
    installed.find((item) => item.compatible && item.id === policy?.policy_id)?.id ||
    installed.find((item) => item.compatible)?.id ||
    '';
  const choice = installed.find((item) => item.id === selectedId);
  const running = lab.session?.mode === 'policy';
  const prepared =
    lab.session?.mode === 'manual' &&
    lab.ownsSession &&
    lab.session.active_joints?.length === robotJoints.length &&
    lab.session.torque_enabled;
  const generation = lab.session?.generation;
  const maxDuration = policy?.limits?.max_run_seconds ?? 120;
  const offline = choice?.offline_only ?? policy?.offline_only ?? false;
  const requiresTask = choice?.task_required ?? policy?.task_required ?? false;
  const taskReady = !requiresTask || !!lab.task.trim();
  // A policy trained on several views needs a camera of this computer for each of them.
  const views = choice?.camera_keys ?? (choice?.camera_key ? [choice.camera_key] : []);
  const cameras =
    views.length > 1
      ? policyCameras(
          views,
          choice?.camera_key ?? views[0],
          lab.cameraIndex,
          lab.availableCameras.map((camera) => camera.index),
          viewCameras,
        )
      : undefined;
  const camerasReady = cameras !== null;
  const preflight = policy?.preflight;
  // A check counts only for the cameras, and the planning allowance, it was made with.
  const checkedTheseCameras =
    preflight?.camera_index === lab.cameraIndex &&
    (!cameras || sameCameras(preflight.cameras, cameras)) &&
    !!preflight.planning_pause === planningPause;
  const pauseSeconds = policy?.limits?.planning_pause_seconds ?? 1.5;
  const checkStamp = preflight?.checked_at;
  const remaining = preflight?.valid_for_ms;
  const checkPassed = preflight?.passed === true;
  // The companion's monotonic clock cannot be compared with the browser's clock.
  // Older companions can only be timed from a check this page just requested.
  const checkExpired =
    checkPassed &&
    (remaining === 0 ||
      (expiredCheck === checkStamp && expiredCheck !== undefined) ||
      (remaining === undefined && (checkStamp === undefined || localCheck !== checkStamp)));
  useEffect(() => {
    if (!checkPassed || checkStamp === undefined || checkExpired) return;
    const timer = window.setTimeout(() => setExpiredCheck(checkStamp), remaining ?? 60_000);
    return () => window.clearTimeout(timer);
  }, [checkPassed, checkStamp, checkExpired, remaining]);
  const loadPolicy = (
    <button
      className="secondary-button"
      disabled={!lab.canAct || lab.active || !choice?.compatible || !catalog.data?.runtime.configured}
      onClick={() => void lab.perform('Policy loaded.', () => robotApi.loadPolicy(selectedId))}
    >
      {policy?.phase === 'loading' ? 'Loading policy…' : 'Load policy'}
    </button>
  );
  return (
    <>
      <Heading title="Run policy" />
      <div className="robot-lab-form">
        {!lab.state?.capabilities?.run_policy ? (
          <Notice kind="warning">Update the robot control service to enable local policy execution.</Notice>
        ) : (
          <>
            <Notice>
              Load a compatible native policy for offline inference or an attended arm session. Choose the same camera
              view, task setup and arm calibration used for training. Stay here with Stop arm within reach.
            </Notice>
            {!offline && (
              <label>
                Follower device
                <select
                  value={lab.catalog.find((item) => item.port === lab.port && item.robot_id === lab.robotId)?.id ?? ''}
                  disabled={lab.active || lab.busy}
                  onChange={(event) => lab.selectControlDevice(event.target.value, 'follower')}
                >
                  <option value="">Choose a device</option>
                  {lab.catalog.map((device) => (
                    <option key={device.id} value={device.id} disabled={!device.connected || !device.role}>
                      {device.name || device.port || device.robot_id}
                    </option>
                  ))}
                </select>
              </label>
            )}
            <RobotLabPolicyInstall
              lab={lab}
              installed={(id) => {
                setSelected(id);
                void catalog.refetch();
              }}
            />
            <label>
              Installed policy
              <select
                value={selectedId}
                disabled={lab.busy || running}
                onChange={(event) => setSelected(event.target.value)}
              >
                <option value="">Choose a policy</option>
                {installed.map((item) => (
                  <option key={item.id} value={item.id} disabled={!item.compatible}>
                    {item.name}
                    {item.offline_only ? ' · Offline' : ''}
                  </option>
                ))}
              </select>
            </label>
            <button className="text-button" disabled={lab.busy} onClick={() => void catalog.refetch()}>
              Refresh policies
            </button>
            {catalog.isPending && <p role="status">Checking installed policies…</p>}
            {catalog.error && <Notice kind="warning">{catalog.error.message}</Notice>}
            {!catalog.isPending && !installed.length && (
              <Notice>
                No policies are installed on this computer yet. Native checkpoints must include their saved processors
                and inference compatibility metadata.
              </Notice>
            )}
            {installed
              .filter((item) => !item.compatible)
              .map((item) => (
                <Notice key={item.id} kind="warning">
                  {item.name}: {item.message}
                </Notice>
              ))}
            {choice && views.length < 2 && (
              <p className="robot-lab-subtle">
                {choice.policy_type === 'smolvla' ? 'SmolVLA' : 'ACT'} ·{' '}
                {choice.fps ? choice.fps + ' FPS' : 'FPS not declared'} · {choice.camera_key} ·{' '}
                {choice.image_shape?.slice(1).join(' × ')}. Camera selected:{' '}
                {lab.availableCameras.find((camera) => camera.index === lab.cameraIndex)?.name ?? lab.cameraIndex}.
                Select and inspect its preview before running.
              </p>
            )}
            {choice && views.length > 1 && (
              <div className="robot-lab-form" role="group" aria-label="Policy cameras">
                <p className="robot-lab-subtle">
                  {choice.policy_type === 'smolvla' ? 'SmolVLA' : 'ACT'} ·{' '}
                  {choice.fps ? choice.fps + ' FPS' : 'FPS not declared'} · trained on {views.length} camera views. Give
                  each view the camera that sees what it saw while recording.
                </p>
                <div className="robot-lab-fields">
                  {views.map((view) => (
                    <label key={view}>
                      Camera for {viewName(view)}
                      <select
                        aria-label={'Camera for ' + viewName(view)}
                        value={cameras?.[view] ?? ''}
                        disabled={lab.busy || running || lab.active}
                        onChange={(event) => {
                          const index = Number(event.target.value);
                          if (view === (choice.camera_key ?? views[0])) lab.setCameraIndex(index);
                          else setViewCameras((current) => ({ ...current, [view]: index }));
                        }}
                      >
                        {!cameras && <option value="">Choose a camera</option>}
                        {lab.availableCameras.map((camera) => (
                          <option
                            key={camera.index}
                            value={camera.index}
                            disabled={
                              !!cameras && Object.entries(cameras).some(([k, v]) => k !== view && v === camera.index)
                            }
                          >
                            {camera.name || 'Camera ' + camera.index}
                          </option>
                        ))}
                      </select>
                    </label>
                  ))}
                </div>
                {!camerasReady && (
                  <Notice kind="warning">
                    This policy needs {views.length} cameras and this computer shows {lab.availableCameras.length}.
                    Connect the cameras used for recording, then refresh devices.
                  </Notice>
                )}
              </div>
            )}
            {catalog.data?.runtime && !catalog.data.runtime.configured && (
              <Notice kind="warning">{catalog.data.runtime.message}</Notice>
            )}
            {policy?.error && <Notice kind="warning">{policy.error}</Notice>}
            {policy?.phase === 'ready' && (
              <Notice>
                Ready: {policy.name} · {policy.device}
                {policy.fps ? ' · ' + policy.fps + ' FPS' : ''}. Loading does not enable motor torque.
              </Notice>
            )}
            {requiresTask && (
              <label>
                Task instruction
                <textarea
                  value={lab.task}
                  maxLength={1000}
                  disabled={lab.busy || lab.active}
                  onChange={(event) => lab.setTask(event.target.value)}
                  placeholder="Describe what the policy should do."
                />
              </label>
            )}
            {!offline && (
              <p className="robot-lab-subtle">
                Load the policy, then check the cameras and inference rate. Confirm attendance, enable policy control
                and start the run within 60 seconds of a passing check.
              </p>
            )}
            {offline ? (
              <>
                <div className="robot-lab-actions">{loadPolicy}</div>
                <Notice>
                  Offline inference only. Physical control units and timing must be verified before an arm session.
                </Notice>
              </>
            ) : (
              <>
                {checkExpired && !running ? (
                  <Notice kind="warning">
                    Camera and inference check expired. {lab.active ? 'Stop the arm and run' : 'Run'} the check again.
                  </Notice>
                ) : preflight ? (
                  <Notice kind={preflight.passed ? 'success' : 'warning'}>
                    {preflight.message} · {preflight.measured_fps} FPS · {preflight.mean_latency_ms} ms mean latency
                    {preflight.planning_pause
                      ? ' · planning takes up to ' + Math.round(preflight.max_latency_ms) + ' ms'
                      : ''}
                    .
                  </Notice>
                ) : (
                  <p className="robot-lab-subtle">Run the camera and inference check before enabling policy control.</p>
                )}
                {(planningPause || preflight?.planning_pause_needed) && (
                  <label className="robot-lab-check">
                    <input
                      type="checkbox"
                      checked={planningPause}
                      disabled={running || lab.busy || lab.active}
                      onChange={(event) => setPlanningPause(event.target.checked)}
                    />
                    <span>
                      <strong>Allow planning pauses for this run.</strong>This policy plans a set of actions at a time
                      and needs longer than the usual limit to do it. While it plans, the arm holds its last target. A
                      pause longer than {pauseSeconds} s still disables torque, and Stop, the heartbeat and the speed
                      limit apply as always. Check again after changing this.
                    </span>
                  </label>
                )}
                <label>
                  Run duration (seconds)
                  <input
                    aria-label="Policy run duration"
                    type="number"
                    min={1}
                    max={maxDuration}
                    value={duration}
                    disabled={lab.active || lab.busy}
                    onChange={(event) => setDuration(Number(event.target.value))}
                  />
                </label>
                <label className="robot-lab-check">
                  <input
                    type="checkbox"
                    checked={attending}
                    disabled={running || lab.busy}
                    onChange={(event) => setAttending(event.target.checked)}
                  />
                  <span>
                    <strong>I am present; the arm is secured and its path is clear.</strong>Policy motion is limited to
                    10°/s (gripper 10%/s). Stop, a missing heartbeat or a stale prediction disables torque.
                  </span>
                </label>
                {lab.active && !prepared && !running && (
                  <Notice kind="warning">Stop the current session before preparing policy control.</Notice>
                )}
                <div className="robot-lab-actions" role="group" aria-label="Policy run controls">
                  {loadPolicy}
                  <button
                    className="secondary-button"
                    disabled={
                      !lab.canAct ||
                      lab.active ||
                      policy?.phase !== 'ready' ||
                      policy.policy_id !== selectedId ||
                      !lab.availableCameras.length ||
                      !camerasReady ||
                      !taskReady
                    }
                    onClick={() =>
                      void lab.perform(
                        'Policy camera and inference check completed.',
                        () =>
                          robotApi.preflightPolicy(
                            lab.cameraIndex,
                            requiresTask ? lab.task : undefined,
                            cameras ?? undefined,
                            planningPause,
                          ),
                        (result) => setLocalCheck(result.policy?.preflight?.checked_at),
                      )
                    }
                  >
                    Check camera and inference rate
                  </button>
                  {!lab.active && (
                    <button
                      className="primary-button"
                      disabled={
                        !lab.canUseArm ||
                        !lab.calibrated ||
                        !attending ||
                        !taskReady ||
                        policy?.phase !== 'ready' ||
                        policy.policy_id !== selectedId ||
                        !preflight?.passed ||
                        checkExpired ||
                        !checkedTheseCameras ||
                        !!lab.recording?.active
                      }
                      onClick={() =>
                        void lab.perform(
                          'Policy control enabled; holding the current pose.',
                          () =>
                            robotApi.startControl({
                              port: lab.port,
                              robot_id: lab.robotId,
                              mode: 'manual',
                              joints: robotJoints.map((joint) => joint.name),
                              move_to_middle: false,
                            }),
                          lab.ownSession,
                        )
                      }
                    >
                      Enable policy control
                    </button>
                  )}
                  <button
                    className="primary-button"
                    disabled={
                      !lab.canAct ||
                      !prepared ||
                      !attending ||
                      !taskReady ||
                      policy?.phase !== 'ready' ||
                      !preflight?.passed ||
                      checkExpired ||
                      !checkedTheseCameras ||
                      policy.policy_id !== selectedId ||
                      generation === undefined ||
                      !Number.isInteger(duration) ||
                      duration < 1 ||
                      duration > maxDuration ||
                      !lab.availableCameras.length
                    }
                    onClick={() => {
                      if (generation === undefined) return;
                      void lab.perform(
                        'Policy running. Stay with the arm.',
                        () =>
                          robotApi.startPolicy({
                            expected_generation: generation,
                            camera_index: lab.cameraIndex,
                            ...(cameras ? { cameras } : {}),
                            ...(planningPause ? { allow_planning_pause: true } : {}),
                            duration_seconds: duration,
                            ...(requiresTask ? { task: lab.task } : {}),
                          }),
                        lab.ownSession,
                      );
                    }}
                  >
                    Run policy
                  </button>
                  {lab.active && (
                    <button className="danger-button" onClick={lab.stop}>
                      Stop arm
                    </button>
                  )}
                </div>
                {running && (
                  <p role="status">
                    Running {policy?.name} · {policy?.steps ?? 0} predictions · {policy?.inference_ms ?? '—'} ms
                    inference. The arm stops at the end of this run.
                  </p>
                )}
              </>
            )}
            <RobotLabPolicyPreview
              lab={lab}
              selectedId={selectedId}
              requiresTask={requiresTask}
              cameras={cameras ?? undefined}
            />
          </>
        )}
      </div>
    </>
  );
}
