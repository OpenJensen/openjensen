'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { robotApi, robotJoints } from '@/lib/robot-api';
import { RobotLabPolicyPreview } from './robot-lab-policy-preview';
import { RobotLabPolicyInstall } from './robot-lab-policy-install';
import { Heading, Notice } from './robot-lab-shared';
import type { RobotLabState } from './robot-lab-state';

export function RobotLabPolicy({ lab }: { lab: RobotLabState }) {
  const [selected, setSelected] = useState('');
  const [duration, setDuration] = useState(10);
  const [attending, setAttending] = useState(false);
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
  const offline = choice?.offline_only ?? policy?.offline_only ?? false;
  const requiresTask = choice?.task_required ?? policy?.task_required ?? false;
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
            {choice && (
              <p className="robot-lab-subtle">
                {choice.policy_type === 'smolvla' ? 'SmolVLA' : 'ACT'} ·{' '}
                {choice.fps ? choice.fps + ' FPS' : 'FPS not declared'} · {choice.camera_key} ·{' '}
                {choice.image_shape?.slice(1).join(' × ')}. Camera selected:{' '}
                {lab.availableCameras.find((camera) => camera.index === lab.cameraIndex)?.name ?? lab.cameraIndex}.
                Select and inspect its preview before running.
              </p>
            )}
            {catalog.data?.runtime && !catalog.data.runtime.configured && (
              <Notice kind="warning">{catalog.data.runtime.message}</Notice>
            )}
            <div className="robot-lab-actions">
              <button
                disabled={!lab.canAct || !choice?.compatible || !catalog.data?.runtime.configured || running}
                onClick={() => void lab.perform('Policy loaded and checked.', () => robotApi.loadPolicy(selectedId))}
              >
                {policy?.phase === 'loading' ? 'Loading policy…' : 'Load and check policy'}
              </button>
              <button className="text-button" disabled={lab.busy} onClick={() => void catalog.refetch()}>
                Refresh policies
              </button>
            </div>
            {policy?.error && <Notice kind="warning">{policy.error}</Notice>}
            {policy?.phase === 'ready' && (
              <Notice>
                Ready: {policy.name} · {policy.device}
                {policy.fps ? ' · ' + policy.fps + ' FPS' : ''}. Loading does not enable motor torque.
              </Notice>
            )}
            <RobotLabPolicyPreview lab={lab} selectedId={selectedId} requiresTask={requiresTask} />
            {offline ? (
              <Notice>
                Offline inference only. Physical control units and timing must be verified before an arm session.
              </Notice>
            ) : (
              <>
                <button
                  disabled={
                    !lab.canAct ||
                    lab.active ||
                    policy?.phase !== 'ready' ||
                    policy.policy_id !== selectedId ||
                    !lab.availableCameras.length
                  }
                  onClick={() =>
                    void lab.perform('Policy camera and inference check completed.', () =>
                      robotApi.preflightPolicy(lab.cameraIndex, requiresTask ? lab.task : undefined),
                    )
                  }
                >
                  Check camera and inference rate
                </button>
                {policy?.preflight && (
                  <Notice kind={policy.preflight.passed ? 'success' : 'warning'}>
                    {policy.preflight.message} · {policy.preflight.measured_fps} FPS ·{' '}
                    {policy.preflight.mean_latency_ms} ms mean latency.
                  </Notice>
                )}
                <label>
                  Run duration (seconds)
                  <input
                    aria-label="Policy run duration"
                    type="number"
                    min={1}
                    max={120}
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
                {!lab.active && (
                  <button
                    className="primary-button"
                    disabled={
                      !lab.canUseArm ||
                      !lab.calibrated ||
                      !attending ||
                      policy?.phase !== 'ready' ||
                      !policy.preflight?.passed ||
                      policy.preflight.camera_index !== lab.cameraIndex ||
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
                {lab.active && !prepared && !running && (
                  <Notice kind="warning">Stop the current session before preparing policy control.</Notice>
                )}
                <div className="robot-lab-actions">
                  <button
                    className="primary-button"
                    disabled={
                      !lab.canAct ||
                      !prepared ||
                      !attending ||
                      policy?.phase !== 'ready' ||
                      !policy.preflight?.passed ||
                      policy.preflight.camera_index !== lab.cameraIndex ||
                      policy.policy_id !== selectedId ||
                      generation === undefined ||
                      !Number.isInteger(duration) ||
                      duration < 1 ||
                      duration > 120 ||
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
          </>
        )}
      </div>
    </>
  );
}
