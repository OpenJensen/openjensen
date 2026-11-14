'use client';

import { useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { robotApi, robotJoints, type RobotStatus } from '@/lib/robot-api';
import { policyCameras, viewName } from '@/lib/policy-cameras';
import { RobotLabControlDevices } from './robot-lab-devices';
import { RobotLabPolicyPreview } from './robot-lab-policy-preview';
import { RobotLabPolicyInstall } from './robot-lab-policy-install';
import { Heading, Notice, readable, statusKey } from './robot-lab-shared';
import type { RobotLabState } from './robot-lab-state';
import { ChoiceCards, Segmented } from './ui/choice';

// One press of Start walks these in order; the list shows where it is and where it stopped.
const stages = [
  { id: 'load', title: 'Load the policy' },
  { id: 'check', title: 'Check the cameras and speed' },
  { id: 'hold', title: 'Hold the arm where it is' },
  { id: 'run', title: 'Run' },
] as const;
type Stage = (typeof stages)[number]['id'];
const lengths = [10, 30, 60, 120, 300, 600];
const lengthLabel = (seconds: number) => (seconds < 60 ? `${seconds} s` : `${seconds / 60} min`);
const countdown = (seconds: number) =>
  seconds < 60 ? `${seconds} s` : `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
const taskKey = 'openjensen.robot-lab.policy-tasks';

function savedTasks(): Record<string, string> {
  try {
    const value = JSON.parse(localStorage.getItem(taskKey) ?? '{}');
    return value && typeof value === 'object' ? value : {};
  } catch {
    return {};
  }
}

export function RobotLabPolicy({ lab }: { lab: RobotLabState }) {
  const [selected, setSelected] = useState('');
  const [duration, setDuration] = useState(30);
  const [attending, setAttending] = useState(false);
  const [viewCameras, setViewCameras] = useState<Record<string, number>>({});
  const [stage, setStage] = useState<Stage | null>(null);
  const [failure, setFailure] = useState<{ stage: Stage; message: string } | null>(null);
  // Set when the check finds a policy that plans slowly: the run waits for one more yes.
  const [pauseOffer, setPauseOffer] = useState<number | null>(null);
  const [finishAt, setFinishAt] = useState<number | null>(null);
  const [now, setNow] = useState(0);
  const sequence = useRef(0);
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
  const maxDuration = policy?.limits?.max_run_seconds ?? 120;
  // The service decides the longest run; a choice above it falls back to the longest offered.
  const offered = lengths.filter((seconds) => seconds <= maxDuration);
  const runSeconds = offered.includes(duration) ? duration : (offered.at(-1) ?? Math.max(1, Math.floor(maxDuration)));
  const offline = choice?.offline_only ?? false;
  const requiresTask = choice?.task_required ?? false;
  const taskReady = !requiresTask || !!lab.task.trim();
  // A policy needs a camera of this computer for each view it was trained on.
  const views = choice?.camera_keys ?? (choice?.camera_key ? [choice.camera_key] : []);
  const primaryView = choice?.camera_key ?? views[0];
  const cameras =
    views.length > 0 && primaryView
      ? policyCameras(
          views,
          primaryView,
          lab.cameraIndex,
          lab.availableCameras.map((camera) => camera.index),
          viewCameras,
        )
      : null;
  const camerasReady = views.length > 0 && cameras !== null;
  const pauseSeconds = policy?.limits?.planning_pause_seconds ?? 3;
  const starting = stage !== null && !running;
  const typeName = (type?: string | null) => (type === 'smolvla' ? 'SmolVLA' : 'ACT');

  // The preview beside this form shows what each of the policy's views sees.
  const viewList = cameras ? views.map((view) => `${view}:${cameras[view]}`).join(',') : '';
  const { setPolicyViews } = lab;
  useEffect(() => {
    setPolicyViews(
      viewList
        ? viewList.split(',').map((entry) => {
            const split = entry.lastIndexOf(':');
            return { view: entry.slice(0, split), index: Number(entry.slice(split + 1)) };
          })
        : [],
    );
    return () => setPolicyViews([]);
  }, [viewList, setPolicyViews]);
  // A language-conditioned policy is run with the words it was last given.
  const { setTask } = lab;
  useEffect(() => {
    if (!selectedId || !requiresTask) return;
    const saved = savedTasks()[selectedId];
    if (saved) setTask(saved);
  }, [selectedId, requiresTask, setTask]);
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(timer);
  }, [running]);
  useEffect(() => {
    if (!running && stage === 'run' && !lab.busy) setStage(null);
  }, [running, stage, lab.busy]);

  function chooseCamera(view: string, index: number) {
    if (!cameras) return;
    const next = { ...cameras, [view]: index };
    // Two views cannot share a camera: taking another view's camera swaps the two.
    const other = Object.keys(cameras).find((key) => key !== view && cameras[key] === index);
    if (other) next[other] = cameras[view];
    if (primaryView && next[primaryView] !== lab.cameraIndex) lab.setCameraIndex(next[primaryView]);
    setViewCameras(next);
  }

  function start(allowPause: boolean) {
    if (!choice || !cameras) return;
    const run = ++sequence.current;
    const task = requiresTask ? lab.task.trim() : undefined;
    setFailure(null);
    setPauseOffer(null);
    if (task) {
      try {
        localStorage.setItem(taskKey, JSON.stringify({ ...savedTasks(), [choice.id]: task }));
      } catch {
        /* Remembering the instruction is a convenience. */
      }
    }
    void lab.perform(
      'Policy running. Stay with the arm.',
      async () => {
        const token = lab.epoch.current;
        // Stop, or anything else started meanwhile, ends this sequence without another motor command.
        const current = () => lab.epoch.current === token && sequence.current === run;
        const abandon = () => {
          if (sequence.current === run) setStage(null);
          return undefined;
        };
        let at: Stage = 'load';
        let holding = false;
        let status: RobotStatus | undefined;
        try {
          if (!(policy?.phase === 'ready' && policy.policy_id === choice.id)) {
            setStage('load');
            status = await robotApi.loadPolicy(choice.id);
            if (!current()) return abandon();
            if (status.policy?.phase !== 'ready' || status.policy.policy_id !== choice.id)
              throw new Error(status.policy?.error ?? 'The policy did not load.');
          }
          at = 'check';
          setStage('check');
          status = await robotApi.preflightPolicy(
            lab.cameraIndex,
            task,
            views.length > 1 ? cameras : undefined,
            allowPause,
          );
          if (!current()) return abandon();
          const check = status.policy?.preflight;
          if (!check?.passed) {
            if (check?.planning_pause_needed && !allowPause) {
              lab.client.setQueryData(statusKey, status);
              setPauseOffer(check.max_latency_ms);
              setStage(null);
              return undefined;
            }
            throw new Error(check?.message ?? 'The camera and speed check did not pass.');
          }
          at = 'hold';
          setStage('hold');
          status = await robotApi.startControl({
            port: lab.port,
            robot_id: lab.robotId,
            mode: 'manual',
            joints: robotJoints.map((joint) => joint.name),
            move_to_middle: false,
          });
          holding = true;
          if (!current()) {
            void robotApi.stop().catch(() => undefined);
            return abandon();
          }
          lab.client.setQueryData(statusKey, status);
          lab.ownSession();
          const generation = status.session?.generation;
          if (generation === undefined) throw new Error('The arm did not report a control session.');
          at = 'run';
          setStage('run');
          status = await robotApi.startPolicy({
            expected_generation: generation,
            camera_index: lab.cameraIndex,
            ...(views.length > 1 ? { cameras } : {}),
            ...(allowPause ? { allow_planning_pause: true } : {}),
            duration_seconds: runSeconds,
            ...(task ? { task } : {}),
          });
          if (!current()) {
            void robotApi.stop().catch(() => undefined);
            return abandon();
          }
          setFinishAt(Date.now() + runSeconds * 1000);
          setNow(Date.now());
          return status;
        } catch (error) {
          if (!current()) return abandon();
          setFailure({ stage: at, message: readable(error) });
          setStage(null);
          // A run that did not start must not leave the motors powered and unattended.
          if (holding) lab.stop();
          return undefined;
        }
      },
      undefined,
      false,
    );
  }

  const blocker = !selectedId
    ? 'Choose a policy.'
    : !catalog.data?.runtime.configured
      ? null
      : !lab.canUseArm
        ? lab.online
          ? 'Plug in the arm.'
          : 'The robot service is not reachable.'
        : !lab.calibrated
          ? 'Calibrate this arm under Devices first.'
          : !camerasReady
            ? views.length > 1
              ? `Connect ${views.length} cameras: this policy was trained on ${views.length} views.`
              : 'Connect a camera.'
            : !taskReady
              ? 'Type the task: what the arm should do, in the words used while recording.'
              : lab.recording?.active
                ? 'Finish the recording that is open.'
                : !attending
                  ? 'Tick the box above once you are at the arm.'
                  : null;
  const canStart = !!choice?.compatible && !!catalog.data?.runtime.configured && !blocker && !lab.active && !lab.busy;
  const stageIndex = running ? 3 : stage ? stages.findIndex((item) => item.id === stage) : -1;
  const failedIndex = failure ? stages.findIndex((item) => item.id === failure.stage) : -1;
  const remaining = running && finishAt ? Math.max(0, Math.ceil((finishAt - now) / 1000)) : null;

  return (
    <>
      <Heading title="Run policy" />
      <div className="robot-lab-form">
        {!lab.state?.capabilities?.run_policy ? (
          <Notice kind="warning">Update the robot control service to enable local policy execution.</Notice>
        ) : (
          <>
            <div className="robot-lab-field">
              <span className="robot-lab-field-label">Policy</span>
              {installed.length > 0 && (
                <ChoiceCards
                  label="Installed policy"
                  value={selectedId}
                  disabled={lab.busy || lab.active}
                  onChange={(id) => {
                    setSelected(id);
                    setFailure(null);
                    setPauseOffer(null);
                  }}
                  options={installed.map((item) => ({
                    value: item.id,
                    label: item.name,
                    detail: item.compatible
                      ? [
                          typeName(item.policy_type),
                          item.fps ? `${item.fps} FPS` : null,
                          `${item.camera_keys?.length ?? 1} ${(item.camera_keys?.length ?? 1) === 1 ? 'camera' : 'cameras'}`,
                          item.task_required ? 'follows a typed task' : null,
                        ]
                          .filter(Boolean)
                          .join(' · ')
                      : item.message,
                    badge: !item.compatible
                      ? 'Cannot run'
                      : item.offline_only
                        ? 'Test only'
                        : policy?.phase === 'ready' && policy.policy_id === item.id
                          ? 'Loaded'
                          : undefined,
                    disabled: !item.compatible,
                  }))}
                />
              )}
              {catalog.isPending && <p role="status">Looking for installed policies…</p>}
              {catalog.error && <Notice kind="warning">{catalog.error.message}</Notice>}
              {!catalog.isPending && !catalog.error && !installed.length && (
                <p className="robot-lab-subtle">
                  No policy is on this computer yet. Install a trained model below, or train one first.
                </p>
              )}
              <RobotLabPolicyInstall
                lab={lab}
                policies={installed}
                installed={(id) => {
                  setSelected(id);
                  void catalog.refetch();
                }}
              />
            </div>
            {catalog.data?.runtime && !catalog.data.runtime.configured && (
              <Notice kind="warning">{catalog.data.runtime.message}</Notice>
            )}
            {choice && !offline && (
              <>
                <RobotLabControlDevices lab={lab} leader={false} hideLeaders />
                {views.map((view) => (
                  <div className="robot-lab-field" key={view}>
                    <span className="robot-lab-field-label">
                      {views.length > 1 ? `Camera for the “${viewName(view)}” view` : 'Camera'}
                    </span>
                    {lab.availableCameras.length ? (
                      <Segmented
                        label={`Camera for ${viewName(view)}`}
                        value={cameras?.[view] ?? null}
                        disabled={lab.busy || lab.active || !cameras}
                        onChange={(index) => chooseCamera(view, index)}
                        options={lab.availableCameras.map((camera) => ({
                          value: camera.index,
                          label: lab.cameraName(camera.index),
                        }))}
                      />
                    ) : (
                      <p className="robot-lab-subtle">No camera found. Plug one in, then refresh devices.</p>
                    )}
                  </div>
                ))}
                {views.length > 0 && lab.availableCameras.length > 0 && (
                  <p className="robot-lab-subtle">
                    {views.length > 1
                      ? 'Each view must see what it saw while recording. Start the preview on the right to check; picking a camera another view uses swaps the two.'
                      : 'It must see what it saw while recording. Start the preview on the right to check.'}
                  </p>
                )}
                {requiresTask && (
                  <label>
                    Task
                    <textarea
                      aria-label="Task instruction"
                      value={lab.task}
                      maxLength={1000}
                      disabled={lab.busy || lab.active}
                      onChange={(event) => lab.setTask(event.target.value)}
                      placeholder="What the arm should do, in the words used while recording."
                    />
                  </label>
                )}
                <div className="robot-lab-field">
                  <span className="robot-lab-field-label">Run for</span>
                  <Segmented
                    label="Policy run duration"
                    value={runSeconds}
                    disabled={lab.busy || lab.active}
                    onChange={setDuration}
                    options={(offered.length ? offered : [runSeconds]).map((seconds) => ({
                      value: seconds,
                      label: lengthLabel(seconds),
                    }))}
                  />
                </div>
                <label className="robot-lab-check">
                  <input
                    type="checkbox"
                    checked={attending}
                    disabled={running || lab.busy}
                    onChange={(event) => setAttending(event.target.checked)}
                  />
                  <span>
                    <strong>I am at the arm and its path is clear.</strong>The arm moves slowly (10°/s). Stop, a lost
                    connection or a late prediction switches the motors off.
                  </span>
                </label>
                {pauseOffer !== null && !running && (
                  <Notice kind="warning">
                    <strong>This policy thinks for up to {Math.round(pauseOffer)} ms between moves.</strong>
                    That is longer than the usual limit. While it plans, the arm holds still; a pause longer than{' '}
                    {pauseSeconds} s still switches the motors off.
                    <div className="robot-lab-actions">
                      <button className="primary-button" disabled={!canStart} onClick={() => start(true)}>
                        Allow planning pauses and start
                      </button>
                      <button className="text-button" onClick={() => setPauseOffer(null)}>
                        Cancel
                      </button>
                    </div>
                  </Notice>
                )}
                {(starting || running || failure) && (
                  <ol className="robot-lab-run-steps" aria-label="Run progress">
                    {stages.map((item, index) => {
                      const state =
                        failedIndex === index
                          ? 'failed'
                          : failure
                            ? index < failedIndex
                              ? 'done'
                              : 'waiting'
                            : index < stageIndex
                              ? 'done'
                              : index === stageIndex
                                ? 'active'
                                : 'waiting';
                      return (
                        <li key={item.id} data-state={state}>
                          <span className="robot-lab-run-step-mark" aria-hidden="true" />
                          <span>
                            {item.title}
                            {state === 'active' && item.id !== 'run' ? '…' : ''}
                            {item.id === 'check' && state === 'done' && policy?.preflight
                              ? ` · ${policy.preflight.measured_fps} FPS`
                              : ''}
                          </span>
                        </li>
                      );
                    })}
                  </ol>
                )}
                {failure && !running && (
                  <Notice kind="warning">
                    <strong>The run did not start.</strong>
                    {failure.message}
                  </Notice>
                )}
                {running ? (
                  <div className="robot-lab-run-live" role="status">
                    <div>
                      <strong>
                        {remaining !== null
                          ? `Running · ${countdown(remaining)} left`
                          : `Running ${policy?.name ?? ''}`}
                      </strong>
                      <small>
                        {policy?.steps ?? 0} predictions · {policy?.inference_ms ?? '—'} ms each. The arm stops by
                        itself when the time is up.
                      </small>
                    </div>
                    <button className="danger-button robot-lab-stop" onClick={lab.stop}>
                      <span />
                      Stop arm
                    </button>
                  </div>
                ) : (
                  <div className="robot-lab-run-start">
                    <button
                      className="primary-button robot-lab-run-button"
                      disabled={!canStart}
                      onClick={() => start(false)}
                    >
                      {starting ? 'Starting…' : 'Start run'}
                    </button>
                    {starting && (
                      <button
                        className="secondary-button"
                        onClick={() => {
                          sequence.current += 1;
                          setStage(null);
                          lab.stop();
                        }}
                      >
                        Cancel
                      </button>
                    )}
                    {blocker && !lab.busy && !lab.active && (
                      <p className="robot-lab-subtle" role="status" aria-label="Next step">
                        {blocker}
                      </p>
                    )}
                    {lab.active && !starting && (
                      <p className="robot-lab-subtle" role="status" aria-label="Next step">
                        The arm is in use. Press Stop arm before starting a run.
                      </p>
                    )}
                  </div>
                )}
              </>
            )}
            {choice && offline && (
              <>
                <Notice>
                  This policy was installed for testing only, so it cannot move the arm. Try it below, or install the
                  model again without “Offline inference only”.
                </Notice>
                {requiresTask && (
                  <label>
                    Task
                    <textarea
                      aria-label="Task instruction"
                      value={lab.task}
                      maxLength={1000}
                      disabled={lab.busy || lab.active}
                      onChange={(event) => lab.setTask(event.target.value)}
                      placeholder="What the arm should do, in the words used while recording."
                    />
                  </label>
                )}
              </>
            )}
            {policy?.error && !failure && !lab.session?.error?.includes(policy.error) && (
              <Notice kind="warning">{policy.error}</Notice>
            )}
            {choice && (
              <details className="robot-lab-details robot-lab-policy-test" open={offline}>
                <summary>Test the policy without moving the arm</summary>
                <RobotLabPolicyPreview
                  lab={lab}
                  selectedId={selectedId}
                  requiresTask={requiresTask}
                  cameras={cameras && views.length > 1 ? cameras : undefined}
                />
              </details>
            )}
          </>
        )}
      </div>
    </>
  );
}
