'use client';

import { useCallback, useEffect, useState } from 'react';
import { apiOrigin } from '@/lib/api';
import { agentModelName } from '@/lib/agent-models';
import { Heading, Notice } from './robot-lab-shared';
import type { RobotLabState } from './robot-lab-state';
import { ChoiceCards, Segmented } from './ui/choice';

type Run = {
  id: string;
  status: string;
  frames: number;
  error: string | null;
  config: { mode: string; task: string; model?: string };
  observation_id?: string;
  proposal: { command_id: string; tool: string; arguments: unknown } | null;
  dataset: { id: string } | null;
};
const terminal = new Set(['saved', 'stopped', 'failed', 'interrupted']);
// Models known to read images and call tools; any other model ID can be typed in.
const models = ['gpt-6.1-sol', 'gpt-6-astra', 'gpt-6.1-terra', 'gpt-6-luna'];
const modelKey = 'openjensen.robot-lab.collection-model';
const statusWords: Record<string, string> = {
  starting: 'Starting',
  running: 'Running',
  awaiting_approval: 'Waiting for your approval',
  finalizing: 'Saving',
  saved: 'Saved',
  stopped: 'Stopped',
  failed: 'Failed',
  interrupted: 'Interrupted',
};

export function RobotLabCollection({ lab }: { lab: RobotLabState }) {
  const base = `${apiOrigin}/api/v1/projects/${lab.projectId}/robot-lab/collection-runs`;
  const [runs, setRuns] = useState<Run[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [mode, setMode] = useState<'mock' | 'physical'>('mock');
  const [model, setModel] = useState('gpt-6-astra');
  const [otherModel, setOtherModel] = useState(false);
  const [task, setTask] = useState('Place one object into the tray');
  const [fixture, setFixture] = useState('');
  const [criterion, setCriterion] = useState('');
  const [wrist, setWrist] = useState(0);
  const [side, setSide] = useState(1);
  const [present, setPresent] = useState(false);
  const [clear, setClear] = useState(false);
  const [fps, setFps] = useState(10);
  const [seconds, setSeconds] = useState(120);
  const [steps, setSteps] = useState(16);
  const [tokens, setTokens] = useState(16000);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const active = runs.find((run) => !terminal.has(run.status));
  const current = runs.find((run) => run.id === selected) ?? active ?? runs[0];
  const request = useCallback(
    async (suffix = '', body?: unknown) => {
      const response = await fetch(base + suffix, {
        method: body === undefined ? 'GET' : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: body === undefined ? undefined : JSON.stringify(body),
      });
      const value = await response.json();
      if (!response.ok) throw new Error(value.detail ?? 'Collection request failed');
      return value;
    },
    [base],
  );
  const refresh = useCallback(async () => {
    if (lab.projectId) setRuns(await request());
  }, [lab.projectId, request]);
  useEffect(() => {
    try {
      const saved = localStorage.getItem(modelKey);
      if (saved && /^[A-Za-z0-9._:-]{1,128}$/.test(saved)) {
        setModel(saved);
        setOtherModel(!models.includes(saved));
      }
    } catch {
      /* The remembered model is a convenience. */
    }
  }, []);
  useEffect(() => {
    const initial = setTimeout(() => void refresh().catch((e: Error) => setError(e.message)), 0);
    const timer = setInterval(() => void refresh().catch((e: Error) => setError(e.message)), 1000);
    return () => {
      clearTimeout(initial);
      clearInterval(timer);
    };
  }, [refresh]);
  useEffect(() => {
    if (!active || active.config.mode !== 'physical') return;
    const pulse = () => {
      if (document.visibilityState === 'visible')
        void request(`/${active.id}/heartbeat`, {}).catch((e: Error) => setError(e.message));
    };
    pulse();
    const timer = setInterval(pulse, 400);
    return () => clearInterval(timer);
  }, [active?.id, active?.config.mode, request]); // eslint-disable-line react-hooks/exhaustive-deps
  const cameras = lab.availableCameras;
  const wristIndex = cameras.some((camera) => camera.index === wrist) ? wrist : (cameras[0]?.index ?? 0);
  const sideIndex =
    cameras.some((camera) => camera.index === side) && side !== wristIndex
      ? side
      : (cameras.find((camera) => camera.index !== wristIndex)?.index ?? 1);
  const validModel = /^[A-Za-z0-9._:-]{1,128}$/.test(model);
  // The model takes over the arm that Teleoperation is holding under joint control.
  const armReady = lab.session?.mode === 'manual' && lab.session.torque_enabled && !lab.partialControl;
  const blocker = !lab.projectId
    ? 'Choose a project first.'
    : !task.trim()
      ? 'Say what the arm should do.'
      : mode === 'physical'
        ? !validModel
          ? 'Enter a model ID.'
          : cameras.length < 2
            ? 'Connect two cameras: one on the wrist, one beside the scene.'
            : !armReady
              ? 'Start joint control for the whole arm under Teleoperation first.'
              : !present || !clear
                ? 'Tick both boxes once you are at the arm.'
                : null
        : null;
  async function start() {
    setBusy(true);
    setError('');
    try {
      if (mode === 'physical') {
        try {
          localStorage.setItem(modelKey, model);
        } catch {
          /* The remembered model is a convenience. */
        }
      }
      const run = await request('', {
        mode,
        model: mode === 'mock' ? 'gpt-6-astra' : model,
        task,
        experiment: { configuration_id: fixture.trim() || null, success_criterion: criterion.trim() || null },
        camera_indices: mode === 'mock' ? [0, 1] : [wristIndex, sideIndex],
        fps,
        max_seconds: seconds,
        max_steps: steps,
        max_tokens: tokens,
        expected_generation: lab.session?.generation ?? 0,
        // The two ticked boxes are the operator's confirmation; the service still asks for its phrase.
        confirm: mode === 'physical' && present && clear ? 'ARM AI CONTROL' : '',
        operator_present: present,
        workspace_clear: clear,
      });
      setSelected(run.id);
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function act(action: string, body = {}) {
    if (!current) return;
    try {
      await request(`/${current.id}/${action}`, body);
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    }
  }
  const number = (label: string, value: number, set: (value: number) => void, min: number, max: number) => (
    <label>
      {label}
      <input type="number" min={min} max={max} value={value} onChange={(e) => set(Number(e.target.value))} />
    </label>
  );
  return (
    <section aria-label="AI collection">
      <Heading title="AI collection" />
      <div className="robot-lab-form">
        <p className="robot-lab-subtle">
          An AI model looks through two cameras and moves the arm one small step at a time to record an episode. On the
          real arm you approve every move.
        </p>
        {error && <Notice kind="warning">{error}</Notice>}
        <fieldset className="robot-lab-fieldset robot-lab-form" disabled={!!active || busy}>
          <div className="robot-lab-field">
            <span className="robot-lab-field-label">Where to run</span>
            <ChoiceCards
              label="Run mode"
              value={mode}
              onChange={setMode}
              options={[
                { value: 'mock', label: 'Practice run', detail: 'A pretend arm and cameras. Free; nothing moves.' },
                { value: 'physical', label: 'Real arm', detail: 'Your SO-101 and cameras. Uses your OpenAI API key.' },
              ]}
            />
          </div>
          {mode === 'physical' && (
            <div className="robot-lab-field">
              <span className="robot-lab-field-label">AI model</span>
              <Segmented
                label="Model"
                value={otherModel ? 'other' : model}
                onChange={(value) => {
                  setOtherModel(value === 'other');
                  if (value !== 'other') setModel(value);
                }}
                options={[
                  ...models.map((id) => ({ value: id, label: agentModelName(id) })),
                  { value: 'other', label: 'Other…' },
                ]}
              />
              {otherModel && (
                <input
                  aria-label="Model ID"
                  value={model}
                  placeholder="An OpenAI model that reads images, such as gpt-6.1-sol"
                  maxLength={128}
                  onChange={(e) => setModel(e.target.value.trim())}
                />
              )}
            </div>
          )}
          <label>
            Task
            <input
              value={task}
              onChange={(e) => setTask(e.target.value)}
              maxLength={1000}
              placeholder="What the arm should do"
            />
          </label>
          <details>
            <summary>Experiment settings</summary>
            <label>
              Fixture configuration ID
              <input value={fixture} maxLength={128} onChange={(e) => setFixture(e.target.value)} />
            </label>
            <label>
              Success criterion
              <input value={criterion} maxLength={500} onChange={(e) => setCriterion(e.target.value)} />
            </label>
          </details>
          {mode === 'physical' && (
            <>
              <p className="robot-lab-subtle">
                {armReady ? (
                  <>
                    Arm: <strong>{lab.session?.robot_id}</strong>, under joint control.
                  </>
                ) : (
                  <>
                    No arm is under control yet.{' '}
                    <button className="text-button" onClick={() => lab.openStep(3)}>
                      Start joint control in Teleoperation
                    </button>
                  </>
                )}
              </p>
              {cameras.length >= 2 &&
                (
                  [
                    ['Wrist camera', wristIndex, sideIndex, setWrist, setSide],
                    ['Side camera', sideIndex, wristIndex, setSide, setWrist],
                  ] as const
                ).map(([label, value, other, set, setOther]) => (
                  <div className="robot-lab-field" key={label}>
                    <span className="robot-lab-field-label">{label}</span>
                    <Segmented
                      label={label}
                      value={value}
                      onChange={(index) => {
                        // The two views need different cameras: taking the other's camera swaps them.
                        if (index === other) setOther(value);
                        set(index);
                      }}
                      options={cameras.map((camera) => ({ value: camera.index, label: lab.cameraName(camera.index) }))}
                    />
                  </div>
                ))}
              <label className="robot-lab-check">
                <input type="checkbox" checked={present} onChange={(e) => setPresent(e.target.checked)} />
                <span>
                  <strong>I am at the arm and can stop it.</strong>The model has no collision avoidance and cannot feel
                  force. Leaving this page stops it.
                </span>
              </label>
              <label className="robot-lab-check">
                <input type="checkbox" checked={clear} onChange={(e) => setClear(e.target.checked)} />
                <span>
                  <strong>The workspace is clear.</strong>
                </span>
              </label>
            </>
          )}
          <details className="robot-lab-details">
            <summary>Limits</summary>
            <div className="robot-lab-fields robot-lab-limits">
              {number('Frames per second', fps, setFps, 1, 20)}
              {number('Longest run (seconds)', seconds, setSeconds, 10, 300)}
              {number('Most moves', steps, setSteps, 1, 40)}
              {number('Token budget', tokens, setTokens, 1024, 64000)}
            </div>
          </details>
          <div className="robot-lab-run-start">
            <button
              className="primary-button robot-lab-run-button"
              onClick={() => void start()}
              disabled={!!blocker || !!active || busy}
            >
              {busy ? 'Starting…' : mode === 'mock' ? 'Start practice run' : 'Start collection'}
            </button>
            {blocker && !active && (
              <p className="robot-lab-subtle" role="status">
                {blocker}
              </p>
            )}
          </div>
        </fieldset>
        {active && (
          <button
            className="danger-button robot-lab-stop"
            onClick={() =>
              void request(`/${active.id}/stop`, {})
                .then(refresh)
                .catch((e: Error) => setError(e.message))
            }
          >
            <span />
            Stop collection
          </button>
        )}
        {current && (
          <section aria-label="Collection progress" className="robot-lab-collection robot-lab-form">
            <div className="robot-lab-mapping-heading">
              <h3>
                {current.config.mode === 'mock' ? 'Practice run' : 'Real arm'} ·{' '}
                {statusWords[current.status] ?? current.status}
              </h3>
              <span>{current.frames} frames</span>
            </div>
            <p className="robot-lab-subtle">
              {current.config.task}
              {current.config.mode !== 'mock' && current.config.model
                ? ` · ${agentModelName(current.config.model)}`
                : ''}
              . Watch the episode before trusting it: nobody has checked that the task succeeded.
            </p>
            {current.error && <Notice kind="warning">{current.error}</Notice>}
            {current.observation_id && (
              <div className="robot-lab-collection-views">
                {['wrist', 'side'].map((role) => (
                  <figure key={role}>
                    <img
                      alt={`${role} model observation`}
                      src={`${base}/${current.id}/image/${role}?observation=${current.observation_id}`}
                    />
                    <figcaption>{role === 'wrist' ? 'Wrist' : 'Side'} · what the model last saw</figcaption>
                  </figure>
                ))}
              </div>
            )}
            {current.proposal && (
              <div role="region" aria-label="Motion approval" className="robot-lab-form">
                <h4>The model wants to {current.proposal.tool === 'set_gripper' ? 'move the gripper' : 'move'}</h4>
                <pre className="robot-lab-collection-proposal">
                  {JSON.stringify(current.proposal.arguments, null, 2)}
                </pre>
                <div className="robot-lab-actions">
                  <button
                    className="primary-button"
                    onClick={() => void act('approve', { command_id: current.proposal?.command_id })}
                  >
                    Approve this move
                  </button>
                  <button className="danger-button" onClick={() => void act('stop')}>
                    Reject and stop
                  </button>
                </div>
              </div>
            )}
            <div className="robot-lab-actions">
              <a className="text-link" href={`${base}/${current.id}/trace`} target="_blank" rel="noreferrer">
                See every step the model took
              </a>
              {current.dataset && (
                <a className="text-link" href={`${apiOrigin}/api/v1/datasets/${current.dataset.id}/download`}>
                  Download the dataset
                </a>
              )}
            </div>
            {current.dataset && <p className="robot-lab-subtle">Saved in this project’s datasets.</p>}
          </section>
        )}
        {runs.length > 0 && (
          <div className="robot-lab-field">
            <span className="robot-lab-field-label">Earlier runs</span>
            <ChoiceCards
              label="Saved runs"
              value={current?.id ?? null}
              onChange={setSelected}
              options={runs.map((run) => ({
                value: run.id,
                label: run.config.task,
                detail: `${run.config.mode === 'mock' ? 'Practice run' : 'Real arm'} · ${run.frames} frames`,
                badge: statusWords[run.status] ?? run.status,
              }))}
            />
          </div>
        )}
      </div>
    </section>
  );
}
