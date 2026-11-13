'use client';

import { useCallback, useEffect, useState } from 'react';
import { apiOrigin } from '@/lib/api';
import type { RobotLabState } from './robot-lab-state';

type Run = {
  id: string;
  status: string;
  frames: number;
  error: string | null;
  config: { mode: string; task: string };
  observation_id?: string;
  proposal: { command_id: string; tool: string; arguments: unknown } | null;
  dataset: { id: string } | null;
};
const terminal = new Set(['saved', 'stopped', 'failed', 'interrupted']);

export function RobotLabCollection({ lab }: { lab: RobotLabState }) {
  const base = `${apiOrigin}/api/v1/projects/${lab.projectId}/robot-lab/collection-runs`;
  const [runs, setRuns] = useState<Run[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [mode, setMode] = useState('mock');
  const [task, setTask] = useState('Place one object into the tray');
  const [wrist, setWrist] = useState(0);
  const [side, setSide] = useState(1);
  const [confirm, setConfirm] = useState('');
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
  async function start() {
    setBusy(true);
    setError('');
    try {
      const run = await request('', {
        mode,
        model: 'gpt-6-astra',
        task,
        camera_indices: mode === 'mock' ? [0, 1] : [wrist, side],
        fps,
        max_seconds: seconds,
        max_steps: steps,
        max_tokens: tokens,
        expected_generation: lab.session?.generation ?? 0,
        confirm,
        operator_present: present,
        workspace_clear: clear,
      });
      setSelected(run.id);
      setConfirm('');
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
  return (
    <section aria-label="Astra collection" className="robot-lab-panel">
      <h2>Astra collection</h2>
      <p>One supervised episode, two camera views, robot actions and an auditable trace.</p>
      {!lab.projectId && <p role="alert">Choose a project before collecting.</p>}
      {error && <p role="alert">{error}</p>}
      <fieldset disabled={!!active || busy} style={{ display: 'grid', gap: 12, border: 0, padding: 0 }}>
        <label>
          Run mode{' '}
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="mock">Dry run · fake robot and cameras · no API cost</option>
            <option value="physical">SO101 · Astra via Pi</option>
          </select>
        </label>
        <label>
          Model <input value="gpt-6-astra" readOnly />
        </label>
        <label>
          Task <input value={task} onChange={(e) => setTask(e.target.value)} maxLength={1000} />
        </label>
        {mode === 'physical' && (
          <>
            <p>
              Connected arm: <strong>{lab.session?.robot_id ?? 'none'}</strong>. Enable full-arm joint control in
              Teleoperation first. Uses the OpenAI API key saved in Settings.
            </p>
            {(['Wrist camera', 'Side camera'] as const).map((label, i) => (
              <label key={label}>
                {label}
                <select
                  value={i === 0 ? wrist : side}
                  onChange={(e) => (i === 0 ? setWrist : setSide)(Number(e.target.value))}
                >
                  {lab.availableCameras.map((camera) => (
                    <option key={camera.index} value={camera.index}>
                      {camera.name} · {camera.index}
                    </option>
                  ))}
                </select>
              </label>
            ))}
            <p>
              Joint segments have no collision avoidance or force sensing. Keep the workspace clear and the physical
              stop within reach. Every proposed move needs your approval. Leaving this page stops control.
            </p>
            <label>
              <input type="checkbox" checked={present} onChange={(e) => setPresent(e.target.checked)} /> I am present
              and ready to stop the arm
            </label>
            <label>
              <input type="checkbox" checked={clear} onChange={(e) => setClear(e.target.checked)} /> The workspace is
              clear
            </label>
            <label>
              Type ARM AI CONTROL <input value={confirm} onChange={(e) => setConfirm(e.target.value)} />
            </label>
          </>
        )}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
          <label>
            FPS <input type="number" min={1} max={20} value={fps} onChange={(e) => setFps(Number(e.target.value))} />
          </label>
          <label>
            Seconds{' '}
            <input
              type="number"
              min={10}
              max={300}
              value={seconds}
              onChange={(e) => setSeconds(Number(e.target.value))}
            />
          </label>
          <label>
            Steps{' '}
            <input type="number" min={1} max={40} value={steps} onChange={(e) => setSteps(Number(e.target.value))} />
          </label>
          <label>
            Token budget{' '}
            <input
              type="number"
              min={1024}
              max={64000}
              value={tokens}
              onChange={(e) => setTokens(Number(e.target.value))}
            />
          </label>
        </div>
        <button
          className="primary-button"
          onClick={() => void start()}
          disabled={!lab.projectId || (mode === 'physical' && (!present || !clear || confirm !== 'ARM AI CONTROL'))}
        >
          Start collection
        </button>
      </fieldset>
      {active && (
        <button
          className="danger-button"
          onClick={() =>
            void request(`/${active.id}/stop`, {})
              .then(refresh)
              .catch((e: Error) => setError(e.message))
          }
        >
          Stop collection
        </button>
      )}
      {current && (
        <section aria-label="Collection progress">
          <h3>
            {current.config.mode === 'mock' ? 'Dry run' : 'SO101 episode'} · {current.status}
          </h3>
          <p>{current.frames} paired frames · Task success needs human review</p>
          {current.error && <p role="alert">{current.error}</p>}
          {current.observation_id && (
            <div style={{ display: 'flex', gap: 8 }}>
              {['wrist', 'side'].map((role) => (
                <figure key={role} style={{ margin: 0, width: '48%' }}>
                  <img
                    alt={`${role} model observation`}
                    src={`${base}/${current.id}/image/${role}?observation=${current.observation_id}`}
                    style={{ width: '100%' }}
                  />
                  <figcaption>{role} · last model observation</figcaption>
                </figure>
              ))}
            </div>
          )}
          {current.proposal && (
            <div role="region" aria-label="Motion approval">
              <h4>Review proposed {current.proposal.tool}</h4>
              <pre>{JSON.stringify(current.proposal.arguments, null, 2)}</pre>
              <button
                className="primary-button"
                onClick={() => void act('approve', { command_id: current.proposal?.command_id })}
              >
                Approve this segment
              </button>
              <button className="danger-button" onClick={() => void act('stop')}>
                Reject and stop
              </button>
            </div>
          )}
          <p>
            <a href={`${base}/${current.id}/trace`} target="_blank" rel="noreferrer">
              Inspect trace
            </a>
            {current.dataset && (
              <>
                {' '}
                ·{' '}
                <a href={`${apiOrigin}/api/v1/datasets/${current.dataset.id}/download`}>
                  Export LeRobot dataset + raw evidence
                </a>{' '}
                · Saved in this project’s dataset library for review and replay.
              </>
            )}
          </p>
        </section>
      )}
      <h3>Saved runs</h3>
      {runs.map((run) => (
        <button
          key={run.id}
          className="secondary-button"
          onClick={() => setSelected(run.id)}
          style={{ display: 'block', marginBottom: 8 }}
        >
          {run.config.task} · {run.status} · {run.frames} frames
        </button>
      ))}
    </section>
  );
}
