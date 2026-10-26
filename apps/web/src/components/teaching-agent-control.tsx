'use client';

import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { agentApi, type AgentSession, type AgentSessionRequest } from '@/lib/agent-api';
import './workbench-form.css';
import './agent-control.css';

const terminal = (session: AgentSession) => ['interrupted', 'complete', 'stopped', 'error'].includes(session.status);
const pending = (session: AgentSession) => ['observing', 'planning', 'executing', 'stopping'].includes(session.status);
const problemText = (problem: unknown) =>
  problem instanceof Error ? problem.message : 'Agent Control request could not be verified.';
const secondsSince = (time: string, now: number) => Math.max(0, (now - Date.parse(time)) / 1000);

export function TeachingAgentControl({ projectId }: { projectId: string }) {
  const [open, setOpen] = useState(false);
  return (
    <details className="panel agent-control-panel" onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>Agent Control</summary>
      {open &&
        (projectId ? (
          <AgentControls key={projectId} projectId={projectId} />
        ) : (
          <p>Select a project to start or review an agent session.</p>
        ))}
    </details>
  );
}

function AgentControls({ projectId }: { projectId: string }) {
  const client = useQueryClient();
  const [providerId, setProviderId] = useState('mock');
  const [target, setTarget] = useState<'replay' | 'teaching'>('replay');
  const [teachingJob, setTeachingJob] = useState('');
  const [replayObservation, setReplayObservation] = useState('');
  const [mode, setMode] = useState<'advisory' | 'shadow' | 'supervised' | 'bounded'>('supervised');
  const [clock, setClock] = useState<'stepped' | 'continuous'>('stepped');
  const [task, setTask] = useState('');
  const [maxCalls, setMaxCalls] = useState(4);
  const [maxSeconds, setMaxSeconds] = useState(120);
  const [maxOutput, setMaxOutput] = useState(1024);
  const [maxTokens, setMaxTokens] = useState(16000);
  const [consent, setConsent] = useState(false);
  const [selectedId, setSelectedId] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [now, setNow] = useState(() => Date.now());
  const working = useRef(false);
  const mounted = useRef(true);
  const selectedRef = useRef(selectedId);
  selectedRef.current = selectedId;
  const providers = useQuery({ queryKey: ['agent-providers'], queryFn: agentApi.providers, retry: false });
  const history = useQuery({
    queryKey: ['agent-sessions', projectId],
    queryFn: () => agentApi.sessions(projectId),
    retry: false,
    refetchInterval: 5000,
  });
  const selected = useQuery({
    queryKey: ['agent-session', projectId, selectedId],
    queryFn: () => agentApi.session(selectedId),
    enabled: !!selectedId,
    retry: false,
    refetchInterval: (query) => (query.state.data && terminal(query.state.data) ? false : 1000),
  });
  const events = useQuery({
    queryKey: ['agent-events', projectId, selectedId],
    queryFn: () => agentApi.events(selectedId),
    enabled: !!selectedId,
    retry: false,
    refetchInterval: selected.data && terminal(selected.data) ? false : 2000,
  });
  useEffect(() => {
    mounted.current = true;
    const timer = setInterval(() => setNow(Date.now()), 500);
    return () => {
      mounted.current = false;
      clearInterval(timer);
    };
  }, []);
  const provider = providers.data?.find((item) => item.id === providerId);
  const cloud = provider?.provider === 'openai_responses';
  const session = selected.data?.project_id === projectId && selected.data.id === selectedId ? selected.data : null;
  const proposal = session?.pending_proposal;
  const proposalFresh =
    !!proposal &&
    proposal.authority_epoch === session?.authority_epoch &&
    Date.parse(proposal.expires_at) > now &&
    proposal.status === 'proposed';
  const observation = session?.observation;
  const canCreate =
    !!provider &&
    (provider.provider === 'mock' || (provider.checked && consent)) &&
    !!task.trim() &&
    (target === 'replay' || !!teachingJob.trim()) &&
    !busy &&
    !history.isError &&
    !providers.isError;
  const latestReceipt = session?.receipts?.at(-1);

  async function refresh(id = selectedRef.current) {
    await history.refetch();
    if (id) {
      await client.invalidateQueries({ queryKey: ['agent-session', projectId, id] });
      await client.invalidateQueries({ queryKey: ['agent-events', projectId, id] });
    }
  }
  async function run(action: () => Promise<AgentSession>, creating = false) {
    if (working.current) return;
    working.current = true;
    setBusy(true);
    setError('');
    const expectedId = selectedRef.current;
    try {
      const result = await action();
      if (!mounted.current) return;
      if (result.project_id !== projectId || (!creating && result.id !== expectedId))
        throw new Error('Agent session identity changed. Refresh saved sessions before continuing.');
      client.setQueryData(['agent-session', projectId, result.id], result);
      if (creating) {
        selectedRef.current = result.id;
        setSelectedId(result.id);
        setConsent(false);
      }
      await refresh(result.id);
    } catch (problem) {
      if (mounted.current) {
        setError(problemText(problem));
        await refresh();
      }
    } finally {
      working.current = false;
      if (mounted.current) setBusy(false);
    }
  }
  function create() {
    if (!canCreate) return;
    const request: AgentSessionRequest = {
      provider_id: providerId,
      task: task.trim(),
      target_kind: target,
      teaching_job_id: target === 'teaching' ? teachingJob.trim() : null,
      replay_observation_id: target === 'replay' && replayObservation.trim() ? replayObservation.trim() : null,
      mode,
      clock_policy: target === 'replay' ? 'replay' : clock,
      max_calls: maxCalls,
      max_seconds: maxSeconds,
      max_output_tokens: maxOutput,
      max_total_tokens: maxTokens,
      cloud_consent: cloud ? consent : false,
    };
    void run(() => agentApi.create(projectId, request), true);
  }
  return (
    <div className="agent-control-body">
      <p>
        Describe the intended outcome. Astra can choose only the skills enabled for the selected target and must ask for
        help when observations or calibration are missing. Replay uses a saved observation or a labeled synthetic
        fixture.
      </p>
      <form
        className="workbench-form"
        onSubmit={(event) => {
          event.preventDefault();
          create();
        }}
      >
        <label className="workbench-field">
          Task goal
          <textarea
            value={task}
            maxLength={2048}
            rows={3}
            required
            disabled={busy}
            placeholder="Pick up the object and place it in the tray"
            onChange={(event) => setTask(event.target.value)}
          />
        </label>
        <div className="workbench-field-grid">
          <label className="workbench-field">
            Agent provider profile
            <select
              value={providerId}
              disabled={busy}
              onChange={(event) => {
                setProviderId(event.target.value);
                setConsent(false);
              }}
            >
              {providers.data?.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.id} · {profile.model}
                  {profile.checked ? '' : ' · unverified'}
                </option>
              ))}
            </select>
          </label>
          <label className="workbench-field">
            Agent target
            <select
              value={target}
              disabled={busy}
              onChange={(event) => {
                setTarget(event.target.value as 'replay' | 'teaching');
                setConsent(false);
              }}
            >
              <option value="replay">Replay observation</option>
              <option value="teaching">AI teleoperation simulation</option>
            </select>
          </label>
          <label className="workbench-field">
            Agent mode
            <select value={mode} disabled={busy} onChange={(event) => setMode(event.target.value as typeof mode)}>
              <option value="advisory">Advice</option>
              <option value="shadow">Shadow proposals</option>
              <option value="supervised">Supervised acceptance</option>
              <option value="bounded">Bounded execution</option>
            </select>
          </label>
          {target === 'teaching' ? (
            <label className="workbench-field">
              Simulation clock
              <select value={clock} disabled={busy} onChange={(event) => setClock(event.target.value as typeof clock)}>
                <option value="stepped">Freeze during planning, then step</option>
                <option value="continuous">Continuously running scene</option>
              </select>
            </label>
          ) : (
            <label className="workbench-field">
              Saved observation ID (optional)
              <input
                value={replayObservation}
                maxLength={96}
                disabled={busy}
                placeholder="Blank uses the synthetic fixture"
                onChange={(event) => setReplayObservation(event.target.value)}
              />
            </label>
          )}
        </div>
        {target === 'teaching' && (
          <label className="workbench-field">
            Managed AI teleoperation job ID
            <input
              value={teachingJob}
              maxLength={96}
              required
              disabled={busy}
              placeholder="Use a running session in this project"
              onChange={(event) => setTeachingJob(event.target.value)}
            />
          </label>
        )}
        <p>
          {mode === 'bounded'
            ? 'Each planning turn may execute one valid correction within this session’s limits.'
            : mode === 'supervised'
              ? 'Review and accept each exact proposal while its observation remains valid.'
              : 'Proposals and evidence are recorded without applying motion.'}
        </p>
        <details>
          <summary>Run limits</summary>
          <div className="workbench-field-grid agent-control-body">
            <label className="workbench-field">
              Maximum model calls
              <input
                type="number"
                value={maxCalls}
                min={1}
                max={20}
                required
                disabled={busy}
                onChange={(event) => setMaxCalls(Number(event.target.value))}
              />
            </label>
            <label className="workbench-field">
              Maximum wall time (seconds)
              <input
                type="number"
                value={maxSeconds}
                min={1}
                max={600}
                required
                disabled={busy}
                onChange={(event) => setMaxSeconds(Number(event.target.value))}
              />
            </label>
            <label className="workbench-field">
              Maximum output tokens per call
              <input
                type="number"
                value={maxOutput}
                min={128}
                max={4096}
                required
                disabled={busy}
                onChange={(event) => setMaxOutput(Number(event.target.value))}
              />
            </label>
            <label className="workbench-field">
              Maximum total tokens
              <input
                type="number"
                value={maxTokens}
                min={1024}
                max={100000}
                required
                disabled={busy}
                onChange={(event) => setMaxTokens(Number(event.target.value))}
              />
            </label>
          </div>
        </details>
        {cloud && (
          <label className="workbench-check">
            <input
              type="checkbox"
              checked={consent}
              disabled={busy}
              onChange={(event) => setConsent(event.target.checked)}
            />
            Send this task, admitted camera images and measured joint state to {provider.model} through OpenAI
            Responses, within the displayed run limits. Provider charges apply.
          </label>
        )}
        {!providers.isPending && !provider && <p role="status">Configure a provider in Settings → AI providers.</p>}
        {provider && !provider.checked && provider.provider !== 'mock' && (
          <p role="status">Run a capability check in Settings → AI providers before creating this session.</p>
        )}
        {providers.error && <p role="alert">Saved provider profiles could not be checked.</p>}
        <div className="workbench-actions">
          <button type="submit" className="primary-button" disabled={!canCreate}>
            Create agent session
          </button>
          <button
            type="button"
            className="secondary-button"
            disabled={busy || history.isFetching}
            onClick={() => void refresh()}
          >
            Refresh agent sessions
          </button>
        </div>
      </form>
      <label className="workbench-field">
        Saved agent session
        <select
          value={selectedId}
          disabled={busy}
          onChange={(event) => {
            setSelectedId(event.target.value);
            setError('');
          }}
        >
          <option value="">Choose a saved session</option>
          {history.data?.map((saved) => (
            <option key={saved.id} value={saved.id}>
              {saved.id} · {saved.target_kind} · {saved.status}
            </option>
          ))}
        </select>
      </label>
      {history.error && <p role="alert">Saved sessions are unavailable. Refresh to reconcile an uncertain request.</p>}
      {selectedId && selected.isPending && <p role="status">Loading saved agent evidence…</p>}
      {selectedId && selected.error && (
        <p role="alert">This session could not be checked. Refresh before sending another action.</p>
      )}
      {session && (
        <section className="agent-control-evidence" aria-label="Agent session evidence">
          <h3>{session.target_kind === 'replay' ? 'Replay evidence' : 'Simulation evidence'}</h3>
          <p role="status">
            {session.status.replaceAll('_', ' ')} · {session.message}
          </p>
          <p>Task: {session.task}</p>
          <dl className="dataset-facts">
            <div>
              <dt>Agent session</dt>
              <dd>{session.id}</dd>
            </div>
            <div>
              <dt>Authority epoch</dt>
              <dd>{session.authority_epoch}</dd>
            </div>
            <div>
              <dt>Mode and clock</dt>
              <dd>
                {session.mode} · {session.clock_policy}
              </dd>
            </div>
            <div>
              <dt>Calls used</dt>
              <dd>
                {session.calls_used} / {session.max_calls} · {Math.max(0, session.max_calls - session.calls_used)}{' '}
                remaining
              </dd>
            </div>
            <div>
              <dt>Tokens reported</dt>
              <dd>
                {session.input_tokens} input + {session.output_tokens} output ·{' '}
                {Math.max(0, session.max_total_tokens - session.input_tokens - session.output_tokens)} remaining
              </dd>
            </div>
            <div>
              <dt>Wall time</dt>
              <dd>
                {secondsSince(session.created_at, terminal(session) ? Date.parse(session.updated_at) : now).toFixed(1)}{' '}
                s / {session.max_seconds} s
              </dd>
            </div>
            <div>
              <dt>Task outcome</dt>
              <dd>Unverified</dd>
            </div>
          </dl>
          {observation && (
            <>
              <h4>Admitted observation</h4>
              <p>
                {observation.id} · {observation.environment_kind} · {observation.source}
              </p>
              <p>
                Admitted {secondsSince(observation.captured_at, now).toFixed(1)} s ago ·{' '}
                {Date.parse(observation.expires_at) > now ? 'Within validity window' : 'Expired'} · step{' '}
                {observation.step} · revision {observation.revision}
              </p>
              <p>
                Embodiment: {observation.embodiment_manifest_id} · {observation.units}
              </p>
              <p>
                Measured joints:{' '}
                {Object.entries(observation.joints)
                  .map(([joint, value]) => `${joint}: ${value.toFixed(4)}`)
                  .join(' · ')}
              </p>
              <details>
                <summary>Observation identity</summary>
                <p>
                  Snapshot hash: <code>{observation.sha256}</code>
                </p>
                <p>
                  Image hash: <code>{observation.image_sha256 ?? 'No admitted image'}</code>
                </p>
              </details>
            </>
          )}
          {proposal && (
            <section aria-label="Pending agent proposal">
              <h4>Proposal</h4>
              <p>
                {proposal.arguments.joint}: {proposal.arguments.delta_rad > 0 ? '+' : ''}
                {proposal.arguments.delta_rad} rad · {proposal.status}
              </p>
              <p>
                Model {proposal.model} ·{' '}
                {proposalFresh
                  ? `Expires in ${Math.max(0, (Date.parse(proposal.expires_at) - now) / 1000).toFixed(1)} s`
                  : 'Expired or superseded'}
              </p>
              <p>
                Proposal <code>{proposal.id}</code> · observation <code>{proposal.observation_id}</code>
              </p>
              {session.mode === 'supervised' && (
                <button
                  type="button"
                  className="primary-button"
                  disabled={busy || !proposalFresh || selected.isError || pending(session) || terminal(session)}
                  onClick={() => {
                    const exact = { session: session.id, proposal: proposal.id, epoch: proposal.authority_epoch };
                    void run(() => agentApi.approve(exact.session, exact.proposal, exact.epoch));
                  }}
                >
                  Accept this proposal
                </button>
              )}
            </section>
          )}
          {latestReceipt && (
            <section aria-label="Latest execution receipt">
              <h4>{latestReceipt.status === 'replayed' ? 'Replay receipt' : 'Execution receipt'}</h4>
              <p>
                {latestReceipt.status} · {latestReceipt.reason} · {latestReceipt.elapsed_seconds.toFixed(3)} s
              </p>
              <p>
                Applied targets:{' '}
                {Object.entries(latestReceipt.applied_targets)
                  .map(([joint, value]) => `${joint}: ${value.toFixed(4)} rad`)
                  .join(' · ') || 'None'}
              </p>
              <p>
                Receipt <code>{latestReceipt.id}</code> · task success remains unverified.
              </p>
              <dl className="dataset-facts">
                <div>
                  <dt>Source observation</dt>
                  <dd>{latestReceipt.pre_observation_id}</dd>
                </div>
                <div>
                  <dt>Next observation</dt>
                  <dd>{latestReceipt.post_observation_id ?? 'Unavailable'}</dd>
                </div>
              </dl>
              {latestReceipt.executor_evidence && (
                <details>
                  <summary>Executor measurements and limits</summary>
                  <div className="agent-control-body">
                    <p>
                      Requested targets, applied targets and measured next state are separate fields. Joint values are
                      in radians; wall and simulation durations are in seconds.
                    </p>
                    <pre className="agent-control-json">{JSON.stringify(latestReceipt.executor_evidence, null, 2)}</pre>
                  </div>
                </details>
              )}
            </section>
          )}
          <div className="workbench-actions">
            <button
              type="button"
              className="secondary-button"
              disabled={busy || terminal(session) || pending(session) || selected.isError}
              onClick={() => void run(() => agentApi.action(session.id, 'observe'))}
            >
              Capture fresh observation
            </button>
            <button
              type="button"
              className="primary-button"
              disabled={
                busy ||
                terminal(session) ||
                pending(session) ||
                !!proposalFresh ||
                session.calls_used >= session.max_calls ||
                selected.isError
              }
              onClick={() => {
                const requestId = crypto.randomUUID();
                void run(() => agentApi.propose(session.id, requestId));
              }}
            >
              Plan one correction
            </button>
            <button
              type="button"
              className="secondary-button"
              disabled={busy || session.status !== 'planning'}
              onClick={() => void run(() => agentApi.action(session.id, 'cancel-inference'))}
            >
              Cancel inference
            </button>
            <button
              type="button"
              className="secondary-button"
              disabled={busy || terminal(session)}
              onClick={() => void run(() => agentApi.action(session.id, 'pause'))}
            >
              Pause planning
            </button>
            <button
              type="button"
              className="secondary-button"
              disabled={busy || terminal(session)}
              onClick={() => void run(() => agentApi.action(session.id, 'stop'))}
            >
              Stop agent session
            </button>
          </div>
          <p>Closing this panel leaves the saved session available. Use Stop agent session to revoke its authority.</p>
          <details>
            <summary>Ordered session events</summary>
            {events.error && <p role="alert">Event history is unavailable. Refresh to recover saved evidence.</p>}
            <ol className="agent-control-events">
              {[...(events.data ?? [])]
                .sort((a, b) => a.id - b.id)
                .map((event) => (
                  <li key={event.id}>
                    <span>
                      {event.id} · {event.kind} · {new Date(event.created_at).toLocaleTimeString()}
                    </span>
                    <pre className="agent-control-json">{JSON.stringify(event.record, null, 2)}</pre>
                  </li>
                ))}
            </ol>
          </details>
        </section>
      )}
      {busy && <p role="status">Agent request in progress…</p>}
      {error && <p role="alert">{error}</p>}
    </div>
  );
}
