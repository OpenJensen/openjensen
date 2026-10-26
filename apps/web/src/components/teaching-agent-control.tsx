'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { agentApi, type AgentSession, type AgentSessionRequest } from '@/lib/agent-api';
import { AiPhysicalConnection, useAiPhysicalConnection } from './ai-physical-connection';
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
  const [target, setTarget] = useState<'replay' | 'teaching' | 'robot_lab'>('replay');
  const [teachingJob, setTeachingJob] = useState('');
  const [replayObservation, setReplayObservation] = useState('');
  const [mode, setMode] = useState<'advisory' | 'shadow' | 'supervised' | 'bounded'>('supervised');
  const [clock, setClock] = useState<'stepped' | 'continuous'>('stepped');
  const [task, setTask] = useState('');
  const [planningMode, setPlanningMode] = useState<'task' | 'joint'>('task');
  const [maxSteps, setMaxSteps] = useState(12);
  const [sceneApproval, setSceneApproval] = useState('');
  const [admissionImageLoaded, setAdmissionImageLoaded] = useState('');
  const [reviewOutcome, setReviewOutcome] = useState<'success' | 'failure' | 'unknown'>('unknown');
  const [reviewNote, setReviewNote] = useState('');
  const [reviewConfirmed, setReviewConfirmed] = useState('');
  const [reviewImageLoaded, setReviewImageLoaded] = useState('');
  const [maxCalls, setMaxCalls] = useState(12);
  const [maxSeconds, setMaxSeconds] = useState(300);
  const [maxOutput, setMaxOutput] = useState(2048);
  const [maxTokens, setMaxTokens] = useState(64000);
  const [consent, setConsent] = useState(false);
  const [selectedId, setSelectedId] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [now, setNow] = useState(() => Date.now());
  const working = useRef(false);
  const operationVersion = useRef(0);
  const mounted = useRef(true);
  const selectedRef = useRef(selectedId);
  selectedRef.current = selectedId;
  const cacheSession = useCallback(
    (result: AgentSession) => {
      if (result.project_id !== projectId) return result;
      const key = ['agent-session', projectId, result.id];
      const previous = client.getQueryData<AgentSession>(key);
      if (
        previous &&
        (previous.authority_epoch > result.authority_epoch ||
          Date.parse(previous.updated_at) > Date.parse(result.updated_at))
      )
        return previous;
      client.setQueryData(key, result);
      return result;
    },
    [client, projectId],
  );
  const providers = useQuery({ queryKey: ['agent-providers'], queryFn: agentApi.providers, retry: false });
  const history = useQuery({
    queryKey: ['agent-sessions', projectId],
    queryFn: () => agentApi.sessions(projectId),
    retry: false,
    refetchInterval: 5000,
  });
  const selected = useQuery({
    queryKey: ['agent-session', projectId, selectedId],
    queryFn: async () => cacheSession(await agentApi.session(selectedId)),
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
    const resetReview = () => {
      setSceneApproval('');
      setReviewConfirmed('');
      setAdmissionImageLoaded('');
      setReviewImageLoaded('');
    };
    const hidden = () => {
      if (document.visibilityState !== 'visible') resetReview();
    };
    window.addEventListener('blur', resetReview);
    document.addEventListener('visibilitychange', hidden);
    return () => {
      window.removeEventListener('blur', resetReview);
      document.removeEventListener('visibilitychange', hidden);
      mounted.current = false;
      clearInterval(timer);
    };
  }, []);
  const provider = providers.data?.find((item) => item.id === providerId);
  const cloud = provider?.provider === 'openai_responses';
  const session = selected.data?.project_id === projectId && selected.data.id === selectedId ? selected.data : null;
  const reservedTokens =
    session && 'reserved_tokens' in session && typeof session.reserved_tokens === 'number'
      ? session.reserved_tokens
      : 0;
  const physical = useAiPhysicalConnection(target === 'robot_lab', (result) => {
    if (selectedRef.current === result.id) cacheSession(result);
  });
  if (session) physical.updateLease(session);
  const physicalSession = session?.target_kind === 'robot_lab';
  const proposal = session?.pending_proposal;
  const proposalId = proposal?.id;
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
    (target === 'replay' ||
      (target === 'robot_lab' && physical.canArm) ||
      (target === 'teaching' && !!teachingJob.trim())) &&
    !busy &&
    !history.isError &&
    !providers.isError;
  const latestReceipt = session?.receipts?.at(-1);
  const admission = session?.execution_observation;
  const sceneKey =
    session && proposal ? `${session.id}:${session.authority_epoch}:${proposal.id}:${physical.cameraIndex}` : '';
  const admissionKey = admission ? `${admission.id}:${admission.sha256}` : '';
  const reviewObservation = session?.execution_observation ?? session?.observation;
  const reviewImageKey = reviewObservation ? `${reviewObservation.id}:${reviewObservation.sha256}` : '';
  const reviewImageReady = !reviewObservation?.image_sha256 || reviewImageLoaded === reviewImageKey;
  const reviewKey =
    session && reviewObservation
      ? `${session.id}:${session.authority_epoch}:${reviewObservation.id}:${reviewOutcome}`
      : '';
  const admissionFresh =
    !!admission &&
    Date.parse(admission.expires_at) > now &&
    secondsSince(admission.captured_at, now) < 1 &&
    admissionImageLoaded === admissionKey;
  const physicalApproval =
    !!physicalSession &&
    !!session?.control_armed &&
    physical.owned &&
    physical.operatorPresent &&
    physical.workspaceClear &&
    sceneApproval === sceneKey &&
    admissionFresh;
  const admissionContext = useRef({ session: selectedId, epoch: session?.authority_epoch, proposal: proposal?.id });
  admissionContext.current = { session: selectedId, epoch: session?.authority_epoch, proposal: proposal?.id };
  useEffect(() => {
    if (
      !physicalSession ||
      !session?.control_armed ||
      !proposalId ||
      session.status !== 'waiting_for_acceptance' ||
      !physical.owned ||
      !physical.operatorPresent ||
      !physical.workspaceClear
    )
      return;
    const expected = { session: session.id, epoch: session.authority_epoch, proposal: proposalId };
    let disposed = false;
    let inFlight = false;
    const capture = async () => {
      if (disposed || inFlight || document.visibilityState !== 'visible' || !document.hasFocus()) return;
      inFlight = true;
      try {
        const result = await agentApi.executionObservation(expected.session);
        const current = admissionContext.current;
        if (
          !disposed &&
          current.session === expected.session &&
          current.epoch === expected.epoch &&
          current.proposal === expected.proposal
        )
          cacheSession(result);
      } catch {
        if (!disposed) setAdmissionImageLoaded('');
      } finally {
        inFlight = false;
      }
    };
    void capture();
    const timer = setInterval(() => void capture(), 500);
    return () => {
      disposed = true;
      clearInterval(timer);
    };
  }, [
    physicalSession,
    session?.id,
    session?.authority_epoch,
    session?.control_armed,
    session?.status,
    proposalId,
    physical.owned,
    physical.operatorPresent,
    physical.workspaceClear,
    cacheSession,
  ]);

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
    const expectedVersion = operationVersion.current;
    try {
      const result = await action();
      if (!mounted.current || expectedVersion !== operationVersion.current) {
        if (creating && result.target_kind === 'robot_lab') void agentApi.stop(result.id, true);
        return;
      }
      if (result.project_id !== projectId || (!creating && result.id !== expectedId))
        throw new Error('Agent session identity changed. Refresh saved sessions before continuing.');
      cacheSession(result);
      if (creating) {
        if (result.target_kind === 'robot_lab') physical.adopt(result);
        selectedRef.current = result.id;
        setSelectedId(result.id);
        setConsent(false);
      }
      await refresh(result.id);
    } catch (problem) {
      if (creating && target === 'robot_lab')
        await physical.stop('AI arming could not be verified. Stop was requested.');
      if (mounted.current && expectedVersion === operationVersion.current) {
        setSceneApproval('');
        setError(problemText(problem));
        await refresh();
      }
    } finally {
      if (expectedVersion === operationVersion.current) {
        working.current = false;
        if (mounted.current) setBusy(false);
      }
    }
  }
  async function interrupt(action: 'cancel-inference' | 'pause' | 'stop') {
    if (!session) return;
    operationVersion.current += 1;
    working.current = false;
    setBusy(false);
    setSceneApproval('');
    setError('');
    try {
      if (action === 'stop' && physicalSession && physical.owned)
        await physical.stop('Stop requested. Support the arm while termination is verified.');
      else cacheSession(await agentApi.action(session.id, action));
      await refresh(session.id);
    } catch (problem) {
      if (mounted.current) setError(problemText(problem));
    }
  }
  function create() {
    if (!canCreate) return;
    const arming = physical.arming;
    if (target === 'robot_lab') {
      try {
        physical.beginTransfer();
      } catch (problem) {
        setError(problemText(problem));
        return;
      }
    }
    const request: AgentSessionRequest = {
      provider_id: providerId,
      task: task.trim(),
      target_kind: target,
      teaching_job_id: target === 'teaching' ? teachingJob.trim() : null,
      replay_observation_id: target === 'replay' && replayObservation.trim() ? replayObservation.trim() : null,
      mode,
      clock_policy: target === 'replay' ? 'replay' : target === 'robot_lab' ? 'physical' : clock,
      planning_mode: planningMode,
      max_steps: maxSteps,
      physical: target === 'robot_lab' ? arming : null,
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
                if (physical.owned || physical.connecting)
                  void physical.stop('The selected target changed. Stop was requested.');
                const next = event.target.value as 'replay' | 'teaching' | 'robot_lab';
                setTarget(next);
                if (next === 'robot_lab') {
                  setMode('supervised');
                  setPlanningMode('task');
                }
                setSceneApproval('');
                setConsent(false);
              }}
            >
              <option value="replay">Replay observation</option>
              <option value="teaching">Simulation</option>
              <option value="robot_lab">Physical SO101</option>
            </select>
          </label>
          <label className="workbench-field">
            Planning style
            <select
              value={planningMode}
              disabled={busy}
              onChange={(event) => setPlanningMode(event.target.value as 'task' | 'joint')}
            >
              <option value="task">Goal-directed task</option>
              <option value="joint" disabled={target === 'robot_lab'}>
                Single correction
              </option>
            </select>
          </label>
          <label className="workbench-field">
            Agent mode
            <select value={mode} disabled={busy} onChange={(event) => setMode(event.target.value as typeof mode)}>
              <option value="advisory">Advice</option>
              <option value="shadow">Shadow proposals</option>
              <option value="supervised">Supervised acceptance</option>
              <option value="bounded" disabled={target === 'robot_lab'}>
                Bounded execution
              </option>
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
          ) : target === 'replay' ? (
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
          ) : null}
        </div>
        {target === 'robot_lab' && <AiPhysicalConnection physical={physical} />}
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
            ? 'Astra may execute enabled actions within this session’s limits, observing the result before each next step.'
            : mode === 'supervised'
              ? 'Review and accept each exact proposal while its observation remains valid.'
              : 'Proposals and evidence are recorded without applying motion.'}
        </p>
        <details>
          <summary>Run limits</summary>
          <div className="workbench-field-grid agent-control-body">
            <label className="workbench-field">
              Maximum execution steps
              <input
                type="number"
                value={maxSteps}
                min={1}
                max={60}
                required
                disabled={busy}
                onChange={(event) => setMaxSteps(Number(event.target.value))}
              />
            </label>
            <label className="workbench-field">
              Maximum model calls
              <input
                type="number"
                value={maxCalls}
                min={1}
                max={100}
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
                max={1000000}
                required
                disabled={busy}
                onChange={(event) => setMaxTokens(Number(event.target.value))}
              />
            </label>
          </div>
        </details>
        {cloud && (
          <p>
            Run budget: up to {maxCalls} model calls, {maxSteps} execution steps, {maxSeconds} seconds and{' '}
            {maxTokens.toLocaleString()} total tokens. Token limits are not a dollar quote.
          </p>
        )}
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
            {target === 'robot_lab' ? 'Arm AI session' : 'Create agent session'}
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
            if (physical.owned) void physical.stop('The selected saved session changed. Stop was requested.');
            setSelectedId(event.target.value);
            setSceneApproval('');
            setReviewConfirmed('');
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
          <h3>
            {session.target_kind === 'replay'
              ? 'Replay evidence'
              : session.target_kind === 'robot_lab'
                ? 'Physical execution evidence'
                : 'Simulation evidence'}
          </h3>
          <p role="status">
            {session.status.replaceAll('_', ' ')} · {session.message}
          </p>
          <p>Task: {session.task}</p>
          <p>Enabled skills: {session.supported_skills?.join(' · ') || 'No executable skills advertised'}</p>
          <p>
            Execution steps: {session.step_count ?? session.receipts?.length ?? 0} / {session.max_steps ?? 12} ·{' '}
            {session.run_enabled ? 'Task loop running' : 'Task loop paused'}
          </p>
          {session.model_assessment && (
            <section aria-label="Astra task assessment">
              <h4>Model assessment</h4>
              <p>
                {session.model_assessment.outcome.replaceAll('_', ' ')} · {session.model_assessment.summary}
              </p>
              <p>This is the model’s assessment. It does not verify task success.</p>
            </section>
          )}
          {physicalSession && session.control_armed && (
            <p role="status">
              The physical controller remains armed and holds locally while the task loop is paused. Keep this page
              focused and operator presence confirmed. Stop disables torque.
            </p>
          )}
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
                {session.input_tokens} input + {session.output_tokens} output · {reservedTokens} reserved ·{' '}
                {Math.max(0, session.max_total_tokens - session.input_tokens - session.output_tokens - reservedTokens)}{' '}
                remaining
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
              <dd>
                {session.task_review
                  ? `Operator reported ${session.task_review.outcome}`
                  : session.task_success_verified
                    ? 'Verified by task evidence'
                    : 'Unverified'}
              </dd>
            </div>
          </dl>
          {observation && (
            <>
              <h4>{physicalSession ? 'Original model observation' : 'Model observation'}</h4>
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
                  .map(
                    ([joint, value]) =>
                      `${joint}: ${value.toFixed(4)} ${observation.joint_units?.[joint] ?? observation.units}`,
                  )
                  .join(' · ')}
              </p>
              {observation.image_sha256 && (
                <img
                  className="agent-control-camera"
                  src={agentApi.imageUrl(session.id, observation.id)}
                  alt="Original model observation"
                />
              )}
              {observation.calibration_sha256 && (
                <p>
                  Calibration: <code>{observation.calibration_sha256}</code>
                </p>
              )}
              {observation.task_evidence && (
                <details>
                  <summary>Independent task evidence</summary>
                  <pre className="agent-control-json">{JSON.stringify(observation.task_evidence, null, 2)}</pre>
                </details>
              )}
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
                {proposal.skill} · {proposal.status}
                {proposal.purpose ? ` · ${proposal.purpose}` : ''}
              </p>
              <pre className="agent-control-json">{JSON.stringify(proposal.arguments, null, 2)}</pre>
              <p>
                Model {proposal.model} ·{' '}
                {proposalFresh
                  ? `Expires in ${Math.max(0, (Date.parse(proposal.expires_at) - now) / 1000).toFixed(1)} s`
                  : 'Expired or superseded'}
              </p>
              <p>
                Proposal <code>{proposal.id}</code> · observation <code>{proposal.observation_id}</code>
              </p>
              {physicalSession && (
                <section aria-label="Current physical admission observation">
                  <h4>Current execution observation</h4>
                  <p>
                    The model used its original observation above. Review this separate current camera and encoder
                    sample before admitting the same proposed targets.
                  </p>
                  {admission ? (
                    <>
                      <p>
                        {admission.id} · admitted {secondsSince(admission.captured_at, now).toFixed(2)} s ago ·{' '}
                        {admissionFresh ? 'Current image loaded' : 'Waiting for a fresh displayed image'}
                      </p>
                      <img
                        key={admissionKey}
                        className="agent-control-camera"
                        src={agentApi.imageUrl(session.id, admission.id)}
                        alt="Current physical execution observation"
                        onLoad={() => setAdmissionImageLoaded(admissionKey)}
                        onError={() => {
                          setAdmissionImageLoaded('');
                          setSceneApproval('');
                        }}
                      />
                      <p>
                        Calibration: <code>{admission.calibration_sha256 ?? 'Unavailable'}</code>
                      </p>
                      <p>
                        Measured joints:{' '}
                        {Object.entries(admission.joints)
                          .map(
                            ([joint, value]) =>
                              `${joint}: ${value.toFixed(3)} ${admission.joint_units?.[joint] ?? admission.units}`,
                          )
                          .join(' · ')}
                      </p>
                    </>
                  ) : (
                    <p role="status">Waiting for a current physical observation.</p>
                  )}
                  <label className="workbench-check">
                    <input
                      type="checkbox"
                      checked={sceneApproval === sceneKey && physical.operatorPresent && physical.owned}
                      disabled={!admissionFresh || !physical.owned || !physical.operatorPresent}
                      onChange={(event) => setSceneApproval(event.target.checked ? sceneKey : '')}
                    />
                    I reviewed the current image and proposed targets; the object, workspace and path remain suitable
                    for this exact action.
                  </label>
                </section>
              )}
              {session.mode === 'supervised' && (
                <button
                  type="button"
                  className="primary-button"
                  disabled={
                    busy ||
                    !proposalFresh ||
                    selected.isError ||
                    pending(session) ||
                    terminal(session) ||
                    (physicalSession && !physicalApproval)
                  }
                  onClick={() => {
                    const exact = { session: session.id, proposal: proposal.id, epoch: proposal.authority_epoch };
                    const admissionApproval =
                      physicalSession && admission
                        ? {
                            execution_observation_id: admission.id,
                            execution_observation_sha256: admission.sha256,
                            scene_unchanged_confirmed: sceneApproval === sceneKey,
                          }
                        : undefined;
                    setSceneApproval('');
                    void run(() => agentApi.approve(exact.session, exact.proposal, exact.epoch, admissionApproval));
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
                  .map(
                    ([joint, value]) =>
                      `${joint}: ${value.toFixed(4)} ${latestReceipt.joint_units?.[joint] ?? observation?.joint_units?.[joint] ?? 'radians'}`,
                  )
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
                      Requested targets, applied targets and measured next state are separate fields. Units come from
                      the target manifest; wall and simulation durations are in seconds.
                    </p>
                    <pre className="agent-control-json">{JSON.stringify(latestReceipt.executor_evidence, null, 2)}</pre>
                  </div>
                </details>
              )}
            </section>
          )}
          <div className="workbench-actions">
            {session.planning_mode !== 'joint' && (
              <button
                type="button"
                className="primary-button"
                disabled={
                  busy ||
                  terminal(session) ||
                  pending(session) ||
                  session.run_enabled ||
                  selected.isError ||
                  (physicalSession && (!physical.owned || !physical.operatorPresent || !session.control_armed))
                }
                onClick={() => void run(() => agentApi.run(session.id))}
              >
                {session.step_count ? 'Continue task' : 'Start task'}
              </button>
            )}
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
              {session.planning_mode === 'joint' ? 'Plan one correction' : 'Plan next action'}
            </button>
            <button
              type="button"
              className="secondary-button"
              disabled={session.status !== 'planning'}
              onClick={() => void interrupt('cancel-inference')}
            >
              Cancel inference
            </button>
            <button
              type="button"
              className="secondary-button"
              disabled={terminal(session)}
              onClick={() => void interrupt('pause')}
            >
              Pause planning
            </button>
            <button
              type="button"
              className="secondary-button"
              disabled={terminal(session) && !session.control_armed}
              onClick={() => void interrupt('stop')}
            >
              Stop agent session
            </button>
          </div>
          <p>
            {physicalSession
              ? 'Hiding or closing this panel, losing focus or withdrawing operator presence stops this page’s physical control. The local lease also expires independently.'
              : 'Closing this panel leaves the saved session available. Use Stop agent session to revoke its authority.'}
          </p>
          {reviewObservation && !pending(session) && !session.run_enabled && (
            <details>
              <summary>Review task outcome</summary>
              <div className="agent-control-body">
                <p>
                  Record your review of observation <code>{reviewObservation.id}</code>. A model assessment or a
                  completed movement does not verify the task.
                </p>
                {reviewObservation.image_sha256 && (
                  <img
                    key={reviewImageKey}
                    className="agent-control-camera"
                    src={agentApi.imageUrl(session.id, reviewObservation.id)}
                    alt="Task outcome review observation"
                    onLoad={() => setReviewImageLoaded(reviewImageKey)}
                    onError={() => {
                      setReviewImageLoaded('');
                      setReviewConfirmed('');
                    }}
                  />
                )}
                <label className="workbench-field">
                  Reviewed outcome
                  <select
                    value={reviewOutcome}
                    disabled={busy}
                    onChange={(event) => {
                      setReviewOutcome(event.target.value as typeof reviewOutcome);
                      setReviewConfirmed('');
                    }}
                  >
                    <option value="unknown">Unknown</option>
                    <option value="success">Success</option>
                    <option value="failure">Failure</option>
                  </select>
                </label>
                <label className="workbench-field">
                  Review note
                  <textarea
                    value={reviewNote}
                    maxLength={1024}
                    rows={2}
                    disabled={busy}
                    onChange={(event) => setReviewNote(event.target.value)}
                  />
                </label>
                <label className="workbench-check">
                  <input
                    type="checkbox"
                    checked={reviewConfirmed === reviewKey && !!reviewKey && reviewImageReady}
                    disabled={!reviewImageReady}
                    onChange={(event) => setReviewConfirmed(event.target.checked ? reviewKey : '')}
                  />
                  I reviewed this retained observation and am reporting the task outcome myself.
                </label>
                <button
                  type="button"
                  className="secondary-button"
                  disabled={
                    busy || reviewConfirmed !== reviewKey || !reviewKey || !reviewImageReady || selected.isError
                  }
                  onClick={() => {
                    const exact = {
                      authority_epoch: session.authority_epoch,
                      observation_id: reviewObservation.id,
                      outcome: reviewOutcome,
                      note: reviewNote,
                    };
                    setReviewConfirmed('');
                    void run(() => agentApi.review(session.id, exact));
                  }}
                >
                  Save operator review
                </button>
              </div>
            </details>
          )}
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
