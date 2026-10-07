'use client';

import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api, type Job } from '@/lib/api';
import { useDurableSubmission, submissionOf, submissionStorageKey, type SubmissionRecovery } from '@/lib/durable-submission';
import { intakeAcknowledgement, reviewedIntakeHistory } from '@/lib/dataset-submission';
import { DatasetSubmissionRecovery } from './dataset-submission-recovery';
import { type PolicyJobAttempt } from '@/lib/policy-job-attempt';
import { UncertainPolicyJob, sameJson } from '@/lib/policy-job-mutation';
import type { ManagedPublishedCapture } from '@/lib/managed-teaching';
import { teachingRecordingRecipe } from '@/lib/teaching-recording-handoff';
import {
  beginRecordingAttempt, clearRecordingAttempt, failRecordingAttempt, finishRecordingAttempt, ownsRecordingAttempt,
  cancelRecording, emptyRecordingState, readRecordingAttempt, readRecordingContext, readRecordingJob,
  readRecordingHistory, readRecordingReceipt, readRecordingState, recordingIntent, recordingRecipe, recordingReceipt,
  RecordingStorageUnavailable, selectionMatches,
  recordingRequest, storeRecordingReceipt, writeRecordingState, type RecordingJob, type RecordingState,
} from '@/lib/recording-preparation';
import { WorkbenchDisclosure } from './workbench-disclosure';

const isRecording = (job: Job) => 'recordings' in job.request && job.request.recordings != null;
const active = (job: RecordingJob) => job.status === 'queued' || job.status === 'running';
const message = (cause: unknown) => cause instanceof Error ? cause.message : 'Recording preparation is unavailable.';

/** Mount with key=projectId. This owns no live Teaching executor or microphone state. */
export function RecordingPreparationPanel({ projectId, onInspect, onTrain, teachingCapture, onTeachingCaptureConsumed, isActive = () => true }: {
  projectId: string; onInspect: (job: RecordingJob) => void; onTrain: (job: RecordingJob) => void;
  teachingCapture?: ManagedPublishedCapture; onTeachingCaptureConsumed?: () => void;
  isActive?: () => boolean;
}) {
  const client = useQueryClient();
  const submission = useDurableSubmission({ project: projectId, operation: 'dataset.inspect' });
  const sharedKey = ['durable-submission', submissionStorageKey({ project: projectId, operation: 'dataset.inspect' })];
  const ownerKey = ['recording-attempt-owner', projectId];
  useQuery<{ attemptId: string | null }>({ queryKey: ownerKey, queryFn: async () => ({ attemptId: null }), initialData: { attemptId: null }, enabled: false, gcTime: Infinity });
  const cachedOwner = () => client.getQueryData<{ attemptId: string | null }>(ownerKey)?.attemptId;
  const cacheOwner = (attemptId: string | null) => client.setQueryData(ownerKey, { attemptId });
  const [state, setState] = useState<RecordingState>(() => emptyRecordingState(projectId));
  const stateRef = useRef(state), mounted = useRef(false), currentProject = useRef(projectId);
  currentProject.current = projectId;
  const captureRef = useRef(teachingCapture); captureRef.current = teachingCapture;
  const busy = useRef(false), generation = useRef(0), attemptVersion = useRef(0), reviewedToken = useRef<string | null | undefined>(undefined);
  const [storageReady, setStorageReady] = useState(false), [pending, setPending] = useState(false);
  const [attempt, setAttempt] = useState<PolicyJobAttempt>(null), [reviewed, setReviewed] = useState(false);
  const [accepted, setAccepted] = useState<RecordingJob | null>(null);
  const [error, setError] = useState(''), [consent, setConsent] = useState(false), [timeout, setTimeoutValue] = useState('600');
  const [cancelId, setCancelId] = useState('');
  const live = () => mounted.current && currentProject.current === projectId && isActive();
  function storageFailure() { const e = new RecordingStorageUnavailable(); if (live()) { setStorageReady(false); setError(e.message); setReviewed(false); setConsent(false); } return e; }
  function persist(next: RecordingState) {
    try { writeRecordingState(projectId, next); stateRef.current = next; if (live()) setState(next); }
    catch { throw storageFailure(); }
  }
  function loadSession() {
    if (!projectId || busy.current) return;
    try {
      const saved = readRecordingState(projectId), receipt = readRecordingReceipt(projectId);
      const previous = readRecordingAttempt(projectId);
      stateRef.current = saved; setState(saved); setTimeoutValue(String(saved.recipe?.timeout_seconds ?? 600)); setAccepted(receipt);
      setAttempt(saved.pending?.action === 'cancel' ? previous ?? { state: 'uncertain', message: recordingIntent(projectId, saved.pending.recipe, 'cancel', saved.pending.job_id ?? undefined) + ' The saved outcome needs review.' } : null);
      // A round-trip verifies writes are available before another request is admitted.
      writeRecordingState(projectId, saved); cacheOwner(saved.pending?.attempt_id ?? null); setStorageReady(true); setConsent(false); setReviewed(false); setError('');
    } catch { storageFailure(); }
  }
  useEffect(() => {
    mounted.current = true; loadSession();
    return () => { mounted.current = false; generation.current += 1; };
    // The parent project key isolates this controller; no mutation is started by mounting.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);
  const previousLegacy = useRef(false);
  useEffect(() => {
    const legacy = !!submission.attempt && !submissionOf(submission.attempt);
    if (previousLegacy.current && !submission.attempt) loadSession();
    previousLegacy.current = legacy;
    // Clearing shared legacy intake recovery also clears only the old submit marker.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [submission.attempt]);
  let sharedReceipt: RecordingJob | null = null;
  if (submission.receipt && isRecording(submission.receipt)) {
    try { sharedReceipt = recordingReceipt(submission.receipt, projectId); } catch { /* No unverified recording identity is displayed. */ }
  }
  const context = useQuery({ queryKey: ['recording-context', projectId], queryFn: () => readRecordingContext(projectId), enabled: !!projectId, retry: false });
  const jobs = useQuery({ queryKey: ['recording-history', projectId], queryFn: () => readRecordingHistory(projectId), enabled: !!projectId, retry: false,
    refetchInterval: query => query.state.data?.some(active) ? 1000 : 5000 });
  const history = [...(jobs.data ?? [])].sort((a, b) => b.created_at.localeCompare(a.created_at));
  const selectedId = state.selected_job_id;
  const selectedQuery = useQuery({ queryKey: ['recording-job', projectId, selectedId], queryFn: () => readRecordingJob(projectId, selectedId), enabled: !!projectId && !!selectedId, retry: false,
    refetchInterval: query => query.state.data && active(query.state.data) ? 1000 : false });
  const cached = accepted?.id === selectedId ? accepted : history.find(j => j.id === selectedId) ?? (sharedReceipt?.id === selectedId ? sharedReceipt : undefined);
  const selected = selectedQuery.data ?? cached;
  const result = !selectedQuery.isError && selectedQuery.isSuccess ? selectedQuery.data.result : null;
  const catalog = !context.isError ? context.data?.catalog : null;
  const stale = !!state.recipe && (!catalog || !selectionMatches(state.recipe, catalog));
  const count = state.recipe?.captures.reduce((n, c) => n + c.episodes.length, 0) ?? 0;
  const timeoutValid = /^\d+$/.test(timeout) && Number.isSafeInteger(Number(timeout)) && Number(timeout) >= 60 && Number(timeout) <= 1800;
  const ready = submission.hydrated && submission.available && !submission.attempt && !submission.busy && storageReady && !!projectId && !!catalog && !!state.recipe && !stale && timeoutValid && consent && !attempt && !pending && jobs.isSuccess && !jobs.isError;
  const knownActive = history.find(active);

  function changeSelection(next: RecordingState['recipe']) {
    generation.current += 1; setConsent(false); setError(''); setCancelId('');
    try { persist({ ...stateRef.current, recipe: next }); } catch { /* Failure visibly latches requests off. */ }
  }
  async function selectTeachingCapture() {
    const original = captureRef.current;
    if (!original || original.project_id !== projectId || busy.current || !storageReady || !submission.available || submission.busy || submission.attempt || attempt || !timeoutValid) return;
    const version = generation.current, limit = Number(timeout);
    busy.current = true; setPending(true); setConsent(false); setError('');
    try {
      const before = readRecordingState(projectId);
      const fresh = await readRecordingContext(projectId);
      if (!live() || captureRef.current !== original || version !== generation.current) return;
      const shared = client.getQueryData<SubmissionRecovery>(sharedKey);
      const saved = readRecordingState(projectId);
      if (!shared?.available || shared.busy || shared.attempt || saved.pending || readRecordingAttempt(projectId) ||
          sessionStorage.getItem(submissionStorageKey({ project: projectId, operation: 'dataset.inspect' })) !== null || !sameJson(before, saved))
        throw new Error('Recording selection or recovery changed during review. Inspect it before selecting the teaching capture.');
      if (!fresh.catalog) throw new Error('The published capture is not available in this project’s configured recording catalog.');
      const recipe = teachingRecordingRecipe(original, projectId, fresh.catalog, limit);
      client.setQueryData(['recording-context', projectId], fresh);
      generation.current += 1; setCancelId('');
      persist({ ...saved, recipe, selected_job_id: '' });
      onTeachingCaptureConsumed?.();
    } catch (cause) { if (live()) { if (cause instanceof RecordingStorageUnavailable) storageFailure(); else setError(message(cause)); } }
    finally { busy.current = false; if (live()) setPending(false); }
  }
  function toggle(session: string, episode: string, checked: boolean) {
    if (!catalog || stale || !storageReady || attempt || submission.attempt || submission.busy || busy.current) return;
    const source = catalog.captures.find(c => c.session_id === session), row = source?.episodes.find(e => e.episode_id === episode);
    if (!source || !row) return;
    const captures = (stateRef.current.recipe?.captures ?? []).map(c => ({ ...c, episodes: [...c.episodes] }));
    const current = captures.find(c => c.session_id === session);
    if (checked && !current) captures.push({ session_id: session, session_sha256: source.session_sha256, episodes: [{ episode_id: episode, receipt_sha256: row.receipt_sha256 }] });
    else if (current) current.episodes = checked ? [...current.episodes, { episode_id: episode, receipt_sha256: row.receipt_sha256 }] : current.episodes.filter(e => e.episode_id !== episode);
    const nonempty = captures.filter(c => c.episodes.length);
    try { changeSelection(nonempty.length ? recordingRecipe({ schema_version: 1, configuration_sha256: catalog.configuration_sha256, timeout_seconds: timeoutValid ? Number(timeout) : 600, captures: nonempty }) : null); }
    catch (cause) { setError(message(cause)); }
  }
  function chooseJob(id: string) {
    generation.current += 1; setCancelId('');
    try { persist({ ...stateRef.current, selected_job_id: id }); } catch { /* Keep current identity on a failed write. */ }
  }
  async function refresh() {
    const version = attemptVersion.current, wasUncertain = attempt?.state === 'uncertain';
    setConsent(false); setReviewed(false); reviewedToken.current = undefined;
    let owner: string | null;
    try { owner = readRecordingState(projectId).pending?.attempt_id ?? null; } catch { storageFailure(); return; }
    const response = await jobs.refetch(); void context.refetch(); if (selectedId) void selectedQuery.refetch();
    try {
      const current = readRecordingState(projectId).pending?.attempt_id ?? null;
      if (live() && wasUncertain && version === attemptVersion.current && owner === current && response.isSuccess && !response.isError) { reviewedToken.current = owner; setReviewed(true); }
    } catch { storageFailure(); }
  }
  async function prepare(mode: 'submit' | 'reconcile' | 'retry') {
    if (busy.current || !storageReady || attempt || (mode === 'submit' && !ready)) return;
    const version = generation.current;
    busy.current = true; setPending(true); setError(''); setConsent(false);
    const validate = (value: unknown, original: Record<string, unknown>) => intakeAcknowledgement(value, projectId, original);
    const beforePost = async () => { if (!live() || generation.current !== version) throw new Error('Selection changed; preparation was not submitted.'); };
    try {
      const job = mode === 'submit' ? await submission.submit(recordingRequest(stateRef.current.recipe!), validate, beforePost)
        : mode === 'reconcile' ? await submission.reconcile(validate) : await submission.retry(validate, beforePost);
      if (!job || !live()) return;
      const shared = client.getQueryData<SubmissionRecovery>(sharedKey);
      if (shared?.receipt?.id !== job.id) return; // A successor owns the shared receipt.
      if (!isRecording(job)) return; // A generic intake stays in its project history.
      const receipt = recordingReceipt(job, projectId);
      setAccepted(receipt); // Retain a known ACK before optional local draft writes.
      storeRecordingReceipt(receipt);
      if (generation.current === version) persist({ ...stateRef.current, selected_job_id: receipt.id });
    } catch (cause) {
      if (live()) { if (cause instanceof RecordingStorageUnavailable) storageFailure(); else setError(message(cause)); }
    } finally {
      busy.current = false; if (live()) setPending(false);
      void client.invalidateQueries({ queryKey: ['recording-history', projectId] }); void client.invalidateQueries({ queryKey: ['recording-job', projectId] });
    }
  }
  async function reviewSubmissionHistory() {
    const fresh = reviewedIntakeHistory(await api.jobs(projectId), projectId);
    const recordings = fresh.filter(job => job.kind === 'dataset.inspect' && isRecording(job)).map(job => recordingReceipt(job, projectId));
    client.setQueryData(['jobs', projectId], fresh); client.setQueryData(['recording-history', projectId], recordings);
    return live() ? fresh : false;
  }
  async function cancel() {
    const source = selected;
    if (busy.current || submission.busy || submission.attempt || !storageReady || attempt || !source || cancelId !== source.id || !active(source) || jobs.isError || selectedQuery.isError) return;
    const recipe = source.request.recordings, action = 'cancel' as const;
    const version = generation.current, originalId = source.id;
    const intent = recordingIntent(projectId, recipe, action, originalId), token = crypto.randomUUID().replaceAll('-', '');
    let received: RecordingJob | null = null;
    busy.current = true; setPending(true); setError(''); setReviewed(false); reviewedToken.current = undefined; attemptVersion.current += 1;
    try {
      const started = beginRecordingAttempt(projectId, { attempt_id: token, action, recipe, job_id: originalId ?? null }, intent);
      cacheOwner(token);
      stateRef.current = started; if (live()) { setState(started); setAttempt({ state: 'pending', message: intent }); }
      const stillSelected = () => live() && cachedOwner() === token && generation.current === version && stateRef.current.selected_job_id === originalId && ownsRecordingAttempt(projectId, token);
      const job = await cancelRecording(source, stillSelected);
      received = job;
      // Cache ownership cannot throw on a denied storage read after a valid ACK.
      // A successor owns a different token; old responses cannot retain its receipt.
      if (cachedOwner() !== token) return;
      if (live()) setAccepted(recordingReceipt(job, projectId, recipe, originalId));
      const completed = finishRecordingAttempt(projectId, token, job, live() && generation.current === version);
      if (completed) {
        cacheOwner(null);
        stateRef.current = completed.state;
        if (live()) { setState(completed.state); setAccepted(completed.receipt); setAttempt(null); setConsent(false); setCancelId(''); }
      }
      // A late old completion cannot touch a remounted controller's receipt or journal.
    } catch (cause) {
      // First-write admission failures have no persisted owner but must still
      // disable this controller before any request can be attempted again.
      if (cause instanceof RecordingStorageUnavailable && live()) storageFailure();
      if (cachedOwner() !== token) return;
      let owned = false;
      try { owned = ownsRecordingAttempt(projectId, token); } catch { if (live()) storageFailure(); }
      if (owned) {
        attemptVersion.current += 1;
        if (live()) { setError(`${intent} ${message(cause)}`); setReviewed(false); reviewedToken.current = undefined; }
        if (cause instanceof RecordingStorageUnavailable) {
          if (live()) storageFailure();
          // An admitted receipt may have saved before a later cleanup write failed.
          try { const receipt = readRecordingReceipt(projectId); if (live() && receipt && received && receipt.id === received.id && ownsRecordingAttempt(projectId, token)) setAccepted(receipt); } catch { /* Keep recovery disabled. */ }
        } else {
          try {
            const uncertain = cause instanceof UncertainPolicyJob ? `${intent} ${message(cause)}` : null;
            const changed = failRecordingAttempt(projectId, token, uncertain);
            if (changed) { cacheOwner(changed.pending?.attempt_id ?? null); stateRef.current = changed; if (live()) { setState(changed); setAttempt(uncertain ? { state: 'uncertain', message: uncertain } : null); } }
          } catch { if (live()) storageFailure(); }
        }
      } else if (live() && !stateRef.current.pending && !(cause instanceof RecordingStorageUnavailable)) {
        // Admission may have refused a different controller's existing request.
        setError(message(cause)); setConsent(false);
      }
    } finally {
      busy.current = false; if (live()) setPending(false);
      void client.invalidateQueries({ queryKey: ['jobs', projectId] }); void client.invalidateQueries({ queryKey: ['recording-history', projectId] }); void client.invalidateQueries({ queryKey: ['recording-job', projectId] });
    }
  }
  function acknowledge() {
    if (!storageReady || !reviewed || reviewedToken.current === undefined || !attempt || attempt.state !== 'uncertain' || jobs.isError || knownActive || busy.current) return;
    try {
      const cleared = clearRecordingAttempt(projectId, reviewedToken.current);
      if (!cleared) { setReviewed(false); reviewedToken.current = undefined; setError('Recording recovery changed. Refresh and inspect it again.'); return; }
      cacheOwner(null);
      stateRef.current = cleared; setState(cleared); setAttempt(null); setReviewed(false); reviewedToken.current = undefined; setError('');
    } catch { storageFailure(); }
  }
  async function handoff(kind: 'inspect' | 'train') {
    if (!selected || !result || busy.current || selectedQuery.isError) return;
    const version = generation.current, id = selected.id; busy.current = true; setPending(true); setError('');
    try {
      const fresh = await readRecordingJob(projectId, id, selected.request.recordings);
      if (!live() || generation.current !== version || stateRef.current.selected_job_id !== id) return;
      if (!fresh.result || fresh.status !== 'succeeded') throw new Error('The selected dataset is no longer ready.');
      if (kind === 'train' && (fresh.result.recording_preparation?.lineage_group_count ?? 0) < 2) throw new Error('Held-out training requires at least two declared lineage groups.');
      client.setQueryData<Job[]>(['jobs', projectId], previous => [fresh, ...(previous ?? []).filter(j => j.id !== id)]);
      if (kind === 'train') onTrain(fresh); else onInspect(fresh);
    } catch (cause) { if (live()) setError(message(cause)); }
    finally { busy.current = false; if (live()) setPending(false); }
  }
  return <section className="panel native-simulation" aria-label="Recording dataset preparation" style={{ overflowWrap: 'anywhere' }}>
    <div className="cloud-heading"><div><h2>Prepare a dataset from recordings</h2><p>Choose finalized episodes published for this project. Preparation verifies the original files and creates an immutable dataset; it does not train a policy.</p></div><button type="button" className="secondary-button" disabled={!projectId || jobs.isFetching || pending} onClick={() => void refresh()}>Refresh recordings and jobs</button></div>
    {!projectId && <p role="status">Select a project to see its published recordings.</p>}
    {context.isPending && projectId && <p role="status">Loading recording configuration…</p>}
    {context.isError && <p role="alert">Recording availability is unverified. {context.error.message}</p>}
    {context.data && <p role="status">{context.data.options.setup_message}</p>}
    {context.data?.options.configured === false && <p>Remote Isaac captures need an operator-published copy on this application host. This panel does not transfer captures or configure the simulator.</p>}
    {teachingCapture?.project_id === projectId && <aside className="warning-box" aria-label="Published teaching capture">
      <p>Teaching session {teachingCapture.session_id.slice(0, 8)} published {teachingCapture.episodes.length} episodes. Review them for a dataset; nothing is selected or submitted automatically.</p>
      <p>Selecting these episodes replaces the current draft selection and clears preparation consent. It preserves saved jobs and still requires the normal source verification before preparation.</p>
      <div className="native-result-actions"><button type="button" className="secondary-button" disabled={pending || !storageReady || !submission.available || submission.busy || !!submission.attempt || !!attempt || !timeoutValid} onClick={() => void selectTeachingCapture()}>Select these recorded episodes</button>
        <button type="button" className="text-link" disabled={pending} onClick={() => onTeachingCaptureConsumed?.()}>Dismiss teaching selection</button></div>
    </aside>}
    {jobs.isError && <p role="alert">Saved preparation history is unavailable; displayed status may be stale.</p>}
    {error && <p role="alert">{error}</p>}
    <DatasetSubmissionRecovery submission={submission} onReconcile={() => prepare('reconcile')} onRetry={() => prepare('retry')} onReviewHistory={reviewSubmissionHistory} />
    {submission.hydrated && !submission.available && <p role="alert">Intake recovery is unavailable. Restore browser storage and reload before preparing another dataset.</p>}
    {submission.receipt && !isRecording(submission.receipt) && <p role="status">Recovered dataset intake {submission.receipt.id}. Open Dataset to inspect this original request; it is not a recording preparation.</p>}
    {accepted && (accepted.id !== selectedId || !storageReady) && <p role="status">Retained acknowledgement: job {accepted.id} · {accepted.status}. {storageReady ? 'Select it in saved preparations to verify its current result.' : 'Restore session recovery and inspect saved jobs to verify its current result.'}</p>}
    {sharedReceipt && sharedReceipt.id !== accepted?.id && sharedReceipt.id !== selectedId && <p role="status">Retained acknowledgement: job {sharedReceipt.id} · {sharedReceipt.status} at intake acceptance. Select it to verify its current state.</p>}
    {!storageReady && projectId && <div className="warning-box"><p>Recording session recovery is not ready. No preparation or cancellation can be submitted.</p><button className="secondary-button" disabled={pending} onClick={loadSession}>Read session recovery again</button></div>}
    {storageReady && attempt && <div className="warning-box" role={attempt.state === 'uncertain' ? 'alert' : 'status'}><p>{attempt.message}</p>{attempt.state === 'uncertain' && <><p>Refresh and inspect saved jobs. A missing response does not mean the request failed.</p>{knownActive && <p>Known active preparation: {knownActive.id} · {knownActive.status}. Wait for a terminal state before clearing uncertainty.</p>}<button className="secondary-button" disabled={!reviewed || jobs.isError || !!knownActive || pending} onClick={acknowledge}>I checked recording jobs; allow a new request</button></>}</div>}
    <label className="distillation-history">Saved recording preparation<select aria-label="Saved recording preparation" value={selectedId} disabled={!storageReady} onChange={event => chooseJob(event.target.value)}><option value="">Choose a saved preparation</option>{sharedReceipt && sharedReceipt.id !== accepted?.id && !history.some(j => j.id === sharedReceipt.id) && <option value={sharedReceipt.id}>{sharedReceipt.id.slice(0, 8)} · intake acknowledged {sharedReceipt.status}</option>}{accepted && !history.some(j => j.id === accepted.id) && <option value={accepted.id}>{accepted.id.slice(0, 8)} · acknowledged {accepted.status}</option>}{history.map(j => <option key={j.id} value={j.id}>{j.id.slice(0, 8)} · {j.status} · {j.request.recordings.captures.reduce((n, c) => n + c.episodes.length, 0)} episodes</option>)}</select></label>
    {selectedId && selectedQuery.isPending && <p role="status">Loading the exact saved preparation…</p>}
    {selectedQuery.isError && <p role="alert">Saved preparation could not be verified. {selectedQuery.error.message}</p>}
    {selected && <article aria-label="Saved recording preparation details" data-job-id={selected.id} className="native-simulation-result"><h3>Saved preparation · {selected.id.slice(0, 8)}</h3><p>{selectedQuery.isError ? 'Last observed: ' : ''}{selected.status} · {selected.stage ?? selected.id}</p>{selected.error && <p role="alert">{selected.error}</p>}
      {result && <><p>{result.total_episodes} episodes · {result.total_frames} frames · {result.recording_preparation!.lineage_group_count} declared lineage groups</p><p>Original joint-position targets and native radians are preserved. Readback verifies dataset bytes, not robot task success or compatibility with unrelated checkpoint processors.</p><p>Snapshot: {result.snapshot!.id}</p>{result.warnings.map((warning, i) => <p key={`${i}:${warning}`} className="warning-box">{warning}</p>)}{result.recording_preparation!.lineage_group_count < 2 && <p role="status">This dataset has one lineage group. It can be inspected, but cannot provide the independent held-out training split.</p>}<div className="native-result-actions"><button className="secondary-button" disabled={pending} onClick={() => void handoff('inspect')}>View this dataset</button><button className="primary-button" disabled={pending || result.recording_preparation!.lineage_group_count < 2} onClick={() => void handoff('train')}>Train on this dataset</button></div></>}
      {active(selected) && <button className="secondary-button" disabled={pending || submission.busy || !!submission.attempt || !!attempt || !storageReady || selectedQuery.isError} onClick={() => setCancelId(selected.id)}>Cancel selected preparation</button>}
      {cancelId === selected.id && active(selected) && <div className="warning-box"><p>Cancel job {selected.id} and await its owned local writer cleanup? Original recordings are retained.</p><button className="secondary-button" disabled={pending || submission.busy || !!submission.attempt || !!attempt} onClick={() => void cancel()}>Confirm preparation cancellation</button><button className="text-link" disabled={pending} onClick={() => setCancelId('')}>Keep preparing</button></div>}
    </article>}
    <WorkbenchDisclosure title={selected ? 'Prepare another recording dataset' : 'Choose recorded episodes'} initiallyOpen={!selected}>
      <p>Catalog entries are finalized metadata candidates. Full contents, FFmpeg and the pinned writer are verified only during preparation. Keep the published capture folders unchanged until it finishes.</p>
      {catalog && !catalog.captures.length && <p role="status">No finalized episodes are published for this project.</p>}
      {stale && <div className="warning-box" role="alert"><p>The saved selection is unavailable or its metadata changed. It has not been replaced with new source bytes.</p><button className="secondary-button" disabled={!!attempt || !!submission.attempt || submission.busy || pending || !storageReady} onClick={() => changeSelection(null)}>Discard stale recording selection</button></div>}
      <fieldset className="native-simulation-form" disabled={!projectId || !storageReady || !submission.available || !!submission.attempt || submission.busy || !!attempt || pending || !catalog || stale}><legend>Exact episode selection</legend>
        {catalog?.captures.map(c => <fieldset key={c.session_id} className="native-simulation-form"><legend>{c.origin === 'synthetic' ? 'Synthetic capture' : 'Recorded capture'} · {c.session_id.slice(0, 8)}</legend><p>Group {c.lineage_group} · {c.width} × {c.height} · {c.fps} fps · radians</p>{c.episodes.map(e => <label key={e.episode_id} className="native-confirm"><input type="checkbox" checked={state.recipe?.captures.some(item => item.session_id === c.session_id && item.episodes.some(row => row.episode_id === e.episode_id)) ?? false} onChange={event => toggle(c.session_id, e.episode_id, event.target.checked)} />Episode {e.episode_id} · {e.frames} frames · {e.outcome === 'unknown' ? 'outcome unknown' : 'operator-reported failure'}</label>)}</fieldset>)}
        <label>Preparation timeout (seconds)<input type="number" min={60} max={1800} value={timeout} onChange={event => { const next = event.target.value; setTimeoutValue(next); setConsent(false); generation.current += 1; if (/^\d+$/.test(next) && Number(next) >= 60 && Number(next) <= 1800 && stateRef.current.recipe) changeSelection({ ...stateRef.current.recipe, timeout_seconds: Number(next) }); }} /></label>
        <p>{count} of at most 100 episodes selected. Same-group episodes do not become independent training/validation data.</p>
        {!timeoutValid && <p role="alert">Choose a whole-number timeout from 60 to 1800 seconds.</p>}
        <label className="native-confirm"><input type="checkbox" checked={consent} onChange={event => setConsent(event.target.checked)} />I reviewed these exact published episodes and will keep their capture files unchanged during preparation.</label>
        <button type="button" className="primary-button" disabled={!ready} onClick={() => void prepare('submit')}>Prepare selected recordings</button>
      </fieldset>
    </WorkbenchDisclosure>
  </section>;
}
