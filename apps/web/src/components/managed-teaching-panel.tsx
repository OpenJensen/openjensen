'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import type { Job } from '@/lib/api';
import { submissionOf, useDurableSubmission } from '@/lib/durable-submission';
import {
  managedTeachingJob, managedTeachingRequest, managedTeachingTransport, publishedCapture,
  readManagedHistory, readManagedOptions, readManagedStatus, requireManagedProfile, retainedManagedTeachingReceipt, stopManagedSession,
  type ManagedPublishedCapture, type ManagedTeachingStatus,
} from '@/lib/managed-teaching';
import { TeachingPanel } from './teaching-panel';
import './workbench-form.css';
import './teaching-panel.css';

const active = (job: Job) => job.status === 'queued' || job.status === 'running';
const message = (cause: unknown) => cause instanceof Error ? cause.message : 'Teaching request could not be verified.';
type StopState = { pending: boolean; uncertain: boolean; message: string };
const noStop: StopState = { pending: false, uncertain: false, message: '' };

/** Parent keys this controller by project. Navigation never stops a saved session. */
export function ManagedTeachingPanel({ projectId, onPublishedCapture }: {
  projectId: string; onPublishedCapture: (capture: ManagedPublishedCapture) => void | Promise<void>;
}) {
  const client = useQueryClient();
  const submission = useDurableSubmission({ project: projectId, operation: 'teaching.capture' });
  const mounted = useRef(false), generation = useRef(0), working = useRef(false), selectedRef = useRef('');
  const [profileId, setProfileId] = useState(''), [seconds, setSeconds] = useState('300');
  const [consent, setConsent] = useState(false), [selectedId, setSelectedId] = useState('');
  const [error, setError] = useState(''), [busy, setBusy] = useState(false), [confirmStop, setConfirmStop] = useState(false);
  const historyReview = useRef<unknown>(null);
  const [reviewed, setReviewed] = useState(false);
  selectedRef.current = selectedId;
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; generation.current += 1; }; }, []);
  const profiles = useQuery({ queryKey: ['managed-teaching-profiles', projectId], queryFn: () => readManagedOptions(projectId), enabled: !!projectId, retry: false });
  const history = useQuery({ queryKey: ['managed-teaching-history', projectId], queryFn: () => readManagedHistory(projectId), enabled: !!projectId, retry: false, refetchInterval: 3000 });
  const accepted = (() => { try { return submission.receipt ? retainedManagedTeachingReceipt(submission.receipt, projectId) : null; } catch { return null; } })();
  const retainedOnly = !!accepted && (history.isError || !history.data?.some(job => job.id === accepted.id));
  const jobs = [...(accepted && retainedOnly ? [accepted] : []), ...(history.data ?? []).filter(job => !retainedOnly || job.id !== accepted?.id)];
  const knownActive = jobs.find(active);
  const saved = jobs.find(j => j.id === selectedId);
  const selected = useQuery({ queryKey: ['managed-teaching-status', projectId, selectedId], queryFn: () => readManagedStatus(projectId, selectedId, saved?.request), enabled: !!projectId && !!selectedId, retry: false, refetchInterval: query => query.state.data && !active(query.state.data.job) ? false : 1000 });
  const stopKey = ['managed-teaching-stop', projectId, selectedId];
  const stop = useQuery<StopState>({ queryKey: stopKey, queryFn: async () => noStop, initialData: noStop, enabled: false, gcTime: Infinity });
  const status = selected.isSuccess && !selected.isError ? selected.data : null;
  const stopState = stop.data ?? noStop;
  // An explicitly selected missing job never falls back to a different session.
  const transport = useMemo(() => status?.ready && status.session_id && !stopState.pending && !stopState.uncertain
    ? managedTeachingTransport(projectId, status.job.id, status.session_id, managedTeachingRequest(status.job.request)) : undefined,
  [projectId, status, stopState.pending, stopState.uncertain]);
  const identity = submissionOf(submission.attempt);
  const profile = profiles.data?.profiles.find(p => p.id === profileId);
  const secondsValid = /^\d+$/.test(seconds) && Number.isSafeInteger(Number(seconds)) && Number(seconds) >= 1 && Number(seconds) <= (profile?.max_seconds ?? 0);
  const ready = !!projectId && profiles.isSuccess && !profiles.isError && profiles.data.available && !!profile && secondsValid && consent && history.isSuccess && !history.isError && !knownActive && submission.hydrated && submission.available && !submission.attempt && !submission.busy && !busy;

  function choose(id: string) { generation.current += 1; setSelectedId(id); selectedRef.current = id; setConfirmStop(false); setError(''); }
  function edit() { generation.current += 1; setConsent(false); setReviewed(false); historyReview.current = null; }
  const live = (version: number) => mounted.current && generation.current === version;
  function cacheStatus(value: ManagedTeachingStatus) {
    client.setQueryData(['managed-teaching-status', projectId, value.job.id], value);
    void client.invalidateQueries({ queryKey: ['managed-teaching-history', projectId] });
  }
  async function submit(mode: 'start' | 'check' | 'retry') {
    if (working.current || !projectId || (mode === 'start' && !ready)) return;
    const version = generation.current;
    working.current = true; setBusy(true); setError(''); setConsent(false);
    try {
      const validate = (value: unknown, original: Record<string, unknown>) => managedTeachingJob(value, projectId, original);
      const beforePost = async (original: Record<string, unknown>) => {
        const originalRequest = managedTeachingRequest(original), options = await readManagedOptions(projectId);
        requireManagedProfile(options, originalRequest);
        if (!live(version)) throw new Error('Teaching selection changed before starting; no session was requested.');
      };
      const job = mode === 'check' ? await submission.reconcile(validate) : mode === 'retry' ? await submission.retry(validate, beforePost)
        : await submission.submit(managedTeachingRequest({ operation: 'teaching.capture', profile_id: profileId, profile_sha256: profile?.profile_sha256, timeout_seconds: Number(seconds) }), validate, beforePost);
      if (job && live(version)) choose(job.id);
      void client.invalidateQueries({ queryKey: ['managed-teaching-history', projectId] });
    } catch (cause) { if (live(version)) setError(message(cause)); }
    finally { working.current = false; if (mounted.current) setBusy(false); }
  }
  async function refresh() {
    const version = generation.current, attempt = submission.attempt;
    historyReview.current = null; setReviewed(false); setConsent(false);
    const result = await history.refetch();
    void profiles.refetch(); if (selectedId) void selected.refetch();
    if (live(version) && result.isSuccess && !result.isError && !result.data.some(active) && attempt && !submissionOf(attempt)) { historyReview.current = attempt; setReviewed(true); }
  }
  async function acknowledgeLegacy() {
    if (!reviewed || working.current || historyReview.current !== submission.attempt || identity) return;
    const version = generation.current, attempt = submission.attempt;
    try {
      const fresh = await readManagedHistory(projectId);
      if (!live(version) || attempt !== historyReview.current || fresh.some(active)) throw new Error('Review the current teaching jobs before clearing recovery.');
      if (!submission.clearLegacy(attempt, fresh)) throw new Error('Teaching recovery changed. Refresh and review it again.');
      setReviewed(false); historyReview.current = null;
    } catch (cause) { if (live(version)) setError(message(cause)); }
  }
  async function stopAndPublish() {
    if (working.current || !confirmStop || !status || !active(status.job)) return;
    const version = generation.current, job = status.job, key = ['managed-teaching-stop', projectId, job.id];
    working.current = true; setBusy(true); setError('');
    try {
      const fresh = await readManagedStatus(projectId, job.id, job.request);
      if (!live(version) || selectedRef.current !== job.id) return;
      if (!active(fresh.job)) { cacheStatus(fresh); return; }
      client.setQueryData<StopState>(key, { pending: true, uncertain: false, message: 'Stop requested. Waiting for verified capture publication.' });
      const result = await stopManagedSession(projectId, job.id, job.request);
      cacheStatus(result);
      client.setQueryData<StopState>(key, { pending: false, uncertain: false, message: active(result.job) ? 'Stop acknowledged; capture verification and publication are still pending.' : 'Session ended. Review the saved result.' });
    } catch (cause) {
      client.setQueryData<StopState>(key, { pending: false, uncertain: true, message: 'Stop outcome is unverified. Check the saved session; do not assume the capture was published. You may explicitly repeat Stop for this same job.' });
      if (live(version)) setError(message(cause));
    } finally { working.current = false; if (mounted.current) { setBusy(false); setConfirmStop(false); } }
  }
  async function reviewCapture() {
    if (working.current || !status || status.job.status !== 'succeeded') return;
    const version = generation.current, job = status.job;
    working.current = true; setBusy(true); setError('');
    try {
      const fresh = await readManagedStatus(projectId, job.id, job.request);
      const capture = publishedCapture(fresh.job);
      if (!live(version) || selectedRef.current !== job.id) return;
      cacheStatus(fresh); await onPublishedCapture(capture);
    } catch (cause) { if (live(version)) setError(message(cause)); }
    finally { working.current = false; if (mounted.current) setBusy(false); }
  }
  return <section className="panel teaching-panel" aria-label="Managed teaching sessions">
    <div className="cloud-heading"><div><h2>Record a teaching session</h2><p>Start one configured local Isaac session, teach with its controls, then stop and publish finalized recordings.</p></div><button type="button" className="secondary-button" disabled={!projectId || busy || history.isFetching} onClick={() => void refresh()}>Refresh teaching sessions</button></div>
    {!projectId && <p role="status">Select a project to see its configured teaching profiles.</p>}
    {profiles.data && <p role="status">{profiles.data.message}</p>}
    {profiles.error && <p role="alert">Teaching profiles could not be checked. Starting is disabled.</p>}
    {history.error && <p role="alert">Saved teaching sessions could not be checked. Starting is disabled.</p>}
    {accepted && retainedOnly && <p role="status">Acknowledgement retained for job {accepted.id} ({accepted.status}). Select this job to check its current session and publication; this receipt alone does not verify either.</p>}
    <form className="workbench-form" onSubmit={event => { event.preventDefault(); if (ready) void submit('start'); }}>
      <div className="workbench-field"><label htmlFor={`teaching-profile-${projectId}`}>Teaching profile</label><select id={`teaching-profile-${projectId}`} value={profileId} disabled={!profiles.data?.available || busy || submission.busy || !!submission.attempt} onChange={event => { edit(); setProfileId(event.target.value); }}><option value="">Choose a configured profile</option>{profiles.data?.profiles.map(p => <option key={p.id} value={p.id}>{p.label}</option>)}</select></div>
      <div className="workbench-field"><label htmlFor={`teaching-time-${projectId}`}>Session time limit (seconds)</label><input id={`teaching-time-${projectId}`} inputMode="numeric" value={seconds} disabled={busy || submission.busy || !!submission.attempt} onChange={event => { edit(); setSeconds(event.target.value); }} /></div>
      {profile && <p>Maximum {profile.max_seconds} seconds · capture limit {(profile.max_capture_bytes / 1024 ** 2).toFixed(0)} MiB. Runtime readiness is checked after starting. Stop and publish before the deadline; a timeout does not publish an active capture.</p>}
      <label className="native-confirm"><input type="checkbox" checked={consent} disabled={busy || submission.busy || !!submission.attempt} onChange={event => setConsent(event.target.checked)} />Start this configured session and record simulator observations. This does not start training or verify task success.</label>
      <button type="submit" className="primary-button" disabled={!ready}>Start teaching session</button>
    </form>
    {knownActive && <p role="status">A saved teaching session is {knownActive.status}: {knownActive.id}. Select it below to inspect or stop it.</p>}
    {submission.error && <p role="alert">{submission.error}</p>}
    {submission.attempt && <div className="warning-box"><p>{submission.attempt.message}</p>{identity ? <><p>Saved profile {String(identity.body.profile_id)} · {String(identity.body.timeout_seconds)} seconds. Recovery retains this exact request.</p><button type="button" className="secondary-button" disabled={busy || submission.busy} onClick={() => void submit('check')}>Check saved teaching request</button>{submission.canRetry && <button type="button" className="secondary-button" disabled={busy || submission.busy} onClick={() => void submit('retry')}>Retry same teaching request</button>}</> : <button type="button" className="secondary-button" disabled={!reviewed || busy || history.isError || !!knownActive} onClick={() => void acknowledgeLegacy()}>I reviewed teaching jobs; clear legacy recovery</button>}</div>}
    <div className="workbench-field"><label htmlFor={`teaching-saved-${projectId}`}>Saved teaching session</label><select id={`teaching-saved-${projectId}`} value={selectedId} onChange={event => choose(event.target.value)}><option value="">Choose a saved session</option>{jobs.map(job => <option key={job.id} value={job.id}>{job.id} · {job.status}</option>)}</select></div>
    {selectedId && selected.isPending && <p role="status">Checking selected session…</p>}
    {selectedId && selected.error && <p role="alert">The selected session is unavailable or changed. No other session was selected.</p>}
    {error && <p role="alert">{error}</p>}
    {status && <section aria-label="Selected managed session">
      <p>Job {status.job.id} · {status.job.status} · {status.job.stage ?? 'waiting'}</p>
      {status.job.error && <p role="alert">{status.job.error}</p>}
      {active(status.job) && stopState.message && <p role={stopState.uncertain ? 'alert' : 'status'}>{stopState.message}</p>}
      {active(status.job) && status.stop_requested && <p role="status">Stopping and verifying this capture. Publication is not confirmed yet.</p>}
      {transport && <TeachingPanel transport={transport} />}
      {active(status.job) && !status.stop_requested && <div className="workbench-actions"><button type="button" className="secondary-button" disabled={busy || stopState.pending} onClick={() => setConfirmStop(true)}>Stop and publish</button></div>}
      {confirmStop && active(status.job) && <div className="warning-box"><p>Stop only session {status.job.id}, await its process cleanup and verify finalized captures before publication? Unmounting this panel alone does not stop it.</p><button type="button" className="primary-button" disabled={busy} onClick={() => void stopAndPublish()}>Confirm stop and publish</button><button type="button" className="text-link" disabled={busy} onClick={() => setConfirmStop(false)}>Keep session running</button></div>}
      {status.job.status === 'succeeded' && <><p>Capture {publishedCapture(status.job).session_id} published with {publishedCapture(status.job).episodes.length} finalized episodes. Simulator coordinates are preserved; physical calibration and task success are unverified.</p><button type="button" className="primary-button" disabled={busy} onClick={() => void reviewCapture()}>Review for dataset preparation</button></>}
      {['failed', 'cancelled', 'interrupted'].includes(status.job.status) && <p>Capture publication is not confirmed by this job. Inspect its saved error and retained evidence; no retry or dataset preparation was started.</p>}
    </section>}
  </section>;
}
