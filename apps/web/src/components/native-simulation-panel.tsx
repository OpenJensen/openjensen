'use client';

import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api, artifactDownloadUrl, isActive } from '@/lib/api';
import { storeAttempt, type PolicyJobAttempt } from '@/lib/policy-job-attempt';
import { minimalSimulationReceipt, simulationAttemptOperation, simulationIntentMessage, storedSimulationAttempt, storedSimulationReceipt, storeSimulationReceipt, type SimulationIntent } from '@/lib/native-simulation-recovery';
import { Icon } from '@/components/icon';
import './simulation-workspace.css';
import { NativePreparation } from './native-preparation';
import { publicPath } from '@/lib/base-path';
import { WorkbenchDisclosure } from '@/components/workbench-disclosure';
import { cancelSimulation, isSimulationJob, nativeInput, simulationOptions, simulationTarget, simulationTaskSummary, simulationVideoUrl, startSimulation, UncertainSubmission, uploadModel, type NativeArtifact, type SimulationJob } from '@/lib/native-simulation';

class SimulationJournalUnavailable extends Error {
  constructor() { super('Browser session storage is unavailable or its simulation receipt is unreadable. Restore it and reload, then inspect recorded jobs before another request.'); }
}

export function NativeSimulationPanel({ projectId, preferredJobId, onJobSelected, onTraining }: { projectId: string; preferredJobId?: string; onJobSelected?: (id: string) => void; onTraining: () => void }) {
  const client = useQueryClient();
  const options = useQuery({ queryKey: ['simulation-options'], queryFn: simulationOptions, retry: false, refetchInterval: 10_000 });
  const jobs = useQuery({ queryKey: ['jobs', projectId], queryFn: () => api.jobs(projectId), enabled: !!projectId, retry: false, refetchInterval: 3_000 });
  const artifacts = useQuery({ queryKey: ['artifacts', projectId], queryFn: () => api.artifacts(projectId), enabled: !!projectId, retry: false, refetchInterval: 3_000 });
  const [profileId, setProfileId] = useState('');
  const profile = options.data?.profiles.find(item => item.id === profileId) ?? options.data?.profiles[0];
  const [artifactId, setArtifactId] = useState('');
  const inputs = (artifacts.data ?? []).filter(item => nativeInput(item, projectId, profile));
  const input = inputs.find(item => item.id === artifactId);
  const [file, setFile] = useState<File | null>(null);
  const [policySource, setPolicySource] = useState<'saved' | 'upload'>('saved');
  const [timeout, setTimeoutValue] = useState('7200');
  const [experimentalProfile, setExperimentalProfile] = useState<string | null>(null);
  const experimental = !!profile && experimentalProfile === profile.id;
  const setExperimental = (accepted: boolean) => setExperimentalProfile(accepted ? profile?.id ?? null : null);
  const [pending, setPending] = useState<'upload' | 'run' | 'cancel' | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');
  const attemptKey = ['native-simulation-attempt', projectId];
  const acceptedKey = ['native-simulation-accepted', projectId];
  const storageKey = ['native-simulation-storage-error', projectId];
  const attempt = useQuery<PolicyJobAttempt>({ queryKey: attemptKey, queryFn: async () => null, enabled: false, initialData: null, gcTime: Infinity });
  const accepted = useQuery<SimulationJob | null>({ queryKey: acceptedKey, queryFn: async () => null, enabled: false, initialData: null, gcTime: Infinity });
  const storageError = useQuery<string | null>({ queryKey: storageKey, queryFn: async () => null, enabled: false, initialData: null, gcTime: Infinity });
  const [journalProject, setJournalProject] = useState('');
  const [reviewedAttempt, setReviewedAttempt] = useState<PolicyJobAttempt>(null);
  const [jobId, setJobId] = useState(preferredJobId ?? '');
  const [confirmCancel, setConfirmCancel] = useState<string | null>(null);
  const [videoFailed, setVideoFailed] = useState(false);
  const selection = useRef({ id: preferredJobId ?? '', generation: 0 });
  const busy = useRef(false);
  const mounted = useRef(true);
  const abortUpload = useRef<(() => void) | null>(null);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; abortUpload.current?.(); }; }, []);
  useEffect(() => {
    try {
      if (!client.getQueryData<PolicyJobAttempt>(['native-simulation-attempt', projectId])) client.setQueryData(['native-simulation-attempt', projectId], storedSimulationAttempt(projectId));
      if (!client.getQueryData<SimulationJob | null>(['native-simulation-accepted', projectId])) client.setQueryData(['native-simulation-accepted', projectId], storedSimulationReceipt(projectId));
      setJournalProject(projectId);
    } catch { client.setQueryData(['native-simulation-storage-error', projectId], new SimulationJournalUnavailable().message); }
  }, [client, projectId]);
  const saved = (jobs.data ?? []).filter(item => item.project_id === projectId && isSimulationJob(item)).sort((a, b) => b.created_at.localeCompare(a.created_at));
  const retained = accepted.data?.project_id === projectId ? accepted.data : null;
  const receipt = retained ? saved.find(item => item.id === retained.id) ?? retained : null;
  const selected = saved.find(item => item.id === jobId) ?? (retained?.id === jobId ? retained : undefined);
  // A terminal transition must fetch final events even when the active poll was empty.
  // Job status has a fixed six-value domain; timestamps would grow this cache unboundedly.
  const events = useQuery({ queryKey: ['events', selected?.id, selected?.status], queryFn: () => api.events(selected!.id), enabled: !!selected, retry: false, refetchInterval: selected && isActive(selected) ? 2_000 : false });
  const result = selected?.result && 'reports' in selected.result ? selected.result : null;
  const outputs = (result?.artifacts ?? []) as NativeArtifact[];
  const report = result?.reports?.find(item => item.stage === 'simulation');
  const records = outputs.filter(item => item.project_id === projectId && item.job_id === selected?.id && item.format === 'simulation_record');
  // Media is served from a verified job-owned record, never a URL inside a report.
  const videoReady = selected?.status === 'succeeded' && records.length > 0 && Array.isArray(report?.artifacts) && report.artifacts.some((item: unknown) => typeof item === 'object' && item !== null && 'path' in item && typeof item.path === 'string' && item.path === 'artifacts/outputs/video.mp4');
  const observedTasks = [...(events.data ?? [])].reverse().map(event => simulationTaskSummary(event.data)).find(Boolean);
  const seconds = Number(timeout);
  const timeoutValid = /^\d+$/.test(timeout) && Number.isSafeInteger(seconds) && seconds >= 30 && seconds <= 7200;
  const journalReady = journalProject === projectId && !storageError.data;
  const ready = !!projectId && journalReady && jobs.isSuccess && !jobs.isError && options.isSuccess && !options.isError && !!profile && !pending && !attempt.data;
  const refresh = async () => {
    const before = client.getQueryData<PolicyJobAttempt>(attemptKey);
    const response = await jobs.refetch();
    void artifacts.refetch(); void options.refetch(); if (selected) void events.refetch();
    if (mounted.current && before?.state === 'uncertain' && before === client.getQueryData<PolicyJobAttempt>(attemptKey) && !response.isError) setReviewedAttempt(before);
  };
  function showJob(id: string) { selection.current = { id, generation: selection.current.generation + 1 }; onJobSelected?.(id); setJobId(id); setConfirmCancel(null); setVideoFailed(false); }
  function storageFailed() {
    const failure = new SimulationJournalUnavailable();
    client.setQueryData(storageKey, failure.message);
    const previous = client.getQueryData<PolicyJobAttempt>(attemptKey);
    if (previous?.state === 'pending') client.setQueryData<PolicyJobAttempt>(attemptKey, (): PolicyJobAttempt => ({ state: 'uncertain', message: `${previous.message} ${failure.message}` }));
    return failure;
  }
  function saveAttempt(value: PolicyJobAttempt) {
    try { storeAttempt(simulationAttemptOperation, projectId, value); } catch { throw storageFailed(); }
    client.setQueryData<PolicyJobAttempt>(attemptKey, value);
  }
  function acknowledgeRecovery() {
    if (!journalReady || jobs.isError || pending || !reviewedAttempt || reviewedAttempt !== client.getQueryData<PolicyJobAttempt>(attemptKey)) return;
    try { saveAttempt(null); setError(''); setReviewedAttempt(null); } catch (cause) { setError((cause as Error).message); }
  }
  async function mutate(kind: 'upload' | 'run' | 'cancel') {
    if (busy.current || !projectId || !journalReady || client.getQueryData<PolicyJobAttempt>(attemptKey) || client.getQueryData<string | null>(storageKey) || (kind !== 'cancel' && !ready)) return;
    if (kind === 'upload' && (!file || !profile || !file.size || file.size > (options.data?.max_archive_bytes ?? 0))) return;
    if (kind === 'run' && (policySource !== 'saved' || !input || !profile || !experimental || !timeoutValid || artifacts.isError || jobs.isError)) return;
    if (kind === 'cancel' && (!selected || selected.id !== confirmCancel || !isActive(selected) || jobs.isError)) return;
    const originalSelection = selection.current;
    const intent: SimulationIntent = kind === 'run' ? { action: 'run', project: projectId, profile: profile!.id, artifact: input!.id, timeout: seconds }
      : kind === 'upload' ? { action: 'upload', project: projectId, profile: profile!.id, filename: file!.name, bytes: file!.size }
      : { action: 'cancel', project: projectId, profile: 'runtime_id' in selected!.request ? selected!.request.runtime_id ?? '' : '', job: selected!.id };
    let context: string;
    try { context = simulationIntentMessage(intent); } catch (cause) { setError((cause as Error).message); return; }
    busy.current = true; setPending(kind); setError(''); setProgress(0); setReviewedAttempt(null);
    try {
      saveAttempt({ state: 'pending', message: context });
      let value: SimulationJob;
      if (kind === 'upload') {
        const transfer = uploadModel(projectId, profile!.id, file!, percent => { if (mounted.current) setProgress(percent); });
        abortUpload.current = transfer.abort; value = await transfer.result;
      } else if (kind === 'run') value = await startSimulation(projectId, profile!.id, input!.id, seconds, input!.manifest_sha256);
      else value = await cancelSimulation(selected!, () => mounted.current && selection.current === originalSelection);
      const acknowledged = minimalSimulationReceipt(value, projectId);
      // Admission survives history/storage failure and unmount; never cache unverified result/media claims.
      client.setQueryData<SimulationJob | null>(acceptedKey, acknowledged);
      if (mounted.current && selection.current === originalSelection) { showJob(acknowledged.id); setExperimental(false); if (kind === 'upload') setPolicySource('saved'); }
      try { storeSimulationReceipt(acknowledged); } catch { throw storageFailed(); }
      saveAttempt(null);
      void client.invalidateQueries({ queryKey: ['jobs', projectId] });
      void client.invalidateQueries({ queryKey: ['artifacts', projectId] });
    } catch (cause) {
      let message = cause instanceof Error ? cause.message : 'The request failed.';
      try {
        if (cause instanceof SimulationJournalUnavailable) { /* Preserve the recovery record; do not retry failed storage writes. */ }
        else if (cause instanceof UncertainSubmission) { message = `${context} ${message}`; saveAttempt({ state: 'uncertain', message }); }
        else saveAttempt(null);
      } catch (failure) { message = failure instanceof Error ? failure.message : message; }
      if (mounted.current) setError(message);
    } finally {
      busy.current = false; abortUpload.current = null;
      if (mounted.current) { setPending(null); setConfirmCancel(null); }
    }
  }
  return <section className="panel native-simulation native-workflow" aria-label="Native Isaac simulation">
    <div className="native-workflow-toolbar"><span className="native-model-badge">Isaac Sim</span><button type="button" className="text-link" aria-label="Refresh simulation jobs" disabled={!projectId || jobs.isFetching} onClick={() => void refresh()}>Refresh</button></div>
    {options.isError && <p className="error-notice" role="alert">Simulation options are unavailable. {options.error.message}</p>}
    {(storageError.data || error) && <p className="error-notice" role="alert">{storageError.data || error}</p>}
    {journalReady && attempt.data?.state === 'pending' && <p role="status">{attempt.data.message} Waiting for the application receipt; another request is blocked.</p>}
    {attempt.data?.state === 'uncertain' && <section aria-label="Native simulation recovery" className="warning-box"><p>{attempt.data.message} Further submissions are paused. Refresh and inspect recorded jobs before explicitly allowing another request.</p><p>Recovery is limited to this browser tab; it does not prevent requests in another tab or prove cloud resources stopped.</p><button className="secondary-button" disabled={!journalReady || reviewedAttempt !== attempt.data || jobs.isError || pending !== null} onClick={acknowledgeRecovery}>I checked the jobs; allow a new request</button></section>}
    {receipt && receipt.id !== selected?.id && <section aria-label="Native submission receipt"><p>Last acknowledged {receipt.kind === 'policy.import' ? 'import' : 'rollout'} · {receipt.id} · {receipt.status}. Recorded history may contain newer details.</p><button className="text-link" onClick={() => showJob(receipt.id)}>View acknowledged job</button></section>}
    {(saved.length > 0 || retained || jobs.isError || preferredJobId) && <section className="native-simulation-history" aria-label="Native simulation jobs">
      {jobs.isError && <p role="alert">Job updates are unavailable. Previously received status may be stale. {jobs.error.message}</p>}
      {!saved.length && !retained && <p>{jobs.isPending && projectId ? 'Loading jobs…' : 'No native imports or simulation runs in this project yet.'}</p>}
      {saved.length > 0 && <label>Saved simulation job<select aria-label="Saved simulation job" value={selected?.id ?? ''} onChange={event => showJob(event.target.value)}><option value="">Choose a recorded job</option>{saved.map(item => <option key={item.id} value={item.id}>{item.kind === 'policy.import' ? 'Policy import' : 'Isaac rollout'} · {item.id.slice(0, 8)} · {item.status}</option>)}</select></label>}
      {preferredJobId && !selected && jobs.isSuccess && <p role="status">The requested job is not available in this project's native simulation history.</p>}
      {selected && <article className="native-simulation-result" aria-label="Native simulation job details" data-job-id={selected.id}>
        <div className="native-result-header"><div><span className="eyebrow">{selected.kind === 'policy.import' ? 'Package intake' : 'Recorded execution'}</span><h3>{selected.kind === 'policy.import' ? 'Native policy import' : 'Isaac rollout'}</h3></div><span className={`status status-${selected.status}`}><span className="status-dot" />{selected.status}</span></div>
        {videoReady && !videoFailed && <figure className="native-recording"><video controls preload="metadata" aria-label="Recorded cup rollout" src={simulationVideoUrl(selected.id)} onError={() => setVideoFailed(true)} /><figcaption><strong>Recorded cup rollout</strong><span>Execution recording · pickup success not measured</span></figcaption></figure>}
        {videoFailed && <p role="alert">The recorded video is unavailable. You can still download the verified simulation record.</p>}
        <dl className="cloud-run-facts"><div><dt>Execution target</dt><dd>{simulationTarget(selected)?.accelerators.join(' + ') ?? (selected.kind === 'policy.import' ? 'Local package validation' : 'Awaiting recorded cloud target')}</dd></div><div><dt>Cup pickup success</dt><dd>Not measured</dd></div><div><dt>Calibration</dt><dd>Unverified</dd></div></dl>
        {events.isError && <p className="error-notice" role="alert">Activity updates are unavailable. Worker states and event history may be stale.</p>}
        {observedTasks && <p className="native-worker-summary" role="status" aria-label="Observed simulation workers">{events.isError ? 'Last observed: ' : ''}{observedTasks}</p>}
        {selected.error && <p role="alert" className="error-notice">{selected.error}</p>}
        {selected.status === 'succeeded' && <p className="native-result-summary">{!saved.some(item => item.id === selected.id) ? 'Request acknowledged. Refresh recorded history to inspect its final output; this cached receipt does not verify a policy package or recording.' : selected.kind === 'policy.import' ? 'The native package is saved and its package integrity checked. Runtime compatibility and task quality remain unverified. Open Prepare another run to select it for an explicit experimental Run.' : 'Execution completed. This is not a scored evaluation or proof of cup pickup.'}</p>}
        {isActive(selected) && <><progress aria-label="Native job in progress" /><button className="secondary-button" disabled={!journalReady || !!attempt.data || pending !== null || jobs.isError} onClick={() => setConfirmCancel(selected.id)}>Cancel selected native job</button></>}
        {confirmCancel === selected.id && isActive(selected) && <div className="warning-box" role="group" aria-label="Confirm native cancellation"><p>Request cancellation of {selected.id}? This does not prove cloud resources have been deleted.</p><button className="secondary-button" disabled={!journalReady || !!attempt.data || pending !== null || jobs.isError} onClick={() => void mutate('cancel')}>Confirm cancellation</button><button className="text-link" disabled={pending !== null} onClick={() => setConfirmCancel(null)}>Keep running</button></div>}
        {selected.status === 'succeeded' && records.map(item => <p key={item.id}><a className="secondary-button" href={artifactDownloadUrl(projectId, item.id)}>Download simulation record</a></p>)}
        <WorkbenchDisclosure key={selected.id} title="Activity and technical details">
          <dl className="native-job-identity"><div><dt>Job ID</dt><dd>{selected.id}</dd></div><div><dt>Last recorded update</dt><dd>{new Date(selected.updated_at).toLocaleString()}</dd></div></dl>
          {events.isError && <p>{events.error.message}</p>}
          <pre className="cloud-log-tail" role="region" aria-label="Native simulation event log" tabIndex={0}>{events.data?.length ? events.data.slice(-100).map(event => `${event.timestamp} · ${event.stage} · ${event.message}${simulationTaskSummary(event.data) ? ` · ${simulationTaskSummary(event.data)}` : ''}`).join('\n') : 'No recorded events yet.'}</pre>
          {report && <details><summary>Recorded simulation report</summary><pre className="cloud-log-tail">{JSON.stringify(report, null, 2)}</pre></details>}
        </WorkbenchDisclosure>
      </article>}
    </section>}
    <NativePreparation key={selected?.id ?? 'first'} title="Prepare another run" hasResult={!!selected}>
      {!projectId || !profile ? <div className="native-setup-empty">
        <p role="status">{!projectId ? 'Select a project to continue.' : options.isPending ? 'Loading scenes…' : options.isError ? 'Scene availability is unknown.' : 'Simulator not connected'}</p>
        {options.data?.unavailable_reason && options.data.unavailable_reason !== 'No Isaac simulation profile is configured.' && <p>{options.data.unavailable_reason}</p>}
        <a className="text-link" href={publicPath('/guide/#run')}>Set up simulation</a>
        <button className="primary-button" disabled>Start experimental simulation</button>
      </div> : <>
    <fieldset disabled={!ready} className="native-simulation-form simulation-setup">
      <legend className="visually-hidden">Choose a native policy</legend>
      <section className="simulation-step" aria-labelledby="simulation-scene-title">
        <h3 id="simulation-scene-title">Scene</h3>
        <div className="simulation-choice-grid" role="radiogroup" aria-label="Isaac profile">
          {options.data?.profiles.map(item => <label className="simulation-choice" key={item.id}>
            <input type="radio" name="simulation-profile" aria-label={item.label} value={item.id} checked={profile?.id === item.id} onChange={() => { setProfileId(item.id); setArtifactId(''); setExperimental(false); }} />
            <span className="simulation-choice-icon"><Icon name="play" /></span><span className="simulation-choice-copy"><strong>{item.label}</strong><span>{item.architectures.map(a => a === 'act' ? 'ACT' : 'SmolVLA').join(' / ')}</span></span>
          </label>)}
        </div>
      </section>
      <section className="simulation-step" aria-labelledby="simulation-policy-title">
        <h3 id="simulation-policy-title">Policy</h3>
        <div className="simulation-choice-grid simulation-source-grid" role="radiogroup" aria-label="Policy source">
          <label className="simulation-choice"><input type="radio" name="simulation-source" aria-label="Saved policies" checked={policySource === 'saved'} onChange={() => setPolicySource('saved')} /><span className="simulation-choice-icon"><Icon name="layers" /></span><span className="simulation-choice-copy"><strong>Saved policies</strong><span>{inputs.length} available</span></span></label>
          <label className="simulation-choice"><input type="radio" name="simulation-source" aria-label="Import package" checked={policySource === 'upload'} onChange={() => { setPolicySource('upload'); setArtifactId(''); setExperimental(false); }} /><span className="simulation-choice-icon"><Icon name="folder" /></span><span className="simulation-choice-copy"><strong>Import package</strong><span>TAR archive</span></span></label>
        </div>
        {policySource === 'upload' ? <div className="native-import">
          <label>Native policy TAR<input type="file" accept=".tar,.tar.gz,.tgz,application/x-tar,application/gzip" onChange={event => setFile(event.target.files?.[0] ?? null)} /></label>
          {file && <p>{file.name} · {(file.size / 1024 ** 2).toFixed(1)} MiB</p>}
          {file && (!file.size || file.size > (options.data?.max_archive_bytes ?? 0)) && <p role="alert">Choose a nonempty TAR no larger than {((options.data?.max_archive_bytes ?? 0) / 1024 ** 3).toFixed(1)} GiB.</p>}
          <WorkbenchDisclosure title="Package requirements"><p>Include safetensors weights, configuration, saved processors and normalization statistics. Import saves and validates locally; no GPU job starts.</p></WorkbenchDisclosure>
          <button type="button" className="secondary-button" disabled={!file || !file.size || file.size > (options.data?.max_archive_bytes ?? 0)} onClick={() => void mutate('upload')}>Upload and validate policy</button>
        </div> : <>
          <div className="simulation-choice-grid" role="radiogroup" aria-label="Native policy">
            {inputs.map(item => <label className="simulation-choice" key={item.id}><input type="radio" name="simulation-policy" aria-label={item.label} value={item.id} checked={input?.id === item.id} onChange={() => { setArtifactId(item.id); setExperimental(false); }} /><span className="simulation-choice-icon"><Icon name="layers" /></span><span className="simulation-choice-copy"><strong>{item.label}</strong><span>{item.metadata?.architecture === 'act' ? 'ACT' : 'SmolVLA'} · {item.id.slice(0, 8)}</span></span></label>)}
          </div>
          {artifacts.isPending && <p role="status">Loading saved policies…</p>}
          {!artifacts.isPending && !artifacts.isError && !inputs.length && <div className="native-setup-empty"><p>No compatible policy yet</p><button className="text-link" type="button" onClick={() => setPolicySource('upload')}>Import a policy</button></div>}
        </>}
      </section>
      {policySource === 'saved' && <section className="simulation-step" aria-labelledby="simulation-launch-title">
        <h3 id="simulation-launch-title" className="visually-hidden">Run settings</h3>
        <label className="simulation-timeout">Timeout (seconds)<input aria-label="Simulation timeout (seconds)" type="number" min="30" max="7200" step="1" value={timeout} onChange={event => { setTimeoutValue(event.target.value); setExperimental(false); }} /></label>
        {!timeoutValid && <p role="alert">Choose a whole number from 30 to 7200 seconds.</p>}
        <p className="simulation-scope-note">Paid L4 + H100 workers. Timeout is not a spending cap.</p>
        <WorkbenchDisclosure title="Cloud limits"><p>Cancellation is supervised; resource deletion is not verified. The job has a maximum two-hour timeout.</p></WorkbenchDisclosure>
        <label className="native-confirm"><input type="checkbox" checked={experimental} onChange={event => setExperimental(event.target.checked)} />I understand this is an experimental, paid cloud rollout with unverified cup pickup and calibration.</label>
        <button type="button" className="primary-button simulation-submit" disabled={policySource !== 'saved' || !input || !experimental || !timeoutValid || artifacts.isError || jobs.isError} onClick={() => void mutate('run')}><Icon name="play" size={17} />Start experimental simulation</button>
      </section>}
    </fieldset>
    <button className="text-link simulation-inline-link" type="button" onClick={onTraining}>Open training checkpoints <Icon name="arrow" size={15} /></button>
    {artifacts.isError && <p role="alert">Saved policies are unavailable. {artifacts.error.message}</p>}
      </>}
    {pending === 'upload' && <div role="status"><p>{progress === 100 ? 'Upload sent; waiting for the import job receipt…' : `Uploading policy: ${progress}%`}</p><progress max="100" value={progress} aria-label="Policy upload progress" /><button className="secondary-button" onClick={() => abortUpload.current?.()}>Stop upload</button></div>}
    {pending === 'run' && <p role="status">Submitting one simulation job…</p>}
    </NativePreparation>
  </section>;
}
