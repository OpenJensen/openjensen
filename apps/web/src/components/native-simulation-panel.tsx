'use client';

import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api, artifactDownloadUrl, isActive } from '@/lib/api';
import { cancelSimulation, isSimulationJob, nativeInput, simulationOptions, simulationTarget, simulationTaskSummary, simulationVideoUrl, startSimulation, UncertainSubmission, uploadModel, type NativeArtifact, type SimulationJob } from '@/lib/native-simulation';

export function NativeSimulationPanel({ projectId, preferredJobId, onTraining }: { projectId: string; preferredJobId?: string; onTraining: () => void }) {
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
  const [timeout, setTimeoutValue] = useState('7200');
  const [experimentalProfile, setExperimentalProfile] = useState<string | null>(null);
  const experimental = !!profile && experimentalProfile === profile.id;
  const setExperimental = (accepted: boolean) => setExperimentalProfile(accepted ? profile?.id ?? null : null);
  const [pending, setPending] = useState<'upload' | 'run' | 'cancel' | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');
  const [ambiguous, setAmbiguous] = useState(false);
  const [reviewed, setReviewed] = useState(false);
  const [jobId, setJobId] = useState(preferredJobId ?? '');
  const [accepted, setAccepted] = useState<SimulationJob | null>(null);
  const [confirmCancel, setConfirmCancel] = useState<string | null>(null);
  const [videoFailed, setVideoFailed] = useState(false);
  const busy = useRef(false);
  const mounted = useRef(true);
  const abortUpload = useRef<(() => void) | null>(null);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; abortUpload.current?.(); }; }, []);
  const saved = (jobs.data ?? []).filter(item => item.project_id === projectId && isSimulationJob(item)).sort((a, b) => b.created_at.localeCompare(a.created_at));
  const selected = saved.find(item => item.id === jobId) ?? (accepted?.id === jobId ? accepted : undefined);
  // A terminal transition must fetch final events even when the active poll was empty.
  // Job status has a fixed six-value domain; timestamps would grow this cache unboundedly.
  const events = useQuery({ queryKey: ['events', selected?.id, selected?.status], queryFn: () => api.events(selected!.id), enabled: !!selected, retry: false, refetchInterval: selected && isActive(selected) ? 2_000 : false });
  const result = selected?.result && 'reports' in selected.result ? selected.result : null;
  const outputs = (result?.artifacts ?? []) as NativeArtifact[];
  const report = result?.reports?.find(item => item.stage === 'simulation');
  const records = outputs.filter(item => item.project_id === projectId && item.job_id === selected?.id && item.format === 'simulation_record');
  // Media is served from a verified job-owned record, never a URL inside a report.
  const videoReady = selected?.status === 'succeeded' && records.length > 0 && Array.isArray(report?.artifacts) && report.artifacts.some((item: unknown) => typeof item === 'object' && item !== null && 'path' in item && typeof item.path === 'string' && item.path === 'artifacts/outputs/video.mp4');
  const seconds = Number(timeout);
  const timeoutValid = /^\d+$/.test(timeout) && Number.isSafeInteger(seconds) && seconds >= 30 && seconds <= 7200;
  const ready = !!projectId && options.isSuccess && !options.isError && !!profile && !pending && !ambiguous;
  const refresh = async () => {
    const response = await jobs.refetch();
    void artifacts.refetch();
    void options.refetch();
    if (selected) void events.refetch();
    if (!response.isError) setReviewed(true);
  };
  function showJob(id: string) { setJobId(id); setConfirmCancel(null); setVideoFailed(false); }
  async function mutate(kind: 'upload' | 'run' | 'cancel') {
    if (busy.current || !projectId || (kind !== 'cancel' && !ready)) return;
    if (kind === 'upload' && (!file || !profile || !file.size || file.size > (options.data?.max_archive_bytes ?? 0))) return;
    if (kind === 'run' && (!input || !profile || !experimental || !timeoutValid || artifacts.isError || jobs.isError)) return;
    if (kind === 'cancel' && (!selected || selected.id !== confirmCancel || !isActive(selected) || jobs.isError)) return;
    busy.current = true; setPending(kind); setError(''); setProgress(0); setReviewed(false);
    try {
      let receipt: SimulationJob;
      if (kind === 'upload') {
        const transfer = uploadModel(projectId, profile!.id, file!, value => { if (mounted.current) setProgress(value); });
        abortUpload.current = transfer.abort;
        receipt = await transfer.result;
      } else if (kind === 'run') receipt = await startSimulation(projectId, profile!.id, input!.id, seconds);
      else receipt = await cancelSimulation(selected!);
      if (mounted.current) { setAccepted(receipt); showJob(receipt.id); setExperimental(false); }
      await client.invalidateQueries({ queryKey: ['jobs', projectId] });
      await client.invalidateQueries({ queryKey: ['artifacts', projectId] });
    } catch (cause) {
      if (mounted.current) { setError(cause instanceof Error ? cause.message : 'The request failed.'); if (cause instanceof UncertainSubmission) setAmbiguous(true); }
    } finally {
      busy.current = false; abortUpload.current = null;
      if (mounted.current) { setPending(null); setConfirmCancel(null); }
    }
  }
  return <section className="panel native-simulation" aria-labelledby="native-simulation-title">
    <div className="cloud-heading"><div><h2 id="native-simulation-title">Native Isaac simulation</h2><p>Run an ACT or SmolVLA policy in the registered SO-101 cup scene. This is an experimental rollout, not a scored benchmark.</p></div><button type="button" className="secondary-button" disabled={!projectId || jobs.isFetching} onClick={() => void refresh()}>Refresh simulation jobs</button></div>
    <p className="warning-box">Launching requests the profile's cloud workers (L4 and H100). The job has a maximum two-hour timeout; cancellation is supervised; resource deletion is not verified. This is not a spending cap. Completed execution does not establish cup pickup success or calibration.</p>
    {!projectId && <p role="status">Select a project before importing a policy or starting simulation.</p>}
    {options.isPending && <p role="status">Loading simulation profiles…</p>}
    {options.isError && <p className="error-notice" role="alert">Simulation options are unavailable. {options.error.message}</p>}
    {options.isSuccess && !options.data.profiles.length && <p role="status">{options.data.unavailable_reason ?? 'No Isaac simulation profile is configured.'} An operator must connect the simulator; this page does not start it automatically.</p>}
    {error && <p className="error-notice" role="alert">{error}</p>}
    {ambiguous && <div className="warning-box"><p>Further submissions are paused. Refresh and inspect the recorded jobs first.</p><button className="secondary-button" disabled={!reviewed || jobs.isError || pending !== null} onClick={() => { setAmbiguous(false); setError(''); setReviewed(false); }}>I checked the jobs; allow a new request</button></div>}
    <fieldset disabled={!ready} className="native-simulation-form">
      <legend>Choose a native policy</legend>
      <label>Isaac profile<select aria-label="Isaac profile" value={profile?.id ?? ''} onChange={event => { setProfileId(event.target.value); setArtifactId(''); setExperimental(false); }}><option value="" disabled>No profile selected</option>{options.data?.profiles.map(item => <option key={item.id} value={item.id}>{item.label} · cup · {item.architectures.map(a => a.toUpperCase()).join(' / ')}</option>)}</select></label>
      <div className="native-import"><h3>Import your weights</h3><p>Choose a complete TAR with safetensors weights, policy configuration, saved processors and normalization statistics. A weights-only file is insufficient. Import validates and copies the package locally; it does not launch a GPU job.</p><label>Native policy TAR<input type="file" accept=".tar,.tar.gz,.tgz,application/x-tar,application/gzip" onChange={event => setFile(event.target.files?.[0] ?? null)} /></label>
        {file && <p>{file.name} · {(file.size / 1024 ** 2).toFixed(1)} MiB</p>}
        {file && (!file.size || file.size > (options.data?.max_archive_bytes ?? 0)) && <p role="alert">Choose a nonempty TAR within the 4 GiB limit.</p>}
        <button type="button" className="secondary-button" disabled={!file || !file.size || file.size > (options.data?.max_archive_bytes ?? 0)} onClick={() => void mutate('upload')}>Upload and validate policy</button>
      </div>
      <label>Native policy<select aria-label="Native policy" value={input?.id ?? ''} onChange={event => { setArtifactId(event.target.value); setExperimental(false); }}><option value="">Choose a saved native policy</option>{inputs.map(item => <option key={item.id} value={item.id}>{item.label} · {String(item.metadata?.architecture).toUpperCase()} · {item.id.slice(0, 8)}</option>)}</select></label>
      {!artifacts.isPending && !inputs.length && <p>No compatible local policy is available. Import a complete native package, or export a completed training checkpoint first.</p>}
      <label>Simulation timeout (seconds)<input type="number" min="30" max="7200" step="1" value={timeout} onChange={event => setTimeoutValue(event.target.value)} /></label>
      {!timeoutValid && <p role="alert">Choose a whole number from 30 to 7200 seconds.</p>}
      <label className="native-confirm"><input type="checkbox" checked={experimental} onChange={event => setExperimental(event.target.checked)} />I understand this is an experimental, paid cloud rollout with unverified cup pickup and calibration.</label>
      <button type="button" className="primary-button" disabled={!input || !experimental || !timeoutValid || artifacts.isError || jobs.isError} onClick={() => void mutate('run')}>Start experimental simulation</button>
    </fieldset>
    {pending === 'upload' && <div role="status"><p>{progress === 100 ? 'Upload sent; waiting for the import job receipt…' : `Uploading policy: ${progress}%`}</p><progress max="100" value={progress} aria-label="Policy upload progress" /><button className="secondary-button" onClick={() => abortUpload.current?.()}>Stop upload</button></div>}
    {pending === 'run' && <p role="status">Submitting one simulation job…</p>}
    <button className="text-link" type="button" onClick={onTraining}>Open training checkpoints</button>
    {artifacts.isError && <p role="alert">Saved policies are unavailable. {artifacts.error.message}</p>}
    <section className="native-simulation-history" aria-label="Native simulation jobs"><h3>Imports and simulation jobs</h3>
      {jobs.isError && <p role="alert">Job updates are unavailable. Previously received status may be stale. {jobs.error.message}</p>}
      {!saved.length && !accepted && <p>{jobs.isPending && projectId ? 'Loading jobs…' : 'No native imports or simulation runs in this project yet.'}</p>}
      {saved.length > 0 && <label>Saved simulation job<select aria-label="Saved simulation job" value={selected?.id ?? ''} onChange={event => showJob(event.target.value)}><option value="">Choose a recorded job</option>{saved.map(item => <option key={item.id} value={item.id}>{item.kind === 'policy.import' ? 'Policy import' : 'Isaac rollout'} · {item.id.slice(0, 8)} · {item.status}</option>)}</select></label>}
      {preferredJobId && !selected && jobs.isSuccess && <p role="status">The requested job is not available in this project's native simulation history.</p>}
      {selected && <article className="native-simulation-result" aria-label="Native simulation job details" data-job-id={selected.id}>
        <h3>{selected.kind === 'policy.import' ? 'Native policy import' : 'Isaac rollout'} · {selected.status}</h3>
        <dl className="cloud-run-facts"><div><dt>Job ID</dt><dd>{selected.id}</dd></div><div><dt>Last recorded update</dt><dd>{new Date(selected.updated_at).toLocaleString()}</dd></div><div><dt>Execution target</dt><dd>{simulationTarget(selected)?.accelerators.join(' + ') ?? 'Local package validation'}</dd></div><div><dt>Cup pickup success</dt><dd>Not measured</dd></div><div><dt>Calibration</dt><dd>Unverified</dd></div></dl>
        {selected.error && <p role="alert" className="error-notice">{selected.error}</p>}
        {selected.status === 'succeeded' && <p>{selected.kind === 'policy.import' ? 'The native package is saved and its package integrity checked. Runtime compatibility and task quality remain unverified. Select it above for an explicit experimental Run.' : 'Execution completed. This is not a scored evaluation or proof of cup pickup.'}</p>}
        {isActive(selected) && <><progress aria-label="Native job in progress" /><button className="secondary-button" disabled={pending !== null || jobs.isError} onClick={() => setConfirmCancel(selected.id)}>Cancel selected native job</button></>}
        {confirmCancel === selected.id && isActive(selected) && <div className="warning-box" role="group" aria-label="Confirm native cancellation"><p>Request cancellation of {selected.id}? This does not prove cloud resources have been deleted.</p><button className="secondary-button" disabled={pending !== null || jobs.isError} onClick={() => void mutate('cancel')}>Confirm cancellation</button><button className="text-link" disabled={pending !== null} onClick={() => setConfirmCancel(null)}>Keep running</button></div>}
        {events.isError && <p role="alert">Recorded activity is unavailable. {events.error.message}</p>}
        <pre className="cloud-log-tail" role="region" aria-label="Native simulation event log" tabIndex={0}>{events.data?.length ? events.data.slice(-100).map(event => `${event.timestamp} · ${event.stage} · ${event.message}${simulationTaskSummary(event.data) ? ` · ${simulationTaskSummary(event.data)}` : ''}`).join('\n') : 'No recorded events yet.'}</pre>
        {videoReady && !videoFailed && <video controls preload="metadata" aria-label="Recorded cup rollout" src={simulationVideoUrl(selected.id)} onError={() => setVideoFailed(true)} />}
        {videoFailed && <p role="alert">The recorded video is unavailable. You can still download the verified simulation record.</p>}
        {selected.status === 'succeeded' && records.map(item => <p key={item.id}><a className="secondary-button" href={artifactDownloadUrl(projectId, item.id)}>Download simulation record</a></p>)}
        {report && <details><summary>Recorded simulation report</summary><pre className="cloud-log-tail">{JSON.stringify(report, null, 2)}</pre></details>}
      </article>}
    </section>
  </section>;
}
