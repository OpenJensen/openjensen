'use client';

import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api, artifactDownloadUrl, isActive, isDatasetJob } from '@/lib/api';
import { record, UncertainPolicyJob } from '@/lib/policy-job-mutation';
import { storedAttempt, storeAttempt, type PolicyJobAttempt } from '@/lib/policy-job-attempt';
import { cancelReplay, readReplay, replayDataset, replayJob, replayPolicy, replayReport, replayRuntime, replaySelection, startReplay, type ReplayJob, type ReplayRecipe } from '@/lib/native-replay';
import { WorkbenchDisclosure } from './workbench-disclosure';

class JournalUnavailable extends Error {
  constructor() { super('Browser session storage is unavailable. Restore it and reload, then inspect recorded jobs before submitting again.'); }
}

function ActionTrace({ values, name, unit }: { values: number[]; name: string; unit: string }) {
  const low = Math.min(...values), high = Math.max(...values), extent = high - low;
  const points = values.map((value, index) => `${4 + index * 252 / 99},${extent ? 50 - (value - low) * 44 / extent : 28}`).join(' ');
  return <figure className="replay-action-trace"><figcaption>{name}<span>{unit}</span></figcaption><svg viewBox="0 0 260 56" role="img" aria-label={`${name}: 100 predicted actions, minimum ${low.toPrecision(4)}, maximum ${high.toPrecision(4)} ${unit}`}><line x1="4" y1="52" x2="256" y2="52" stroke="var(--line)" /><polyline points={points} fill="none" stroke="currentColor" strokeWidth="2" /></svg><small>{low.toPrecision(4)} to {high.toPrecision(4)} · actions 1–100</small></figure>;
}

export function NativeReplayPanel({ projectId, preferredArtifactId, onDataset }: { projectId: string; preferredArtifactId?: string; onDataset: () => void }) {
  const client = useQueryClient();
  const options = useQuery({ queryKey: ['policy-options'], queryFn: api.policyOptions, retry: false, refetchInterval: 10000 });
  const jobs = useQuery({ queryKey: ['jobs', projectId], queryFn: () => api.jobs(projectId), enabled: !!projectId, retry: false, refetchInterval: 2000 });
  const artifacts = useQuery({ queryKey: ['artifacts', projectId], queryFn: () => api.artifacts(projectId), enabled: !!projectId, retry: false, refetchInterval: 5000 });
  const attemptKey = ['native-replay-attempt', projectId];
  const attempt = useQuery<PolicyJobAttempt>({ queryKey: attemptKey, queryFn: async () => null, initialData: null, enabled: false, gcTime: Infinity });
  const [journalReady, setJournalReady] = useState(false), [reviewed, setReviewed] = useState(false);
  const [policyId, setPolicy] = useState(''), [datasetId, setDataset] = useState(''), [runtimeId, setRuntime] = useState('');
  const [selection, setSelection] = useState(''), [units, setUnits] = useState(''), [generated, setGenerated] = useState(false), [attested, setAttested] = useState(false);
  const [timeout, setTimeoutValue] = useState('600'), [selectedId, setSelectedId] = useState(''), [accepted, setAccepted] = useState<ReplayJob | null>(null);
  const [error, setError] = useState(''), [cancelId, setCancelId] = useState(''), [cancelling, setCancelling] = useState(false), [observationIndex, setObservationIndex] = useState(0);
  const mounted = useRef(true), busy = useRef(false), currentId = useRef(''), attemptVersion = useRef(0), preferred = useRef('');
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  useEffect(() => {
    try { if (!client.getQueryData(attemptKey)) client.setQueryData(attemptKey, storedAttempt('policy.run.replay', projectId)); setJournalReady(true); }
    catch { setError('Browser session storage is unavailable. Enable it before submitting a recoverable job.'); }
  }, [client, projectId]);
  const runtimes = (options.data?.runtimes ?? []).filter(replayRuntime);
  const runtime = runtimeId ? runtimes.find(item => item.id === runtimeId) : runtimes[0];
  const policies = (artifacts.data ?? []).filter(item => replayPolicy(item, projectId));
  const policy = policies.find(item => item.id === policyId);
  const datasets = (jobs.data ?? []).filter(isDatasetJob).filter(item => replayDataset(item, projectId));
  const dataset = datasets.find(item => item.id === datasetId);
  useEffect(() => { if (preferredArtifactId && preferred.current !== preferredArtifactId && policies.some(item => item.id === preferredArtifactId)) { preferred.current = preferredArtifactId; setPolicy(preferredArtifactId); setAttested(false); } }, [preferredArtifactId, artifacts.data, projectId]);
  const history = (jobs.data ?? []).filter(replayJob).filter(item => item.project_id === projectId).sort((a, b) => b.created_at.localeCompare(a.created_at));
  const picked = history.find(item => item.id === selectedId) ?? (accepted?.id === selectedId ? accepted : undefined);
  const selected = picked?.project_id === projectId ? picked : undefined;
  const events = useQuery({ queryKey: ['events', selected?.id, selected?.status], queryFn: () => api.events(selected!.id), enabled: !!selected, retry: false, refetchInterval: selected && isActive(selected) ? 2000 : false });
  const result: Record<string, unknown> | null = selected?.result && record(selected.result) ? selected.result : null;
  const report = selected && Array.isArray(result?.reports) ? result.reports.map(item => replayReport(item, selected)).find(Boolean) : null;
  const outputs = selected?.result && 'artifacts' in selected.result ? (selected.result.artifacts ?? []).filter(item => item.project_id === projectId && item.job_id === selected.id && item.format === 'native_run_record' && item.metadata?.recipe === 'native-observation-replay-v1' && item.metadata?.model_id === report?.model_id) : [];
  const output = outputs[0];
  const preview = useQuery({ queryKey: ['native-replay-record', projectId, selected?.id, output?.id, output?.manifest_sha256], queryFn: () => readReplay(projectId, output!.id, selected!, report!), enabled: !!selected && !!output && !!report, retry: false });
  const observed = preview.data?.records[observationIndex];
  function selectJob(id: string) { currentId.current = id; setSelectedId(id); setCancelId(''); setObservationIndex(0); }
  function saveAttempt(value: PolicyJobAttempt) {
    try { storeAttempt('policy.run.replay', projectId, value); }
    catch {
      const failure = new JournalUnavailable();
      if (client.getQueryData<PolicyJobAttempt>(attemptKey)?.state === 'pending') client.setQueryData<PolicyJobAttempt>(attemptKey, (): PolicyJobAttempt => ({ state: 'uncertain', message: failure.message }));
      if (mounted.current) setJournalReady(false);
      throw failure;
    }
    client.setQueryData(attemptKey, value);
  }
  let recipe: ReplayRecipe | undefined, invalid = '';
  try {
    const rows = replaySelection(selection), coordinates = units.split(',').map(item => item.trim());
    if (dataset && rows.some(row => row.episode_index >= dataset.result!.total_episodes || row.frame_index >= dataset.result!.total_frames)) throw new Error('An observation exceeds the selected dataset bounds.');
    if (coordinates.length !== 6 || coordinates.some(item => !item || item.length > 80 || /[\x00-\x1f\x7f]/.test(item))) throw new Error('Enter six coordinate units in dataset order, separated by commas.');
    if (!/^\d+$/.test(timeout) || !Number.isSafeInteger(Number(timeout)) || Number(timeout) < 30 || Number(timeout) > 600) throw new Error('Choose a whole-number timeout from 30 to 600 seconds.');
    recipe = { adapter: 'act-packed-observation-v1', selection: rows, units: coordinates, coordinate_attestation: generated ? 'generated_fixture' : 'policy_recorded_coordinates' };
  } catch (cause) { invalid = cause instanceof Error ? cause.message : 'Review the observations.'; }
  const ready = journalReady && !!projectId && options.isSuccess && !options.isError && jobs.isSuccess && !jobs.isError && artifacts.isSuccess && !artifacts.isError && !!runtime && !!policy && !!dataset && !!recipe && attested && !attempt.data && !cancelling;
  async function refresh() {
    const version = attemptVersion.current, canReview = attempt.data?.state === 'uncertain';
    const response = await jobs.refetch(); void options.refetch(); void artifacts.refetch();
    if (canReview && version === attemptVersion.current && !response.isError && mounted.current) setReviewed(true);
  }
  async function submit() {
    if (!ready || busy.current) return;
    busy.current = true; attemptVersion.current += 1; setError(''); setReviewed(false);
    try {
      saveAttempt({ state: 'pending', message: 'Submitting one CPU observation replay…' });
      const job = await startReplay(projectId, { operation: 'policy.run', runtime_id: runtime!.id, artifact_id: policy!.id, dataset_job_id: dataset!.id, native_replay: recipe!, timeout_seconds: Number(timeout) });
      if (mounted.current) { setAccepted(job); selectJob(job.id); }
      saveAttempt(null);
      await client.invalidateQueries({ queryKey: ['jobs', projectId] });
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : 'Replay could not be submitted.';
      attemptVersion.current += 1; if (mounted.current) { setReviewed(false); setError(message); }
      if (!(cause instanceof JournalUnavailable)) {
        try { saveAttempt(cause instanceof UncertainPolicyJob ? { state: 'uncertain', message } : null); } catch { if (mounted.current) setJournalReady(false); }
      }
    } finally { busy.current = false; }
  }
  async function cancel() {
    if (!selected || selected.id !== cancelId || !isActive(selected) || jobs.isError || busy.current) return;
    busy.current = true; setCancelling(true); setError('');
    try {
      const job = await cancelReplay(selected, () => mounted.current && currentId.current === selected.id);
      if (mounted.current && currentId.current === selected.id) { setAccepted(job); selectJob(job.id); }
      await client.invalidateQueries({ queryKey: ['jobs', projectId] });
    } catch (cause) { if (mounted.current) setError(cause instanceof Error ? cause.message : 'Cancellation outcome is unverified; refresh this job.'); }
    finally { busy.current = false; if (mounted.current) { setCancelling(false); setCancelId(''); } }
  }
  return <section className="panel native-simulation native-replay" aria-label="CPU observation replay">
    <div className="cloud-heading"><div><h2>Inspect the actions your policy predicts</h2><p>Run a packed ACT policy on selected dataset observations. Each observation starts from a reset policy and produces a complete 100-action chunk.</p></div><button className="secondary-button" disabled={!projectId || jobs.isFetching} onClick={() => void refresh()}>Refresh replay jobs</button></div>
    <p>Local CPU · INT8 / INT4 ACT · actions are saved, never applied to a robot or simulator. Use Native Isaac for a scene rollout.</p>
    {selected && <article className="native-simulation-result" aria-label="Observation replay details" data-job-id={selected.id}>
      <div className="native-result-header"><h3>CPU observation replay</h3><span className={`status status-${selected.status}`}>{selected.status}</span></div><p className="native-result-summary">{selected.stage ?? 'Queued'} · {selected.id}</p>
      {selected.error && <p role="alert">{selected.error}</p>}
      {report && <><p>{report.observation_source.kind === 'generated_fixture' ? 'Generated observations · software verification only' : 'Dataset observations · execution check only'}</p><dl className="cloud-run-facts"><div><dt>Observations</dt><dd>{report.observations}</dd></div><div><dt>Predictions per observation</dt><dd>100 × 6</dd></div><div><dt>Reset repeatability</dt><dd>Exact repeat</dd></div></dl><p>The saved policy returned the same full chunk after reset. This does not measure task success, action accuracy, GPU performance, or calibration.</p></>}
      {selected.status === 'succeeded' && !report && <p role="alert">Complete replay evidence is unavailable. Job completion alone does not verify the output.</p>}
      {preview.isError && <p role="alert">Saved action preview unavailable. {preview.error.message}</p>}
      {report && preview.isPending && output && <p role="status">Verifying the saved action record…</p>}
      {observed && preview.data && !preview.isError && <section aria-label="Predicted action chunks"><label className="distillation-history">Recorded observation<select aria-label="Recorded observation" value={observationIndex} onChange={event => setObservationIndex(Number(event.target.value))}>{preview.data.records.map((row, index) => <option key={`${row.episode_index}:${row.frame_index}`} value={index}>Episode {row.episode_index} · frame {row.frame_index}</option>)}</select></label><p>Saved processor output coordinates. Each chart has its own vertical scale.</p><div className="replay-action-grid">{preview.data.coordinate_names.map((name, index) => <ActionTrace key={`${index}-${name}`} name={name} unit={preview.data!.units[index]} values={observed.actions.map(action => action[index])} />)}</div></section>}
      {selected.status === 'succeeded' && report && outputs.map(item => <p key={item.id}><a className="secondary-button" href={artifactDownloadUrl(projectId, item.id)}>Download verified replay record</a></p>)}
      {isActive(selected) && <><progress aria-label="Observation replay in progress" /><button className="secondary-button" disabled={cancelling || jobs.isError} onClick={() => setCancelId(selected.id)}>Cancel selected replay</button></>}
      {cancelId === selected.id && isActive(selected) && <div className="warning-box" role="group" aria-label="Confirm replay cancellation"><p>Stop this job and its owned CPU worker?</p><button className="secondary-button" disabled={cancelling || jobs.isError} onClick={() => void cancel()}>Confirm cancellation</button><button className="text-link" disabled={cancelling} onClick={() => setCancelId('')}>Keep running</button></div>}
      <WorkbenchDisclosure title="Replay activity and provenance">{events.isError && <p role="alert">Activity is unavailable.</p>}<pre className="cloud-log-tail">{events.data?.map(item => `${item.timestamp} · ${item.stage} · ${item.message}`).join('\n') || 'No recorded activity yet.'}</pre>{result && <pre className="cloud-log-tail">{JSON.stringify(result.reports, null, 2)}</pre>}</WorkbenchDisclosure>
    </article>}
    {history.length > 0 && <label className="distillation-history">Saved replay<select aria-label="Saved replay" value={selected?.id ?? ''} onChange={event => selectJob(event.target.value)}><option value="">Choose a recorded replay</option>{history.map(item => <option key={item.id} value={item.id}>{item.id.slice(0, 8)} · {item.status}</option>)}</select></label>}
    {jobs.isError && <p role="alert">Job updates are unavailable; displayed status may be stale.</p>}{options.isError && <p role="alert">Worker options are unavailable.</p>}{artifacts.isError && <p role="alert">Saved policies are unavailable.</p>}{error && <p role="alert">{error}</p>}
    {journalReady && attempt.data?.state === 'pending' && <p role="status">{attempt.data.message}</p>}
    {attempt.data?.state === 'uncertain' && <div className="warning-box"><p>{attempt.data.message}</p><button className="secondary-button" disabled={!reviewed || jobs.isError} onClick={() => { try { saveAttempt(null); setReviewed(false); setError(''); } catch { setJournalReady(false); setError('Browser session storage is unavailable.'); } }}>I checked replay jobs; allow a new request</button></div>}
    <WorkbenchDisclosure key={selected ? 'another' : 'first'} title={selected ? 'Prepare another replay' : 'Choose policy and observations'} initiallyOpen={!selected}>
      {options.isSuccess && !runtimes.length && <p role="status">No local CPU replay worker is configured. Its isolated model environment and dataset reader must be registered first.</p>}
      <fieldset className="native-simulation-form" disabled={!projectId || !!attempt.data || cancelling}><legend>Explicit observation selection</legend>
        <label>Replay worker<select aria-label="Replay worker" value={runtime?.id ?? ''} onChange={event => setRuntime(event.target.value)}><option value="">Choose a local worker</option>{runtimes.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
        <label>Packed ACT policy<select aria-label="Packed ACT policy" value={policyId} onChange={event => { setPolicy(event.target.value); setAttested(false); }}><option value="">Choose a complete local INT8 or INT4 policy</option>{policies.map(item => <option key={item.id} value={item.id}>{item.label} · {item.id.slice(0, 8)}</option>)}</select></label>
        <label>Observation dataset<select aria-label="Observation dataset" value={datasetId} onChange={event => { setDataset(event.target.value); setSelection(''); setAttested(false); }}><option value="">Choose a complete dataset snapshot</option>{datasets.map(item => <option key={item.id} value={item.id}>{item.result!.repo_id ?? 'Local robotics dataset'} · {item.result!.total_episodes} episodes · {item.id.slice(0, 8)}</option>)}</select></label>
        <button className="text-link" type="button" onClick={onDataset}>Open Dataset intake</button>
        <label>Episode and frame pairs<input value={selection} onChange={event => setSelection(event.target.value)} placeholder="For example: 0:3, 2:1" /></label><p>Choose 1–32 distinct observations. Original camera resolution, raw states, exact frame identity, and a 128 MiB input limit are verified before execution.</p>
        <label>Six replay coordinate units<input value={units} maxLength={500} onChange={event => { setUnits(event.target.value); setAttested(false); }} placeholder="One unit per recorded action coordinate, in order" /></label>
        <label className="distillation-check"><input type="checkbox" checked={generated} onChange={event => { setGenerated(event.target.checked); setAttested(false); }} /> These are generated test observations.</label>
        <label className="distillation-check"><input type="checkbox" checked={attested} onChange={event => setAttested(event.target.checked)} /> I verified that these coordinates, order, and camera match the packed policy’s saved processors.</label>
        <label>Replay timeout (seconds)<input type="number" min="30" max="600" value={timeout} onChange={event => setTimeoutValue(event.target.value)} /></label>
        {invalid && <p className="field-hint">{invalid}</p>}<button className="primary-button" type="button" disabled={!ready} onClick={() => void submit()}>Run CPU observation replay</button>
      </fieldset>
    </WorkbenchDisclosure>
  </section>;
}
