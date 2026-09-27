'use client';

import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api, artifactDownloadUrl, isActive, type Job } from '@/lib/api';
import { storeAttempt, storedAttempt, type PolicyJobAttempt } from '@/lib/policy-job-attempt';
import { WorkflowChoiceGrid } from './workflow-choice-grid';
import { NativePreparation } from './native-preparation';
import { publicPath } from '@/lib/base-path';
import { WorkbenchDisclosure } from './workbench-disclosure';
import { simulationOptions } from '@/lib/native-simulation';
import { quantizedSimulationHandoff, simulationHandoffProfiles, type SimulationHandoff } from '@/lib/native-simulation-handoff';
import { availableNativeQuantizer, nativeTransformIssue, nativeTransformMetadata, cancelNativeQuantization, nativeQuantizationInput, nativeQuantizationOf, measuredNativeReport, replayableNativeOutput, object, startNativeQuantization, UncertainQuantization, type NativeQuantizationRuntime, type PackedArtifact } from '@/lib/native-quantization';

class QuantizationJournalUnavailable extends Error {
  constructor() { super('Browser session storage is unavailable. Restore it and reload, then inspect recorded jobs before submitting again.'); }
}

export function NativeQuantizationPanel({ projectId, preferredArtifactId, preferredJobId, onJobSelected, onPrepare, onReplay, onPrepareSimulation }: { projectId: string; preferredArtifactId?: string; preferredJobId?: string; onJobSelected?: (id: string) => void; onPrepare: () => void; onReplay?: (artifactId: string) => void; onPrepareSimulation?: (source: SimulationHandoff) => void }) {
  const client = useQueryClient();
  const options = useQuery({ queryKey: ['policy-options'], queryFn: api.policyOptions, retry: false, refetchInterval: 10_000 });
  const jobs = useQuery({ queryKey: ['jobs', projectId], queryFn: () => api.jobs(projectId), enabled: !!projectId, retry: false, refetchInterval: 2_000 });
  const artifacts = useQuery({ queryKey: ['artifacts', projectId], queryFn: () => api.artifacts(projectId), enabled: !!projectId, retry: false, refetchInterval: 3_000 });
  const configured = (options.data?.runtimes ?? []).filter(item => (item as NativeQuantizationRuntime).native_quantization === true) as NativeQuantizationRuntime[];
  const runtimes = configured.filter(availableNativeQuantizer);
  const [runtimeId, setRuntimeId] = useState('');
  const runtime = runtimeId ? runtimes.find(item => item.id === runtimeId) : runtimes[0];
  const [artifactId, setArtifactId] = useState('');
  const inputs = (artifacts.data ?? []).filter(item => nativeQuantizationInput(item, projectId));
  const excludedInputs = (artifacts.data ?? []).some(item => item.project_id === projectId && ['native_checkpoint', 'inference_export'].includes(item.format) && item.metadata?.architecture === 'act' && nativeTransformIssue(item));
  const input = inputs.find(item => item.id === artifactId);
  const inputSemantics = input ? nativeTransformMetadata(input) : null;
  const [bits, setBits] = useState<4 | 8>(8);
  const [timeout, setTimeoutValue] = useState('600');
  const seconds = Number(timeout);
  const validTimeout = /^\d+$/.test(timeout) && Number.isSafeInteger(seconds) && seconds >= 30 && seconds <= 600;
  const [pending, setPending] = useState<'submit' | 'cancel' | null>(null);
  const [error, setError] = useState('');
  const attemptKey = ['native-quantization-attempt', projectId];
  const attempt = useQuery<PolicyJobAttempt>({ queryKey: attemptKey, queryFn: async () => null, enabled: false, initialData: null, gcTime: Infinity });
  const [journalReady, setJournalReady] = useState(false);
  const attemptVersion = useRef(0);
  const preference = useRef({ id: preferredArtifactId, consumed: false });
  const [reviewed, setReviewed] = useState(false);
  const [jobId, setJobId] = useState(preferredArtifactId ? '' : preferredJobId ?? '');
  const [accepted, setAccepted] = useState<Job | null>(null);
  const [confirmCancel, setConfirmCancel] = useState<string | null>(null);
  const busy = useRef(false);
  const mounted = useRef(true);
  const selectedId = useRef(preferredArtifactId ? '' : preferredJobId ?? '');
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  useEffect(() => {
    try {
      if (!client.getQueryData<PolicyJobAttempt>(['native-quantization-attempt', projectId])) client.setQueryData<PolicyJobAttempt>(['native-quantization-attempt', projectId], storedAttempt('policy.quantize', projectId));
      setJournalReady(true);
    } catch { setError('Browser session storage is unavailable. Enable it before submitting a job so uncertain requests can be recovered after a reload.'); }
  }, [client, projectId]);
  useEffect(() => {
    if (preference.current.id !== preferredArtifactId) preference.current = { id: preferredArtifactId, consumed: false };
    if (!preferredArtifactId || preference.current.consumed || !inputs.some(item => item.id === preferredArtifactId)) return;
    preference.current.consumed = true;
    setArtifactId(preferredArtifactId);
  }, [preferredArtifactId, inputs]);
  function chooseArtifact(id: string) { preference.current = { id: preferredArtifactId, consumed: true }; setArtifactId(id); }
  function saveAttempt(value: PolicyJobAttempt) {
    attemptVersion.current += 1;
    if (mounted.current) setReviewed(false);
    try { storeAttempt('policy.quantize', projectId, value); }
    catch {
      const failure = new QuantizationJournalUnavailable();
      // The disk record stays untouched; a remounted pane must not remain 'Submitting'.
      if (client.getQueryData<PolicyJobAttempt>(attemptKey)?.state === 'pending') client.setQueryData<PolicyJobAttempt>(attemptKey, (): PolicyJobAttempt => ({ state: 'uncertain', message: failure.message }));
      if (mounted.current) setJournalReady(false);
      throw failure;
    }
    client.setQueryData<PolicyJobAttempt>(attemptKey, value);
  }
  const saved = (jobs.data ?? []).filter(item => item.project_id === projectId && nativeQuantizationOf(item)).sort((a, b) => b.created_at.localeCompare(a.created_at));
  const selected = saved.find(item => item.id === jobId) ?? (accepted?.id === jobId && accepted.project_id === projectId ? accepted : undefined);
  const events = useQuery({ queryKey: ['events', selected?.id, selected?.status], queryFn: () => api.events(selected!.id), enabled: !!selected, retry: false, refetchInterval: selected && isActive(selected) ? 2_000 : false });
  const result: Record<string, unknown> | null = selected?.result && object(selected.result) ? selected.result : null;
  const outputs = (Array.isArray(result?.artifacts) ? result.artifacts : []) as PackedArtifact[];
  const downloads = outputs.filter(item => item.project_id === projectId && item.job_id === selected?.id && item.format === 'native_quantized' && item.metadata?.architecture === 'act');
  const reports: Record<string, unknown>[] = Array.isArray(result?.reports) ? result.reports.filter(object) : [];
  const measured = selected ? reports.map(item => measuredNativeReport(item, selected)).find(Boolean) : null;
  const simulation = useQuery({ queryKey: ['simulation-options'], queryFn: simulationOptions, enabled: !!onPrepareSimulation && selected?.status === 'succeeded' && !!measured, retry: false, refetchInterval: selected?.status === 'succeeded' && !!measured ? 10_000 : false });
  const editable = journalReady && !!projectId && options.isSuccess && !options.isError && artifacts.isSuccess && !artifacts.isError && jobs.isSuccess && !jobs.isError && !pending && !attempt.data;
  const ready = editable && !!runtime;
  function showJob(id: string) { onJobSelected?.(id); selectedId.current = id; setJobId(id); setConfirmCancel(null); }
  async function refresh() {
    const version = attemptVersion.current, canReview = attempt.data?.state === 'uncertain';
    const response = await jobs.refetch();
    void artifacts.refetch(); void options.refetch(); if (selected) void events.refetch();
    if (canReview && version === attemptVersion.current && !response.isError && mounted.current) setReviewed(true);
  }
  async function mutate(kind: 'submit' | 'cancel') {
    if (busy.current || !projectId) return;
    if (kind === 'submit' && (!ready || !input || !runtime || !validTimeout)) return;
    if (kind === 'cancel' && (!selected || selected.id !== confirmCancel || !isActive(selected) || jobs.isError)) return;
    busy.current = true; setPending(kind); setError(''); setReviewed(false);
    try {
      if (kind === 'submit') saveAttempt({ state: 'pending', message: 'Submitting one local quantization job…' });
      const receipt = kind === 'submit'
        ? await startNativeQuantization({ project: projectId, runtime: runtime!.id, artifact: input!.id, bits, timeout: seconds })
        : await cancelNativeQuantization(selected!, () => mounted.current && selectedId.current === selected!.id);
      if (mounted.current && (kind === 'submit' || selectedId.current === selected!.id)) { setAccepted(receipt); showJob(receipt.id); }
      if (kind === 'submit') saveAttempt(null);
      await client.invalidateQueries({ queryKey: ['jobs', projectId] });
      await client.invalidateQueries({ queryKey: ['artifacts', projectId] });
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : 'The request failed.';
      try {
        if (cause instanceof QuantizationJournalUnavailable) { /* Keep the existing recovery record; never retry failed storage cleanup. */ }
        else if (cause instanceof UncertainQuantization) saveAttempt({ state: 'uncertain', message });
        else if (kind === 'submit') saveAttempt(null);
      } catch { if (mounted.current) setJournalReady(false); }
      if (mounted.current) setError(message);
    } finally { busy.current = false; if (mounted.current) { setPending(null); setConfirmCancel(null); } }
  }
  return <section className="panel native-simulation native-quantization native-workflow" aria-label="Native ACT quantization">
    <div className="native-workflow-toolbar"><span className="native-model-badge">ACT · INT8 / INT4</span><button className="text-link" aria-label="Refresh ACT quantization jobs" disabled={!projectId || jobs.isFetching} onClick={() => void refresh()}>Refresh</button></div>
    {selected && <article className="native-simulation-result" aria-label="ACT quantization job details" data-job-id={selected.id}>
      <div className="native-result-header"><h3>INT{nativeQuantizationOf(selected)?.bits} candidate</h3><span className={`status status-${selected.status}`}>{selected.status}</span></div>
      <p className="native-result-summary">{isActive(selected) ? selected.stage ?? selected.status : 'Recorded job'} · {selected.id}</p>
      {selected.error && <p role="alert" className="error-notice">{selected.error}</p>}
        {selected.status === 'succeeded' && (measured ? <section aria-label="Measured ACT quantization results">
          <h4>Stored bytes and action drift</h4>
          <dl className="cloud-run-facts"><div><dt>Source FP32 weights</dt><dd>{measured.source_weight_bytes.toLocaleString()} bytes</dd></div><div><dt>Packed weights</dt><dd>{measured.packed_weight_bytes.toLocaleString()} bytes</dd></div><div><dt>Inference payload</dt><dd>{measured.policy_package_bytes.toLocaleString()} bytes</dd></div><div><dt>Fresh packed CPU reload</dt><dd>Exact agreement with the packed candidate</dd></div></dl>
          <p>Generated observations only. Exact reload does not mean unchanged FP32 actions.</p>
          <WorkbenchDisclosure title="Action differences"><p>Payload size excludes the outer download envelope. These observations are not held-out robotics evaluation.</p><div className="native-quantization-drift" role="region" aria-label="FP32 action differences" tabIndex={0}><table><caption>Difference from FP32 across each full {measured.prediction_horizon} × 6 action chunk</caption><thead><tr><th>Fixture seed</th><th>Raw RMSE</th><th>Raw maximum</th><th>Postprocessed RMSE</th><th>Postprocessed maximum</th></tr></thead><tbody>{measured.drift_from_fp32.map((row, index) => <tr key={`${row.input_sha256}-${index}`}><th scope="row">{row.seed}</th><td>{row.raw.rmse.toPrecision(6)}</td><td>{row.raw.maximum_absolute_difference.toPrecision(6)}</td><td>{row.postprocessed.rmse.toPrecision(6)}</td><td>{row.postprocessed.maximum_absolute_difference.toPrecision(6)}</td></tr>)}</tbody></table></div>
          <p>Postprocessed differences use saved processor output coordinates; physical units are unverified. No accepted quality threshold or GPU memory/latency improvement is established.</p></WorkbenchDisclosure>
        </section> : <p role="alert">A complete measured quantization report is unavailable. Do not infer reload or quality acceptance from the job status alone.</p>)}

      {selected.status === 'succeeded' && <><p>Task quality, calibration, speed and GPU memory savings remain unverified.</p>{downloads.map(item => {
        const handoff = quantizedSimulationHandoff(item, selected, measured ?? null);
        const compatible = simulation.data ? simulationHandoffProfiles(item, simulation.data.profiles) : [];
        return <div key={item.id} data-artifact-id={item.id}>
          <div className="native-result-actions">{onReplay && replayableNativeOutput(item, selected, measured ?? null) && <button className="primary-button" onClick={() => onReplay(item.id)}>Replay recorded observations</button>}{onPrepareSimulation && handoff && <button className="secondary-button" disabled={!simulation.isSuccess || simulation.isError || jobs.isError || !compatible.length} onClick={() => onPrepareSimulation(handoff)}>Prepare simulation</button>}<a className="secondary-button" href={artifactDownloadUrl(projectId, item.id)}>Download INT{nativeQuantizationOf(selected)?.bits} package</a></div>
          {onPrepareSimulation && handoff && <p className="field-help" role="status">{simulation.isError ? 'Simulation profile availability is unknown. Open Run to refresh its configuration.' : simulation.isPending ? 'Checking configured simulation profiles…' : compatible.length ? 'Continue with this exact package, then choose a profile and review paid rollout consent. Nothing starts here.' : 'No packed ACT simulation profile is configured. Observation replay and download remain available.'}</p>}
        </div>;
      })}</>}
      {isActive(selected) && <><progress aria-label="ACT quantization in progress" /><button className="secondary-button" disabled={pending !== null || jobs.isError} onClick={() => setConfirmCancel(selected.id)}>Cancel selected ACT quantization</button></>}
      {confirmCancel === selected.id && isActive(selected) && <div role="group" aria-label="Confirm ACT quantization cancellation" className="warning-box"><p>Stop this job and its owned local processes?</p><button className="secondary-button" disabled={pending !== null || jobs.isError} onClick={() => void mutate('cancel')}>Confirm cancellation</button><button className="text-link" disabled={pending !== null} onClick={() => setConfirmCancel(null)}>Keep running</button></div>}
      {events.isError && <p role="alert">Activity updates are unavailable; previously received activity may be stale. {events.error.message}</p>}
      <WorkbenchDisclosure key={selected.id} title="Activity and recorded report">
        <pre className="cloud-log-tail" role="region" aria-label="ACT quantization event log" tabIndex={0}>{events.data?.length ? events.data.slice(-100).map(event => `${event.timestamp} · ${event.stage} · ${event.message}`).join('\n') : 'No recorded events yet.'}</pre>
        {reports.length > 0 && <pre className="cloud-log-tail">{JSON.stringify(reports, null, 2)}</pre>}
      </WorkbenchDisclosure>
    </article>}
    {(saved.length > 0 || accepted || jobs.isError) && <section className="native-simulation-history" aria-label="Native ACT quantization jobs">
      {jobs.isError && <p role="alert">Job updates are unavailable. Previously received status may be stale. {jobs.error.message}</p>}
      {!saved.length && !accepted && <p>{jobs.isPending && projectId ? 'Loading jobs…' : 'No native ACT quantization jobs in this project yet.'}</p>}
      {saved.length > 0 && <label>Saved ACT quantization job<select aria-label="Saved ACT quantization job" value={selected?.id ?? ''} onChange={event => showJob(event.target.value)}><option value="">Choose a recorded job</option>{saved.map(item => <option key={item.id} value={item.id}>INT{nativeQuantizationOf(item)!.bits} · {item.id.slice(0, 8)} · {item.status}</option>)}</select></label>}
    </section>}
    {options.isError && <p role="alert">Native quantization options are unavailable. {options.error.message}</p>}
    {artifacts.isError && <p role="alert">Saved policies are unavailable. {artifacts.error.message}</p>}
    {error && <p role="alert" className="error-notice">{error}</p>}
    {journalReady && attempt.data?.state === 'pending' && <p role="status">{attempt.data.message}</p>}
    {attempt.data?.state === 'uncertain' && <div className="warning-box"><p>{attempt.data.message} Further submissions are paused.</p><button className="secondary-button" disabled={!reviewed || jobs.isError || pending !== null} onClick={() => { try { saveAttempt(null); setError(''); } catch { setJournalReady(false); setError('Browser session storage is unavailable.'); } }}>I checked the jobs; allow a new request</button></div>}
    <NativePreparation key={selected ? 'another' : 'first'} title="Prepare another ACT candidate" hasResult={!!selected}>
      {!projectId || !runtimes.length ? <div className="native-setup-empty">
        <p role="status">{!projectId ? 'Select a project to continue.' : options.isPending ? 'Loading workers…' : options.isError ? 'Worker availability is unknown.' : configured.length ? 'The configured ACT quantization worker is unavailable or disabled.' : 'No local ACT quantization worker is configured.'}</p>
        <a className="text-link" href={publicPath('/guide/#quantize')}>Set up quantization</a>
        <button className="primary-button" disabled>Create ACT quantized package</button>
      </div> : <>
        <fieldset disabled={!editable} className="native-simulation-form"><legend className="visually-hidden">ACT quantization setup</legend>
          <WorkflowChoiceGrid name="act-policy" label="Policy" value={input?.id ?? ''} onChange={chooseArtifact} options={inputs.map(item => ({ value: item.id, label: item.label, meta: item.id.slice(0, 8), icon: 'layers' }))} emptyMessage="No ACT inference policies. Import or export a complete policy first." />
          {excludedInputs && <p role="status">Some ACT packages are excluded because their model format, timing or simulator details are incomplete or unsupported. Refresh or export a complete inference package.</p>}
          {inputSemantics && <p className="field-help">Inherited action timing: plans {inputSemantics.prediction_horizon} actions and applies {inputSemantics.execution_horizon} per update. Packing keeps the saved timing; the server checks the policy before starting.</p>}
          {inputSemantics?.control_contract && <p role="status">This package carries a simulator control contract. Packing must preserve it. A compatible Run profile and separate rollout consent are still required; calibration and task success remain unverified.</p>}
          {(runtimes.length > 1 || !runtime) && <WorkflowChoiceGrid name="act-worker" label="Compute" value={runtime?.id ?? ''} onChange={setRuntimeId} options={runtimes.map(item => ({ value: item.id, label: item.label, icon: 'sliders' }))} />}
          <WorkflowChoiceGrid name="act-precision" label="Compression" value={String(bits)} onChange={value => setBits(value === '4' ? 4 : 8)} options={[{ value: '8', label: '8-bit', description: 'Balanced', icon: 'compress' }, { value: '4', label: '4-bit', description: 'Smaller package', icon: 'compress' }]} />
          <label className="simulation-timeout">Timeout (seconds)<input aria-label="Quantization timeout (seconds)" type="number" min="30" max="600" step="1" value={timeout} onChange={event => setTimeoutValue(event.target.value)} /></label>
          {!validTimeout && <p role="alert">Choose a whole number from 30 to 600 seconds.</p>}
          <button className="primary-button" disabled={!ready || !input || !validTimeout} onClick={() => void mutate('submit')}>Create ACT quantized package</button>
        </fieldset>
        {!inputs.length && <button className="text-link" onClick={onPrepare}>Open training and inference exports</button>}
      </>}
    </NativePreparation>
  </section>;
}
