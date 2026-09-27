'use client';

import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api, artifactDownloadUrl, isActive, type Job } from '@/lib/api';
import { availableNativeQuantizer, cancelNativeQuantization, nativeQuantizationInput, nativeQuantizationOf, measuredNativeReport, object, startNativeQuantization, UncertainQuantization, type NativeQuantizationRuntime, type PackedArtifact } from '@/lib/native-quantization';

export function NativeQuantizationPanel({ projectId, onPrepare }: { projectId: string; onPrepare: () => void }) {
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
  const input = inputs.find(item => item.id === artifactId);
  const [bits, setBits] = useState<4 | 8>(8);
  const [timeout, setTimeoutValue] = useState('600');
  const seconds = Number(timeout);
  const validTimeout = /^\d+$/.test(timeout) && Number.isSafeInteger(seconds) && seconds >= 30 && seconds <= 600;
  const [pending, setPending] = useState<'submit' | 'cancel' | null>(null);
  const [error, setError] = useState('');
  const [ambiguous, setAmbiguous] = useState(false);
  const [reviewed, setReviewed] = useState(false);
  const [jobId, setJobId] = useState('');
  const [accepted, setAccepted] = useState<Job | null>(null);
  const [confirmCancel, setConfirmCancel] = useState<string | null>(null);
  const busy = useRef(false);
  const mounted = useRef(true);
  const selectedId = useRef('');
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  const saved = (jobs.data ?? []).filter(item => item.project_id === projectId && nativeQuantizationOf(item)).sort((a, b) => b.created_at.localeCompare(a.created_at));
  const selected = saved.find(item => item.id === jobId) ?? (accepted?.id === jobId && accepted.project_id === projectId ? accepted : undefined);
  const events = useQuery({ queryKey: ['events', selected?.id, selected?.status], queryFn: () => api.events(selected!.id), enabled: !!selected, retry: false, refetchInterval: selected && isActive(selected) ? 2_000 : false });
  const result: Record<string, unknown> | null = selected?.result && object(selected.result) ? selected.result : null;
  const outputs = (Array.isArray(result?.artifacts) ? result.artifacts : []) as PackedArtifact[];
  const downloads = outputs.filter(item => item.project_id === projectId && item.job_id === selected?.id && item.format === 'native_quantized' && item.metadata?.architecture === 'act');
  const reports: Record<string, unknown>[] = Array.isArray(result?.reports) ? result.reports.filter(object) : [];
  const measured = selected ? reports.map(item => measuredNativeReport(item, selected)).find(Boolean) : null;
  const ready = !!projectId && options.isSuccess && !options.isError && !!runtime && artifacts.isSuccess && !artifacts.isError && jobs.isSuccess && !jobs.isError && !pending && !ambiguous;
  function showJob(id: string) { selectedId.current = id; setJobId(id); setConfirmCancel(null); }
  async function refresh() {
    const response = await jobs.refetch();
    void artifacts.refetch(); void options.refetch(); if (selected) void events.refetch();
    if (!response.isError) setReviewed(true);
  }
  async function mutate(kind: 'submit' | 'cancel') {
    if (busy.current || !projectId) return;
    if (kind === 'submit' && (!ready || !input || !runtime || !validTimeout)) return;
    if (kind === 'cancel' && (!selected || selected.id !== confirmCancel || !isActive(selected) || jobs.isError)) return;
    busy.current = true; setPending(kind); setError(''); setReviewed(false);
    try {
      const receipt = kind === 'submit'
        ? await startNativeQuantization({ project: projectId, runtime: runtime!.id, artifact: input!.id, bits, timeout: seconds })
        : await cancelNativeQuantization(selected!, () => mounted.current && selectedId.current === selected!.id);
      if (mounted.current) { setAccepted(receipt); showJob(receipt.id); }
      await client.invalidateQueries({ queryKey: ['jobs', projectId] });
      await client.invalidateQueries({ queryKey: ['artifacts', projectId] });
    } catch (cause) {
      if (mounted.current) { setError(cause instanceof Error ? cause.message : 'The request failed.'); if (cause instanceof UncertainQuantization) setAmbiguous(true); }
    } finally { busy.current = false; if (mounted.current) { setPending(null); setConfirmCancel(null); } }
  }
  return <section className="panel native-simulation native-quantization" aria-labelledby="native-quantization-title">
    <div className="cloud-heading"><div><h2 id="native-quantization-title">Native ACT quantization</h2><p>Create a packed INT8 or INT4 candidate and verify a fresh CPU reload using generated inputs.</p></div><button className="secondary-button" disabled={!projectId || jobs.isFetching} onClick={() => void refresh()}>Refresh ACT quantization jobs</button></div>
    <p className="warning-box">Smaller stored weights can still expand during execution. Action drift is measured against FP32; task quality, calibration, speed and GPU memory savings are unverified. These packages are download-only: native Isaac Run does not support this packed format.</p>
    {!projectId && <p role="status">Select a project before quantizing a policy.</p>}
    {options.isPending && <p role="status">Loading native quantization capability…</p>}
    {options.isError && <p role="alert">Native quantization options are unavailable. {options.error.message}</p>}
    {options.isSuccess && !runtimes.length && <p role="status">{configured.length ? 'The configured ACT quantization worker is unavailable or disabled.' : 'No local ACT quantization worker is configured.'} Connect an operator-installed worker in the application configuration; this page does not install or start cloud resources.</p>}
    {artifacts.isError && <p role="alert">Saved policies are unavailable. {artifacts.error.message}</p>}
    {error && <p role="alert" className="error-notice">{error}</p>}
    {ambiguous && <div className="warning-box"><p>Further submissions are paused. Refresh and inspect the recorded jobs first.</p><button className="secondary-button" disabled={!reviewed || jobs.isError || pending !== null} onClick={() => { setAmbiguous(false); setError(''); setReviewed(false); }}>I checked the jobs; allow a new request</button></div>}
    <fieldset disabled={!ready} className="native-simulation-form">
      <legend>Prepare a local ACT candidate</legend>
      <label>ACT quantization worker<select aria-label="ACT quantization worker" value={runtime?.id ?? ''} onChange={event => setRuntimeId(event.target.value)}><option value="" disabled>No worker selected</option>{runtimes.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
      <label>ACT inference policy<select aria-label="ACT inference policy" value={input?.id ?? ''} onChange={event => setArtifactId(event.target.value)}><option value="">Choose a complete local ACT policy</option>{inputs.map(item => <option key={item.id} value={item.id}>{item.label} · {item.id.slice(0, 8)}</option>)}</select></label>
      <p>Use an imported complete ACT inference package or a saved inference export: 100-action chunks, six action coordinates, one camera and no training-only VAE. Full training checkpoints and cloud descriptors must be prepared first. The worker checks every package before conversion.</p>
      {!artifacts.isPending && !inputs.length && <p>No compatible local ACT input is available.</p>}
      <label>Native precision<select aria-label="Native precision" value={bits} onChange={event => setBits(event.target.value === '4' ? 4 : 8)}><option value="8">INT8 · default</option><option value="4">INT4 · greater compression, measured drift may increase</option></select></label>
      <label>Quantization timeout (seconds)<input type="number" min="30" max="600" step="1" value={timeout} onChange={event => setTimeoutValue(event.target.value)} /></label>
      {!validTimeout && <p role="alert">Choose a whole number from 30 to 600 seconds.</p>}
      <button className="primary-button" disabled={!input || !validTimeout} onClick={() => void mutate('submit')}>Create ACT quantized package</button>
    </fieldset>
    {pending === 'submit' && <p role="status">Submitting one local quantization job…</p>}
    <button className="text-link" onClick={onPrepare}>Open training and inference exports</button>
    <section className="native-simulation-history" aria-label="Native ACT quantization jobs"><h3>Recorded ACT quantization jobs</h3>
      {jobs.isError && <p role="alert">Job updates are unavailable. Previously received status may be stale. {jobs.error.message}</p>}
      {!saved.length && !accepted && <p>{jobs.isPending && projectId ? 'Loading jobs…' : 'No native ACT quantization jobs in this project yet.'}</p>}
      {saved.length > 0 && <label>Saved ACT quantization job<select aria-label="Saved ACT quantization job" value={selected?.id ?? ''} onChange={event => showJob(event.target.value)}><option value="">Choose a recorded job</option>{saved.map(item => <option key={item.id} value={item.id}>INT{nativeQuantizationOf(item)!.bits} · {item.id.slice(0, 8)} · {item.status}</option>)}</select></label>}
      {selected && <article className="native-simulation-result" aria-label="ACT quantization job details" data-job-id={selected.id}>
        <h3>INT{nativeQuantizationOf(selected)?.bits} candidate · {selected.status}</h3>
        <dl className="cloud-run-facts"><div><dt>Job ID</dt><dd>{selected.id}</dd></div><div><dt>Recorded stage</dt><dd>{selected.stage ?? 'Queued'}</dd></div><div><dt>Task quality</dt><dd>Unverified</dd></div><div><dt>Isaac compatibility</dt><dd>Not implemented for this packed format</dd></div></dl>
        {selected.error && <p role="alert" className="error-notice">{selected.error}</p>}
        {isActive(selected) && <><progress aria-label="ACT quantization in progress" /><button className="secondary-button" disabled={pending !== null || jobs.isError} onClick={() => setConfirmCancel(selected.id)}>Cancel selected ACT quantization</button></>}
        {confirmCancel === selected.id && isActive(selected) && <div role="group" aria-label="Confirm ACT quantization cancellation" className="warning-box"><p>Request cancellation of {selected.id}?</p><button className="secondary-button" disabled={pending !== null || jobs.isError} onClick={() => void mutate('cancel')}>Confirm cancellation</button><button className="text-link" disabled={pending !== null} onClick={() => setConfirmCancel(null)}>Keep running</button></div>}
        {events.isError && <p role="alert">Recorded activity is unavailable. {events.error.message}</p>}
        <pre className="cloud-log-tail" role="region" aria-label="ACT quantization event log" tabIndex={0}>{events.data?.length ? events.data.slice(-100).map(event => `${event.timestamp} · ${event.stage} · ${event.message}`).join('\n') : 'No recorded events yet.'}</pre>
        {selected.status === 'succeeded' && <><p>The candidate was saved. Generated-input checks do not establish robot task quality or calibration.</p>{downloads.map(item => <p key={item.id}><a className="secondary-button" href={artifactDownloadUrl(projectId, item.id)}>Download INT{nativeQuantizationOf(selected)?.bits} package</a></p>)}</>}
        {selected.status === 'succeeded' && (measured ? <section aria-label="Measured ACT quantization results">
          <h4>Stored bytes and action drift</h4>
          <dl className="cloud-run-facts"><div><dt>Source FP32 weights</dt><dd>{measured.source_weight_bytes.toLocaleString()} bytes</dd></div><div><dt>Packed weights</dt><dd>{measured.packed_weight_bytes.toLocaleString()} bytes</dd></div><div><dt>Inference payload</dt><dd>{measured.policy_package_bytes.toLocaleString()} bytes</dd></div><div><dt>Fresh packed CPU reload</dt><dd>Exact agreement with the packed candidate</dd></div></dl>
          <p>Payload size excludes the outer download envelope. Exact reload does not mean unchanged FP32 actions. The two generated observations are not held-out robotics evaluation.</p>
          <div className="native-quantization-drift" role="region" aria-label="FP32 action differences" tabIndex={0}><table><caption>Difference from FP32 across each full 100 × 6 action chunk</caption><thead><tr><th>Fixture seed</th><th>Raw RMSE</th><th>Raw maximum</th><th>Postprocessed RMSE</th><th>Postprocessed maximum</th></tr></thead><tbody>{measured.drift_from_fp32.map((row, index) => <tr key={`${row.input_sha256}-${index}`}><th scope="row">{row.seed}</th><td>{row.raw.rmse.toPrecision(6)}</td><td>{row.raw.maximum_absolute_difference.toPrecision(6)}</td><td>{row.postprocessed.rmse.toPrecision(6)}</td><td>{row.postprocessed.maximum_absolute_difference.toPrecision(6)}</td></tr>)}</tbody></table></div>
          <p>Postprocessed differences use saved processor output coordinates; physical units are unverified. No accepted quality threshold or GPU memory/latency improvement is established.</p>
        </section> : <p role="alert">A complete measured quantization report is unavailable. Do not infer reload or quality acceptance from the job status alone.</p>)}
        {reports.length > 0 && <details><summary>Recorded quantization report</summary><pre className="cloud-log-tail">{JSON.stringify(reports, null, 2)}</pre></details>}
      </article>}
    </section>
  </section>;
}
