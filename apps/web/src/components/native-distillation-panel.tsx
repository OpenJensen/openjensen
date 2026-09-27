'use client';

import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api, artifactDownloadUrl, isActive, isDatasetJob, type Job } from '@/lib/api';
import { cancelStudent, episodeSelection, startStudent, studentDataset, studentJob, studentReport, studentRuntime, studentTeacher, type DistillationRecipe, type DistillationRequest } from '@/lib/native-distillation';
import { record, UncertainPolicyJob } from '@/lib/policy-job-mutation';
import { storeAttempt, storedAttempt, type PolicyJobAttempt } from '@/lib/policy-job-attempt';
import { WorkbenchDisclosure } from './workbench-disclosure';

function coordinateNames(value: unknown): string { return record(value) && Array.isArray(value.names) && value.names.every(item => typeof item === 'string') ? value.names.join(', ') : 'not declared'; }

type Attempt = PolicyJobAttempt;
const size = (bytes: number) => `${(bytes / 1024 ** 2).toFixed(1)} MiB`;

export function NativeDistillationPanel({ projectId, onDataset, onQuantize }: { projectId: string; onDataset: () => void; onQuantize: (artifactId: string) => void }) {
  const client = useQueryClient();
  const options = useQuery({ queryKey: ['policy-options'], queryFn: api.policyOptions, retry: false, refetchInterval: 10_000 });
  const jobs = useQuery({ queryKey: ['jobs', projectId], queryFn: () => api.jobs(projectId), enabled: !!projectId, retry: false, refetchInterval: 2_000 });
  const artifacts = useQuery({ queryKey: ['artifacts', projectId], queryFn: () => api.artifacts(projectId), enabled: !!projectId, retry: false, refetchInterval: 5_000 });
  const attemptKey = ['distillation-attempt', projectId];
  const attempt = useQuery<Attempt>({ queryKey: attemptKey, queryFn: async () => null, enabled: false, initialData: null, gcTime: Infinity });
  const [teacherId, setTeacher] = useState(''), [datasetId, setDataset] = useState(''), [runtimeId, setRuntime] = useState('');
  const [train, setTrain] = useState(''), [validation, setValidation] = useState(''), [final, setFinal] = useState('');
  const [units, setUnits] = useState(''), [attested, setAttested] = useState(false), [generated, setGenerated] = useState(false);
  const [steps, setSteps] = useState('100'), [stride, setStride] = useState('30'), [rate, setRate] = useState('0.0001'), [seed, setSeed] = useState('1729'), [timeout, setTimeoutValue] = useState('600');
  const [selectedId, setSelectedId] = useState(''), [accepted, setAccepted] = useState<Job | null>(null);
  const [journalReady, setJournalReady] = useState(false);
  const [error, setError] = useState(''), [cancelId, setCancelId] = useState(''), [cancelling, setCancelling] = useState(false), [reviewed, setReviewed] = useState(false);
  const mounted = useRef(true), busy = useRef(false), currentId = useRef('');
  const attemptVersion = useRef(0);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  useEffect(() => {
    try {
      if (!client.getQueryData<Attempt>(['distillation-attempt', projectId])) client.setQueryData<Attempt>(['distillation-attempt', projectId], storedAttempt('policy.distill', projectId));
      setJournalReady(true);
    } catch { setError('Browser session storage is unavailable. Enable it before submitting a job so uncertain requests can be recovered after a reload.'); }
  }, [client, projectId]);
  function saveAttempt(value: Attempt) { storeAttempt('policy.distill', projectId, value); client.setQueryData<Attempt>(attemptKey, value); }
  const runtimes = (options.data?.runtimes ?? []).filter(studentRuntime);
  const runtime = runtimeId ? runtimes.find(item => item.id === runtimeId) : runtimes[0];
  const teachers = (artifacts.data ?? []).filter(item => studentTeacher(item, projectId));
  const teacher = teachers.find(item => item.id === teacherId);
  const datasets = (jobs.data ?? []).filter(isDatasetJob).filter(item => studentDataset(item, projectId));
  const dataset = datasets.find(item => item.id === datasetId);
  const history = (jobs.data ?? []).filter(studentJob).filter(item => item.project_id === projectId).sort((a, b) => b.created_at.localeCompare(a.created_at));
  const picked = history.find(item => item.id === selectedId) ?? (accepted?.id === selectedId && studentJob(accepted) ? accepted : undefined);
  const selected = picked && picked.project_id === projectId ? picked : undefined;
  const events = useQuery({ queryKey: ['events', selected?.id, selected?.status], queryFn: () => api.events(selected!.id), enabled: !!selected, retry: false, refetchInterval: selected && isActive(selected) ? 2_000 : false });
  const result: Record<string, unknown> | null = selected?.result && record(selected.result) ? selected.result : null;
  const report = selected && Array.isArray(result?.reports) ? result.reports.map(value => studentReport(value, selected)).find(Boolean) : null;
  const outputs = selected?.result && 'artifacts' in selected.result ? (selected.result.artifacts ?? []).filter(item => item.project_id === projectId && item.job_id === selected.id && item.format === 'native_checkpoint' && item.metadata?.recipe === 'act-action-distillation-v1') : [];
  function selectJob(id: string) { currentId.current = id; setSelectedId(id); setCancelId(''); }
  let recipe: DistillationRecipe | undefined, invalid = '';
  try {
    const partitions = { train: episodeSelection(train), validation: episodeSelection(validation), final: episodeSelection(final) };
    const all = [...partitions.train, ...partitions.validation, ...partitions.final];
    if (new Set(all).size !== all.length || all.length > 256) throw new Error('Select distinct episodes across all three partitions, at most 256 in total.');
    if (dataset && all.some(id => id >= dataset.result!.total_episodes)) throw new Error('An episode number exceeds the selected dataset.');
    const coordinates = units.split(',').map(value => value.trim());
    if (coordinates.length !== 6 || coordinates.some(value => !value || value.length > 80 || /[\x00-\x1f\x7f]/.test(value))) throw new Error('Enter six coordinate units, in dataset order, separated by commas.');
    for (const [label, value, min, max] of [['Steps', steps, 1, 10000], ['Frame stride', stride, 1, 10000], ['Seed', seed, 0, 2147483647], ['Timeout', timeout, 30, 3600]] as const) {
      if (!/^\d+$/.test(value) || !Number.isSafeInteger(Number(value)) || Number(value) < min || Number(value) > max) throw new Error(`${label} must be a whole number from ${min} to ${max}.`);
    }
    if (!Number.isFinite(Number(rate)) || Number(rate) < 1e-7 || Number(rate) > 1e-3) throw new Error('Learning rate must be between 0.0000001 and 0.001.');
    recipe = { adapter: 'act-act-v1', student: 'act-256', steps: Number(steps), learning_rate: Number(rate), seed: Number(seed), frame_stride: Number(stride), splits: partitions, units: coordinates, coordinate_attestation: generated ? 'generated_fixture' : 'teacher_recorded_coordinates' };
  } catch (cause) { invalid = cause instanceof Error ? cause.message : 'Review the recipe.'; }
  const ready = journalReady && !!projectId && options.isSuccess && !options.isError && jobs.isSuccess && !jobs.isError && artifacts.isSuccess && !artifacts.isError && !!runtime && !!teacher && !!dataset && !!recipe && attested && !attempt.data && !cancelling;
  async function refresh() {
    const version = attemptVersion.current;
    const canReview = attempt.data?.state === 'uncertain';
    const value = await jobs.refetch(); void artifacts.refetch(); void options.refetch();
    if (canReview && version === attemptVersion.current && !value.isError && mounted.current) setReviewed(true);
  }
  async function submit() {
    if (!ready || busy.current) return;
    const request: DistillationRequest = { operation: 'policy.distill', runtime_id: runtime!.id, artifact_id: teacher!.id, dataset_job_id: dataset!.id, native_distillation: recipe, timeout_seconds: Number(timeout) };
    busy.current = true; setError(''); setReviewed(false); attemptVersion.current += 1;
    try {
      saveAttempt({ state: 'pending', message: 'Submitting one local distillation job…' });
      const job = await startStudent(projectId, request);
      saveAttempt(null);
      if (mounted.current) { setAccepted(job); selectJob(job.id); }
      await client.invalidateQueries({ queryKey: ['jobs', projectId] });
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : 'The request failed.';
      attemptVersion.current += 1;
      if (mounted.current) setReviewed(false);
      try { saveAttempt(cause instanceof UncertainPolicyJob ? { state: 'uncertain', message } : null); } catch { if (mounted.current) setJournalReady(false); }
      if (mounted.current) setError(message);
    } finally { busy.current = false; }
  }
  async function cancel() {
    if (!selected || cancelId !== selected.id || !isActive(selected) || jobs.isError || busy.current) return;
    busy.current = true; setCancelling(true); setError('');
    try {
      const job = await cancelStudent(selected, () => mounted.current && currentId.current === selected.id);
      if (mounted.current && currentId.current === selected.id) { setAccepted(job); selectJob(job.id); }
      await client.invalidateQueries({ queryKey: ['jobs', projectId] });
    } catch (cause) { if (mounted.current) setError(cause instanceof Error ? cause.message : 'Cancellation outcome is unknown; refresh the job.'); }
    finally { busy.current = false; if (mounted.current) { setCancelling(false); setCancelId(''); } }
  }
  return <section className="panel native-simulation native-distillation" aria-label="ACT distillation">
    <div className="cloud-heading"><div><h2>Teach a smaller ACT policy</h2><p>Train an ACT256 student to imitate your teacher’s action chunks, then reload the saved student in a fresh process.</p></div><button className="secondary-button" disabled={!projectId || jobs.isFetching} onClick={() => void refresh()}>Refresh distillation jobs</button></div>
    <p>ACT → ACT256 · local CPU · saved inference policy. SmolVLA and cross-family distillation are not supported yet.</p>
    {selected && <article className="native-simulation-result" aria-label="Distillation job details" data-job-id={selected.id}>
      <div className="native-result-header"><h3>ACT256 student</h3><span className={`status status-${selected.status}`}>{selected.status}</span></div>
      <p className="native-result-summary">{selected.stage ?? 'Queued'} · {selected.id}</p>
      {selected.error && <p role="alert">{selected.error}</p>}
      {report && <><p>{report.dataset_kind === 'generated_fixture' ? 'Generated observations · software verification only' : 'Recorded observations · offline imitation only'}</p><dl className="cloud-run-facts distillation-facts"><div><dt>Teacher inference tensors</dt><dd title={`${report.teacher_inference_tensor_bytes.toLocaleString()} bytes`}>{size(report.teacher_inference_tensor_bytes)}</dd></div><div><dt>Student weights</dt><dd title={`${report.student_weights_bytes.toLocaleString()} bytes`}>{size(report.student_weights_bytes)}</dd><small>{(100 * (1 - report.student_weights_bytes / report.teacher_inference_tensor_bytes)).toFixed(1)}% fewer weight bytes</small></div><div><dt>Validation imitation error</dt><dd>{report.trained_student.validation.teacher_normalized_l1.toPrecision(5)}</dd></div><div><dt>Final imitation error</dt><dd>{report.trained_student.final.teacher_normalized_l1.toPrecision(5)}</dd></div></dl><p>Masked normalized L1 against teacher actions. Smaller weights and lower imitation error do not prove task success or faster execution. Calibration and simulation performance remain unverified.</p></>}
      {selected.status === 'succeeded' && !report && <p role="alert">Complete distillation measurements are unavailable. Job completion alone does not establish student quality.</p>}
      {selected.status === 'succeeded' && outputs.map(item => <p key={item.id}><a className="secondary-button" href={artifactDownloadUrl(projectId, item.id)}>Download tested student package</a></p>)}
      {selected.status === 'succeeded' && outputs.length > 0 && <button className="text-link" onClick={() => onQuantize(outputs[0].id)}>Open ACT quantization</button>}
      {isActive(selected) && <><progress aria-label="Distillation in progress" /><button className="secondary-button" disabled={cancelling || jobs.isError} onClick={() => setCancelId(selected.id)}>Cancel selected distillation</button></>}
      {cancelId === selected.id && isActive(selected) && <div className="warning-box" role="group" aria-label="Confirm distillation cancellation"><p>Stop this job and its owned local processes?</p><button className="secondary-button" disabled={cancelling || jobs.isError} onClick={() => void cancel()}>Confirm cancellation</button><button className="text-link" disabled={cancelling} onClick={() => setCancelId('')}>Keep running</button></div>}
      <WorkbenchDisclosure title="Activity and measurements"><p>Inference-only output; interrupted training cannot resume from this student.</p>{events.isError && <p role="alert">Activity unavailable. {events.error.message}</p>}<pre className="cloud-log-tail" aria-label="Distillation activity">{events.data?.map(item => `${item.timestamp} · ${item.stage} · ${item.message}`).join('\n') || 'No recorded activity yet.'}</pre>{result && <pre className="cloud-log-tail">{JSON.stringify(result.reports, null, 2)}</pre>}</WorkbenchDisclosure>
    </article>}
    {history.length > 0 && <label className="distillation-history">Saved distillation job<select aria-label="Saved distillation job" value={selected?.id ?? ''} onChange={event => selectJob(event.target.value)}><option value="">Choose a recorded job</option>{history.map(item => <option key={item.id} value={item.id}>{item.id.slice(0, 8)} · {item.status}</option>)}</select></label>}
    {jobs.isError && <p role="alert">Job updates are unavailable; displayed status may be stale. {jobs.error.message}</p>}
    {options.isError && <p role="alert">Worker options are unavailable. {options.error.message}</p>}
    {artifacts.isError && <p role="alert">Teacher policies are unavailable. {artifacts.error.message}</p>}
    {error && <p role="alert">{error}</p>}
    {attempt.data?.state === 'pending' && <p role="status">{attempt.data.message}</p>}
    {attempt.data?.state === 'uncertain' && <div className="warning-box"><p>{attempt.data.message}</p><button className="secondary-button" disabled={!reviewed || jobs.isError} onClick={() => { try { saveAttempt(null); setReviewed(false); setError(''); } catch { setJournalReady(false); setError('Browser session storage is unavailable.'); } }}>I checked recorded jobs; allow a new request</button></div>}
    <WorkbenchDisclosure key={selected ? 'another' : 'first'} title={selected ? 'Prepare another student' : 'Prepare a student'} initiallyOpen={!selected}>
      {!projectId && <p role="status">Select a project to prepare a student.</p>}
      {options.isSuccess && !runtimes.length && <p role="status">No local ACT distillation worker is configured. Install the isolated worker and dataset reader, then register both in application settings.</p>}
      <fieldset className="native-simulation-form" disabled={!projectId || !!attempt.data || cancelling}><legend>Teacher and observations</legend>
        <label>Distillation worker<select aria-label="Distillation worker" value={runtime?.id ?? ''} onChange={event => setRuntime(event.target.value)}><option value="">Choose a local worker</option>{runtimes.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
        <label>ACT teacher<select aria-label="ACT teacher" value={teacherId} onChange={event => { setTeacher(event.target.value); setAttested(false); }}><option value="">Choose a complete local policy</option>{teachers.map(item => <option key={item.id} value={item.id}>{item.label} · {item.id.slice(0, 8)}</option>)}</select></label>
        <label>Verified dataset<select aria-label="Verified dataset" value={datasetId} onChange={event => { setDataset(event.target.value); setAttested(false); setTrain(''); setValidation(''); setFinal(''); }}><option value="">Choose a dataset snapshot</option>{datasets.map(item => <option key={item.id} value={item.id}>{item.result!.repo_id ?? 'Local robotics dataset'} · {item.result!.total_episodes} episodes · {item.id.slice(0, 8)}</option>)}</select></label>
        {!datasets.length && <p>A complete local LeRobot-v3 snapshot with episode lineage is required. Metadata-only Hugging Face inspection cannot supply training frames.</p>}
        <button type="button" className="text-link" onClick={onDataset}>Open Dataset intake</button>
        {dataset && <p>State order: {coordinateNames(dataset.result!.features['observation.state'])}. Action order: {coordinateNames(dataset.result!.features.action)}. Exact camera resolution and all selected lineage groups are verified before training.</p>}
        <div className="distillation-partitions"><label>Training episodes<input value={train} onChange={event => setTrain(event.target.value)} placeholder="Episode numbers, separated by commas" /></label>
        <label>Validation episodes<input value={validation} onChange={event => setValidation(event.target.value)} placeholder="Separate episodes" /></label>
        <label>Final episodes<input value={final} onChange={event => setFinal(event.target.value)} placeholder="Untouched final episodes" /></label></div>
        <p>Related demonstrations must stay in one partition. The final partition is assessed only after saving the fixed last-step student. No automatic random split is applied.</p>
        <label>Six coordinate units<input value={units} maxLength={500} onChange={event => { setUnits(event.target.value); setAttested(false); }} placeholder="One unit for each recorded action coordinate" /></label>
        <label className="distillation-check"><input type="checkbox" checked={generated} onChange={event => { setGenerated(event.target.checked); setAttested(false); }} /> This snapshot contains generated test observations.</label>
        <label className="distillation-check"><input type="checkbox" checked={attested} onChange={event => setAttested(event.target.checked)} /> I verified that the dataset coordinates, order and camera match this teacher’s saved processors. No implicit unit conversion is allowed.</label>
        <WorkbenchDisclosure title="Training recipe" initiallyOpen={false}>
          <label>Training steps<input type="number" min="1" max="10000" value={steps} onChange={event => setSteps(event.target.value)} /></label>
          <label>Frame stride<input type="number" min="1" max="10000" value={stride} onChange={event => setStride(event.target.value)} /></label>
          <label>Learning rate<input type="number" min="0.0000001" max="0.001" step="any" value={rate} onChange={event => setRate(event.target.value)} /></label>
          <label>Random seed<input type="number" min="0" max="2147483647" value={seed} onChange={event => setSeed(event.target.value)} /></label>
          <label>Distillation timeout (seconds)<input type="number" min="30" max="3600" value={timeout} onChange={event => setTimeoutValue(event.target.value)} /></label>
          <p>ACT256 · 2 encoder layers · 1 decoder layer · frozen copied visual backbone. At most 256 sampled observations; this recipe does not resume optimizer state.</p>
        </WorkbenchDisclosure>
        {invalid && <p className="field-hint">{invalid}</p>}
        <button type="button" className="primary-button" disabled={!ready} onClick={() => void submit()}>Train ACT256 student</button>
      </fieldset>
    </WorkbenchDisclosure>
  </section>;
}
