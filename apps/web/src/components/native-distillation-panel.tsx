'use client';

import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api, artifactDownloadUrl, isActive, isDatasetJob, type Job } from '@/lib/api';
import { cancelStudent, episodeSelection, startStudent, studentDataset, studentJob, studentReport, studentRuntime, studentTeacher, type DistillationRecipe, type DistillationRequest } from '@/lib/native-distillation';
import { record, UncertainPolicyJob } from '@/lib/policy-job-mutation';
import { hasSimulatorControlContract } from '@/lib/native-quantization';
import { storeAttempt, storedAttempt, type PolicyJobAttempt } from '@/lib/policy-job-attempt';
import { useNativeCancellation } from '@/lib/native-cancellation-attempt';
import { WorkflowChoiceGrid } from './workflow-choice-grid';
import { NativePreparation } from './native-preparation';
import { publicPath } from '@/lib/base-path';
import { WorkbenchDisclosure } from './workbench-disclosure';

function coordinateNames(value: unknown): string { return record(value) && Array.isArray(value.names) && value.names.every(item => typeof item === 'string') ? value.names.join(', ') : 'not declared'; }

type Attempt = PolicyJobAttempt;
const size = (bytes: number) => `${(bytes / 1024 ** 2).toFixed(1)} MiB`;

class JournalUnavailable extends Error {
  constructor() { super('Browser session storage is unavailable. Restore it and reload, then inspect recorded jobs before submitting again.'); }
}

export function NativeDistillationPanel({ projectId, preferredJobId, preferredTeacherArtifactId, onDataset, onQuantize }: { projectId: string; preferredJobId?: string; preferredTeacherArtifactId?: string; onDataset: () => void; onQuantize: (artifactId: string) => void }) {
  const client = useQueryClient();
  const options = useQuery({ queryKey: ['policy-options'], queryFn: api.policyOptions, retry: false, refetchInterval: 10_000 });
  const jobs = useQuery({ queryKey: ['jobs', projectId], queryFn: () => api.jobs(projectId), enabled: !!projectId, retry: false, refetchInterval: 2_000 });
  const artifacts = useQuery({ queryKey: ['artifacts', projectId], queryFn: () => api.artifacts(projectId), enabled: !!projectId, retry: false, refetchInterval: 5_000 });
  const attemptKey = ['distillation-attempt', projectId];
  const attempt = useQuery<Attempt>({ queryKey: attemptKey, queryFn: async () => null, enabled: false, initialData: null, gcTime: Infinity });
  const preference = useRef({ id: preferredTeacherArtifactId, consumed: false });
  const [pendingTeacher, setPendingTeacher] = useState(!!preferredTeacherArtifactId);
  const [teacherId, setTeacher] = useState(''), [datasetId, setDataset] = useState(''), [runtimeId, setRuntime] = useState('');
  const [train, setTrain] = useState(''), [validation, setValidation] = useState(''), [final, setFinal] = useState('');
  const [units, setUnits] = useState(''), [attested, setAttested] = useState(false), [generated, setGenerated] = useState(false);
  const [steps, setSteps] = useState('100'), [stride, setStride] = useState('30'), [rate, setRate] = useState('0.0001'), [seed, setSeed] = useState('1729'), [timeout, setTimeoutValue] = useState('600');
  const [selectedId, setSelectedId] = useState(preferredTeacherArtifactId ? '' : preferredJobId ?? ''), [accepted, setAccepted] = useState<Job | null>(null);
  const [journalReady, setJournalReady] = useState(false);
  const [error, setError] = useState(''), [cancelId, setCancelId] = useState(''), [cancelling, setCancelling] = useState(false), [reviewed, setReviewed] = useState(false);
  const mounted = useRef(true), busy = useRef(false), currentId = useRef(preferredTeacherArtifactId ? '' : preferredJobId ?? '');
  const attemptVersion = useRef(0);
  const selectionGeneration = useRef(0);
  const cancellation = useNativeCancellation({ project: projectId, workflow: 'distillation' });
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  useEffect(() => {
    try {
      if (!client.getQueryData<Attempt>(['distillation-attempt', projectId])) client.setQueryData<Attempt>(['distillation-attempt', projectId], storedAttempt('policy.distill', projectId));
      setJournalReady(true);
    } catch { setError('Browser session storage is unavailable. Enable it before submitting a job so uncertain requests can be recovered after a reload.'); }
  }, [client, projectId]);
  function saveAttempt(value: Attempt) {
    try { storeAttempt('policy.distill', projectId, value); }
    catch {
      const failure = new JournalUnavailable();
      if (client.getQueryData<PolicyJobAttempt>(attemptKey)?.state === 'pending') client.setQueryData<PolicyJobAttempt>(attemptKey, (): PolicyJobAttempt => ({ state: 'uncertain', message: failure.message }));
      if (mounted.current) setJournalReady(false);
      throw failure;
    }
    client.setQueryData<Attempt>(attemptKey, value);
  }
  const runtimes = (options.data?.runtimes ?? []).filter(studentRuntime);
  const runtime = runtimeId ? runtimes.find(item => item.id === runtimeId) : runtimes[0];
  const teachers = (artifacts.data ?? []).filter(item => studentTeacher(item, projectId));
  const contractTeachers = (artifacts.data ?? []).some(item => item.project_id === projectId && ['native_checkpoint', 'inference_export'].includes(item.format) && item.metadata?.architecture === 'act' && hasSimulatorControlContract(item));
  const teacher = teachers.find(item => item.id === teacherId);
  useEffect(() => {
    if (preference.current.id !== preferredTeacherArtifactId) { preference.current = { id: preferredTeacherArtifactId, consumed: false }; setPendingTeacher(!!preferredTeacherArtifactId); }
    if (!preferredTeacherArtifactId || preference.current.consumed || !artifacts.isSuccess || artifacts.isError) return;
    const candidate = teachers.find(item => item.id === preferredTeacherArtifactId);
    if (!candidate) return;
    preference.current.consumed = true; setPendingTeacher(false);
    setTeacher(candidate.id); setAttested(false);
  }, [preferredTeacherArtifactId, teachers, artifacts.isSuccess, artifacts.isError]);
  function consumeTeacherPreference() { preference.current = { id: preferredTeacherArtifactId, consumed: true }; setPendingTeacher(false); }
  function chooseTeacher(id: string) { consumeTeacherPreference(); setTeacher(id); setAttested(false); }
  const continuedTeacher = teacher && teacher.id === preferredTeacherArtifactId ? teacher : undefined;
  const datasets = (jobs.data ?? []).filter(isDatasetJob).filter(item => studentDataset(item, projectId));
  const dataset = datasets.find(item => item.id === datasetId);
  const history = (jobs.data ?? []).filter(studentJob).filter(item => item.project_id === projectId).sort((a, b) => b.created_at.localeCompare(a.created_at));
  const picked = history.find(item => item.id === selectedId) ?? (accepted?.id === selectedId && studentJob(accepted) ? accepted : undefined);
  const selected = picked && picked.project_id === projectId ? picked : undefined;
  const events = useQuery({ queryKey: ['events', selected?.id, selected?.status], queryFn: () => api.events(selected!.id), enabled: !!selected, retry: false, refetchInterval: selected && isActive(selected) ? 2_000 : false });
  const result: Record<string, unknown> | null = selected?.result && record(selected.result) ? selected.result : null;
  const report = selected && Array.isArray(result?.reports) ? result.reports.map(value => studentReport(value, selected)).find(Boolean) : null;
  const outputs = selected?.result && 'artifacts' in selected.result ? (selected.result.artifacts ?? []).filter(item => item.project_id === projectId && item.job_id === selected.id && item.format === 'native_checkpoint' && item.metadata?.recipe === 'act-action-distillation-v1') : [];
  function selectJob(id: string) { consumeTeacherPreference(); selectionGeneration.current += 1; cancellation.selectionChanged(); currentId.current = id; setSelectedId(id); setCancelId(''); }
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
  const ready = journalReady && !!projectId && options.isSuccess && !options.isError && jobs.isSuccess && !jobs.isError && artifacts.isSuccess && !artifacts.isError && !!runtime && !!teacher && !!dataset && !!recipe && attested && !attempt.data && !cancelling && cancellation.attempt?.state !== 'pending';
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
      if (mounted.current) { setAccepted(job); selectJob(job.id); }
      saveAttempt(null);
      await client.invalidateQueries({ queryKey: ['jobs', projectId] });
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : 'The request failed.';
      attemptVersion.current += 1;
      if (mounted.current) setReviewed(false);
      if (!(cause instanceof JournalUnavailable)) {
        try { saveAttempt(cause instanceof UncertainPolicyJob ? { state: 'uncertain', message } : null); } catch { if (mounted.current) setJournalReady(false); }
      }
      if (mounted.current) setError(message);
    } finally { busy.current = false; }
  }
  async function cancel() {
    if (!selected || selected.id !== cancelId || !isActive(selected) || jobs.isError || busy.current || !cancellation.available || cancellation.attempt) return;
    const generation = selectionGeneration.current;
    const stillSelected = () => mounted.current && currentId.current === selected.id && selectionGeneration.current === generation;
    busy.current = true; setCancelling(true); setError('');
    try {
      const job = await cancellation.run(selected, stillSelected, guard => cancelStudent(selected, guard));
      if (job && stillSelected()) { setAccepted(job as Job); selectJob(job.id); }
    } finally { busy.current = false; if (mounted.current) { setCancelling(false); setCancelId(''); } }
  }
  return <section className="panel native-simulation native-distillation native-workflow" aria-label="ACT distillation">
    <div className="native-workflow-toolbar"><span className="native-model-badge">ACT → ACT256</span><button className="text-link" aria-label="Refresh distillation jobs" disabled={!projectId || jobs.isFetching} onClick={() => void refresh()}>Refresh</button></div>
    {cancellation.error && <p role="alert">{cancellation.error}</p>}
    {cancellation.receipt && <p role="status">Cancellation response for {cancellation.receipt.jobId}: {cancellation.receipt.status}. Recorded job history remains authoritative.</p>}
    {cancellation.attempt && <section className="warning-box" aria-label="Cancellation recovery">
      <p>{cancellation.attempt.state === 'pending' ? 'Cancellation request pending' : 'Cancellation outcome is unverified'} for {cancellation.attempt.jobId}. No automatic retry was made. Recovery is limited to this browser tab.</p>
      {cancellation.attempt.state === 'uncertain' && <>
        {cancellation.reviewedTarget && <p role="status">Fresh cancellation history for {cancellation.reviewedTarget.jobId}: {cancellation.reviewedTarget.status}. This is the status returned by your recovery refresh.</p>}
        <button className="secondary-button" disabled={!cancellation.available || cancellation.refreshing} onClick={() => { const generation = selectionGeneration.current; void cancellation.refresh(generation, () => mounted.current && selectionGeneration.current === generation); }}>Refresh cancellation history</button>
        <button className="secondary-button" disabled={!cancellation.available || !cancellation.canAcknowledge || cancellation.refreshing || jobs.isError} onClick={() => { if (!jobs.isError) cancellation.acknowledge(selectionGeneration.current); }}>I reviewed cancellation history; allow another cancellation</button>
      </>}
    </section>}
    {selected && <article className="native-simulation-result" aria-label="Distillation job details" data-job-id={selected.id}>
      <div className="native-result-header"><h3>ACT256 student</h3><span className={`status status-${selected.status}`}>{selected.status}</span></div>
      <p className="native-result-summary">{isActive(selected) ? selected.stage ?? selected.status : 'Recorded job'} · {selected.id}</p>
      {selected.error && <p role="alert">{selected.error}</p>}
      {report && <><p>{report.dataset_kind === 'generated_fixture' ? 'Generated observations · software verification only' : 'Recorded observations · offline imitation only'}</p><dl className="cloud-run-facts distillation-facts"><div><dt>Teacher inference tensors</dt><dd title={`${report.teacher_inference_tensor_bytes.toLocaleString()} bytes`}>{size(report.teacher_inference_tensor_bytes)}</dd></div><div><dt>Student weights</dt><dd title={`${report.student_weights_bytes.toLocaleString()} bytes`}>{size(report.student_weights_bytes)}</dd><small>{(100 * (1 - report.student_weights_bytes / report.teacher_inference_tensor_bytes)).toFixed(1)}% fewer weight bytes</small></div><div><dt>Validation imitation error</dt><dd>{report.trained_student.validation.teacher_normalized_l1.toPrecision(5)}</dd></div><div><dt>Final imitation error</dt><dd>{report.trained_student.final.teacher_normalized_l1.toPrecision(5)}</dd></div></dl><p>Imitation metrics do not prove task success. Speed, calibration and simulation performance remain unverified.</p></>}
      {selected.status === 'succeeded' && !report && <p role="alert">Complete distillation measurements are unavailable. Job completion alone does not establish student quality.</p>}
      {selected.status === 'succeeded' && outputs.length > 0 && <div className="native-result-actions"><button className="primary-button" onClick={() => onQuantize(outputs[0].id)}>Open ACT quantization</button>{outputs.map(item => <a key={item.id} className="secondary-button" href={artifactDownloadUrl(projectId, item.id)}>Download tested student package</a>)}</div>}
      {isActive(selected) && <><progress aria-label="Distillation in progress" /><button className="secondary-button" disabled={cancelling || jobs.isError || !cancellation.available || !!cancellation.attempt} onClick={() => setCancelId(selected.id)}>Cancel selected distillation</button></>}
      {cancelId === selected.id && isActive(selected) && <div className="warning-box" role="group" aria-label="Confirm distillation cancellation"><p>Stop this job and its owned local processes?</p><button className="secondary-button" disabled={cancelling || jobs.isError || !cancellation.available || !!cancellation.attempt} onClick={() => void cancel()}>Confirm cancellation</button><button className="text-link" disabled={cancelling} onClick={() => setCancelId('')}>Keep running</button></div>}
      <WorkbenchDisclosure title="Activity and measurements"><p>Errors use masked normalized L1 against teacher actions.</p><p>Inference-only output; interrupted training cannot resume from this student.</p>{events.isError && <p role="alert">Activity unavailable. {events.error.message}</p>}<pre className="cloud-log-tail" aria-label="Distillation activity">{events.data?.map(item => `${item.timestamp} · ${item.stage} · ${item.message}`).join('\n') || 'No recorded activity yet.'}</pre>{result && <pre className="cloud-log-tail">{JSON.stringify(result.reports, null, 2)}</pre>}</WorkbenchDisclosure>
    </article>}
    {history.length > 0 && <label className="distillation-history">Saved distillation job<select aria-label="Saved distillation job" value={selected?.id ?? ''} onChange={event => selectJob(event.target.value)}><option value="">Choose a recorded job</option>{history.map(item => <option key={item.id} value={item.id}>{item.id.slice(0, 8)} · {item.status}</option>)}</select></label>}
    {jobs.isError && <p role="alert">Job updates are unavailable; displayed status may be stale. {jobs.error.message}</p>}
    {options.isError && <p role="alert">Worker options are unavailable. {options.error.message}</p>}
    {artifacts.isError && <p role="alert">Teacher policies are unavailable. {artifacts.error.message}</p>}
    {error && <p role="alert">{error}</p>}
    {journalReady && attempt.data?.state === 'pending' && <p role="status">{attempt.data.message}</p>}
    {attempt.data?.state === 'uncertain' && <div className="warning-box"><p>{attempt.data.message}</p><button className="secondary-button" disabled={!reviewed || jobs.isError} onClick={() => { try { saveAttempt(null); setReviewed(false); setError(''); } catch { setJournalReady(false); setError('Browser session storage is unavailable.'); } }}>I checked recorded jobs; allow a new request</button></div>}
    {preferredTeacherArtifactId && pendingTeacher && <p role="status" className="field-help">{artifacts.isPending ? 'Loading the selected training package…' : artifacts.isError ? 'The selected training package could not be checked. Refresh teacher policies before continuing.' : 'The selected training package is unavailable or unsupported. Refresh, or choose another teacher explicitly; no replacement has been selected.'}</p>}
    <NativePreparation key={selected ? 'another' : 'first'} title="Prepare another student" hasResult={!!selected}>
      {!projectId || !runtimes.length ? <div className="native-setup-empty">
        <p role="status">{!projectId ? 'Select a project to continue.' : options.isPending ? 'Loading workers…' : options.isError ? 'Worker availability is unknown.' : 'No local ACT distillation worker is configured.'}</p>
        <a className="text-link" href={publicPath('/guide/#distill')}>Set up distillation</a>
        <button className="primary-button" disabled>Train ACT256 student</button>
      </div> : <fieldset className="native-simulation-form" disabled={!!attempt.data || cancelling || cancellation.attempt?.state === 'pending'}><legend className="visually-hidden">Distillation setup</legend>
        {continuedTeacher && <div className="distillation-continuation" aria-label="Teacher from training">
          <span className="distillation-continuation-step">Training → Distill</span>
          <h3>Your exported teacher is selected</h3>
          <p>{continuedTeacher.label} <span>· Export {continuedTeacher.job_id.slice(0, 8)}</span></p>
          <small>{continuedTeacher.id}</small>
          <p>Choose a prepared dataset and independent episode splits, then confirm its coordinates. Training starts only when you submit.</p>
        </div>}
        <WorkflowChoiceGrid name="teacher" label="Teacher" value={teacherId} onChange={chooseTeacher} options={teachers.map(item => ({ value: item.id, label: item.label, meta: item.id.slice(0, 8), icon: 'layers' }))} emptyMessage="No ACT teacher policies. Import or export a complete policy first." />
        {contractTeachers && <p role="status">Simulator-bound ACT packages are excluded because distillation cannot preserve their control contract yet. Keep the original package for Run compatibility checks.</p>}
        <WorkflowChoiceGrid name="student-dataset" label="Dataset" value={datasetId} onChange={value => { setDataset(value); setAttested(false); setTrain(''); setValidation(''); setFinal(''); }} options={datasets.map(item => ({ value: item.id, label: item.result!.repo_id ?? 'Local robotics dataset', description: `${item.result!.total_episodes} episodes`, icon: 'database' }))} emptyMessage="Import a complete dataset snapshot to continue." />
        {!datasets.length && <button type="button" className="text-link" onClick={onDataset}>Open Dataset intake</button>}
        {(runtimes.length > 1 || !runtime) && <WorkflowChoiceGrid name="student-worker" label="Compute" value={runtime?.id ?? ''} onChange={setRuntime} options={runtimes.map(item => ({ value: item.id, label: item.label, icon: 'sliders' }))} />}
        {teacher && dataset && <>
          <section className="native-form-section" aria-labelledby="distillation-splits-title"><h3 id="distillation-splits-title">Episode split</h3><div className="distillation-partitions">
            <label>Training episodes<input value={train} onChange={event => setTrain(event.target.value)} placeholder="0, 1" /></label>
            <label>Validation episodes<input value={validation} onChange={event => setValidation(event.target.value)} placeholder="2, 3" /></label>
            <label>Final episodes<input value={final} onChange={event => setFinal(event.target.value)} placeholder="4, 5" /></label>
          </div></section>
          <label>Six coordinate units<input value={units} maxLength={500} onChange={event => { setUnits(event.target.value); setAttested(false); }} placeholder="Six units in dataset order, comma-separated" /></label>
          <label className="distillation-check"><input type="checkbox" checked={generated} onChange={event => { setGenerated(event.target.checked); setAttested(false); }} /> This snapshot contains generated test observations.</label>
          <label className="distillation-check"><input type="checkbox" checked={attested} onChange={event => setAttested(event.target.checked)} /> I verified that the dataset coordinates, order and camera match this teacher’s saved processors. No implicit unit conversion is allowed.</label>
          <WorkbenchDisclosure title="Training recipe">
            <div className="native-recipe-controls">
              <label>Training steps<input type="number" min="1" max="10000" value={steps} onChange={event => setSteps(event.target.value)} /></label>
              <label>Frame stride<input type="number" min="1" max="10000" value={stride} onChange={event => setStride(event.target.value)} /></label>
              <label>Learning rate<input type="number" min="0.0000001" max="0.001" step="any" value={rate} onChange={event => setRate(event.target.value)} /></label>
              <label>Random seed<input type="number" min="0" max="2147483647" value={seed} onChange={event => setSeed(event.target.value)} /></label>
              <label>Distillation timeout (seconds)<input type="number" min="30" max="3600" value={timeout} onChange={event => setTimeoutValue(event.target.value)} /></label>
            </div>
            <p>State order: {coordinateNames(dataset.result!.features['observation.state'])}. Action order: {coordinateNames(dataset.result!.features.action)}.</p>
            <p>Related demonstrations must stay in one partition. Final episodes are checked only after saving the last-step student; splits are never random.</p>
            <p>ACT256 · 2 encoder layers · 1 decoder layer · frozen visual backbone. At most 256 sampled observations. No optimizer-state resume.</p>
          </WorkbenchDisclosure>
          {invalid && (train || validation || final || units) && <p className="field-hint">{invalid}</p>}
        </>}
        <button type="button" className="primary-button" disabled={!ready} onClick={() => void submit()}>Train ACT256 student</button>
      </fieldset>}
    </NativePreparation>
  </section>;
}
