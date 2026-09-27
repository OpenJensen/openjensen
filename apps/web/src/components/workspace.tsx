'use client';

import { QueryClient, QueryClientProvider, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useRef, useState, type FormEvent } from 'react';
import { isAugmentationSource } from '@/lib/augmentation-source';
import { api, isActive, isDatasetJob, type DatasetJob, type DatasetProfile, type Job, type Project } from '@/lib/api';
import { engineRuntime, initialQuantizeEntry, initialRunEntry, runJobMode, type Entry, type ProjectEntry, type QuantizeMode, type RunMode } from '@/lib/workflow-entry';
import { storedSimulationAttempt } from '@/lib/native-simulation-recovery';
import { storedAttempt, type PolicyJobAttempt } from '@/lib/policy-job-attempt';
import { replayRuntime } from '@/lib/native-replay';
import { simulationOptions } from '@/lib/native-simulation';
import type { SimulationHandoff } from '@/lib/native-simulation-handoff';
import { NativeQuantizationPanel } from '@/components/native-quantization-panel';
import { NativeReplayPanel } from '@/components/native-replay-panel';
import { DistillationPanel, type DistillationModel } from '@/components/distillation-panel';
import { NativeSimulationPanel } from '@/components/native-simulation-panel';
import { CloudRuns } from '@/components/cloud-runs';
import { WorkflowPanel } from '@/components/workflow-panel';
import { TeachingPanel } from '@/components/teaching-panel';
import { ManagedTeachingPanel } from '@/components/managed-teaching-panel';
import type { ManagedPublishedCapture } from '@/lib/managed-teaching';
import { RecordingPreparationPanel } from '@/components/recording-preparation-panel';
import { DecisionPanel } from '@/components/decision-panel';
import { TrainingPanel } from '@/components/training-panel';
import { AugmentationPanel } from '@/components/augmentation-panel';
import { Icon } from '@/components/icon';
import { WorkspaceShell } from '@/components/workspace-shell';
import { WorkspaceJourney } from '@/components/workspace-journey';
import { DatasetExplorer } from '@/components/dataset-explorer';
import { DatasetStarters } from '@/components/dataset-starters';
import { datasetStarters, type DatasetStarter } from '@/lib/dataset-starters';
import { publicPath } from '@/lib/base-path';
import { guideSectionId } from '@/lib/guide-section';
import { useDurableSubmission } from '@/lib/durable-submission';
import { intakeAcknowledgement, validatedProjectHistory } from '@/lib/dataset-submission';
import { DatasetSubmissionRecovery } from '@/components/dataset-submission-recovery';
import { WorkbenchDisclosure } from '@/components/workbench-disclosure';

const stages = [
  { name: 'Dataset', icon: 'database' },
  { name: 'Fine-tune', icon: 'sliders' },
  { name: 'Distill', icon: 'layers' },
  { name: 'Quantize', icon: 'compress' },
  { name: 'Evaluate', icon: 'chart' },
  { name: 'Run', icon: 'play' },
  { name: 'Settings & diagnostics', icon: 'sliders' },
  { name: 'Cloud runs', icon: 'clock' },
  { name: 'Augmentation', icon: 'spark' },
  { name: 'Teaching', icon: 'plus' },
  { name: 'Decision lab', icon: 'chart' },
] as const;

const navigationGroups = [
  { name: 'Data', items: [0, 8, 9] },
  { name: 'Train', items: [1, 2, 3] },
  { name: 'Test', items: [4, 5, 10] },
  { name: 'Workspace', items: [7, 6] },
];

function ModeCard({ title, detail, icon, selected, disabled, onClick }: { title: string; detail: string; icon?: Parameters<typeof Icon>[0]['name']; selected: boolean; disabled: boolean; onClick: () => void }) {
  return <button type="button" className={`mode-card${icon ? '' : ' mode-card-plain'}${selected ? ' selected' : ''}`} aria-label={title} aria-pressed={selected} disabled={disabled} onClick={onClick}>
    <span className="mode-card-top">{icon ? <span className="mode-card-icon"><Icon name={icon} size={23} /></span> : <strong>{title}</strong>}<span className="mode-card-check">{selected && <Icon name="check" size={13} />}</span></span>
    {icon && <strong>{title}</strong>}<span className="mode-card-detail">{detail}</span>
  </button>;
}

function ErrorNotice({ error }: { error: Error | null }) {
  return error ? <p className="error-notice" role="alert">{error.message}</p> : null;
}

function JobStatus({ status }: { status: Job['status'] }) {
  return <span className={`status status-${status}`}><span className="status-dot" />{status}</span>;
}

function formatNumber(value: number) {
  return new Intl.NumberFormat('en-US').format(value);
}

function displayDate(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function DatasetResult({ profile }: { profile: DatasetProfile }) {
  return <section className="result inspection-overview" aria-labelledby="result-title">
    {profile.snapshot && <p role="status">Training copy verified · {profile.snapshot.file_count} files · {profile.snapshot.lineage_validated ? 'Recorded lineage retained' : 'Ancestry unknown; not independent evaluation evidence'}</p>}
    <div className="result-heading">
      <div><h3 id="result-title">{profile.repo_id || 'Local dataset'}</h3></div>
      <span className="metadata-badge"><Icon name="check" size={14} /> {profile.format.replace('_', ' ')}</span>
    </div>
    <dl className="dataset-facts">
      <div><dt>episodes</dt><dd>{formatNumber(profile.total_episodes)}</dd></div>
      <div><dt>frames</dt><dd>{formatNumber(profile.total_frames)}</dd></div>
      <div><dt>fps</dt><dd>{profile.fps}</dd></div>
      <div><dt className="visually-hidden">Robot type</dt><dd>{profile.robot_type || 'Unknown robot'}</dd></div>
    </dl>
    <details className="provenance inspection-provenance">
      <summary>Source details</summary>
      {profile.warnings.length > 0 && <div className="warning-box"><h4>Inspection notes</h4><ul>{profile.warnings.map((warning, index) => <li key={`${index}-${warning}`}>{warning}</li>)}</ul></div>}
      <dl>
        <div><dt>Source</dt><dd>{profile.repo_id || 'Local metadata snapshot'}</dd></div>
        <div><dt>Declared license</dt><dd>{profile.license || 'Not declared'}</dd></div>
        <div><dt>Resolved revision</dt><dd><code>{profile.revision}</code></dd></div>
        <div><dt>Metadata SHA-256</dt><dd><code>{profile.metadata_sha256}</code></dd></div>
        <div><dt>Inspected</dt><dd>{displayDate(profile.inspected_at)}</dd></div>
      </dl>
      {profile.source === 'huggingface' && profile.repo_id && <a className="text-link" href={`https://huggingface.co/datasets/${profile.repo_id.split('/').map(encodeURIComponent).join('/')}/tree/${encodeURIComponent(profile.revision)}`} target="_blank" rel="noreferrer">Open pinned source <Icon name="external" size={13} /></a>}
    </details>
  </section>;
}

function IntakeForm({ project, readinessMessage, localAvailable, onCreated, starter, onSourceEdited, navigationToken }: { navigationToken: string; project: Project | undefined; readinessMessage: string; localAvailable: boolean; onCreated: (job: Job) => void; starter: DatasetStarter; onSourceEdited: () => void }) {
  const queryClient = useQueryClient();
  const [source, setSource] = useState<'huggingface' | 'local'>('huggingface');
  const [repoId, setRepoId] = useState(starter.repoId);
  const [revision, setRevision] = useState(starter.revision);
  const [path, setPath] = useState('');
  const [snapshotForTraining, setSnapshotForTraining] = useState(false);
  const submission = useDurableSubmission({ project: project?.id ?? '', operation: 'dataset.inspect' });
  const selection = useRef({ token: navigationToken, generation: 0 });
  if (selection.current.token !== navigationToken) selection.current = { token: navigationToken, generation: selection.current.generation + 1 };
  const mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  const validate = (value: unknown, body: unknown) => intakeAcknowledgement(value, project?.id ?? '', body);
  function accept(job: Job | null, generation: number) {
    if (!job) return;
    queryClient.setQueryData<Job[]>(['jobs', job.project_id], previous => [job, ...(previous ?? []).filter(item => item.id !== job.id)]);
    void queryClient.invalidateQueries({ queryKey: ['jobs', job.project_id] });
    if (mounted.current && selection.current.generation === generation) onCreated(job);
  }
  const blocked = !project || !submission.hydrated || !submission.available || !!submission.attempt || submission.busy;

  useEffect(() => {
    if (!localAvailable && source === 'local') setSource('huggingface');
  }, [localAvailable, source]);
  const mutation = useMutation({
    mutationFn: async () => {
      if (!project) throw new Error('Create or select a project first.');
      if (source === 'local' && !localAvailable) throw new Error('Local intake is not enabled on this application.');
      if (blocked) throw new Error('Verify the saved request before starting another inspection.');
      const generation = selection.current.generation;
      const job = await submission.submit(source === 'huggingface'
        ? { source, repo_id: repoId.trim(), revision: revision.trim() || 'main' }
        : { source, path: path.trim(), revision: 'main', snapshot_for_training: snapshotForTraining }, validate);
      accept(job, generation);
      return job;
    },
  });
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!blocked && !mutation.isPending) mutation.mutate();
  }
  return <section className="panel intake-panel" aria-labelledby="intake-title">
    <div className="panel-title"><h2 id="intake-title">Import a dataset</h2></div>
    <form onSubmit={submit}>
      <fieldset className="source-options">
        <legend className="visually-hidden">Dataset source</legend>
        <label className={source === 'huggingface' ? 'source-option selected' : 'source-option'}><input type="radio" name="source" value="huggingface" disabled={!project} checked={source === 'huggingface'} onChange={() => { setSource('huggingface'); mutation.reset(); onSourceEdited(); }} /><span>Hugging Face</span></label>
        <label className={`source-option${source === 'local' ? ' selected' : ''}${!localAvailable ? ' unavailable' : ''}`}><input type="radio" name="source" value="local" checked={source === 'local'} disabled={!project || !localAvailable} onChange={() => { setSource('local'); mutation.reset(); onSourceEdited(); }} /><span>Local directory</span>{!localAvailable && <span className="source-detail">Not enabled</span>}</label>
      </fieldset>
      {source === 'huggingface' ? <>
        <label className="field-label" htmlFor="repo-id">Dataset repository</label>
        <input id="repo-id" name="repo_id" disabled={!project} value={repoId} onChange={event => {
          setRepoId(event.target.value);
          onSourceEdited();
          if (revision === starter.revision) setRevision('');
        }} required placeholder="owner/dataset-name" autoCapitalize="none" autoCorrect="off" spellCheck={false} />
        <details className="intake-advanced"><summary>Revision (optional)</summary><label className="field-label visually-hidden" htmlFor="revision">Revision</label>
        <input id="revision" className="mono-input" name="revision" disabled={!project} value={revision} onChange={event => { setRevision(event.target.value); onSourceEdited(); }} placeholder="Latest (main)" aria-describedby="revision-help" autoCapitalize="none" autoCorrect="off" spellCheck={false} />
        <p id="revision-help" className="field-help">Leave blank for the latest revision, or enter a branch, tag, or commit.</p>
        </details>
      </> : <>
        <label className="field-label" htmlFor="local-path">Dataset directory</label>
        <input id="local-path" name="path" disabled={!project} value={path} onChange={event => setPath(event.target.value)} required placeholder="Path to a LeRobot dataset" autoCapitalize="none" autoCorrect="off" spellCheck={false} aria-describedby="path-help" />
        <p id="path-help" className="field-help">Path on the API host, inside the configured dataset directory.</p>
        <label className="field-label"><input type="checkbox" checked={snapshotForTraining} onChange={event => setSnapshotForTraining(event.target.checked)} /> Prepare immutable training copy</label>
        <p className="field-help">Validate all rows and videos in a finalized LeRobot v3 dataset, then copy it into this workspace. Up to 16 GiB; source files stay unchanged.</p>
      </>}
      <ErrorNotice error={mutation.error} />
      <DatasetSubmissionRecovery submission={submission} onReconcile={async () => { const generation = selection.current.generation; accept(await submission.reconcile(validate), generation); }}
        onRetry={async () => { const generation = selection.current.generation; accept(await submission.retry(validate, async () => { if (!mounted.current || selection.current.generation !== generation) throw new Error('Selection changed; the saved request was not retried.'); }), generation); }} onReviewHistory={async () => {
          if (!project) return false;
          const fresh = validatedProjectHistory(await api.jobs(project.id), project.id);
          queryClient.setQueryData(['jobs', project.id], fresh);
          return fresh;
        }} />
      {!project && <p className="form-note" role="status">{readinessMessage}</p>}
            <button className="primary-button inspect-button" type="submit" disabled={blocked || mutation.isPending}>{mutation.isPending || submission.busy ? 'Checking dataset…' : 'Inspect dataset'}<Icon name="arrow" size={17} /></button>
    </form>
  </section>;
}

function JobDetail({ job, projectId }: { job: DatasetJob; projectId: string }) {
  const queryClient = useQueryClient();
  const cancel = useMutation({
    mutationFn: () => api.cancel(job.id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['jobs', projectId] }),
  });
  return <>
    {!job.result && <div className="job-detail-heading"><JobStatus status={job.status} /></div>}
    {isActive(job) && <div className="working-state" role="status"><span className="spinner" /><div><h3>{job.status === 'queued' ? 'Waiting to inspect' : 'Reading source metadata'}</h3></div><button className="secondary-button" onClick={() => cancel.mutate()} disabled={cancel.isPending}>{cancel.isPending ? 'Cancelling…' : 'Cancel'}</button></div>}
    <ErrorNotice error={cancel.error} />
    {job.error && <p className="error-notice" role="alert">{job.error}</p>}
    {(job.status === 'cancelled' || job.status === 'interrupted') && <p className="muted">Inspection did not complete.</p>}
    {job.result && 'inspection_scope' in job.result && <DatasetResult profile={job.result} />}
  </>;
}

function Workbench() {
  const queryClient = useQueryClient();
  const [projectId, setProjectId] = useState('');
  const [projectName, setProjectName] = useState('');
  const [selectedJobId, setSelectedJobId] = useState('');
  const [activeStage, setActiveStage] = useState(0);
  const [entries, setEntries] = useState<Record<string, ProjectEntry>>({});
  const [distillationModels, setDistillationModels] = useState<Record<string, DistillationModel | undefined>>({});
  const [entryError, setEntryError] = useState<{ projectId: string; stage: number; message: string } | null>(null);
  const [openSimulation, setOpenSimulation] = useState<{ projectId: string; id: string } | null>(null);
  const [quantizeArtifact, setQuantizeArtifact] = useState<{ projectId: string; artifactId: string } | null>(null);
  const [distillTeacher, setDistillTeacher] = useState<{ projectId: string; artifactId: string } | null>(null);
  const [replayArtifact, setReplayArtifact] = useState<{ projectId: string; artifactId: string } | null>(null);
  const [simulationArtifact, setSimulationArtifact] = useState<SimulationHandoff | null>(null);
  const [teachingMode, setTeachingMode] = useState<'manual' | 'managed'>('manual');
  const [teachingCapture, setTeachingCapture] = useState<ManagedPublishedCapture | null>(null);
  const [trainingNavigation, setTrainingNavigation] = useState(0);
  const [openTrainingRun, setOpenTrainingRun] = useState<{ projectId: string; id: string } | null>(null);
  const [workflowNavigation, setWorkflowNavigation] = useState(0);
  const [startTraining, setStartTraining] = useState<{ id: number; datasetId?: string }>();
  const [settingsTab, setSettingsTab] = useState<'compute' | 'settings' | 'diagnostics'>('compute');
  const [datasetView, setDatasetView] = useState<'sources' | 'inspection'>('sources');
  const [starter, setStarter] = useState(datasetStarters[0]);
  const [starterSelection, setStarterSelection] = useState(0);
  const [activeStarterId, setActiveStarterId] = useState(datasetStarters[0].id);
  const health = useQuery({ queryKey: ['health'], queryFn: api.health, refetchInterval: 15_000, retry: false });
  const projects = useQuery({ queryKey: ['projects'], queryFn: api.projects, retry: false });
  const capabilities = useQuery({ queryKey: ['capabilities'], queryFn: api.capabilities, retry: false });
  useEffect(() => {
    if (!health.isSuccess || health.isError) return;
    // A successful health poll can be the first request after API startup or
    // recovery. Retry the views that may have failed while it was unavailable.
    void queryClient.invalidateQueries({ queryKey: ['projects'] });
    void queryClient.invalidateQueries({ queryKey: ['capabilities'] });
  }, [health.isSuccess, health.isError, queryClient]);
  const jobs = useQuery({
    queryKey: ['jobs', projectId],
    queryFn: () => api.jobs(projectId),
    enabled: !!projectId,
    refetchInterval: query => query.state.data?.some(isActive) ? 1_000 : 5_000,
    retry: false,
  });
  useEffect(() => {
    if (!projects.data || projects.data.some(project => project.id === projectId)) return;
    let stored = '';
    try { stored = localStorage.getItem('firebird.project') ?? ''; } catch { /* Storage may be unavailable. */ }
    const selected = projects.data.find(project => project.id === stored) ?? projects.data[0];
    setProjectId(selected?.id ?? '');
  }, [projects.data, projectId]);
  function navigateStage(index: number) {
    setOpenTrainingRun(null);
    setOpenSimulation(null);
    setQuantizeArtifact(null);
    setDistillTeacher(null);
    setReplayArtifact(null);
    setSimulationArtifact(null);
    setTeachingCapture(null);
    setWorkflowNavigation(value => value + 1);
    if (index === 1) {
      setStartTraining(undefined);
      setTrainingNavigation(value => value + 1);
    }
    setActiveStage(index);
  }
  function startTrainingOnDataset(datasetId: string) {
    const id = trainingNavigation + 1;
    setTrainingNavigation(id);
    setStartTraining({ id, datasetId });
    setActiveStage(1);
  }
  function selectProject(id: string) {
    setOpenTrainingRun(null);
    setOpenSimulation(null);
    setQuantizeArtifact(null);
    setDistillTeacher(null);
    setReplayArtifact(null);
    setSimulationArtifact(null);
    setTeachingCapture(null); setTeachingMode('manual');
    setStartTraining(undefined);
    setProjectId(id);
    setSelectedJobId('');
    setDatasetView('sources');
    try { localStorage.setItem('firebird.project', id); } catch { /* Session selection still works. */ }
  }
  const projectMutation = useMutation({
    mutationFn: () => api.createProject(projectName.trim()),
    onSuccess: (project) => {
      queryClient.setQueryData<Project[]>(['projects'], previous => [...(previous ?? []), project]);
      setProjectName('');
      selectProject(project.id);
    },
  });
  const project = projects.data?.find(item => item.id === projectId);
  // Only confirmed membership may enable project-scoped workflow controls.
  const workflowProjectId = projects.isSuccess ? project?.id ?? '' : '';
  const teachingContext = useRef({ project: workflowProjectId, stage: activeStage, mode: teachingMode });
  teachingContext.current = { project: workflowProjectId, stage: activeStage, mode: teachingMode };
  function reviewTeachingCapture(capture: ManagedPublishedCapture) {
    const context = teachingContext.current;
    if (context.project !== capture.project_id || context.stage !== 9 || context.mode !== 'managed') return;
    setTeachingCapture(capture);
  }
  const choosingWorkflow = [3, 4, 5].includes(activeStage);
  const options = useQuery({ queryKey: ['policy-options'], queryFn: api.policyOptions, enabled: choosingWorkflow, retry: false, refetchInterval: choosingWorkflow ? 10_000 : false });
  const simulation = useQuery({ queryKey: ['simulation-options'], queryFn: simulationOptions, enabled: activeStage === 4 || activeStage === 5, retry: false, refetchInterval: activeStage === 4 || activeStage === 5 ? 10_000 : false });
  const context = workflowProjectId ? entries[workflowProjectId] : undefined;
  const quantizeMode = context?.quantize?.mode;
  const runMode = context?.run?.mode;
  function chooseQuantize(mode: QuantizeMode, origin: Entry<QuantizeMode>['origin'] = 'manual', jobId?: string) {
    if (!workflowProjectId) return;
    setEntries(previous => ({ ...previous, [workflowProjectId]: { ...previous[workflowProjectId], quantize: { mode, origin, jobId: jobId ?? (origin === 'manual' && previous[workflowProjectId]?.quantize?.mode === mode ? previous[workflowProjectId].quantize?.jobId : undefined) } } }));
    setEntryError(null);
  }
  function chooseRun(mode: RunMode, origin: Entry<RunMode>['origin'] = 'manual', jobId?: string) {
    if (!workflowProjectId) return;
    setEntries(previous => ({ ...previous, [workflowProjectId]: { ...previous[workflowProjectId], run: { mode, origin, jobId: jobId ?? (origin === 'manual' && previous[workflowProjectId]?.run?.mode === mode ? previous[workflowProjectId].run?.jobId : undefined) } } }));
    setEntryError(null);
  }
  useEffect(() => {
    const stage = activeStage === 3 ? 'quantize' : activeStage === 5 ? 'run' : null;
    if (!stage || !workflowProjectId || context?.[stage]) return;
    // An unfinished request has priority over restoring owned history.
    // Read only: the child owns its journal and explicit recovery acknowledgement.
    let recovery = false, simulationRecovery = false;
    try {
      const operation = stage === 'quantize' ? 'policy.quantize' : 'policy.run.replay';
      const cacheKey = [stage === 'quantize' ? 'native-quantization-attempt' : 'native-replay-attempt', workflowProjectId];
      recovery = !!queryClient.getQueryData<PolicyJobAttempt>(cacheKey) || !!storedAttempt(operation, workflowProjectId);
      if (stage === 'run') simulationRecovery = !!queryClient.getQueryData<PolicyJobAttempt>(['native-simulation-attempt', workflowProjectId]) || !!storedSimulationAttempt(workflowProjectId);
      if (recovery && simulationRecovery) { setEntryError({ projectId: workflowProjectId, stage: activeStage, message: 'Both observation replay and native simulation have unresolved requests. Choose either workflow to inspect its recovery record; neither request was retried.' }); return; }
    } catch {
      setEntryError({ projectId: workflowProjectId, stage: activeStage, message: 'Saved request recovery could not be read. Choose a workflow to inspect its history.' });
      return;
    }
    if (!recovery && !simulationRecovery && (!jobs.isSuccess || jobs.isError)) return;
    const next = stage === 'quantize'
      ? recovery ? { mode: 'native' as const, origin: 'recovery' as const } : initialQuantizeEntry(workflowProjectId, jobs.data!)
      : simulationRecovery ? { mode: 'native' as const, origin: 'recovery' as const } : recovery ? { mode: 'replay' as const, origin: 'recovery' as const } : initialRunEntry(workflowProjectId, jobs.data!);
    if (next) setEntries(previous => previous[workflowProjectId]?.[stage] ? previous : { ...previous, [workflowProjectId]: { ...previous[workflowProjectId], [stage]: next } });
    setEntryError(null);
  }, [activeStage, workflowProjectId, context, jobs.data, jobs.isSuccess, jobs.isError, queryClient]);
  const ownedJobs = (jobs.data ?? []).filter(job => job.project_id === workflowProjectId);
  const replayHistory = ownedJobs.some(job => runJobMode(job, workflowProjectId) === 'replay');
  const simulationHistory = ownedJobs.some(job => runJobMode(job, workflowProjectId) === 'native');
  const replayConfigured = options.isSuccess && !options.isError && options.data.runtimes.some(replayRuntime);
  const simulationConfigured = simulation.isSuccess && !simulation.isError && simulation.data.profiles.length > 0;
  const engineConfigured = options.isSuccess && !options.isError && options.data.runtimes.some(item => engineRuntime(item, activeStage === 4 ? 'Evaluate' : activeStage === 3 ? 'Quantize' : 'Run'));
  const entryPending = (activeStage === 3 && !quantizeMode) || (activeStage === 5 && !runMode);
  const entryFailed = jobs.isError || options.isError || (activeStage === 5 && simulation.isError);
  const recoveryError = entryError?.projectId === workflowProjectId && entryError.stage === activeStage ? entryError.message : null;
  const sortedJobs = [...(jobs.data ?? [])].filter(job => job.project_id === workflowProjectId).filter(isDatasetJob).sort((a, b) => b.created_at.localeCompare(a.created_at));
  const selectedJob = selectedJobId ? sortedJobs.find(job => job.id === selectedJobId) : sortedJobs[0];
  const connected = health.isSuccess && !health.isError;

  const stage = stages[activeStage] ?? stages[0];

  return <WorkspaceShell showGuideShortcut={false}
    breadcrumb={<><Icon name={stage.icon} size={18} /><strong>{stage.name}</strong><span className="breadcrumb-divider">/</span><span className="breadcrumb-project">{project?.name ?? 'No project selected'}</span></>}
    navigation={<>
      <section className="projects-section" aria-labelledby="projects-heading">
        <div className="sidebar-section-label"><h2 id="projects-heading">Project</h2><span>{projects.data?.length ?? '—'}</span></div>
        {projects.isPending && <p className="sidebar-note" role="status">Loading projects…</p>}
        <ErrorNotice error={projects.error} />
        {projects.isError && <button className="text-button" onClick={() => void projects.refetch()} disabled={projects.isFetching}>Retry projects</button>}
        {!!projects.data?.length && <div className="project-picker"><Icon name="folder" size={16} /><label className="visually-hidden" htmlFor="project-select">Current project</label><select id="project-select" value={projectId} onChange={event => selectProject(event.target.value)}>{projects.data.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></div>}
        {projects.data?.length === 0 && <p className="sidebar-note">No projects yet.</p>}
        <form className="project-form" onSubmit={event => { event.preventDefault(); if (projectName.trim()) projectMutation.mutate(); }}>
          <label htmlFor="project-name">New project</label>
          <div className="project-input-row"><input id="project-name" name="name" value={projectName} onChange={event => setProjectName(event.target.value)} required maxLength={100} placeholder="Project name" /><button type="submit" aria-label="Create project" title="Create project" disabled={!projectName.trim() || projectMutation.isPending}><Icon name="plus" size={17} /></button></div>
          {projectMutation.isPending && <p className="sidebar-note" role="status">Creating project…</p>}
          <ErrorNotice error={projectMutation.error} />
        </form>
      </section>
      <nav className="stage-navigation grouped-navigation" aria-label="Policy lifecycle">
        {navigationGroups.map(group => <div className="navigation-group" key={group.name}>
          <p className="sidebar-section-label">{group.name}</p>
          <ul className="stage-list">{group.items.map(index => {
            const item = stages[index];
            return <li key={item.name}><button type="button" className={`stage-button${activeStage === index ? ' selected' : ''}`} onClick={() => { if (index === 6) setSettingsTab('compute'); navigateStage(index); }} aria-current={activeStage === index ? 'page' : undefined}>
              <Icon name={item.icon} size={18} /><span>{item.name}</span>{index < 6 && <span className="stage-order" aria-hidden="true">{index + 1}</span>}
            </button></li>;
          })}</ul>
        </div>)}
      </nav>
    </>}
  >
        <div className={`page-heading${activeStage === 1 ? ' training-page-heading' : ''}`}><h1>{stage.name}</h1><div className="page-actions"><a className="page-guide" href={publicPath(`/guide/#${guideSectionId(stage.name)}`)}><Icon name="book" size={16} />Guide</a><div className={`connection ${connected ? 'connected' : ''}`} role="status" aria-label="Application API connection" title="Application API connection; simulator and voice connections are shown separately."><span />{health.isPending ? 'Connecting' : connected ? 'App connected' : 'App offline'}</div></div></div>
        {!connected && !health.isPending && <div className="connection-notice"><ErrorNotice error={health.error} /><button className="text-button" onClick={() => { void health.refetch(); void projects.refetch(); void capabilities.refetch(); }}>Retry connection</button></div>}
        {capabilities.error && connected && <div className="connection-notice"><ErrorNotice error={capabilities.error} /><button className="text-button" onClick={() => void capabilities.refetch()} disabled={capabilities.isFetching}>Retry capabilities</button></div>}
        <WorkspaceJourney key={`journey-${workflowProjectId}`} stage={activeStage} projectId={workflowProjectId} projectName={workflowProjectId ? project?.name : undefined}
          jobs={jobs.data ?? []} historyState={!workflowProjectId ? 'unselected' : jobs.isError ? 'unavailable' : jobs.isPending ? 'loading' : 'ready'}
          inspection={selectedJob} inspectionExplicit={!!selectedJobId}
          workflow={activeStage === 3 ? quantizeMode === 'native' ? 'ACT · native quantization' : quantizeMode === 'gguf' ? 'SmolVLA · GGUF' : undefined : activeStage === 5 ? runMode === 'native' ? '3D simulation · ACT or SmolVLA' : runMode === 'replay' ? 'ACT · observation replay' : runMode === 'engine' ? 'GGUF · inference check' : undefined : activeStage === 2 && distillationModels[workflowProjectId] === 'act' ? 'ACT → ACT256' : undefined}
          onNavigate={navigateStage} onTrain={startTrainingOnDataset} onRefresh={() => void jobs.refetch()} />
        <div className="dataset-view" hidden={activeStage !== 0}>
          <div className="section-tabs"><nav className="dataset-tab-buttons" aria-label="Dataset views"><button type="button" className={`section-tab${datasetView === 'sources' ? ' active' : ''}`} aria-pressed={datasetView === 'sources'} onClick={() => setDatasetView('sources')}>Sources</button><button type="button" className={`section-tab${datasetView === 'inspection' ? ' active' : ''}`} aria-pressed={datasetView === 'inspection'} disabled={!selectedJob} onClick={() => setDatasetView('inspection')}>Inspection{sortedJobs.length > 0 && <span className="tab-count">{sortedJobs.length}</span>}</button></nav><span className="section-note">LeRobot v2 / v3</span></div>
          <div className="content-grid source-grid" hidden={datasetView !== 'sources'}>
            <div className="intake-column"><IntakeForm key={`${projectId}-${starterSelection}`} navigationToken={`${workflowNavigation}:${activeStage}:${datasetView}:${selectedJobId}`} project={workflowProjectId ? project : undefined} readinessMessage={projects.isPending ? 'Loading projects before importing a dataset.' : projects.isError ? 'Project list unavailable. Retry projects to continue.' : 'Create or select a project to import a dataset.'} starter={starter} onSourceEdited={() => setActiveStarterId('')} localAvailable={capabilities.data?.some(item => item.operation === 'dataset.inspect.local' && (item.status === 'available' || item.status === 'untested')) ?? false} onCreated={job => { setSelectedJobId(job.id); setDatasetView('inspection'); }} /></div>
            <DatasetStarters selected={activeStarterId} onSelect={item => { setStarter(item); setActiveStarterId(item.id); setStarterSelection(previous => previous + 1); }} />
          </div>
          <div className="inspection-view" hidden={datasetView !== 'inspection'}>
            <section className="inspection-record" aria-labelledby="activity-title">
              <div className="activity-heading"><div><h2 id="activity-title">Dataset inspection</h2></div><button type="button" className="secondary-button" onClick={() => setDatasetView('sources')}>Change source</button></div>
              <ErrorNotice error={jobs.error} />
              {jobs.isPending && projectId && <p className="loading-note" role="status">Loading inspections…</p>}
              {selectedJobId && !selectedJob && !jobs.isPending && <p className="warning-box" role="alert">Selected inspection {selectedJobId} is unavailable in this project. Another dataset has not been substituted. Refresh or choose a source explicitly.</p>}
              {selectedJob && <>
                {sortedJobs.length > 1 && <div className="history-control"><label htmlFor="inspection-history">History</label><select id="inspection-history" value={selectedJob.id} onChange={event => setSelectedJobId(event.target.value)}>{sortedJobs.map(job => <option key={job.id} value={job.id}>{job.request.repo_id || 'Local dataset'} · {displayDate(job.created_at)} · {job.status}</option>)}</select></div>}
                <JobDetail key={selectedJob.id} job={selectedJob} projectId={projectId} />
                {selectedJob.result && <DatasetExplorer key={`explorer-${selectedJob.id}`} job={selectedJob} active={activeStage === 0 && datasetView === 'inspection'} />}
                {selectedJob.status === 'succeeded' && (selectedJob.result?.source === 'huggingface' || selectedJob.result?.snapshot) && <div className="dataset-train-action">{isAugmentationSource(selectedJob) && <button className="secondary-button" onClick={() => navigateStage(8)}><Icon name="spark" size={16} />Augment this dataset</button>}<button className="primary-button" onClick={() => startTrainingOnDataset(selectedJob.id)}>Train on this dataset <Icon name="arrow" size={16} /></button></div>}
              </>}
            </section>
          </div>
          {jobs.error && datasetView === 'sources' && <ErrorNotice error={jobs.error} />}
        </div>
        {activeStage === 8 && <AugmentationPanel key={projectId} projectId={workflowProjectId} preferredDatasetId={selectedJob?.id} onOpenSettings={() => { setSettingsTab('compute'); navigateStage(6); }} onChooseDataset={() => { navigateStage(0); setDatasetView('sources'); }} />}
        {activeStage === 9 && <>
          <section className="panel" aria-label="Teaching connection"><h2>Choose a teaching connection</h2><div className="workbench-actions" role="group" aria-label="Teaching mode">
            <button type="button" className={teachingMode === 'manual' ? 'primary-button' : 'secondary-button'} aria-pressed={teachingMode === 'manual'} onClick={() => { setTeachingCapture(null); setTeachingMode('manual'); }}>Existing executor</button>
            <button type="button" className={teachingMode === 'managed' ? 'primary-button' : 'secondary-button'} aria-pressed={teachingMode === 'managed'} disabled={!workflowProjectId} onClick={() => { setTeachingCapture(null); setTeachingMode('managed'); }}>Managed session</button>
          </div><p>Use an existing teaching connection, or explicitly start a configured session for this project. Switching views does not stop a managed session.</p></section>
          {teachingMode === 'managed' ? <ManagedTeachingPanel key={workflowProjectId} projectId={workflowProjectId} onPublishedCapture={reviewTeachingCapture} /> : <TeachingPanel key={`manual:${workflowProjectId}`} />}
          <RecordingPreparationPanel key={workflowProjectId} projectId={workflowProjectId} teachingCapture={teachingCapture?.project_id === workflowProjectId ? teachingCapture : undefined} onTeachingCaptureConsumed={() => setTeachingCapture(null)} onInspect={job => { navigateStage(0); setSelectedJobId(job.id); setDatasetView('inspection'); }} onTrain={job => startTrainingOnDataset(job.id)} />
        </>}
        {activeStage === 10 && <DecisionPanel />}
        {(activeStage === 1 || activeStage === 6) && <div className="training-view" hidden={activeStage !== 1}><TrainingPanel active={activeStage === 1} key={projectId} projectId={workflowProjectId} startNew={startTraining} showJobsRequest={trainingNavigation} preferredRunId={openTrainingRun?.projectId === projectId ? openTrainingRun.id : undefined} preferredDatasetId={selectedJob?.id} onChooseDataset={() => { setActiveStage(0); setDatasetView('sources'); }} onDiagnostics={() => { setSettingsTab('diagnostics'); setActiveStage(6); }} onComputeSettings={() => { setSettingsTab('compute'); setActiveStage(6); }} onQuantize={artifactId => { chooseQuantize('gguf', 'handoff'); setQuantizeArtifact({ projectId, artifactId }); setWorkflowNavigation(value => value + 1); setActiveStage(3); }} onNativeQuantize={artifactId => { navigateStage(3); setQuantizeArtifact({ projectId: workflowProjectId, artifactId }); chooseQuantize('native', 'handoff'); }} onNativeDistill={artifactId => { if (!workflowProjectId) return; navigateStage(2); setDistillTeacher({ projectId: workflowProjectId, artifactId }); setDistillationModels(previous => ({ ...previous, [workflowProjectId]: 'act' })); }} /></div>}
        {activeStage === 7 && <CloudRuns key={workflowProjectId} projectId={workflowProjectId} onOpenSimulation={id => { setOpenSimulation({ projectId: workflowProjectId, id }); chooseRun('native', 'handoff', id); setActiveStage(5); }} onOpenTraining={id => { setStartTraining(undefined); setOpenTrainingRun({ projectId: workflowProjectId, id }); setActiveStage(1); }} />}
        {activeStage === 3 && <section className="workflow-choices" aria-label="Quantization workflow">
          <h2 className="workflow-choice-title">Supported models</h2>
          <div className="mode-card-grid two-columns" role="group" aria-label="Quantization mode">
            <ModeCard title="SmolVLA" detail="GGUF · 4-bit or 8-bit" selected={quantizeMode === 'gguf'} disabled={!workflowProjectId} onClick={() => { if (quantizeMode !== 'gguf') setQuantizeArtifact(null); chooseQuantize('gguf'); }} />
            <ModeCard title="ACT" detail="INT8 or INT4 · Local worker" selected={quantizeMode === 'native'} disabled={!workflowProjectId} onClick={() => { if (quantizeMode !== 'native') setQuantizeArtifact(null); chooseQuantize('native'); }} />
          </div>
        </section>}
        {entryPending && (!workflowProjectId || jobs.isPending || entryFailed || recoveryError) && <section className="panel" aria-label="Workflow selection status"><p role={entryFailed || recoveryError ? 'alert' : 'status'}>{!workflowProjectId ? 'Select a project to see its workflow history.' : recoveryError ?? (entryFailed ? 'Workflow availability or history could not be loaded. Choose a mode to inspect it, or retry these reads.' : 'Loading this project’s workflow history and configured workers…')}</p>{entryFailed && <button className="secondary-button" onClick={() => { void jobs.refetch(); void options.refetch(); if (activeStage === 5) void simulation.refetch(); }}>Retry workflow context</button>}</section>}
        {activeStage === 3 && quantizeMode === 'native' && <NativeQuantizationPanel key={workflowProjectId} projectId={workflowProjectId} preferredJobId={context?.quantize?.jobId} onJobSelected={id => chooseQuantize('native', 'manual', id)} preferredArtifactId={quantizeArtifact?.projectId === workflowProjectId ? quantizeArtifact.artifactId : undefined} onPrepare={() => navigateStage(1)} onReplay={artifactId => { navigateStage(5); setReplayArtifact({ projectId: workflowProjectId, artifactId }); chooseRun('replay', 'handoff'); }} onPrepareSimulation={source => { if (source.projectId !== workflowProjectId) return; navigateStage(5); setSimulationArtifact(source); chooseRun('native', 'handoff'); }} />}
        {activeStage === 4 && <section className="workflow-context" aria-label="Evaluation purpose"><WorkbenchDisclosure title="Evaluation details"><p>Engine checks measure loading, finite actions and runtime performance. LIBERO measures closed-loop task success with a configured benchmark runtime. Observation replay and experimental Isaac rollouts do not establish task success; scored ACT / Isaac evaluation is not configured.</p></WorkbenchDisclosure>
          {options.isPending ? <p role="status">Checking evaluation targets…</p> : options.isError ? <p role="alert">Evaluation targets could not be loaded. Availability is unknown.</p> : <p role="status">{engineConfigured ? 'Engine evaluation is configured.' : 'No engine evaluation target is configured.'} {options.data?.runtimes.some(item => engineRuntime(item, 'Evaluate') && item.simulation) ? 'A LIBERO target is configured; its policy and protocol still require validation.' : 'No LIBERO evaluation target is configured.'}</p>}
          {simulation.isError && <p role="alert">Isaac profile availability could not be loaded.</p>}
          {jobs.isError && <p role="alert">Saved workflow history could not be refreshed; previously received records may be stale.</p>}
          <div className="native-result-actions">
            {(replayConfigured || replayHistory) && <button className="text-link" onClick={() => { navigateStage(5); chooseRun('replay', 'handoff'); }}>Open observation replay</button>}
            {(simulationConfigured || simulationHistory) && <button className="text-link" onClick={() => { navigateStage(5); chooseRun('native', 'handoff'); }}>Open native Isaac Run</button>}
          </div>
        </section>}
        {activeStage === 5 && <section className="workflow-choices" aria-label="Run workflow">
          <h2 className="workflow-choice-title">Workflows</h2>
          <div className="mode-card-grid" role="group" aria-label="Run mode">
            <ModeCard title="3D simulation" detail="Isaac Sim · ACT or SmolVLA" icon="play" selected={runMode === 'native'} disabled={!workflowProjectId} onClick={() => { setSimulationArtifact(null); if (runMode !== 'native') { setOpenSimulation(null); setReplayArtifact(null); } chooseRun('native'); }} />
            <ModeCard title="Replay observations" detail="Offline replay · ACT" icon="database" selected={runMode === 'replay'} disabled={!workflowProjectId} onClick={() => { setSimulationArtifact(null); if (runMode !== 'replay') { setOpenSimulation(null); setReplayArtifact(null); } chooseRun('replay'); }} />
            <ModeCard title="Check inference" detail="Inference engine · GGUF" icon="chart" selected={runMode === 'engine'} disabled={!workflowProjectId} onClick={() => { setSimulationArtifact(null); if (runMode !== 'engine') { setOpenSimulation(null); setReplayArtifact(null); } chooseRun('engine'); }} />
          </div>
        </section>}
        {activeStage === 5 && runMode === 'replay' && <NativeReplayPanel key={workflowProjectId} projectId={workflowProjectId} preferredJobId={context?.run?.mode === 'replay' ? context.run.jobId : undefined} onJobSelected={id => chooseRun('replay', 'manual', id)} preferredArtifactId={replayArtifact?.projectId === workflowProjectId ? replayArtifact.artifactId : undefined} onDataset={() => { setDatasetView('sources'); navigateStage(0); }} />}
        {activeStage === 5 && runMode === 'native' && <NativeSimulationPanel key={workflowProjectId} projectId={workflowProjectId} onJobSelected={id => { setOpenSimulation(null); setSimulationArtifact(null); chooseRun('native', 'manual', id); }} preferredArtifact={simulationArtifact?.projectId === workflowProjectId ? simulationArtifact : undefined} preferredJobId={openSimulation?.projectId === workflowProjectId ? openSimulation.id : context?.run?.mode === 'native' ? context.run.jobId : undefined} onTraining={() => navigateStage(1)} />}
        {[3, 4, 5, 6].includes(activeStage) && (activeStage !== 3 || quantizeMode === 'gguf') && (activeStage !== 5 || runMode === 'engine') && <WorkflowPanel key={`${workflowProjectId}-${activeStage}-${workflowNavigation}`} projectId={workflowProjectId} tab={settingsTab} onTabChange={setSettingsTab} onOpenQuantize={() => { navigateStage(3); chooseQuantize('gguf', 'handoff'); }} stage={activeStage === 6 ? 'settings' : stage.name} preferredArtifactId={activeStage === 3 && quantizeArtifact?.projectId === projectId ? quantizeArtifact.artifactId : undefined} onViewTraining={() => navigateStage(1)} />}
        {activeStage === 2 && <DistillationPanel key={workflowProjectId} projectId={workflowProjectId} model={distillationModels[workflowProjectId]} preferredTeacherArtifactId={distillTeacher?.projectId === workflowProjectId ? distillTeacher.artifactId : undefined} onSelectModel={model => { setDistillTeacher(null); if (workflowProjectId) setDistillationModels(previous => ({ ...previous, [workflowProjectId]: model })); }} onDataset={() => { setDatasetView('sources'); navigateStage(0); }} onQuantize={artifactId => { navigateStage(3); setQuantizeArtifact({ projectId, artifactId }); chooseQuantize('native', 'handoff'); }} />}
  </WorkspaceShell>;
}

export function Workspace() {
  const [queryClient] = useState(() => new QueryClient({ defaultOptions: { queries: { refetchOnWindowFocus: true, staleTime: 5_000 } } }));
  return <QueryClientProvider client={queryClient}><Workbench /></QueryClientProvider>;
}
