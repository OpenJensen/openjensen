'use client';

import { QueryClient, QueryClientProvider, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useState, type FormEvent } from 'react';
import { api, isActive, isDatasetJob, type DatasetJob, type DatasetProfile, type Job, type Project } from '@/lib/api';
import { CloudRuns } from '@/components/cloud-runs';
import { WorkflowPanel } from '@/components/workflow-panel';
import { TrainingPanel } from '@/components/training-panel';
import { AugmentationPanel } from '@/components/augmentation-panel';
import { Icon } from '@/components/icon';
import { WorkspaceShell } from '@/components/workspace-shell';
import { DatasetExplorer } from '@/components/dataset-explorer';
import { DatasetStarters } from '@/components/dataset-starters';
import { datasetStarters, type DatasetStarter } from '@/lib/dataset-starters';

const stages = [
  { name: 'Dataset', icon: 'database' },
  { name: 'Fine-tune', icon: 'sliders' },
  { name: 'Distill', icon: 'layers' },
  { name: 'Quantize', icon: 'compress' },
  { name: 'Evaluate', icon: 'chart' },
  { name: 'Run', icon: 'play' },
] as const;

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

function IntakeForm({ project, localAvailable, onCreated, starter, onSourceEdited }: { project: Project | undefined; localAvailable: boolean; onCreated: (job: Job) => void; starter: DatasetStarter; onSourceEdited: () => void }) {
  const queryClient = useQueryClient();
  const [source, setSource] = useState<'huggingface' | 'local'>('huggingface');
  const [repoId, setRepoId] = useState(starter.repoId);
  const [revision, setRevision] = useState(starter.revision);
  const [path, setPath] = useState('');
  useEffect(() => {
    if (!localAvailable && source === 'local') setSource('huggingface');
  }, [localAvailable, source]);
  const mutation = useMutation({
    mutationFn: () => {
      if (!project) throw new Error('Create or select a project first.');
      if (source === 'local' && !localAvailable) throw new Error('Local intake is not enabled on this application.');
      return api.inspect(project.id, source === 'huggingface'
        ? { source, repo_id: repoId.trim(), revision: revision.trim() || 'main' }
        : { source, path: path.trim(), revision: 'main' });
    },
    onSuccess: (job) => {
      queryClient.setQueryData<Job[]>(['jobs', job.project_id], previous => [job, ...(previous ?? []).filter(item => item.id !== job.id)]);
      void queryClient.invalidateQueries({ queryKey: ['jobs', job.project_id] });
      onCreated(job);
    },
  });
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    mutation.mutate();
  }
  return <section className="panel intake-panel" aria-labelledby="intake-title">
    <div className="panel-title"><h2 id="intake-title">Import a dataset</h2></div>
    <form onSubmit={submit}>
      <fieldset className="source-options">
        <legend className="visually-hidden">Dataset source</legend>
        <label className={source === 'huggingface' ? 'source-option selected' : 'source-option'}><input type="radio" name="source" value="huggingface" checked={source === 'huggingface'} onChange={() => { setSource('huggingface'); mutation.reset(); onSourceEdited(); }} /><span>Hugging Face</span></label>
        <label className={`source-option${source === 'local' ? ' selected' : ''}${!localAvailable ? ' unavailable' : ''}`}><input type="radio" name="source" value="local" checked={source === 'local'} disabled={!localAvailable} onChange={() => { setSource('local'); mutation.reset(); onSourceEdited(); }} /><span>Local directory</span>{!localAvailable && <span className="source-detail">Not enabled</span>}</label>
      </fieldset>
      {source === 'huggingface' ? <>
        <label className="field-label" htmlFor="repo-id">Dataset repository</label>
        <input id="repo-id" name="repo_id" value={repoId} onChange={event => {
          setRepoId(event.target.value);
          onSourceEdited();
          if (revision === starter.revision) setRevision('');
        }} required placeholder="owner/dataset-name" autoCapitalize="none" autoCorrect="off" spellCheck={false} />
        <details className="intake-advanced"><summary>Revision (optional)</summary><label className="field-label visually-hidden" htmlFor="revision">Revision</label>
        <input id="revision" className="mono-input" name="revision" value={revision} onChange={event => { setRevision(event.target.value); onSourceEdited(); }} placeholder="Latest (main)" aria-describedby="revision-help" autoCapitalize="none" autoCorrect="off" spellCheck={false} />
        <p id="revision-help" className="field-help">Leave blank for the latest revision, or enter a branch, tag, or commit.</p>
        </details>
      </> : <>
        <label className="field-label" htmlFor="local-path">Dataset directory</label>
        <input id="local-path" name="path" value={path} onChange={event => setPath(event.target.value)} required placeholder="Path to a LeRobot dataset" autoCapitalize="none" autoCorrect="off" spellCheck={false} aria-describedby="path-help" />
        <p id="path-help" className="field-help">Path on the API host, inside the configured dataset directory.</p>
      </>}
      <ErrorNotice error={mutation.error} />
      {!project && <p className="form-note">Create a project in the sidebar to begin.</p>}
      <p className="field-help">Saved inspections are reused when the dataset revision is unchanged.</p>
      <button className="primary-button inspect-button" type="submit" disabled={!project || mutation.isPending}>{mutation.isPending ? 'Checking dataset…' : 'Inspect dataset'}<Icon name="arrow" size={17} /></button>
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
  const [quantizeArtifact, setQuantizeArtifact] = useState<{ projectId: string; artifactId: string } | null>(null);
  const [settingsTab, setSettingsTab] = useState<'compute' | 'settings' | 'diagnostics'>('compute');
  const [datasetView, setDatasetView] = useState<'sources' | 'inspection' | 'augmentation'>('sources');
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
  function selectProject(id: string) {
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
  const sortedJobs = [...(jobs.data ?? [])].filter(isDatasetJob).sort((a, b) => b.created_at.localeCompare(a.created_at));
  const selectedJob = sortedJobs.find(job => job.id === selectedJobId) ?? sortedJobs[0];
  const connected = health.isSuccess && !health.isError;

  const stage = activeStage === 7 ? { name: 'Cloud runs', icon: 'clock' as const } : stages[activeStage] ?? { name: 'Settings & diagnostics', icon: 'sliders' as const };

  return <WorkspaceShell
    breadcrumb={<><Icon name={stage.icon} size={18} /><strong>{stage.name}</strong><span className="breadcrumb-divider">/</span><span className="breadcrumb-project">{project?.name ?? 'No project selected'}</span></>}
    navigation={<>
      <nav className="stage-navigation" aria-label="Policy lifecycle">
        <p className="sidebar-section-label">Workspace</p>
        <ul className="stage-list">{stages.map((item, index) => <li key={item.name}>
          <button type="button" className={`stage-button${activeStage === index ? ' selected' : ''}`} onClick={() => setActiveStage(index)} aria-current={activeStage === index ? 'page' : undefined}>
            <Icon name={item.icon} size={19} /><span>{item.name}</span>{index === 2 && <small>Planned</small>}
          </button>
        </li>)}</ul>
        <button className={`stage-button${activeStage === 6 ? ' selected' : ''}`} onClick={() => { setSettingsTab('compute'); setActiveStage(6); }}><Icon name="sliders" size={19} /><span>Settings & diagnostics</span></button>
        <button className={`stage-button${activeStage === 7 ? ' selected' : ''}`} onClick={() => setActiveStage(7)} aria-current={activeStage === 7 ? 'page' : undefined}><Icon name="clock" size={19} /><span>Cloud runs</span></button>
      </nav>
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
    </>}
  >
        <div className={`page-heading${activeStage === 1 ? ' training-page-heading' : ''}`}><div><h1>{stage.name}</h1></div><div className={`connection ${connected ? 'connected' : ''}`} role="status"><span />{health.isPending ? 'Connecting' : connected ? 'Connected' : 'Offline'}</div></div>
        {!connected && !health.isPending && <div className="connection-notice"><ErrorNotice error={health.error} /><button className="text-button" onClick={() => { void health.refetch(); void projects.refetch(); void capabilities.refetch(); }}>Retry connection</button></div>}
        {capabilities.error && connected && <div className="connection-notice"><ErrorNotice error={capabilities.error} /><button className="text-button" onClick={() => void capabilities.refetch()} disabled={capabilities.isFetching}>Retry capabilities</button></div>}
        <div className="dataset-view" hidden={activeStage !== 0}>
          <div className="section-tabs"><nav className="dataset-tab-buttons" aria-label="Dataset views"><button type="button" className={`section-tab${datasetView === 'sources' ? ' active' : ''}`} aria-pressed={datasetView === 'sources'} onClick={() => setDatasetView('sources')}>Sources</button><button type="button" className={`section-tab${datasetView === 'inspection' ? ' active' : ''}`} aria-pressed={datasetView === 'inspection'} disabled={!selectedJob} onClick={() => setDatasetView('inspection')}>Inspection{sortedJobs.length > 0 && <span className="tab-count">{sortedJobs.length}</span>}</button><button type="button" className={`section-tab${datasetView === 'augmentation' ? ' active' : ''}`} aria-pressed={datasetView === 'augmentation'} onClick={() => setDatasetView('augmentation')}>Augmentation</button></nav><span className="section-note">LeRobot v2 / v3</span></div>
          <div className="content-grid source-grid" hidden={datasetView !== 'sources'}>
            <div className="intake-column"><IntakeForm key={`${projectId}-${starterSelection}`} project={project} starter={starter} onSourceEdited={() => setActiveStarterId('')} localAvailable={capabilities.data?.some(item => item.operation === 'dataset.inspect.local' && (item.status === 'available' || item.status === 'untested')) ?? false} onCreated={job => { setSelectedJobId(job.id); setDatasetView('inspection'); }} /></div>
            <DatasetStarters selected={activeStarterId} onSelect={item => { setStarter(item); setActiveStarterId(item.id); setStarterSelection(previous => previous + 1); }} />
          </div>
          <div className="inspection-view" hidden={datasetView !== 'inspection'}>
            <section className="inspection-record" aria-labelledby="activity-title">
              <div className="activity-heading"><div><h2 id="activity-title">Dataset inspection</h2></div><button type="button" className="secondary-button" onClick={() => setDatasetView('sources')}>Change source</button></div>
              <ErrorNotice error={jobs.error} />
              {jobs.isPending && projectId && <p className="loading-note" role="status">Loading inspections…</p>}
              {selectedJob && <>
                {sortedJobs.length > 1 && <div className="history-control"><label htmlFor="inspection-history">History</label><select id="inspection-history" value={selectedJob.id} onChange={event => setSelectedJobId(event.target.value)}>{sortedJobs.map(job => <option key={job.id} value={job.id}>{job.request.repo_id || 'Local dataset'} · {displayDate(job.created_at)} · {job.status}</option>)}</select></div>}
                <JobDetail key={selectedJob.id} job={selectedJob} projectId={projectId} />
                {selectedJob.result && <DatasetExplorer key={`explorer-${selectedJob.id}`} job={selectedJob} active={activeStage === 0 && datasetView === 'inspection'} />}
                {selectedJob.status === 'succeeded' && selectedJob.result?.source === 'huggingface' && <div className="dataset-train-action"><button className="primary-button" onClick={() => setActiveStage(1)}>Train on this dataset <Icon name="arrow" size={16} /></button></div>}
              </>}
            </section>
          </div>
          {activeStage === 0 && datasetView === 'augmentation' && <AugmentationPanel key={`${projectId}-${selectedJob?.id ?? ''}`} projectId={projectId} preferredDatasetId={selectedJob?.id} onChooseDataset={() => setDatasetView('sources')} />}
          {jobs.error && datasetView === 'sources' && <ErrorNotice error={jobs.error} />}
        </div>
        {(activeStage === 1 || activeStage === 6) && <div className="training-view" hidden={activeStage !== 1}><TrainingPanel active={activeStage === 1} key={projectId} projectId={workflowProjectId} preferredDatasetId={selectedJob?.id} onChooseDataset={() => { setActiveStage(0); setDatasetView('sources'); }} onDiagnostics={() => { setSettingsTab('diagnostics'); setActiveStage(6); }} onComputeSettings={() => { setSettingsTab('compute'); setActiveStage(6); }} onQuantize={artifactId => { setQuantizeArtifact({ projectId, artifactId }); setActiveStage(3); }} /></div>}
        {activeStage === 7 && <CloudRuns />}
        {activeStage > 1 && activeStage !== 2 && activeStage !== 7 && <WorkflowPanel key={`${workflowProjectId}-${activeStage}`} projectId={workflowProjectId} tab={settingsTab} onTabChange={setSettingsTab} onOpenQuantize={() => setActiveStage(3)} stage={activeStage === 6 ? 'settings' : stage.name} preferredArtifactId={activeStage === 3 && quantizeArtifact?.projectId === projectId ? quantizeArtifact.artifactId : undefined} onViewTraining={() => setActiveStage(1)} />}
        {activeStage === 2 && <section className="planned-panel" aria-labelledby="planned-title"><span className="empty-icon"><Icon name={stage.icon} size={28} /></span><span className="planned-badge">Planned</span><h2 id="planned-title">{stage.name}</h2><button className="secondary-button" onClick={() => setActiveStage(0)}>Go to Dataset <Icon name="arrow" size={15} /></button></section>}
  </WorkspaceShell>;
}

export function Workspace() {
  const [queryClient] = useState(() => new QueryClient({ defaultOptions: { queries: { refetchOnWindowFocus: true, staleTime: 5_000 } } }));
  return <QueryClientProvider client={queryClient}><Workbench /></QueryClientProvider>;
}
