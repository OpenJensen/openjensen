'use client';

import { QueryClient, QueryClientProvider, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useState, type FormEvent } from 'react';
import { api, isActive, type DatasetProfile, type Job, type Project } from '@/lib/api';
import { WorkflowPanel } from '@/components/workflow-panel';
import { Icon } from '@/components/icon';
import { WorkspaceShell } from '@/components/workspace-shell';

const candidate = 'codywang/so101_pickup_test';
const candidateRevision = 'ecef85bc07005f771ad86deeff1427f9d72953ed';
const stages = [
  { name: 'Dataset', icon: 'database', description: 'Bring in your robotics data. Understand it before you train.' },
  { name: 'Fine-tune', icon: 'sliders', description: 'Adapt a base policy to your dataset and your task.' },
  { name: 'Distill', icon: 'layers', description: 'Transfer what a larger policy knows into a smaller model.' },
  { name: 'Quantize', icon: 'compress', description: 'Reduce model size for the hardware you want to run on.' },
  { name: 'Evaluate', icon: 'chart', description: 'Measure policy behavior before taking it to your robot.' },
  { name: 'Run', icon: 'play', description: 'Put a tested policy to work on your target hardware.' },
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
  const features = Object.entries(profile.features);
  return <section className="result" aria-labelledby="result-title">
    <div className="result-heading">
      <div><p className="eyebrow">Dataset profile</p><h3 id="result-title">Inspection complete</h3></div>
      <span className="metadata-badge"><Icon name="check" size={14} /> Metadata only</span>
    </div>
    <p className="result-explainer">This inspection reads metadata. It does not decode videos, validate trajectories, or establish training or simulator compatibility.</p>
    <dl className="metric-grid">
      <div><dt>Episodes</dt><dd>{formatNumber(profile.total_episodes)}</dd></div>
      <div><dt>Frames</dt><dd>{formatNumber(profile.total_frames)}</dd></div>
      <div><dt>Frame rate</dt><dd>{profile.fps}<span> fps</span></dd></div>
      <div><dt>Robot type</dt><dd className="metric-text">{profile.robot_type || 'Not declared'}</dd></div>
    </dl>
    <div className="profile-facts">
      <div><span>Native format</span><strong>{profile.format}</strong></div>
      <div><span>Declared license</span><strong>{profile.license || 'Not declared'}</strong></div>
      <div><span>Source</span><strong>{profile.source === 'huggingface' ? 'Hugging Face' : 'Local directory'}</strong></div>
    </div>
    {profile.warnings.length > 0 && <div className="warning-box">
      <h4>Before you use this dataset</h4>
      <ul>{profile.warnings.map((warning, index) => <li key={`${index}-${warning}`}>{warning}</li>)}</ul>
    </div>}
    <div className="feature-heading"><h4>Observation & action schema</h4><span>{features.length} features</span></div>
    {features.length ? <div className="table-scroll"><table className="feature-table">
      <caption className="visually-hidden">Declared dataset feature metadata</caption>
      <thead><tr><th scope="col">Feature</th><th scope="col">Declared metadata</th></tr></thead>
      <tbody>{features.map(([name, value]) => <tr key={name}><th scope="row">{name}</th><td><code>{JSON.stringify(value)}</code></td></tr>)}</tbody>
    </table></div> : <p className="muted">No feature declarations found.</p>}
    <details className="provenance">
      <summary>Source & provenance</summary>
      <dl>
        <div><dt>Repository</dt><dd>{profile.repo_id || 'Local metadata snapshot'}</dd></div>
        <div><dt>Resolved revision</dt><dd><code>{profile.revision}</code></dd></div>
        <div><dt>Metadata SHA-256</dt><dd><code>{profile.metadata_sha256}</code></dd></div>
        <div><dt>Inspected</dt><dd>{displayDate(profile.inspected_at)}</dd></div>
      </dl>
      {profile.source === 'huggingface' && profile.repo_id && <a className="text-link" href={`https://huggingface.co/datasets/${profile.repo_id.split('/').map(encodeURIComponent).join('/')}/tree/${encodeURIComponent(profile.revision)}`} target="_blank" rel="noreferrer">Open pinned source <Icon name="external" size={13} /></a>}
    </details>
  </section>;
}

function IntakeForm({ project, localAvailable, onCreated }: { project: Project | undefined; localAvailable: boolean; onCreated: (job: Job) => void }) {
  const queryClient = useQueryClient();
  const [source, setSource] = useState<'huggingface' | 'local'>('huggingface');
  const [repoId, setRepoId] = useState(candidate);
  const [revision, setRevision] = useState(candidateRevision);
  const [path, setPath] = useState('');
  useEffect(() => {
    if (!localAvailable && source === 'local') setSource('huggingface');
  }, [localAvailable, source]);
  const mutation = useMutation({
    mutationFn: () => {
      if (!project) throw new Error('Create or select a project first.');
      if (source === 'local' && !localAvailable) throw new Error('Local intake is not enabled on this application.');
      return api.inspect(project.id, source === 'huggingface'
        ? { source, repo_id: repoId.trim(), revision: revision.trim() }
        : { source, path: path.trim(), revision: 'main' });
    },
    onSuccess: (job) => {
      void queryClient.invalidateQueries({ queryKey: ['jobs', job.project_id] });
      onCreated(job);
    },
  });
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    mutation.mutate();
  }
  return <section className="panel intake-panel" aria-labelledby="intake-title">
    <div className="panel-title"><h2 id="intake-title">Import a dataset</h2><p>Choose a source to inspect its metadata.</p></div>
    <form onSubmit={submit}>
      <fieldset className="source-options">
        <legend className="visually-hidden">Dataset source</legend>
        <label className={source === 'huggingface' ? 'source-option selected' : 'source-option'}><input type="radio" name="source" value="huggingface" checked={source === 'huggingface'} onChange={() => { setSource('huggingface'); mutation.reset(); }} /><span>Hugging Face</span><span className="source-detail">Public dataset</span></label>
        <label className={`source-option${source === 'local' ? ' selected' : ''}${!localAvailable ? ' unavailable' : ''}`}><input type="radio" name="source" value="local" checked={source === 'local'} disabled={!localAvailable} aria-describedby={!localAvailable ? 'local-source-help' : undefined} onChange={() => { setSource('local'); mutation.reset(); }} /><span>Local directory</span><span className="source-detail">{localAvailable ? 'On the API host' : 'Not enabled'}</span></label>
      </fieldset>
      {!localAvailable && <p id="local-source-help" className="field-help local-source-help">Local sources are not configured for this workspace.</p>}
      {source === 'huggingface' ? <>
        <label className="field-label" htmlFor="repo-id">Dataset repository</label>
        <input id="repo-id" name="repo_id" value={repoId} onChange={event => {
          setRepoId(event.target.value);
          if (revision === candidateRevision) setRevision('main');
        }} required placeholder="owner/dataset-name" autoCapitalize="none" autoCorrect="off" spellCheck={false} aria-describedby="repo-help" />
        <p id="repo-help" className="field-help">A public LeRobot v2 or v3 dataset.</p>
        <label className="field-label" htmlFor="revision">Revision</label>
        <input id="revision" className="mono-input" name="revision" value={revision} onChange={event => setRevision(event.target.value)} required autoCapitalize="none" autoCorrect="off" spellCheck={false} aria-describedby="revision-help" />
        <p id="revision-help" className="field-help">A branch, tag, or commit. The resolved revision is saved with the result.</p>
      </> : <>
        <label className="field-label" htmlFor="local-path">Dataset directory</label>
        <input id="local-path" name="path" value={path} onChange={event => setPath(event.target.value)} required placeholder="Path to a LeRobot dataset" autoCapitalize="none" autoCorrect="off" spellCheck={false} aria-describedby="path-help" />
        <p id="path-help" className="field-help">The directory must exist on the computer running the API, inside its configured allowed root. A browser file upload is not used.</p>
      </>}
      <ErrorNotice error={mutation.error} />
      {!project && <p className="form-note">Create a project in the sidebar to begin.</p>}
      <button className="primary-button inspect-button" type="submit" disabled={!project || mutation.isPending}>{mutation.isPending ? 'Starting inspection…' : 'Inspect dataset'}<Icon name="arrow" size={17} /></button>
      <p className="privacy-note"><Icon name="check" size={13} /> Metadata only. No videos or model weights downloaded.</p>
    </form>
  </section>;
}

function JobDetail({ job, projectId }: { job: Job; projectId: string }) {
  const queryClient = useQueryClient();
  const cancel = useMutation({
    mutationFn: () => api.cancel(job.id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['jobs', projectId] }),
  });
  return <>
    <div className="job-detail-heading"><div><span className="eyebrow">INSPECTION</span><p className="job-id" title={job.id}>{job.id}</p></div><JobStatus status={job.status} /></div>
    {isActive(job) && <div className="working-state" role="status"><span className="spinner" /><div><h3>{job.status === 'queued' ? 'Waiting to inspect' : 'Reading source metadata'}</h3><p>The result will appear here automatically.</p></div><button className="secondary-button" onClick={() => cancel.mutate()} disabled={cancel.isPending}>{cancel.isPending ? 'Cancelling…' : 'Cancel'}</button></div>}
    <ErrorNotice error={cancel.error} />
    {job.error && <p className="error-notice" role="alert">{job.error}</p>}
    {(job.status === 'cancelled' || job.status === 'interrupted') && <p className="muted">This inspection did not complete. Start another inspection when you are ready.</p>}
    {job.result && 'inspection_scope' in job.result && <DatasetResult profile={job.result} />}
  </>;
}

function Workbench() {
  const queryClient = useQueryClient();
  const [projectId, setProjectId] = useState('');
  const [projectName, setProjectName] = useState('');
  const [selectedJobId, setSelectedJobId] = useState('');
  const [activeStage, setActiveStage] = useState(0);
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
    if (!projects.data || projectId) return;
    let stored = '';
    try { stored = localStorage.getItem('firebird.project') ?? ''; } catch { /* Storage may be unavailable. */ }
    const selected = projects.data.find(project => project.id === stored) ?? projects.data[0];
    if (selected) setProjectId(selected.id);
  }, [projects.data, projectId]);
  function selectProject(id: string) {
    setProjectId(id);
    setSelectedJobId('');
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
  const sortedJobs = [...(jobs.data ?? [])].filter(job => job.kind === 'dataset.inspect').sort((a, b) => b.created_at.localeCompare(a.created_at));
  const selectedJob = sortedJobs.find(job => job.id === selectedJobId) ?? sortedJobs[0];
  const connected = health.isSuccess && !health.isError;

  const stage = stages[activeStage] ?? { name: 'Settings & diagnostics', icon: 'sliders' as const, description: 'Manage workflow preferences and inspect recorded evidence.' };

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
        <button className={`stage-button${activeStage === 6 ? ' selected' : ''}`} onClick={() => setActiveStage(6)}><Icon name="sliders" size={19} /><span>Settings & diagnostics</span></button>
      </nav>
      <section className="projects-section" aria-labelledby="projects-heading">
        <div className="sidebar-section-label"><h2 id="projects-heading">Project</h2><span>{projects.data?.length ?? '—'}</span></div>
        {projects.isPending && <p className="sidebar-note" role="status">Loading projects…</p>}
        <ErrorNotice error={projects.error} />
        {projects.isError && <button className="text-button" onClick={() => void projects.refetch()} disabled={projects.isFetching}>Retry projects</button>}
        {!!projects.data?.length && <div className="project-picker"><Icon name="folder" size={16} /><label className="visually-hidden" htmlFor="project-select">Current project</label><select id="project-select" value={projectId} onChange={event => selectProject(event.target.value)}>{projects.data.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></div>}
        {projects.data?.length === 0 && <p className="sidebar-note">Create a project to keep your datasets and inspections together.</p>}
        <form className="project-form" onSubmit={event => { event.preventDefault(); if (projectName.trim()) projectMutation.mutate(); }}>
          <label htmlFor="project-name">New project</label>
          <div className="project-input-row"><input id="project-name" name="name" value={projectName} onChange={event => setProjectName(event.target.value)} required maxLength={100} placeholder="Project name" /><button type="submit" aria-label="Create project" title="Create project" disabled={!projectName.trim() || projectMutation.isPending}><Icon name="plus" size={17} /></button></div>
          {projectMutation.isPending && <p className="sidebar-note" role="status">Creating project…</p>}
          <ErrorNotice error={projectMutation.error} />
        </form>
      </section>
    </>}
  >
        <div className="page-heading"><div><p className="eyebrow">Step {String(activeStage + 1).padStart(2, '0')}</p><h1>{stage.name}</h1><p>{stage.description}</p></div><div className={`connection ${connected ? 'connected' : ''}`} role="status"><span />{health.isPending ? 'Connecting' : connected ? 'Connected' : 'Offline'}</div></div>
        {!connected && !health.isPending && <div className="connection-notice"><ErrorNotice error={health.error} /><button className="text-button" onClick={() => { void health.refetch(); void projects.refetch(); void capabilities.refetch(); }}>Retry connection</button></div>}
        {capabilities.error && connected && <div className="connection-notice"><ErrorNotice error={capabilities.error} /><button className="text-button" onClick={() => void capabilities.refetch()} disabled={capabilities.isFetching}>Retry capabilities</button></div>}
        <div className="dataset-view" hidden={activeStage !== 0}>
          <div className="section-tabs"><span className="section-tab active">Dataset intake</span><span className="section-note">LeRobot v2 / v3</span></div>
          <div className="content-grid">
            <div className="intake-column"><IntakeForm key={projectId} project={project} localAvailable={capabilities.data?.some(item => item.operation === 'dataset.inspect.local' && (item.status === 'available' || item.status === 'untested')) ?? false} onCreated={job => setSelectedJobId(job.id)} /><div className="source-note"><Icon name="database" size={16} /><p>Start with the included SO-101 example, or enter your own dataset repository.</p></div></div>
            <section className="panel activity-panel" aria-labelledby="activity-title">
              <div className="activity-heading"><div><h2 id="activity-title">Inspection</h2><p>{project ? 'Metadata and history for this project.' : 'Your dataset profile will appear here.'}</p></div><span className="quiet-icon"><Icon name="clock" size={18} /></span></div>
              <ErrorNotice error={jobs.error} />
              {jobs.isPending && projectId && <p className="loading-note" role="status">Loading inspections…</p>}
              {selectedJob ? <>
                {sortedJobs.length > 1 && <div className="history-control"><label htmlFor="inspection-history">History</label><select id="inspection-history" value={selectedJob.id} onChange={event => setSelectedJobId(event.target.value)}>{sortedJobs.map(job => <option key={job.id} value={job.id}>{displayDate(job.created_at)} · {job.status}</option>)}</select></div>}
                <JobDetail key={selectedJob.id} job={selectedJob} projectId={projectId} />
              </> : !jobs.isFetching && <div className="empty-state"><span className="empty-icon"><Icon name="database" size={25} /></span><h3>No dataset inspected yet</h3><p>{project ? 'Choose a source and run an inspection to explore its structure, features, and provenance.' : 'Create a project in the sidebar, then inspect a dataset to explore what is inside.'}</p><div className="empty-metrics" aria-hidden="true"><div><span>Episodes</span><strong>—</strong></div><div><span>Frames</span><strong>—</strong></div><div><span>Features</span><strong>—</strong></div></div><div className="empty-caption"><Icon name="check" size={14} /> Source metadata, without the full download.</div></div>}
            </section>
          </div>
          <footer className="workspace-footer"><span>Inspect first. Build on what you know.</span><span>Metadata intake</span></footer>
        </div>
        {activeStage > 0 && activeStage !== 2 && <WorkflowPanel key={`${projectId}-${activeStage}`} projectId={projectId} stage={activeStage === 6 ? 'settings' : stage.name} />}
        {activeStage === 2 && <section className="planned-panel" aria-labelledby="planned-title"><span className="empty-icon"><Icon name={stage.icon} size={28} /></span><span className="planned-badge">Planned</span><h2 id="planned-title">{stage.name} is on the roadmap</h2><p>{capabilities.data?.find(item => item.stage.toLowerCase() === stage.name.toLowerCase())?.description ?? 'This stage is not available in the current application.'}</p><p className="planned-note">You can start by inspecting your dataset. Your project and inspection history will be here when this stage is ready.</p><button className="secondary-button" onClick={() => setActiveStage(0)}>Go to Dataset <Icon name="arrow" size={15} /></button></section>}
  </WorkspaceShell>;
}

export function Workspace() {
  const [queryClient] = useState(() => new QueryClient({ defaultOptions: { queries: { refetchOnWindowFocus: true, staleTime: 5_000 } } }));
  return <QueryClientProvider client={queryClient}><Workbench /></QueryClientProvider>;
}
