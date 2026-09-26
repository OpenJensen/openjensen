'use client';

import { QueryClient, QueryClientProvider, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useState, type FormEvent, type ReactNode } from 'react';
import { api, apiReferenceUrl, isActive, type DatasetProfile, type Job, type Project } from '@/lib/api';
import { ThemeToggle } from '@/components/theme-toggle';
import { DatasetExplorer } from '@/components/dataset-explorer';
import { DatasetStarters } from '@/components/dataset-starters';
import { datasetStarters, type DatasetStarter } from '@/lib/dataset-starters';

const stages = [
  { name: 'Dataset', icon: 'database', description: 'Bring in your robotics data. Understand it before you train.' },
  { name: 'Fine-tune', icon: 'sliders', description: 'Adapt a base policy to your dataset and your task.' },
  { name: 'Distill', icon: 'layers', description: 'Transfer what a larger policy knows into a smaller model.' },
  { name: 'Quantize', icon: 'compress', description: 'Reduce model size for the hardware you want to run on.' },
  { name: 'Evaluate', icon: 'chart', description: 'Measure policy behavior before taking it to your robot.' },
  { name: 'Run', icon: 'play', description: 'Put a tested policy to work on your target hardware.' },
] as const;

function Icon({ name, size = 20 }: { name: 'arrow' | 'plus' | 'database' | 'folder' | 'check' | 'clock' | 'spark' | 'external' | 'sliders' | 'layers' | 'compress' | 'chart' | 'play' | 'book'; size?: number }) {
  const paths: Record<typeof name, ReactNode> = {
    arrow: <><path d="M5 12h14M13 6l6 6-6 6" /></>,
    plus: <><path d="M12 5v14M5 12h14" /></>,
    database: <><ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" /></>,
    folder: <><path d="M3 7V5a1 1 0 0 1 1-1h5l2 3h9a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V7Z" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
    spark: <><path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Z" /></>,
    external: <><path d="M14 3h7v7M21 3l-9 9M10 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-5" /></>,
    sliders: <><path d="M4 7h7m4 0h5M4 17h3m4 0h9" /><circle cx="13" cy="7" r="2" /><circle cx="9" cy="17" r="2" /></>,
    layers: <><path d="m12 3 9 5-9 5-9-5 9-5ZM3 12l9 5 9-5M3 16l9 5 9-5" /></>,
    compress: <><path d="M4 9h5V4M20 9h-5V4M4 15h5v5M20 15h-5v5M9 9 3 3M15 9l6-6M9 15l-6 6M15 15l6 6" /></>,
    chart: <><path d="M4 3v17h17M8 15v-4M13 15V7M18 15V5" /></>,
    play: <path d="m8 4 12 8-12 8V4Z" />,
    book: <><path d="M12 5v16M3 3h5a4 4 0 0 1 4 2 4 4 0 0 1 4-2h5v16h-5a4 4 0 0 0-4 2 4 4 0 0 0-4-2H3V3Z" /></>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
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
    <div className="result-heading">
      <div><p className="eyebrow">Dataset overview</p><h3 id="result-title">{profile.repo_id || 'Local dataset'}</h3></div>
      <span className="metadata-badge"><Icon name="check" size={14} /> {profile.format.replace('_', ' ')}</span>
    </div>
    <p className="result-explainer">Source-declared metadata from the inspected revision. Load the visual preview below to explore episodes and sample data.</p>
    <dl className="metric-grid">
      <div><dt>Episodes</dt><dd>{formatNumber(profile.total_episodes)}</dd></div>
      <div><dt>Frames</dt><dd>{formatNumber(profile.total_frames)}</dd></div>
      <div><dt>Frame rate</dt><dd>{profile.fps}<span> fps</span></dd></div>
      <div><dt>Robot type</dt><dd className="metric-text">{profile.robot_type || 'Not declared'}</dd></div>
    </dl>
    <details className="provenance inspection-provenance">
      <summary>Inspection notes & source provenance</summary>
      <p className="field-help">These notes describe the original metadata-only inspection. Loading a visual preview is a separate operation.</p>
      {profile.warnings.length > 0 && <div className="warning-box"><h4>Original inspection notes</h4><ul>{profile.warnings.map((warning, index) => <li key={`${index}-${warning}`}>{warning}</li>)}</ul></div>}
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
        ? { source, repo_id: repoId.trim(), revision: revision.trim() }
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
    <div className="panel-title"><h2 id="intake-title">Import a dataset</h2><p>Choose a source to inspect its metadata.</p></div>
    <form onSubmit={submit}>
      <fieldset className="source-options">
        <legend className="visually-hidden">Dataset source</legend>
        <label className={source === 'huggingface' ? 'source-option selected' : 'source-option'}><input type="radio" name="source" value="huggingface" checked={source === 'huggingface'} onChange={() => { setSource('huggingface'); mutation.reset(); onSourceEdited(); }} /><span>Hugging Face</span><span className="source-detail">Public dataset</span></label>
        <label className={`source-option${source === 'local' ? ' selected' : ''}${!localAvailable ? ' unavailable' : ''}`}><input type="radio" name="source" value="local" checked={source === 'local'} disabled={!localAvailable} aria-describedby={!localAvailable ? 'local-source-help' : undefined} onChange={() => { setSource('local'); mutation.reset(); onSourceEdited(); }} /><span>Local directory</span><span className="source-detail">{localAvailable ? 'On the API host' : 'Not enabled'}</span></label>
      </fieldset>
      {!localAvailable && <p id="local-source-help" className="field-help local-source-help">Local sources are not configured for this workspace.</p>}
      {source === 'huggingface' ? <>
        <label className="field-label" htmlFor="repo-id">Dataset repository</label>
        <input id="repo-id" name="repo_id" value={repoId} onChange={event => {
          setRepoId(event.target.value);
          onSourceEdited();
          if (revision === starter.revision) setRevision('main');
        }} required placeholder="owner/dataset-name" autoCapitalize="none" autoCorrect="off" spellCheck={false} aria-describedby="repo-help" />
        <p id="repo-help" className="field-help">A public LeRobot v2 or v3 dataset.</p>
        <label className="field-label" htmlFor="revision">Revision</label>
        <input id="revision" className="mono-input" name="revision" value={revision} onChange={event => { setRevision(event.target.value); onSourceEdited(); }} required autoCapitalize="none" autoCorrect="off" spellCheck={false} aria-describedby="revision-help" />
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
    {job.result && <DatasetResult profile={job.result} />}
  </>;
}

function Workbench() {
  const queryClient = useQueryClient();
  const [projectId, setProjectId] = useState('');
  const [projectName, setProjectName] = useState('');
  const [selectedJobId, setSelectedJobId] = useState('');
  const [activeStage, setActiveStage] = useState(0);
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
    if (!projects.data || projectId) return;
    let stored = '';
    try { stored = localStorage.getItem('firebird.project') ?? ''; } catch { /* Storage may be unavailable. */ }
    const selected = projects.data.find(project => project.id === stored) ?? projects.data[0];
    if (selected) setProjectId(selected.id);
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
  const sortedJobs = [...(jobs.data ?? [])].sort((a, b) => b.created_at.localeCompare(a.created_at));
  const selectedJob = sortedJobs.find(job => job.id === selectedJobId) ?? sortedJobs[0];
  const connected = health.isSuccess && !health.isError;

  const stage = stages[activeStage];

  return <div className="workspace">
    <a href="#main" className="skip-link">Skip to workspace</a>
    <aside className="sidebar" aria-label="Workspace navigation">
      <a className="brand" href="/" aria-label="Firebird workspace home"><span className="brand-mark"><Icon name="layers" size={21} /></span><span>Firebird<span className="brand-subtitle">Robotics workspace</span></span></a>
      <nav className="stage-navigation" aria-label="Policy lifecycle">
        <p className="sidebar-section-label">Workspace</p>
        <ul className="stage-list">{stages.map((item, index) => <li key={item.name}>
          <button type="button" className={`stage-button${activeStage === index ? ' selected' : ''}`} onClick={() => setActiveStage(index)} aria-current={activeStage === index ? 'page' : undefined}>
            <Icon name={item.icon} size={19} /><span>{item.name}</span>{index > 0 && <small>Planned</small>}
          </button>
        </li>)}</ul>
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
      <div className="sidebar-bottom"><a className="sidebar-link" href={apiReferenceUrl} target="_blank" rel="noreferrer"><Icon name="book" size={18} /> API reference <Icon name="external" size={13} /></a><div className="workspace-identity"><span className="workspace-avatar">F</span><div><strong>Local workspace</strong><span>Firebird · v0.1</span></div></div></div>
    </aside>

    <div className="main-shell">
      <header className="topbar"><div className="breadcrumb"><Icon name={stage.icon} size={18} /><strong>{stage.name}</strong><span className="breadcrumb-divider">/</span><span className="breadcrumb-project">{project?.name ?? 'No project selected'}</span></div><ThemeToggle /></header>
      <main id="main" className="main-content" tabIndex={-1}>
        <div className="page-heading"><div><p className="eyebrow">Step {String(activeStage + 1).padStart(2, '0')}</p><h1>{stage.name}</h1><p>{stage.description}</p></div><div className={`connection ${connected ? 'connected' : ''}`} role="status"><span />{health.isPending ? 'Connecting' : connected ? 'Connected' : 'Offline'}</div></div>
        {!connected && !health.isPending && <div className="connection-notice"><ErrorNotice error={health.error} /><button className="text-button" onClick={() => { void health.refetch(); void projects.refetch(); void capabilities.refetch(); }}>Retry connection</button></div>}
        {capabilities.error && connected && <div className="connection-notice"><ErrorNotice error={capabilities.error} /><button className="text-button" onClick={() => void capabilities.refetch()} disabled={capabilities.isFetching}>Retry capabilities</button></div>}
        <div className="dataset-view" hidden={activeStage !== 0}>
          <div className="section-tabs"><nav className="dataset-tab-buttons" aria-label="Dataset views"><button type="button" className={`section-tab${datasetView === 'sources' ? ' active' : ''}`} aria-pressed={datasetView === 'sources'} onClick={() => setDatasetView('sources')}>Sources</button><button type="button" className={`section-tab${datasetView === 'inspection' ? ' active' : ''}`} aria-pressed={datasetView === 'inspection'} disabled={!selectedJob} onClick={() => setDatasetView('inspection')}>Inspection{sortedJobs.length > 0 && <span className="tab-count">{sortedJobs.length}</span>}</button></nav><span className="section-note">LeRobot v2 / v3</span></div>
          <div className="content-grid source-grid" hidden={datasetView !== 'sources'}>
            <div className="intake-column"><IntakeForm key={`${projectId}-${starterSelection}`} project={project} starter={starter} onSourceEdited={() => setActiveStarterId('')} localAvailable={capabilities.data?.some(item => item.operation === 'dataset.inspect.local' && item.status === 'available') ?? false} onCreated={job => { setSelectedJobId(job.id); setDatasetView('inspection'); }} /><div className="source-note"><Icon name="database" size={16} /><p>Inspect metadata first. Camera media and sample rows load when you open a visual preview.</p></div></div>
            <DatasetStarters selected={activeStarterId} onSelect={item => { setStarter(item); setActiveStarterId(item.id); setStarterSelection(previous => previous + 1); }} />
          </div>
          <div className="inspection-view" hidden={datasetView !== 'inspection'}>
            <section className="inspection-record" aria-labelledby="activity-title">
              <div className="activity-heading"><div><h2 id="activity-title">Dataset inspection</h2><p>Explore the source, then look inside an episode.</p></div><button type="button" className="secondary-button" onClick={() => setDatasetView('sources')}>Change source</button></div>
              <ErrorNotice error={jobs.error} />
              {jobs.isPending && projectId && <p className="loading-note" role="status">Loading inspections…</p>}
              {selectedJob && <>
                {sortedJobs.length > 1 && <div className="history-control"><label htmlFor="inspection-history">History</label><select id="inspection-history" value={selectedJob.id} onChange={event => setSelectedJobId(event.target.value)}>{sortedJobs.map(job => <option key={job.id} value={job.id}>{job.request.repo_id || 'Local dataset'} · {displayDate(job.created_at)} · {job.status}</option>)}</select></div>}
                <JobDetail key={selectedJob.id} job={selectedJob} projectId={projectId} />
                {selectedJob.result && <DatasetExplorer key={`explorer-${selectedJob.id}`} job={selectedJob} active={activeStage === 0 && datasetView === 'inspection'} />}
              </>}
            </section>
          </div>
          {jobs.error && datasetView === 'sources' && <ErrorNotice error={jobs.error} />}
          <footer className="workspace-footer"><span>Inspect first. Build on what you know.</span><span>Dataset workspace</span></footer>
        </div>
        {activeStage > 0 && <section className="planned-panel" aria-labelledby="planned-title"><span className="empty-icon"><Icon name={stage.icon} size={28} /></span><span className="planned-badge">Planned</span><h2 id="planned-title">{stage.name} is on the roadmap</h2><p>{capabilities.data?.find(item => item.stage.toLowerCase() === stage.name.toLowerCase())?.description ?? 'This stage is not available in the current application.'}</p><p className="planned-note">You can start by inspecting your dataset. Your project and inspection history will be here when this stage is ready.</p><button className="secondary-button" onClick={() => setActiveStage(0)}>Go to Dataset <Icon name="arrow" size={15} /></button></section>}
      </main>
    </div>
  </div>;
}

export function Workspace() {
  const [queryClient] = useState(() => new QueryClient({ defaultOptions: { queries: { refetchOnWindowFocus: true, staleTime: 5_000 } } }));
  return <QueryClientProvider client={queryClient}><Workbench /></QueryClientProvider>;
}
