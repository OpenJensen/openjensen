'use client';

import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useEffect, useRef, useState, type FormEvent, type ReactNode } from 'react';
import { api, isActive, type DatasetJob, type DatasetProfile, type Job, type Project } from '@/lib/api';
import { isAugmentationSource } from '@/lib/augmentation-source';
import { Icon } from '@/components/icon';
import { DatasetExplorer, FeatureChips } from '@/components/dataset-explorer';
import { DatasetLibrary, DatasetLabeling, LocalDatasetImport } from '@/components/dataset-library';
import type { LibraryDataset } from '@/lib/dataset-library';
import { publicPath } from '@/lib/base-path';
import { useDurableSubmission } from '@/lib/durable-submission';
import { intakeAcknowledgement, validatedProjectHistory } from '@/lib/dataset-submission';
import { DatasetSubmissionRecovery } from '@/components/dataset-submission-recovery';
import { ErrorNotice, displayDate } from '@/components/workspace-ui';

import { useWorkspace } from '@/components/workspace-context';
function JobStatus({ status }: { status: Job['status'] }) {
  return <span className={`status status-${status}`}><span className="status-dot" />{status}</span>;
}

function formatNumber(value: number) {
  return new Intl.NumberFormat('en-US').format(value);
}

function DatasetResult({ profile, history }: { profile: DatasetProfile; history?: ReactNode }) {
  return <section className="result inspection-overview" aria-labelledby="result-title">
    {profile.snapshot && <p role="status">Training copy verified · {profile.snapshot.file_count} files · {profile.snapshot.lineage_validated ? 'Recorded lineage retained' : 'Ancestry unknown; not independent evaluation evidence'}</p>}
    <div className="result-heading">
      <div><h3 id="result-title">{profile.repo_id || 'Local dataset'}</h3></div>
      <span className="metadata-badge"><Icon name="check" size={14} /> {profile.format.replace('_', ' ')}</span>
    </div>
    <details className="provenance inspection-provenance"><summary>Advanced</summary>
    {history}
    <dl className="dataset-facts">
      <div><dt>episodes</dt><dd>{formatNumber(profile.total_episodes)}</dd></div>
      <div><dt>frames</dt><dd>{formatNumber(profile.total_frames)}</dd></div>
      <div><dt>fps</dt><dd>{profile.fps}</dd></div>
      <div><dt className="visually-hidden">Robot type</dt><dd>{profile.robot_type || 'Unknown robot'}</dd></div>
    </dl>
      <FeatureChips features={profile.features} /><h4>Source details</h4>
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

function IntakeForm({ project, readinessMessage, localAvailable, onCreated, navigationToken, initialLibrary }: { initialLibrary?: LibraryDataset | null; navigationToken: string; project: Project | undefined; readinessMessage: string; localAvailable: boolean; onCreated: (job: Job) => void; }) {
  const queryClient = useQueryClient();
  const [source, setSource] = useState<'huggingface' | 'local'>(initialLibrary ? 'local' : 'huggingface');
  const [library, setLibrary] = useState<LibraryDataset | null>(initialLibrary ?? null);
  const [repoId, setRepoId] = useState('');
  const [revision, setRevision] = useState('');
  const [path, setPath] = useState('');
  const [snapshotForTraining, setSnapshotForTraining] = useState(Boolean(initialLibrary));
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

  const mutation = useMutation({
    mutationFn: async () => {
      if (!project) throw new Error('Create or select a project first.');
      if (source === 'local' && !library && !localAvailable) throw new Error('Choose a dataset folder or file first.');
      if (source === 'local' && library && library.status !== 'ready') throw new Error('Wait for the dataset conversion to finish.');
      if (blocked) throw new Error('Verify the saved request before starting another inspection.');
      const generation = selection.current.generation;
      const job = await submission.submit(source === 'huggingface'
        ? { source, repo_id: repoId.trim(), revision: revision.trim() || 'main' }
        : library ? { source, library_id: library.id, revision: 'main', snapshot_for_training: snapshotForTraining }
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
    <form className={`dataset-intake-form${source === 'huggingface' ? ' hub-intake' : ''}`} onSubmit={submit}>
      <fieldset className="source-options">
        <legend className="visually-hidden">Dataset source</legend>
        <label className={source === 'huggingface' ? 'source-option selected' : 'source-option'}><input type="radio" name="source" value="huggingface" disabled={!project} checked={source === 'huggingface'} onChange={() => { setSource('huggingface'); mutation.reset(); }} /><span>Hugging Face</span></label>
        <label className={`source-option${source === 'local' ? ' selected' : ''}`}><input type="radio" name="source" value="local" checked={source === 'local'} disabled={!project} onChange={() => { setSource('local'); mutation.reset(); }} /><span>Local files</span></label>
      </fieldset>
      {source === 'huggingface' ? <>
        <div className="intake-repository">
          <label className="field-label" htmlFor="repo-id">Dataset repository</label>
          <input id="repo-id" name="repo_id" disabled={!project} value={repoId} onChange={event => {
            setRepoId(event.target.value);
            setRevision('');
          }} required placeholder="owner/dataset-name" autoCapitalize="none" autoCorrect="off" spellCheck={false} />
        </div>
        <details className="intake-advanced"><summary>Revision (optional)</summary><label className="field-label visually-hidden" htmlFor="revision">Revision</label>
        <input id="revision" className="mono-input" name="revision" disabled={!project} value={revision} onChange={event => { setRevision(event.target.value); }} placeholder="Latest (main)" aria-describedby="revision-help" autoCapitalize="none" autoCorrect="off" spellCheck={false} />
        <p id="revision-help" className="field-help">Leave blank for the latest revision, or enter a branch, tag, or commit.</p>
        </details>
      </> : <>
        <LocalDatasetImport projectId={project?.id ?? ''} value={library} onChange={setLibrary} disabled={!project || blocked} />
        {localAvailable && !library && <details className="intake-advanced"><summary>Folder on the app host</summary><label className="field-label" htmlFor="local-path">Dataset directory</label>
        <input id="local-path" name="path" disabled={!project} value={path} onChange={event => setPath(event.target.value)} placeholder="Path inside the configured dataset directory" autoCapitalize="none" autoCorrect="off" spellCheck={false} /></details>}
        <label className="field-label"><input type="checkbox" checked={snapshotForTraining} onChange={event => setSnapshotForTraining(event.target.checked)} /> Prepare immutable training copy</label>
        <p className="field-help">Store a verified training copy. At least two complete episodes are required; original files stay unchanged.</p>
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
      <button className="primary-button inspect-button" type="submit" disabled={blocked || mutation.isPending || (source === 'local' && (!library ? !path : library.status !== 'ready'))}>{mutation.isPending || submission.busy ? 'Checking dataset…' : 'Inspect dataset'}<Icon name="arrow" size={17} /></button>
    </form>
  </section>;
}

function JobDetail({ job, projectId, history }: { job: DatasetJob; projectId: string; history?: ReactNode }) {
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
    {job.result && 'inspection_scope' in job.result && <DatasetResult profile={job.result} history={history} />}
  </>;
}

export default function DatasetsSection() {
  const { activeStage, datasetView, setDatasetView, selectedJob, sortedJobs, projectId, intakeSelection, workflowNavigation, selectedJobId, workflowProjectId, project, projects, importLibrary, capabilities, setSelectedJobId, openDataset, selectedLibrary, setImportLibrary, setIntakeSelection, jobs, navigateStage, startTrainingOnDataset } = useWorkspace();
  return <>
        <div className="dataset-view" hidden={activeStage !== 0}>
          <div className="dataset-sources" hidden={datasetView !== 'sources'}>
            <div className="intake-column"><IntakeForm key={`${projectId}-${intakeSelection}`} navigationToken={`${workflowNavigation}:${activeStage}:${datasetView}:${selectedJobId}`} project={workflowProjectId ? project : undefined} readinessMessage={projects.isPending ? 'Loading projects before importing a dataset.' : projects.isError ? 'Project list unavailable. Retry projects to continue.' : 'Create or select a project to import a dataset.'} initialLibrary={importLibrary} localAvailable={capabilities.data?.some(item => item.operation === 'dataset.inspect.local' && (item.status === 'available' || item.status === 'untested')) ?? false} onCreated={job => { setSelectedJobId(job.id); setDatasetView('inspection'); }} /></div>
            <div className="dataset-choice-divider"><span>or</span></div>
            <DatasetLibrary active={activeStage === 0 && datasetView === 'sources'} projectId={workflowProjectId} onOpen={openDataset} />
          </div>
          {datasetView === 'labels' && selectedLibrary?.project_id === workflowProjectId && <><div className="dataset-back-navigation"><button type="button" className="text-link" onClick={() => setDatasetView('sources')}>← Back to library</button></div><DatasetLabeling key={selectedLibrary.id} entry={selectedLibrary} onInspect={() => { setImportLibrary(selectedLibrary); setIntakeSelection(value=>value+1); setDatasetView('sources'); }} /></>}
          <div className="inspection-view" hidden={datasetView !== 'inspection'}>
            <section className="inspection-record" aria-label="Dataset inspection">
              <div className="dataset-back-navigation"><button type="button" className="text-link" onClick={() => setDatasetView('sources')}>← Back to library</button></div>
              <ErrorNotice error={jobs.error} />
              {jobs.isPending && projectId && <p className="loading-note" role="status">Loading inspections…</p>}
              {selectedJobId && !selectedJob && !jobs.isPending && <p className="warning-box" role="alert">Selected inspection {selectedJobId} is unavailable in this project. Another dataset has not been substituted. Refresh or choose a source explicitly.</p>}
              {selectedJob && <>
                <JobDetail key={selectedJob.id} job={selectedJob} projectId={projectId} history={sortedJobs.length > 1 && <div className="history-control"><label htmlFor="inspection-history">History</label><select id="inspection-history" value={selectedJob.id} onChange={event => setSelectedJobId(event.target.value)}>{sortedJobs.map(job => <option key={job.id} value={job.id}>{job.request.repo_id || 'Local dataset'} · {displayDate(job.created_at)} · {job.status}</option>)}</select></div>} />
                {selectedJob.result && !selectedJob.request.library_id && <DatasetExplorer key={`explorer-${selectedJob.id}`} job={selectedJob} active={activeStage === 0 && datasetView === 'inspection'} />}
                {selectedJob.status === 'succeeded' && (selectedJob.result?.source === 'huggingface' || selectedJob.result?.snapshot) && <div className="dataset-train-action">{isAugmentationSource(selectedJob) && <button className="secondary-button" onClick={() => navigateStage(8)}><Icon name="spark" size={16} />Augment this dataset</button>}<button className="primary-button" onClick={() => startTrainingOnDataset(selectedJob.id)}>Train on this dataset <Icon name="arrow" size={16} /></button></div>}
              </>}
            </section>
          </div>
          {jobs.error && datasetView === 'sources' && <ErrorNotice error={jobs.error} />}
        </div>
  </>;
}
