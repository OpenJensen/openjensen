'use client';

import { useEffect, useState } from 'react';
import { useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query';
import { api, artifactDownloadUrl, isDatasetJob, type Job, type PolicyArtifact, type Project } from '@/lib/api';
import { episodePartitions, modelActionIssue, modelDataset, modelFamily, modelFormat, modelLineage, modelRunGroups, modelRunName, ownedModels, record, textValue, type LineageStep, type ModelAction } from '@/lib/model-library';
import { isCloudArtifact } from '@/lib/checkpoints';
import './model-library.css';

const actionNames: Record<ModelAction, string> = { distill: 'Distill', quantize: 'Quantize', evaluate: 'Evaluate', replay: 'Replay observations', simulate: 'Run in simulation' };
function date(value?: string) { return value && Number.isFinite(Date.parse(value)) ? new Date(value).toLocaleString() : 'Date not recorded'; }
function size(bytes: number) { return bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(2)} GB` : bytes >= 1024 ** 2 ? `${(bytes / 1024 ** 2).toFixed(1)} MB` : bytes >= 1024 ? `${(bytes / 1024).toFixed(1)} KB` : `${bytes} B`; }
function description(value: unknown) { return value === undefined || value === null || value === '' ? 'Not recorded' : typeof value === 'object' ? JSON.stringify(value) : String(value); }
const operationNames: Record<string, string> = { 'policy.import': 'Imported', 'policy.finetune': 'Trained', 'policy.distill': 'Distilled', 'policy.export': 'Exported for inference', 'policy.quantize': 'Quantized', 'policy.workflow': 'Converted and quantized', 'policy.evaluate': 'Evaluated', 'policy.run': 'Run' };

function ModelRunIdentity({ artifact, compact = false }: { artifact: PolicyArtifact; compact?: boolean }) {
  const queryClient = useQueryClient();
  const [editing, setEditing] = useState(false), [name, setName] = useState('');
  const rename = useMutation({
    mutationFn: () => api.renameModelRun(artifact.project_id, artifact.job_id, name.trim()),
    onSuccess: async () => { await queryClient.invalidateQueries({ queryKey: ['artifacts', artifact.project_id] }); setEditing(false); },
  });
  return <div className="model-run-identity">
    {!compact && <div><strong>{modelRunName(artifact)}</strong><small>Run {artifact.job_id.slice(0, 8)}</small></div>}
    {editing ? <form className="model-run-rename" onSubmit={event => { event.preventDefault(); rename.mutate(); }}>
      <label>Model name<input aria-label="Model name" autoFocus maxLength={80} value={name} disabled={rename.isPending} onChange={event => setName(event.target.value)} /></label>
      <button type="submit" className="secondary-button" disabled={!name.trim() || rename.isPending}>{rename.isPending ? 'Saving…' : 'Save name'}</button>
      <button type="button" className="text-link" disabled={rename.isPending} onClick={() => setEditing(false)}>Cancel</button>
      {rename.isError && <p role="alert">{rename.error instanceof Error ? rename.error.message : 'The model name could not be saved.'}</p>}
    </form> : <button type="button" className="text-link" aria-label={`Rename ${modelRunName(artifact)}`} onClick={() => { setName(modelRunName(artifact)); rename.reset(); setEditing(true); }}>Rename</button>}
  </div>;
}

export function ModelWorkflowPicker({ projectId, action, selectedId, onSelect, onLibrary }: { projectId: string; action: 'distill' | 'quantize'; selectedId?: string; onSelect: (artifact: PolicyArtifact) => void; onLibrary: () => void }) {
  const artifacts = useQuery({ queryKey: ['artifacts', projectId], queryFn: () => api.artifacts(projectId), enabled: !!projectId, retry: false, refetchInterval: 3000 });
  const jobs = useQuery({ queryKey: ['jobs', projectId], queryFn: () => api.jobs(projectId), enabled: !!projectId, retry: false, refetchInterval: 5000 });
  const models = ownedModels(artifacts.data ?? [], projectId);
  return <section className="model-workflow-picker" aria-label={action === 'distill' ? 'Your teacher models' : 'Your quantization models'}>
    <div className="model-section-heading"><div><h2>{action === 'distill' ? 'Choose a teacher from My models' : 'Choose a model to quantize'}</h2><p>{action === 'distill' ? 'The student learns from this exact saved teacher.' : 'Compression creates a new version linked to the selected model.'}</p></div><button type="button" className="text-link" onClick={onLibrary}>My models →</button></div>
    {!projectId ? <p role="status">Select a project to see its models.</p> : artifacts.isError || jobs.isError ? <div role="alert"><p>Models or their history could not be refreshed.</p><button type="button" className="secondary-button" onClick={() => { void artifacts.refetch(); void jobs.refetch(); }}>Retry models</button></div> : artifacts.isPending || jobs.isPending ? <p role="status">Loading your models…</p> : !models.length ? <div className="model-empty"><h3>No saved models in this project yet</h3><p>Train or import a model first. Its checkpoints and derived versions will appear in My models.</p><button type="button" className="secondary-button" onClick={onLibrary}>Open My models</button></div> : <div className="model-run-list">{modelRunGroups(models).map(group => <details className="model-run-group" key={group.key} open={group.models.length === 1 || group.models.some(artifact => artifact.id === selectedId)}>
      <summary><span><strong>{group.name}</strong><small>{modelFamily(group.models[0], jobs.data ?? [], models)} · Run {group.models[0].job_id.slice(0, 8)}</small></span><span>{group.models.length} saved {group.models.length === 1 ? 'version' : 'versions'}</span></summary>
      <div className="model-run-content"><ModelRunIdentity artifact={group.models[0]} compact /><div className="model-choice-list">{group.models.map(artifact => {
        const issue = modelActionIssue(artifact, action);
        return <button type="button" className={`model-choice${artifact.id === selectedId ? ' selected' : ''}`} key={artifact.id} disabled={!!issue} aria-pressed={artifact.id === selectedId} aria-label={`Choose ${artifact.label} · ${artifact.id}`} onClick={() => onSelect(artifact)}>
          <span><strong>{artifact.label}</strong><small>{modelFormat(artifact)}</small><small>{modelDataset(artifact, models, jobs.data ?? [])}</small></span><span className="model-choice-status">{issue ?? (artifact.id === selectedId ? 'Selected' : 'Select version')}</span>
        </button>;
      })}</div></div>
    </details>)}</div>}
  </section>;
}

function ModelStep({ step, jobs }: { step: LineageStep; jobs: Job[] }) {
  const job = step.job;
  const telemetry = useQuery({ queryKey: ['training-telemetry', job?.id], queryFn: () => api.trainingTelemetry(job!.id), enabled: job?.kind === 'policy.finetune', retry: false });
  if (step.missing) return <li className="model-lineage-missing"><p role="status">{step.missing}</p></li>;
  const request = record(job?.request), reproducibility = record(telemetry.data?.reproducibility), evidence = record(reproducibility.evidence);
  const datasetJob = jobs.find(item => item.project_id === job?.project_id && item.id === request.dataset_job_id && isDatasetJob(item));
  const dataset = Object.keys(record(reproducibility.dataset)).length ? record(reproducibility.dataset) : record(datasetJob?.result);
  const recipe = Object.keys(record(reproducibility.recipe)).length ? record(reproducibility.recipe) : record(request.native_distillation ?? request.training ?? request.native_quantization);
  const splits = episodePartitions(evidence.splits ?? recipe.splits, dataset.total_episodes);
  const compute = record(reproducibility.compute_target ?? record(job).compute_target), runtime = record(reproducibility.runtime), environment = record(evidence.environment);
  const gpu = environment.gpu_name ?? environment.gpu ?? environment.device_name;
  const baseModel = record(reproducibility.model ?? step.artifact?.metadata?.base_model);
  const affected = job?.kind === 'policy.finetune' || job?.kind === 'policy.distill';
  return <li>
    <div className="model-lineage-heading"><span className="model-lineage-dot" /><div><strong>{operationNames[job?.kind ?? ''] ?? 'Saved model'}{step.artifact ? ` · ${step.artifact.label}` : ''}</strong><span>{date(job?.created_at)}{job ? ` · ${job.status}` : ' · Producing job unavailable'}</span></div></div>
    <div className="model-step-body">
      <dl className="model-facts">
        <div><dt>Model ID</dt><dd>{step.artifact?.id ?? 'No checkpoint identity recorded'}</dd></div>
        <div><dt>Run ID</dt><dd>{job?.id ?? step.artifact?.job_id ?? 'Not recorded'}</dd></div>
        {affected && <><div><dt>Dataset</dt><dd>{textValue(dataset.repo_id) ?? (dataset.source === 'local' ? 'Local dataset' : 'Not recorded')}</dd></div><div><dt>Dataset revision</dt><dd>{description(dataset.revision ?? record(dataset.snapshot).id)}</dd></div>
        <div><dt>Base model</dt><dd>{description(baseModel.repository ?? recipe.model_id)}</dd></div><div><dt>Base model revision</dt><dd>{description(baseModel.revision ?? recipe.model_revision)}</dd></div>
        <div><dt>Dataset manifest</dt><dd>{description(record(dataset.snapshot).manifest_sha256 ?? dataset.metadata_sha256)}</dd></div></>}
        <div><dt>GPU reported by worker</dt><dd>{description(gpu)}</dd></div><div><dt>Recorded compute selection</dt><dd>{[textValue(compute.accelerator ?? runtime.gpu_name ?? runtime.device), textValue(compute.provider ?? runtime.provider), textValue(compute.region ?? runtime.region), textValue(request.runtime_id)].filter(Boolean).join(' · ') || 'Not recorded'}</dd></div>
      </dl>
      {affected && <div className="model-episode-splits"><h4>Data used in this step</h4>{splits.partitions.length ? <dl className="model-facts">{splits.partitions.map(part => <div key={part.name}><dt>{part.name === 'train' ? 'Training' : part.name === 'final' ? 'Final held-out' : part.name === 'test' ? 'Test' : 'Validation'} episodes</dt><dd>{part.ids.length} · {part.ids.join(', ') || 'None'}</dd></div>)}{splits.outside !== null && <div><dt>Outside recorded partitions</dt><dd>{splits.outside.length} · {splits.outside.join(', ') || 'None'}</dd></div>}</dl> : <p>{telemetry.isPending && job?.kind === 'policy.finetune' ? 'Loading recorded training evidence…' : 'Exact episode membership was not recorded.'}</p>}
      {recipe.excluded_episodes !== undefined && <p>Explicitly excluded episodes: {description(recipe.excluded_episodes)}</p>}
      {telemetry.isError && <p role="alert">Training evidence could not be loaded. The saved request is shown; observed splits and hardware are unavailable.</p>}
      </div>}
      <details className="model-record-details"><summary>Recorded recipe and identity</summary><pre>{JSON.stringify({ request: job?.request ?? null, resolved_recipe: Object.keys(recipe).length ? recipe : null, metadata: step.artifact?.metadata ?? null, manifest_sha256: step.artifact?.manifest_sha256 ?? null }, null, 2)}</pre></details>
    </div>
  </li>;
}

type Props = { projects: Project[]; currentProjectId: string; onAction: (artifact: PolicyArtifact, action: ModelAction) => void; onTrain: () => void; onImport: () => void; preferredModelId?: string; onModelRun?: (projectId:string,jobId:string,kind:string,artifactId?:string)=>void; onTrainingRun: (projectId: string, jobId: string, artifactId?: string) => void };
export function ModelLibrary({ projects, currentProjectId, onAction, onTrain, onImport, onTrainingRun, preferredModelId, onModelRun }: Props) {
  const [scope, setScope] = useState('all'), [search, setSearch] = useState(''), [selected, setSelected] = useState<{ project: string; artifact: string } | null>(null);
  useEffect(()=>{if(preferredModelId)setSelected({project:currentProjectId,artifact:preferredModelId});},[preferredModelId,currentProjectId]);
  const visibleProjects = projects.filter(project => scope === 'all' || project.id === currentProjectId);
  const artifactQueries = useQueries({ queries: visibleProjects.map(project => ({ queryKey: ['artifacts', project.id], queryFn: () => api.artifacts(project.id), retry: false, refetchInterval: 5000 })) });
  const jobQueries = useQueries({ queries: visibleProjects.map(project => ({ queryKey: ['jobs', project.id], queryFn: () => api.jobs(project.id), retry: false, refetchInterval: 5000 })) });
  const rows = visibleProjects.flatMap((project, index) => {
    const artifacts = ownedModels(artifactQueries[index].data ?? [], project.id), jobs = (jobQueries[index].data ?? []).filter(job => job.project_id === project.id);
    return artifacts.map(artifact => ({ project, artifact, artifacts, jobs, fresh: artifactQueries[index].isSuccess && !artifactQueries[index].isError && jobQueries[index].isSuccess && !jobQueries[index].isError }));
  }).sort((a, b) => (b.jobs.find(job => job.id === b.artifact.job_id)?.created_at ?? '').localeCompare(a.jobs.find(job => job.id === a.artifact.job_id)?.created_at ?? '') || a.artifact.label.localeCompare(b.artifact.label));
  const chosen = selected ? rows.find(row => row.project.id === selected.project && row.artifact.id === selected.artifact) : undefined;
  const filtered = rows.filter(row => [modelRunName(row.artifact), row.artifact.job_id, row.artifact.label, row.artifact.id, row.project.name, modelFamily(row.artifact, row.jobs, row.artifacts), modelDataset(row.artifact, row.artifacts, row.jobs)].join(' ').toLowerCase().includes(search.toLowerCase().trim()));
  const pending = [...artifactQueries, ...jobQueries].some(query => query.isPending), failed = [...artifactQueries, ...jobQueries].some(query => query.isError);
  function refresh() { for (const query of [...artifactQueries, ...jobQueries]) void query.refetch(); }
  if (chosen) {
    const { artifact, artifacts, jobs, project } = chosen;
    const lineage = modelLineage(artifact, artifacts, jobs);
    const activity = jobs.filter(job => record(job.request).artifact_id === artifact.id && !lineage.some(step => step.job?.id === job.id));
    const producer = jobs.find(job => job.id === artifact.job_id);
    return <article className="model-detail" aria-label="Model details" data-model-id={artifact.id}>
      <button type="button" className="text-link" onClick={() => setSelected(null)}>← All my models</button>
      <header className="model-detail-heading"><div><p className="model-eyebrow">{project.name} · {modelFamily(artifact, jobs, artifacts)}</p><h2>{artifact.label}</h2><p>{modelFormat(artifact)} · {size(artifact.file_bytes)} · {isCloudArtifact(artifact) ? 'Cloud storage' : 'Local storage'}</p><p>Model {artifact.id}</p></div><a className="secondary-button" href={artifactDownloadUrl(project.id, artifact.id)}>Download model</a></header>
      <ModelRunIdentity artifact={artifact} />
      {!chosen.fresh && <p className="warning-box" role="alert">This model’s records could not be refreshed. Refresh before continuing with it.<button type="button" className="text-link" onClick={refresh}>Refresh records</button></p>}
      <section className="model-next-actions" aria-label="Continue with this model"><h3>Continue with this model</h3><div className="model-actions">{(Object.keys(actionNames) as ModelAction[]).map(action => {
        const issue = modelActionIssue(artifact, action);
        return <div key={action}><button type="button" className="secondary-button" disabled={!chosen.fresh || !!issue} onClick={() => onAction(artifact, action)}>{actionNames[action]}</button>{issue && <small>{issue}</small>}</div>;
      })}</div>{producer && ['policy.distill','policy.quantize','policy.workflow'].includes(producer.kind) && onModelRun && <button type="button" className="text-link" disabled={!chosen.fresh} onClick={()=>onModelRun(project.id,producer.id,producer.kind,artifact.id)}>Open this model’s {producer.kind==='policy.distill'?'distillation':'quantization'} job →</button>}{producer?.kind === 'policy.finetune' && <button type="button" className="text-link" disabled={!chosen.fresh} onClick={() => onTrainingRun(project.id, producer.id, artifact.id)}>Open this checkpoint’s training run and export →</button>}</section>
      <section className="model-lineage" aria-label="Model lineage"><div className="model-section-heading"><div><h3>How this model was made</h3><p>Recorded parent models and producing runs, from the original model to this version.</p></div><span>{lineage.length} recorded steps</span></div><ol>{lineage.map(step => <ModelStep key={step.id} step={step} jobs={jobs} />)}</ol></section>
      <section className="model-activity" aria-label="Model activity"><h3>Work using this model</h3>{activity.length ? <ul>{activity.map(job => <li key={job.id}><strong>{operationNames[job.kind] ?? job.kind}</strong><span>{date(job.created_at)} · {job.status} · Run {job.id}</span>{job.error && <p>{job.error}</p>}</li>)}</ul> : <p>No subsequent work has been recorded for this version.</p>}</section>
    </article>;
  }
  return <section className="model-library" aria-label="My models collection">
    <div className="model-section-heading"><div><h2>Your saved models</h2><p>Checkpoints, imported policies and derived versions, with their recorded history.</p></div><div className="model-library-create"><button type="button" className="secondary-button" disabled={!currentProjectId} onClick={onImport}>Import a model</button><button type="button" className="primary-button" disabled={!currentProjectId} onClick={onTrain}>Train a model</button></div></div>
    <div className="model-library-toolbar"><label>Find a model<input type="search" value={search} onChange={event => setSearch(event.target.value)} placeholder="Name, dataset, model ID…" /></label><label>Show<select value={scope} onChange={event => setScope(event.target.value)}><option value="all">All projects</option><option value="current">Current project</option></select></label><button type="button" className="secondary-button" onClick={refresh}>Refresh models</button></div>
    {failed && <p className="warning-box" role="alert">Some model records could not be loaded. Refresh to see their latest history.</p>}
    {pending && <p role="status">Loading your models…</p>}
    {selected && !pending && !chosen && <p role="status">The selected model is unavailable in these records. Refresh the collection or select another model explicitly.</p>}
    {!projects.length && <p role="status">Create a project to train or import your first model.</p>}
    {!pending && !failed && projects.length > 0 && !rows.length && <div className="model-empty"><h3>Your models will live here</h3><p>Train a model on your dataset, or import an existing policy. Every saved checkpoint, distilled student and quantized version will keep a link to its source model and run.</p></div>}
    {!!rows.length && !filtered.length && <p role="status">No models match this search.</p>}
    <div className="model-run-list">{modelRunGroups(filtered.map(row => row.artifact)).map(group => {
      const first = filtered.find(row => row.artifact.id === group.models[0].id && row.project.id === group.models[0].project_id)!;
      return <section className="model-run-group model-library-run" key={group.key} aria-label={`${group.name} · Run ${first.artifact.job_id.slice(0, 8)}`}>
        <ModelRunIdentity artifact={first.artifact} /><p className="model-run-description">{first.project.name} · {modelFamily(first.artifact, first.jobs, first.artifacts)} · {group.models.length} saved {group.models.length === 1 ? 'version' : 'versions'}</p>
        <div className="model-library-grid">{group.models.map(artifact => {
          const row = filtered.find(item => item.project.id === artifact.project_id && item.artifact.id === artifact.id)!;
          return <button type="button" className="model-library-card" key={artifact.id} aria-label={`Open model ${artifact.label} · ${artifact.id}`} onClick={() => setSelected({ project: row.project.id, artifact: artifact.id })}>
            <strong>{artifact.label}</strong><span>{modelFormat(artifact)}</span><span>{modelDataset(artifact, row.artifacts, row.jobs)}</span><span className="model-card-footer">{size(artifact.file_bytes)} · {modelLineage(artifact, row.artifacts, row.jobs).length > 1 ? 'Derived model' : 'Saved version'}<span>View history →</span></span><small>Model {artifact.id}</small>
          </button>;
        })}</div>
      </section>;
    })}</div>
  </section>;
}
