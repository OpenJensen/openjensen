import { isActive, isDatasetJob, type DatasetJob, type Job } from '@/lib/api';
import { Icon } from './icon';
import './workspace-journey.css';

const steps = [
  { name: 'Dataset', purpose: 'Inspect your data, then choose a dataset for training.', kinds: ['dataset.inspect'] },
  { name: 'Fine-tune', purpose: 'Train a policy on your data. Review its checkpoint before exporting or refining it.', kinds: ['policy.finetune', 'policy.export'] },
  { name: 'Distill', purpose: 'Optionally train a smaller student from a teacher. Compare the recorded results before continuing.', kinds: ['policy.distill'] },
  { name: 'Quantize', purpose: 'Optionally reduce weight storage. Review measured drift before using the packed policy.', kinds: ['policy.quantize', 'policy.workflow'] },
  { name: 'Evaluate', purpose: 'Choose a configured evaluation protocol. A completed execution is not proof of task success.', kinds: ['policy.evaluate'] },
  { name: 'Run', purpose: 'Review saved runs or choose a runner. Observation replay and simulation are separate workflows.', kinds: ['policy.run'] },
];

type Props = {
  stage: number;
  projectId: string;
  projectName?: string;
  jobs: Job[];
  historyState: 'unselected' | 'loading' | 'unavailable' | 'ready';
  inspection?: DatasetJob;
  inspectionExplicit: boolean;
  workflow?: string;
  onNavigate: (stage: number) => void;
  onTrain: (datasetId: string) => void;
  onRefresh: () => void;
};

/** Navigation context only: records do not establish model readiness or quality. */
export function WorkspaceJourney({ stage, projectId, projectName, jobs, historyState, inspection, inspectionExplicit, workflow, onNavigate, onTrain, onRefresh }: Props) {
  const step = steps[stage];
  if (!step) return null;
  const ready = historyState === 'ready';
  const owned = ready ? jobs.filter(job => job.project_id === projectId) : [];
  const dataset = ready && inspection?.project_id === projectId && isDatasetJob(inspection) ? inspection : undefined;
  const profile = dataset?.status === 'succeeded' ? dataset.result : undefined;
  // Same admission as the inspection's existing Train action; no recipe is submitted.
  const trainable = profile && (profile.source === 'huggingface' || profile.snapshot);
  const records = owned.filter(job => steps.some(item => item.kinds.includes(job.kind)));
  const count = records.length;
  const active = records.filter(isActive).length;
  return <section className="workspace-journey" aria-label="Workflow context">
    <div className="journey-purpose"><span className="journey-step" aria-hidden="true">{String(stage + 1).padStart(2, '0')}</span><p>{step.purpose}</p>
      {stage === 0 && trainable && dataset && <button className="journey-continue" type="button" onClick={() => onTrain(dataset.id)}>Continue with this dataset <Icon name="arrow" size={15} /></button>}
    </div>
    <dl className="journey-context">
      <div><dt>Project</dt><dd>{projectName ?? 'Choose a project'}</dd></div>
      {stage <= 2 && profile && <div><dt>{inspectionExplicit ? 'Inspection viewed' : 'Latest inspection'}</dt><dd title={dataset?.id}>{profile.repo_id || 'Local dataset'} <span className="journey-id">· {dataset?.id.slice(0, 8)}</span></dd></div>}
      {workflow && <div><dt>Workflow</dt><dd>{workflow}</dd></div>}
    </dl>
    {historyState === 'loading' && <p className="journey-read-state" role="status">Loading project activity…</p>}
    {historyState === 'unavailable' && <div className="journey-read-state"><p role="alert">Project activity is unavailable; saved context may be stale.</p><button type="button" className="text-button" onClick={onRefresh}>Refresh project activity</button></div>}
    {ready && <details className="journey-activity">
      <summary>Project activity <span>{count} recorded {count === 1 ? 'job' : 'jobs'}{active > 0 ? ` · ${active} active` : ''}</span></summary>
      <p className="journey-history-note">Execution history, not a completion checklist. Model quality and task success require their own evidence. Choose a stage to review its records.</p>
      <ol>{steps.map((item, index) => {
        const records = owned.filter(job => item.kinds.includes(job.kind));
        const running = records.filter(isActive).length;
        const succeeded = records.filter(job => job.status === 'succeeded').length;
        const attention = records.filter(job => ['failed', 'cancelled', 'interrupted'].includes(job.status)).length;
        return <li key={item.name}><button type="button" aria-label={`Review ${item.name} activity`} aria-current={index === stage ? 'step' : undefined} onClick={() => onNavigate(index)}>
          <span className="journey-record-title"><span aria-hidden="true">{index + 1}</span>{item.name}<Icon name="arrow" size={13} /></span>
          <span className="journey-record-status">{records.length ? `${records.length} recorded` : 'No recorded jobs'}</span>
          {records.length > 0 && <small>{[running ? `${running} active` : '', succeeded ? `${succeeded} succeeded` : '', attention ? `${attention} stopped or failed` : ''].filter(Boolean).join(' · ')}</small>}
        </button></li>;
      })}</ol>
    </details>}
  </section>;
}
