import { isDatasetJob, type DatasetJob } from '@/lib/api';
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
  historyState: 'unselected' | 'loading' | 'unavailable' | 'ready';
  inspection?: DatasetJob;
  inspectionExplicit: boolean;
  workflow?: string;
  onTrain: (datasetId: string) => void;
};

/** Navigation context only: records do not establish model readiness or quality. */
export function WorkspaceJourney({ stage, projectId, projectName, historyState, inspection, inspectionExplicit, workflow, onTrain }: Props) {
  const step = steps[stage];
  if (!step) return null;
  const ready = historyState === 'ready';
  const dataset = ready && inspection?.project_id === projectId && isDatasetJob(inspection) ? inspection : undefined;
  const profile = dataset?.status === 'succeeded' ? dataset.result : undefined;
  // Same admission as the inspection's existing Train action; no recipe is submitted.
  const trainable = profile && (profile.source === 'huggingface' || profile.snapshot);
  return <section className="workspace-journey" aria-label="Workflow context">
    <div className="journey-purpose"><span className="journey-step" aria-hidden="true">{String(stage + 1).padStart(2, '0')}</span><p>{step.purpose}</p>
      {stage === 0 && trainable && dataset && <button className="journey-continue" type="button" onClick={() => onTrain(dataset.id)}>Continue with this dataset <Icon name="arrow" size={15} /></button>}
    </div>
    <dl className="journey-context">
      <div><dt>Project</dt><dd>{projectName ?? 'Choose a project'}</dd></div>
      {stage <= 2 && profile && <div><dt>{inspectionExplicit ? 'Inspection viewed' : 'Latest inspection'}</dt><dd title={dataset?.id}>{profile.repo_id || 'Local dataset'} <span className="journey-id">· {dataset?.id.slice(0, 8)}</span></dd></div>}
      {workflow && <div><dt>Workflow</dt><dd>{workflow}</dd></div>}
    </dl>
  </section>;
}
