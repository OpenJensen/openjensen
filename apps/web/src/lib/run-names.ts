import { isDatasetJob, type Job, type PolicyArtifact } from './api';
import { trainingRunModelLabel } from './checkpoints';
import { datasetName, shortDate, type NamedDataset } from './dataset-names';
import { modelRunName, record, textValue } from './model-library';
import type { TrainingModel } from './training-models';

export type RunNameContext = {
  /** The project's jobs, to find the dataset a run was trained on. */
  jobs?: Job[];
  artifacts?: PolicyArtifact[];
  library?: NamedDataset[];
  catalog?: TrainingModel[];
};

/** How a run was trained, in words. */
export const trainingMethodLabels: Record<string, string> = {
  full: 'Full training',
  lora: 'LoRA',
  qlora: 'QLoRA',
};

/** The name someone gave a run through Rename, or null while it still has its automatic one. */
export function chosenRunName(artifacts: PolicyArtifact[]): string | null {
  for (const artifact of artifacts) {
    const saved = textValue(artifact.run_name);
    if (saved && saved !== modelRunName({ ...artifact, run_name: null })) return saved;
  }
  return null;
}

/** The dataset, or datasets, a run was trained on, by name. */
export function runDatasetName(job: Job, { jobs = [], library = [] }: RunNameContext = {}): string | null {
  const request = record(job.request);
  const ids = Array.isArray(request.dataset_job_ids)
    ? request.dataset_job_ids.filter((id): id is string => typeof id === 'string')
    : [];
  if (!ids.length && textValue(request.dataset_job_id)) ids.push(request.dataset_job_id as string);
  const names = ids
    .map((id) => jobs.find((item) => item.project_id === job.project_id && item.id === id))
    .filter((item): item is Job => !!item && isDatasetJob(item))
    .map((item) => datasetName(item, library));
  if (!names.length) return null;
  return names.length === 1 ? names[0] : `${names[0]} + ${names.length - 1} more`;
}

/**
 * The one name for a training run, used by every page that lists runs or their models:
 * the name someone chose, or else model · dataset, with the start time where the page
 * does not already show it. Two runs of one recipe are told apart by that time.
 */
export function runName(job: Job, context: RunNameContext = {}, { date = false }: { date?: boolean } = {}): string {
  const { jobs = [], artifacts = [], catalog = [] } = context;
  const chosen = chosenRunName(
    artifacts.filter((item) => item.project_id === job.project_id && item.job_id === job.id),
  );
  if (chosen) return chosen;
  return [
    trainingRunModelLabel(job, catalog, artifacts, jobs) ?? 'Training run',
    runDatasetName(job, context),
    date ? shortDate(job.created_at) : null,
  ]
    .filter(Boolean)
    .join(' · ');
}
