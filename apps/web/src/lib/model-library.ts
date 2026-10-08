import { isDatasetJob, type Job, type PolicyArtifact } from './api';
import { checkpointStep, quantizationIssue, trainingRunModelLabel } from './checkpoints';
import { studentTeacher } from './native-distillation';
import { nativeQuantizationInput } from './native-quantization';
import { replayPolicy } from './native-replay';
import { nativeInput } from './native-simulation';

export type ModelAction = 'distill' | 'quantize' | 'evaluate' | 'replay' | 'simulate';
export const modelFormats = new Set(['training_checkpoint', 'native_checkpoint', 'inference_export', 'gguf', 'native_quantized', 'deployment_package']);
export function record(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {};
}
export function textValue(value: unknown): string | null { return typeof value === 'string' && value.trim() ? value : null; }
export function ownedModels(artifacts: PolicyArtifact[], project: string): PolicyArtifact[] {
  const seen = new Set<string>();
  return artifacts.filter(item => {
    if (item.project_id !== project || !modelFormats.has(item.format) || !item.id || seen.has(item.id)) return false;
    seen.add(item.id); return true;
  });
}
export function modelRunName(artifact: PolicyArtifact): string {
  return textValue(artifact.run_name) ?? `Run ${artifact.job_id.slice(0, 8)}`;
}
export function modelRunGroups(artifacts: PolicyArtifact[]): { key: string; name: string; models: PolicyArtifact[] }[] {
  const groups = new Map<string, { key: string; name: string; models: PolicyArtifact[] }>();
  for (const artifact of artifacts) {
    const key = JSON.stringify([artifact.project_id, artifact.job_id]);
    const group = groups.get(key) ?? { key, name: modelRunName(artifact), models: [] };
    group.models.push(artifact);
    groups.set(key, group);
  }
  for (const group of groups.values()) group.models.sort((a, b) => (checkpointStep(b) ?? -1) - (checkpointStep(a) ?? -1) || a.label.localeCompare(b.label));
  return [...groups.values()];
}
export function modelFamily(artifact: PolicyArtifact, jobs: Job[], artifacts: PolicyArtifact[] = []): string {
  const job = jobs.find(item => item.project_id === artifact.project_id && item.id === artifact.job_id);
  return (job ? trainingRunModelLabel(job, [], [artifact, ...artifacts], jobs) : null)
    ?? textValue(artifact.metadata?.architecture) ?? 'Architecture not recorded';
}
export function modelFormat(artifact: PolicyArtifact): string {
  const labels: Record<string, string> = { training_checkpoint: 'Training checkpoint', native_checkpoint: 'Policy checkpoint', inference_export: 'Inference export', native_quantized: 'Packed policy', gguf: 'GGUF policy', deployment_package: 'Deployment package' };
  const precision = textValue(artifact.metadata?.precision);
  const step = checkpointStep(artifact);
  return [labels[artifact.format] ?? artifact.format, precision, step !== null ? `Step ${step.toLocaleString()}` : null].filter(Boolean).join(' · ');
}
export function quantizeModeFor(artifact: PolicyArtifact): 'native' | 'gguf' | null {
  if (artifact.metadata?.architecture === 'smolvla' && quantizationIssue(artifact)) return null;
  if (nativeQuantizationInput(artifact, artifact.project_id)) return 'native';
  const architecture = artifact.metadata?.architecture;
  if (artifact.format === 'gguf' && artifact.metadata?.precision === 'float' && (!architecture || architecture === 'smolvla')) return 'gguf';
  if (architecture === 'smolvla' && (['training_checkpoint', 'native_checkpoint'].includes(artifact.format) || (artifact.format === 'gguf' && artifact.metadata?.precision === 'float'))) return 'gguf';
  return null;
}
export function modelActionIssue(artifact: PolicyArtifact, action: ModelAction): string | null {
  if (action === 'quantize' && artifact.metadata?.architecture === 'smolvla' && quantizationIssue(artifact)) return quantizationIssue(artifact);
  if (action === 'distill' && studentTeacher(artifact, artifact.project_id)) return null;
  if (action === 'quantize' && quantizeModeFor(artifact)) return null;
  if (action === 'evaluate' && artifact.format === 'gguf') return null;
  if (action === 'replay' && replayPolicy(artifact, artifact.project_id)) return null;
  if (action === 'simulate' && nativeInput(artifact, artifact.project_id) && artifact.format !== 'native_quantized') return null;
  if (['distill', 'quantize', 'simulate'].includes(action) && artifact.format === 'training_checkpoint' && artifact.metadata?.architecture === 'act') return 'Export this ACT checkpoint as an inference model first.';
  const reasons: Record<ModelAction, string> = {
    distill: 'Distillation currently requires a compatible local ACT teacher package.',
    quantize: 'Quantization requires a SmolVLA checkpoint or a compatible local ACT inference model.',
    evaluate: 'Task and engine evaluation currently require a GGUF policy.',
    replay: 'Observation replay requires a compatible packed ACT policy.',
    simulate: 'Simulation requires an imported ACT or SmolVLA inference model and a compatible simulator profile.',
  };
  return reasons[action];
}
export type LineageStep = { id: string; artifact?: PolicyArtifact; job?: Job; missing?: string };
/** Follow only recorded identities inside this project. Missing ancestry stays visible. */
export function modelLineage(artifact: PolicyArtifact, artifacts: PolicyArtifact[], jobs: Job[]): LineageStep[] {
  const ownArtifacts = artifacts.filter(item => item.project_id === artifact.project_id);
  const ownJobs = jobs.filter(item => item.project_id === artifact.project_id);
  const result: LineageStep[] = [], visited = new Set<string>(), visiting = new Set<string>();
  function resumeJob(id: string) {
    const key = `job:${id}`;
    if (visited.has(key)) return;
    if (visiting.has(key)) { result.push({ id: `cycle:${key}`, missing: `Circular training resume reference: ${id}` }); return; }
    visiting.add(key);
    const job = ownJobs.find(candidate => candidate.id === id), request = record(job?.request);
    const source = ownArtifacts.find(candidate => candidate.id === request.artifact_id);
    if (source) visit(source);
    else if (textValue(request.artifact_id)) result.push({ id: `missing:${key}`, missing: `Parent model unavailable: ${request.artifact_id}` });
    if (textValue(request.resume_job_id)) resumeJob(request.resume_job_id as string);
    visiting.delete(key); visited.add(key);
    result.push({ id: key, job, missing: job ? undefined : `Resumed training run unavailable: ${id}` });
  }
  function visit(item: PolicyArtifact) {
    if (visited.has(item.id)) return;
    if (visiting.has(item.id)) { result.push({ id: `cycle:${item.id}`, missing: `Circular lineage reference: ${item.id}` }); return; }
    visiting.add(item.id);
    const job = ownJobs.find(candidate => candidate.id === item.job_id);
    const request = record(job?.request);
    const parents = new Set([...(item.parent_ids ?? []), ...(textValue(request.artifact_id) ? [request.artifact_id as string] : [])]);
    for (const id of parents) {
      const parent = ownArtifacts.find(candidate => candidate.id === id);
      if (parent) visit(parent);
      else if (!visited.has(id)) { visited.add(id); result.push({ id, missing: `Parent model unavailable: ${id}` }); }
    }
    const resume = textValue(request.resume_job_id);
    if (resume) resumeJob(resume);
    visiting.delete(item.id); visited.add(item.id);
    result.push({ id: item.id, artifact: item, job });
  }
  visit(artifact);
  return result;
}
export function modelDataset(artifact: PolicyArtifact, artifacts: PolicyArtifact[], jobs: Job[]): string {
  for (const step of modelLineage(artifact, artifacts, jobs).reverse()) {
    const id = textValue(record(step.job?.request).dataset_job_id);
    const dataset = jobs.find(item => item.project_id === artifact.project_id && item.id === id && isDatasetJob(item));
    if (dataset && isDatasetJob(dataset)) return dataset.result?.repo_id || (dataset.result?.source === 'local' ? 'Local dataset' : 'Dataset details unavailable');
  }
  return textValue(record(artifact.metadata?.dataset).repo_id) ?? 'Dataset not recorded';
}
export function episodePartitions(splits: unknown, total: unknown): { partitions: { name: string; ids: number[] }[]; outside: number[] | null } {
  const value = record(splits), partitions: { name: string; ids: number[] }[] = [];
  for (const name of ['train', 'validation', 'final', 'test']) {
    const ids = value[name];
    if (Array.isArray(ids) && ids.every(id => typeof id === 'number' && Number.isSafeInteger(id) && id >= 0)) partitions.push({ name, ids });
  }
  const complete = partitions.some(part => part.name === 'train') && partitions.some(part => part.name === 'validation');
  const used = new Set(partitions.flatMap(part => part.ids));
  const bounded = typeof total === 'number' && Number.isSafeInteger(total) && total >= 0 && total <= 20000 && [...used].every(id => id < total);
  return { partitions, outside: complete && bounded ? Array.from({ length: total }, (_, id) => id).filter(id => !used.has(id)) : null };
}
