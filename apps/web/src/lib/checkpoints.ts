import type { Job, PolicyArtifact } from './api';
import { trainingModels, type TrainingModel } from './training-models';

const architectureLabels: Record<string, string> = {
  smolvla: 'SmolVLA', act: 'ACT', diffusion: 'Diffusion Policy', eo1: 'EO-1', evo1: 'EVO-1',
  groot: 'GR00T N1.7', gr00t_n17: 'GR00T N1.7', multi_task_dit: 'Multi-Task DiT',
  pi0: 'π₀', pi0_fast: 'π₀-FAST', pi05: 'π₀.₅', psi0: 'Psi-Zero',
  vla_jepa: 'VLA-JEPA', vqbet: 'VQ-BeT', wall_x: 'WALL-X', xvla: 'XVLA',
  openvla: 'OpenVLA', openvla_oft: 'OpenVLA-OFT',
};
const repositoryArchitectures: Record<string, string> = {
  'code://lerobot/act': 'act', 'code://lerobot/diffusion': 'diffusion',
  'code://lerobot/multi_task_dit': 'multi_task_dit', 'code://lerobot/vqbet': 'vqbet',
  'lerobot/eo1-base': 'eo1', 'zuoxingdong/evo1_libero': 'evo1',
  'lerobot/pi0fast-base': 'pi0_fast', 'lerobot/pi05_base': 'pi05',
  'USC-PSI-Lab/psi-model': 'psi0', 'lerobot/VLA-JEPA-Pretrain': 'vla_jepa',
  'x-square-robot/wall-oss-flow': 'wall_x', 'lerobot/xvla-base': 'xvla',
};

function modelLabel(repository: unknown, architecture: unknown, catalog: TrainingModel[]): string | null {
  const models = [...catalog, ...trainingModels];
  const byRepository = typeof repository === 'string' ? models.find(model => model.model_id === repository) : undefined;
  if (byRepository) return byRepository.label;
  if (typeof architecture === 'string' && architecture.trim()) {
    return models.find(model => model.id === architecture)?.label ?? architectureLabels[architecture.toLowerCase()] ?? architecture;
  }
  if (typeof repository === 'string' && repository.trim()) {
    return architectureLabels[repositoryArchitectures[repository]] ?? repository;
  }
  return null;
}

function artifactModelLabel(artifact: PolicyArtifact | undefined, catalog: TrainingModel[]): string | null {
  const metadata = artifact?.metadata ?? {};
  const baseModel = metadata.base_model;
  const repository = baseModel && typeof baseModel === 'object' && 'repository' in baseModel ? baseModel.repository : metadata.model_id;
  return modelLabel(repository, metadata.architecture ?? metadata.policy_type, catalog);
}

export function trainingRunModelLabel(job: Job, catalog: TrainingModel[] = [], artifacts: PolicyArtifact[] = [], jobs: Job[] = []): string | null {
  const visited = new Set<string>();
  let current: Job | undefined = job;
  while (current && !visited.has(current.id)) {
    visited.add(current.id);
    const request: Job['request'] = current.request;
    const label = modelLabel('training' in request ? request.training?.model_id : null, null, catalog) ??
      artifactModelLabel(artifacts.find(artifact => artifact.job_id === current!.id), catalog);
    if (label) return label;
    const sourceArtifact: PolicyArtifact | undefined = 'artifact_id' in request ? artifacts.find(artifact => artifact.id === request.artifact_id) : undefined;
    const sourceLabel = artifactModelLabel(sourceArtifact, catalog);
    if (sourceLabel) return sourceLabel;
    const sourceJobId: string | null | undefined = ('resume_job_id' in request ? request.resume_job_id : null) ?? sourceArtifact?.job_id;
    current = jobs.find((candidate): boolean => candidate.id === sourceJobId);
  }
  return null;
}

export function checkpointStep(artifact: PolicyArtifact): number | null {
  const metadata = artifact.metadata ?? {};
  for (const value of [metadata.step, metadata.checkpoint_step, metadata.completed_steps]) {
    if (typeof value === 'number' && Number.isInteger(value) && value >= 0) return value;
  }
  const match = /(?:\bstep\s*|checkpoint[-_])(\d+)/i.exec(artifact.label);
  return match ? Number(match[1]) : null;
}

export function isCloudArtifact(artifact: PolicyArtifact): boolean {
  const metadata = artifact.metadata ?? {};
  return artifact.path.startsWith('gs://') || metadata.storage === 'gcs' || metadata.storage_provider === 'gcp' ||
    [metadata.remote_uri, metadata.gcs_uri, metadata.storage_uri].some(value => typeof value === 'string' && value.startsWith('gs://'));
}

export function quantizationIssue(artifact: PolicyArtifact | undefined): string | null {
  const architecture = artifact?.metadata?.architecture;
  if (architecture === 'smolvla' && artifact?.metadata?.training_backend === 'lerobot') return 'GGUF export for native SmolVLA checkpoints is not available yet. Download or resume this checkpoint.';
  return typeof architecture === 'string' && architecture.toLowerCase() !== 'smolvla'
    ? 'GGUF quantization currently supports SmolVLA checkpoints.' : null;
}

export function sortCheckpoints(artifacts: PolicyArtifact[], jobs: Job[] = []): PolicyArtifact[] {
  const dates = new Map(jobs.map(job => [job.id, job.created_at]));
  const unique = new Map<string, PolicyArtifact>();
  for (const artifact of artifacts) {
    const step = checkpointStep(artifact);
    const verifiedAct = artifact.metadata?.architecture === 'act' && artifact.metadata?.reload_verified === true;
    const key = artifact.format === 'training_checkpoint' && step !== null && !verifiedAct ? `${artifact.job_id}:${step}` : artifact.id;
    if (!unique.has(key)) unique.set(key, artifact);
  }
  return [...unique.values()].sort((a, b) => {
    if (a.job_id !== b.job_id) {
      const dateOrder = (dates.get(b.job_id) ?? '').localeCompare(dates.get(a.job_id) ?? '');
      if (dateOrder) return dateOrder;
    }
    return (checkpointStep(b) ?? -1) - (checkpointStep(a) ?? -1) ||
      Number(b.metadata?.architecture === 'act' && b.metadata?.reload_verified === true) - Number(a.metadata?.architecture === 'act' && a.metadata?.reload_verified === true) || a.label.localeCompare(b.label);
  });
}

export function checkpointLabel(artifact: PolicyArtifact, jobs: Job[] = [], catalog: TrainingModel[] = []): string {
  const step = checkpointStep(artifact);
  const job = jobs.find(item => item.id === artifact.job_id);
  const date = job ? new Date(job.created_at).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : null;
  const model = artifactModelLabel(artifact, catalog) ?? (job ? trainingRunModelLabel(job, catalog, [], jobs) : null);
  return [step !== null ? `Step ${step.toLocaleString()}` : artifact.label,
    artifact.metadata?.architecture === 'act' && artifact.metadata?.reload_verified === true ? 'Reload-verified bundle' : null, model, date, `Run ${artifact.job_id.slice(0, 8)}`].filter(Boolean).join(' · ');
}
