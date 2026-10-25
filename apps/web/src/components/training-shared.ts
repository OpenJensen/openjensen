import type { DatasetJob, DatasetProfile, Job } from '@/lib/api';
import type { TrainingModel } from '@/lib/training-models';
import { defaultRecipe, type Recipe } from '@/lib/training-recipe';

export function validTrainingHistory(value: unknown, project: string): value is Job[] {
  return (
    Array.isArray(value) &&
    value.every(
      (item) =>
        item &&
        typeof item === 'object' &&
        !Array.isArray(item) &&
        typeof item.id === 'string' &&
        item.id.length > 0 &&
        item.project_id === project &&
        typeof item.kind === 'string' &&
        item.kind.length > 0 &&
        typeof item.status === 'string' &&
        ['queued', 'running', 'succeeded', 'failed', 'cancelled', 'interrupted'].includes(item.status) &&
        typeof item.created_at === 'string' &&
        Number.isFinite(Date.parse(item.created_at)) &&
        typeof item.updated_at === 'string' &&
        Number.isFinite(Date.parse(item.updated_at)) &&
        item.request &&
        typeof item.request === 'object' &&
        !Array.isArray(item.request),
    )
  );
}
export type InspectedDataset = DatasetJob & { result: DatasetProfile };

export function positiveInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value > 0;
}

export function savedRecipe(saved: Record<string, unknown>): Recipe {
  const version = typeof saved.trainingDefaultsVersion === 'number' ? saved.trainingDefaultsVersion : 0;
  const trainingSteps =
    positiveInteger(saved.trainingSteps) && !(version < 2 && saved.trainingSteps === 1000)
      ? saved.trainingSteps
      : defaultRecipe.trainingSteps;
  const batchSize =
    positiveInteger(saved.batchSize) && !(version < 2 && saved.batchSize === 1)
      ? saved.batchSize
      : defaultRecipe.batchSize;
  // The old five-step value shipped as a default. Other interval choices remain
  // exact until the user edits the new checkpoint-count control.
  const interval =
    version < 3
      ? positiveInteger(saved.checkpointEvery) && saved.checkpointEvery !== 5
        ? saved.checkpointEvery
        : null
      : positiveInteger(saved.checkpointIntervalOverride)
        ? saved.checkpointIntervalOverride
        : null;
  const checkpointCount = interval
    ? Math.ceil(trainingSteps / interval)
    : version >= 3 && positiveInteger(saved.checkpointCount)
      ? saved.checkpointCount
      : defaultRecipe.checkpointCount;
  return {
    trainingSteps,
    batchSize,
    checkpointCount,
    checkpointIntervalOverride: interval,
    seed:
      typeof saved.seed === 'number' && Number.isInteger(saved.seed) && saved.seed >= 0
        ? saved.seed
        : defaultRecipe.seed,
    learningRate:
      typeof saved.learningRate === 'number' && Number.isFinite(saved.learningRate) && saved.learningRate > 0
        ? saved.learningRate
        : defaultRecipe.learningRate,
    gradientAccumulation: positiveInteger(saved.gradientAccumulation)
      ? saved.gradientAccumulation
      : defaultRecipe.gradientAccumulation,
    validationFraction:
      typeof saved.validationFraction === 'number' && saved.validationFraction > 0 && saved.validationFraction < 1
        ? saved.validationFraction
        : defaultRecipe.validationFraction,
    evalEvery: positiveInteger(saved.evalEvery) ? saved.evalEvery : defaultRecipe.evalEvery,
  };
}

export function cameras(profile?: DatasetProfile) {
  return Object.entries(profile?.features ?? {})
    .filter(
      ([, feature]) =>
        feature &&
        typeof feature === 'object' &&
        'dtype' in feature &&
        ['image', 'video'].includes(String(feature.dtype)),
    )
    .map(([key]) => key);
}
export function datasetIssue(profile: DatasetProfile) {
  if (profile.source === 'local' && !profile.snapshot)
    return 'Prepare an immutable training copy when importing this local dataset.';
  if (profile.source === 'huggingface' && !/^[0-9a-f]{40}$/i.test(profile.revision))
    return 'Inspect an immutable dataset revision before training.';
  if (profile.total_episodes < 2) return 'At least two episodes are needed for training and validation.';
  if (!cameras(profile).length) return 'No image or video camera was found.';
  for (const key of ['observation.state' in profile.features ? 'observation.state' : 'states', 'action']) {
    const feature = profile.features[key];
    const shape = feature && typeof feature === 'object' && 'shape' in feature ? feature.shape : undefined;
    if (
      !Array.isArray(shape) ||
      shape.length !== 1 ||
      typeof shape[0] !== 'number' ||
      !Number.isInteger(shape[0]) ||
      shape[0] < 1
    )
      return `Training needs a non-empty ${key} vector.`;
  }
  return null;
}
export function number(value: number) {
  return value.toLocaleString();
}
// Mirrors the reviewed adapters in native_profiles.py and psi_profile.py. The
// server remains authoritative; camera/dimension limits are model-specific.
const dimensionLimits: Record<string, number> = {
  smolvla: 32,
  psi0: 36,
  eo1: 32,
  evo1: 24,
  gr00t_n17: 132,
  pi0: 32,
  pi05: 32,
  pi0_fast: 32,
  wall_x: 20,
  xvla: 20,
};
export function modelDatasetIssue(
  model: TrainingModel | undefined,
  profile: DatasetProfile | undefined,
  cameraKeys: string[],
) {
  if (!model || !profile) return null;
  if (profile.source === 'local' && model.backend !== 'lerobot')
    return 'Local snapshots currently support native LeRobot models, including ACT.';
  if (model.id === 'psi0' && profile.format !== 'lerobot_v2')
    return 'Psi-Zero currently needs a LeRobot v2 dataset. Choose a v2 inspection or another model.';
  if (model.backend === 'lerobot' && profile.format !== 'lerobot_v3')
    return `${model.label} currently needs a LeRobot v3 dataset. Choose a v3 inspection or another model.`;
  if (model.required_cameras && cameraKeys.length !== model.required_cameras)
    return `${model.label} requires ${model.required_cameras} selected camera${model.required_cameras === 1 ? '' : 's'}. Change the camera selection in Dataset.`;
  if (model.id === 'evo1' && cameraKeys.length > 3)
    return 'EVO-1 supports at most three selected cameras. Change the camera selection in Dataset.';
  if (model.id !== 'psi0' && !('observation.state' in profile.features))
    return `${model.label} requires an observation.state feature.`;
  const maximum = dimensionLimits[model.id];
  if (maximum) {
    const state = 'observation.state' in profile.features ? 'observation.state' : 'states';
    for (const key of [state, 'action']) {
      const feature = profile.features[key] as { shape?: number[] } | undefined;
      if ((feature?.shape?.[0] ?? 0) > maximum) return `${model.label} supports at most ${maximum} ${key} dimensions.`;
    }
  }
  return null;
}
export function modelStatusLabel(model: TrainingModel, supported: boolean): string | null {
  switch (model.status) {
    case 'connect_account':
      return 'Connect Google Cloud';
    case 'setup_required':
      return 'Setup required';
    case 'coming_soon':
      return 'Coming soon';
    default:
      return supported ? null : 'Compute unavailable';
  }
}
