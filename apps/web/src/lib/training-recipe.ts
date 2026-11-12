export type Recipe = {
  trainingSteps: number;
  batchSize: number;
  checkpointCount: number;
  checkpointIntervalOverride: number | null;
  seed: number;
  learningRate: number;
  gradientAccumulation: number;
  validationFraction: number;
  evalEvery: number;
};

// SmolVLA recipe defaults; automatic saving targets five checkpoints per run.
export const defaultRecipe: Recipe = {
  trainingSteps: 20000,
  batchSize: 64,
  checkpointCount: 5,
  checkpointIntervalOverride: null,
  seed: 42,
  learningRate: 0.0001,
  gradientAccumulation: 1,
  validationFraction: 0.2,
  evalEvery: 1000,
};

/**
 * Starting points each model's authors recommend (the LeRobot policy presets). One shared
 * form value cannot suit every model: ACT's learning rate is a tenth of SmolVLA's, and
 * training ACT at SmolVLA's rate was the default here until this table existed.
 */
const modelDefaults: Record<string, { batchSize: number; learningRate: number }> = {
  act: { batchSize: 8, learningRate: 0.00001 },
  smolvla: { batchSize: 64, learningRate: 0.0001 },
  psi0: { batchSize: 2, learningRate: 0.0001 },
};
export const defaultBatchSize = (modelId: string | undefined) => modelDefaults[modelId ?? '']?.batchSize ?? 4;
/** An untouched learning-rate field means "this model's recommended rate". */
export function learningRateFor(modelId: string | undefined, recipe: Pick<Recipe, 'learningRate'>) {
  return recipe.learningRate === defaultRecipe.learningRate
    ? (modelDefaults[modelId ?? '']?.learningRate ?? defaultRecipe.learningRate)
    : recipe.learningRate;
}

/** Never ask for validation less often than the run is long. */
export const evaluationInterval = (recipe: Pick<Recipe, 'evalEvery' | 'trainingSteps'>) =>
  Math.max(1, Math.min(recipe.evalEvery, recipe.trainingSteps));
