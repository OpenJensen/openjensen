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

// KiteML SmolVLA form defaults, verified in its signed-in UI on 2026-09-26.
// Its automatic save cadence targets about five checkpoints across the run.
export const defaultRecipe: Recipe = {
  trainingSteps: 20000,
  batchSize: 64,
  checkpointCount: 5,
  checkpointIntervalOverride: null,
  seed: 42,
  learningRate: 0.0001,
  gradientAccumulation: 1,
  validationFraction: 0.2,
  evalEvery: 100,
};
