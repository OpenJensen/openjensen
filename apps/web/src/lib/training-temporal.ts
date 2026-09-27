/** Mirrors lifecycle/temporal.py; the worker still validates saved model shapes. */
export type TemporalFamily = 'act' | 'smolvla';
export type TemporalDraft = { enabled: boolean; prediction: number; execution: number };
export type TemporalDrafts = Partial<Record<TemporalFamily, TemporalDraft>>;
export function temporalFamily(id?: string): TemporalFamily | null {
  return id === 'act' || id === 'smolvla' ? id : null;
}
export function defaultTemporal(family: TemporalFamily): TemporalDraft {
  const horizon = family === 'act' ? 100 : 50;
  return { enabled: false, prediction: horizon, execution: horizon };
}
export function temporalIssue(draft: TemporalDraft): string | null {
  if (!draft.enabled) return null;
  if (![draft.prediction, draft.execution].every(value => Number.isInteger(value) && value >= 1 && value <= 1024))
    return 'Use whole numbers from 1 to 1,024 for both horizons.';
  return draft.execution > draft.prediction ? 'Execution horizon cannot exceed prediction horizon.' : null;
}
export function temporalRecipe(family: TemporalFamily | null, draft: TemporalDraft | null): Record<string, number> {
  if (!family || !draft?.enabled) return {};
  const issue = temporalIssue(draft);
  if (issue) throw new Error(issue);
  return { prediction_horizon: draft.prediction, execution_horizon: draft.execution, observation_history: 1, frame_stride: 1 };
}
export function restoreTemporal(value: unknown): TemporalDrafts {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
  const restored: TemporalDrafts = {};
  for (const family of ['act', 'smolvla'] as const) {
    const raw = (value as Record<string, unknown>)[family];
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) continue;
    const draft = raw as Record<string, unknown>;
    // An invalid saved customization stays invalid; never silently substitute a recipe.
    if (draft.enabled === true || draft.enabled === false) restored[family] = {
      enabled: draft.enabled,
      prediction: typeof draft.prediction === 'number' && Number.isFinite(draft.prediction) ? draft.prediction : 0,
      execution: typeof draft.execution === 'number' && Number.isFinite(draft.execution) ? draft.execution : 0,
    };
  }
  return restored;
}
export function checkpointTiming(recipe: Record<string, unknown> | null | undefined): string {
  const prediction = recipe?.prediction_horizon ?? recipe?.chunk_size;
  const execution = recipe?.execution_horizon ?? prediction;
  if ([prediction, execution].every(value => typeof value === 'number' && Number.isInteger(value) && value >= 1 && value <= 1024) && Number(execution) <= Number(prediction))
    return `Checkpoint-owned timing · predict ${prediction} · execute ${execution}`;
  return 'Checkpoint-owned timing · verified by the worker when resumed';
}
