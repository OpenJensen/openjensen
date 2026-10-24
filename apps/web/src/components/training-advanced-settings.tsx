import type { ReactNode } from 'react';
import type { Recipe } from '@/lib/training-recipe';
import type { TemporalDraft } from '@/lib/training-temporal';
import { TrainingHelp } from './training-help';

function NumberField({
  label,
  help,
  helpId,
  value,
  min,
  max,
  step = '1',
  disabled,
  onChange,
}: {
  label: string;
  help: ReactNode;
  helpId?: string;
  value: number;
  min: string;
  max?: string;
  step?: string;
  disabled: boolean;
  onChange: (value: number) => void;
}) {
  return (
    <label>
      <span className="training-field-name">
        {label}
        <TrainingHelp label={label} id={helpId}>
          {help}
        </TrainingHelp>
      </span>
      <input
        aria-label={label}
        aria-describedby={helpId}
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

/** Learning settings and action timing for a new run, or the preserved recipe when resuming. */
export function TrainingAdvancedSettings({
  savedTiming,
  recipe,
  busy,
  accumulationSupported,
  checkpointCount,
  checkpointInterval,
  cloudStorage,
  timing,
  effectiveTiming,
  timingIssue,
  onChange,
  onCheckpointCount,
  onTiming,
  onReset,
}: {
  /** Set when resuming: the saved run's action timing, which cannot change. */
  savedTiming?: string;
  recipe: Recipe;
  busy: boolean;
  accumulationSupported: boolean;
  checkpointCount: number;
  checkpointInterval: number;
  cloudStorage: boolean;
  timing: TemporalDraft | null;
  effectiveTiming: TemporalDraft | null;
  timingIssue: string | null;
  onChange: <K extends keyof Recipe>(key: K, value: Recipe[K]) => void;
  onCheckpointCount: (count: number) => void;
  onTiming: (change: Partial<TemporalDraft>) => void;
  onReset: () => void;
}) {
  if (savedTiming !== undefined) {
    return (
      <div className="training-saved-recipe">
        <p>Original recipe preserved.</p>
        <p>{savedTiming}</p>
        <p>Resume loads the saved architecture and processors. Start a new run to request different horizons.</p>
      </div>
    );
  }
  const nominalBatch = recipe.batchSize * recipe.gradientAccumulation;
  return (
    <>
      <div className="training-settings-actions">
        <button type="button" className="text-button" disabled={busy} onClick={onReset}>
          Reset settings
        </button>
      </div>
      <fieldset className="training-hyperparameters">
        <legend>Learning settings</legend>
        <div className="training-fields">
          <NumberField
            label="Steps"
            help="Optimizer updates. Learning-rate schedules and checkpoint intervals use the same count."
            min="2"
            value={recipe.trainingSteps}
            disabled={busy}
            onChange={(value) => onChange('trainingSteps', value)}
          />
          <NumberField
            label="Batch size"
            help="Examples in each GPU microbatch. Larger batches generally need more GPU memory."
            min="1"
            value={recipe.batchSize}
            disabled={busy}
            onChange={(value) => onChange('batchSize', value)}
          />
          <NumberField
            label="Learning rate"
            help="Size of each optimizer update."
            min="0.000000001"
            step="any"
            value={recipe.learningRate}
            disabled={busy}
            onChange={(value) => onChange('learningRate', value)}
          />
          <NumberField
            label="Gradient accumulation"
            helpId="training-accumulation-help"
            help={
              accumulationSupported
                ? `Nominal effective batch: ${nominalBatch.toLocaleString()} examples (${recipe.batchSize} × ${recipe.gradientAccumulation} × 1 GPU). Short final windows contain fewer examples.`
                : 'Only accumulation 1 is available for this model, method and compute target.'
            }
            min="1"
            value={recipe.gradientAccumulation}
            disabled={busy || !accumulationSupported}
            onChange={(value) => onChange('gradientAccumulation', value)}
          />
          <NumberField
            label="Random seed"
            help="Controls episode splitting and training randomness."
            min="0"
            max="2147483647"
            value={recipe.seed}
            disabled={busy}
            onChange={(value) => onChange('seed', value)}
          />
          <NumberField
            label="Validation fraction"
            help="Fraction of separate episodes held out for validation."
            min="0.01"
            max="0.99"
            step="any"
            value={recipe.validationFraction}
            disabled={busy}
            onChange={(value) => onChange('validationFraction', value)}
          />
          <NumberField
            label="Validate every (steps)"
            help="Optimizer updates between held-out validation checks."
            min="1"
            value={recipe.evalEvery}
            disabled={busy}
            onChange={(value) => onChange('evalEvery', value)}
          />
          <NumberField
            label="Checkpoints"
            helpId="training-checkpoint-help"
            help={`Every ${checkpointInterval.toLocaleString()} steps, including the final checkpoint. ${cloudStorage ? 'Saved on Google Cloud.' : 'Saved on the selected compute storage.'}`}
            min="1"
            value={checkpointCount}
            disabled={busy}
            onChange={onCheckpointCount}
          />
        </div>
      </fieldset>
      {timing && effectiveTiming ? (
        <fieldset className="training-temporal" aria-label="Action timing" aria-describedby="training-timing-help">
          <legend>
            <span className="training-field-name">
              Action timing
              <TrainingHelp label="Action timing" id="training-timing-help">
                Prediction: future actions learned together. Execution: actions used before the next observation.
                Existing checkpoints keep their shape.
              </TrainingHelp>
            </span>
          </legend>
          <div className="training-temporal-fields">
            <label htmlFor="training-prediction">
              <span className="training-field-name">
                Prediction horizon
                <TrainingHelp label="Prediction horizon" id="training-prediction-help">
                  Actions predicted together.
                </TrainingHelp>
              </span>
              <input
                id="training-prediction"
                aria-label="Prediction horizon"
                type="number"
                min="1"
                max="1024"
                step="1"
                value={effectiveTiming.prediction}
                disabled={busy}
                aria-invalid={!!timingIssue}
                aria-describedby="training-timing-help training-prediction-help"
                onChange={(event) => onTiming({ enabled: true, prediction: Number(event.target.value) })}
              />
            </label>
            <label htmlFor="training-execution">
              <span className="training-field-name">
                Execution horizon
                <TrainingHelp label="Execution horizon" id="training-execution-help">
                  Actions used before observing again.
                </TrainingHelp>
              </span>
              <input
                id="training-execution"
                aria-label="Execution horizon"
                type="number"
                min="1"
                max={effectiveTiming.prediction >= 1 ? effectiveTiming.prediction : 1024}
                step="1"
                value={effectiveTiming.execution}
                disabled={busy}
                aria-invalid={!!timingIssue}
                aria-describedby="training-timing-help training-execution-help"
                onChange={(event) => onTiming({ enabled: true, execution: Number(event.target.value) })}
              />
            </label>
          </div>
          {timingIssue && (
            <p className="error-notice" role="alert">
              {timingIssue}
            </p>
          )}
        </fieldset>
      ) : (
        <p className="training-native-timing">Action timing follows this model’s native configuration.</p>
      )}
    </>
  );
}
