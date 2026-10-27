import { Icon } from '@/components/icon';
import { GpuPicker } from '@/components/gpu-picker';
import { checkpointTiming, defaultTemporal, temporalFamily } from '@/lib/training-temporal';
import { trainingMemory } from '@/lib/training-memory';
import { TrainingHelp } from './training-help';
import { TrainingAdvancedSettings } from './training-advanced-settings';
import { defaultRecipe } from '@/lib/training-recipe';
import { number } from './training-shared';
import type { TrainingPanelState } from './training-state';

/** Setup step 3: the recipe review, GPU choice, advanced settings and the start button. */
export function TrainingComputeStep({
  panel,
  onComputeSettings,
}: {
  panel: TrainingPanelState;
  onComputeSettings: () => void;
}) {
  const {
    accumulationIssue,
    accumulationSupported,
    activeCameraKeys,
    activeDataset,
    activeMethod,
    batchFor,
    blocked,
    busy,
    checkpointCount,
    checkpointInterval,
    cloudGpuMemory,
    cloudInstalled,
    effectiveTiming,
    gpuChoices,
    headingRef,
    localRuntimes,
    methodFor,
    model,
    needsAccount,
    originalTraining,
    recipe,
    resumeId,
    runStarting,
    runtime,
    selectedDatasets,
    selectedGpu,
    selectedModels,
    setModelBatchSizes,
    setRecipe,
    setRuntimeId,
    setStep,
    setTemporalDrafts,
    startLabel,
    summaryModel,
    temporalDrafts,
    timing,
    timingIssue,
    trainingBudget,
    update,
    updateTiming,
  } = panel;
  return (
    <section aria-labelledby="training-compute-title">
      <div className="training-heading">
        <h2 id="training-compute-title" ref={headingRef} tabIndex={-1}>
          Compute
        </h2>
      </div>
      <section className="training-recipe-review" aria-label="Training recipe review">
        <div className="training-recipe-title">
          <span>{resumeId ? 'Resume saved training' : 'Your training recipe'}</span>
          <strong>
            {selectedModels.length > 1 && !resumeId
              ? `${selectedModels.length} independent jobs · 1 GPU each`
              : `${summaryModel} · ${activeMethod?.toUpperCase()}`}
          </strong>
        </div>
        <dl>
          <div>
            <dt>Dataset</dt>
            <dd>
              {activeDataset?.result.repo_id ?? (activeDataset ? 'Local training snapshot' : 'Choose a dataset')}
              {activeDataset && (
                <small
                  className="training-recipe-identity"
                  title={activeDataset.result.snapshot?.manifest_sha256 ?? activeDataset.result.revision}
                >
                  {activeDataset.result.snapshot ? 'Snapshot' : 'Pinned revision'}{' '}
                  {(activeDataset.result.snapshot?.manifest_sha256 ?? activeDataset.result.revision).slice(0, 12)}
                </small>
              )}
            </dd>
          </div>
          <div>
            <dt>Observations</dt>
            <dd>
              {activeCameraKeys.length} {activeCameraKeys.length === 1 ? 'camera' : 'cameras'}
              {activeDataset
                ? ` · ${number(resumeId ? activeDataset.result.total_episodes : selectedDatasets.reduce((sum, item) => sum + item.result.total_episodes, 0))} episodes`
                : ''}
            </dd>
          </div>
          <div>
            <dt>Training budget</dt>
            <dd>
              {resumeId
                ? 'Saved recipe'
                : selectedModels.length > 1
                  ? `${number(recipe.trainingSteps)} steps per job`
                  : trainingBudget}
            </dd>
          </div>
          {(resumeId || selectedModels.length <= 1) && (
            <div>
              <dt>Action timing</dt>
              <dd>
                {resumeId
                  ? checkpointTiming(originalTraining)
                  : effectiveTiming
                    ? `Predict ${effectiveTiming.prediction} · execute ${effectiveTiming.execution}${timing?.enabled ? '' : ' · default'}`
                    : 'Model-owned settings'}
              </dd>
            </div>
          )}
        </dl>
        {!resumeId &&
          selectedModels.map((item) => {
            const family = temporalFamily(item.id);
            const draft = family ? (temporalDrafts[family] ?? defaultTemporal(family)) : null;
            const modelTiming = family ? (draft?.enabled ? draft : defaultTemporal(family)) : null;
            const estimate = trainingMemory(
              item,
              methodFor(item),
              batchFor(item),
              activeCameraKeys,
              activeDataset?.result,
              modelTiming?.prediction,
            );
            return (
              <div className="training-memory-estimate" key={item.id}>
                <strong>{item.label}</strong>
                <span>
                  {selectedModels.length > 1 ? `${methodFor(item).toUpperCase()} · batch ${batchFor(item)} · ` : ''}
                  {estimate ? `≈ ${estimate.gb} GB GPU memory` : 'Memory estimate unavailable'}
                  <TrainingHelp label={`${item.label} memory estimate`}>
                    {estimate?.source ?? 'This model has no verified catalog memory budget.'}
                  </TrainingHelp>
                </span>
                {selectedModels.length > 1 && (
                  <small>
                    {modelTiming
                      ? `Predict ${modelTiming.prediction} · execute ${modelTiming.execution}${draft?.enabled ? '' : ' · default'}`
                      : 'Model-owned action timing'}
                  </small>
                )}
                {estimate && (
                  <small>
                    {estimate.recommended
                      ? `Suggested GPU: ${estimate.recommended}`
                      : 'No listed GPU meets this estimate. Try a smaller batch.'}
                  </small>
                )}
              </div>
            );
          })}
        {!resumeId && selectedDatasets.length > 1 && (
          <div className="training-mixture-recipe">
            <strong>Combined training set</strong>
            {selectedDatasets.map((item) => (
              <span key={item.id}>
                {item.result.repo_id} · {item.result.revision.slice(0, 12)}
              </span>
            ))}
          </div>
        )}
        {resumeId && <p>Dataset, model and timing stay bound to the saved checkpoint.</p>}
        {accumulationIssue && (
          <>
            <p role="alert">{accumulationIssue}</p>
            <button
              type="button"
              className="secondary-button"
              disabled={busy}
              onClick={() => update('gradientAccumulation', 1)}
            >
              Use accumulation 1
            </button>
          </>
        )}
      </section>
      <GpuPicker
        value={selectedGpu}
        onChange={setRuntimeId}
        disabled={busy}
        choices={[
          ...gpuChoices.map((gpu) => ({
            id: gpu,
            label: gpu === 'A100-80GB' ? 'A100' : gpu,
            memory: `${cloudGpuMemory[gpu]} GB`,
          })),
          ...localRuntimes.map((item) => ({
            id: item.id,
            label: item.label,
            memory: item.gpu_memory_mib ? `${number(item.gpu_memory_mib / 1024)} GB` : undefined,
          })),
        ]}
      />
      <details className="training-disclosure">
        <summary>Advanced settings</summary>
        <TrainingAdvancedSettings
          savedTiming={resumeId ? checkpointTiming(originalTraining) : undefined}
          recipe={recipe}
          busy={busy}
          accumulationSupported={accumulationSupported}
          checkpointCount={checkpointCount}
          checkpointInterval={checkpointInterval}
          cloudStorage={runtime?.execution === 'skypilot'}
          timing={timing}
          effectiveTiming={effectiveTiming}
          timingIssue={timingIssue}
          onChange={update}
          onCheckpointCount={(count) =>
            setRecipe((previous) => ({
              ...previous,
              checkpointCount: count,
              checkpointIntervalOverride: null,
            }))
          }
          onTiming={updateTiming}
          onReset={() => {
            setRecipe({
              ...defaultRecipe,
              batchSize: model?.id === 'smolvla' ? defaultRecipe.batchSize : model?.id === 'psi0' ? 2 : 4,
            });
            setModelBatchSizes({});
            setTemporalDrafts({});
          }}
        />
      </details>
      <div className="training-footer">
        <button type="button" className="text-button" onClick={() => setStep(1)}>
          Back
        </button>
        <span className="training-launch">
          {needsAccount ? (
            <button type="button" className="text-button" onClick={onComputeSettings}>
              Connect account
            </button>
          ) : !runtime && !cloudInstalled ? (
            <button type="button" className="text-button" onClick={onComputeSettings}>
              Configure local compute
            </button>
          ) : (
            <span role="status">{runStarting ? '' : blocked}</span>
          )}
          <button type="submit" className="primary-button" disabled={!!blocked || busy}>
            {startLabel}
            <Icon name="arrow" size={15} />
          </button>
        </span>
      </div>
    </section>
  );
}
