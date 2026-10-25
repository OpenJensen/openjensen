import { groupedTrainingModels } from '@/lib/training-models';
import { Icon } from '@/components/icon';
import { WorkflowChoiceGrid } from '@/components/workflow-choice-grid';
import { defaultRecipe } from '@/lib/training-recipe';
import { modelStatusLabel, number } from './training-shared';
import type { TrainingPanelState } from './training-state';

/** Setup step 2: one or more base models and the training method for the model being edited. */
export function TrainingModelStep({ panel }: { panel: TrainingPanelState }) {
  const {
    activeDataset,
    activeMethod,
    availableMethods,
    batchFor,
    busy,
    displayedModelId,
    headingRef,
    model,
    modelBatchSizes,
    modelIssue,
    models,
    modelSupported,
    mutation,
    recipe,
    resumeId,
    selectedModels,
    setMethod,
    setModelBatchSizes,
    setModelId,
    setModelIds,
    setRecipe,
    setRuntimeId,
    setStep,
  } = panel;
  return (
    <section aria-labelledby="training-model-title">
      <div className="training-heading">
        <h2 id="training-model-title" ref={headingRef} tabIndex={-1}>
          Base model
        </h2>
        <span className="training-model-selection-count" aria-live="polite">
          {selectedModels.length} {selectedModels.length === 1 ? 'model' : 'models'} selected
        </span>
      </div>
      <fieldset className="training-model-groups">
        <legend className="visually-hidden">Base model</legend>
        {groupedTrainingModels(models).map((group) => (
          <section className="training-model-group" key={group.id} aria-labelledby={`training-family-${group.id}`}>
            <h3 id={`training-family-${group.id}`}>
              {group.label}
              <span className="training-model-family-count" aria-hidden="true">
                {group.models.length}
              </span>
            </h3>
            <div className="training-model-grid">
              {group.models.map((item) => {
                const statusLabel = modelStatusLabel(item, modelSupported(item));
                const memory = item.minimum_gpu_memory_gb ?? item.suggested_gpu_memory_gb;
                return (
                  <label
                    key={item.id}
                    className={`training-model-tile model-${item.id}${!modelSupported(item) ? ' unavailable' : ''}`}
                    title={`${item.model_id} · ${item.description}`}
                  >
                    <input
                      type="checkbox"
                      name="training-model"
                      aria-label={item.label}
                      checked={selectedModels.some((model) => model.id === item.id)}
                      disabled={!!resumeId || busy || !modelSupported(item)}
                      onChange={() => {
                        if (displayedModelId)
                          setModelBatchSizes((previous) => ({
                            ...previous,
                            [displayedModelId]: recipe.batchSize,
                          }));
                        const selected = selectedModels.map((model) => model.id);
                        const next = selected.includes(item.id)
                          ? selected.filter((id) => id !== item.id)
                          : [...selected, item.id];
                        const editId = next.includes(item.id) ? item.id : (next[0] ?? '');
                        setRecipe((previous) => ({
                          ...previous,
                          batchSize:
                            modelBatchSizes[editId] ??
                            (editId === 'smolvla' ? defaultRecipe.batchSize : editId === 'psi0' ? 2 : 4),
                        }));
                        setModelIds(next);
                        setModelId(next.includes(item.id) ? item.id : (next[0] ?? ''));
                        setRuntimeId('');
                        mutation.reset();
                      }}
                    />
                    <span className="training-model-copy">
                      <span>
                        <strong>{item.label}</strong>
                        {statusLabel && <small className="training-model-status">{statusLabel}</small>}
                      </span>
                      <span className="training-model-memory">
                        {typeof memory === 'number' && Number.isFinite(memory) && memory > 0 ? (
                          <>
                            GPU budget · <strong>{number(memory)} GB+</strong>
                          </>
                        ) : (
                          <strong>GPU budget not verified</strong>
                        )}
                      </span>
                    </span>
                  </label>
                );
              })}
            </div>
          </section>
        ))}
      </fieldset>
      {selectedModels.length > 1 && (
        <p className="training-selection-note">{selectedModels.length} independent jobs · one cloud GPU per model</p>
      )}
      {model && (
        <div className="training-method-choices">
          <div className="training-edit-model">
            {selectedModels.length > 1 &&
              selectedModels.map((item) => (
                <button
                  type="button"
                  key={item.id}
                  aria-pressed={model.id === item.id}
                  className="secondary-button"
                  disabled={busy}
                  onClick={() => {
                    setModelBatchSizes((previous) => ({ ...previous, [model.id]: recipe.batchSize }));
                    setModelId(item.id);
                    setRecipe((previous) => ({ ...previous, batchSize: batchFor(item) }));
                  }}
                >
                  Settings for {item.label}
                </button>
              ))}
          </div>
          <WorkflowChoiceGrid
            name="training-method"
            label="Training method"
            value={activeMethod ?? ''}
            disabled={!!resumeId || busy}
            options={availableMethods.map((item) => ({
              value: item.id,
              label: item.label,
              meta:
                item.id === 'lora'
                  ? 'Adapters'
                  : item.id === 'qlora'
                    ? 'Quantized adapters'
                    : item.id === 'full'
                      ? 'Trainable weights'
                      : undefined,
              icon: item.id === 'lora' ? 'sliders' : 'layers',
            }))}
            onChange={setMethod}
          />
        </div>
      )}
      {modelIssue && (
        <p className="error-notice" role="alert">
          {modelIssue}
        </p>
      )}
      <div className="training-footer">
        <button type="button" className="text-button" onClick={() => setStep(0)}>
          Back
        </button>
        <button
          type="button"
          className="primary-button"
          disabled={!model || !activeMethod || !activeDataset || !!modelIssue}
          onClick={() => setStep(2)}
        >
          Next <Icon name="arrow" size={15} />
        </button>
      </div>
    </section>
  );
}
