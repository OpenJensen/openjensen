'use client';

import { QuantizationCompute } from '@/components/quantization-settings';
import { WorkflowChoiceGrid } from '@/components/workflow-choice-grid';
import { WorkflowCheckFields, WorkflowPolicyFields, WorkflowTrainingFields } from './workflow-job-fields';
import type { WorkflowState } from './workflow-state';

export function WorkflowNewJob({
  workflow,
  onOpenQuantize,
  onViewTraining,
}: {
  workflow: WorkflowState;
  onOpenQuantize: () => void;
  onViewTraining?: () => void;
}) {
  const {
    artifacts,
    backToHistory,
    cloudEngine,
    errors,
    evaluationMode,
    executionIssue,
    inputArtifact,
    inputIssue,
    modelRunId,
    mutation,
    nativeQuantization,
    options,
    preferences,
    preferencesBlocker,
    ready,
    runtime,
    runtimes,
    selectedInput,
    setPreferences,
    setRuntimeId,
    stage,
    starting,
    update,
  } = workflow;
  const selectionIssue =
    stage !== 'Fine-tune' && selectedInput && artifacts.isSuccess && !artifacts.isError && !inputArtifact
      ? 'The selected model is unavailable. Refresh or choose another saved model explicitly.'
      : null;
  const formIssue = selectionIssue ?? inputIssue ?? executionIssue;
  return (
    <section className="panel workflow-panel workflow-new-job">
      <div className="workflow-view-navigation">
        <button className="text-link" type="button" disabled={mutation.isPending} onClick={backToHistory}>
          {modelRunId ? '← Back to models' : '← Back to jobs'}
        </button>
      </div>
      {preferencesBlocker && (
        <p className="warning-box" role="status">
          {preferencesBlocker}
        </p>
      )}
      {options.data && !runtimes.length && (
        <div className="workflow-storage-note" role="status">
          <p>
            {stage === 'Evaluate'
              ? 'No evaluation target is available.'
              : stage === 'Run'
                ? 'No policy runner is available.'
                : 'Connect a worker to start quantization.'}
          </p>
          {onViewTraining && (stage === 'Evaluate' || stage === 'Run') && (
            <button type="button" className="text-link" onClick={onViewTraining}>
              View training metrics
            </button>
          )}
        </div>
      )}
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (ready && !mutation.isPending && !formIssue) mutation.mutate();
        }}
      >
        <fieldset className="workflow-fields workflow-controls" disabled={!ready || mutation.isPending}>
          <legend className="visually-hidden">Project policy job</legend>
          {stage === 'Fine-tune' ? (
            <WorkflowTrainingFields workflow={workflow} />
          ) : (
            <WorkflowPolicyFields workflow={workflow} onOpenQuantize={onOpenQuantize} />
          )}
          {stage === 'Quantize' ? (
            <QuantizationCompute
              artifact={inputArtifact}
              runtime={runtime}
              runtimes={runtimes}
              disabled={!ready || mutation.isPending}
              onChange={setRuntimeId}
              localGpuInference={nativeQuantization && runtime?.device === 'cuda'}
            />
          ) : (
            <WorkflowChoiceGrid
              name="execution-target"
              label="Compute"
              value={runtime?.id ?? ''}
              disabled={!runtimes.length}
              options={runtimes.map((target) => ({
                value: target.id,
                label: target.label,
                meta: `${target.execution === 'skypilot' ? 'Cloud' : 'Local'} · ${target.device === 'cuda' ? 'GPU' : 'CPU'}`,
                icon: 'layers',
              }))}
              onChange={setRuntimeId}
              emptyMessage="Connect a compatible worker in Compute settings."
            />
          )}
          {(stage === 'Evaluate' || stage === 'Run') && (
            <WorkflowCheckFields
              cloudEngine={cloudEngine}
              evaluationMode={evaluationMode}
              preferences={preferences}
              runtime={runtime}
              update={update}
              setPreferences={setPreferences}
            />
          )}
          {formIssue && (
            <p className="warning-box" role="status">
              {formIssue}
            </p>
          )}
          <button
            className="primary-button"
            type="submit"
            disabled={
              !ready ||
              !runtime ||
              starting ||
              (stage !== 'Fine-tune' && !inputArtifact) ||
              !!formIssue ||
              (stage === 'Fine-tune' && !runtime.training)
            }
          >
            {mutation.isPending
              ? 'Starting…'
              : stage === 'Quantize'
                ? nativeQuantization
                  ? 'Run quantization workflow'
                  : 'Start quantization'
                : stage === 'Evaluate'
                  ? 'Start evaluation'
                  : stage === 'Fine-tune'
                    ? 'Start fine-tuning'
                    : 'Reload and run'}
          </button>
        </fieldset>
      </form>
      {errors.map((error, index) => (
        <p className="error-notice" role="alert" key={index}>
          {error?.message}
        </p>
      ))}
    </section>
  );
}
