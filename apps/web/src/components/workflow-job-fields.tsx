'use client';

import type { Dispatch, SetStateAction } from 'react';
import { ModelVersionPicker } from '@/components/model-version-picker';
import { QuantizationCompression } from '@/components/quantization-settings';
import { WorkflowChoiceGrid, type WorkflowChoice } from '@/components/workflow-choice-grid';
import type { PolicyOptions } from '@/lib/api';
import { checkpointLabel, checkpointStep, isCloudArtifact, trainingRunModelLabel } from '@/lib/checkpoints';
import type { Preferences } from '@/lib/workflow-preferences';
import { precisionName, sizeLabel } from './workflow-shared';
import type { WorkflowState } from './workflow-state';

export function WorkflowTrainingFields({ workflow }: { workflow: WorkflowState }) {
  const {
    checkpoints,
    datasetId,
    datasets,
    method,
    options,
    policyJobs,
    resumeId,
    setDatasetId,
    setMethod,
    setResumeId,
  } = workflow;
  return (
    <>
      <label>
        Fine-tuning method
        <select value={method} onChange={(event) => setMethod(event.target.value)}>
          {options.data?.training_methods.map((item) => (
            <option key={item.id} value={item.id}>
              {item.label}
            </option>
          ))}
        </select>
      </label>
      <label>
        Inspected dataset
        <select value={datasetId || datasets[0]?.id || ''} onChange={(event) => setDatasetId(event.target.value)}>
          <option value="" disabled>
            Select an inspection
          </option>
          {datasets.map((job) => (
            <option key={job.id} value={job.id}>
              {'repo_id' in job.request ? job.request.repo_id : job.id}
            </option>
          ))}
        </select>
      </label>
      <label>
        Resume checkpoint
        <select value={resumeId} onChange={(event) => setResumeId(event.target.value)}>
          <option value="">Start a new training run</option>
          {checkpoints.map((artifact) => (
            <option key={artifact.id} value={artifact.id}>
              {checkpointLabel(artifact, policyJobs, options.data?.training_models)}
            </option>
          ))}
          {policyJobs
            .filter(
              (job) => job.kind === 'policy.finetune' && ['failed', 'interrupted', 'cancelled'].includes(job.status),
            )
            .map((job) => (
              <option key={job.id} value={`job:${job.id}`}>
                Last saved checkpoint · {job.id.slice(0, 8)}
              </option>
            ))}
        </select>
      </label>
    </>
  );
}

export function WorkflowPolicyFields({
  workflow,
  onOpenQuantize,
}: {
  workflow: WorkflowState;
  onOpenQuantize: () => void;
}) {
  const {
    artifacts,
    checkpoints,
    inputs,
    language,
    modelRunId,
    mutation,
    nativeQuantization,
    options,
    policyJobs,
    preferences,
    projectId,
    selectedInput,
    setInput,
    stage,
    update,
  } = workflow;
  const policyChoices: WorkflowChoice[] = [
    ...(stage === 'Quantize' ? checkpoints : inputs).map((artifact, index) => {
      const job = policyJobs.find((item) => item.id === artifact.job_id);
      const step = checkpointStep(artifact);
      const model = job ? trainingRunModelLabel(job, options.data?.training_models, [artifact], policyJobs) : null;
      const checkpoint = ['training_checkpoint', 'native_checkpoint'].includes(artifact.format);
      return {
        value: artifact.id,
        label: checkpoint
          ? [model, step !== null ? `Step ${step.toLocaleString()}` : artifact.label].filter(Boolean).join(' · ')
          : artifact.label,
        description: checkpoint
          ? `${index === 0 ? 'Latest checkpoint · ' : ''}${job ? new Date(job.created_at).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : 'Saved checkpoint'}`
          : `${precisionName(artifact.metadata?.precision)} · ${sizeLabel(artifact.file_bytes)}`,
        meta: checkpoint
          ? `Run ${artifact.job_id.slice(0, 8)} · ${isCloudArtifact(artifact) ? 'Cloud storage' : 'Local storage'}`
          : isCloudArtifact(artifact)
            ? 'Cloud storage'
            : 'Local storage',
        icon: checkpoint ? ('layers' as const) : ('compress' as const),
      };
    }),
    ...(stage === 'Quantize'
      ? inputs
          .filter((artifact) => !['training_checkpoint', 'native_checkpoint'].includes(artifact.format))
          .map((artifact) => ({
            value: artifact.id,
            label: artifact.label,
            meta: 'Floating-point',
            icon: 'layers' as const,
          }))
      : []),
  ];
  return (
    <>
      {stage === 'Quantize' && modelRunId ? (
        <ModelVersionPicker
          runId={modelRunId}
          projectId={projectId}
          artifacts={artifacts.data ?? []}
          jobs={policyJobs}
          value={selectedInput}
          onChange={(value) => {
            setInput(value);
            mutation.reset();
          }}
        />
      ) : (
        <WorkflowChoiceGrid
          name="input-policy"
          label="My model"
          value={selectedInput}
          options={policyChoices}
          onChange={(value) => {
            setInput(value);
            mutation.reset();
          }}
          emptyMessage={artifacts.isPending ? 'Loading policies…' : 'Your saved policies will appear here.'}
        />
      )}
      {(stage === 'Evaluate' || stage === 'Run') && !artifacts.isPending && !artifacts.isError && !inputs.length && (
        <div className="workflow-storage-note" role="status">
          <p>Quantize a SmolVLA checkpoint first.</p>
          <button type="button" className="text-link" onClick={onOpenQuantize}>
            Open Quantize
          </button>
        </div>
      )}
      {stage === 'Quantize' && (
        <>
          <QuantizationCompression
            language={language}
            vision={preferences.vision}
            onLanguage={(value) => update('precision', value)}
            onVision={(value) => update('vision', value)}
          />
          {nativeQuantization && (
            <details className="workflow-advanced">
              <summary>Compare language formats</summary>
              <label className="workflow-check">
                <input
                  type="checkbox"
                  checked={preferences.compareQ4}
                  onChange={(event) => update('compareQ4', event.target.checked)}
                />
                Compare Q8 and Q4 (experimental)
              </label>
            </details>
          )}
        </>
      )}
    </>
  );
}

export function WorkflowCheckFields({
  cloudEngine,
  evaluationMode,
  preferences,
  runtime,
  update,
  setPreferences,
}: {
  cloudEngine: boolean;
  evaluationMode: Preferences['mode'];
  preferences: Preferences;
  runtime?: PolicyOptions['runtimes'][number];
  update: <K extends keyof Preferences>(key: K, value: Preferences[K]) => void;
  setPreferences: Dispatch<SetStateAction<Preferences>>;
}) {
  return (
    <>
      {cloudEngine ? (
        <p className="workflow-input-help">Synthetic inputs · no task success score</p>
      ) : (
        <WorkflowChoiceGrid
          name="evaluation-method"
          label="Check type"
          value={preferences.mode}
          disabled={preferences.suite === 'libero_spatial'}
          options={[
            { value: 'engine', label: 'Inference checks', meta: 'Synthetic inputs · timing', icon: 'chart' },
            {
              value: 'libero',
              label: 'Robot task evaluation',
              meta: runtime?.simulation ? 'LIBERO · task success' : 'LIBERO worker required',
              icon: 'play',
              disabled: !runtime?.simulation,
            },
          ]}
          onChange={(value) => update('mode', value as Preferences['mode'])}
        />
      )}

      {evaluationMode === 'engine' && (
        <details className="workflow-advanced">
          <summary>Engine check settings</summary>
          <div className="workflow-form-row">
            <label>
              Warmup predictions
              <input
                type="number"
                min="1"
                max="100"
                value={preferences.warmup}
                onChange={(event) => update('warmup', Number(event.target.value))}
              />
            </label>
            <label>
              Timed predictions
              <input
                type="number"
                min="2"
                max="1000"
                value={preferences.repetitions}
                onChange={(event) => update('repetitions', Number(event.target.value))}
              />
            </label>
          </div>
        </details>
      )}
      {!cloudEngine && preferences.mode === 'libero' && !runtime?.simulation && (
        <button
          type="button"
          className="text-link"
          onClick={() => setPreferences((old) => ({ ...old, mode: 'engine', suite: 'libero_object' }))}
        >
          Use engine checks
        </button>
      )}
    </>
  );
}
