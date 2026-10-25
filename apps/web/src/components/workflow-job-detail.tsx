'use client';

import { QuantizationComparison } from '@/components/quantization-comparison';
import { isActive } from '@/lib/api';
import { conciseRunError, runSummary } from '@/lib/run-summary';
import { WorkflowJobLogs, WorkflowJobOutput, WorkflowJobResults } from './workflow-job-report';
import { currentStage, finiteNumber, objectValue, policyRequest } from './workflow-shared';
import type { WorkflowState } from './workflow-state';

export function WorkflowJobDetail({
  workflow,
  onModel,
}: {
  workflow: WorkflowState;
  onModel?: (artifactId: string) => void;
}) {
  const {
    backToHistory,
    cancel,
    checkpointFor,
    data,
    events,
    jobs,
    modelFor,
    newLabel,
    openNew,
    options,
    precisionFor,
    projectId,
    ready,
    selected,
    stage,
  } = workflow;
  if (!selected)
    return (
      <section className="panel workflow-panel workflow-job-detail">
        <button className="text-link" onClick={backToHistory}>
          ← Back to jobs
        </button>
        <p role="status">{jobs.isPending ? 'Loading job…' : 'This job is unavailable.'}</p>
      </section>
    );
  const latestEvent = events.data?.at(-1);
  const phase = currentStage(selected, latestEvent?.stage ?? selected.stage);
  const request = policyRequest(selected);
  const target =
    options.data?.runtimes.find((item) => item.id === request?.runtime_id)?.label ?? request?.runtime_id ?? '—';
  const engineReport = data?.reports?.findLast((report) => finiteNumber(report.p95_ms));
  const inference = data?.reports?.map((report) => objectValue(report.inference)).find(Boolean);
  const engineOnly = request?.evaluation?.mode !== 'libero';
  const error = conciseRunError(selected.error);
  return (
    <article className="panel workflow-panel workflow-job-detail" aria-label={`${stage} job details`}>
      <div className="workflow-view-navigation">
        <button className="text-link" onClick={backToHistory}>
          ← Back to jobs
        </button>
        <button className="secondary-button" onClick={openNew} disabled={!ready}>
          {newLabel}
        </button>
      </div>
      <header className="workflow-job-heading">
        <div>
          <p className="workflow-job-type">
            {stage === 'Quantize' ? 'Quantization' : stage === 'Evaluate' ? 'Evaluation' : 'Policy run'}
          </p>
          <h2>
            {modelFor(selected)}
            {precisionFor(selected) !== '—' ? ` · ${precisionFor(selected)}` : ''}
          </h2>
        </div>
        <span className={`status status-${selected.status}`}>{selected.status}</span>
      </header>
      <section
        className={`workflow-current-stage workflow-current-stage-${selected.status}`}
        aria-label="Current job stage"
      >
        <div>
          <span
            className={isActive(selected) ? 'workflow-stage-indicator active' : 'workflow-stage-indicator'}
            aria-hidden="true"
          />
          <h3 aria-live="polite">{phase}</h3>
        </div>
        {isActive(selected) && (
          <>
            <p>{latestEvent?.message.split('\n')[0] ?? runSummary(selected)}</p>
            <progress className="workflow-progress" aria-label="Run in progress" />
          </>
        )}
        {error && (
          <p className="error-notice" role="alert">
            {error}
          </p>
        )}
        {selected.status === 'succeeded' && (
          <p>
            {selected.kind === 'policy.quantize'
              ? 'Quantized policy ready.'
              : data?.decision === 'validated'
                ? 'The model meets your requested limits.'
                : data?.decision === 'no_feasible_candidate'
                  ? 'No candidate meets your requested limits.'
                  : data?.decision === 'diagnostics_only'
                    ? 'Diagnostics complete.'
                    : 'Results are ready.'}
          </p>
        )}
      </section>
      <dl className="workflow-job-facts">
        <div>
          <dt>Checkpoint</dt>
          <dd>{checkpointFor(selected)}</dd>
        </div>
        <div>
          <dt>Precision</dt>
          <dd>{precisionFor(selected)}</dd>
        </div>
        <div>
          <dt>Target</dt>
          <dd>{target}</dd>
        </div>
        <div>
          <dt>Started</dt>
          <dd>{new Date(selected.created_at).toLocaleString()}</dd>
        </div>
      </dl>
      {isActive(selected) && (
        <button className="secondary-button" disabled={cancel.isPending} onClick={() => cancel.mutate()}>
          {cancel.isPending ? 'Cancelling…' : 'Cancel run'}
        </button>
      )}
      {cancel.error && (
        <p className="error-notice" role="alert">
          {cancel.error.message}
        </p>
      )}
      {events.error && (
        <p className="error-notice" role="alert">
          Activity is unavailable: {events.error.message}
        </p>
      )}
      {stage === 'Quantize' && selected.kind === 'policy.quantize' && (
        <QuantizationComparison job={selected} reports={data?.reports} />
      )}
      {(engineReport || inference) && (
        <WorkflowJobResults engineReport={engineReport} inference={inference} engineOnly={engineOnly} />
      )}
      <WorkflowJobOutput projectId={projectId} selected={selected} data={data} onModel={onModel} />
      <WorkflowJobLogs selected={selected} events={events} data={data} />
    </article>
  );
}
