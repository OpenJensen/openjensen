'use client';

import { Icon } from '@/components/icon';
import { JobHistory } from '@/components/job-history';
import { BatchRecoveryNotice, SubmissionRecoveryNotice } from './training-recovery';
import { useTrainingPanel } from './training-state';
import { TrainingDatasetStep } from './training-dataset-step';
import { TrainingModelStep } from './training-model-step';
import { TrainingComputeStep } from './training-compute-step';
import { TrainingJobDetail } from './training-job-detail';
import './training-panel.css';

const steps = ['Dataset', 'Model', 'Compute'];

export function TrainingPanel({
  projectId,
  preferredDatasetId,
  onChooseDataset,
  onDiagnostics,
  onComputeSettings,
  onQuantize,
  onNativeQuantize,
  onNativeDistill,
  startNew,
  showJobsRequest,
  preferredRunId,
  preferredCheckpointId,
  active = true,
}: {
  projectId: string;
  preferredDatasetId?: string;
  onChooseDataset: () => void;
  onDiagnostics: () => void;
  onComputeSettings: () => void;
  onQuantize?: (artifactId: string) => void;
  onNativeQuantize?: (artifactId: string) => void;
  onNativeDistill?: (artifactId: string) => void;
  startNew?: { id: number; datasetId?: string };
  showJobsRequest?: number;
  preferredRunId?: string;
  preferredCheckpointId?: string;
  active?: boolean;
}) {
  const panel = useTrainingPanel({ projectId, preferredDatasetId, startNew, showJobsRequest, preferredRunId, active });
  const {
    artifacts,
    attempt,
    batch,
    blocked,
    busy,
    chooseRun,
    historyEntries,
    historyReviewed,
    jobs,
    keyedAttempt,
    mutation,
    openNew,
    options,
    recoverSubmission,
    reviewHistory,
    setHistoryReviewed,
    setStep,
    showHistory,
    step,
    stepComplete,
    submission,
    view,
  } = panel;

  return (
    <>
      <BatchRecoveryNotice batch={batch} onOpenJob={chooseRun} />
      <SubmissionRecoveryNotice
        submission={submission}
        keyed={!!keyedAttempt}
        historyReviewed={historyReviewed === attempt}
        historyRefreshing={jobs.isFetching}
        historyError={jobs.error}
        submitting={mutation.isPending}
        onCheck={() => void recoverSubmission()}
        onRetry={() => void recoverSubmission(true)}
        onRefreshHistory={() => {
          showHistory();
          void reviewHistory();
        }}
        onAllowNew={() => {
          if (submission.clearLegacy(attempt)) {
            setHistoryReviewed(null);
            mutation.reset();
          }
        }}
        onOpenJob={chooseRun}
      />
      {view === 'jobs' && (
        <JobHistory
          title="Fine-tuning jobs"
          newLabel="Start a new fine-tuning"
          entries={historyEntries}
          onNew={() => openNew()}
          onSelect={chooseRun}
          loading={!!projectId && jobs.isPending}
          error={jobs.error}
          emptyMessage="No fine-tuning jobs yet."
          disabled={!projectId || mutation.isPending}
        />
      )}
      {view === 'new' && (
        <>
          <div className="training-view-toolbar">
            <button type="button" className="text-button training-back" onClick={showHistory}>
              <Icon name="arrow" size={14} />
              Back to jobs
            </button>
          </div>
          <form
            className="training-workspace"
            onSubmit={(event) => {
              event.preventDefault();
              if (step === 2 && !blocked && !busy) mutation.mutate();
            }}
          >
            <nav className="training-nav" aria-label="Training setup">
              {steps.map((label, index) => (
                <button
                  type="button"
                  key={label}
                  aria-label={label}
                  aria-current={step === index ? 'step' : undefined}
                  onClick={() => setStep(index)}
                >
                  <span>{index < step && stepComplete[index] ? <Icon name="check" size={13} /> : index + 1}</span>
                  {label}
                </button>
              ))}
            </nav>

            {step === 0 && <TrainingDatasetStep panel={panel} onChooseDataset={onChooseDataset} />}

            {step === 1 && <TrainingModelStep panel={panel} />}

            {step === 2 && <TrainingComputeStep panel={panel} onComputeSettings={onComputeSettings} />}

            {submission.error && submission.available && !attempt && (
              <p className="error-notice" role="alert">
                {submission.error}
              </p>
            )}
            {[options.error, jobs.error, artifacts.error, mutation.error].filter(Boolean).map((error, index) => (
              <p className="error-notice" role="alert" key={index}>
                {error?.message}
              </p>
            ))}
            {(options.isError || jobs.isError) && (
              <button
                type="button"
                className="secondary-button"
                onClick={() => {
                  void options.refetch();
                  void jobs.refetch();
                }}
              >
                Retry
              </button>
            )}
          </form>
        </>
      )}

      {view === 'run' && (
        <TrainingJobDetail
          panel={panel}
          preferredRunId={preferredRunId}
          preferredCheckpointId={preferredCheckpointId}
          onDiagnostics={onDiagnostics}
          onQuantize={onQuantize}
          onNativeQuantize={onNativeQuantize}
          onNativeDistill={onNativeDistill}
        />
      )}
    </>
  );
}
