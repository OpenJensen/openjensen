import { trainingRunModelLabel } from '@/lib/checkpoints';
import { Icon } from '@/components/icon';
import { TrainingMonitor } from '@/components/training-monitor';
import type { TrainingPanelState } from './training-state';

/** One fine-tuning job: its monitor, cancel, export and resume actions. */
export function TrainingJobDetail({
  panel,
  preferredRunId,
  preferredCheckpointId,
  onDiagnostics,
  onQuantize,
  onNativeQuantize,
  onNativeDistill,
}: {
  panel: TrainingPanelState;
  preferredRunId?: string;
  preferredCheckpointId?: string;
  onDiagnostics: () => void;
  onQuantize?: (artifactId: string) => void;
  onNativeQuantize?: (artifactId: string) => void;
  onNativeDistill?: (artifactId: string) => void;
}) {
  const {
    active,
    artifacts,
    cancel,
    headingRef,
    jobs,
    models,
    options,
    projectId,
    resumeOptions,
    resumeRun,
    selectedRun,
    showHistory,
    trainingRuns,
  } = panel;
  return (
    <section className="training-activity training-job-detail" aria-label="Fine-tuning job">
      <button type="button" className="text-button training-back" onClick={showHistory}>
        <Icon name="arrow" size={14} />
        Back to jobs
      </button>
      <div className="training-heading">
        <h2 ref={headingRef} tabIndex={-1}>
          {selectedRun
            ? (trainingRunModelLabel(selectedRun, models, artifacts.data ?? [], trainingRuns) ?? 'Fine-tuning job')
            : 'Fine-tuning job'}
        </h2>
        <button type="button" className="text-button" onClick={onDiagnostics}>
          Diagnostics <Icon name="arrow" size={14} />
        </button>
      </div>
      {selectedRun && (
        <TrainingMonitor
          key={selectedRun.id}
          run={selectedRun}
          projectId={projectId}
          active={active}
          artifacts={(artifacts.data ?? []).filter((item) => item.job_id === selectedRun.id)}
          preferredCheckpointId={selectedRun.id === preferredRunId ? preferredCheckpointId : undefined}
          modelCatalog={models}
          exportRuntimes={options.isSuccess ? options.data.runtimes : []}
          exportJobs={jobs.data ?? []}
          exportArtifacts={artifacts.data ?? []}
          onCancel={() => cancel.mutate(selectedRun.id)}
          cancelling={cancel.isPending && cancel.variables === selectedRun.id}
          cancelError={cancel.variables === selectedRun.id ? cancel.error : null}
          onQuantize={onQuantize}
          onNativeQuantize={onNativeQuantize}
          onNativeDistill={onNativeDistill}
          onResume={
            resumeOptions.some((item) => item.jobId === selectedRun.id)
              ? (checkpointId) => resumeRun(selectedRun.id, checkpointId)
              : undefined
          }
        />
      )}
      {!selectedRun && <p className="muted">This job is no longer available.</p>}
    </section>
  );
}
