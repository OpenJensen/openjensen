import { isProjectDataset, type LibraryDataset } from '@/lib/dataset-library';
import { runName } from '@/lib/run-names';
import { Icon } from '@/components/icon';
import { TrainingMonitor } from '@/components/training-monitor';
import type { TrainingPanelState } from './training-state';

/** One training run: its monitor, cancel, export and resume actions. */
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
    library,
    models,
    options,
    projectId,
    resumeOptions,
    resumeRun,
    selectedRun,
    showHistory,
  } = panel;
  return (
    <section className="training-activity training-job-detail" aria-label="Training run">
      <button type="button" className="text-button training-back" onClick={showHistory}>
        <Icon name="arrow" size={14} />
        Back to jobs
      </button>
      <div className="training-heading">
        <h2 ref={headingRef} tabIndex={-1}>
          {selectedRun
            ? runName(
                selectedRun,
                {
                  jobs: jobs.data ?? [],
                  artifacts: artifacts.data ?? [],
                  library: (library.data ?? []).filter((entry): entry is LibraryDataset =>
                    isProjectDataset(entry, projectId),
                  ),
                  catalog: models,
                },
                { date: true },
              )
            : 'Training run'}
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
