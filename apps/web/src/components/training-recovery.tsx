import type { useDurableSubmission } from '@/lib/durable-submission';
import type { useTrainingBatch } from '@/lib/training-batch';

type TrainingBatch = ReturnType<typeof useTrainingBatch>;
type DurableSubmission = ReturnType<typeof useDurableSubmission>;

/** Lists each job of a saved multi-model plan until every entry is resolved. */
export function BatchRecoveryNotice({ batch, onOpenJob }: { batch: TrainingBatch; onOpenJob: (id: string) => void }) {
  if (!batch.journal && !batch.error) return null;
  return (
    <section className="warning-box training-recovery" aria-label="Multi-model submission recovery">
      <h2>Independent training jobs</h2>
      {batch.error && <p role="alert">{batch.error}</p>}
      {batch.journal && (
        <>
          <ul>
            {batch.journal.entries.map((entry) => (
              <li key={entry.key}>
                {String(entry.body.training?.model_id)} · {entry.state}
                {entry.job && (
                  <button type="button" className="text-button" onClick={() => onOpenJob(entry.job!.id)}>
                    Open job
                  </button>
                )}
              </li>
            ))}
          </ul>
          <div className="training-recovery-actions">
            <button type="button" className="secondary-button" disabled={batch.busy} onClick={() => void batch.check()}>
              Check saved jobs
            </button>
            <button
              type="button"
              className="secondary-button"
              disabled={batch.busy || !batch.canContinue}
              onClick={() => void batch.continue()}
            >
              Continue saved plan
            </button>
            <button
              type="button"
              className="text-button"
              disabled={batch.busy || batch.journal.entries.some((item) => ['pending', 'unknown'].includes(item.state))}
              onClick={batch.dismiss}
            >
              Dismiss unsubmitted recipes
            </button>
          </div>
        </>
      )}
    </section>
  );
}

/** Shows a pending or uncertain single training request and the safe ways to resolve it. */
export function SubmissionRecoveryNotice({
  submission,
  keyed,
  historyReviewed,
  historyRefreshing,
  historyError,
  submitting,
  onCheck,
  onRetry,
  onRefreshHistory,
  onAllowNew,
  onOpenJob,
}: {
  submission: DurableSubmission;
  /** The request carries a server idempotency key, so it can be checked or retried exactly. */
  keyed: boolean;
  historyReviewed: boolean;
  historyRefreshing: boolean;
  historyError: Error | null;
  submitting: boolean;
  onCheck: () => void;
  onRetry: () => void;
  onRefreshHistory: () => void;
  onAllowNew: () => void;
  onOpenJob: (id: string) => void;
}) {
  const attempt = submission.attempt;
  if (!attempt && (submission.available || !submission.error)) return null;
  return (
    <section className="warning-box training-recovery" aria-label="Training submission recovery">
      <h2>{attempt?.state === 'pending' && !submission.error ? 'Starting training' : 'Check your training request'}</h2>
      <p role={attempt?.state === 'pending' ? 'status' : 'alert'}>{submission.error || attempt?.message}</p>
      {attempt?.state === 'uncertain' && (
        <>
          <p>New submissions are paused until this request is resolved.</p>
          {keyed ? (
            <>
              <p>Checking the saved request does not start training.</p>
              <div className="training-recovery-actions">
                <button
                  type="button"
                  className="secondary-button"
                  disabled={!submission.available || submission.busy}
                  onClick={onCheck}
                >
                  Check saved request
                </button>
                <button
                  type="button"
                  className="secondary-button"
                  disabled={!submission.canRetry || submission.busy}
                  onClick={onRetry}
                >
                  Retry same request
                </button>
              </div>
              <p className="training-recovery-note">
                Retry resends the saved recipe only if the server has no matching job.
              </p>
            </>
          ) : (
            <>
              <p>This older request has no server key. Check recorded jobs before allowing a new request.</p>
              <div className="training-recovery-actions">
                <button
                  type="button"
                  className="secondary-button"
                  disabled={historyRefreshing}
                  onClick={onRefreshHistory}
                >
                  Refresh training jobs
                </button>
                <button
                  type="button"
                  className="secondary-button"
                  disabled={!submission.available || !historyReviewed || !!historyError || submitting}
                  onClick={onAllowNew}
                >
                  I checked the jobs; allow a new request
                </button>
              </div>
              {historyError && <p role="alert">Training history could not refresh. {historyError.message}</p>}
            </>
          )}
        </>
      )}
      {submission.receipt && (
        <div className="training-recovery-actions">
          <button type="button" className="secondary-button" onClick={() => onOpenJob(submission.receipt!.id)}>
            View saved request {submission.receipt.id}
          </button>
        </div>
      )}
    </section>
  );
}
