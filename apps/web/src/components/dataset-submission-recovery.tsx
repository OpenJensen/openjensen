'use client';

import type { Job } from '@/lib/api';
import { reviewedIntakeHistory } from '@/lib/dataset-submission';
import { useRef, useState } from 'react';
import { submissionOf, type useDurableSubmission } from '@/lib/durable-submission';
import type { PolicyJobAttempt } from '@/lib/policy-job-attempt';
import './dataset-submission-recovery.css';

type Props = {
  submission: ReturnType<typeof useDurableSubmission>;
  onReconcile: () => Promise<void>;
  onRetry: () => Promise<void>;
  onReviewHistory: () => Promise<boolean | Job[]>;
};

/** Only an explicit history review can release an old, unkeyed recovery record. */
export function DatasetSubmissionRecovery({ submission, onReconcile, onRetry, onReviewHistory }: Props) {
  const [reviewed, setReviewed] = useState<PolicyJobAttempt>(null);
  const [reviewing, setReviewing] = useState(false);
  const [reviewError, setReviewError] = useState('');
  const reviewBusy = useRef(false), reviewedHistory = useRef<Job[] | undefined>(undefined);
  const latestAttempt = useRef(submission.attempt);
  latestAttempt.current = submission.attempt;
  const identity = submissionOf(submission.attempt);
  async function review() {
    if (reviewBusy.current || submission.busy) return;
    const expected = submission.attempt;
    reviewBusy.current = true; reviewedHistory.current = undefined; setReviewed(null); setReviewing(true); setReviewError('');
    try {
      const result = await onReviewHistory();
      if (result && latestAttempt.current === expected) {
        if (Array.isArray(result)) {
          const project = result[0]?.project_id;
          // Empty history is also a valid freshly fetched response. The controller
          // validates its actual scope again at the explicit acknowledgment.
          reviewedHistory.current = project ? reviewedIntakeHistory(result, project) : result;
        }
        setReviewed(expected);
      }
      else setReviewError('Job history could not be verified. Keep this request unresolved.');
    } catch (error) { setReviewError(error instanceof Error ? error.message : 'Job history could not be verified.'); }
    finally { reviewBusy.current = false; setReviewing(false); }
  }
  if (!submission.attempt) return submission.error ? <p className="error-notice" role="alert">{submission.error}</p> : null;
  return <aside className="dataset-submission-recovery" aria-label="Request recovery">
    <div className="dataset-submission-heading"><strong>{submission.busy ? 'Checking your saved request' : submission.receipt ? 'Job accepted; recovery needs attention' : 'Request needs verification'}</strong></div>
    <p role="status">{submission.error || submission.attempt?.message}</p>
    {submission.receipt && <p>Accepted job: <code>{submission.receipt.id}</code>. Check its current state in the project history.</p>}
    {identity ? <>
      <p>A retry uses the saved source and settings, even if you have edited the form. It reuses the saved request ID so the application can return an existing job instead of creating a duplicate.</p>
      <details><summary>Saved request</summary><dl><div><dt>Request ID</dt><dd><code>{identity.key}</code></dd></div><div><dt>Operation</dt><dd>{identity.operation}</dd></div></dl><pre>{JSON.stringify(identity.body, null, 2)}</pre></details>
      <div className="dataset-submission-actions">
        <button type="button" className="secondary-button" disabled={submission.busy || !submission.available} onClick={() => void onReconcile()}>Check saved request</button>
        {submission.canRetry && <button type="button" className="secondary-button" disabled={submission.busy} onClick={() => void onRetry()}>Retry same request</button>}
      </div>
    </> : submission.attempt && <>
      <p>This older or unreadable recovery record has no verified request ID. Review the project’s jobs before allowing another submission.</p>
      <div className="dataset-submission-actions">
        <button type="button" className="secondary-button" disabled={reviewing || submission.busy || !submission.available} onClick={() => void review()}>{reviewing ? 'Checking history…' : 'Refresh job history'}</button>
        <button type="button" className="secondary-button" disabled={!reviewed || reviewed !== submission.attempt || reviewing || submission.busy || !submission.available} onClick={() => { if (submission.clearLegacy(reviewed, reviewedHistory.current)) setReviewed(null); }}>I reviewed the jobs; allow a new request</button>
      </div>
    </>}
    {reviewError && <p className="error-notice" role="alert">{reviewError}</p>}
  </aside>;
}
