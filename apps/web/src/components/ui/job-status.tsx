import { jobStatusLabel } from '@/lib/job-status';

/** A job's state as a coloured pill. `label` overrides the word for a page-specific state. */
export function JobStatus({ status, label }: { status: string; label?: string }) {
  return (
    <span className={`status status-${status}`}>
      <span className="status-dot" />
      {label ?? jobStatusLabel(status)}
    </span>
  );
}
