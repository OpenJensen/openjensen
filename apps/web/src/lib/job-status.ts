/**
 * The one set of words for a job's state. Every page that shows a job uses these,
 * so a finished run is "Completed" on Jobs, on Train and on its own page alike.
 */
export const jobStatusLabels: Record<string, string> = {
  queued: 'Queued',
  preparing: 'Preparing',
  running: 'Running',
  cancelling: 'Cancelling',
  waiting: 'Needs attention',
  succeeded: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
  interrupted: 'Interrupted',
};

export const jobStatusLabel = (status: string) => jobStatusLabels[status] ?? status.replaceAll('_', ' ');

/** The same word for use inside a sentence or a compact row: "completed", "needs attention". */
export const jobStatusWord = (status: string) => jobStatusLabel(status).toLowerCase();
