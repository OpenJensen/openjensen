"use client";

import { publicDemo } from "@/lib/public-demo";
import type { Job } from "@/lib/api";
import { Icon } from "@/components/icon";
import "./job-history.css";

export type JobHistoryEntry = {
  id: string;
  title: string;
  subtitle?: string;
  status: Job["status"];
  createdAt: string;
  progress?: string;
};

export function JobHistory({ title, newLabel, entries, onNew, onSelect, loading = false, error,
  emptyMessage = "No jobs yet.", disabled = false }: {
  title: string;
  newLabel: string;
  entries: JobHistoryEntry[];
  onNew: () => void;
  onSelect: (id: string) => void;
  loading?: boolean;
  error?: Error | null;
  emptyMessage?: string;
  disabled?: boolean;
}) {
  return <section className="job-history" aria-label={title}>
    <header className="job-history-heading">
      <h2>{title}</h2>
      <button type="button" className="primary-button" disabled={publicDemo || disabled} title={publicDemo ? "Owner access is required to start jobs" : undefined} onClick={onNew}>
        <Icon name="plus" size={15} />{newLabel}
      </button>
    </header>
    {error && <p className="error-notice" role="alert">{error.message}</p>}
    {loading && <p className="job-history-note" role="status">Loading jobs…</p>}
    {!loading && !error && entries.length === 0 && <p className="job-history-empty">{emptyMessage}</p>}
    {entries.length > 0 && <ul className="job-history-list">
      {entries.map(entry => <li key={entry.id}>
        <button type="button" className="job-history-entry" onClick={() => onSelect(entry.id)} data-job-id={entry.id}>
          <span className="job-history-copy">
            <strong>{entry.title}</strong>
            {entry.subtitle && <span className="job-history-subtitle">{entry.subtitle}</span>}
            {entry.progress && <span className="job-history-progress">{entry.progress}</span>}
          </span>
          <span className="job-history-meta">
            <span className={`status status-${entry.status}`}><span className="status-dot" />{entry.status}</span>
            <time dateTime={entry.createdAt}>{new Date(entry.createdAt).toLocaleString()}</time>
          </span>
          <Icon name="arrow" size={16} />
        </button>
      </li>)}
    </ul>}
  </section>;
}
