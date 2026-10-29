'use client';

import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { workJobs, type WorkJob, type WorkLogRecord } from '@/lib/work-jobs';
import { Notice } from './ui/notice';

const maxVisible = 2000;

export function WorkJobLogs({ job }: { job: WorkJob }) {
  const [after, setAfter] = useState(0);
  const [records, setRecords] = useState<WorkLogRecord[]>([]);
  const [severity, setSeverity] = useState('');
  const [stream, setStream] = useState('');
  const [search, setSearch] = useState('');
  const [query, setQuery] = useState('');
  const [follow, setFollow] = useState(true);
  const [limited, setLimited] = useState(false);
  const logs = useQuery({
    queryKey: ['work-job-logs', job.project_id, job.id, after, severity, stream, query],
    queryFn: ({ signal }) => workJobs.logs(job.project_id, job.id, after, { severity, stream, query }, signal),
    retry: false,
    refetchInterval: follow ? 2000 : false,
  });
  useEffect(() => {
    const page = logs.data;
    if (!page) return;
    setRecords((previous) => {
      const bySequence = new Map(previous.map((record) => [record.sequence, record]));
      for (const record of page.records) bySequence.set(record.sequence, record);
      const merged = [...bySequence.values()].sort((a, b) => a.sequence - b.sequence);
      return merged.slice(-maxVisible);
    });
    if (page.truncated || (page.records.length > 0 && after >= maxVisible)) setLimited(true);
    if (follow && page.next_cursor > after) setAfter(page.next_cursor);
  }, [logs.data, after, follow]);
  function reset(next: { severity?: string; stream?: string; query?: string }) {
    setAfter(0);
    setRecords([]);
    setLimited(false);
    if (next.severity !== undefined) setSeverity(next.severity);
    if (next.stream !== undefined) setStream(next.stream);
    if (next.query !== undefined) setQuery(next.query);
  }
  return (
    <section className="panel work-job-logs" aria-label="Job logs">
      <div className="work-job-title">
        <h3>Logs</h3>
        <a className="text-link" href={workJobs.diagnosticsUrl(job.project_id, job.id)} download>
          Download diagnostics
        </a>
      </div>
      <form
        className="work-job-filters"
        onSubmit={(event) => {
          event.preventDefault();
          reset({ query: search });
        }}
      >
        <label>
          Search logs
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            maxLength={200}
            placeholder="Search retained messages"
          />
        </label>
        <label>
          Severity
          <select value={severity} onChange={(event) => reset({ severity: event.target.value })}>
            <option value="">All severities</option>
            <option value="error">Error</option>
            <option value="warning">Warning</option>
            <option value="info">Info</option>
            <option value="debug">Debug</option>
          </select>
        </label>
        <label>
          Output
          <select value={stream} onChange={(event) => reset({ stream: event.target.value })}>
            <option value="">All output</option>
            <option value="stdout">Standard output</option>
            <option value="stderr">Standard error</option>
          </select>
        </label>
        <button type="submit" className="secondary-button">
          Search
        </button>
      </form>
      <div className="workbench-actions work-log-actions">
        <label className="work-log-follow">
          <input type="checkbox" checked={follow} onChange={(event) => setFollow(event.target.checked)} /> Follow live
          logs
        </label>
        <button className="text-button" onClick={() => void logs.refetch()} disabled={logs.isFetching}>
          Refresh logs
        </button>
      </div>
      {logs.error && <Notice tone="danger" title={logs.error.message} live="alert" />}
      {logs.isPending && <p role="status">Loading logs…</p>}
      {logs.data && !logs.data.available && (
        <p>No structured logs were retained for this job. Its timeline and saved results remain available.</p>
      )}
      {logs.data?.available && !records.length && !logs.isPending && <p>No log messages match these filters.</p>}
      {limited && (
        <p className="work-log-note">
          Log history was limited or rotated. This view retains up to {maxVisible.toLocaleString()} recent messages.
        </p>
      )}
      {records.length > 0 && (
        <ol className="work-log-list" aria-label="Log messages">
          {records.map((record) => (
            <li key={record.sequence} className={`work-log-row work-log-${record.severity}`}>
              <div className="work-log-meta">
                <time dateTime={record.timestamp}>{new Date(record.timestamp).toLocaleString()}</time>
                <span>{record.severity}</span>
                <span>
                  {record.component}
                  {record.stage ? ` · ${record.stage}` : ''}
                </span>
                {record.stream && <span>{record.stream}</span>}
                <span>{record.attempt_id}</span>
              </div>
              <pre>{record.message}</pre>
            </li>
          ))}
        </ol>
      )}
      {!follow && logs.data?.has_more && (
        <button className="secondary-button" onClick={() => setAfter(logs.data!.next_cursor)}>
          Load more logs
        </button>
      )}
    </section>
  );
}
