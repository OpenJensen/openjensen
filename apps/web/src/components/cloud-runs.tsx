'use client';

import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api, isActive } from '@/lib/api';
import { runLabel, runSummary } from '@/lib/run-summary';

function reported(value: boolean | null | undefined) {
  return value == null ? 'Not reported' : value ? 'Yes' : 'No';
}

function ApplicationCloudJobs({ projectId, onOpenTraining }: { projectId: string; onOpenTraining: (id: string) => void }) {
  const jobs = useQuery({ queryKey: ['jobs', projectId], queryFn: () => api.jobs(projectId), enabled: !!projectId, refetchInterval: 3_000, retry: false });
  const [jobId, setJobId] = useState('');
  const managed = (jobs.data ?? []).filter(job => job.project_id === projectId && job.compute_target != null).sort((a, b) => b.created_at.localeCompare(a.created_at));
  const job = managed.find(item => item.id === jobId) ?? managed[0];
  const events = useQuery({ queryKey: ['events', job?.id, job?.status], queryFn: () => api.events(job!.id), enabled: !!job, refetchInterval: job && isActive(job) ? 3_000 : false, retry: false });
  return <section className="panel managed-cloud-jobs" aria-labelledby="managed-cloud-title">
    <div className="cloud-heading"><div><h2 id="managed-cloud-title">Application cloud jobs</h2><p>Recorded GCP jobs for this project.</p></div><button className="secondary-button" disabled={!projectId || jobs.isFetching} onClick={() => { void jobs.refetch(); if (job) void events.refetch(); }}>Refresh application jobs</button></div>
    {!projectId && <p role="status">Select a project to see its cloud jobs.</p>}
    {projectId && jobs.isPending && <p role="status">Loading cloud jobs…</p>}
    {jobs.isError && <p className="error-notice" role="alert">Cloud job updates are unavailable. {jobs.error.message}</p>}
    {projectId && jobs.isSuccess && !managed.length && <p>No application cloud jobs in this project yet.</p>}
    {job && <>
      <div className="cloud-run-picker"><label htmlFor="managed-cloud-job">Application cloud job</label><select id="managed-cloud-job" value={job.id} onChange={event => setJobId(event.target.value)}>{managed.map(item => <option key={item.id} value={item.id}>{runLabel(item)} · {item.id.slice(0, 8)} · {item.status}</option>)}</select></div>
      <h3>{runSummary(job)}</h3>
      <dl className="cloud-run-facts"><div><dt>Recorded status</dt><dd>{job.status}</dd></div><div><dt>GPU</dt><dd>{job.compute_target?.accelerator}</dd></div><div><dt>Region</dt><dd>{job.compute_target?.region}</dd></div><div><dt>Last update</dt><dd>{new Date(job.updated_at).toLocaleString()}</dd></div></dl>
      {job.error && <p className="error-notice" role="alert">{job.error}</p>}
      {job.kind === 'policy.finetune' && <button className="secondary-button" onClick={() => onOpenTraining(job.id)}>Open training job</button>}
      <h3>Recorded job events</h3>
      {events.isPending && <p role="status">Loading events…</p>}
      {events.isError && <p role="alert">Job events are unavailable. {events.error.message}</p>}
      <pre className="cloud-log-tail" role="region" aria-label="Application job event log" tabIndex={0}>{events.data?.length ? events.data.slice(-100).map(event => `${event.timestamp} · ${event.stage} · ${event.message}`).join('\n') : 'No events recorded yet.'}</pre>
      <p className="field-help">Job status is recorded history; it does not indicate whether a VM is still running.</p>
    </>}
  </section>;
}

export function CloudRuns({ projectId, onOpenTraining }: { projectId: string; onOpenTraining: (id: string) => void }) {
  const feed = useQuery({ queryKey: ['cloud-runs'], queryFn: api.cloudRuns, refetchInterval: 3_000, staleTime: 0, retry: false });
  const [runId, setRunId] = useState('');
  const [stream, setStream] = useState<'isaac' | 'vla'>('isaac');
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1_000);
    return () => clearInterval(timer);
  }, []);
  const runs = feed.data?.runs ?? [];
  const errors = feed.data?.errors ?? [];
  const selected = runs.find(run => run.run_id === runId) ?? runs[0];
  // Age the last response locally too, so a disconnected browser cannot keep
  // displaying "Current" forever. Do not depend on agreement between host clocks.
  const elapsed = Math.max(0, (now - feed.dataUpdatedAt) / 1_000);
  const age = selected?.age_seconds == null ? null : selected.age_seconds + elapsed;
  const stale = feed.isError || selected?.stale || age === null || age > (feed.data?.stale_after_seconds ?? 90);
  const streamName = stream === 'isaac' ? 'Isaac' : 'VLA';

  return <div className="cloud-runs"><ApplicationCloudJobs projectId={projectId} onOpenTraining={onOpenTraining} />
    <details className="panel external-cloud-observations"><summary>External simulator monitor</summary>
    <section aria-labelledby="cloud-runs-title">
    <div className="cloud-heading">
      <div><h2 id="cloud-runs-title">Cloud run monitor</h2><p>Read-only status and recent logs from the connected monitor. Updates every 3 seconds.</p></div>
      <button className="secondary-button" onClick={() => void feed.refetch()} disabled={feed.isFetching}>{feed.isFetching ? 'Checking…' : 'Refresh cloud runs'}</button>
    </div>
    {feed.isPending && <p role="status">Loading cloud runs…</p>}
    {feed.isError && <p className="error-notice" role="alert">Cloud updates are unavailable. {feed.error.message}{feed.data && ' Previously received information remains below.'}</p>}
    {feed.data && !feed.data.enabled && <p className="warning-box" role="status">External simulator monitoring is not configured. Application cloud jobs are available above.</p>}
    {errors.map((error, index) => <p className="error-notice" role="alert" key={`${error.run_id}-${index}`}>Snapshot read error{error.run_id ? ` for ${error.run_id}` : ''}: {error.message}</p>)}
    {feed.data?.enabled && runs.length === 0 && errors.length === 0 && <p>No cloud runs have been published yet.</p>}
    {selected && <>
      <div className="cloud-run-picker"><label htmlFor="cloud-run-select">Cloud run</label>
        <select id="cloud-run-select" value={selected.run_id} onChange={event => setRunId(event.target.value)}>
          {runs.map(run => <option key={run.run_id} value={run.run_id}>{run.label} · {run.run_id}</option>)}
        </select>
      </div>
      <div className="cloud-freshness" role="status">
        <strong>{stale ? 'Stale or unavailable observation' : selected.collection_error ? 'Last observation (refresh failed)' : 'Current observation'}</strong>
        <span>{selected.collected_at ? <>Last successful collection: <time dateTime={selected.collected_at}>{new Date(selected.collected_at).toLocaleString()}</time>{age !== null && ` · ${Math.floor(age)}s ago`}</> : 'No successful remote collection yet.'}</span>
        <span>Observations become stale after {feed.data?.stale_after_seconds ?? 90} seconds.</span>
      </div>
      {selected.collection_error && <p className="error-notice" role="alert">Remote collection error: {selected.collection_error}</p>}
      <dl className="cloud-run-facts">
        <div><dt>Job group</dt><dd>{selected.cluster}</dd></div>
        <div><dt>SkyPilot job ID</dt><dd>{selected.job_id ?? 'Not reported'}</dd></div>
        <div><dt>Primary task status</dt><dd>{selected.status}</dd></div>
        <div><dt>Rollout completed</dt><dd>{reported(selected.outcomes.rollout_completed)}</dd></div>
        <div><dt>Pickup success</dt><dd>{reported(selected.outcomes.pickup_success)}</dd></div>
        <div><dt>Calibration</dt><dd>{selected.outcomes.calibration ?? 'unknown'}</dd></div>
      </dl>
      <p className="cloud-outcome-note">Task status and logs do not establish pickup success, calibration or optimizer validation. Outcomes above are explicit monitor reports.</p>
      <div className="cloud-log-switch" role="group" aria-label="Log stream">
        <button className="secondary-button" aria-pressed={stream === 'isaac'} onClick={() => setStream('isaac')}>Isaac logs</button>
        <button className="secondary-button" aria-pressed={stream === 'vla'} onClick={() => setStream('vla')}>VLA logs</button>
      </div>
      <p className="cloud-log-caption">{streamName} recent log tail</p>
      <pre className="cloud-log-tail" role="region" aria-label={`${streamName} log tail`} tabIndex={0}>{selected.logs[stream] || 'No log lines collected for this stream yet.'}</pre>
    </>}
  </section></details></div>;
}
