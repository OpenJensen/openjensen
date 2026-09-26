'use client';

import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';

function reported(value: boolean | null | undefined) {
  return value == null ? 'Not reported' : value ? 'Yes' : 'No';
}

export function CloudRuns() {
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

  return <section className="panel cloud-runs" aria-labelledby="cloud-runs-title">
    <div className="cloud-heading">
      <div><h2 id="cloud-runs-title">Cloud run monitor</h2><p>Read-only status and recent logs from the connected monitor. Updates every 3 seconds.</p></div>
      <button className="secondary-button" onClick={() => void feed.refetch()} disabled={feed.isFetching}>{feed.isFetching ? 'Checking…' : 'Refresh cloud runs'}</button>
    </div>
    {feed.isPending && <p role="status">Loading cloud runs…</p>}
    {feed.isError && <p className="error-notice" role="alert">Cloud updates are unavailable. {feed.error.message}{feed.data && ' Previously received information remains below.'}</p>}
    {feed.data && !feed.data.enabled && <p className="warning-box" role="status">Cloud monitoring is not configured. The application operator can connect a read-only snapshot feed.</p>}
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
  </section>;
}
