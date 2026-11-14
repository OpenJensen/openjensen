'use client';

import { jobStatusLabel } from '@/lib/job-status';
import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api, isActive } from '@/lib/api';
import { simulationTarget, simulationTaskSummary } from '@/lib/native-simulation';
import { runLabel, runSummary } from '@/lib/run-summary';
import { ChoiceCards } from './ui/choice';

function reported(value: boolean | null | undefined) {
  return value == null ? 'Not reported' : value ? 'Yes' : 'No';
}

export function CloudRuns({
  projectId,
  onOpenTraining,
  onOpenSimulation,
}: {
  projectId: string;
  onOpenTraining: (id: string) => void;
  onOpenSimulation: (id: string) => void;
}) {
  const jobs = useQuery({
    queryKey: ['jobs', projectId],
    queryFn: () => api.jobs(projectId),
    enabled: !!projectId,
    refetchInterval: 3_000,
    retry: false,
  });
  // The saved execution target is evidence of where this job ran. Current
  // runtime settings and a remote input artifact do not establish cloud execution.
  const managed = (jobs.data ?? [])
    .filter((job) => job.project_id === projectId && (job.compute_target != null || simulationTarget(job) !== null))
    .sort((a, b) => b.created_at.localeCompare(a.created_at));
  const [jobId, setJobId] = useState('');
  const job = managed.find((item) => item.id === jobId) ?? managed[0];
  // A terminal transition must fetch final events even when the active poll was empty.
  // Job status has a fixed six-value domain; timestamps would grow this cache unboundedly.
  const events = useQuery({
    queryKey: ['events', job?.id, job?.status],
    queryFn: () => api.events(job!.id),
    enabled: !!job,
    refetchInterval: job && isActive(job) ? 3_000 : false,
    retry: false,
  });
  const feed = useQuery({
    queryKey: ['cloud-runs'],
    queryFn: api.cloudRuns,
    refetchInterval: 3_000,
    staleTime: 0,
    retry: false,
  });
  // The same message often arrives many times in a row; one line says so, newest first.
  const steps: { key: string; stage: string; message: string; detail: string; last: string; count: number }[] = [];
  for (const event of (events.data ?? []).slice(-100)) {
    const detail = simulationTaskSummary(event.data) ?? '';
    const previous = steps[0];
    if (
      previous &&
      previous.stage === event.stage &&
      previous.message === event.message &&
      previous.detail === detail
    ) {
      previous.last = event.timestamp;
      previous.count += 1;
    } else
      steps.unshift({
        key: `${event.timestamp}:${steps.length}`,
        stage: event.stage,
        message: event.message,
        detail,
        last: event.timestamp,
        count: 1,
      });
  }
  const [runId, setRunId] = useState('');
  const [stream, setStream] = useState<'isaac' | 'vla'>('isaac');
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1_000);
    return () => clearInterval(timer);
  }, []);
  const runs = feed.data?.runs ?? [];
  const errors = feed.data?.errors ?? [];
  const selected = runs.find((run) => run.run_id === runId) ?? runs[0];
  // Age the last response locally too, so a disconnected browser cannot keep
  // displaying "Current" forever. Do not depend on agreement between host clocks.
  const elapsed = Math.max(0, (now - feed.dataUpdatedAt) / 1_000);
  const age = selected?.age_seconds == null ? null : selected.age_seconds + elapsed;
  const stale = feed.isError || selected?.stale || age === null || age > (feed.data?.stale_after_seconds ?? 90);
  const streamName = stream === 'isaac' ? 'Isaac' : 'VLA';

  return (
    <div className="cloud-runs">
      <section className="panel managed-cloud-jobs" aria-labelledby="managed-cloud-title">
        <div className="cloud-heading">
          <div>
            <h2 id="managed-cloud-title">Jobs</h2>
          </div>
          <button
            className="secondary-button"
            disabled={!projectId || jobs.isFetching}
            onClick={() => {
              void jobs.refetch();
              if (job) void events.refetch();
            }}
          >
            Refresh application jobs
          </button>
        </div>
        {!projectId && <p role="status">Select a project to see its cloud jobs.</p>}
        {projectId && jobs.isPending && <p role="status">Loading application cloud jobs…</p>}
        {jobs.isError && (
          <p className="error-notice" role="alert">
            Application job updates are unavailable. {jobs.error.message} Previously received jobs may be out of date.
          </p>
        )}
        {projectId && jobs.isSuccess && !managed.length && <p>No application cloud jobs in this project yet.</p>}
        {job && (
          <>
            <ChoiceCards
              className="cloud-run-cards"
              label="Application cloud job"
              value={job.id}
              onChange={setJobId}
              options={managed.map((item) => ({
                value: item.id,
                label: runLabel(item),
                detail: `${new Date(item.created_at).toLocaleString()} · ${item.id.slice(0, 8)}`,
                badge: jobStatusLabel(item.status),
              }))}
            />
            <h3>{runSummary(job)}</h3>
            <dl className="cloud-run-facts">
              <div>
                <dt>Job ID</dt>
                <dd>{job.id}</dd>
              </div>
              <div>
                <dt>Recorded status</dt>
                <dd>{jobStatusLabel(job.status)}</dd>
              </div>
              <div>
                <dt>Execution target</dt>
                <dd>
                  {simulationTarget(job)
                    ? `${simulationTarget(job)!.accelerators.join(' + ')} · ${simulationTarget(job)!.profile_id}`
                    : `${job.compute_target?.accelerator} · ${job.compute_target?.region}`}{' '}
                </dd>
              </div>
              <div>
                <dt>Last job update</dt>
                <dd>
                  <time dateTime={job.updated_at}>{new Date(job.updated_at).toLocaleString()}</time>
                </dd>
              </div>
            </dl>
            {job.error && (
              <p className="error-notice" role="alert">
                {job.error}
              </p>
            )}
            {simulationTarget(job) && (
              <button className="secondary-button" disabled={jobs.isError} onClick={() => onOpenSimulation(job.id)}>
                Open simulation job
              </button>
            )}
            {job.kind === 'policy.finetune' && (
              <button className="secondary-button" disabled={jobs.isError} onClick={() => onOpenTraining(job.id)}>
                Open training job
              </button>
            )}
            <h3>What happened</h3>
            {events.isPending && <p role="status">Loading job events…</p>}
            {events.isError && (
              <p role="alert">
                Job events are unavailable. {events.error.message} Previously received events may be out of date.
              </p>
            )}
            <div className="cloud-events" role="region" aria-label="Application job event log" tabIndex={0}>
              {steps.length ? (
                <ol>
                  {steps.map((step) => (
                    <li key={step.key}>
                      <time dateTime={step.last}>{new Date(step.last).toLocaleTimeString()}</time>
                      <span>
                        {step.message}
                        {step.detail ? ` · ${step.detail}` : ''}
                      </span>
                      <small>
                        {step.stage}
                        {step.count > 1 ? ` · reported ${step.count} times` : ''}
                      </small>
                    </li>
                  ))}
                </ol>
              ) : (
                <p>No events received for this job yet.</p>
              )}
            </div>
            <p className="field-help">
              Newest first, from the latest 100 events. A finished job does not prove the task succeeded or that its
              cloud machine was removed.
            </p>
          </>
        )}
      </section>
      <section className="panel external-cloud-observations" aria-labelledby="cloud-runs-title">
        <div className="cloud-heading">
          <div>
            <h2 id="cloud-runs-title">External observations</h2>
          </div>
          <button className="secondary-button" onClick={() => void feed.refetch()} disabled={feed.isFetching}>
            {feed.isFetching ? 'Checking…' : 'Refresh cloud runs'}
          </button>
        </div>
        {feed.isPending && <p role="status">Loading cloud runs…</p>}
        {feed.isError && (
          <p className="error-notice" role="alert">
            Cloud updates are unavailable. {feed.error.message}
            {feed.data && ' Previously received information remains below.'}
          </p>
        )}
        {feed.data && !feed.data.enabled && (
          <p className="warning-box" role="status">
            External rollout monitoring is not configured.
          </p>
        )}
        {errors.map((error, index) => (
          <p className="error-notice" role="alert" key={`${error.run_id}-${index}`}>
            Snapshot read error{error.run_id ? ` for ${error.run_id}` : ''}: {error.message}
          </p>
        ))}
        {feed.data?.enabled && runs.length === 0 && errors.length === 0 && (
          <p>No external rollout observations have been published yet.</p>
        )}
        {selected && (
          <>
            <ChoiceCards
              className="cloud-run-cards"
              label="Cloud run"
              value={selected.run_id}
              onChange={setRunId}
              options={runs.map((run) => ({
                value: run.run_id,
                label: run.label,
                detail: run.run_id,
                badge: run.status,
              }))}
            />
            <div className="cloud-freshness" role="status">
              <strong>
                {stale
                  ? 'Stale or unavailable observation'
                  : selected.collection_error
                    ? 'Last observation (refresh failed)'
                    : 'Current observation'}
              </strong>
              <span>
                {selected.collected_at ? (
                  <>
                    Last successful collection:{' '}
                    <time dateTime={selected.collected_at}>{new Date(selected.collected_at).toLocaleString()}</time>
                    {age !== null && ` · ${Math.floor(age)}s ago`}
                  </>
                ) : (
                  'No successful remote collection yet.'
                )}
              </span>
              <span>Stale after {feed.data?.stale_after_seconds ?? 90}s · historical snapshot</span>
            </div>
            {selected.collection_error && (
              <p className="error-notice" role="alert">
                Remote collection error: {selected.collection_error}
              </p>
            )}
            <dl className="cloud-run-facts">
              <div>
                <dt>Job group</dt>
                <dd>{selected.cluster}</dd>
              </div>
              <div>
                <dt>SkyPilot job ID</dt>
                <dd>{selected.job_id ?? 'Not reported'}</dd>
              </div>
              <div>
                <dt>Primary task status</dt>
                <dd>{selected.status}</dd>
              </div>
              <div>
                <dt>Rollout completed</dt>
                <dd>{reported(selected.outcomes.rollout_completed)}</dd>
              </div>
              <div>
                <dt>Pickup success</dt>
                <dd>{reported(selected.outcomes.pickup_success)}</dd>
              </div>
              <div>
                <dt>Calibration</dt>
                <dd>{selected.outcomes.calibration ?? 'unknown'}</dd>
              </div>
            </dl>
            <p className="cloud-outcome-note">
              Observer-reported outcomes · logs do not verify pickup success or calibration.
            </p>
            <div className="cloud-log-switch" role="group" aria-label="Log stream">
              <button className="secondary-button" aria-pressed={stream === 'isaac'} onClick={() => setStream('isaac')}>
                Isaac logs
              </button>
              <button className="secondary-button" aria-pressed={stream === 'vla'} onClick={() => setStream('vla')}>
                VLA logs
              </button>
            </div>
            <pre className="cloud-log-tail" role="region" aria-label={`${streamName} log tail`} tabIndex={0}>
              {selected.logs[stream] || 'No log lines collected for this stream yet.'}
            </pre>
          </>
        )}
      </section>
    </div>
  );
}
