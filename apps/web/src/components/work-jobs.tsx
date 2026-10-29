'use client';

import { useEffect, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiOrigin } from '@/lib/api';
import { WorkbenchDisclosure } from './workbench-disclosure';
import { Icon } from './icon';
import { JobHistory } from './job-history';
import { WorkJobLogs } from './work-job-logs';
import { workJobs, workStatusLabels, isWorkActive, type WorkJob, type JobOperation } from '@/lib/work-jobs';
import { computeTargets } from '@/lib/compute-targets';
import { selectedJobCompute, saveJobCompute } from '@/lib/job-compute-selection';
import { Notice } from './ui/notice';
import './work-jobs.css';

function NewWorkJob({
  projectId,
  choices,
  onBack,
  onNew,
}: {
  projectId: string;
  choices: JobOperation[];
  onBack: () => void;
  onNew: (operation: JobOperation) => void;
}) {
  const [chosen, setChosen] = useState<JobOperation | null>(null);
  const [targetId, setTargetId] = useState('');
  const [error, setError] = useState('');
  const targets = useQuery({ queryKey: ['compute-targets'], queryFn: () => computeTargets.list(), retry: false });
  const options = (targets.data?.targets ?? []).filter((target) => target.operations.includes(chosen?.id ?? ''));
  const target = options.find((item) => item.definition.id === targetId);
  const configurable = chosen?.id.startsWith('policy.') && chosen.submission === 'job';
  function choose(operation: JobOperation) {
    setChosen(operation);
    setError('');
    try {
      setTargetId(selectedJobCompute(projectId, operation.id)?.id ?? '');
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Compute selection could not be read.');
    }
  }
  function configure() {
    if (!chosen) return;
    try {
      if (configurable) {
        if (targetId && (!target || !target.launchable))
          throw new Error('Review the selected compute target before continuing.');
        saveJobCompute(
          projectId,
          chosen.id,
          target
            ? {
                id: target.definition.id,
                label: target.definition.label,
                runtimeId: target.definition.runtime_id,
                operation: chosen.id,
              }
            : null,
        );
      }
      onNew(chosen);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Compute selection could not be saved.');
    }
  }
  return (
    <section className="panel work-job-picker" aria-label="New job">
      <button className="text-button" onClick={onBack}>
        ← Jobs
      </button>
      <h2>New job</h2>
      <p>Choose an operation, then configure its inputs and settings.</p>
      <div className="work-job-operations" role="group" aria-label="Job operation">
        {choices.map((operation) => (
          <button
            key={operation.id}
            className={chosen?.id === operation.id ? 'primary-button' : 'secondary-button'}
            aria-pressed={chosen?.id === operation.id}
            onClick={() => choose(operation)}
          >
            {operation.title}
            <Icon name="arrow" size={15} />
          </button>
        ))}
      </div>
      {chosen && (
        <div className="work-job-compose">
          {configurable && (
            <>
              <label className="work-job-compute">
                Compute target
                <select
                  value={targetId}
                  onChange={(event) => {
                    setTargetId(event.target.value);
                    setError('');
                  }}
                >
                  <option value="">Use module compute settings</option>
                  {targetId && !target && (
                    <option value={targetId} disabled>
                      Saved target is unavailable
                    </option>
                  )}
                  {options.map((item) => (
                    <option key={item.definition.id} value={item.definition.id} disabled={!item.launchable}>
                      {item.definition.label}
                      {!item.launchable ? ` — ${item.status.replaceAll('_', ' ')}` : ''}
                    </option>
                  ))}
                </select>
              </label>
              {targets.isError && (
                <p role="alert">Compute targets could not be loaded. Retry before using a configured target.</p>
              )}
              {target && <p>{target.message}</p>}
              {!options.length && targets.isSuccess && (
                <p>Add compatible compute targets in Settings, or use the module’s existing compute settings.</p>
              )}
            </>
          )}
          {chosen.submission === 'interactive' && (
            <p>
              This operation continues in an interactive workspace. Required confirmations remain part of that workflow.
            </p>
          )}
          {error && <Notice tone="danger" title={error} live="alert" />}
          <button
            className="primary-button"
            disabled={!!(configurable && targetId && (!target || !target.launchable))}
            onClick={configure}
          >
            Configure {chosen.title.toLowerCase()}
          </button>
        </div>
      )}
    </section>
  );
}

function JobDetail({
  job,
  operations,
  onOpen,
  onSelect,
  onBack,
}: {
  job: WorkJob;
  operations: JobOperation[];
  onOpen: (job: WorkJob) => void;
  onSelect: (id: string) => void;
  onBack: () => void;
}) {
  const client = useQueryClient();
  const [view, setView] = useState<'overview' | 'timeline' | 'logs' | 'details'>('overview');
  const failure = useQuery({
    queryKey: ['work-job-failure', job.project_id, job.id],
    queryFn: ({ signal }) => workJobs.failure(job.project_id, job.id, signal),
    retry: false,
    refetchInterval: isWorkActive(job) ? 2000 : false,
  });
  const cancel = useMutation({
    mutationFn: () => workJobs.cancel(job.project_id, job.id),
    onSuccess: (updated) => {
      client.setQueryData(['work-job', job.project_id, job.id], updated);
      void client.invalidateQueries({ queryKey: ['work-jobs', job.project_id] });
      void client.invalidateQueries({ queryKey: ['jobs', job.project_id] });
    },
  });
  const operation = operations.find((item) => item.id === job.operation);
  return (
    <section className="work-job-detail" aria-label="Job details">
      <button className="text-button work-job-back" onClick={onBack}>
        ← Jobs
      </button>
      <section className="panel">
        <div className="work-job-title">
          <h2>{job.title}</h2>
          <span className={`status status-${job.status}`}>{workStatusLabels[job.status] ?? job.status}</span>
        </div>
        <p className="work-job-stage" role="status">
          {job.message || job.stage?.replaceAll('_', ' ') || workStatusLabels[job.status]}
        </p>
        <div className="workbench-actions">
          <button className="secondary-button" onClick={() => onOpen(job)}>
            Open workspace
          </button>
          {job.can_cancel && (
            <button className="secondary-button" disabled={cancel.isPending} onClick={() => cancel.mutate()}>
              {cancel.isPending ? 'Requesting cancellation…' : 'Cancel job'}
            </button>
          )}
          <a className="text-link" href={workJobs.diagnosticsUrl(job.project_id, job.id)} download>
            Download diagnostics
          </a>
        </div>
        {cancel.error && <Notice tone="danger" title={cancel.error.message} live="alert" />}
      </section>
      <div className="workbench-actions work-job-tabs" role="group" aria-label="Job detail view">
        {(['overview', 'timeline', 'logs', 'details'] as const).map((item) => (
          <button
            key={item}
            type="button"
            className={view === item ? 'primary-button' : 'secondary-button'}
            aria-pressed={view === item}
            onClick={() => setView(item)}
          >
            {item[0].toUpperCase() + item.slice(1)}
          </button>
        ))}
      </div>
      {view === 'overview' && (
        <>
          {failure.data && (
            <section className="panel work-job-failure" aria-label="Failure details">
              <h3>What failed</h3>
              <p>{failure.data.message}</p>
              <dl className="work-job-facts">
                <div>
                  <dt>Stage</dt>
                  <dd>{failure.data.stage}</dd>
                </div>
                <div>
                  <dt>Error code</dt>
                  <dd>
                    <code>{failure.data.code}</code>
                  </dd>
                </div>
                <div>
                  <dt>Cleanup</dt>
                  <dd>{failure.data.cleanup}</dd>
                </div>
                {failure.data.exit_code !== null && (
                  <div>
                    <dt>Exit code</dt>
                    <dd>{failure.data.exit_code}</dd>
                  </div>
                )}
              </dl>
              {failure.data.cause && (
                <WorkbenchDisclosure title="Technical cause">
                  <pre className="work-job-json">{failure.data.cause}</pre>
                </WorkbenchDisclosure>
              )}
              {failure.data.retryable && <p>Review the cause and compute settings before starting another job.</p>}
            </section>
          )}
          {failure.error && (
            <Notice
              tone="danger"
              title="Failure diagnostics could not be loaded. The saved job state remains visible."
              live="alert"
            />
          )}
          <section className="panel">
            <h3>Overview</h3>
            {job.status === 'waiting' && (
              <p>Continue in the workspace to review the result or provide the next input.</p>
            )}
            <dl className="work-job-facts">
              <div>
                <dt>Operation</dt>
                <dd>{operation?.title ?? job.operation}</dd>
              </div>
              <div>
                <dt>Compute</dt>
                <dd>
                  {job.execution?.label ?? 'Existing job'}
                  {job.execution?.infrastructure && ` · ${job.execution.infrastructure}`}
                </dd>
              </div>
              <div>
                <dt>Created</dt>
                <dd>
                  <time dateTime={job.created_at}>{new Date(job.created_at).toLocaleString()}</time>
                </dd>
              </div>
              <div>
                <dt>Updated</dt>
                <dd>
                  <time dateTime={job.updated_at}>{new Date(job.updated_at).toLocaleString()}</time>
                </dd>
              </div>
            </dl>
          </section>
          <section className="panel">
            <h3>Results</h3>
            {job.outputs.length ? (
              <div className="workbench-actions">
                {job.outputs.map((output) => (
                  <a key={output.url} className="secondary-button" href={`${apiOrigin}${output.url}`}>
                    {output.label}
                  </a>
                ))}
              </div>
            ) : (
              <p>
                {isWorkActive(job)
                  ? 'Results appear here when the job publishes them.'
                  : 'No downloadable outputs were published. Open the workspace for its saved result.'}
              </p>
            )}
          </section>
          {job.parent_ids.length > 0 && (
            <section className="panel">
              <h3>Inputs from other jobs</h3>
              <div className="workbench-actions">
                {job.parent_ids.map((id) => (
                  <button className="text-button" key={id} onClick={() => onSelect(id)}>
                    {id}
                  </button>
                ))}
              </div>
            </section>
          )}
        </>
      )}
      {view === 'timeline' && (
        <section className="panel">
          <h3>Timeline</h3>
          {job.events.length ? (
            <ol className="work-job-events">
              {job.events.map((event, index) => (
                <li key={`${event.timestamp}-${index}`}>
                  <time dateTime={event.timestamp}>{new Date(event.timestamp).toLocaleString()}</time>
                  <span>{event.message}</span>
                </li>
              ))}
            </ol>
          ) : (
            <p>No detailed lifecycle events were retained for this job.</p>
          )}
        </section>
      )}
      {view === 'logs' && <WorkJobLogs key={`${job.project_id}:${job.id}`} job={job} />}
      {view === 'details' && (
        <section className="panel">
          <h3>Inputs</h3>
          <pre className="work-job-json">{JSON.stringify(job.inputs, null, 2)}</pre>
          {job.result !== null && (
            <>
              <h3>Result</h3>
              <pre className="work-job-json">{JSON.stringify(job.result, null, 2)}</pre>
            </>
          )}
        </section>
      )}
    </section>
  );
}

export function WorkJobs({
  projectId,
  module,
  onOpen,
  onNew,
  preferredJobId,
}: {
  projectId: string;
  module?: string;
  onOpen: (job: WorkJob) => void;
  onNew: (operation: JobOperation) => void;
  preferredJobId?: string;
}) {
  const [selected, setSelected] = useState(preferredJobId ?? '');
  const [filter, setFilter] = useState('all');
  const [state, setState] = useState('all');
  const [search, setSearch] = useState('');
  const [query, setQuery] = useState('');
  const [offset, setOffset] = useState(0);
  useEffect(() => {
    const timer = setTimeout(() => {
      setQuery(search);
      setOffset(0);
    }, 250);
    return () => clearTimeout(timer);
  }, [search]);
  const [creating, setCreating] = useState(false);
  useEffect(() => {
    setSelected(preferredJobId ?? '');
    setCreating(false);
    setOffset(0);
  }, [projectId, module, preferredJobId]);
  const jobs = useQuery({
    queryKey: ['work-jobs', projectId, module ?? filter, state, query, offset],
    queryFn: ({ signal }) =>
      workJobs.page(projectId, offset, { module: module ?? filter, status: state, q: query }, signal),
    enabled: !!projectId,
    retry: false,
    refetchInterval: (result) => (result.state.data?.jobs.some(isWorkActive) ? 1500 : 5000),
  });
  const operations = useQuery({ queryKey: ['job-operations'], queryFn: workJobs.operations, retry: false });
  const detail = useQuery({
    queryKey: ['work-job', projectId, selected],
    queryFn: ({ signal }) => workJobs.get(projectId, selected, signal),
    enabled: !!projectId && !!selected,
    retry: false,
    refetchInterval: (query) =>
      query.state.data && !isWorkActive(query.state.data) && query.state.data.status !== 'waiting' ? false : 2000,
  });
  const owned = (jobs.data?.jobs ?? []).filter((job) => job.project_id === projectId);
  const current = detail.data?.project_id === projectId && detail.data.id === selected ? detail.data : undefined;
  const choices = (operations.data ?? []).filter((operation) => !module || operation.module === module);
  const modules = [...new Set((operations.data ?? []).map((item) => item.module))];
  if (creating)
    return (
      <NewWorkJob
        key={`${projectId}:${module ?? 'all'}`}
        projectId={projectId}
        choices={choices}
        onBack={() => setCreating(false)}
        onNew={onNew}
      />
    );
  if (selected)
    return current ? (
      <JobDetail
        key={`${projectId}:${selected}`}
        job={current}
        operations={operations.data ?? []}
        onOpen={onOpen}
        onSelect={setSelected}
        onBack={() => setSelected('')}
      />
    ) : (
      <section className="panel">
        <button className="text-button" onClick={() => setSelected('')}>
          ← Jobs
        </button>
        {detail.error ? (
          <Notice tone="danger" title={detail.error.message} live="alert" />
        ) : (
          <p role="status">Loading job…</p>
        )}
      </section>
    );
  const visible = owned.filter((job) => !module || job.module === module);
  return (
    <div className="work-jobs" aria-label="Project jobs">
      <div className="work-job-filters">
        <label>
          Search jobs
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Name, operation, or compute"
            maxLength={200}
          />
        </label>
        {!module && (
          <label>
            Module
            <select
              value={filter}
              onChange={(event) => {
                setFilter(event.target.value);
                setOffset(0);
              }}
            >
              <option value="all">All modules</option>
              {modules.map((id) => (
                <option key={id} value={id}>
                  {id.replaceAll('-', ' ')}
                </option>
              ))}
            </select>
          </label>
        )}
        <label>
          Status
          <select
            value={state}
            onChange={(event) => {
              setState(event.target.value);
              setOffset(0);
            }}
          >
            <option value="all">All statuses</option>
            <option value="active">Active</option>
            {Object.entries(workStatusLabels)
              .filter(([id]) => !['queued', 'running', 'preparing', 'cancelling'].includes(id))
              .map(([id, label]) => (
                <option key={id} value={id}>
                  {label}
                </option>
              ))}
          </select>
        </label>
      </div>
      {operations.error && (
        <Notice
          tone="danger"
          title="Job operations could not be loaded. Refresh before starting a new job."
          live="alert"
        />
      )}
      <JobHistory
        title="Jobs"
        newLabel="New job"
        loading={jobs.isPending && !!projectId}
        error={jobs.error}
        disabled={!projectId || operations.isError || operations.isPending}
        emptyMessage={
          projectId
            ? query || state !== 'all' || filter !== 'all'
              ? 'No jobs match these filters.'
              : 'No jobs yet. Start a job to see its progress and results here.'
            : 'Select a project to see its jobs.'
        }
        entries={visible.map((job) => ({
          id: job.id,
          title: job.title,
          subtitle: `${operations.data?.find((item) => item.id === job.operation)?.title ?? job.operation} · ${job.execution?.label ?? 'Existing job'}`,
          status: job.status,
          statusLabel: workStatusLabels[job.status] ?? job.status,
          createdAt: job.created_at,
          progress: isWorkActive(job) ? job.stage?.replaceAll('_', ' ') : undefined,
        }))}
        onNew={() => setCreating(true)}
        onSelect={setSelected}
      />
      {(offset > 0 || (jobs.data?.nextOffset !== null && jobs.data?.nextOffset !== undefined)) && (
        <nav className="workbench-actions work-job-pagination" aria-label="Job history pages">
          <button
            className="secondary-button"
            disabled={offset === 0 || jobs.isFetching}
            onClick={() => setOffset(Math.max(0, offset - 50))}
          >
            Previous jobs
          </button>
          <span>Page {Math.floor(offset / 50) + 1}</span>
          <button
            className="secondary-button"
            disabled={jobs.data?.nextOffset == null || jobs.isFetching}
            onClick={() => {
              if (jobs.data?.nextOffset != null) setOffset(jobs.data.nextOffset);
            }}
          >
            Next jobs
          </button>
        </nav>
      )}
    </div>
  );
}
