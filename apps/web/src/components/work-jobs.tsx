'use client';

import { useEffect, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiOrigin } from '@/lib/api';
import { WorkbenchDisclosure } from './workbench-disclosure';
import { Icon } from './icon';
import { JobHistory } from './job-history';
import { WorkJobLogs } from './work-job-logs';
import {
  workJobs,
  workStatusLabels,
  jobCategoryActions,
  jobResultLabel,
  isWorkActive,
  type WorkJob,
  type JobOperation,
} from '@/lib/work-jobs';
import { Notice } from './ui/notice';
import './work-jobs.css';

function NewWorkJob({
  choices,
  onBack,
  onNew,
}: {
  choices: JobOperation[];
  onBack: () => void;
  onNew: (operation: JobOperation) => void;
}) {
  return (
    <section className="panel work-job-picker" aria-label="New job">
      <button className="text-button" onClick={onBack}>
        ← Jobs
      </button>
      <h2>What would you like to do?</h2>
      <div className="work-job-operations" role="group" aria-label="Job operation">
        {choices.map((operation) => (
          <button key={operation.id} className="secondary-button" onClick={() => onNew(operation)}>
            {operation.title}
            <Icon name="arrow" size={15} />
          </button>
        ))}
      </div>
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
            {jobResultLabel(job.operation)}
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
            {job.status === 'waiting' && <p>Review the result or open the job controls to continue.</p>}
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
                  : 'No downloadable outputs were published. Open the saved result for more detail.'}
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
  const category = module ? jobCategoryActions[module] : undefined;
  const primary = category && choices.find((item) => item.id === category.operation);
  const additional = module ? choices.filter((item) => item !== primary) : [];

  if (creating)
    return (
      <NewWorkJob
        key={`${projectId}:${module ?? 'all'}`}
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
        title={category?.title ?? 'Jobs'}
        newLabel={category?.label ?? 'New job'}
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
        onNew={() => (primary ? onNew(primary) : setCreating(true))}
        onSelect={setSelected}
      />
      {additional.length > 0 && (
        <div className="workbench-actions work-job-secondary-actions" role="group" aria-label="Other jobs">
          {additional.map((operation) => (
            <button
              key={operation.id}
              className="secondary-button"
              disabled={!projectId}
              onClick={() => onNew(operation)}
            >
              {operation.title}
            </button>
          ))}
        </div>
      )}
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
