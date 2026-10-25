'use client';

import type { UseQueryResult } from '@tanstack/react-query';
import { artifactDownloadUrl, isActive, type Job, type JobEvent, type LifecycleResult } from '@/lib/api';
import { isCloudArtifact } from '@/lib/checkpoints';
import { ownedModels } from '@/lib/model-library';
import { finiteNumber, sizeLabel } from './workflow-shared';

export function WorkflowJobResults({
  engineReport,
  inference,
  engineOnly,
}: {
  engineReport?: Record<string, unknown>;
  inference?: Record<string, unknown>;
  engineOnly: boolean;
}) {
  return (
    <section className="workflow-measurements" aria-label="Measured results">
      <h3>Results</h3>
      <dl className="workflow-job-facts">
        {engineReport && (
          <>
            {finiteNumber(engineReport.p50_ms) && (
              <div>
                <dt>Median latency</dt>
                <dd>{engineReport.p50_ms.toLocaleString(undefined, { maximumFractionDigits: 1 })} ms</dd>
              </div>
            )}
            {finiteNumber(engineReport.p95_ms) && (
              <div>
                <dt>p95 latency</dt>
                <dd>{engineReport.p95_ms.toLocaleString(undefined, { maximumFractionDigits: 1 })} ms</dd>
              </div>
            )}
            {finiteNumber(engineReport.peak_device_mib) && (
              <div>
                <dt>Peak memory</dt>
                <dd>{engineReport.peak_device_mib.toLocaleString(undefined, { maximumFractionDigits: 1 })} MiB</dd>
              </div>
            )}
            {finiteNumber(engineReport.success_rate) && (
              <div>
                <dt>Task success</dt>
                <dd>{(engineReport.success_rate * 100).toLocaleString(undefined, { maximumFractionDigits: 1 })}%</dd>
              </div>
            )}
          </>
        )}
        {inference && finiteNumber(inference.finite_action_values) && (
          <div>
            <dt>Native inference</dt>
            <dd>{inference.finite_action_values.toLocaleString()} finite action values</dd>
          </div>
        )}
        {inference && finiteNumber(inference.wall_seconds) && (
          <div>
            <dt>Native check</dt>
            <dd>{inference.wall_seconds.toLocaleString(undefined, { maximumFractionDigits: 1 })} s</dd>
          </div>
        )}
      </dl>
      {engineOnly && <p className="muted">Synthetic input checks. Robot task success was not measured.</p>}
    </section>
  );
}

export function WorkflowJobOutput({
  projectId,
  selected,
  data,
  onModel,
}: {
  projectId: string;
  selected: Job;
  data?: LifecycleResult;
  onModel?: (artifactId: string) => void;
}) {
  return (
    <section className="workflow-job-output" aria-label="Job output">
      {onModel &&
        ownedModels(data?.artifacts ?? [], projectId)
          .filter((item) => item.job_id === selected.id)
          .map((item) => (
            <button type="button" className="secondary-button" key={item.id} onClick={() => onModel(item.id)}>
              View {item.label} in Models
            </button>
          ))}
      <h3>Output</h3>
      {data?.artifacts?.length ? (
        <ul className="workflow-output-list">
          {data.artifacts.map((artifact) => (
            <li key={artifact.id}>
              <div>
                <strong>{artifact.label}</strong>
                <span>
                  {artifact.format === 'gguf' ? 'GGUF' : artifact.format.replaceAll('_', ' ')}
                  {artifact.file_bytes > 0 ? ` · ${sizeLabel(artifact.file_bytes)}` : ''}
                  {isCloudArtifact(artifact) ? ' · Google Cloud' : ''}
                </span>
              </div>
              <a className="secondary-button" href={artifactDownloadUrl(projectId, artifact.id)}>
                Download {artifact.label}
              </a>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted">
          {isActive(selected)
            ? 'Output will appear when this job finishes.'
            : selected.status === 'succeeded'
              ? 'This job produced a report. Open the details below.'
              : 'No completed output.'}
        </p>
      )}
    </section>
  );
}

export function WorkflowJobLogs({
  selected,
  events,
  data,
}: {
  selected: Job;
  events: UseQueryResult<JobEvent[]>;
  data?: LifecycleResult;
}) {
  return (
    <details className="workflow-job-technical" key={selected.id}>
      <summary>Details and logs</summary>
      <dl className="workflow-job-facts">
        <div>
          <dt>Job ID</dt>
          <dd>{selected.id}</dd>
        </div>
        <div>
          <dt>Updated</dt>
          <dd>{new Date(selected.updated_at).toLocaleString()}</dd>
        </div>
      </dl>
      {selected.error && <pre className="workflow-json">{selected.error}</pre>}
      <ol className="workflow-events" aria-label="Detailed run activity">
        {events.data?.map((event) => (
          <li key={event.sequence}>
            <time>{new Date(event.timestamp).toLocaleString()}</time>
            <span>{event.message}</span>
          </li>
        ))}
      </ol>
      {data?.reports?.map((report, index) => (
        <details className="provenance" key={index}>
          <summary>
            {String(report.stage ?? `Report ${index + 1}`)} · {String(report.scope ?? report.operation ?? 'Details')}
          </summary>
          <pre className="workflow-json">{JSON.stringify(report, null, 2)}</pre>
        </details>
      ))}
      <details className="provenance">
        <summary>Saved request and lineage</summary>
        <pre className="workflow-json">
          {JSON.stringify({ request: selected.request, artifacts: data?.artifacts }, null, 2)}
        </pre>
      </details>
    </details>
  );
}
