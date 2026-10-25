'use client';

import { isActive } from '@/lib/api';
import { conciseRunError, runLabel, runSummary } from '@/lib/run-summary';
import type { Preferences } from '@/lib/workflow-preferences';
import type { SettingsTab } from './workflow-shared';
import type { WorkflowState } from './workflow-state';

export function WorkflowDiagnostics({
  workflow,
  onTabChange: setTab,
  onOpenQuantize,
}: {
  workflow: WorkflowState;
  onTabChange: (tab: SettingsTab) => void;
  onOpenQuantize: () => void;
}) {
  const {
    artifacts,
    diagnosticBlocker,
    errors,
    input,
    inputs,
    mutation,
    options,
    policyJobs,
    preferences,
    ready,
    runtime,
    runtimes,
    setInput,
    setRuntimeId,
    update,
  } = workflow;
  return (
    <>
      <h2>Run diagnostics</h2>
      <form
        className="workflow-fields diagnostic-launcher"
        onSubmit={(event) => {
          event.preventDefault();
          if (!diagnosticBlocker && !mutation.isPending) mutation.mutate();
        }}
      >
        <label>
          Diagnostic execution target
          <select value={runtime?.id ?? ''} disabled={!ready} onChange={(event) => setRuntimeId(event.target.value)}>
            <option value="" disabled>
              Select a target
            </option>
            {runtimes.map((target) => (
              <option key={target.id} value={target.id}>
                {target.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Diagnostic policy
          <select value={input} disabled={!ready} onChange={(event) => setInput(event.target.value)}>
            <option value="">Choose a project policy</option>
            {inputs.map((artifact) => (
              <option key={artifact.id} value={artifact.id}>
                {artifact.label} · {artifact.id.slice(0, 8)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Diagnostic mode
          <select
            value={preferences.mode}
            disabled={!ready || preferences.suite === 'libero_spatial'}
            onChange={(event) => update('mode', event.target.value as Preferences['mode'])}
          >
            <option value="engine">Engine checks</option>
            <option value="libero" disabled={!runtime?.simulation}>
              LIBERO task evaluation
            </option>
          </select>
        </label>
        <details className="workflow-advanced">
          <summary>Protocol details</summary>
          <p className="muted">
            {preferences.mode === 'engine'
              ? `Checks loading, finite actions, prediction latency and memory with ${preferences.warmup} warmups and ${preferences.repetitions} repetitions.`
              : preferences.suite === 'libero_spatial'
                ? `Evaluates LIBERO Spatial tasks ${preferences.taskIds} with initial states ${preferences.searchStates}, seed 42 and the full 280-step horizon. The policy must include pinned Spatial assets and approved parity limits.`
                : `Evaluates LIBERO Object task ${preferences.task} with initial states ${preferences.searchStates}, seed 42 and up to ${preferences.steps} steps. The policy must declare LIBERO Object compatibility.`}
          </p>
        </details>
        <button className="text-link" type="button" onClick={() => setTab('settings')}>
          Edit diagnostic settings
        </button>
        {diagnosticBlocker && (
          <p className="warning-box" id="diagnostic-readiness" role="status">
            {diagnosticBlocker}
          </p>
        )}
        {!runtime && !options.isPending && (
          <p className="muted">
            <a
              className="text-link"
              href="https://github.com/OpenJensen/openjensen/blob/main/docs/policy-workflow.md#configure-the-execution-host"
              target="_blank"
              rel="noreferrer"
            >
              Worker setup instructions ↗
            </a>
          </p>
        )}
        {!!runtime && !inputs.length && !artifacts.isPending && (
          <button className="secondary-button" type="button" onClick={onOpenQuantize}>
            Open Quantize
          </button>
        )}
        <button
          className="primary-button"
          type="submit"
          disabled={!!diagnosticBlocker || mutation.isPending}
          aria-describedby={diagnosticBlocker ? 'diagnostic-readiness' : undefined}
        >
          {mutation.isPending ? 'Starting…' : 'Start diagnostics'}
        </button>
      </form>
      {errors.map((error, index) => (
        <p className="error-notice" role="alert" key={index}>
          {error?.message}
        </p>
      ))}
      <h2 className="diagnostic-runs-heading">Diagnostic runs</h2>
      {policyJobs.length ? <DiagnosticRuns workflow={workflow} /> : <p>No policy runs in this project yet.</p>}
    </>
  );
}

function DiagnosticRuns({ workflow }: { workflow: WorkflowState }) {
  const { cancel, data, detailsJobId, events, policyJobs, progress, selected, setDetailsJobId, setSelectedJobId } =
    workflow;
  return (
    <>
      <label>
        Run
        <select value={selected?.id ?? ''} onChange={(e) => setSelectedJobId(e.target.value)}>
          {policyJobs.map((job) => (
            <option key={job.id} value={job.id}>
              {runLabel(job)} · {job.status} · {new Date(job.created_at).toLocaleString()}
            </option>
          ))}
        </select>
      </label>
      {selected && (
        <div className="workflow-run-summary" role="status">
          <strong>{runSummary(selected)}</strong>
          {progress.step !== null && <span>Last reported step: {progress.step.toLocaleString()}</span>}
          {progress.checkpoint !== null && <span>Latest checkpoint: step {progress.checkpoint.toLocaleString()}</span>}
        </div>
      )}
      {selected?.error && (
        <p className="error-notice" role="alert">
          {conciseRunError(selected.error)}
        </p>
      )}
      {selected && <p className={`status status-${selected.status}`}>{selected.status}</p>}
      {selected && isActive(selected) && (
        <button className="secondary-button" disabled={cancel.isPending} onClick={() => cancel.mutate()}>
          Cancel run
        </button>
      )}
      {!!data?.reports?.some((report) => typeof report.p95_ms === 'number') && (
        <div className="table-scroll">
          <table className="feature-table workflow-metrics">
            <caption>Recorded policy measurements</caption>
            <thead>
              <tr>
                <th>Stage</th>
                <th>p50 (ms)</th>
                <th>p95 (ms)</th>
                <th>Peak target memory (MiB)</th>
                <th>Task success</th>
              </tr>
            </thead>
            <tbody>
              {data.reports
                .filter((report) => typeof report.p95_ms === 'number')
                .map((report, index) => (
                  <tr key={index}>
                    <th>{String(report.stage).replaceAll('-', ' ')}</th>
                    <td>{typeof report.p50_ms === 'number' ? report.p50_ms.toFixed(1) : '—'}</td>
                    <td>{typeof report.p95_ms === 'number' ? report.p95_ms.toFixed(1) : '—'}</td>
                    <td>{typeof report.peak_device_mib === 'number' ? report.peak_device_mib.toFixed(1) : '—'}</td>
                    <td>
                      {typeof report.success_rate === 'number'
                        ? `${(report.success_rate * 100).toFixed(0)}% · ${report.complete_episodes} episodes`
                        : '—'}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}
      {data?.reports?.map((report, index) => (
        <details className="provenance" key={index}>
          <summary>
            {String(report.stage)} · {String(report.scope ?? report.operation)}
          </summary>
          <pre className="workflow-json">{JSON.stringify(report, null, 2)}</pre>
        </details>
      ))}
      <details className="provenance">
        <summary>Recorded recipe and artifact lineage</summary>
        <pre className="workflow-json">
          {JSON.stringify(
            {
              request: selected?.request,
              artifacts: data?.artifacts,
            },
            null,
            2,
          )}
        </pre>
      </details>
      <details
        className="provenance workflow-activity-details"
        key={selected?.id}
        onToggle={(event) => setDetailsJobId(event.currentTarget.open ? (selected?.id ?? null) : null)}
      >
        <summary>
          {selected && runLabel(selected) === 'Training' ? 'Training details' : 'Run details'}
          <span>Recorded activity and errors</span>
        </summary>
        {detailsJobId === selected?.id && (
          <>
            {selected?.error && <pre className="workflow-json">{selected.error}</pre>}
            <ol className="workflow-events" aria-label="Detailed run activity">
              {events.data?.map((event) => (
                <li key={event.sequence}>
                  <time>{new Date(event.timestamp).toLocaleString()}</time>
                  <span>{event.message}</span>
                </li>
              ))}
            </ol>
            {events.error && (
              <p className="error-notice" role="alert">
                {events.error.message}
              </p>
            )}
          </>
        )}
      </details>
    </>
  );
}
