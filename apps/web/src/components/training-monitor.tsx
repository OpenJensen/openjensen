"use client";

import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  api, ApiError, artifactDownloadUrl, isActive, trainingReproducibilityUrl,
  type Job, type PolicyArtifact, type PolicyOptions, type TrainingMetric, type TrainingTelemetry,
} from "@/lib/api";
import { conciseRunError, observedProgress, runSummary } from "@/lib/run-summary";
import "./training-monitor.css";
import { checkpointLabel, isCloudArtifact, quantizationIssue, sortCheckpoints } from "@/lib/checkpoints";
import { ActExportControl } from "./act-export-control";
import type { TrainingModel } from "@/lib/training-models";

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}
function duration(value: unknown) {
  if (!finite(value) || value < 0) return "—";
  const seconds = Math.floor(value);
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}
function metric(value: unknown) {
  if (!finite(value)) return "—";
  if (value !== 0 && Math.abs(value) < 0.0001) return value.toExponential(3);
  return value.toLocaleString(undefined, { maximumFractionDigits: 5 });
}
function time(value?: string | null) {
  const parsed = value ? new Date(value) : null;
  return parsed && Number.isFinite(parsed.getTime()) ? parsed.toLocaleString() : "Time unavailable";
}
function object(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}
function description(value: unknown) {
  if (value === null || value === undefined) return "Not recorded";
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

const phaseNames: Record<string, string> = {
  queued: "Queued", preparing: "Preparing your GPU", training: "Training", operation: "Training worker",
  validation: "Validating on held-out episodes", checkpoint: "Saving checkpoint",
  verifying: "Verifying the saved model", completed: "Completed", succeeded: "Completed",
  failed: "Failed", cancelled: "Cancelled", interrupted: "Interrupted",
};

function LossChart({ metrics }: { metrics: TrainingMetric[] }) {
  const chartRef = useRef<HTMLElement>(null);
  const [width, setWidth] = useState(720);
  const series = (["train_loss", "validation_loss"] as const).map(key => ({
    key,
    label: key === "train_loss" ? "Training loss" : "Validation loss",
    points: metrics.filter(point => finite(point.step) && finite(point[key])),
  }));
  const all = series.flatMap(line => line.points.map(point => ({ step: point.step, value: point[line.key] as number })));
  const hasSamples = all.length > 0;
  useEffect(() => {
    if (!chartRef.current) return;
    const observer = new ResizeObserver(entries => setWidth(Math.max(240, entries[0].contentRect.width)));
    observer.observe(chartRef.current);
    return () => observer.disconnect();
  }, [hasSamples]);
  if (!all.length) return <div className="training-chart-empty">Loss curves appear when the worker reports its first metrics.</div>;
  const minStep = Math.min(...all.map(point => point.step));
  const maxStep = Math.max(...all.map(point => point.step));
  const minimum = Math.min(...all.map(point => point.value));
  const maximum = Math.max(...all.map(point => point.value));
  const padding = Math.max((maximum - minimum) * 0.1, Math.abs(maximum) * 0.05, 0.01);
  const minValue = minimum - padding;
  const maxValue = maximum + padding;
  const x = (step: number) => 56 + ((step - minStep) / Math.max(maxStep - minStep, 1)) * (width - 84);
  const y = (value: number) => 16 + (1 - (value - minValue) / (maxValue - minValue)) * 172;
  return (
    <figure className="training-loss-chart" ref={chartRef}>
      <svg viewBox={`0 0 ${width} 226`} role="img" aria-label={`Training and validation loss across observed steps ${minStep} to ${maxStep}`}>
        {[0, 0.5, 1].map(fraction => {
          const value = minValue + (maxValue - minValue) * fraction;
          return <g key={fraction}><line x1="56" x2={width - 28} y1={y(value)} y2={y(value)} className="training-chart-grid" /><text x="48" y={y(value) + 4} textAnchor="end">{value.toPrecision(3)}</text></g>;
        })}
        {series.map(line => <g key={line.key} className={`training-chart-${line.key}`}>
          <polyline fill="none" strokeWidth="2" points={line.points.map(point => `${x(point.step)},${y(point[line.key] as number)}`).join(" ")} />
          {line.points.length === 1 && <circle cx={x(line.points[0].step)} cy={y(line.points[0][line.key] as number)} r="3" />}
        </g>)}
        <text x="56" y="210">{minStep.toLocaleString()}</text>
        <text x={width - 28} y="210" textAnchor="end">{maxStep.toLocaleString()}</text>
        <text x={width / 2} y="224" textAnchor="middle">Optimizer step</text>
      </svg>
      <figcaption>{series.map(line => <span key={line.key} className={`training-legend-${line.key}`}>{line.label}</span>)}</figcaption>
    </figure>
  );
}

export function TrainingMonitor({ run, projectId, active, artifacts, onCancel, cancelling, cancelError, onResume, onQuantize, modelCatalog = [], exportRuntimes = [], exportJobs = [], exportArtifacts = [] }: {
  run: Job;
  projectId: string;
  active: boolean;
  artifacts: PolicyArtifact[];
  onCancel: () => void;
  cancelling: boolean;
  cancelError?: Error | null;
  onResume?: () => void;
  onQuantize?: (artifactId: string) => void;
  modelCatalog?: TrainingModel[];
  exportRuntimes?: PolicyOptions["runtimes"];
  exportJobs?: Job[];
  exportArtifacts?: PolicyArtifact[];
}) {
  const [now, setNow] = useState(() => Date.now());
  const [logFilter, setLogFilter] = useState("");
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [checkpointId, setCheckpointId] = useState("latest");
  const savedCheckpoints = sortCheckpoints(artifacts.filter(item => item.format === "training_checkpoint" || item.format === "native_checkpoint"));
  const chosenCheckpoint = savedCheckpoints.find(item => item.id === checkpointId) ?? savedCheckpoints[0];
  const checkpointIssue = quantizationIssue(chosenCheckpoint);
  const telemetry = useQuery({
    queryKey: ["training-telemetry", run.id, run.status],
    queryFn: () => api.trainingTelemetry(run.id),
    enabled: active,
    retry: (failures, error) => !(error instanceof ApiError && error.status === 404) && failures < 3,
    refetchInterval: query => active && isActive(run) && !(query.state.error instanceof ApiError && query.state.error.status === 404) ? 2000 : false,
  });
  useEffect(() => {
    if (!active || !isActive(run)) return;
    const timer = window.setInterval(() => setNow(Date.now()), 10000);
    return () => window.clearInterval(timer);
  }, [active, run.id, run.status]);
  const legacy = telemetry.error instanceof ApiError && telemetry.error.status === 404;
  const legacyEvents = useQuery({
    queryKey: ["events", run.id],
    queryFn: () => api.events(run.id),
    enabled: active && legacy,
    refetchInterval: active && legacy && isActive(run) ? 2000 : false,
  });
  const reportedSteps = (legacyEvents.data ?? []).flatMap(event => {
    const match = /^(?:Optimizer step (\d+)\b|Checkpoint (\d+) saved locally$)/i.exec(event.message);
    return match ? [Number(match[1] ?? match[2])].filter(Number.isFinite) : [];
  });
  const acceptedRecipe = "training" in run.request ? run.request.training : null;
  const legacyData: TrainingTelemetry | undefined = legacy ? {
    job_id: run.id, status: run.status, phase: run.stage ?? run.status,
    current_action: legacyEvents.data?.at(-1)?.message ?? "Waiting for recorded activity",
    updated_at: run.updated_at, completed_steps: reportedSteps.length ? Math.max(...reportedSteps) : null,
    total_steps: finite(acceptedRecipe?.steps) ? acceptedRecipe.steps : null,
    events: legacyEvents.data ?? [], metrics: [], metrics_truncated: false,
  } : undefined;
  const data = telemetry.data ?? legacyData;
  const status = data?.status ?? run.status;
  const running = status === "running" || status === "queued";
  const completed = finite(data?.completed_steps) && data.completed_steps >= 0 ? data.completed_steps : null;
  const total = finite(data?.total_steps) && data.total_steps > 0 ? data.total_steps : null;
  const percent = completed !== null && total !== null ? Math.min(100, Math.max(0, completed / total * 100)) : null;
  const metrics = (data?.metrics ?? []).filter(point => finite(point.step) && point.step >= 0);
  const latest = data?.latest;
  const metricHistory = [...(latest && finite(latest.step) ? [latest] : []), ...[...metrics].reverse()]
    .sort((a, b) => b.step - a.step);
  const latestTraining = metricHistory.find(point => finite(point.train_loss));
  const latestValidation = metricHistory.find(point => finite(point.validation_loss));
  const latestLearningRate = metricHistory.find(point => finite(point.learning_rate));
  const lastMetricTime = latest?.timestamp ? Date.parse(latest.timestamp) : NaN;
  const stale = running && Number.isFinite(lastMetricTime) && now - lastMetricTime > 120000;
  const phase = running ? data?.phase ?? run.stage ?? status : status === "succeeded" ? "completed" : status;
  const events = data?.events ?? [];
  const eventProgress = observedProgress(events);
  const checkpointSteps = (data?.checkpoints ?? []).map(item => item.step).filter(value => finite(value) && value >= 0);
  if (eventProgress.checkpoint !== null) checkpointSteps.push(eventProgress.checkpoint);
  const lastCheckpoint = checkpointSteps.length ? Math.max(...checkpointSteps) : null;
  const filteredEvents = logFilter.trim()
    ? events.filter(event => `${event.stage} ${event.message}`.toLowerCase().includes(logFilter.toLowerCase()))
    : events;
  const reproducibility = object(data?.reproducibility);
  const recipe = object(reproducibility.recipe);
  const model = object(reproducibility.model);
  const dataset = object(reproducibility.dataset);
  const evidence = object(reproducibility.evidence);
  const splits = object(evidence.splits);
  const limitations = Array.isArray(reproducibility.limitations) ? reproducibility.limitations.filter((item): item is string => typeof item === "string") : [];

  return (
    <article className="training-monitor" aria-label="Training run monitor" data-run-id={run.id}>
      <header className="training-monitor-header">
        <div><span className="training-monitor-eyebrow">Run {run.id.slice(0, 8)}</span><h3>{phaseNames[phase] ?? phase.replaceAll("_", " ")}</h3></div>
        <div className="training-monitor-actions">
          <span className={`status status-${status}`}>{status}</span>
          {running && <button type="button" className="secondary-button" disabled={cancelling} onClick={onCancel}>{cancelling ? "Cancelling…" : "Cancel run"}</button>}
          {!running && onResume && <button type="button" className="secondary-button" onClick={onResume}>Resume from checkpoint</button>}
        </div>
      </header>
      <p className="training-current-action" role="status">{runSummary(run, phase, status)}</p>
      {telemetry.isPending && <p className="training-monitor-note" role="status">Loading run telemetry…</p>}
      {legacy && <p className="training-stale-notice" role="status">Showing saved activity from the running application. Detailed telemetry becomes available after the application is restarted once active runs finish.</p>}
      {legacyEvents.error && legacy && <p className="error-notice" role="alert">Saved activity is unavailable: {legacyEvents.error.message}</p>}
      {telemetry.error && !legacy && <div className="error-notice" role="alert">Run telemetry is unavailable: {telemetry.error.message} <button type="button" className="text-button" onClick={() => void telemetry.refetch()}>Retry telemetry</button></div>}
      {run.error && <p className="error-notice" role="alert">{conciseRunError(run.error)}</p>}
      {cancelError && <p className="error-notice" role="alert">{cancelError.message}</p>}
      <div className="training-progress-heading">
        <span>{completed !== null ? `${completed.toLocaleString()}${total !== null ? ` / ${total.toLocaleString()}` : ""} optimizer steps` : "Waiting for the first reported step"}</span>
        <strong>{percent !== null ? `${percent.toLocaleString(undefined, { maximumFractionDigits: 1 })}%` : "—"}</strong>
      </div>
      <progress className="training-progress" aria-label="Training step progress" max="100" {...(percent !== null ? { value: percent } : {})} />
      <p className="training-monitor-note">Progress counts completed optimizer steps. Setup, validation and checkpoint verification can take additional time.</p>
      {stale && <p className="training-stale-notice" role="status">No new training metric for {duration((now - lastMetricTime) / 1000)}. Last reported values are shown; the run may still be preparing data, validating, or saving.</p>}
      <dl className="training-metrics">
        <div><dt>Training loss</dt><dd>{metric(latestTraining?.train_loss)}</dd><small>{latestTraining ? `Step ${latestTraining.step.toLocaleString()}` : "Awaiting a metric"}</small></div>
        <div><dt>Validation loss</dt><dd>{metric(latestValidation?.validation_loss)}</dd><small>{latestValidation ? `Step ${latestValidation.step.toLocaleString()}` : "Awaiting validation"}</small></div>
        <div><dt>Learning rate</dt><dd>{metric(latestLearningRate?.learning_rate)}</dd><small>{latestLearningRate ? `Step ${latestLearningRate.step.toLocaleString()}` : "Reported by the optimizer"}</small></div>
        <div><dt>Worker elapsed</dt><dd>{duration(data?.elapsed_seconds)}</dd><small>{finite(data?.wall_seconds) ? `${duration(data.wall_seconds)} since launch` : "Worker-reported duration"}</small></div>
        <div><dt>Estimated remaining</dt><dd>{running && !stale ? duration(data?.eta_seconds) : "—"}</dd><small>{!running ? "Run ended" : stale ? "Awaiting a fresh metric" : "Based on observed step speed"}</small></div>
      </dl>
      <div className="training-chart-heading"><h4>Loss over time</h4><span>{latest ? latest.timestamp ? `Last metric ${time(latest.timestamp)}` : "Metric timestamp unavailable" : "No metrics yet"}</span></div>
      <LossChart metrics={metrics} />
      {data?.metrics_truncated && <p className="training-monitor-note">Showing the most recent recorded metrics. Earlier samples remain in the run files.</p>}
      {lastCheckpoint !== null && <p className="training-latest-checkpoint">Latest checkpoint: step {lastCheckpoint.toLocaleString()}</p>}
      {!!savedCheckpoints.length && onQuantize && <section className="training-checkpoint-actions" aria-label="Use a trained checkpoint">
        <div><h4>Use a trained checkpoint</h4><p>{chosenCheckpoint?.metadata?.architecture === "act" ? "Create a separate ACT inference package while preserving the training checkpoint." : chosenCheckpoint && isCloudArtifact(chosenCheckpoint) ? "Saved on Google Cloud. Quantization runs from the cloud copy." : "Choose a saved step to create a compressed policy."}</p></div>
        <label>Checkpoint<select aria-label="Checkpoint" value={checkpointId} onChange={event => setCheckpointId(event.target.value)}>
          <option value="latest">Latest checkpoint · {checkpointLabel(savedCheckpoints[0], [run], modelCatalog)}</option>
          {savedCheckpoints.map(item => <option key={item.id} value={item.id}>{checkpointLabel(item, [run], modelCatalog)}</option>)}
        </select></label>
        {chosenCheckpoint?.metadata?.architecture === "act" ? <ActExportControl
          projectId={projectId} checkpoint={chosenCheckpoint} runtimes={exportRuntimes}
          jobs={exportJobs} artifacts={exportArtifacts} active={active}
        /> : <><button type="button" className="primary-button" disabled={!!checkpointIssue} onClick={() => chosenCheckpoint && onQuantize(chosenCheckpoint.id)}>Quantize checkpoint</button>
        {checkpointIssue && <p className="training-monitor-note" role="status">{checkpointIssue}</p>}</>}
      </section>}
      <details className="training-monitor-disclosure training-details" onToggle={event => setDetailsOpen(event.currentTarget.open)}>
        <summary>Training details<span>Activity, checkpoints and worker output</span></summary>
        {detailsOpen && <>
        {data?.current_action && <p className="training-monitor-note">{data.current_action}</p>}
        {run.error && <pre className="training-error-detail">{run.error}</pre>}
      {!!metrics.length && <details className="training-monitor-disclosure"><summary>Recent metric values</summary>
        <div className="training-metric-table"><table><thead><tr><th>Step</th><th>Training loss</th><th>Validation loss</th><th>Learning rate</th><th>Gradient norm</th></tr></thead>
          <tbody>{metrics.slice(-20).reverse().map((point, index) => <tr key={`${point.step}-${index}`}><th>{point.step.toLocaleString()}</th><td>{metric(point.train_loss)}</td><td>{metric(point.validation_loss)}</td><td>{metric(point.learning_rate)}</td><td>{metric(point.grad_norm)}</td></tr>)}</tbody></table></div>
      </details>}
      {!!data?.checkpoints?.length && <div className="training-saved-checkpoints"><h4>Saved checkpoints</h4><ul>{data.checkpoints.map((checkpoint, index) => <li key={`${checkpoint.name}-${index}`}><strong>Step {checkpoint.step.toLocaleString()}</strong><span>{checkpoint.name}</span><time>{time(checkpoint.timestamp)}</time></li>)}</ul></div>}
      {!!artifacts.length && <div className="training-run-downloads">{artifacts.map(item => <a className="text-link" key={item.id} href={artifactDownloadUrl(projectId, item.id)}>Download {item.label}</a>)}</div>}
      <section className="training-detail-activity"><h4>Activity log</h4>
        <label className="training-log-filter">Filter activity<input type="search" value={logFilter} onChange={event => setLogFilter(event.target.value)} placeholder="Search stages and messages" /></label>
        <ol className="training-event-log" aria-label="Persisted training activity">{filteredEvents.map(event => <li key={event.sequence}><time>{time(event.timestamp)}</time><span>{event.stage}</span><p>{event.message}</p></li>)}</ol>
        {!filteredEvents.length && <p className="training-monitor-note">{logFilter ? "No matching events." : "No activity events have been recorded yet."}</p>}
      </section>
      <details className="training-monitor-disclosure training-worker-logs"><summary>Worker output <span>Latest setup and training messages</span></summary>
        {data?.logs?.length ? <pre aria-label="Worker output">{data.logs.join("\n")}</pre> : <p className="training-monitor-note">No worker output has been recorded yet.</p>}
      </details>
        </>}
      </details>
      <details className="training-monitor-disclosure training-reproducibility"><summary>Reproducibility <span>Recipe, revisions and run evidence</span></summary>
        <p className="training-monitor-note">The saved recipe and available evidence describe this run. Downloads include the accepted request, dataset snapshot, compute selection and checkpoint lineage.</p>
        {!!Object.keys(reproducibility).length && <>
          <dl className="training-reproducibility-facts">
            <div><dt>Model</dt><dd>{description(model.repository ?? recipe.model_id)}</dd></div>
            <div><dt>Model revision</dt><dd>{description(model.revision ?? recipe.model_revision)}</dd></div>
            <div><dt>Dataset</dt><dd>{description(dataset.repo_id ?? recipe.dataset_id)}</dd></div>
            <div><dt>Dataset revision</dt><dd>{description(dataset.revision ?? recipe.dataset_revision)}</dd></div>
            <div><dt>Seed</dt><dd>{description(recipe.seed)}</dd></div>
            <div><dt>Recipe source</dt><dd>{description(reproducibility.recipe_source).replaceAll("_", " ")}</dd></div>
            <div><dt>Episode split</dt><dd>{Array.isArray(splits.train) && Array.isArray(splits.validation) ? `${splits.train.length} train · ${splits.validation.length} validation episodes` : "Not recorded yet"}</dd></div>
            <div><dt>Runtime</dt><dd>{description(object(reproducibility.runtime).label ?? object(reproducibility.runtime).id ?? ("runtime_id" in run.request ? run.request.runtime_id : undefined))}</dd></div>
          </dl>
          {limitations.length > 0 && <ul className="training-monitor-limitations">{limitations.map((item, index) => <li key={index}>{item}</li>)}</ul>}
          <a className="secondary-button training-reproducibility-download" href={trainingReproducibilityUrl(run.id)}>Download reproducibility JSON</a>
          <details><summary>Full recorded configuration</summary><pre>{JSON.stringify(reproducibility, null, 2)}</pre></details>
        </>}
        {!Object.keys(reproducibility).length && <p className="training-monitor-note">The recorded configuration is not available yet.</p>}
      </details>
    </article>
  );
}
