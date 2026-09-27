import { objectRecord } from './native-simulation';
import { storedAttempt, type PolicyJobAttempt } from './policy-job-attempt';
import { UncertainSubmission, type SimulationJob } from './native-simulation';

export const simulationAttemptOperation = 'policy.run.simulation';
const statuses = ['queued', 'running', 'succeeded', 'failed', 'cancelled', 'interrupted'];
const text = (value: unknown): value is string => typeof value === 'string' && value.length > 0 && value.length <= 512;
const timestamp = (value: unknown): value is string => typeof value === 'string' && value.length <= 64 && Number.isFinite(Date.parse(value));
const uncertain = () => new UncertainSubmission('The submission outcome is unverified. Check recorded jobs before another request. No automatic retry was made.');
export type SimulationIntent = { project: string; profile: string } & (
  { action: 'run'; artifact: string; timeout: number } |
  { action: 'upload'; filename: string; bytes: number } |
  { action: 'cancel'; job: string }
);
export function simulationIntentMessage(intent: SimulationIntent): string {
  if (![intent.project, intent.profile].every(text)) throw new Error('Simulation request identity is invalid. Refresh the project.');
  let detail: string;
  if (intent.action === 'run') {
    if (!text(intent.artifact) || !Number.isInteger(intent.timeout) || intent.timeout < 30 || intent.timeout > 7200) throw new Error('Simulation recipe is invalid.');
    detail = `Policy: ${intent.artifact} · Budget: ${intent.timeout} seconds · Experimental paid cloud rollout.`;
  } else if (intent.action === 'upload') {
    if (!text(intent.filename) || !Number.isSafeInteger(intent.bytes) || intent.bytes <= 0) throw new Error('Upload identity is invalid.');
    detail = `File: ${JSON.stringify(intent.filename)} · ${intent.bytes} bytes · Local import; no GPU launch.`;
  } else {
    if (!text(intent.job)) throw new Error('Cancellation identity is invalid.');
    detail = `Job: ${intent.job} · Cancellation request; resource deletion is unverified.`;
  }
  const message = `Native simulation · Project: ${intent.project} · Action: ${intent.action} · Profile: ${intent.profile} · ${detail}`;
  if (message.length > 1800) throw new Error('Simulation request identity exceeds the recovery limit.');
  return message;
}
/** Per-tab restart recovery; no credentials, archive bytes or server configuration are persisted. */
export function storedSimulationAttempt(project: string): PolicyJobAttempt {
  const raw = sessionStorage.getItem(`firebird:job-attempt:${simulationAttemptOperation}:${project}`);
  if (raw !== null && raw.length > 16_384) throw new Error('Simulation recovery record is too large.');
  let value: unknown;
  try { value = raw === null ? null : JSON.parse(raw); } catch { /* The shared reader preserves malformed records as uncertainty. */ }
  if (objectRecord(value) && value.state === 'pending' && typeof value.message === 'string' && value.message.length <= 1800) {
    return { state: 'uncertain', message: `${value.message} ${uncertain().message}` };
  }
  return storedAttempt(simulationAttemptOperation, project);
}
/** A receipt records admission only. Never persist result, media, target or model-verification claims. */
export function minimalSimulationReceipt(value: unknown, project: string): SimulationJob {
  if (!objectRecord(value) || !text(value.id) || value.project_id !== project || (typeof value.kind !== 'string' || !['policy.run', 'policy.import'].includes(value.kind)) ||
      (typeof value.status !== 'string' || !statuses.includes(value.status)) || !timestamp(value.created_at) || !timestamp(value.updated_at) || !objectRecord(value.request) ||
      value.request.operation !== value.kind || !text(value.request.runtime_id) || !objectRecord(value.request.simulation) ||
      value.request.simulation.profile_id !== value.request.runtime_id ||
      (value.stage != null && (typeof value.stage !== 'string' || value.stage.length > 512)) ||
      (value.error != null && (typeof value.error !== 'string' || value.error.length > 2000))) throw uncertain();
  const request = value.request, simulation = request.simulation as Record<string, unknown>;
  if (value.kind === 'policy.run' ? !text(request.artifact_id) || !Number.isInteger(request.timeout_seconds) || Number(request.timeout_seconds) < 30 || Number(request.timeout_seconds) > 7200 || simulation.experimental !== true
    : !text(request.source_id) || request.timeout_seconds !== 600 || simulation.experimental !== false) throw uncertain();
  return { id: value.id, project_id: project, kind: value.kind, status: value.status, created_at: value.created_at, updated_at: value.updated_at,
    stage: value.stage ?? null, error: value.error ?? null, result: null, simulation_target: null,
    request: { operation: value.kind, runtime_id: request.runtime_id, ...(value.kind === 'policy.run' ? { artifact_id: request.artifact_id } : { source_id: request.source_id }), timeout_seconds: request.timeout_seconds,
      simulation: { profile_id: request.runtime_id, experimental: simulation.experimental } } } as SimulationJob;
}
const receiptKey = (project: string) => `firebird:native-simulation-receipt:${project}`;
export function storeSimulationReceipt(job: SimulationJob): void {
  sessionStorage.setItem(receiptKey(job.project_id), JSON.stringify(minimalSimulationReceipt(job, job.project_id)));
}
export function storedSimulationReceipt(project: string): SimulationJob | null {
  const raw = sessionStorage.getItem(receiptKey(project));
  if (raw === null) return null;
  if (raw.length > 16_384) throw new Error('Simulation receipt is too large.');
  return minimalSimulationReceipt(JSON.parse(raw) as unknown, project);
}
