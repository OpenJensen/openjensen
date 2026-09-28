import type { Job } from './api';
import { replayableNativeOutput, type PackedArtifact, type QuantizationReport } from './native-quantization';
import { nativeInput, type NativeArtifact, type SimulationProfile } from './native-simulation';

/** Navigation identity only. Profile selection, admission and paid consent remain in Run. */
export type SimulationHandoff = { projectId: string; artifactId: string; jobId: string; manifestSha256: string; modelId: string };
const sha = (value: unknown): value is string => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
export function simulationHandoffKey(value: SimulationHandoff | undefined, projectId: string): string | null {
  return value && value.projectId === projectId && !!projectId && typeof value.artifactId === 'string' && !!value.artifactId && typeof value.jobId === 'string' && !!value.jobId && sha(value.manifestSha256) && typeof value.modelId === 'string' && /^sha256:[a-f0-9]{64}$/.test(value.modelId)
    ? JSON.stringify([value.projectId, value.artifactId, value.jobId, value.manifestSha256, value.modelId]) : null;
}
export function quantizedSimulationHandoff(artifact: PackedArtifact, job: Job, report: QuantizationReport | null): SimulationHandoff | null {
  if (!replayableNativeOutput(artifact, job, report) || !sha(artifact.manifest_sha256) || !report) return null;
  const value = { projectId: artifact.project_id, artifactId: artifact.id, jobId: artifact.job_id, manifestSha256: artifact.manifest_sha256, modelId: report.model_id };
  return simulationHandoffKey(value, job.project_id) ? value : null;
}
export function simulationHandoffArtifact(value: SimulationHandoff, projectId: string, artifacts: NativeArtifact[]): NativeArtifact | null {
  if (!simulationHandoffKey(value, projectId)) return null;
  const matches = artifacts.filter(item => item.id === value.artifactId);
  if (matches.length !== 1) return null;
  const item = matches[0];
  return item.project_id === projectId && item.job_id === value.jobId && item.manifest_sha256 === value.manifestSha256 && item.format === 'native_quantized' && item.metadata?.architecture === 'act' && item.metadata?.format === 'firebird_quant' && item.metadata?.model_id === value.modelId && item.metadata?.storage !== 'gcs' && !item.metadata?.remote && !item.metadata?.remote_uri ? item : null;
}
export function simulationHandoffProfiles(artifact: NativeArtifact, profiles: SimulationProfile[]): SimulationProfile[] {
  return profiles.filter(profile => profile.experimental === true && profile.task_object === 'cup' && profile.architectures.length === 1 && profile.architectures[0] === 'act' && profile.policy_runtime === 'packed-act-cpu' && profile.policy_device === 'cpu' && profile.policy_formats?.length === 1 && profile.policy_formats[0] === 'firebird_quant' && profile.provider === 'gcp' && profile.accelerators?.length === 1 && profile.accelerators[0] === 'L4' && nativeInput(artifact, artifact.project_id, profile));
}
