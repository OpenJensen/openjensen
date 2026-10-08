import type { Job } from './api';
import { measuredNativeReport } from './native-quantization';

export type QuantizationComparison = {
  reference: string;
  description: string;
  samples: { label: string; rmse: number; maximum: number; coordinates: number }[];
  sourceBytes: number;
  packedBytes: number;
  sizeLabel: string;
};
const object = (value: unknown): Record<string, unknown> | null => value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null;
const nonnegative = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0;
const positiveInteger = (value: unknown): value is number => Number.isSafeInteger(value) && Number(value) > 0;
const hash = (value: unknown) => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);

/** Only paired reports from the selected job's source can establish action drift. */
export function quantizationComparison(job: Job, reports: Record<string, unknown>[] = []): QuantizationComparison | null {
  if (job.status !== 'succeeded' || job.kind !== 'policy.quantize') return null;
  const request = object(job.request);
  for (const report of reports) {
    const native = measuredNativeReport(report, job);
    if (native && native.drift_from_fp32.every(row => nonnegative(row.raw.rmse) && nonnegative(row.raw.maximum_absolute_difference) && row.raw.maximum_absolute_difference >= row.raw.rmse)) return {
      reference: 'Original FP32',
      description: `Native ACT outputs on ${native.drift_from_fp32.length} generated observations, across each full ${native.prediction_horizon} × 6 action chunk.`,
      samples: native.drift_from_fp32.map(row => ({ label: `Seed ${row.seed}`, rmse: row.raw.rmse, maximum: row.raw.maximum_absolute_difference, coordinates: row.raw.coordinates })),
      sourceBytes: native.source_weight_bytes,
      packedBytes: native.packed_weight_bytes,
      sizeLabel: 'Weight bytes',
    };
    const value = object(report.comparison);
    if (!value || report.source_artifact_id !== request?.artifact_id || !hash(report.source_manifest_sha256) ||
        value.schema_version !== 1 || value.scope !== 'paired_synthetic_native_actions' ||
        value.reference !== 'floating_gguf_before_quantization' || value.backend !== 'cpu' || value.samples !== 1 ||
        value.validation_loss !== null || value.task_success !== null ||
        !['input_sha256', 'executable_sha256', 'source_model_sha256', 'quantized_model_sha256'].every(key => hash(value[key])) ||
        !['action_rmse', 'action_mse', 'action_mae', 'action_max_abs_difference'].every(key => nonnegative(value[key])) ||
        !['coordinates', 'action_chunk_size', 'real_action_dim', 'source_file_bytes', 'quantized_file_bytes'].every(key => positiveInteger(value[key])) ||
        value.coordinates !== Number(value.action_chunk_size) * Number(value.real_action_dim) ||
        Number(value.action_max_abs_difference) < Number(value.action_rmse)) continue;
    return {
      reference: 'Original floating GGUF',
      description: `One native CPU prediction on identical fixed images, language, state and noise. Compares ${value.coordinates} active action coordinates; padding is excluded.`,
      samples: [{ label: 'Fixed input', rmse: Number(value.action_rmse), maximum: Number(value.action_max_abs_difference), coordinates: Number(value.coordinates) }],
      sourceBytes: Number(value.source_file_bytes),
      packedBytes: Number(value.quantized_file_bytes),
      sizeLabel: 'Model file bytes',
    };
  }
  return null;
}

export function pooledActionRmse(comparison: QuantizationComparison): number {
  const coordinates = comparison.samples.reduce((sum, sample) => sum + sample.coordinates, 0);
  const largest = Math.max(...comparison.samples.map(sample => sample.rmse));
  return largest === 0 ? 0 : largest * Math.sqrt(comparison.samples.reduce((sum, sample) => sum + (sample.rmse / largest) ** 2 * sample.coordinates, 0) / coordinates);
}
