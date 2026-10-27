import type { Job, PolicyRequest } from '@/lib/api';

export type SettingsTab = 'compute' | 'modules' | 'providers' | 'cloud-apis' | 'settings' | 'diagnostics';

export function policyRequest(job?: Job): PolicyRequest | undefined {
  return job && 'runtime_id' in job.request ? (job.request as PolicyRequest) : undefined;
}

export function precisionName(precision: unknown): string {
  if (precision === 'float') return 'Floating point';
  if (typeof precision === 'string') return precision.replace('_0', '');
  if (precision && typeof precision === 'object' && 'language' in precision) {
    const language = String(precision.language).replace('_0', '');
    return 'vision' in precision && precision.vision ? `${language} + vision Q8` : language;
  }
  return '—';
}

export function sizeLabel(bytes: number): string {
  return bytes >= 1024 ** 3
    ? `${(bytes / 1024 ** 3).toLocaleString(undefined, { maximumFractionDigits: 2 })} GiB`
    : `${(bytes / 1024 ** 2).toLocaleString(undefined, { maximumFractionDigits: 1 })} MiB`;
}

export function currentStage(job: Job, phase = job.stage): string {
  if (job.status === 'succeeded') return 'Completed';
  if (job.status === 'failed') return 'Failed';
  if (job.status === 'cancelled') return 'Cancelled';
  if (job.status === 'interrupted') return 'Interrupted';
  if (job.status === 'queued') return 'Queued';
  const labels: Record<string, string> = {
    preparing: 'Preparing worker',
    setup: 'Installing dependencies',
    compiling: 'Compiling native engine',
    exporting: 'Merging checkpoint',
    converting: 'Converting policy',
    quantizing: 'Quantizing weights',
    quantize: 'Quantizing weights',
    verifying: 'Checking model',
    saving: 'Saving output',
    finishing: 'Finalizing',
    evaluating: 'Evaluating policy',
    running: 'Running policy',
    'training-export': 'Exporting checkpoint',
    'floating-conversion': 'Converting policy',
    baseline: 'Checking reference',
  };
  if (phase && labels[phase]) return labels[phase];
  if (phase?.startsWith('quantize-')) return 'Quantizing weights';
  if (phase?.includes('evaluation') || phase?.startsWith('evaluate-')) return 'Evaluating policy';
  return job.kind === 'policy.quantize' || job.kind === 'policy.workflow'
    ? 'Quantizing policy'
    : job.kind === 'policy.evaluate'
      ? 'Evaluating policy'
      : 'Running policy';
}

export function objectValue(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

export function finiteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}
