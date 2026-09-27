import type { Job, JobEvent } from './api';

export function runLabel(job: Job) {
  if (job.kind === 'policy.finetune' || ('training' in job.request && job.request.training)) return 'Training';
  return ({ 'policy.evaluate': 'Evaluation', 'policy.quantize': 'Compression', 'policy.workflow': 'Policy workflow', 'policy.export': 'Export', 'policy.import': 'Import', 'policy.run': 'Policy run' } as Record<string, string>)[job.kind] ?? 'Run';
}

export function runSummary(job: Job, phase = job.stage, status: string = job.status) {
  const label = runLabel(job);
  if (status === 'failed') return `${label} stopped`;
  if (status === 'cancelled') return `${label} cancelled`;
  if (status === 'interrupted') return `${label} interrupted`;
  if (status === 'succeeded') return `${label} complete`;
  if (status === 'queued') return phase === 'preparing' ? 'Preparing your GPU' : `${label} queued`;
  if (phase === 'preparing') return 'Preparing your GPU';
  if (phase === 'validation') return 'Validating the model';
  if (phase === 'checkpoint') return 'Saving a checkpoint';
  if (phase === 'verifying') return 'Verifying the saved model';
  return `${label} in progress`;
}

export function observedProgress(events: JobEvent[]) {
  let step: number | null = null;
  let checkpoint: number | null = null;
  for (const event of events) {
    const trainingMatch = /\bOptimizer step (\d+)\b/i.exec(event.message);
    const checkpointMatch = /\bCheckpoint (\d+) saved\b/i.exec(event.message);
    if (trainingMatch && Number.isSafeInteger(Number(trainingMatch[1]))) step = Math.max(step ?? 0, Number(trainingMatch[1]));
    if (checkpointMatch && Number.isSafeInteger(Number(checkpointMatch[1]))) {
      checkpoint = Math.max(checkpoint ?? 0, Number(checkpointMatch[1]));
      step = Math.max(step ?? 0, Number(checkpointMatch[1]));
    }
  }
  return { step, checkpoint };
}

export function conciseRunError(error?: string | null) {
  if (!error) return null;
  if (/out of memory|CUDA.*memory/i.test(error)) return 'The GPU ran out of memory. Try a smaller batch size.';
  if (/timed? ?out|timeout/i.test(error)) return 'The run reached its time limit.';
  return 'The run stopped before completion. Open the details for the recorded error.';
}
