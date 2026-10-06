import type { DatasetProfile } from './api';
import type { TrainingModel } from './training-models';

/** A deterministic planning estimate, not a measured peak or admission limit.
 * Catalog budgets include model/optimizer storage. Only the current microbatch
 * is on the GPU: increasing the number of dataset episodes does not raise VRAM.
 */
export function trainingMemory(model: TrainingModel, method: string, batch: number, cameras: string[], profile: DatasetProfile | undefined, prediction = 50) {
  const catalog = model.minimum_gpu_memory_gb ?? model.suggested_gpu_memory_gb ?? (model.id === 'smolvla' ? 16 : null);
  if (catalog === null || !Number.isFinite(batch) || batch < 1 || !cameras.length) return null;
  const smol = model.id === 'smolvla';
  const resident = smol ? method === 'qlora' ? 2.5 : 4 : catalog * 0.7;
  const referenceBatch = smol ? 64 : 4;
  const referenceCameras = model.required_cameras ?? 2;
  const referenceHorizon = model.id === 'act' ? 100 : 50;
  const referenceImages = 512 * 512 * 3 * 4 * referenceBatch * referenceCameras / 1024 ** 3;
  const activation = Math.max(0, catalog - resident - referenceImages) * (batch / referenceBatch) * Math.max(0.5, cameras.length / referenceCameras) * Math.max(1, prediction / referenceHorizon);
  const imageBytes = cameras.reduce((sum, key) => {
    const feature = profile?.features[key] as { shape?: number[] } | undefined;
    const shape = feature?.shape;
    return sum + (shape?.length === 3 && shape.every(value => Number.isFinite(value) && value > 0) ? shape.reduce((a, b) => a * b, 1) * 4 : 512 * 512 * 3 * 4);
  }, 0) * batch;
  const gb = Math.round(Math.max(model.minimum_gpu_memory_gb ?? 0, resident + activation + imageBytes / 1024 ** 3) * 2) / 2;
  const recommended = ([['T4', 16], ['L4', 24], ['A100', 40]] as const).find(([, memory]) => memory >= Math.max(gb, model.minimum_gpu_memory_gb ?? 0))?.[0] ?? null;
  return { gb, recommended, source: 'Catalog model budget at the reference batch (64 for SmolVLA, 4 otherwise), two 512px views unless the model requires another camera count, plus microbatch and image-tensor adjustments. Rounded to 0.5 GB. Includes optimizer storage in the catalog budget. Camera count, image dimensions and action horizon change the estimate; episode count and gradient accumulation do not multiply simultaneous GPU memory. Actual peaks depend on the worker and precision.' };
}
