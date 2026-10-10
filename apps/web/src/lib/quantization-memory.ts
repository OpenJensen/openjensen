import type { PolicyArtifact } from './api';

const GiB = 1024 ** 3;
const positive = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value) && value > 0;

/** Planning headroom for export, CPU packing and verification, not output size.
 * The base is absent from adapter-only checkpoint files. Q4/Q8 are component
 * policies; without tensor inventories they cannot predict whole-model size.
 */
export function quantizationMemory(artifact: PolicyArtifact | undefined, localGpuInference = false) {
  if (!artifact || artifact.metadata?.architecture !== 'smolvla') return null;
  const metadata = artifact.metadata, base = metadata.base_model as { repository?: string } | undefined;
  let floatingBytes: number, basis: string;
  if (positive(metadata.parameter_count) && Number.isSafeInteger(metadata.parameter_count)) {
    floatingBytes = metadata.parameter_count * 4;
    basis = 'Recorded parameter count, materialized at four bytes per parameter.';
  } else if (base?.repository === 'lerobot/smolvla_base') {
    // https://huggingface.co/docs/lerobot/v0.4.3/en/smolvla
    floatingBytes = 450_000_000 * 4;
    basis = 'The standard 450M-parameter SmolVLA base, materialized in FP32. Adapter file size is not the base model size.';
  } else if (artifact.format === 'gguf' && metadata.precision === 'float' && positive(metadata.weight_bytes)) {
    floatingBytes = metadata.weight_bytes * 2;
    basis = 'Recorded floating weight bytes, doubled conservatively to allow BF16-to-FP32 materialization.';
  } else return null;
  if (!Number.isFinite(floatingBytes) || floatingBytes > Number.MAX_SAFE_INTEGER) return null;
  const gpuRequired = artifact.format === 'training_checkpoint' || localGpuInference;
  // Four floating copies allow checkpoint/processor loading, export buffers,
  // GGUF input/output and tensor workspaces; 2 GiB covers libraries and smoke.
  const ramGiB = Math.ceil(floatingBytes * 4 / GiB + 2);
  const gpuGiB = gpuRequired ? Math.ceil(floatingBytes / GiB + 2) : 0;
  return {
    ramGiB, gpuGiB,
    explanation: `${basis} System RAM allows four floating-weight copies plus 2 GiB for libraries and verification. GPU headroom allows floating weights plus 2 GiB when checkpoint export or local GPU inference requires it. Packing itself runs on the CPU. The full floating source is needed before Q4/Q8 packing, so this peak estimate does not shrink with the selected output precision. These are conservative planning assumptions, not measured peaks or fit guarantees; dataset episode count does not multiply model memory.`,
  };
}
