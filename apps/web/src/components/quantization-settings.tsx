'use client';

import type { PolicyArtifact, PolicyOptions } from '@/lib/api';
import { quantizationMemory } from '@/lib/quantization-memory';
import { GpuPicker } from './gpu-picker';
import { TrainingHelp } from './training-help';
import { WorkflowChoiceGrid } from './workflow-choice-grid';
import './quantization-settings.css';

type Runtime = PolicyOptions['runtimes'][number];
const cloudMemory: Record<string, number> = { L4: 24, T4: 16, A100: 40, 'A100-80GB': 80 };

export function QuantizationCompression({ language, vision, onLanguage, onVision }: {
  language: string; vision: boolean; onLanguage: (value: 'Q8_0' | 'Q4_0') => void; onVision: (value: boolean) => void;
}) {
  return <div className="quantization-components">
    <WorkflowChoiceGrid name="language-compression" label="Language compression" value={language}
      help={<TrainingHelp label="Language compression">Packs supported language-backbone weight matrices. The action expert, embeddings, norms and projectors keep their source precision. Q8 uses 8-bit blocks; Q4 uses smaller 4-bit blocks and is experimental.</TrainingHelp>}
      options={[{value:'Q8_0',label:'8-bit',meta:'Q8',icon:'layers'},{value:'Q4_0',label:'4-bit',meta:'Q4 · Experimental',icon:'compress'}]}
      onChange={value=>onLanguage(value as 'Q8_0' | 'Q4_0')} />
    <WorkflowChoiceGrid name="vision-compression" label="Vision compression" value={vision ? 'Q8_0' : 'source'}
      help={<TrainingHelp label="Vision compression">Applies separately to supported vision-encoder weight matrices. Source precision preserves those weights as they are; 8-bit explicitly enables experimental vision Q8. Changing language compression does not change this setting.</TrainingHelp>}
      options={[{value:'source',label:'Source precision',meta:'No change',icon:'layers'},{value:'Q8_0',label:'8-bit',meta:'Q8 · Experimental',icon:'compress'}]}
      onChange={value=>onVision(value==='Q8_0')} />
  </div>;
}

export function QuantizationCompute({ artifact, runtime, runtimes, onChange, disabled, localGpuInference }: {
  artifact?: PolicyArtifact; runtime?: Runtime; runtimes: Runtime[]; onChange: (id: string) => void;
  disabled: boolean; localGpuInference: boolean;
}) {
  const estimate = quantizationMemory(artifact, localGpuInference);
  return <div className="quantization-compute">
    <section className="quantization-memory" aria-label="Estimated quantization memory">
      <div className="quantization-memory-heading"><strong>Estimated memory</strong><TrainingHelp label="Quantization memory estimate" wide>{estimate?.explanation ?? 'An estimate needs a recorded parameter count, the standard SmolVLA base identity, or floating GGUF weight bytes. Adapter-only file size cannot establish the full model size. Estimates cover the job, not the resulting compressed file.'}</TrainingHelp></div>
      {estimate ? <dl><div><dt>GPU memory</dt><dd>{estimate.gpuGiB ? `≈ ${estimate.gpuGiB} GiB` : 'Not used by this job'}</dd></div><div><dt>System RAM</dt><dd>≈ {estimate.ramGiB} GiB</dd></div></dl> : <p>Estimate unavailable</p>}
    </section>
    <GpuPicker label="Compute" value={runtime?.id ?? ''} disabled={disabled || !runtimes.length} onChange={onChange}
      choices={runtimes.map(target=>{
        const memory = target.gpu_memory_mib ? target.gpu_memory_mib / 1024 : cloudMemory[target.accelerator ?? ''];
        return {id:target.id,label:target.accelerator==='A100-80GB' ? 'A100' : target.accelerator ?? target.gpu_name ?? target.label,
          accelerator:target.accelerator ?? undefined,memory:memory ? `${memory} GB` : undefined,
          description:[target.execution==='skypilot' ? 'Google Cloud' : 'Local',target.device==='cuda' ? 'GPU' : 'CPU',target.region].filter(Boolean).join(' · ')};
      })} />
    {!runtimes.length && <p role="status">Connect a compatible worker in Compute settings.</p>}
  </div>;
}
