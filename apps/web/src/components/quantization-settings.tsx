'use client';

import { TrainingHelp } from './training-help';
import { WorkflowChoiceGrid } from './workflow-choice-grid';
import './quantization-settings.css';

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
