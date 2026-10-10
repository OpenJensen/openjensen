'use client';

import type { Job, PolicyArtifact } from '@/lib/api';
import { modelActionIssue, modelDataset, modelFormat, modelRunGroups, modelRunName } from '@/lib/model-library';
import { isCloudArtifact } from '@/lib/checkpoints';
import './model-version-picker.css';

export function ModelVersionPicker({ runId, projectId, artifacts, jobs, value, onChange, label = 'My model' }: {
  runId: string; projectId: string; artifacts: PolicyArtifact[]; jobs: Job[]; value: string;
  onChange: (id: string) => void; label?: string;
}) {
  const versions = modelRunGroups(artifacts.filter(item => item.project_id === projectId && item.job_id === runId))[0]?.models ?? [];
  const selected = versions.find(item => item.id === value), latest = versions[0];
  return <fieldset className="model-version-picker"><legend>{label}</legend>
    <div className="model-version-heading"><strong>{selected ? modelRunName(selected) : latest ? modelRunName(latest) : 'Selected model unavailable'}</strong><small>Run {runId.slice(0, 8)}</small></div>
    <p>{(selected ?? latest) ? modelDataset((selected ?? latest)!, artifacts, jobs) : 'Loading model records…'}</p>
    {selected && <small>{isCloudArtifact(selected) ? 'Cloud storage' : 'Local storage'}</small>}
    <label>Checkpoint<select aria-label="Checkpoint" value={value} onChange={event => onChange(event.target.value)}>
      {!selected && <option value={value} disabled>{value ? 'Selected checkpoint unavailable' : 'Choose a checkpoint'}</option>}
      {versions.map((artifact, index) => <option key={artifact.id} value={artifact.id} disabled={!!modelActionIssue(artifact, 'quantize')}>
        {index === 0 ? 'Latest checkpoint · ' : ''}{modelFormat(artifact)} · {artifact.label}
      </option>)}
    </select></label>
  </fieldset>;
}
