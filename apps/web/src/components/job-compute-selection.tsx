'use client';

import { useEffect, useState } from 'react';
import { selectedJobCompute, saveJobCompute, type JobComputeSelection } from '@/lib/job-compute-selection';

const moduleOperations: Record<string, string[]> = {
  training: ['policy.finetune'],
  distillation: ['policy.distill'],
  quantization: ['policy.quantize'],
  evaluation: ['policy.evaluate', 'policy.workflow'],
  simulation: ['policy.run'],
  models: ['policy.import', 'policy.export'],
};

/** The recipe override stays visible wherever a module constructs a new request. */
export function JobComputeSelectionNotice({
  projectId,
  module,
  onSettings,
}: {
  projectId: string;
  module: string;
  onSettings: () => void;
}) {
  const [selections, setSelections] = useState<JobComputeSelection[]>([]);
  const [error, setError] = useState('');
  useEffect(() => {
    if (!projectId) return;
    try {
      setSelections(
        (moduleOperations[module] ?? [])
          .map((operation) => selectedJobCompute(projectId, operation))
          .filter((value): value is JobComputeSelection => value !== null),
      );
    } catch {
      setError('Saved compute selection could not be read. Review compute before submitting another job.');
    }
  }, [projectId, module]);
  if (!selections.length && !error) return null;
  function clear(selection: JobComputeSelection) {
    try {
      saveJobCompute(projectId, selection.operation, null);
      setSelections((previous) => previous.filter((item) => item.operation !== selection.operation));
      setError('');
    } catch {
      setError('Compute selection could not be saved. Restore browser storage before submitting.');
    }
  }
  return (
    <section className="panel work-compute-selection" aria-label="Selected job compute">
      {error && <p role="alert">{error}</p>}
      {selections.map((selection) => (
        <div className="workbench-actions" key={selection.operation}>
          <p>
            New {selection.operation.replace('policy.', '')} jobs use <strong>{selection.label}</strong>.
          </p>
          <button className="text-button" onClick={() => clear(selection)}>
            Use module compute settings
          </button>
        </div>
      ))}
      <button className="text-link" onClick={onSettings}>
        Manage compute targets
      </button>
    </section>
  );
}
