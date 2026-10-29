'use client';

import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { computeTargets } from '@/lib/compute-targets';
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
  operation,
  onSettings,
}: {
  projectId: string;
  module: string;
  operation?: string;
  onSettings: () => void;
}) {
  const [selections, setSelections] = useState<JobComputeSelection[]>([]);
  const [error, setError] = useState('');
  const configurable = !!operation?.startsWith('policy.');
  const targets = useQuery({
    queryKey: ['compute-targets'],
    queryFn: ({ signal }) => computeTargets.list(signal),
    enabled: configurable,
    retry: false,
  });
  const compatible = (targets.data?.targets ?? []).filter((target) => target.operations.includes(operation ?? ''));
  const selected = selections.find((item) => item.operation === operation);
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
  if (!configurable && !selections.length && !error) return null;
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
      {configurable && (
        <label className="work-job-compute">
          Compute target
          <select
            value={selected?.id ?? ''}
            onChange={(event) => {
              try {
                const target = compatible.find((item) => item.definition.id === event.target.value);
                if (event.target.value && (!target || !target.launchable))
                  throw new Error('Review the selected compute target.');
                const next = target
                  ? {
                      id: target.definition.id,
                      label: target.definition.label,
                      runtimeId: target.definition.runtime_id,
                      operation: operation!,
                    }
                  : null;
                saveJobCompute(projectId, operation!, next);
                setSelections((items) => [
                  ...items.filter((item) => item.operation !== operation),
                  ...(next ? [next] : []),
                ]);
                setError('');
              } catch (cause) {
                setError(cause instanceof Error ? cause.message : 'Compute selection could not be saved.');
              }
            }}
          >
            <option value="">Use module compute settings</option>
            {selected && !compatible.some((item) => item.definition.id === selected.id) && (
              <option value={selected.id} disabled>
                Saved target is unavailable
              </option>
            )}
            {compatible.map((target) => (
              <option key={target.definition.id} value={target.definition.id} disabled={!target.launchable}>
                {target.definition.label}
              </option>
            ))}
          </select>
        </label>
      )}
      {configurable &&
        selected &&
        targets.isSuccess &&
        !compatible.some((target) => target.definition.id === selected.id && target.launchable) && (
          <p role="alert">
            The selected compute target is unavailable. Choose another target or use module compute settings before
            starting a new job.
          </p>
        )}
      {targets.isError && configurable && (
        <p role="alert">Compute targets could not be loaded. Retry before selecting a configured target.</p>
      )}
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
