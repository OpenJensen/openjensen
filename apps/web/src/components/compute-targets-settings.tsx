'use client';

import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { computeTargets, type ComputeTarget } from '@/lib/compute-targets';
import { Icon } from './icon';
import { WorkbenchDisclosure } from './workbench-disclosure';
import './compute-targets-settings.css';

const infrastructureLabels: Record<ComputeTarget['infrastructure'], string> = {
  local: 'Local machine',
  ssh: 'SSH infrastructure',
  gcp: 'Google Cloud',
  runpod: 'RunPod',
  nebius: 'Nebius',
  slurm: 'Slurm cluster',
};
const operationLabels: Record<string, string> = {
  'policy.finetune': 'Fine-tune',
  'policy.distill': 'Distill',
  'policy.quantize': 'Quantize',
  'policy.export': 'Export',
  'policy.import': 'Import',
  'policy.evaluate': 'Evaluate',
  'policy.run': 'Run policy',
  'policy.workflow': 'Policy workflow',
};
const operationLabel = (value: string) => operationLabels[value] ?? value;
function newTarget(): ComputeTarget {
  return {
    id: `target-${crypto.randomUUID().replaceAll('-', '').slice(0, 16)}`,
    label: '',
    adapter: 'local',
    infrastructure: 'local',
    enabled: false,
    runtime_id: '',
    connection_ref: null,
    location: null,
    partition: null,
    account: null,
    qos: null,
    resources: {
      cpus: 2,
      memory_gib: 4,
      accelerator: null,
      gpu_count: 0,
      disk_gib: 100,
      timeout_seconds: 7200,
      concurrency: 1,
    },
    operations: [],
  };
}

export function ComputeTargetsSettings() {
  const client = useQueryClient();
  const [draft, setDraft] = useState<ComputeTarget | null>(null);
  const [notice, setNotice] = useState('');
  const targets = useQuery({
    queryKey: ['compute-targets'],
    queryFn: ({ signal }) => computeTargets.list(signal),
    retry: false,
  });
  function changed() {
    void client.invalidateQueries({ queryKey: ['compute-targets'] });
    void client.invalidateQueries({ queryKey: ['policy-options'] });
  }
  const save = useMutation({
    mutationFn: computeTargets.save,
    onSuccess: (target) => {
      setNotice(`${target.definition.label} saved. ${target.message}`);
      setDraft(null);
      changed();
    },
  });
  const check = useMutation({
    mutationFn: computeTargets.check,
    onSuccess: (target) => {
      setNotice(`${target.definition.label}: ${target.message}`);
      changed();
    },
  });
  const remove = useMutation({
    mutationFn: computeTargets.remove,
    onSuccess: () => {
      setNotice('Compute target removed. Existing job history is retained.');
      changed();
    },
  });
  const toggle = useMutation({ mutationFn: computeTargets.save, onSuccess: () => changed() });
  const busy = save.isPending || check.isPending || remove.isPending || toggle.isPending;
  const runtime = targets.data?.runtime_options.find((item) => item.id === draft?.runtime_id);
  const resource = (key: keyof ComputeTarget['resources'], value: number | string | null) =>
    setDraft((current) => (current ? { ...current, resources: { ...current.resources, [key]: value } } : current));
  const update = <K extends keyof ComputeTarget>(key: K, value: ComputeTarget[K]) =>
    setDraft((current) => (current ? { ...current, [key]: value } : current));
  const errors = [targets.error, save.error, check.error, remove.error, toggle.error].filter(Boolean);
  return (
    <section className="compute-target-settings" aria-label="Compute targets">
      <header className="compute-target-heading">
        <div>
          <h2>Compute targets</h2>
          <p>Choose which machines, clusters, and cloud resources jobs may use.</p>
        </div>
        <button
          type="button"
          className="primary-button"
          disabled={busy}
          onClick={() => {
            save.reset();
            setNotice('');
            setDraft(newTarget());
          }}
        >
          <Icon name="plus" size={15} /> Add compute target
        </button>
      </header>
      {errors.map((error, index) => (
        <p className="error-notice" role="alert" key={index}>
          {error?.message}
        </p>
      ))}
      {notice && <p role="status">{notice}</p>}
      {targets.isPending && <p role="status">Loading compute targets…</p>}
      {!draft && targets.data?.targets.length === 0 && (
        <p>No compute targets configured. Add one to choose where jobs can run.</p>
      )}
      {!draft &&
        targets.data?.targets.map((target) => (
          <article className="compute-target-entry" key={target.definition.id}>
            <div className="compute-target-heading">
              <div>
                <h3>{target.definition.label}</h3>
                <p>
                  {infrastructureLabels[target.definition.infrastructure]} ·{' '}
                  {targets.data?.runtime_options.find((item) => item.id === target.definition.runtime_id)?.label ??
                    'Runtime unavailable'}
                </p>
              </div>
              <span className="cloud-status">{target.status.replaceAll('_', ' ')}</span>
            </div>
            <p>{target.message}</p>
            <p className="compute-target-summary">
              {target.operations.map(operationLabel).join(' · ') || 'No compatible operations'} ·{' '}
              {target.definition.resources.cpus} CPU · {target.definition.resources.memory_gib} GiB memory
              {target.definition.resources.accelerator ? ` · ${target.definition.resources.accelerator}` : ''}
            </p>
            <div className="workbench-actions">
              <button
                type="button"
                className="secondary-button"
                disabled={busy}
                onClick={() => {
                  save.reset();
                  setDraft(structuredClone(target.definition));
                }}
              >
                Edit
              </button>
              <button
                type="button"
                className="secondary-button"
                disabled={busy}
                onClick={() => check.mutate(target.definition.id)}
              >
                {check.isPending && check.variables === target.definition.id ? 'Checking…' : 'Check configuration'}
              </button>
              <button
                type="button"
                className="text-button"
                disabled={busy}
                onClick={() => toggle.mutate({ ...target.definition, enabled: !target.definition.enabled })}
              >
                {target.definition.enabled ? 'Disable' : 'Enable'}
              </button>
              <button
                type="button"
                className="text-button"
                disabled={busy}
                onClick={() => remove.mutate(target.definition.id)}
              >
                Remove
              </button>
            </div>
          </article>
        ))}
      {draft && (
        <form
          className="workbench-form compute-target-form"
          aria-label="Configure compute target"
          onSubmit={(event) => {
            event.preventDefault();
            if (!busy) save.mutate(draft);
          }}
        >
          <div className="compute-target-fields">
            <label>
              Target name
              <input
                required
                maxLength={100}
                value={draft.label}
                onChange={(event) => update('label', event.target.value)}
                placeholder="GPU server"
              />
            </label>
            <label>
              Infrastructure
              <select
                value={draft.infrastructure}
                onChange={(event) => {
                  const infrastructure = event.target.value as ComputeTarget['infrastructure'];
                  setDraft({
                    ...draft,
                    infrastructure,
                    adapter: infrastructure === 'local' ? 'local' : 'skypilot',
                    runtime_id: '',
                    connection_ref: null,
                    location: null,
                    partition: null,
                    account: null,
                    qos: null,
                    operations: [],
                  });
                }}
              >
                {Object.entries(infrastructureLabels).map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Worker runtime
              <select
                required
                value={draft.runtime_id}
                onChange={(event) => {
                  const selected = targets.data?.runtime_options.find((option) => option.id === event.target.value);
                  setDraft({
                    ...draft,
                    runtime_id: event.target.value,
                    operations: [],
                    resources: {
                      ...draft.resources,
                      ...(selected?.device === 'cuda'
                        ? { gpu_count: 1, accelerator: selected.accelerator ?? draft.resources.accelerator }
                        : {}),
                    },
                  });
                }}
              >
                <option value="">Choose a registered runtime</option>
                {targets.data?.runtime_options
                  .filter((option) => draft.adapter === 'skypilot' || !option.remote_only)
                  .map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
              </select>
            </label>
            {draft.adapter === 'skypilot' && (
              <>
                <label>
                  Connection
                  <select
                    value={draft.connection_ref ?? ''}
                    onChange={(event) => update('connection_ref', event.target.value || null)}
                  >
                    <option value="">Default SkyPilot connection</option>
                    {targets.data?.connection_refs.map((reference) => (
                      <option key={reference} value={reference}>
                        {reference}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  {draft.infrastructure === 'ssh'
                    ? 'Registered node pool'
                    : draft.infrastructure === 'slurm'
                      ? 'Registered cluster'
                      : 'Region (optional)'}
                  <input
                    required={draft.infrastructure === 'ssh' || draft.infrastructure === 'slurm'}
                    value={draft.location ?? ''}
                    maxLength={100}
                    pattern="[A-Za-z0-9][A-Za-z0-9_.-]*"
                    onChange={(event) => update('location', event.target.value || null)}
                  />
                </label>
                {draft.infrastructure === 'slurm' && (
                  <>
                    <label>
                      Partition (optional)
                      <input
                        value={draft.partition ?? ''}
                        onChange={(event) => update('partition', event.target.value || null)}
                      />
                    </label>
                    <label>
                      Account (optional)
                      <input
                        value={draft.account ?? ''}
                        onChange={(event) => update('account', event.target.value || null)}
                      />
                    </label>
                    <label>
                      QoS (optional)
                      <input value={draft.qos ?? ''} onChange={(event) => update('qos', event.target.value || null)} />
                    </label>
                  </>
                )}
              </>
            )}
            <p className="compute-target-resource-note">
              Resource requests select suitable capacity. Local CPU limits are advisory; isolated memory and disk limits
              depend on the worker environment.
            </p>
            <label>
              CPUs
              <input
                required
                type="number"
                min={1}
                max={1024}
                value={draft.resources.cpus}
                onChange={(event) => resource('cpus', Number(event.target.value))}
              />
            </label>
            <label>
              Memory (GiB)
              <input
                required
                type="number"
                min={1}
                max={65536}
                value={draft.resources.memory_gib}
                onChange={(event) => resource('memory_gib', Number(event.target.value))}
              />
            </label>
            <label>
              GPU count
              <select
                value={draft.resources.gpu_count}
                onChange={(event) =>
                  setDraft({
                    ...draft,
                    resources: {
                      ...draft.resources,
                      gpu_count: Number(event.target.value),
                      accelerator: event.target.value === '0' ? null : draft.resources.accelerator,
                    },
                  })
                }
              >
                <option value={0}>CPU only</option>
                <option value={1}>One GPU</option>
              </select>
            </label>
            {draft.resources.gpu_count > 0 && (
              <label>
                Accelerator type
                <input
                  required
                  value={draft.resources.accelerator ?? ''}
                  onChange={(event) => resource('accelerator', event.target.value || null)}
                  placeholder="L4"
                />
              </label>
            )}
            <label>
              Disk (GiB)
              <input
                required
                type="number"
                min={10}
                max={65536}
                value={draft.resources.disk_gib}
                onChange={(event) => resource('disk_gib', Number(event.target.value))}
              />
            </label>
            <label>
              Time limit (minutes)
              <input
                required
                type="number"
                min={1}
                max={10080}
                value={draft.resources.timeout_seconds / 60}
                onChange={(event) => resource('timeout_seconds', Number(event.target.value) * 60)}
              />
            </label>
            <label>
              Concurrent jobs
              <input
                required
                type="number"
                min={1}
                max={128}
                value={draft.resources.concurrency}
                onChange={(event) => resource('concurrency', Number(event.target.value))}
              />
            </label>
          </div>
          {runtime && (
            <WorkbenchDisclosure title="Allowed operations">
              <p>Leave all unchecked to allow every operation supported by this runtime and destination.</p>
              <div className="compute-target-operations">
                {runtime.operations
                  .filter(
                    (operation) =>
                      draft.adapter === 'local' ||
                      ['policy.finetune', 'policy.quantize', 'policy.evaluate', 'policy.run'].includes(operation),
                  )
                  .map((operation) => (
                    <label className="workflow-check" key={operation}>
                      <input
                        type="checkbox"
                        checked={draft.operations.includes(operation)}
                        onChange={(event) =>
                          update(
                            'operations',
                            event.target.checked
                              ? [...draft.operations, operation]
                              : draft.operations.filter((item) => item !== operation),
                          )
                        }
                      />
                      {operationLabel(operation)}
                    </label>
                  ))}
              </div>
            </WorkbenchDisclosure>
          )}
          <label className="workflow-check">
            <input
              type="checkbox"
              checked={draft.enabled}
              onChange={(event) => update('enabled', event.target.checked)}
            />
            Allow jobs on this target
          </label>
          {draft.adapter === 'skypilot' && (
            <p>
              Provider sign-in and registered infrastructure are managed by SkyPilot. Enabling this target allows jobs
              to use its configured resources.
            </p>
          )}
          <div className="workbench-actions">
            <button
              type="submit"
              className="primary-button"
              disabled={busy || !draft.label.trim() || !draft.runtime_id}
            >
              {save.isPending ? 'Saving…' : 'Save compute target'}
            </button>
            <button
              type="button"
              className="secondary-button"
              disabled={save.isPending}
              onClick={() => {
                setDraft(null);
                save.reset();
              }}
            >
              Cancel
            </button>
          </div>
        </form>
      )}
    </section>
  );
}
