'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api, type PolicyArtifact } from '@/lib/api';
import { checkpointStep } from '@/lib/checkpoints';
import { runName } from '@/lib/run-names';
import { useProjectDatasets } from '@/lib/use-project-datasets';
import type { RobotPolicyCatalog } from '@/lib/robot-api';
import { Notice, readable } from './robot-lab-shared';
import type { RobotLabState } from './robot-lab-state';

type InstalledPolicy = RobotPolicyCatalog['policies'][number];

/**
 * Trained models of this project that are not on the robot computer yet. One click
 * copies a model over, ready for the arm; the rarely needed settings sit under Options.
 */
export function RobotLabPolicyInstall({
  lab,
  policies,
  installed,
}: {
  lab: RobotLabState;
  policies: InstalledPolicy[];
  installed: (id: string) => void;
}) {
  const [everyCheckpoint, setEveryCheckpoint] = useState(false);
  const [options, setOptions] = useState<string | null>(null);
  const [installing, setInstalling] = useState<string | null>(null);
  const [failure, setFailure] = useState<{ id: string; message: string } | null>(null);
  const [name, setName] = useState('');
  const [fps, setFps] = useState('');
  const [units, setUnits] = useState(false);
  const [offline, setOffline] = useState(false);
  const models = useQuery({
    queryKey: ['artifacts', lab.projectId],
    queryFn: () => api.artifacts(lab.projectId!),
    enabled: !!lab.projectId && lab.online,
  });
  // The run's jobs and datasets are read only to name each model: two runs of one recipe
  // are otherwise listed under the same name.
  const jobs = useQuery({
    queryKey: ['jobs', lab.projectId],
    queryFn: () => api.jobs(lab.projectId!),
    enabled: !!lab.projectId && lab.online,
    retry: false,
    // A run that finished a moment ago must be listed by name as soon as this opens.
    refetchOnMount: 'always',
  });
  const library = useProjectDatasets(lab.projectId, lab.online);
  const modelName = (item: PolicyArtifact, date: boolean) => {
    const job = (jobs.data ?? []).find((entry) => entry.project_id === item.project_id && entry.id === item.job_id);
    if (!job || job.kind !== 'policy.finetune') return item.run_name || item.label;
    const step = checkpointStep(item);
    return [
      runName(job, { jobs: jobs.data ?? [], artifacts: models.data ?? [], library }, { date }),
      step !== null ? `step ${step.toLocaleString()}` : null,
    ]
      .filter(Boolean)
      .join(' · ');
  };
  const compatible = (models.data ?? []).filter(
    (item) =>
      item.project_id === lab.projectId &&
      ['training_checkpoint', 'native_checkpoint'].includes(item.format) &&
      ['act', 'smolvla'].includes(String(item.metadata?.architecture)) &&
      item.metadata?.training_backend === 'lerobot' &&
      item.metadata?.reload_verified === true,
  );
  // A model already copied over is listed above as a policy, not offered again.
  const onRobot = (item: PolicyArtifact) =>
    policies.some(
      (policy) =>
        (!!policy.source_manifest_sha256 && policy.source_manifest_sha256 === item.manifest_sha256) ||
        policy.name === modelName(item, false),
    );
  const waiting = compatible.filter((item) => !onRobot(item));
  // A run saves several checkpoints; its last one is the model people mean.
  const latest = waiting.filter((item) => {
    const step = checkpointStep(item) ?? 0;
    return !waiting.some(
      (other) => other !== item && other.job_id === item.job_id && (checkpointStep(other) ?? 0) > step,
    );
  });
  const listed = everyCheckpoint ? waiting : latest;
  const supported = !!lab.state?.capabilities?.policy_install && !!lab.state?.capabilities?.native_policy_adapters;
  const rate = fps ? Number(fps) : undefined;
  const validRate = rate === undefined || (Number.isInteger(rate) && rate >= 1 && rate <= 40);

  async function install(model: PolicyArtifact, custom: boolean) {
    if (!lab.projectId) return;
    const title = ((custom && name.trim()) || modelName(model, false)).slice(0, 100);
    setInstalling(model.id);
    setFailure(null);
    try {
      const result = await api.installRobotPolicy(lab.projectId, model.id, {
        name: title,
        ...(custom && rate !== undefined ? { fps: rate } : {}),
        confirm_so101_units: custom && units,
        offline_only: custom && offline,
      });
      setOptions(null);
      installed(result.installed_policy_id);
    } catch (error) {
      const message = readable(error);
      setFailure({ id: model.id, message });
      // A model saved without its units or frame rate needs them stated once.
      if (/confirm six|frame rate/i.test(message)) setOptions(model.id);
    } finally {
      setInstalling(null);
    }
  }

  if (!lab.projectId || (!models.isPending && !compatible.length && !models.error)) return null;
  return (
    <section className="robot-lab-policy-models" aria-label="Trained models">
      {models.error && <Notice kind="warning">{models.error.message}</Notice>}
      {listed.length > 0 && <span className="robot-lab-field-label">Trained models not on this computer yet</span>}
      {listed.map((model) => (
        <div className="robot-lab-policy-model" key={model.id}>
          <div className="robot-lab-policy-model-row">
            <span>
              <strong>{modelName(model, true)}</strong>
              <small>
                {String(model.metadata?.architecture) === 'smolvla' ? 'SmolVLA' : 'ACT'}
                {model.file_bytes ? ` · ${(model.file_bytes / 1024 ** 3).toFixed(1)} GB` : ''}
              </small>
            </span>
            <button
              className="secondary-button"
              aria-label={`Install ${modelName(model, false)}`}
              disabled={!lab.canAct || lab.active || !supported || !!installing}
              onClick={() => void install(model, false)}
            >
              {installing === model.id ? 'Copying to the robot computer…' : 'Install'}
            </button>
            <button
              className="text-button"
              aria-expanded={options === model.id}
              disabled={!!installing}
              onClick={() => {
                setOptions(options === model.id ? null : model.id);
                setName('');
                setFps('');
                setUnits(false);
                setOffline(false);
              }}
            >
              Options
            </button>
          </div>
          {installing === model.id && (
            <p className="robot-lab-subtle" role="status">
              A model kept in cloud storage is fetched first. On a slow connection that can take half an hour; stay on
              this page until it finishes.
            </p>
          )}
          {failure?.id === model.id && <Notice kind="warning">{failure.message}</Notice>}
          {options === model.id && (
            <div className="robot-lab-form robot-lab-policy-options">
              <label>
                Installed policy name
                <input
                  value={name}
                  placeholder={modelName(model, false)}
                  maxLength={100}
                  disabled={!!installing}
                  onChange={(event) => setName(event.target.value)}
                />
              </label>
              <label className="robot-lab-check">
                <input
                  type="checkbox"
                  checked={offline}
                  disabled={!!installing}
                  onChange={(event) => setOffline(event.target.checked)}
                />
                <span>Offline inference only (no arm control)</span>
              </label>
              {!offline && (
                <>
                  <label>
                    Training FPS, if missing from the model
                    <input
                      type="number"
                      min={1}
                      max={40}
                      value={fps}
                      placeholder="Use saved training FPS"
                      disabled={!!installing}
                      onChange={(event) => setFps(event.target.value)}
                    />
                  </label>
                  <label className="robot-lab-check">
                    <input
                      type="checkbox"
                      checked={units}
                      disabled={!!installing}
                      onChange={(event) => setUnits(event.target.checked)}
                    />
                    <span>
                      If the model has no saved units, I confirm shoulder pan, shoulder lift, elbow, wrist flex, wrist
                      roll and gripper in that order; five joints in degrees and gripper in percent.
                    </span>
                  </label>
                </>
              )}
              <button
                className="secondary-button"
                disabled={!lab.canAct || lab.active || !supported || !!installing || !validRate}
                onClick={() => void install(model, true)}
              >
                Install with these options
              </button>
            </div>
          )}
        </div>
      ))}
      {waiting.length > latest.length && (
        <button className="text-button" onClick={() => setEveryCheckpoint(!everyCheckpoint)}>
          {everyCheckpoint ? 'Show only the last checkpoint of each run' : 'Show earlier checkpoints'}
        </button>
      )}
      {!supported && compatible.length > 0 && (
        <Notice kind="warning">Update the robot control service to install policies from Models.</Notice>
      )}
    </section>
  );
}
