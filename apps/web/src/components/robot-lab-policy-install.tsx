'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { Notice } from './robot-lab-shared';
import type { RobotLabState } from './robot-lab-state';

export function RobotLabPolicyInstall({ lab, installed }: { lab: RobotLabState; installed: (id: string) => void }) {
  const [selected, setSelected] = useState('');
  const [name, setName] = useState('');
  const [fps, setFps] = useState('');
  const [units, setUnits] = useState(false);
  const models = useQuery({
    queryKey: ['artifacts', lab.projectId],
    queryFn: () => api.artifacts(lab.projectId!),
    enabled: !!lab.projectId && lab.online,
  });
  const compatible = (models.data ?? []).filter(
    (item) =>
      item.project_id === lab.projectId &&
      ['training_checkpoint', 'native_checkpoint'].includes(item.format) &&
      item.metadata?.architecture === 'act' &&
      item.metadata?.training_backend === 'lerobot' &&
      item.metadata?.reload_verified === true,
  );
  const selectedId = selected || compatible[0]?.id || '';
  const model = compatible.find((item) => item.id === selectedId);
  const title = name || model?.label || '';
  const rate = fps ? Number(fps) : undefined;
  const validRate = rate === undefined || (Number.isInteger(rate) && rate >= 1 && rate <= 40);
  return (
    <details className="robot-lab-transfer">
      <summary>Install from Models</summary>
      <div className="robot-lab-form">
        <p>
          Copy a complete native ACT checkpoint from this project to the USB computer. Installation does not load the
          model or enable motion.
        </p>
        {models.isPending && <p role="status">Checking project models…</p>}
        {models.error && <Notice kind="warning">{models.error.message}</Notice>}
        {!models.isPending && !compatible.length && (
          <Notice>
            No reload-verified native ACT checkpoints are available in this project. Open Models to inspect or import a
            policy.
          </Notice>
        )}
        <label>
          Project model
          <select
            value={selectedId}
            disabled={lab.active || lab.busy}
            onChange={(event) => {
              setSelected(event.target.value);
              setName('');
              setFps('');
              setUnits(false);
            }}
          >
            <option value="">Choose a native ACT model</option>
            {compatible.map((item) => (
              <option key={item.id} value={item.id}>
                {item.run_name || item.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Installed policy name
          <input
            value={title}
            maxLength={100}
            disabled={lab.active || lab.busy}
            onChange={(event) => setName(event.target.value)}
          />
        </label>
        <label>
          Training FPS, if missing from the model
          <input
            type="number"
            min={1}
            max={40}
            value={fps}
            placeholder="Use saved training FPS"
            disabled={lab.active || lab.busy}
            onChange={(event) => setFps(event.target.value)}
          />
        </label>
        <label className="robot-lab-check">
          <input
            type="checkbox"
            checked={units}
            disabled={lab.active || lab.busy}
            onChange={(event) => setUnits(event.target.checked)}
          />
          <span>
            If the model has no saved units, I confirm shoulder pan, shoulder lift, elbow, wrist flex, wrist roll and
            gripper in that order; five joints in degrees and gripper in percent. Use this only for a policy trained
            with those units. Saved training FPS is kept.
          </span>
        </label>
        <small>
          Saved incompatible units or FPS cannot be overridden. Frame-rate and hardware checks still run after
          installation.
        </small>
        <button
          disabled={
            !lab.canAct ||
            lab.active ||
            !model ||
            !title.trim() ||
            !validRate ||
            !lab.projectId ||
            !lab.state?.capabilities?.policy_install
          }
          onClick={() => {
            if (!model || !lab.projectId) return;
            void lab.perform(
              'Policy installed. Load it and check the camera before enabling control.',
              () =>
                api.installRobotPolicy(lab.projectId!, model.id, {
                  name: title.trim(),
                  ...(rate === undefined ? {} : { fps: rate }),
                  confirm_so101_units: units,
                }),
              (result) => installed(result.installed_policy_id),
            );
          }}
        >
          Install policy on USB computer
        </button>
        {!lab.state?.capabilities?.policy_install && (
          <Notice kind="warning">Update the robot control service to install policies from Models.</Notice>
        )}
      </div>
    </details>
  );
}
