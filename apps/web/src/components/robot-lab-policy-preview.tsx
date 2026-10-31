'use client';

import { useState } from 'react';
import { robotApi } from '@/lib/robot-api';
import { Notice } from './robot-lab-shared';
import type { RobotLabState } from './robot-lab-state';

export function RobotLabPolicyPreview({
  lab,
  selectedId,
  requiresTask,
}: {
  lab: RobotLabState;
  selectedId: string;
  requiresTask: boolean;
}) {
  const [values, setValues] = useState([0, 0, 0, 0, 0, 0]);
  const [image, setImage] = useState<string | null>(null);
  const [imageError, setImageError] = useState<string | null>(null);
  const policy = lab.state?.policy;
  const ready =
    !!lab.state?.capabilities?.policy_preview &&
    lab.canAct &&
    !lab.active &&
    policy?.phase === 'ready' &&
    policy.policy_id === selectedId &&
    (!requiresTask || !!lab.task.trim());
  const task = requiresTask ? lab.task : undefined;
  const preview = policy?.preview;
  const run = (input: Parameters<typeof robotApi.previewPolicy>[0]) =>
    void lab.perform('Prediction preview ready.', () =>
      robotApi.previewPolicy({ ...input, ...(task ? { task } : {}) }),
    );
  return (
    <div className="robot-lab-form">
      {requiresTask ? (
        <label>
          Task instruction
          <textarea
            value={lab.task}
            maxLength={1000}
            disabled={lab.busy || lab.active}
            onChange={(event) => lab.setTask(event.target.value)}
            placeholder="Describe what the policy should do."
          />
        </label>
      ) : (
        <p className="robot-lab-subtle">ACT inference uses images and state values.</p>
      )}
      <h3>Try inference</h3>
      {!lab.state?.capabilities?.policy_preview && (
        <Notice kind="warning">Update the robot control service to preview policies without arm control.</Notice>
      )}
      <p>
        Preview model output without connecting or moving robot arms. Output values use the model’s own coordinates and
        units.
      </p>
      <button disabled={!ready} onClick={() => run({ synthetic: true })}>
        Try with test inputs
      </button>
      <small>
        Explicit test: a blank image and six zero state values. This checks inference; it does not test task success.
      </small>
      <details>
        <summary>Use an observation image and state</summary>
        <div className="robot-lab-form">
          <label>
            Observation image
            <input
              type="file"
              accept="image/jpeg,image/png"
              disabled={lab.busy || lab.active}
              onChange={(event) => {
                const file = event.target.files?.[0];
                setImage(null);
                setImageError(null);
                if (!file) return;
                if (!['image/jpeg', 'image/png'].includes(file.type) || file.size > 2 * 1024 * 1024) {
                  setImageError('Choose a JPEG or PNG image smaller than 2 MB.');
                  return;
                }
                const reader = new FileReader();
                reader.onload = () => setImage(String(reader.result).split(',')[1] ?? null);
                reader.onerror = () => setImageError('The observation image could not be read.');
                reader.readAsDataURL(file);
              }}
            />
          </label>
          {imageError && <Notice kind="warning">{imageError}</Notice>}
          <div className="robot-lab-fields">
            {values.map((value, index) => (
              <label key={index}>
                State {index + 1}
                <input
                  type="number"
                  step="any"
                  value={value}
                  disabled={lab.busy || lab.active}
                  onChange={(event) =>
                    setValues((current) => current.map((item, i) => (i === index ? Number(event.target.value) : item)))
                  }
                />
              </label>
            ))}
          </div>
          <small>Supply state values in this model’s expected order and units.</small>
          <div className="robot-lab-actions">
            <button
              disabled={!ready || !image || !values.every(Number.isFinite)}
              onClick={() => {
                if (image) run({ state: values, jpeg: image });
              }}
            >
              Preview uploaded observation
            </button>
            <button
              disabled={!ready || !lab.availableCameras.length || !values.every(Number.isFinite)}
              onClick={() => run({ state: values, camera_index: lab.cameraIndex })}
            >
              Preview selected camera
            </button>
          </div>
        </div>
      </details>
      {lab.pending === 'Prediction preview ready.' && <button onClick={lab.stop}>Cancel preview</button>}
      {preview && preview.policy_id === policy?.policy_id && (
        <div role="status">
          <strong>Latest prediction · {preview.inference_ms} ms · no movement</strong>
          <p className="robot-lab-subtle">{preview.source}</p>
          <div className="robot-lab-fields">
            {preview.action.map((value, index) => (
              <span key={index}>
                Output {index + 1}: {value.toFixed(4)}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
