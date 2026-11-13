'use client';

import { useState } from 'react';
import { robotApi } from '@/lib/robot-api';
import { Notice } from './robot-lab-shared';
import type { RobotLabState } from './robot-lab-state';

export function RobotLabPolicyPreview({
  lab,
  selectedId,
  requiresTask,
  cameras,
}: {
  lab: RobotLabState;
  selectedId: string;
  requiresTask: boolean;
  /** One camera for each view of a policy trained on several; absent for one view. */
  cameras?: Record<string, number>;
}) {
  const [values, setValues] = useState([0, 0, 0, 0, 0, 0]);
  const [image, setImage] = useState<string | null>(null);
  const [imageError, setImageError] = useState<string | null>(null);
  const policy = lab.state?.policy;
  const loaded = policy?.phase === 'ready' && policy.policy_id === selectedId;
  const ready =
    !!lab.state?.capabilities?.policy_preview &&
    lab.canAct &&
    !lab.active &&
    loaded &&
    (!requiresTask || !!lab.task.trim());
  const task = requiresTask ? lab.task : undefined;
  const preview = policy?.preview;
  const run = (input: Parameters<typeof robotApi.previewPolicy>[0]) =>
    void lab.perform('Prediction preview ready.', () =>
      robotApi.previewPolicy({ ...input, ...(task ? { task } : {}) }),
    );
  return (
    <div className="robot-lab-form" role="group" aria-label="Try inference">
      {!lab.state?.capabilities?.policy_preview && (
        <Notice kind="warning">Update the robot control service to preview policies without arm control.</Notice>
      )}
      <p>The policy predicts one action and nothing moves. Values are in the model’s own units.</p>
      <div className="robot-lab-actions">
        {!loaded && (
          <button
            className="secondary-button"
            disabled={!lab.canAct || lab.active || !selectedId}
            onClick={() => void lab.perform('Policy loaded.', () => robotApi.loadPolicy(selectedId))}
          >
            {policy?.phase === 'loading' ? 'Loading policy…' : 'Load policy'}
          </button>
        )}
        <button className="secondary-button" disabled={!ready} onClick={() => run({ synthetic: true })}>
          Try with test inputs
        </button>
      </div>
      <small>Test inputs are a blank image and six zeros: this shows the policy runs, not that it does the task.</small>
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
              className="secondary-button"
              disabled={!ready || !image || !values.every(Number.isFinite)}
              onClick={() => {
                if (image) run({ state: values, jpeg: image });
              }}
            >
              Preview uploaded observation
            </button>
            <button
              className="secondary-button"
              disabled={!ready || !lab.availableCameras.length || !values.every(Number.isFinite)}
              onClick={() => run({ state: values, camera_index: lab.cameraIndex, ...(cameras ? { cameras } : {}) })}
            >
              {cameras ? 'Preview selected cameras' : 'Preview selected camera'}
            </button>
          </div>
        </div>
      </details>
      {lab.pending === 'Prediction preview ready.' && (
        <button className="secondary-button" onClick={lab.stop}>
          Cancel preview
        </button>
      )}
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
