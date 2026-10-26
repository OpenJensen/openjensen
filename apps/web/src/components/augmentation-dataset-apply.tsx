'use client';

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { api, type AugmentationJob, type AugmentationRequest, type DatasetJob } from '@/lib/api';
import { augmentationVideoCameras } from '@/lib/augmentation-source';

export function AugmentationDatasetApply({
  job,
  dataset,
  busy,
  onClose,
  onSubmit,
}: {
  job: AugmentationJob;
  dataset?: DatasetJob;
  busy: boolean;
  onClose: () => void;
  onSubmit: (body: AugmentationRequest) => Promise<void>;
}) {
  const cameras = augmentationVideoCameras(dataset?.result ?? undefined);
  const [selected, setSelected] = useState([job.request.camera_key].filter((key) => cameras.includes(key)));
  const request: AugmentationRequest = {
    operation: 'dataset.augment',
    source_job_id: job.request.source_job_id,
    episode_indices: [],
    camera_key: selected[0] ?? '',
    preset: job.request.preset,
    prompt: job.request.prompt ?? '',
    start_seconds: 0,
    duration_seconds: 10,
    dataset_scope: { recipe_job_id: job.id, camera_keys: selected },
  };
  const plan = useMutation({ mutationFn: () => api.augmentationPlan(job.project_id, request) });
  const start = useMutation({
    mutationFn: async () => {
      if (!plan.data || !request.dataset_scope) throw new Error('Review the dataset plan first.');
      await onSubmit({ ...request, dataset_scope: { ...request.dataset_scope, plan_sha256: plan.data.plan_sha256 } });
    },
  });
  const locked = busy || plan.isPending || start.isPending;
  return (
    <section className="augmentation-dataset-apply" aria-label="Apply augmentation to dataset">
      <div className="augmentation-preview-heading">
        <div>
          <h2>Apply to full dataset</h2>
          <p>
            {dataset?.result?.repo_id ?? job.result?.repo_id} · {dataset?.result?.total_episodes ?? 'All'} full episodes
          </p>
        </div>
        <button type="button" className="text-button" disabled={locked} onClick={onClose}>
          Close dataset plan
        </button>
      </div>
      <p className="augmentation-variant-description">{job.request.prompt || 'Use the saved preset edit.'}</p>
      <p className="augmentation-hint">
        Choose the cameras to augment. Every full episode from each selected camera will be covered.
      </p>
      <fieldset className="augmentation-camera-checks" disabled={locked}>
        <legend>Cameras to augment</legend>
        {cameras.map((key) => (
          <label key={key}>
            <input
              type="checkbox"
              checked={selected.includes(key)}
              onChange={(event) => {
                setSelected((previous) =>
                  event.target.checked ? [...previous, key] : previous.filter((item) => item !== key),
                );
                plan.reset();
                start.reset();
              }}
            />
            {key.replace('observation.images.', '')}
          </label>
        ))}
      </fieldset>
      {!dataset && <p className="error-notice">The source dataset inspection is unavailable.</p>}
      <button
        type="button"
        className="secondary-button"
        disabled={locked || !selected.length || !dataset}
        onClick={() => plan.mutate()}
      >
        {plan.isPending ? 'Checking episode coverage…' : 'Review dataset plan'}
      </button>
      <p className="augmentation-hint">
        Planning checks dataset metadata. It does not generate videos or use generation tokens.
      </p>
      {plan.error && (
        <p role="alert" className="error-notice">
          {plan.error.message}
        </p>
      )}
      {plan.data && (
        <div className="augmentation-plan-summary">
          <strong>
            {plan.data.total_episodes} full episodes · {plan.data.camera_keys.length} cameras
          </strong>
          <p>
            {plan.data.total_source_seconds.toLocaleString()} seconds of source video · {plan.data.total_clips} output
            clips · {plan.data.provider_requests} generation requests
          </p>
          <p>
            Starting this plan sends the selected videos to Google and incurs generation charges. The original dataset
            is kept intact.
          </p>
          <ul>
            {plan.data.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
          <details className="augmentation-provenance">
            <summary>Edits to apply in order ({plan.data.recipe_prompts.length})</summary>
            {plan.data.recipe_prompts.map((prompt, index) => (
              <p className="augmentation-effective-prompt" key={index}>
                {prompt}
              </p>
            ))}
          </details>
          <button type="button" className="primary-button" disabled={locked} onClick={() => start.mutate()}>
            {start.isPending ? 'Starting dataset augmentation…' : 'Generate full-dataset augmentation'}
          </button>
        </div>
      )}
      {start.error && (
        <p role="alert" className="error-notice">
          {start.error.message}
        </p>
      )}
    </section>
  );
}
