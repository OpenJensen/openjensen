'use client';

import { useRef, useState } from 'react';
import { augmentationClipUrl, type AugmentationJob } from '@/lib/api';
import { augmentationGallery, galleryKey, type GalleryEntry } from '@/lib/augmentation-gallery';

export function AugmentationGallery({
  runs,
  projectId,
  selectedRun,
  onReuse,
  onApply,
  onAdd,
  disabled,
}: {
  runs: AugmentationJob[];
  projectId: string;
  selectedRun?: AugmentationJob;
  onReuse: (job: AugmentationJob, index: number, variant: 'original' | 'augmented') => void;
  onApply: (job: AugmentationJob) => void;
  onAdd: (job: AugmentationJob, index: number) => void;
  disabled: boolean;
}) {
  const groups = augmentationGallery(runs, projectId);
  const [selection, setSelection] = useState({ run: '', key: '' });
  const [playError, setPlayError] = useState('');
  const videos = useRef(new Map<string, HTMLVideoElement>());
  const preferred = selectedRun?.result?.clips[0];
  const group =
    groups.find(
      (item) =>
        item.key ===
        (selection.run === selectedRun?.id ? selection.key : preferred ? galleryKey(selectedRun!, preferred) : ''),
    ) ?? groups.at(-1);
  if (!group) return null;
  const focus = group.entries[0];
  const clip = focus.clip;
  const relatedGroups = groups.filter(
    (item) =>
      item.entries[0].job.result?.repo_id === focus.job.result?.repo_id &&
      item.entries[0].job.result?.revision === focus.job.result?.revision &&
      item.entries[0].job.result?.metadata_sha256 === focus.job.result?.metadata_sha256,
  );
  const label = (id: string) => {
    const index = runs.findIndex((job) => job.id === id);
    return index < 0 ? 'Saved run' : 'Run ' + (runs.length - index);
  };
  function pause() {
    videos.current.forEach((video) => video.pause());
  }
  function seek(time: number, source?: HTMLVideoElement) {
    videos.current.forEach((video) => {
      if (video !== source && video.readyState && Math.abs(video.currentTime - time) > 0.08) video.currentTime = time;
    });
  }
  const cards: { entry: GalleryEntry; original: boolean }[] = [
    ...(group.original ? [{ entry: group.original, original: true }] : []),
    ...group.entries.map((entry) => ({ entry, original: false })),
  ];
  return (
    <section className="augmentation-gallery" aria-label="Episode experiments">
      <div className="augmentation-preview-heading">
        <div>
          <h2>Episode experiments</h2>
          <p>Compare the original and every saved variation together.</p>
        </div>
        <span>
          {cards.length} videos · {group.entries.length} variants
        </span>
      </div>
      <p className="augmentation-hint">
        {focus.job.result?.repo_id} · {clip.camera_key.replace('observation.images.', '')} camera ·{' '}
        {clip.source_start_seconds}s–{clip.source_end_seconds}s
      </p>
      {relatedGroups.length > 1 && (
        <div className="augmentation-clip-picker" role="group" aria-label="Preview episode">
          {relatedGroups.map((item) => {
            const c = item.entries[0].clip;
            const duplicateEpisode =
              relatedGroups.filter((other) => other.entries[0].clip.episode_index === c.episode_index).length > 1;
            const name =
              'Episode ' +
              c.episode_index +
              (duplicateEpisode
                ? ' · ' +
                  c.camera_key.replace('observation.images.', '') +
                  ' · ' +
                  c.source_start_seconds +
                  '–' +
                  c.source_end_seconds +
                  's'
                : '');
            return (
              <button
                key={item.key}
                type="button"
                aria-label={'Preview ' + name.toLowerCase()}
                aria-pressed={group.key === item.key}
                onClick={() => {
                  pause();
                  setSelection({ run: selectedRun?.id ?? '', key: item.key });
                  setPlayError('');
                }}
              >
                {name}
              </button>
            );
          })}
        </div>
      )}
      <div className="augmentation-playback" role="group" aria-label="Compare playback">
        <button
          type="button"
          className="secondary-button"
          onClick={async () => {
            setPlayError('');
            const reference = [...videos.current.values()][0];
            seek(reference?.ended ? 0 : (reference?.currentTime ?? 0));
            const results = await Promise.allSettled([...videos.current.values()].map((video) => video.play()));
            if (results.some((result) => result.status === 'rejected'))
              setPlayError('Some videos could not play. Use their individual controls or download links.');
          }}
        >
          Play all
        </button>
        <button type="button" className="text-button" onClick={pause}>
          Pause all
        </button>
        <button
          type="button"
          className="text-button"
          onClick={() => {
            pause();
            seek(0);
          }}
        >
          Restart all
        </button>
      </div>
      {playError && (
        <p className="augmentation-hint" role="status">
          {playError}
        </p>
      )}
      {!group.original && (
        <p className="augmentation-hint">
          The original source run is unavailable. These are the saved generated variants.
        </p>
      )}
      <div className="augmentation-variant-grid">
        {cards.map(({ entry, original }) => {
          const { job, clip: item } = entry;
          const key = job.id + '-' + item.index + '-' + original;
          const title = original ? 'Original' : label(job.id);
          const current = selectedRun?.id === job.id;
          const mediaLabel = original
            ? 'Original episode ' + item.episode_index
            : current
              ? 'Augmented episode ' + item.episode_index
              : title + ' episode ' + item.episode_index;
          const url = augmentationClipUrl(job.id, item.index, original);
          return (
            <article
              key={key}
              className="augmentation-variant"
              aria-label={title + ' video'}
              data-selected={(current && !original) || undefined}
            >
              <div className="augmentation-variant-title">
                <h3>{title}</h3>
                <span>{original ? 'Source' : 'Variant'}</span>
              </div>
              <video
                ref={(node) => {
                  if (node) videos.current.set(key, node);
                  else videos.current.delete(key);
                }}
                controls
                muted
                playsInline
                preload="metadata"
                aria-label={mediaLabel}
                src={url}
                onSeeked={(event) => seek(event.currentTarget.currentTime, event.currentTarget)}
              />
              {original && <p className="augmentation-variant-description">Unedited source clip</p>}
              {!original && job.request.source_clip?.variant === 'augmented' && (
                <p className="augmentation-hint">Refined from {label(job.request.source_clip.job_id)}</p>
              )}
              <div className="augmentation-variant-actions">
                <a className="text-link" href={url} download>
                  Download {original ? 'original' : 'variant'}
                </a>
                {original ? (
                  <button
                    type="button"
                    className="secondary-button"
                    disabled={disabled || !!job.request.dataset_scope}
                    onClick={() => onReuse(job, item.index, 'original')}
                  >
                    New variation from original
                  </button>
                ) : (
                  <>
                    <button
                      type="button"
                      className="secondary-button"
                      disabled={disabled || !!job.request.dataset_scope}
                      onClick={() => onReuse(job, item.index, 'augmented')}
                    >
                      Refine this result
                    </button>
                    {!job.request.dataset_scope && (
                      <button type="button" className="text-button" disabled={disabled} onClick={() => onApply(job)}>
                        Apply to full dataset
                      </button>
                    )}
                  </>
                )}
              </div>
              {!original && (
                <details className="augmentation-provenance">
                  <summary>Prompt used</summary>
                  <p className="augmentation-variant-description">{job.request.prompt || 'Preset appearance edit'}</p>
                  <details>
                    <summary>Full generation instructions</summary>
                    <p className="augmentation-effective-prompt">{job.result?.prompt}</p>
                  </details>
                </details>
              )}
            </article>
          );
        })}
        <button
          type="button"
          className="augmentation-add-variant"
          disabled={disabled || !group.original || !!group.original.job.request.dataset_scope}
          onClick={() => {
            if (group.original) onAdd(group.original.job, group.original.clip.index);
          }}
        >
          <span className="augmentation-add-symbol" aria-hidden="true">
            +
          </span>
          <strong>Add one more</strong>
          <span>Try another style on this original clip</span>
        </button>
      </div>
    </section>
  );
}
