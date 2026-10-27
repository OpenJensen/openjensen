import type { AugmentationJob } from './api';

export type GalleryClip = NonNullable<AugmentationJob['result']>['clips'][number];
export type GalleryEntry = { job: AugmentationJob; clip: GalleryClip };
export type GalleryGroup = { key: string; original?: GalleryEntry; entries: GalleryEntry[] };

export function galleryKey(job: AugmentationJob, clip: GalleryClip) {
  return JSON.stringify([
    job.project_id,
    job.result?.repo_id,
    job.result?.revision,
    job.result?.metadata_sha256,
    clip.episode_index,
    clip.camera_key,
    clip.source_start_seconds,
    clip.source_end_seconds,
  ]);
}

/** Group the same source interval across all runs, never treating a refinement input as the original. */
export function augmentationGallery(runs: AugmentationJob[], project: string): GalleryGroup[] {
  const completed = runs.filter((job) => job.project_id === project && job.status === 'succeeded' && job.result);
  const groups = new Map<string, GalleryGroup>();
  for (const job of completed.sort((a, b) => a.created_at.localeCompare(b.created_at) || a.id.localeCompare(b.id))) {
    for (const clip of job.result!.clips) {
      const key = galleryKey(job, clip);
      const group = groups.get(key) ?? { key, entries: [] };
      group.entries.push({ job, clip });
      if (!job.request.source_clip && !group.original) group.original = { job, clip };
      groups.set(key, group);
    }
  }
  return [...groups.values()];
}
