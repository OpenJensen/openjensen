import { isDatasetJob, type DatasetJob, type DatasetProfile, type Job } from './api';

export function augmentationVideoCameras(profile?: DatasetProfile): string[] {
  return Object.entries(profile?.features ?? {}).filter(([, feature]) =>
    feature && typeof feature === 'object' && 'dtype' in feature && feature.dtype === 'video',
  ).map(([key]) => key);
}

export function isAugmentationSource(job: Job): job is DatasetJob & { result: DatasetProfile } {
  return isDatasetJob(job) && job.status === 'succeeded' && job.result?.source === 'huggingface' &&
    augmentationVideoCameras(job.result).length > 0;
}
