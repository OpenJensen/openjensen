import { apiOrigin, type DatasetProfile } from './api';

export type DatasetDetection = {
  format: string;
  convertible: boolean;
  reason?: string;
  sha256: string;
  file_count: number;
  total_bytes: number;
  episodes?: number;
  frames?: number;
  fps?: number;
  cameras: string[];
  mapping: { action?: string; state?: string[]; cameras?: string[]; table?: string };
};
export type DatasetLabels = { views: Record<string, string>; frames: Record<string, string> };
export type LibraryDataset = {
  id: string;
  project_id: string;
  name: string;
  source: 'huggingface' | 'local';
  status: 'uploading' | 'unavailable' | 'detected' | 'converting' | 'ready' | 'failed';
  created_at: string;
  job_id?: string;
  training_copy?: boolean;
  profile?: DatasetProfile;
  detection?: DatasetDetection;
  converted?: { format: string; episodes: number; frames: number; fps: number; cameras: string[] };
  error?: string;
  preview_note?: string;
  example?: boolean;
  annotations?: DatasetLabels;
  annotation_revision?: number;
  uploaded_files?: number;
  total_bytes?: number;
  can_discard?: boolean;
};
export type DatasetRepairNotice = {
  id: string;
  project_id: null;
  name: string;
  status: 'unavailable';
  created_at: string;
  error: string;
  can_discard: false;
};
export const isProjectDataset = (
  entry: LibraryDataset | DatasetRepairNotice,
  project: string,
): entry is LibraryDataset => !!project && entry.project_id === project;
export type DatasetSample = { episode_index: number; frame_index: number; camera: string; path: string; task: string };
const base = `${apiOrigin}/api/v1`;
export async function datasetRequest<T>(path: string, init?: RequestInit, timeoutMs = 30_000): Promise<T> {
  const controller = new AbortController();
  const cancel = () => controller.abort();
  if (init?.signal?.aborted) cancel();
  else init?.signal?.addEventListener('abort', cancel, { once: true });
  const timer = setTimeout(cancel, timeoutMs);
  try {
    if (controller.signal.aborted) throw new Error('Request cancelled.');
    const response = await fetch(base + path, {
      ...init,
      signal: controller.signal,
      cache: 'no-store',
      headers: { 'Content-Type': 'application/json', ...init?.headers },
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      const detail = typeof error?.detail === 'string' ? error.detail : `Dataset request failed (${response.status})`;
      throw new Error(detail);
    }
    return await response.json();
  } catch (error) {
    if (controller.signal.aborted)
      throw new Error(
        init?.signal?.aborted
          ? 'Dataset request cancelled. A change may already have been saved; refresh its saved state before retrying.'
          : 'The dataset API did not respond in time. A change may already have been saved; refresh its saved state before retrying.',
        { cause: error },
      );
    if (error instanceof TypeError)
      throw new Error(
        'Cannot reach the dataset API. Check the application connection and refresh saved state before retrying.',
        { cause: error },
      );
    throw error;
  } finally {
    clearTimeout(timer);
    init?.signal?.removeEventListener('abort', cancel);
  }
}
export class DatasetUploadError extends Error {
  constructor(
    message: string,
    readonly uploadId: string | null,
  ) {
    super(
      `${message} ${uploadId ? 'Open the saved upload in My datasets to review or discard unfinished files. A completed import may already be saved.' : 'Refresh My datasets before choosing the files again; an upload may already have been created.'} No upload was retried automatically.`,
    );
    this.name = 'DatasetUploadError';
  }
}
export const datasetLibrary = {
  list: (project?: string, signal?: AbortSignal) =>
    datasetRequest<(LibraryDataset | DatasetRepairNotice)[]>(
      `/datasets${project ? `?project_id=${encodeURIComponent(project)}` : ''}`,
      { signal },
    ),
  get: (id: string, signal?: AbortSignal) =>
    datasetRequest<LibraryDataset>(`/datasets/${encodeURIComponent(id)}`, { signal }),
  samples: (id: string, signal?: AbortSignal) =>
    datasetRequest<DatasetSample[]>(`/datasets/${encodeURIComponent(id)}/samples`, { signal }),
  convert: (
    id: string,
    settings: { fps: number; task: string; robot_type: string; mapping?: DatasetDetection['mapping'] },
  ) =>
    datasetRequest<LibraryDataset>(`/datasets/${encodeURIComponent(id)}/convert`, {
      method: 'POST',
      body: JSON.stringify(settings),
    }),
  annotate: (id: string, revision: number, labels: DatasetLabels) =>
    datasetRequest<LibraryDataset>(`/datasets/${encodeURIComponent(id)}/annotations`, {
      method: 'PUT',
      body: JSON.stringify({ revision, ...labels }),
    }),
  discard: (id: string, project: string) =>
    datasetRequest<{ discarded: true; id: string }>(
      `/dataset-uploads/${encodeURIComponent(id)}?project_id=${encodeURIComponent(project)}`,
      { method: 'DELETE' },
    ),
  async upload(
    project: string,
    files: File[],
    onProgress: (value: string) => void,
    signal?: AbortSignal,
    onStarted?: (entry: LibraryDataset) => void,
  ) {
    if (!files.length) throw new Error('Choose dataset files first.');
    if (files.length > 4096 || files.reduce((sum, file) => sum + file.size, 0) > 2 * 1024 ** 3)
      throw new Error('Choose up to 4096 files and 2 GB.');
    let uploadId: string | null = null;
    try {
      const folder = (files[0].webkitRelativePath ?? '').split('/')[0];
      const entry = await datasetRequest<LibraryDataset>(`/projects/${encodeURIComponent(project)}/dataset-uploads`, {
        method: 'POST',
        signal,
        body: JSON.stringify({ name: (folder || files[0].name).slice(0, 100) }),
      });
      uploadId = entry.id;
      onStarted?.(entry);
      for (let index = 0; index < files.length; index++) {
        const file = files[index];
        const path = file.webkitRelativePath ? file.webkitRelativePath.split('/').slice(1).join('/') : file.name;
        onProgress(`Uploading ${index + 1} of ${files.length} files…`);
        await datasetRequest(
          `/dataset-uploads/${encodeURIComponent(entry.id)}/file?path=${encodeURIComponent(path)}`,
          { method: 'PUT', signal, headers: { 'Content-Type': 'application/octet-stream' }, body: file },
          600_000,
        );
      }
      onProgress('Detecting dataset format…');
      return await datasetRequest<LibraryDataset>(
        `/dataset-uploads/${encodeURIComponent(entry.id)}/finish`,
        { method: 'POST', signal },
        240_000,
      );
    } catch (error) {
      throw new DatasetUploadError(
        error instanceof Error ? error.message : 'The upload could not be confirmed.',
        uploadId,
      );
    }
  },
};
export type LabelDraft = { revision: number; labels: DatasetLabels };
export const labelDraftKey = (entry: Pick<LibraryDataset, 'project_id' | 'id'>) =>
  `jensen:dataset-labels:${encodeURIComponent(entry.project_id)}:${encodeURIComponent(entry.id)}`;
export const sameLabelDraft = (left: LabelDraft | undefined | null, right: LabelDraft) =>
  !!left && left.revision === right.revision && JSON.stringify(left.labels) === JSON.stringify(right.labels);
export function readLabelDraft(raw: string | null): LabelDraft | null {
  if (!raw) return null;
  try {
    const value = JSON.parse(raw);
    const record = (fields: unknown, maximum: number, length: number) =>
      !!fields &&
      typeof fields === 'object' &&
      !Array.isArray(fields) &&
      Object.keys(fields).length <= maximum &&
      Object.values(fields).every((text) => typeof text === 'string' && text.length <= length);
    return value?.version === 1 &&
      Number.isSafeInteger(value.revision) &&
      value.revision >= 0 &&
      record(value.labels?.views, 8, 100) &&
      record(value.labels?.frames, 12000, 1000)
      ? { revision: value.revision, labels: value.labels }
      : null;
  } catch {
    return null;
  }
}
export function sampleImage(id: string, name: string) {
  return `${base}/datasets/${id}/preview/${encodeURIComponent(name)}`;
}
export function datasetDownload(id: string) {
  return `${base}/datasets/${id}/download`;
}
export function formatName(value?: string) {
  return (
    (
      {
        lerobot_v2: 'LeRobot v2',
        lerobot_v3: 'LeRobot v3',
        robomimic_hdf5: 'robomimic HDF5',
        aloha_hdf5: 'ALOHA HDF5',
        image_records: 'Images + records',
        rlds_tfrecord: 'RLDS / TFRecord',
        ros_recording: 'ROS recording',
      } as Record<string, string>
    )[value ?? ''] || 'Unknown format'
  );
}
