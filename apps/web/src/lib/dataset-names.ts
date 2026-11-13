import type { LibraryDataset } from './dataset-library';

/** What naming needs from a saved dataset; any list of library entries will do. */
export type NamedDataset = Pick<LibraryDataset, 'id' | 'name' | 'job_id'>;

/** The parts of a dataset job that say which dataset it read. */
type DatasetSource = {
  id: string;
  request: object;
  result?: object | null;
};

const text = (value: unknown) => (typeof value === 'string' && value.trim() ? value : null);

/**
 * A dataset's own name, wherever one of its jobs is shown: the repository it came from,
 * the name it was saved under, or the folder it was read from. Pages used to print
 * "Local dataset" for every dataset saved on the computer, so two could not be told apart.
 */
export function datasetName(job: DatasetSource, library: NamedDataset[] = []): string {
  const request = job.request as { repo_id?: unknown; library_id?: unknown; path?: unknown };
  const repository = text((job.result as { repo_id?: unknown } | null | undefined)?.repo_id) ?? text(request.repo_id);
  if (repository) return repository;
  const saved =
    library.find((entry) => entry.id === request.library_id) ?? library.find((entry) => entry.job_id === job.id);
  if (text(saved?.name)) return saved!.name;
  const folder = text(request.path)?.split(/[\\/]/).filter(Boolean).at(-1);
  return folder ?? 'Unnamed dataset';
}

/** A short local date and time for telling two things of the same name apart. */
export function shortDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
}
