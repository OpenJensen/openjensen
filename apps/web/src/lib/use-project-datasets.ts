'use client';

import { useQuery } from '@tanstack/react-query';
import { datasetLibrary, isProjectDataset, type LibraryDataset } from './dataset-library';

/**
 * The project's saved datasets, for showing a dataset by its own name on pages that
 * otherwise only know one of its jobs. Shares its cache with the Datasets page.
 */
export function useProjectDatasets(projectId: string | null | undefined, enabled = true): LibraryDataset[] {
  const project = projectId ?? '';
  const library = useQuery({
    queryKey: ['datasets', project],
    queryFn: ({ signal }) => datasetLibrary.list(project, signal),
    enabled: enabled && !!project,
    retry: false,
    staleTime: 10_000,
  });
  return (library.data ?? []).filter((entry): entry is LibraryDataset => isProjectDataset(entry, project));
}
