import type { QueryClient } from '@tanstack/react-query';
import type { LibraryDataset } from './dataset-library';

export function datasetPollingInterval(entry: LibraryDataset | undefined, acceptedConversionId: string | null = null): number | false {
  if (entry?.status === 'uploading' || entry?.status === 'converting') return 1000;
  // An accepted conversion remains observed even if a read returns its earlier
  // detection state. Only a recorded terminal state can finish that observation.
  if (entry?.id === acceptedConversionId && !['ready','failed','unavailable'].includes(entry.status)) return 1000;
  return false;
}

export async function publishDatasetReceipt(client: QueryClient, entry: LibraryDataset): Promise<void> {
  const queryKey = ['dataset-import',entry.id];
  // A GET started before the mutation may still hold the old state. Cancel it
  // before publishing the receipt so it cannot overwrite the accepted state.
  await client.cancelQueries({queryKey,exact:true});
  client.setQueryData(queryKey,entry);
  await client.invalidateQueries({queryKey,exact:true});
}
