'use client';

import { useEffect } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { selectedJobCompute } from './job-compute-selection';

export function useJobComputeSelection(project: string, operation: string, active: boolean) {
  const client = useQueryClient();
  const query = useQuery({
    queryKey: ['job-compute-selection', project, operation],
    queryFn: () => selectedJobCompute(project, operation),
    enabled: !!project && active,
    staleTime: 0,
    retry: false,
  });
  useEffect(() => {
    const refresh = () => {
      void client.invalidateQueries({ queryKey: ['job-compute-selection', project, operation] });
    };
    window.addEventListener('openjensen:job-compute-selection', refresh);
    return () => window.removeEventListener('openjensen:job-compute-selection', refresh);
  }, [client, project, operation]);
  return query;
}
