'use client';

import { createContext, useContext } from 'react';
import type { useWorkbenchState } from '@/components/workspace-state';

export const WorkspaceContext = createContext<ReturnType<typeof useWorkbenchState> | null>(null);

export function useWorkspace() {
  const workspace = useContext(WorkspaceContext);
  if (!workspace) throw new Error('Workspace sections require the shared workspace layout.');
  return workspace;
}
