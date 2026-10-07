'use client';

import { WorkspaceDashboard } from '@/components/workspace-dashboard';
import { useWorkspace } from '@/components/workspace-context';

export default function DashboardSection() {
  const { projects, navigateStage, setDatasetView, openDataset, setSettingsTab } = useWorkspace();
  return <>
    <WorkspaceDashboard projects={projects.data ?? []} onModels={() => navigateStage(11)} onDatasets={() => { navigateStage(0); setDatasetView('sources'); }} onDataset={openDataset} onSettings={() => { setSettingsTab('compute'); navigateStage(6); }} />
  </>;
}
