'use client';

import { WorkspaceDashboard } from '@/components/workspace-dashboard';
import { useWorkspace } from '@/components/workspace-context';

export default function DashboardSection() {
  const { projects, workflowProjectId, navigateStage, setDatasetView, openDataset, setSettingsTab } = useWorkspace();
  return <>
    <WorkspaceDashboard projects={projects.data ?? []} projectId={workflowProjectId} onModels={() => navigateStage('models')} onDatasets={() => { navigateStage('datasets'); setDatasetView('sources'); }} onDataset={openDataset} onSettings={() => { setSettingsTab('compute'); navigateStage('settings'); }} />
  </>;
}
