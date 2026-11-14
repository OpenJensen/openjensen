'use client';

import { WorkspaceDashboard } from '@/components/workspace-dashboard';
import { useWorkspace } from '@/components/workspace-context';
import { requestRobotLabTab } from '@/lib/robot-lab-entry';

export default function DashboardSection() {
  const { projects, workflowProjectId, navigateStage, setDatasetView, openDataset, setSettingsTab, modules } =
    useWorkspace();
  return (
    <>
      <WorkspaceDashboard
        projects={projects.data ?? []}
        projectId={workflowProjectId}
        installedModules={modules.data?.filter((item) => item.installed).map((item) => item.id) ?? []}
        onCloudApis={() => navigateStage('cloud-apis')}
        onRobotLab={(tab) => {
          requestRobotLabTab(tab);
          navigateStage('robot-lab');
        }}
        onTrain={() => navigateStage('training')}
        onModels={() => navigateStage('models')}
        onDatasets={() => {
          navigateStage('datasets');
          setDatasetView('sources');
        }}
        onDataset={openDataset}
        onSettings={() => {
          setSettingsTab('compute');
          navigateStage('settings');
        }}
      />
    </>
  );
}
