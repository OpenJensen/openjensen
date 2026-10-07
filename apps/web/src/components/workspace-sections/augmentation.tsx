'use client';

import { AugmentationPanel } from '@/components/augmentation-panel';
import { useWorkspace } from '@/components/workspace-context';

export default function AugmentationSection() {
  const { projectId, workflowProjectId, selectedJob, setSettingsTab, navigateStage, setDatasetView } = useWorkspace();
  return <>
    <AugmentationPanel key={projectId} projectId={workflowProjectId} preferredDatasetId={selectedJob?.id} onOpenSettings={() => { setSettingsTab('compute'); navigateStage(6); }} onChooseDataset={() => { navigateStage(0); setDatasetView('sources'); }} />
  </>;
}
