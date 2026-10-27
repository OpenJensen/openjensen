'use client';

import { AugmentationPanel } from '@/components/augmentation-panel';
import { useWorkspace } from '@/components/workspace-context';

export default function AugmentationSection() {
  const { projectId, workflowProjectId, selectedJob, navigateStage, setDatasetView } = useWorkspace();
  return (
    <>
      <AugmentationPanel
        key={projectId}
        projectId={workflowProjectId}
        preferredDatasetId={selectedJob?.id}
        onOpenSettings={() => {
          navigateStage('cloud-apis');
        }}
        onChooseDataset={() => {
          navigateStage('datasets');
          setDatasetView('sources');
        }}
      />
    </>
  );
}
