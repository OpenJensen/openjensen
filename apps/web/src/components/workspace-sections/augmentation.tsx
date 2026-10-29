'use client';

import { AugmentationPanel } from '@/components/augmentation-panel';
import { useWorkspace } from '@/components/workspace-context';

export default function AugmentationSection() {
  const { requestedWorkJob, projectId, workflowProjectId, selectedJob, navigateStage, setDatasetView } = useWorkspace();
  return (
    <>
      <AugmentationPanel
        key={projectId}
        projectId={workflowProjectId}
        preferredDatasetId={selectedJob?.id}
        preferredJobId={
          requestedWorkJob?.project_id === workflowProjectId && requestedWorkJob.operation === 'dataset.augment'
            ? requestedWorkJob.source_id
            : undefined
        }
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
