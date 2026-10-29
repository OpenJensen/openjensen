'use client';

import { SceneGenerationPanel } from '@/components/scene-generation-panel';
import { useWorkspace } from '@/components/workspace-context';

export default function RealToSimSection() {
  const { workflowProjectId, requestedWorkJob, requestedJobOperation, workflowNavigation } = useWorkspace();
  return (
    <SceneGenerationPanel
      key={workflowProjectId}
      projectId={workflowProjectId}
      startNewRequest={requestedJobOperation?.id === 'scene.generate' ? workflowNavigation : undefined}
      preferredJobId={
        requestedWorkJob?.project_id === workflowProjectId && requestedWorkJob.operation === 'scene.generate'
          ? requestedWorkJob.source_id
          : undefined
      }
    />
  );
}
