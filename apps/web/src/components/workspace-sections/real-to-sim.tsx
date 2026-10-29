'use client';

import { SceneGenerationPanel } from '@/components/scene-generation-panel';
import { useWorkspace } from '@/components/workspace-context';

export default function RealToSimSection() {
  const { workflowProjectId, requestedWorkJob } = useWorkspace();
  return (
    <SceneGenerationPanel
      key={workflowProjectId}
      projectId={workflowProjectId}
      preferredJobId={
        requestedWorkJob?.project_id === workflowProjectId && requestedWorkJob.operation === 'scene.generate'
          ? requestedWorkJob.source_id
          : undefined
      }
    />
  );
}
