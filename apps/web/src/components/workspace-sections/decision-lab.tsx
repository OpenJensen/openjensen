'use client';

import { useWorkspace } from '@/components/workspace-context';
import { DecisionPanel } from '@/components/decision-panel';

export default function DecisionLabSection() {
  const { workflowProjectId, requestedWorkJob } = useWorkspace();
  return (
    <>
      <DecisionPanel
        key={workflowProjectId}
        projectId={workflowProjectId}
        preferredJobId={
          requestedWorkJob?.project_id === workflowProjectId && requestedWorkJob.operation === 'decision.score'
            ? requestedWorkJob.id
            : undefined
        }
      />
    </>
  );
}
