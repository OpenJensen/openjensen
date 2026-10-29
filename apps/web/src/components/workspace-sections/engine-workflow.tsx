'use client';

import { WorkflowPanel } from '@/components/workflow-panel';
import { useWorkspace } from '@/components/workspace-context';

export default function EngineWorkflowSection() {
  const {
    workflowProjectId,
    requestedWorkJob,
    activeStage,
    workflowNavigation,
    settingsTab,
    setSettingsTab,
    navigateStage,
    stage,
    openQuantizationRun,
    quantizeArtifact,
    modelInput,
  } = useWorkspace();
  return (
    <WorkflowPanel
      key={`${workflowProjectId}-${activeStage}-${workflowNavigation}`}
      projectId={workflowProjectId}
      tab={settingsTab}
      onTabChange={setSettingsTab}
      onOpenQuantize={() => navigateStage('quantization')}
      stage={activeStage === 'settings' ? 'settings' : stage.name}
      preferredJobId={
        activeStage === 'quantization' && openQuantizationRun?.projectId === workflowProjectId
          ? openQuantizationRun.id
          : requestedWorkJob?.project_id === workflowProjectId && requestedWorkJob.module === activeStage
            ? requestedWorkJob.source_id
            : undefined
      }
      preferredArtifactId={
        activeStage === 'quantization' && quantizeArtifact?.projectId === workflowProjectId
          ? quantizeArtifact.artifactId
          : modelInput?.projectId === workflowProjectId
            ? modelInput.artifactId
            : undefined
      }
      onViewTraining={() => navigateStage('training')}
    />
  );
}
