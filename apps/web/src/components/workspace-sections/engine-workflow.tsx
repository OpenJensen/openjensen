'use client';

import { WorkflowPanel } from '@/components/workflow-panel';
import { useWorkspace } from '@/components/workspace-context';
import type { StageId } from '@/lib/workspace-routes';

// The workflow panel's logic is keyed by these fixed words. They are not what the sidebar
// calls a page, so renaming a page cannot change which form or job list is shown.
const workflowKeys: Partial<Record<StageId, string>> = {
  settings: 'settings',
  training: 'Fine-tune',
  quantization: 'Quantize',
  evaluation: 'Evaluate',
  simulation: 'Run',
};

export default function EngineWorkflowSection() {
  const {
    workflowProjectId,
    requestedWorkJob,
    requestedJobOperation,
    setModuleJobView,
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
      startNew={requestedJobOperation?.module === activeStage}
      onBackToJobs={activeStage === 'simulation' ? () => setModuleJobView('jobs') : undefined}
      key={`${workflowProjectId}-${activeStage}-${workflowNavigation}`}
      projectId={workflowProjectId}
      tab={settingsTab}
      onTabChange={setSettingsTab}
      onOpenQuantize={() => navigateStage('quantization')}
      stage={workflowKeys[activeStage] ?? stage.name}
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
