'use client';

import { WorkflowPanel } from '@/components/workflow-panel';
import { useWorkspace } from '@/components/workspace-context';

export default function EngineWorkflowSection() {
  const { workflowProjectId, activeStage, workflowNavigation, settingsTab, setSettingsTab, navigateStage, stage, openQuantizationRun, quantizeArtifact, modelInput } = useWorkspace();
  return <WorkflowPanel key={`${workflowProjectId}-${activeStage}-${workflowNavigation}`} projectId={workflowProjectId} tab={settingsTab} onTabChange={setSettingsTab} onOpenQuantize={() => navigateStage(3)} stage={activeStage === 6 ? 'settings' : stage.name} preferredJobId={activeStage === 3 && openQuantizationRun?.projectId === workflowProjectId ? openQuantizationRun.id : undefined} preferredArtifactId={activeStage === 3 && quantizeArtifact?.projectId === workflowProjectId ? quantizeArtifact.artifactId : modelInput?.projectId === workflowProjectId ? modelInput.artifactId : undefined} onViewTraining={() => navigateStage(1)} />;
}
