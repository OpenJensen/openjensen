'use client';

import { QuantizationPanel } from '@/components/quantization-panel';
import { useWorkspace } from '@/components/workspace-context';

export default function QuantizationSection() {
  const {
    workflowNavigation,
    context,
    workflowProjectId,
    requestedJobOperation,
    openQuantizationRun,
    quantizeArtifact,
    navigateStage,
    navigateTask,
    openSavedModel,
    setReplayArtifact,
    chooseRun,
    setSimulationArtifact,
  } = useWorkspace();
  return (
    <QuantizationPanel
      startNew={requestedJobOperation?.id === 'policy.quantize'}
      key={`${workflowProjectId}-${workflowNavigation}`}
      projectId={workflowProjectId}
      preferredJobId={openQuantizationRun?.projectId === workflowProjectId ? openQuantizationRun.id : undefined}
      preferredArtifactId={quantizeArtifact?.projectId === workflowProjectId ? quantizeArtifact.artifactId : undefined}
      preferredMode={quantizeArtifact?.projectId === workflowProjectId ? context?.quantize?.mode : undefined}
      onLibrary={() => navigateStage('models')}
      onModel={openSavedModel}
      onPrepare={() => navigateStage('training')}
      onReplay={(artifactId) => {
        navigateTask('simulation');
        setReplayArtifact({ projectId: workflowProjectId, artifactId });
        chooseRun('replay', 'handoff');
      }}
      onPrepareSimulation={(source) => {
        if (source.projectId !== workflowProjectId) return;
        navigateTask('simulation');
        setSimulationArtifact(source);
        chooseRun('native', 'handoff');
      }}
    />
  );
}
