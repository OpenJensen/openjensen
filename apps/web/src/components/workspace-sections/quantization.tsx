'use client';

import { QuantizationPanel } from '@/components/quantization-panel';
import { useWorkspace } from '@/components/workspace-context';

export default function QuantizationSection() {
  const { workflowNavigation, context, workflowProjectId, openQuantizationRun, quantizeArtifact, openModelWorkflow, navigateStage, setOpenQuantizationRun, openSavedModel, setReplayArtifact, chooseRun, setSimulationArtifact } = useWorkspace();
  return <QuantizationPanel key={`${workflowProjectId}-${workflowNavigation}`} projectId={workflowProjectId} preferredJobId={openQuantizationRun?.projectId === workflowProjectId ? openQuantizationRun.id : undefined} preferredArtifactId={quantizeArtifact?.projectId === workflowProjectId ? quantizeArtifact.artifactId : undefined} preferredMode={quantizeArtifact?.projectId === workflowProjectId ? context?.quantize?.mode : undefined} onLibrary={() => navigateStage('models')} onModel={openSavedModel} onPrepare={() => navigateStage('training')} onReplay={artifactId => {navigateStage('simulation');setReplayArtifact({projectId:workflowProjectId,artifactId});chooseRun('replay','handoff');}} onPrepareSimulation={source => {if(source.projectId !== workflowProjectId)return;navigateStage('simulation');setSimulationArtifact(source);chooseRun('native','handoff');}} />;
}
