'use client';

import { QuantizationPanel } from '@/components/quantization-panel';
import { useWorkspace } from '@/components/workspace-context';

export default function QuantizationSection() {
  const { workflowNavigation, context, workflowProjectId, openQuantizationRun, quantizeArtifact, openModelWorkflow, navigateStage, setOpenQuantizationRun, openSavedModel, setReplayArtifact, chooseRun, setSimulationArtifact } = useWorkspace();
  return <QuantizationPanel key={`${workflowProjectId}-${workflowNavigation}`} projectId={workflowProjectId} preferredJobId={openQuantizationRun?.projectId === workflowProjectId ? openQuantizationRun.id : undefined} preferredArtifactId={quantizeArtifact?.projectId === workflowProjectId ? quantizeArtifact.artifactId : undefined} preferredMode={quantizeArtifact?.projectId === workflowProjectId ? context?.quantize?.mode : undefined} onLibrary={() => navigateStage(11)} onModel={openSavedModel} onPrepare={() => navigateStage(1)} onReplay={artifactId => {navigateStage(5);setReplayArtifact({projectId:workflowProjectId,artifactId});chooseRun('replay','handoff');}} onPrepareSimulation={source => {if(source.projectId !== workflowProjectId)return;navigateStage(5);setSimulationArtifact(source);chooseRun('native','handoff');}} />;
}
