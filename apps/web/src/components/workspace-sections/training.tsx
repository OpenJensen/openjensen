'use client';

import { TrainingPanel } from '@/components/training-panel';
import { useWorkspace } from '@/components/workspace-context';

export default function TrainingSection() {
  const { activeStage, projectId, workflowProjectId, startTraining, trainingNavigation, openTrainingRun, selectedJob, openStage, setDatasetView, setSettingsTab, chooseQuantize, setQuantizeArtifact, setWorkflowNavigation, navigateStage, setDistillTeachers, setDistillationModels } = useWorkspace();
  return <>
    <div className="training-view" hidden={activeStage !== 1}><TrainingPanel active={activeStage === 1} key={projectId} projectId={workflowProjectId} startNew={startTraining} showJobsRequest={trainingNavigation} preferredRunId={openTrainingRun?.projectId === projectId ? openTrainingRun.id : undefined} preferredCheckpointId={openTrainingRun?.projectId === projectId ? openTrainingRun.artifactId : undefined} preferredDatasetId={selectedJob?.id} onChooseDataset={() => { openStage(0); setDatasetView('sources'); }} onDiagnostics={() => { setSettingsTab('diagnostics'); openStage(6); }} onComputeSettings={() => { setSettingsTab('compute'); openStage(6); }} onQuantize={artifactId => { chooseQuantize('gguf', 'handoff'); setQuantizeArtifact({ projectId, artifactId }); setWorkflowNavigation(value => value + 1); openStage(3); }} onNativeQuantize={artifactId => { navigateStage(3); setQuantizeArtifact({ projectId: workflowProjectId, artifactId }); chooseQuantize('native', 'handoff'); }} onNativeDistill={artifactId => { if (!workflowProjectId) return; navigateStage(2); setDistillTeachers(previous => ({ ...previous, [workflowProjectId]: artifactId })); setDistillationModels(previous => ({ ...previous, [workflowProjectId]: 'act' })); }} /></div>
  </>;
}
