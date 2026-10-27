'use client';

import { TrainingPanel } from '@/components/training-panel';
import { useWorkspace } from '@/components/workspace-context';

export default function TrainingSection() {
  const {
    activeStage,
    modules,
    projectId,
    workflowProjectId,
    startTraining,
    trainingNavigation,
    openTrainingRun,
    selectedJob,
    openStage,
    setDatasetView,
    setSettingsTab,
    chooseQuantize,
    setQuantizeArtifact,
    setWorkflowNavigation,
    navigateStage,
    setDistillTeachers,
    setDistillationModels,
  } = useWorkspace();
  return (
    <>
      <div className="training-view" hidden={activeStage !== 'training'}>
        <TrainingPanel
          active={activeStage === 'training'}
          key={projectId}
          projectId={workflowProjectId}
          startNew={startTraining}
          showJobsRequest={trainingNavigation}
          preferredRunId={openTrainingRun?.projectId === projectId ? openTrainingRun.id : undefined}
          preferredCheckpointId={openTrainingRun?.projectId === projectId ? openTrainingRun.artifactId : undefined}
          preferredDatasetId={selectedJob?.id}
          onChooseDataset={() => {
            openStage('datasets');
            setDatasetView('sources');
          }}
          onDiagnostics={() => {
            setSettingsTab('diagnostics');
            openStage('settings');
          }}
          onComputeSettings={() => {
            if (modules.data?.some((item) => item.id === 'google-cloud' && item.installed)) {
              openStage('cloud-apis');
            } else {
              setSettingsTab('compute');
              openStage('settings');
            }
          }}
          onQuantize={(artifactId) => {
            chooseQuantize('gguf', 'handoff');
            setQuantizeArtifact({ projectId, artifactId });
            setWorkflowNavigation((value) => value + 1);
            openStage('quantization');
          }}
          onNativeQuantize={(artifactId) => {
            navigateStage('quantization');
            setQuantizeArtifact({ projectId: workflowProjectId, artifactId });
            chooseQuantize('native', 'handoff');
          }}
          onNativeDistill={(artifactId) => {
            if (!workflowProjectId) return;
            navigateStage('distillation');
            setDistillTeachers((previous) => ({ ...previous, [workflowProjectId]: artifactId }));
            setDistillationModels((previous) => ({ ...previous, [workflowProjectId]: 'act' }));
          }}
        />
      </div>
    </>
  );
}
