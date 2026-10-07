'use client';

import { ModelLibrary } from '@/components/model-library';
import { useWorkspace } from '@/components/workspace-context';

export default function ModelsSection() {
  const { selectedModel, openModelRun, projects, workflowProjectId, openModelWorkflow, navigateStage, setStartTraining, chooseRun, setImportModel, selectProject, setOpenTrainingRun } = useWorkspace();
  return <>
    <ModelLibrary projects={projects.isSuccess ? projects.data : []} currentProjectId={workflowProjectId} preferredModelId={selectedModel?.projectId === workflowProjectId ? selectedModel.id : undefined} onModelRun={openModelRun} onAction={openModelWorkflow}
      onTrain={() => { navigateStage(1); setStartTraining({ id: Date.now() }); }}
      onImport={() => { navigateStage(5); chooseRun('native', 'handoff'); setImportModel(true); }}
      onTrainingRun={(owner, id, artifactId) => { selectProject(owner); navigateStage(1); setOpenTrainingRun({ projectId: owner, id, artifactId }); }} />
  </>;
}
