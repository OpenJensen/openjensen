'use client';

import { DistillationPanel } from '@/components/distillation-panel';
import { useWorkspace } from '@/components/workspace-context';

export default function DistillationSection() {
  const { openDistillationRun, workflowNavigation, openSavedModel, workflowProjectId, distillationModels, distillTeachers, navigateStage, setDistillTeachers, setDistillationModels, setDatasetView, setQuantizeArtifact, projectId, chooseQuantize } = useWorkspace();
  return <>
    <DistillationPanel key={`${workflowProjectId}-${workflowNavigation}`} preferredJobId={openDistillationRun?.projectId === workflowProjectId ? openDistillationRun.id : undefined} onModel={openSavedModel} projectId={workflowProjectId} model={distillationModels[workflowProjectId]} preferredTeacherArtifactId={distillTeachers[workflowProjectId]} onLibrary={() => navigateStage('models')} onSelectModel={(model, artifactId) => { setDistillTeachers(previous => ({ ...previous, [workflowProjectId]: artifactId })); if (workflowProjectId) setDistillationModels(previous => ({ ...previous, [workflowProjectId]: model })); }} onDataset={() => { setDatasetView('sources'); navigateStage('datasets'); }} onQuantize={artifactId => { navigateStage('quantization'); setQuantizeArtifact({ projectId, artifactId }); chooseQuantize('native', 'handoff'); }} />
  </>;
}
