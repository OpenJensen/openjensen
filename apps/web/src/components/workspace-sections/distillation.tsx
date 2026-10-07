'use client';

import { DistillationPanel } from '@/components/distillation-panel';
import { useWorkspace } from '@/components/workspace-context';

export default function DistillationSection() {
  const { workflowProjectId, distillationModels, distillTeachers, navigateStage, setDistillTeachers, setDistillationModels, setDatasetView, setQuantizeArtifact, projectId, chooseQuantize } = useWorkspace();
  return <>
    <DistillationPanel key={workflowProjectId} projectId={workflowProjectId} model={distillationModels[workflowProjectId]} preferredTeacherArtifactId={distillTeachers[workflowProjectId]} onLibrary={() => navigateStage(11)} onSelectModel={(model, artifactId) => { setDistillTeachers(previous => ({ ...previous, [workflowProjectId]: artifactId })); if (workflowProjectId) setDistillationModels(previous => ({ ...previous, [workflowProjectId]: model })); }} onDataset={() => { setDatasetView('sources'); navigateStage(0); }} onQuantize={artifactId => { navigateStage(3); setQuantizeArtifact({ projectId, artifactId }); chooseQuantize('native', 'handoff'); }} />
  </>;
}
