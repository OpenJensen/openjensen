'use client';

import { CloudRuns } from '@/components/cloud-runs';
import { useWorkspace } from '@/components/workspace-context';

export default function CloudRunsSection() {
  const { workflowProjectId, setOpenSimulation, chooseRun, openStage, setStartTraining, setOpenTrainingRun } = useWorkspace();
  return <>
    <CloudRuns key={workflowProjectId} projectId={workflowProjectId} onOpenSimulation={id => { setOpenSimulation({ projectId: workflowProjectId, id }); chooseRun('native', 'handoff', id); openStage(5); }} onOpenTraining={id => { setStartTraining(undefined); setOpenTrainingRun({ projectId: workflowProjectId, id }); openStage(1); }} />
  </>;
}
