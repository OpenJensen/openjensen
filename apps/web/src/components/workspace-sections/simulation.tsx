'use client';

import { NativeReplayPanel } from '@/components/native-replay-panel';
import { NativeSimulationPanel } from '@/components/native-simulation-panel';
import { ModeCard } from '@/components/workspace-ui';
import EngineWorkflowSection from './engine-workflow';
import { useWorkspace } from '@/components/workspace-context';

export default function SimulationSection() {
  const { runMode, workflowProjectId, setSimulationArtifact, setOpenSimulation, setReplayArtifact, chooseRun, context, replayArtifact, setDatasetView, navigateStage, modelInput, importModel, simulationArtifact, openSimulation } = useWorkspace();
  return <>
    <section className="workflow-choices" aria-label="Run workflow">
      <h2 className="workflow-choice-title">Workflows</h2>
      <div className="mode-card-grid" role="group" aria-label="Run mode">
        <ModeCard title="3D simulation" detail="Isaac Sim · ACT or SmolVLA" icon="play" selected={runMode === 'native'} disabled={!workflowProjectId} onClick={() => { setSimulationArtifact(null); if (runMode !== 'native') { setOpenSimulation(null); setReplayArtifact(null); } chooseRun('native'); }} />
        <ModeCard title="Replay observations" detail="Offline replay · ACT" icon="database" selected={runMode === 'replay'} disabled={!workflowProjectId} onClick={() => { setSimulationArtifact(null); if (runMode !== 'replay') { setOpenSimulation(null); setReplayArtifact(null); } chooseRun('replay'); }} />
        <ModeCard title="Check inference" detail="Inference engine · GGUF" icon="chart" selected={runMode === 'engine'} disabled={!workflowProjectId} onClick={() => { setSimulationArtifact(null); if (runMode !== 'engine') { setOpenSimulation(null); setReplayArtifact(null); } chooseRun('engine'); }} />
      </div>
    </section>
    {runMode === 'replay' && <NativeReplayPanel key={workflowProjectId} projectId={workflowProjectId} preferredJobId={context?.run?.mode === 'replay' ? context.run.jobId : undefined} onJobSelected={id => chooseRun('replay', 'manual', id)} preferredArtifactId={replayArtifact?.projectId === workflowProjectId ? replayArtifact.artifactId : undefined} onDataset={() => { setDatasetView('sources'); navigateStage(0); }} />}
    {runMode === 'native' && <NativeSimulationPanel key={`${workflowProjectId}-${modelInput?.artifactId ?? ''}-${importModel}`} projectId={workflowProjectId} preferredModelId={modelInput?.projectId === workflowProjectId ? modelInput.artifactId : undefined} initialPolicySource={importModel ? 'upload' : 'saved'} onJobSelected={id => { setOpenSimulation(null); setSimulationArtifact(null); chooseRun('native', 'manual', id); }} preferredArtifact={simulationArtifact?.projectId === workflowProjectId ? simulationArtifact : undefined} preferredJobId={openSimulation?.projectId === workflowProjectId ? openSimulation.id : context?.run?.mode === 'native' ? context.run.jobId : undefined} onTraining={() => navigateStage(1)} />}
    {runMode === 'engine' && <EngineWorkflowSection />}
  </>;
}
