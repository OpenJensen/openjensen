'use client';

import { ModelWorkflowPicker } from '@/components/model-library';
import { NativeQuantizationPanel } from '@/components/native-quantization-panel';
import { nativeQuantizationOf } from '@/lib/native-quantization';
import { displayDate } from '@/components/workspace-ui';
import EngineWorkflowSection from './engine-workflow';
import { useWorkspace } from '@/components/workspace-context';

export default function QuantizationSection() {
  const { workflowProjectId, quantizeArtifact, openModelWorkflow, navigateStage, ownedJobs, chooseQuantize, setWorkflowNavigation, setOpenQuantizationRun, quantizeMode, workflowNavigation, context, setReplayArtifact, chooseRun, setSimulationArtifact } = useWorkspace();
  return <>
    <ModelWorkflowPicker projectId={workflowProjectId} action="quantize" selectedId={quantizeArtifact?.projectId === workflowProjectId ? quantizeArtifact.artifactId : undefined} onSelect={artifact => openModelWorkflow(artifact, 'quantize')} onLibrary={() => navigateStage(11)} />
    {ownedJobs.some(job => ['policy.quantize', 'policy.workflow'].includes(job.kind)) && <details className="model-saved-work"><summary>Saved quantization work</summary><div className="model-choice-list">{ownedJobs.filter(job => ['policy.quantize', 'policy.workflow'].includes(job.kind)).map(job => <button type="button" className="model-choice" key={job.id} aria-label={`Open quantization ${job.id}`} onClick={() => { navigateStage(3); chooseQuantize(nativeQuantizationOf(job) ? 'native' : 'gguf', 'manual', job.id); setWorkflowNavigation(value => value + 1); setOpenQuantizationRun({ projectId: workflowProjectId, id: job.id }); }}><span><strong>{'artifact_id' in job.request ? job.request.artifact_id : 'Model identity not recorded'}</strong><small>{displayDate(job.created_at)} · Run {job.id}</small></span><span>{job.status}</span></button>)}</div></details>}
    {quantizeMode === 'native' && <NativeQuantizationPanel key={`${workflowProjectId}-${workflowNavigation}`} projectId={workflowProjectId} preferredJobId={context?.quantize?.jobId} onJobSelected={id => chooseQuantize('native', 'manual', id)} preferredArtifactId={quantizeArtifact?.projectId === workflowProjectId ? quantizeArtifact.artifactId : undefined} onPrepare={() => navigateStage(1)} onReplay={artifactId => { navigateStage(5); setReplayArtifact({ projectId: workflowProjectId, artifactId }); chooseRun('replay', 'handoff'); }} onPrepareSimulation={source => { if (source.projectId !== workflowProjectId) return; navigateStage(5); setSimulationArtifact(source); chooseRun('native', 'handoff'); }} />}
    {quantizeMode === 'gguf' && <EngineWorkflowSection />}
  </>;
}
