'use client';

import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api, isActive } from '@/lib/api';
import { checkpointStep, trainingRunModelLabel } from '@/lib/checkpoints';
import { quantizeModeFor } from '@/lib/model-library';
import { nativeQuantizationOf } from '@/lib/native-quantization';
import { storedAttempt, type PolicyJobAttempt } from '@/lib/policy-job-attempt';
import { quantizeJobMode, type QuantizeMode } from '@/lib/workflow-entry';
import type { SimulationHandoff } from '@/lib/native-simulation-handoff';
import { JobHistory } from './job-history';
import { ModelWorkflowPicker } from './model-library';
import { NativeQuantizationPanel } from './native-quantization-panel';
import { currentStage, precisionName, WorkflowPanel } from './workflow-panel';

/** One project-owned history for native packing and GGUF conversion. Adapters
 * keep their existing admission, cancellation and submission recovery logic. */
export function QuantizationPanel({ projectId, preferredArtifactId, preferredMode, preferredJobId, onLibrary, onPrepare, onReplay, onPrepareSimulation, onModel }: {
  projectId: string; preferredArtifactId?: string; preferredMode?: QuantizeMode; preferredJobId?: string;
  onLibrary: () => void; onPrepare: () => void; onReplay: (id: string) => void;
  onPrepareSimulation: (source: SimulationHandoff) => void; onModel: (id: string) => void;
}) {
  const jobs = useQuery({queryKey:['jobs',projectId],queryFn:()=>api.jobs(projectId),enabled:!!projectId,retry:false,refetchInterval:2000});
  const artifacts = useQuery({queryKey:['artifacts',projectId],queryFn:()=>api.artifacts(projectId),enabled:!!projectId,retry:false,refetchInterval:3000});
  const cached = useQuery<PolicyJobAttempt>({queryKey:['native-quantization-attempt',projectId],queryFn:async()=>null,enabled:false,initialData:null,gcTime:Infinity});
  const [recovery,setRecovery] = useState({pending:false,unreadable:false});
  const initialArtifact = artifacts.data?.find(artifact => artifact.id === preferredArtifactId && artifact.project_id === projectId);
  const [entry,setEntry] = useState<{mode:QuantizeMode;artifactId?:string;jobId?:string;modelRunId?:string} | null>(preferredArtifactId && preferredMode ? {mode:preferredMode,artifactId:preferredArtifactId,modelRunId:initialArtifact?.job_id} : null);
  const [view,setView] = useState<'jobs'|'new'>(preferredArtifactId ? 'new' : 'jobs');
  const [handoffConsumed,setHandoffConsumed] = useState(!!(preferredArtifactId && preferredMode));
  useEffect(()=>{
    try {setRecovery({pending:!!storedAttempt('policy.quantize',projectId),unreadable:false});}
    catch {setRecovery({pending:false,unreadable:true});}
  },[projectId,cached.data]);
  const ownArtifacts=(artifacts.data??[]).filter(item=>item.project_id===projectId);
  const ownJobs=(jobs.data??[]).filter(item=>item.project_id===projectId);
  const history=ownJobs.filter(job=>quantizeJobMode(job,projectId)).sort((a,b)=>b.created_at.localeCompare(a.created_at)||a.id.localeCompare(b.id));
  useEffect(()=>{
    if(handoffConsumed) return;
    if(preferredArtifactId && artifacts.isSuccess && !artifacts.isError) {
      const artifact=artifacts.data.find(item=>item.id===preferredArtifactId && item.project_id===projectId);
      const mode=artifact && quantizeModeFor(artifact);
      if(mode) {setEntry({mode,artifactId:artifact.id,modelRunId:artifact.job_id});setHandoffConsumed(true);}
    } else if(preferredJobId && jobs.isSuccess && !jobs.isError) {
      const job=jobs.data.find(item=>item.id===preferredJobId);
      const mode=job && quantizeJobMode(job,projectId);
      if(mode) {setEntry({mode,jobId:job.id});setHandoffConsumed(true);}
    }
  },[preferredArtifactId,preferredJobId,artifacts.data,artifacts.isSuccess,artifacts.isError,jobs.data,jobs.isSuccess,jobs.isError,projectId,handoffConsumed]);
  function back(){setEntry(null);setView('jobs');setHandoffConsumed(true);}
  function backToModels(){setEntry(null);setView('new');setHandoffConsumed(true);}
  const modelRunId=entry?.modelRunId;
  if(entry?.mode==='native') return <NativeQuantizationPanel projectId={projectId} preferredArtifactId={entry.artifactId} preferredJobId={entry.jobId} modelRunId={modelRunId} onBackToModels={backToModels} onBack={back} onPrepare={onPrepare} onReplay={onReplay} onPrepareSimulation={onPrepareSimulation} onModel={onModel}/>;
  if(entry?.mode==='gguf') return <WorkflowPanel projectId={projectId} stage="Quantize" tab="compute" onTabChange={()=>{}} onOpenQuantize={back} preferredArtifactId={entry.artifactId} preferredJobId={entry.jobId} onViewTraining={onPrepare} onBackToJobs={back} modelRunId={modelRunId} onBackToModels={backToModels} onModel={onModel}/>;
  return <section aria-label="Quantization workspace">
    {(preferredArtifactId || preferredJobId) && !handoffConsumed && <p role={artifacts.isError || jobs.isError ? 'alert' : 'status'}>{artifacts.isPending || jobs.isPending ? 'Loading the selected model or job…' : 'The selected model or job is unavailable or unsupported. No replacement has been selected.'}<button className="text-link" onClick={back}>Back to jobs</button></p>}
    {view==='jobs' ? <JobHistory title="Quantization jobs" newLabel="Start a new quantization"
      entries={history.map(job=>{
        const native=nativeQuantizationOf(job);
        const precision=native?`INT${native.bits}`:'precision' in job.request?precisionName(job.request.precision):'GGUF';
        const source=ownArtifacts.find(item=>'artifact_id' in job.request && item.id===job.request.artifact_id);
        const step=source?checkpointStep(source):null;
        return {id:job.id,title:[trainingRunModelLabel(job,[],ownArtifacts,ownJobs)??'Policy',precision==='—'?'GGUF':precision].join(' · '),subtitle:step!==null?`Step ${step.toLocaleString()}`:source?.label,status:job.status,createdAt:job.created_at,progress:isActive(job)?currentStage(job):undefined};
      })}
      onNew={()=>{setHandoffConsumed(true);setView('new');}} onSelect={id=>{const job=history.find(item=>item.id===id)!;setHandoffConsumed(true);setEntry({mode:quantizeJobMode(job,projectId)!,jobId:id});}}
      loading={jobs.isPending && !!projectId} error={jobs.error} disabled={!projectId} emptyMessage={projectId?'No quantization jobs yet.':'Select a project to see its jobs.'}/>
      : <><div className="workflow-view-navigation"><button type="button" className="text-link" onClick={back}>← Back to jobs</button></div><ModelWorkflowPicker projectId={projectId} action="quantize" onLibrary={onLibrary} onSelect={artifact=>{const mode=quantizeModeFor(artifact) ?? ownArtifacts.filter(item=>item.job_id===artifact.job_id).map(quantizeModeFor).find(Boolean);if(mode){setHandoffConsumed(true);setEntry({mode,artifactId:artifact.id,modelRunId:artifact.job_id});}}}/></>}
    {(cached.data || recovery.pending || recovery.unreadable) && <div className="warning-box" role={recovery.unreadable?'alert':'status'}><p>{recovery.unreadable?'Saved request status could not be read.':'A quantization request needs review.'}</p><button type="button" className="secondary-button" onClick={()=>{setHandoffConsumed(true);setEntry({mode:'native'});}}>Review request</button></div>}
  </section>;
}
