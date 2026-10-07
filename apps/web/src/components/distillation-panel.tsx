'use client';

import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api, isActive } from '@/lib/api';
import { studentJob } from '@/lib/native-distillation';
import { storedAttempt, type PolicyJobAttempt } from '@/lib/policy-job-attempt';
import { NativeDistillationPanel } from './native-distillation-panel';
import { JobHistory } from './job-history';
import { ModelWorkflowPicker } from './model-library';
import './distillation-panel.css';

export type DistillationModel = 'act';

type Props = {
  projectId: string;
  model?: DistillationModel;
  preferredTeacherArtifactId?: string;
  onSelectModel: (model: DistillationModel | undefined, artifactId?: string) => void;
  onLibrary: () => void;
  onDataset: () => void;
  onQuantize: (artifactId: string) => void;
  preferredJobId?: string;
  onModel?: (artifactId: string) => void;
};

/** Distillation is a workflow; the ACT implementation is one supported adapter. */
export function DistillationPanel({ projectId, model, preferredTeacherArtifactId, onSelectModel, onDataset, onQuantize, onLibrary, preferredJobId: requestedJobId, onModel }: Props) {
  const jobs = useQuery({ queryKey: ['jobs', projectId], queryFn: () => api.jobs(projectId), enabled: !!projectId, retry: false, refetchInterval: 3_000 });
  const cachedAttempt = useQuery<PolicyJobAttempt>({ queryKey: ['distillation-attempt', projectId], queryFn: async () => null, enabled: false, initialData: null, gcTime: Infinity });
  const [recovery, setRecovery] = useState<{ attempt: PolicyJobAttempt; unreadable: boolean }>({ attempt: null, unreadable: false });
  const [preferredJobId, setPreferredJobId] = useState<string | undefined>(requestedJobId);
  const [view, setView] = useState<'jobs' | 'new'>(model || requestedJobId ? 'new' : 'jobs');
  useEffect(() => {
    if (!projectId || model) return;
    try { setRecovery({ attempt: storedAttempt('policy.distill', projectId), unreadable: false }); }
    catch { setRecovery({ attempt: null, unreadable: true }); }
  }, [projectId, model, cachedAttempt.data]);
  const history = (jobs.data ?? []).filter(job => job.project_id === projectId && studentJob(job)).sort((a, b) => b.created_at.localeCompare(a.created_at));
  const needsReview = !!cachedAttempt.data || !!recovery.attempt || recovery.unreadable;

  function openAct(jobId?: string) {
    if (!projectId) return;
    setPreferredJobId(jobId);
    setView('new');
    onSelectModel('act');
  }

  function back() { setPreferredJobId(undefined); setView('jobs'); onSelectModel(undefined); }

  if (model === 'act') return <div className="distillation-workflow">
    <NativeDistillationPanel projectId={projectId} preferredJobId={preferredJobId ?? requestedJobId} preferredTeacherArtifactId={preferredTeacherArtifactId} onDataset={onDataset} onQuantize={onQuantize} onBack={back} onModel={onModel} />
  </div>;

  return <section className="distillation-overview" aria-label="Distillation models">
    {view === 'jobs' ? <JobHistory title="Distillation jobs" newLabel="Start a new distillation" entries={history.map(job => ({id:job.id,title:'ACT → ACT256',subtitle:('artifact_id' in job.request ? job.request.artifact_id : undefined),status:job.status,createdAt:job.created_at,progress:isActive(job) ? job.stage ?? job.status : undefined}))} onNew={() => setView('new')} onSelect={id => openAct(id)} loading={jobs.isPending && !!projectId} error={jobs.error} disabled={!projectId} emptyMessage={projectId ? 'No distillation jobs yet.' : 'Select a project to see its jobs.'} /> : <><div className="workflow-view-navigation"><button type="button" className="text-link" onClick={back}>← Back to jobs</button></div><ModelWorkflowPicker projectId={projectId} action="distill" onLibrary={onLibrary} onSelect={artifact => { setPreferredJobId(undefined); onSelectModel('act', artifact.id); }} /></>}
    {projectId && needsReview && <div className="distillation-recovery" role={recovery.unreadable ? 'alert' : 'status'}>
      <p>{recovery.unreadable ? 'Saved request status could not be read.' : 'A distillation request needs review.'}</p>
      <button type="button" className="secondary-button" onClick={() => openAct()}>Review request</button>
    </div>}
  </section>;
}
