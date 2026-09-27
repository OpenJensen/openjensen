'use client';

import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api, isActive } from '@/lib/api';
import { studentJob } from '@/lib/native-distillation';
import { storedAttempt, type PolicyJobAttempt } from '@/lib/policy-job-attempt';
import { NativeDistillationPanel } from './native-distillation-panel';
import { Icon } from './icon';
import './distillation-panel.css';

export type DistillationModel = 'act';

type Props = {
  projectId: string;
  model?: DistillationModel;
  onSelectModel: (model: DistillationModel | undefined) => void;
  onDataset: () => void;
  onQuantize: (artifactId: string) => void;
};

/** Distillation is a workflow; the ACT implementation is one supported adapter. */
export function DistillationPanel({ projectId, model, onSelectModel, onDataset, onQuantize }: Props) {
  const jobs = useQuery({ queryKey: ['jobs', projectId], queryFn: () => api.jobs(projectId), enabled: !!projectId, retry: false, refetchInterval: 3_000 });
  const cachedAttempt = useQuery<PolicyJobAttempt>({ queryKey: ['distillation-attempt', projectId], queryFn: async () => null, enabled: false, initialData: null, gcTime: Infinity });
  const [recovery, setRecovery] = useState<{ attempt: PolicyJobAttempt; unreadable: boolean }>({ attempt: null, unreadable: false });
  const [preferredJobId, setPreferredJobId] = useState<string>();
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
    onSelectModel('act');
  }

  if (model === 'act') return <div className="distillation-workflow">
    <button type="button" className="distillation-back" onClick={() => { setPreferredJobId(undefined); onSelectModel(undefined); }}><Icon name="arrow" size={15} />All models</button>
    <NativeDistillationPanel projectId={projectId} preferredJobId={preferredJobId} onDataset={onDataset} onQuantize={onQuantize} />
  </div>;

  return <section className="distillation-overview" aria-label="Distillation models">
    <div className="distillation-model-heading"><h2>Supported models</h2><span>1 model family</span></div>
    <div className="distillation-model-grid">
      <button type="button" className="distillation-model-card" aria-label="ACT" disabled={!projectId} onClick={() => openAct()}>
        <span className="distillation-model-icon"><Icon name="layers" size={24} /></span>
        <span className="distillation-model-copy"><strong>ACT</strong><span>Action Chunking with Transformers</span><small>ACT teacher → ACT256 student</small></span>
        <Icon name="arrow" size={17} />
      </button>
    </div>
    {!projectId && <p className="form-note" role="status">Select a project to continue.</p>}
    {projectId && needsReview && <div className="distillation-recovery" role={recovery.unreadable ? 'alert' : 'status'}>
      <p>{recovery.unreadable ? 'Saved request status could not be read.' : 'A distillation request needs review.'}</p>
      <button type="button" className="secondary-button" onClick={() => openAct()}>Review request</button>
    </div>}
    {projectId && jobs.isPending && <p className="field-help" role="status">Loading saved jobs…</p>}
    {jobs.isError && <div className="distillation-recovery"><p role="alert">Saved jobs could not be refreshed.</p><button type="button" className="secondary-button" disabled={jobs.isFetching} onClick={() => void jobs.refetch()}>Retry saved jobs</button></div>}
    {history.length > 0 && <section className="distillation-history-overview" aria-label="Saved distillation jobs">
      <h2>Saved jobs</h2>
      <div className="distillation-saved-jobs">{history.map(job => <button type="button" key={job.id} aria-label={`Open distillation ${job.id}`} onClick={() => openAct(job.id)}>
        <span><strong>ACT → ACT256</strong><small>{job.id.slice(0, 8)} · {new Date(job.created_at).toLocaleString()}</small></span>
        <span className={`status status-${job.status}`}>{isActive(job) ? job.stage ?? job.status : job.status}</span>
      </button>)}</div>
    </section>}
  </section>;
}
