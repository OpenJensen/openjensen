'use client';

import { TeachingPanel } from '@/components/teaching-panel';
import { ManagedTeachingPanel } from '@/components/managed-teaching-panel';
import { RecordingPreparationPanel } from '@/components/recording-preparation-panel';
import { useWorkspace } from '@/components/workspace-context';

export default function TeachingSection() {
  const { teachingMode, setTeachingCapture, setTeachingMode, workflowProjectId, reviewTeachingCapture, teachingCapture, navigateStage, setSelectedJobId, setDatasetView, startTrainingOnDataset, isSectionCurrent } = useWorkspace();
  return <>
      <section className="panel" aria-label="Teaching connection"><h2>Choose a teaching connection</h2><div className="workbench-actions" role="group" aria-label="Teaching mode">
        <button type="button" className={teachingMode === 'manual' ? 'primary-button' : 'secondary-button'} aria-pressed={teachingMode === 'manual'} onClick={() => { setTeachingCapture(null); setTeachingMode('manual'); }}>Existing executor</button>
        <button type="button" className={teachingMode === 'managed' ? 'primary-button' : 'secondary-button'} aria-pressed={teachingMode === 'managed'} disabled={!workflowProjectId} onClick={() => { setTeachingCapture(null); setTeachingMode('managed'); }}>Managed session</button>
      </div><p>Use an existing teaching connection, or explicitly start a configured session for this project. Switching views does not stop a managed session.</p></section>
      {teachingMode === 'managed' ? <ManagedTeachingPanel key={`managed:${workflowProjectId}`} projectId={workflowProjectId} onPublishedCapture={reviewTeachingCapture} /> : <TeachingPanel key="manual-teaching" />}
      <RecordingPreparationPanel isActive={() => isSectionCurrent(9)} key={`recording:${workflowProjectId}`} projectId={workflowProjectId} teachingCapture={teachingCapture?.project_id === workflowProjectId ? teachingCapture : undefined} onTeachingCaptureConsumed={() => setTeachingCapture(null)} onInspect={job => { navigateStage(0); setSelectedJobId(job.id); setDatasetView('inspection'); }} onTrain={job => startTrainingOnDataset(job.id)} />
  </>;
}
