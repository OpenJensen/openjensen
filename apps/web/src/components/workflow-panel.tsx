'use client';

import { JobHistory } from '@/components/job-history';
import { isActive } from '@/lib/api';
import './workflow-panel.css';
import { WorkflowJobDetail } from './workflow-job-detail';
import { WorkflowNewJob } from './workflow-new-job';
import { WorkflowSettingsPanel } from './workflow-settings-panel';
import { currentStage, type SettingsTab } from './workflow-shared';
import { useWorkflowPanel } from './workflow-state';

export { currentStage, precisionName, type SettingsTab } from './workflow-shared';

export function WorkflowPanel({
  projectId,
  startNew,
  stage,
  tab,
  onTabChange: setTab,
  onOpenQuantize,
  preferredArtifactId,
  preferredJobId,
  onViewTraining,
  onBackToJobs,
  onBackToModels,
  modelRunId,
  onModel,
}: {
  projectId: string;
  startNew?: boolean;
  stage: string;
  tab: SettingsTab;
  onTabChange: (tab: SettingsTab) => void;
  onOpenQuantize: () => void;
  preferredArtifactId?: string;
  preferredJobId?: string;
  onViewTraining?: () => void;
  onBackToJobs?: () => void;
  onBackToModels?: () => void;
  modelRunId?: string;
  onModel?: (artifactId: string) => void;
}) {
  const workflow = useWorkflowPanel({
    projectId,
    startNew,
    stage,
    preferredArtifactId,
    preferredJobId,
    modelRunId,
    onBackToJobs,
    onBackToModels,
  });
  const {
    checkpointFor,
    historyTitle,
    jobs,
    modelFor,
    newLabel,
    openJob,
    openNew,
    precisionFor,
    ready,
    stageJobs,
    view,
  } = workflow;
  if (stage === 'settings')
    return <WorkflowSettingsPanel workflow={workflow} tab={tab} onTabChange={setTab} onOpenQuantize={onOpenQuantize} />;

  if (view === 'history')
    return (
      <JobHistory
        title={historyTitle}
        newLabel={newLabel}
        entries={stageJobs.map((job) => ({
          id: job.id,
          title: [modelFor(job), precisionFor(job) !== '—' ? precisionFor(job) : null].filter(Boolean).join(' · '),
          subtitle: checkpointFor(job),
          status: job.status,
          createdAt: job.created_at,
          progress: isActive(job) ? currentStage(job) : undefined,
        }))}
        onNew={openNew}
        onSelect={openJob}
        loading={jobs.isPending && !!projectId}
        error={jobs.error}
        disabled={!ready}
        emptyMessage={projectId ? 'No jobs yet.' : 'Select a project to see its jobs.'}
      />
    );

  if (view === 'detail') return <WorkflowJobDetail workflow={workflow} onModel={onModel} />;

  return <WorkflowNewJob workflow={workflow} onOpenQuantize={onOpenQuantize} onViewTraining={onViewTraining} />;
}
