'use client';

import { useEffect, useState } from 'react';
import { TeachingAgentControl, type AgentTarget } from '@/components/teaching-agent-control';
import { TeachingPanel } from '@/components/teaching-panel';
import { ManagedTeachingPanel } from '@/components/managed-teaching-panel';
import { RecordingPreparationPanel } from '@/components/recording-preparation-panel';
import { useWorkspace } from '@/components/workspace-context';

export default function TeachingSection() {
  const [agentTarget, setAgentTarget] = useState<AgentTarget>('robot_lab');
  const {
    teachingMode,
    requestedWorkJob,
    requestedJobOperation,
    setTeachingCapture,
    setTeachingMode,
    workflowProjectId,
    reviewTeachingCapture,
    teachingCapture,
    navigateStage,
    setSelectedJobId,
    setDatasetView,
    startTrainingOnDataset,
    isSectionCurrent,
  } = useWorkspace();
  useEffect(() => {
    if (requestedJobOperation?.id === 'teaching.capture' || requestedWorkJob?.operation === 'teaching.capture')
      setAgentTarget('teaching');
    else if (
      requestedWorkJob?.operation === 'agent.control' &&
      ['replay', 'teaching', 'robot_lab'].includes(String(requestedWorkJob.inputs.target))
    )
      setAgentTarget(requestedWorkJob.inputs.target as AgentTarget);
  }, [requestedJobOperation, requestedWorkJob]);
  return (
    <>
      <TeachingAgentControl
        key={`agent:${workflowProjectId}`}
        projectId={workflowProjectId}
        preferredSessionId={
          requestedWorkJob?.project_id === workflowProjectId && requestedWorkJob.operation === 'agent.control'
            ? requestedWorkJob.source_id
            : undefined
        }
        target={agentTarget}
        onTargetChange={setAgentTarget}
        onOpenProviders={() => navigateStage('cloud-apis')}
      />
      {agentTarget === 'teaching' && (
        <section className="agent-simulation-tools" aria-label="Simulation tools">
          <section className="panel" aria-label="AI teleoperation connection">
            <h2>Connection</h2>
            <div className="workbench-actions" role="group" aria-label="AI teleoperation mode">
              <button
                type="button"
                className={teachingMode === 'manual' ? 'primary-button' : 'secondary-button'}
                aria-pressed={teachingMode === 'manual'}
                onClick={() => {
                  setTeachingCapture(null);
                  setTeachingMode('manual');
                }}
              >
                Existing executor
              </button>
              <button
                type="button"
                className={teachingMode === 'managed' ? 'primary-button' : 'secondary-button'}
                aria-pressed={teachingMode === 'managed'}
                disabled={!workflowProjectId}
                onClick={() => {
                  setTeachingCapture(null);
                  setTeachingMode('managed');
                }}
              >
                Managed session
              </button>
            </div>
          </section>
          {teachingMode === 'managed' ? (
            <ManagedTeachingPanel
              key={`managed:${workflowProjectId}`}
              projectId={workflowProjectId}
              preferredJobId={
                requestedWorkJob?.project_id === workflowProjectId && requestedWorkJob.operation === 'teaching.capture'
                  ? requestedWorkJob.source_id
                  : undefined
              }
              onPublishedCapture={reviewTeachingCapture}
            />
          ) : (
            <TeachingPanel key="manual-teaching" />
          )}
          <RecordingPreparationPanel
            isActive={() => isSectionCurrent('teaching')}
            key={`recording:${workflowProjectId}`}
            projectId={workflowProjectId}
            teachingCapture={teachingCapture?.project_id === workflowProjectId ? teachingCapture : undefined}
            onTeachingCaptureConsumed={() => setTeachingCapture(null)}
            onInspect={(job) => {
              navigateStage('datasets');
              setSelectedJobId(job.id);
              setDatasetView('inspection');
            }}
            onTrain={(job) => startTrainingOnDataset(job.id)}
          />
        </section>
      )}
    </>
  );
}
