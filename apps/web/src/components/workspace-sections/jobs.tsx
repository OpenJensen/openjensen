'use client';
import { WorkJobs } from '@/components/work-jobs';
import { useWorkspace } from '@/components/workspace-context';
export default function JobsSection() {
  const { workflowProjectId, newWorkJob, openWorkJob } = useWorkspace();
  return <WorkJobs key={workflowProjectId} projectId={workflowProjectId} onNew={newWorkJob} onOpen={openWorkJob} />;
}
