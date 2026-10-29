'use client';

import { SceneGenerationPanel } from '@/components/scene-generation-panel';
import { useWorkspace } from '@/components/workspace-context';

export default function RealToSimSection() {
  const { workflowProjectId } = useWorkspace();
  return <SceneGenerationPanel key={workflowProjectId} projectId={workflowProjectId} />;
}
