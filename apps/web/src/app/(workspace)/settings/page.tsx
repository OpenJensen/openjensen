import type { Metadata } from 'next';
import EngineWorkflowSection from '@/components/workspace-sections/engine-workflow';

export const metadata: Metadata = { title: 'OpenJensen · Settings' };

export default function SettingsPage() {
  return <EngineWorkflowSection />;
}
