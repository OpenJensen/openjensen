import type { Metadata } from 'next';
import EngineWorkflowSection from '@/components/workspace-sections/engine-workflow';

export const metadata: Metadata = { title: 'Open Jensen · Settings & diagnostics' };

export default function SettingsPage() {
  return <EngineWorkflowSection />;
}
