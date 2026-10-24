import type { Metadata } from 'next';
import CloudRunsSection from '@/components/workspace-sections/cloud-runs';

export const metadata: Metadata = { title: 'OpenJensen · Cloud runs' };

export default function CloudRunsPage() {
  return <CloudRunsSection />;
}
