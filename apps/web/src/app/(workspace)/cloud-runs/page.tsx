import type { Metadata } from 'next';
import CloudRunsSection from '@/components/workspace-sections/cloud-runs';

export const metadata: Metadata = { title: 'Open Jensen · Cloud runs' };

export default function CloudRunsPage() {
  return <CloudRunsSection />;
}
