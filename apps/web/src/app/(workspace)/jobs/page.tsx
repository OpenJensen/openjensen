import type { Metadata } from 'next';
import JobsSection from '@/components/workspace-sections/jobs';
export const metadata: Metadata = { title: 'OpenJensen · Jobs' };
export default function JobsPage() {
  return <JobsSection />;
}
