import type { Metadata } from 'next';
import TeachingSection from '@/components/workspace-sections/teaching';

export const metadata: Metadata = { title: 'OpenJensen · AI teleoperation' };

export default function TeachingPage() {
  return <TeachingSection />;
}
