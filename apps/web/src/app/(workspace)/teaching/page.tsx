import type { Metadata } from 'next';
import TeachingSection from '@/components/workspace-sections/teaching';

export const metadata: Metadata = { title: 'Open Jensen · Teaching' };

export default function TeachingPage() {
  return <TeachingSection />;
}
