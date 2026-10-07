import type { Metadata } from 'next';
import ModelsSection from '@/components/workspace-sections/models';

export const metadata: Metadata = { title: 'Open Jensen · My models' };

export default function ModelsPage() {
  return <ModelsSection />;
}
