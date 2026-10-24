import type { Metadata } from 'next';
import ModelsSection from '@/components/workspace-sections/models';

export const metadata: Metadata = { title: 'OpenJensen · Models' };

export default function ModelsPage() {
  return <ModelsSection />;
}
