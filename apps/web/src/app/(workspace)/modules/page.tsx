import type { Metadata } from 'next';
import ModulesSection from '@/components/workspace-sections/modules';

export const metadata: Metadata = { title: 'OpenJensen · Modules' };

export default function Page() {
  return <ModulesSection />;
}
