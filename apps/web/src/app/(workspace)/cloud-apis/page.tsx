import type { Metadata } from 'next';
import CloudApisSection from '@/components/workspace-sections/cloud-apis';

export const metadata: Metadata = { title: 'OpenJensen · Cloud & APIs' };

export default function Page() {
  return <CloudApisSection />;
}
