import type { Metadata } from 'next';
import { ApiReference } from '@/components/api-reference';

export const metadata: Metadata = {
  title: 'OPEN JENSEN · API reference',
  description: 'Explore the endpoints, request contracts, and response models of your OPEN JENSEN workspace.',
};

export default function DocsPage() {
  return <ApiReference />;
}
