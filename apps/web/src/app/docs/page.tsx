import type { Metadata } from 'next';
import { ApiReference } from '@/components/api-reference';

export const metadata: Metadata = {
  title: 'Firebird · API reference',
  description: 'Explore the endpoints, request contracts, and response models of your Firebird workspace.',
};

export default function DocsPage() {
  return <ApiReference />;
}
