import type { Metadata } from 'next';
import DecisionLabSection from '@/components/workspace-sections/decision-lab';

export const metadata: Metadata = { title: 'OpenJensen · Decision Lab' };

export default function DecisionLabPage() {
  return <DecisionLabSection />;
}
