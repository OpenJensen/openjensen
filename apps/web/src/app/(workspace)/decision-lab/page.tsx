import type { Metadata } from 'next';
import DecisionLabSection from '@/components/workspace-sections/decision-lab';

export const metadata: Metadata = { title: 'Open Jensen · Decision lab' };

export default function DecisionLabPage() {
  return <DecisionLabSection />;
}
