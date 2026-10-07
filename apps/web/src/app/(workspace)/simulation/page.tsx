import type { Metadata } from 'next';
import SimulationSection from '@/components/workspace-sections/simulation';

export const metadata: Metadata = { title: 'Open Jensen · Run' };

export default function SimulationPage() {
  return <SimulationSection />;
}
