import type { Metadata } from 'next';
import SimulationSection from '@/components/workspace-sections/simulation';

export const metadata: Metadata = { title: 'OpenJensen · Simulate' };

export default function SimulationPage() {
  return <SimulationSection />;
}
