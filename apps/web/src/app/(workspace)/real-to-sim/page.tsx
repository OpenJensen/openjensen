import type { Metadata } from 'next';
import RealToSimSection from '@/components/workspace-sections/real-to-sim';

export const metadata: Metadata = { title: 'OpenJensen · Real to sim' };

export default function RealToSimPage() {
  return <RealToSimSection />;
}
