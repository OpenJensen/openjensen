import type { Metadata } from 'next';
import DistillationSection from '@/components/workspace-sections/distillation';

export const metadata: Metadata = { title: 'OpenJensen · Distill' };

export default function DistillationPage() {
  return <DistillationSection />;
}
