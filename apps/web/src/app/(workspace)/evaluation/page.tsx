import type { Metadata } from 'next';
import EvaluationSection from '@/components/workspace-sections/evaluation';

export const metadata: Metadata = { title: 'OpenJensen · Evaluate' };

export default function EvaluationPage() {
  return <EvaluationSection />;
}
