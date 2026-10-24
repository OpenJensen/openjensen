import type { Metadata } from 'next';
import DashboardSection from '@/components/workspace-sections/dashboard';

export const metadata: Metadata = { title: 'OpenJensen · Dashboard' };

export default function DashboardPage() {
  return <DashboardSection />;
}
