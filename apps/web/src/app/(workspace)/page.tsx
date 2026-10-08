import type { Metadata } from 'next';
import DashboardSection from '@/components/workspace-sections/dashboard';

export const metadata: Metadata = { title: 'Open Jensen · Dashboard' };

export default function Home() {
  return <DashboardSection />;
}
