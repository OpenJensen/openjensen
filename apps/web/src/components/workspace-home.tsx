'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import DashboardSection from '@/components/workspace-sections/dashboard';

export function WorkspaceHome() {
  const router = useRouter();
  useEffect(() => { router.replace('/dashboard/', { scroll: false }); }, [router]);
  return <DashboardSection />;
}
