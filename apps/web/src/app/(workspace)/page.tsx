import type { Metadata } from 'next';
import { WorkspaceHome } from '@/components/workspace-home';

export const metadata: Metadata = { title: 'Open Jensen · Dashboard' };

export default function Home() {
  return <WorkspaceHome />;
}
