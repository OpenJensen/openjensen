import type { Metadata } from 'next';
import { RobotLab } from '@/components/robot-lab';

export const metadata: Metadata = { title: 'OpenJensen · Initial setup' };

export default function RobotLabPage() {
  return <RobotLab />;
}
