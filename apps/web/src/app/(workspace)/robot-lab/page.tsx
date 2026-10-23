import type { Metadata } from 'next';
import { RobotLab } from '@/components/robot-lab';

export const metadata: Metadata = { title: 'Open Jensen · Robot Lab' };

export default function RobotLabPage() {
  return <RobotLab />;
}
