import type { Metadata } from 'next';
import AugmentationSection from '@/components/workspace-sections/augmentation';

export const metadata: Metadata = { title: 'Open Jensen · Augmentation' };

export default function AugmentationPage() {
  return <AugmentationSection />;
}
