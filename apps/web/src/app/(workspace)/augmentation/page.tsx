import type { Metadata } from 'next';
import AugmentationSection from '@/components/workspace-sections/augmentation';

export const metadata: Metadata = { title: 'OpenJensen · Augmentation' };

export default function AugmentationPage() {
  return <AugmentationSection />;
}
