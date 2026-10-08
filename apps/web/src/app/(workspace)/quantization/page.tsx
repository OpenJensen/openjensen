import type { Metadata } from 'next';
import QuantizationSection from '@/components/workspace-sections/quantization';

export const metadata: Metadata = { title: 'Open Jensen · Quantize' };

export default function QuantizationPage() {
  return <QuantizationSection />;
}
