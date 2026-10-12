import type { Metadata } from 'next';
import './globals.css';
import { themeBootstrapScript } from '@/lib/public-preferences';

export const metadata: Metadata = {
  title: 'Open Jensen · Workspace',
  description: 'Open Jensen is a local robotics workspace. Inspect dataset metadata and prepare the path from data to a tested policy.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en" data-theme="light" suppressHydrationWarning><head><script dangerouslySetInnerHTML={{ __html: themeBootstrapScript }} /></head><body>{children}</body></html>;
}
