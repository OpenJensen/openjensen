import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Firebird · Dataset workspace',
  description: 'A local robotics workspace. Inspect dataset metadata and prepare the path from data to a tested policy.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
