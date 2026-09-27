import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'OPEN JENSEN · Dataset workspace',
  description: 'A local robotics workspace. Inspect dataset metadata and prepare the path from data to a tested policy.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en" data-theme="light" suppressHydrationWarning><head><script dangerouslySetInnerHTML={{ __html: "try{document.documentElement.dataset.theme=localStorage.getItem('firebird.theme')==='dark'?'dark':'light'}catch{document.documentElement.dataset.theme='light'}" }} /></head><body>{children}</body></html>;
}
