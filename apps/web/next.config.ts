import type { NextConfig } from 'next';
import { basePath } from './src/lib/base-path';

const nextConfig: NextConfig = {
  output: 'export',
  basePath,
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
