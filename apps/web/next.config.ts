import type { NextConfig } from 'next';
import { basePath } from './src/lib/base-path';

const nextConfig: NextConfig = {
  output: 'export',
  // Keep the normal preview available while validating a prefixed static export.
  distDir: process.env.FIREBIRD_PREFIX_SMOKE_EXPORT === '1' ? 'out/prefix-smoke' : '.next',
  basePath,
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
