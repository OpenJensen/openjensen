/** Public mount path, captured at build time so exports also work behind a prefix. */
export function normalizeBasePath(value: string | undefined) {
  const path = (value ?? '').trim().replace(/\/+$/, '');
  if (!path) return '';
  if (!/^\/(?:[A-Za-z0-9_-]+\/)*[A-Za-z0-9_-]+$/.test(path)) {
    throw new Error('NEXT_PUBLIC_BASE_PATH must be an absolute URL path such as /firebird.');
  }
  return path;
}

export const basePath = normalizeBasePath(process.env.NEXT_PUBLIC_BASE_PATH);

export function publicPath(path: string) {
  if (!path.startsWith('/') || path.startsWith('//') || !basePath || path === basePath || path.startsWith(`${basePath}/`)) return path;
  return `${basePath}${path}`;
}
