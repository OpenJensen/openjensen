/** Canonical server/test configuration, with read compatibility for alpha deployments. */
export function getenv(
  name: string,
  values: Record<string, string | undefined> = process.env,
  fallback?: string,
): string | undefined {
  const suffix = name.startsWith('JENSEN_') ? name.slice(7) : name.startsWith('FIREBIRD_') ? name.slice(9) : null;
  return suffix === null
    ? (values[name] ?? fallback)
    : (values[`JENSEN_${suffix}`] ?? values[`FIREBIRD_${suffix}`] ?? fallback);
}
