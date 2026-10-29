/** Readable model labels; the exact API ID remains the saved identity. */
export function agentModelName(model: string): string {
  if (model === 'deterministic-mock' || model === 'deterministic-mock-v1') return 'Deterministic mock';
  if (!model.startsWith('gpt-')) return model;
  const [family, ...suffix] = model.slice(4).split('-');
  return `GPT-${family}${suffix.length ? ` ${suffix.map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join(' ')}` : ''}`;
}
