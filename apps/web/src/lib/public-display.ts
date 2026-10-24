/** Presentation aliases only: stored artifact formats and worker replies stay exact. */
export function publicFormatLabel(value: string): string {
  return value === 'firebird_quant' ? 'OpenJensen Quant' : value;
}
export function publicMessage(value: string): string {
  return value.replace('Experimental Firebird prompt;', 'Experimental OpenJensen prompt;');
}
