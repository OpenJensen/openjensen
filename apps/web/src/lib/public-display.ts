/** Presentation aliases only: stored artifact formats and worker replies stay exact. */
export function publicFormatLabel(value: string): string {
  return value === 'firebird_quant' ? 'OPEN JENSEN Quant' : value;
}
export function publicMessage(value: string): string {
  return value.replace('Experimental Firebird prompt;', 'Experimental OPEN JENSEN prompt;');
}
