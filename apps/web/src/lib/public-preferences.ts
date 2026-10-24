type PreferenceReader = Pick<Storage, 'getItem'>;
type PreferenceWriter = Pick<Storage, 'setItem'>;

/** Only presentation preferences use these helpers; mutation journals retain their exact keys. */
export function readPublicPreference(storage: PreferenceReader, name: string): string | null {
  return storage.getItem(`jensen.${name}`) ?? storage.getItem(`firebird.${name}`);
}
export function writePublicPreference(storage: PreferenceWriter, name: string, value: string): void {
  storage.setItem(`jensen.${name}`, value);
}

// Apply the same canonical-first read before React hydrates to avoid a theme flash.
export const themeBootstrapScript =
  "try{document.documentElement.dataset.theme=(localStorage.getItem('jensen.theme')??localStorage.getItem('firebird.theme'))==='dark'?'dark':'light'}catch{document.documentElement.dataset.theme='light'}";
