import { record } from './policy-job-mutation';

export type PolicyJobAttempt = { state: 'pending' | 'uncertain'; message: string } | null;
const interrupted: PolicyJobAttempt = { state: 'uncertain', message: 'An earlier request did not return a verified outcome in this browser session. Refresh and inspect the recorded jobs before submitting again. No automatic retry was made.' };
const key = (operation: string, project: string) => `firebird:job-attempt:${operation}:${project}`;

/** Persist no credentials or model data. Reload never turns an unknown mutation into a retry. */
export function storedAttempt(operation: string, project: string): PolicyJobAttempt {
  const value = sessionStorage.getItem(key(operation, project));
  if (value === null) return null;
  try {
    const parsed: unknown = JSON.parse(value);
    if (record(parsed) && parsed.state === 'uncertain' && typeof parsed.message === 'string' && parsed.message.length <= 2000) return { state: 'uncertain', message: parsed.message };
  } catch { /* An unreadable journal still requires an explicit history review. */ }
  return interrupted;
}

export function storeAttempt(operation: string, project: string, value: PolicyJobAttempt): void {
  const name = key(operation, project);
  if (value === null) sessionStorage.removeItem(name);
  else sessionStorage.setItem(name, JSON.stringify(value));
}
