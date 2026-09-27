import { apiOrigin } from './api';

export class UncertainPolicyJob extends Error {
  constructor() { super('The request outcome is unverified. Refresh and inspect the recorded jobs before submitting again. No automatic retry was made.'); }
}
export class PolicyJobHttpError extends Error {
  constructor(message: string, readonly status: number, readonly initialRejection = false) { super(message); }
}
export type PolicyJobRequestOptions = { idempotencyKey?: string; allowInitialRejection?: boolean };
export function record(value: unknown): value is Record<string, unknown> { return value !== null && typeof value === 'object' && !Array.isArray(value); }

// Lifecycle mutations must not silently replay after an ambiguous network result.
export async function policyJobRequest(path: string, body?: unknown, options: PolicyJobRequestOptions = {}): Promise<unknown> {
  const key = options.idempotencyKey;
  if (key !== undefined && !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(key)) throw new Error('Invalid submission key.');
  const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 15_000);
  try {
    const response = await fetch(`${apiOrigin}/api/v1${path}`, { method: body === undefined ? 'GET' : 'POST', body: body === undefined ? undefined : JSON.stringify(body), signal: controller.signal, cache: 'no-store', redirect: 'error', headers: { 'Content-Type': 'application/json', ...(key ? { 'Idempotency-Key': key } : {}) } });
    const reader = response.body?.getReader();
    if (!reader) throw new Error('Application returned an empty response.');
    const parts: Uint8Array[] = []; let bytes = 0;
    try { for (;;) { const next = await reader.read(); if (next.done) break; bytes += next.value.length; if (bytes > 1024 * 1024) throw new Error('Application response is too large.'); parts.push(next.value); } }
    finally { await reader.cancel(); }
    const data = new Uint8Array(bytes); let offset = 0; for (const part of parts) { data.set(part, offset); offset += part.length; }
    const value: unknown = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(data));
    const validRejection = record(value) && ((typeof value.detail === 'string' && value.detail.length > 0 && value.detail.length <= 2000) || (Array.isArray(value.detail) && value.detail.length > 0 && value.detail.length <= 100 && value.detail.every(item => record(item) && typeof item.msg === 'string' && item.msg.length > 0 && item.msg.length <= 2000)));
    const initialRejection = body !== undefined && options.allowInitialRejection === true && [400, 422].includes(response.status) && validRejection;
    // Success and authoritative missing-key reads must prove contract support.
    // A valid initial validation rejection cannot have allocated a job; its
    // framework response may omit custom headers. Retried/ambiguous keys do not
    // get this exception and must remain reconcilable.
    if (key && !initialRejection && (response.headers.get('Idempotency-Key') !== key || !response.headers.get('Cache-Control')?.split(',').some(part => part.trim().toLowerCase() === 'no-store'))) throw new Error('The application did not verify durable submission support. The saved request remains unresolved; no automatic retry was made.');
    if (!response.ok) {
      if (body !== undefined && response.status >= 500) throw new UncertainPolicyJob();
      const detail = record(value) && typeof value.detail === 'string' ? value.detail : record(value) && Array.isArray(value.detail) ? value.detail.filter(record).map(item => typeof item.msg === 'string' ? item.msg : 'Invalid request').slice(0, 10).join('; ') : `Application returned HTTP ${response.status}.`;
      throw new PolicyJobHttpError(detail.slice(0, 2000), response.status, initialRejection);
    }
    return value;
  } catch (error) {
    if (body !== undefined && !(error instanceof PolicyJobHttpError)) throw new UncertainPolicyJob();
    throw error instanceof Error ? error : new Error('Application request failed.');
  } finally { clearTimeout(timer); }
}

export function sameJson(left: unknown, right: unknown): boolean {
  if (Array.isArray(left)) return Array.isArray(right) && left.length === right.length && left.every((item, index) => sameJson(item, right[index]));
  if (record(left)) return record(right) && Object.keys(left).length === Object.keys(right).length && Object.keys(left).every(key => Object.hasOwn(right, key) && sameJson(left[key], right[key]));
  return left === right;
}
