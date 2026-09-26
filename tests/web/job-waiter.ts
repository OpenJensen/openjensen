import { setTimeout as delay } from 'node:timers/promises';
import { errors, expect, type APIRequestContext } from '@playwright/test';

const jobWaitMs = 45_000;
type ExpectedState = 'succeeded' | 'failed' | 'cancelled' | 'worker_started';

/** Observe real application jobs without retrying a terminal failure. */
export async function waitForJob(
  request: APIRequestContext,
  id: string,
  expected: ExpectedState,
  { origin = '', timeoutMs = jobWaitMs }: { origin?: string; timeoutMs?: number } = {},
) {
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0 || timeoutMs > jobWaitMs)
    throw new Error(`Worker wait budget must be in (0, ${jobWaitMs}]ms`);
  const deadline = performance.now() + timeoutMs;
  let last = `Job ${id}: no snapshot received`;
  const expired = (cause?: unknown) => new Error(
    `Job did not reach ${expected} within ${timeoutMs}ms. Last observation: ${last}`,
    { cause },
  );
  async function get(path: string) {
    const remaining = deadline - performance.now();
    try {
      return await request.get(`${origin}${path}`, {
        timeout: Math.max(1, Math.min(5_000, remaining)),
      });
    } catch (error) {
      // Only translate transport timeouts limited by the overall job deadline.
      // Earlier network failures remain immediate failures with their own cause.
      if (remaining <= 5_000 && error instanceof errors.TimeoutError) throw expired(error);
      throw error;
    }
  }
  while (performance.now() < deadline) {
    const response = await get(`/api/v1/jobs/${id}`);
    expect(response.status(), `Reading job ${id}: ${await response.text()}`).toBe(200);
    const job = await response.json();
    last = JSON.stringify({ id: job.id, status: job.status, stage: job.stage, error: job.error, updated_at: job.updated_at });
    if (!['queued', 'running'].includes(job.status)) {
      expect(job.status, `Job reached ${job.status}; expected ${expected}. ${last}`).toBe(expected);
      return job;
    }
    if (expected === 'worker_started') {
      const events = await get(`/api/v1/jobs/${id}/events`);
      expect(events.status(), `Reading worker events for ${id}: ${await events.text()}`).toBe(200);
      if ((await events.json()).some((event: { message: string }) => event.message === 'Optimizer step 0')) return job;
    }
    await delay(Math.max(0, Math.min(250, deadline - performance.now())));
  }
  throw expired();
}
