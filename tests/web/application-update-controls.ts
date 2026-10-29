import { jobOperationFixtures } from './work-job-controls';
import type { Route } from '@playwright/test';

// Isolate other workflow fixtures from the header's read-only version discovery.
// Update installation requests still pass through their strict mutation assertions.
export async function fulfillApplicationUpdateRead(route: Route) {
  const request = route.request();
  const path = new URL(request.url()).pathname;
  if (request.method() === 'GET' && path === '/api/v1/job-operations') {
    await route.fulfill({ json: jobOperationFixtures });
    return true;
  }
  if (request.method() !== 'GET' || !['/api/v1/updates', '/api/v1/updates/check'].includes(path)) return false;
  await route.fulfill({ status: 404, json: { detail: 'Update discovery is outside this workflow fixture.' } });
  return true;
}
