import { expect, type Page, type Route } from '@playwright/test';

/** Existing specialized-workflow suites enter their editor explicitly from shared history. */
export async function openModuleWorkspace(page: Page) {
  const view = page.getByRole('group', { name: 'Module view', exact: true });
  const workspace = view.getByRole('button', { name: 'Workspace', exact: true });
  await expect(workspace).toBeVisible();
  await workspace.click();
}

/** Job navigation reads never start a worker in specialized browser fixtures. */
export async function fulfillWorkJobRead(route: Route): Promise<boolean> {
  const request = route.request();
  if (request.method() !== 'GET') return false;
  const path = new URL(request.url()).pathname;
  if (path.endsWith('/work-jobs') || path === '/api/v1/job-operations') {
    await route.fulfill({ json: [] });
    return true;
  }
  if (path === '/api/v1/compute-targets') {
    await route.fulfill({ json: { schema_version: 1, targets: [], runtime_options: [], connection_refs: [] } });
    return true;
  }
  return false;
}
