import { expect, test, type Page } from '@playwright/test';

async function teaching(page: Page, connected = true) {
  const commands: any[] = [];
  const changes = { failState: false, nextSession: false };
  const unexpected: string[] = [];
  const state = { mode: 'running', episode_id: 'episode-1', revision: 7, session_id: 'session-1', instruction: 'Move gripper', outcome: 'unknown', steps: 10, sim_time: .3, joints: ['gripper'], state_rad: [0], fault: null };
  await page.route('**/api/v1/**', async route => {
    const request = route.request(), path = new URL(request.url()).pathname;
    if (request.method() === 'POST' && path === '/api/v1/teaching/commands') {
      const body = request.postDataJSON(); commands.push(body);
      await route.fulfill({ status: 202, json: { command_id: body.command_id, status: 'queued' } }); return;
    }
    if (path.startsWith('/api/v1/teaching/commands/')) {
      await route.fulfill({ json: { command_id: path.split('/').pop(), status: 'acknowledged' } }); return;
    }
    if (path === '/api/v1/teaching/voice/join') {
      await route.fulfill({ status: 503, json: { detail: 'Voice settings are incomplete.' } }); return;
    }
    if (path === '/api/v1/teaching/state' && changes.failState) {
      await route.fulfill({ status: 503, json: { detail: 'Executor unavailable' } }); return;
    }
    if (path === '/api/v1/teaching/state' && changes.nextSession) {
      await route.fulfill({ json: { connected: true, state: { ...state, session_id: 'new-session' }, voice_configured: true } }); return;
    }
    const replies: Record<string, unknown> = {
      '/api/v1/health': { status: 'ok', version: 'test' },
      '/api/v1/capabilities': [],
      '/api/v1/projects': [{ id: 'teaching', name: 'Teaching test', created_at: '2026-09-27T00:00:00Z' }],
      '/api/v1/projects/teaching/jobs': [],
      '/api/v1/teaching/state': { connected, state: connected ? state : null, message: connected ? null : 'Connect a teaching executor in the application host configuration.', voice_configured: connected },
    };
    if (request.method() === 'GET' && path in replies) { await route.fulfill({ json: replies[path] }); return; }
    unexpected.push(request.method() + ' ' + path);
    await route.fulfill({ status: 405, json: { detail: 'No external calls allowed by test.' } });
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Teaching', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Teach in simulation' })).toBeVisible();
  return { commands, unexpected, changes };
}

test('disconnected executor disables recording and voice without pretending monitor is control', async ({ page }) => {
  const { commands, unexpected } = await teaching(page, false);
  await expect(page.getByRole('button', { name: 'Start recording' })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Connect voice' })).toBeDisabled();
  await expect(page.getByText(/existing rollout monitor does not provide teaching control/)).toBeVisible();
  expect(commands).toEqual([]); expect(unexpected).toEqual([]);
  const size = await page.evaluate(() => [document.documentElement.scrollWidth, document.documentElement.clientWidth]);
  expect(size[0]).toBeLessThanOrEqual(size[1] + 1);
});

test('correction binds fresh session revision and waits for applied receipt', async ({ page }) => {
  const { commands, unexpected } = await teaching(page);
  await page.getByRole('button', { name: '+ 0.05 rad', exact: true }).click();
  await expect(page.getByText('Command: acknowledged', { exact: true })).toBeVisible();
  expect(commands).toHaveLength(1);
  expect(commands[0]).toMatchObject({ session_id: 'session-1', episode_id: 'episode-1', expected_revision: 7, operation: 'correct', arguments: { joint: 'gripper', delta_rad: .05 } });
  expect(commands[0].command_id).toMatch(/^[a-f0-9-]{36}$/);
  expect(unexpected).toEqual([]);
});

test('missing voice configuration does not claim microphone or agent is connected', async ({ page }) => {
  const { commands, unexpected } = await teaching(page);
  await page.getByRole('button', { name: 'Connect voice' }).click();
  await expect(page.getByRole('alert').filter({ hasText: 'Voice could not connect' })).toContainText('Voice could not connect');
  await expect(page.getByText('Microphone connected', { exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Connect voice' })).toBeEnabled();
  expect(commands).toEqual([]); expect(unexpected).toEqual([]);
});


test('later connection failure disables cached teaching controls', async ({ page }) => {
  const { changes, commands } = await teaching(page);
  await expect(page.getByRole('button', { name: '+ 0.05 rad', exact: true })).toBeEnabled();
  changes.failState = true;
  await expect(page.getByRole('button', { name: '+ 0.05 rad', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Connect voice' })).toBeDisabled();
  expect(commands).toEqual([]);
});

test('a command cannot silently switch to a restarted simulator', async ({ page }) => {
  const { changes, commands } = await teaching(page);
  await expect(page.getByRole('button', { name: '+ 0.05 rad', exact: true })).toBeEnabled();
  changes.nextSession = true;
  await page.getByRole('button', { name: '+ 0.05 rad', exact: true }).click();
  await expect(page.getByText(/simulator session changed/)).toBeVisible();
  expect(commands).toEqual([]);
});
