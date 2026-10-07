import { expect, test, type Page } from '@playwright/test';
import { createHash } from 'node:crypto';

async function teaching(page: Page, connected = true) {
  const commands: any[] = [];
  const changes = { failState: false, disconnected: false, nextSession: false, frameFailure: false, frameChanges: {} as Record<string, unknown>, voiceConfigured: connected, frames: 0 };
  const unexpected: string[] = [];
  const state = { mode: 'running', episode_id: 'episode-1', revision: 7, session_id: 'session-1', instruction: 'Move gripper', outcome: 'unknown', steps: 10, sim_time: .3, joints: ['gripper'], state_rad: [0], fault: null };
  await page.route('**/api/v1/**', async route => {
    const request = route.request(), path = new URL(request.url()).pathname;
    if (request.method() === 'GET' && path === '/api/v1/datasets') return route.fulfill({ json: [] });
    if (path === '/api/v1/teaching/frame') {
      changes.frames += 1;
      if (changes.frameFailure) return route.fulfill({ status: 503, json: { detail: 'Generated unavailable frame' } });
      return route.fulfill({ json: frameFixture(changes.nextSession ? 'new-session' : 'session-1', changes.frameChanges) });
    }
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
    if (path === '/api/v1/teaching/state' && changes.disconnected) return route.fulfill({ json: { connected: false, state: null, message: 'Executor disconnected', voice_configured: false } });
    if (path === '/api/v1/teaching/state' && changes.failState) {
      await route.fulfill({ status: 503, json: { detail: 'Executor unavailable' } }); return;
    }
    if (path === '/api/v1/teaching/state' && changes.nextSession) {
      await route.fulfill({ json: { connected: true, state: { ...state, session_id: 'new-session' }, voice_configured: true } }); return;
    }
    const replies: Record<string, unknown> = {
      '/api/v1/teaching/intelligence/status': { broker_reachable: false, configured: false, busy: false },
      '/api/v1/teaching/intelligence/settings': { saved: false, revision: null, voice_model: null, message: 'Not saved' },
      '/api/v1/teaching/voice/status': { broker_reachable: changes.voiceConfigured, configuration_present: changes.voiceConfigured, dependencies_present: changes.voiceConfigured, message: 'Fixture configuration only' },
      '/api/v1/health': { status: 'ok', version: 'test' },
      '/api/v1/capabilities': [],
      '/api/v1/projects': [{ id: 'teaching', name: 'Teaching test', created_at: '2026-09-27T00:00:00Z' }],
      '/api/v1/projects/teaching/jobs': [],
      '/api/v1/projects/teaching/recordings/options': { configured: false, runtime_verified: false, configuration_sha256: null, max_episodes: 100, max_source_bytes: 8589934592, setup_message: 'Recording preparation is not configured in this fixture.' },
      '/api/v1/teaching/state': { connected, state: connected ? state : null, message: connected ? null : 'Connect a teaching executor in the application host configuration.', voice_configured: changes.voiceConfigured },
    };
    if (request.method() === 'GET' && path in replies) { await route.fulfill({ json: replies[path] }); return; }
    unexpected.push(request.method() + ' ' + path);
    await route.fulfill({ status: 405, json: { detail: 'No external calls allowed by test.' } });
  });
  await page.goto('/datasets/');
  await page.getByRole('link', { name: 'Teaching', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Teaching', level: 1 })).toBeVisible();
  await expect(page.getByRole('group', { name: 'Control source', exact: true })).toContainText('Isaac Sim');
  return { commands, unexpected, changes };
}

test('disconnected executor disables recording and voice without pretending monitor is control', async ({ page }) => {
  const { commands, unexpected } = await teaching(page, false);
  await expect(page.getByLabel('Current project')).toBeEnabled();
  await expect(page.getByRole('button', { name: 'Start recording' })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Connect voice' })).toBeDisabled();
  await expect(page.getByText('Connect a teaching executor in the application host configuration.', { exact: true })).toBeVisible();
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

function frameFixture(session: string, changes: Record<string, unknown> = {}) {
  const rgb = Buffer.from([200, 20, 30, 10, 200, 30, 10, 20, 200, 200, 200, 20]);
  const context = { session_id: session, revision: 7, active_episode_id: 'episode-1', mode: 'running' };
  return { schema_version: 1, available: true, ...context, current_context: context,
    capture_id: '1152921504606846976', episode_id: 'episode-1', step: 10, sim_time: .3,
    camera_key: 'observation.images.front', camera_prim: '/World/front', width: 2, height: 2,
    rgb_base64: rgb.toString('base64'), rgb_sha256: createHash('sha256').update(rgb).digest('hex'),
    relay_age_ns: 1000, joints: ['gripper'], state_rad: [.25], units: 'rad', ...changes };
}

test('camera is read-only and works without voice while displaying actual RGB and joints', async ({ page }) => {
  const { changes, commands, unexpected } = await teaching(page);
  changes.voiceConfigured = false;
  await expect(page.getByRole('img', { name: 'Fresh simulator camera', exact: true })).toBeVisible();
  const pixel = await page.locator('canvas').evaluate(el => Array.from((el as HTMLCanvasElement).getContext('2d')!.getImageData(0, 0, 1, 1).data));
  expect(pixel).toEqual([200, 20, 30, 255]);
  await expect(page.getByText('0.2500 rad', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Connect voice' })).toBeDisabled();
  await expect(page.getByText('Executor reachable', { exact: true })).toBeVisible();
  expect(commands).toEqual([]); expect(unexpected).toEqual([]);
});

test('invalidated camera capture cannot revive on replay and needs genuinely newer pixels', async ({ page }) => {
  const { changes, commands } = await teaching(page);
  await expect(page.getByRole('img', { name: 'Fresh simulator camera', exact: true })).toBeVisible();
  changes.frameChanges = { capture_id: '1152921504606846977', current_context: { session_id: 'session-1', revision: 8, active_episode_id: 'episode-1', mode: 'paused' } };
  await expect(page.getByRole('img', { name: 'Simulator camera unavailable or stale' })).toBeVisible();
  const previousRequests = changes.frames;
  changes.frameChanges = { capture_id: '1152921504606846977' };
  await expect.poll(() => changes.frames).toBeGreaterThan(previousRequests + 1);
  // A replay with source_age reset must not revive a capture already invalidated.
  await expect(page.getByText(/Previous observation — not live/)).toBeVisible();
  await expect(page.getByRole('img', { name: 'Fresh simulator camera', exact: true })).toHaveCount(0);
  changes.frameChanges = { capture_id: '1152921504606846978' };
  await expect(page.getByRole('img', { name: 'Fresh simulator camera', exact: true })).toBeVisible();
  expect(commands).toEqual([]);
});

test('unchanged observation expires locally even when successful replies reset source age', async ({ page }) => {
  await teaching(page);
  await expect(page.getByRole('img', { name: 'Fresh simulator camera', exact: true })).toBeVisible();
  await expect(page.getByRole('img', { name: 'Simulator camera unavailable or stale' })).toBeVisible({ timeout: 8000 });
  await expect(page.getByText(/Previous observation — not live/)).toBeVisible();
});

test('corrupt RGB and disconnected executor never render a fresh preview', async ({ page }) => {
  const { changes, commands } = await teaching(page);
  changes.frameChanges = { rgb_sha256: '0'.repeat(64) };
  await expect(page.getByText('Preview unavailable, stale or changed.', { exact: false })).toBeVisible();
  await expect(page.getByRole('img', { name: 'Fresh simulator camera', exact: true })).toHaveCount(0);
  changes.failState = true;
  await expect(page.getByRole('button', { name: 'Start recording' })).toBeDisabled();
  expect(commands).toEqual([]);
});


test('executor reconnect cannot revive the last capture after an explicit disconnect', async ({ page }) => {
  const { changes } = await teaching(page);
  await expect(page.getByRole('img', { name: 'Fresh simulator camera', exact: true })).toBeVisible();
  changes.disconnected = true;
  await expect(page.getByRole('img', { name: 'Simulator camera unavailable or stale' })).toBeVisible();
  const requests = changes.frames;
  changes.disconnected = false;
  await expect.poll(() => changes.frames).toBeGreaterThan(requests + 1);
  await expect(page.getByRole('img', { name: 'Fresh simulator camera', exact: true })).toHaveCount(0);
  changes.frameChanges = { capture_id: '1152921504606846978' };
  await expect(page.getByRole('img', { name: 'Fresh simulator camera', exact: true })).toBeVisible();
});

async function intelligence(page: Page, options: { fail?: boolean; hold?: boolean } = {}) {
  const posts: { path: string; body: any }[] = [];
  let release: (() => void) | undefined;
  const pending = new Promise<void>(resolve => { release = resolve; });
  const changes = { revision: 2 };
  let saved = false;
  await page.route('**/api/v1/teaching/**', async route => {
    const request = route.request(), path = new URL(request.url()).pathname;
    if (request.method() === 'GET' && path === '/api/v1/datasets') return route.fulfill({ json: [] });
    if (path === '/api/v1/teaching/state') return route.fulfill({ json: { connected: true, voice_configured: false, state: { mode: 'idle', episode_id: null, revision: changes.revision, session_id: 'session-advice', instruction: 'Review scene', outcome: 'unknown', steps: 0, sim_time: 0, joints: ['gripper'], state_rad: [0], fault: null } } });
    if (!path.includes('/intelligence/')) return route.fallback();
    if (request.method() !== 'GET') posts.push({ path, body: request.postData() ? request.postDataJSON() : undefined });
    if (path.endsWith('/status')) return route.fulfill({ json: { broker_reachable: true, configured: true, busy: false, configuration_revision: null, credential_source: 'environment' } });
    if (path.endsWith('/settings')) { if (request.method() === 'PUT') saved = true; if (request.method() === 'DELETE') saved = false; return route.fulfill({ json: { saved, revision: saved ? 'a'.repeat(32) : null, voice_model: saved ? 'provider/chat' : null, message: 'Saved locally; provider access is unverified.' } }); }
    if (path.endsWith('/cancel')) return route.fulfill({ json: { schema_version: 1, request_id: request.postDataJSON().request_id, status: 'cancellation_requested', billing_verified: false } });
    if (options.hold) await pending;
    if (options.fail) return route.fulfill({ status: 503, json: { detail: 'The provider deadline expired. Completion and billing remain unverified; no automatic retry was made.' } });
    const body = request.postDataJSON(), kind = path.endsWith('/decision') ? 'decision' : 'perception';
    return route.fulfill({ json: { schema_version: 1, request_id: body.request_id, kind, advisory_only: true, current_at_return: true, choices: [{ id: 'review_instruction', description: 'Clarify the recorded instruction.' }], proposal: { ...(kind === 'decision' ? { choice: 'review_instruction', confidence: .8 } : { summary: 'Generated camera fixture.', uncertain: true }), receipt: { context: { revision: body.expected_revision, session_id: body.session_id, episode_id: body.episode_id }, requested_model: kind === 'decision' ? 'typesafe/jev-1.13' : 'perceptron/perceptron-mk1.5', returned_model: kind === 'decision' ? 'typesafe/jev-1.13' : 'perceptron/perceptron-mk1.5', elapsed_seconds: .02, reported_cost_usd: null, frame: kind === 'perception' ? { step: 1, rgb_sha256: 'a'.repeat(64) } : null } } } });
  });
  return { posts, changes, release: () => release?.() };
}

async function consentAdvice(page: Page) {
  await page.getByLabel('Question for intelligence').fill('What should I review?');
  await page.getByRole('checkbox', { name: /I consent to sending/ }).check();
}

test('optional intelligence requires consent and never dispatches a control command', async ({ page }) => {
  const base = await teaching(page);
  const fixture = await intelligence(page);
  await page.reload(); await page.getByRole('link', { name: 'Teaching', exact: true }).click();
  const ask = page.getByRole('button', { name: 'Ask Jev for a review suggestion' });
  await expect(ask).toBeDisabled();
  await consentAdvice(page); await ask.dblclick();
  await expect(page.getByRole('article', { name: 'Intelligence suggestion' })).toContainText('Clarify the recorded instruction.');
  expect(fixture.posts.filter(p => p.path.endsWith('/decision'))).toHaveLength(1);
  expect(fixture.posts[0].body).toMatchObject({ session_id: 'session-advice', expected_revision: 2, episode_id: null, consent: true });
  expect(base.commands).toEqual([]);
  await expect(page.getByRole('checkbox', { name: /I consent to sending/ })).not.toBeChecked();
  fixture.changes.revision = 3;
  await expect(page.getByText('Historical suggestion · context changed')).toBeVisible();
});

test('camera advice displays captured scope and never sends browser pixels', async ({ page }) => {
  await teaching(page);const fixture = await intelligence(page);
  await page.reload(); await page.getByRole('link', { name: 'Teaching', exact: true }).click();
  await consentAdvice(page);await page.getByRole('button', { name: 'Ask Mk1.5 about the camera' }).click();
  await expect(page.getByRole('article', { name: 'Intelligence suggestion' })).toContainText('captured frame 1');
  expect(Object.keys(fixture.posts[0].body).sort()).toEqual(['consent', 'episode_id', 'expected_revision', 'prompt', 'request_id', 'session_id']);
  await expect(page.getByText(/not a live observation, calibrated confidence/)).toBeVisible();
});

test('private intelligence settings clear password without browser persistence or provider verification', async ({ page }) => {
  await teaching(page);const fixture = await intelligence(page);
  await page.reload(); await page.getByRole('link', { name: 'Teaching', exact: true }).click();
  await page.getByText('Configure optional services', { exact: true }).click();
  await page.getByLabel('OpenRouter API key').fill('fixture-private-key-not-real');
  await page.getByLabel('Voice chat model', { exact: true }).fill('provider/chat');
  await page.getByRole('button', { name: 'Save private settings' }).click();
  await expect(page.getByLabel('OpenRouter API key')).toHaveValue('');
  await expect(page.getByText('Saved voice model: provider/chat. Provider access is unverified.')).toBeVisible();
  expect(await page.evaluate(() => JSON.stringify({ ...localStorage, ...sessionStorage }))).not.toContain('fixture-private-key');
  expect(fixture.posts.map(p => p.path)).toEqual(['/api/v1/teaching/intelligence/settings']);
});

test('ambiguous intelligence failure never retries and requires renewed consent', async ({ page }) => {
  await teaching(page);const fixture = await intelligence(page, { fail: true });
  await page.reload(); await page.getByRole('link', { name: 'Teaching', exact: true }).click();
  await consentAdvice(page);await page.getByRole('button', { name: 'Ask Jev for a review suggestion' }).click();
  await expect(page.getByRole('alert').filter({ hasText: 'provider deadline expired' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Ask Jev for a review suggestion' })).toBeDisabled();
  expect(fixture.posts.filter(p => p.path.endsWith('/decision'))).toHaveLength(1);
  await expect(page.getByRole('article', { name: 'Intelligence suggestion' })).toHaveCount(0);
});

test('explicit intelligence cancel preserves request identity and withholds late response', async ({ page }) => {
  await teaching(page);const fixture = await intelligence(page, { hold: true });
  await page.reload(); await page.getByRole('link', { name: 'Teaching', exact: true }).click();
  await consentAdvice(page);await page.getByRole('button', { name: 'Ask Jev for a review suggestion' }).click();
  await expect.poll(() => fixture.posts.filter(p => p.path.endsWith('/decision')).length).toBe(1);
  await page.getByRole('button', { name: 'Cancel intelligence request' }).click();
  await expect.poll(() => fixture.posts.filter(p => p.path.endsWith('/cancel')).length).toBe(1);
  expect(fixture.posts[1].body).toEqual({ request_id: fixture.posts[0].body.request_id, session_id: 'session-advice' });
  fixture.release();
  await expect(page.getByRole('alert').filter({ hasText: 'Cancellation requested' })).toBeVisible();
  await expect(page.getByRole('article', { name: 'Intelligence suggestion' })).toHaveCount(0);
});

test('revision change cancels advice and cannot promote a late response', async ({ page }) => {
  await teaching(page);const fixture = await intelligence(page, { hold: true });
  await page.reload(); await page.getByRole('link', { name: 'Teaching', exact: true }).click();
  await consentAdvice(page);await page.getByRole('button', { name: 'Ask Jev for a review suggestion' }).click();
  await expect.poll(() => fixture.posts.length).toBe(1);
  fixture.changes.revision = 3;
  await expect.poll(() => fixture.posts.filter(p => p.path.endsWith('/cancel')).length).toBe(1);
  fixture.release();
  await expect(page.getByRole('alert').filter({ hasText: 'Teaching context changed' })).toBeVisible();
  await expect(page.getByRole('article', { name: 'Intelligence suggestion' })).toHaveCount(0);
});

for (const width of [1440, 390, 320]) {
  test(`teaching fields keep labels above usable controls and consent on its own row at ${width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 1000 });
    const { commands, unexpected } = await teaching(page, false);
    await page.getByText('Configure optional services', { exact: true }).click();
    const controls = page.getByRole('region', { name: 'Teaching controls', exact: true });
    const geometry = await controls.evaluate(section => {
      const rect = (element: Element) => {
        const value = element.getBoundingClientRect();
        return { top: value.top, bottom: value.bottom, left: value.left, right: value.right, width: value.width, height: value.height };
      };
      const field = (id: string) => {
        const input = section.querySelector(`#${id}`)!;
        return { input: rect(input), label: rect(section.querySelector(`label[for="${id}"]`)!), field: rect(input.parentElement!) };
      };
      const question = section.querySelector('#teaching-advice-question')!;
      const consent = section.querySelector('.teaching-advice-form .workbench-check')!;
      return {
        panel: rect(section), advice: rect(section.querySelector('.teaching-advice')!),
        task: field('teaching-task'), joint: field('teaching-joint'), question: field('teaching-advice-question'),
        key: field('teaching-openrouter-key'), model: field('teaching-voice-model'),
        setInstruction: rect(section.querySelector('.teaching-task-form button')!),
        consent: rect(consent), checkbox: rect(consent.querySelector('input')!), consentText: rect(consent.querySelector('span')!),
        questionValue: (question as HTMLInputElement).value,
      };
    });
    for (const field of [geometry.task, geometry.joint, geometry.question, geometry.key, geometry.model]) {
      expect(field.input.top - field.label.bottom).toBeGreaterThanOrEqual(6);
      expect(field.input.width).toBeGreaterThanOrEqual(field.field.width - 2);
      expect(field.input.width).toBeGreaterThan(140);
      expect(field.input.height).toBeGreaterThanOrEqual(44);
    }
    expect(geometry.task.input.width).toBeGreaterThan(geometry.panel.width * .6);
    expect(geometry.question.input.width).toBeGreaterThan(geometry.advice.width * .65);
    expect(geometry.consent.top - geometry.question.input.bottom).toBeGreaterThanOrEqual(12);
    expect(geometry.consentText.left - geometry.checkbox.right).toBeGreaterThanOrEqual(8);
    expect(geometry.checkbox.width).toBeLessThanOrEqual(20);
    if (width > 700) {
      expect(geometry.setInstruction.left - geometry.task.input.right).toBeGreaterThanOrEqual(10);
      expect(geometry.model.input.left - geometry.key.input.right).toBeGreaterThanOrEqual(16);
    } else {
      expect(geometry.setInstruction.top - geometry.task.input.bottom).toBeGreaterThanOrEqual(10);
      expect(geometry.model.label.top - geometry.key.input.bottom).toBeGreaterThanOrEqual(16);
    }
    await expect(page.getByRole('checkbox', { name: /I consent to sending/ })).not.toBeChecked();
    await expect(page.getByRole('button', { name: 'Connect voice', exact: true })).toBeDisabled();
    expect(commands).toEqual([]);
    expect(unexpected).toEqual([]);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width + 1);
    const path = testInfo.outputPath(`teaching-form-${width}.png`);
    await controls.screenshot({ path });
    await testInfo.attach(`Teaching form ${width}px`, { path, contentType: 'image/png' });
  });
}
