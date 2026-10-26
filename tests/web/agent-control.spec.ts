import { expect, test, type Page } from '@playwright/test';
import { fulfillApplicationUpdateRead } from './application-update-controls';
import { installedModules } from './module-fixtures';

// These fixtures verify browser authority and evidence boundaries without model or motor calls.
async function fixture(page: Page) {
  const requests: { path: string; body: Record<string, unknown> }[] = [];
  const stamp = new Date().toISOString();
  const observation = {
    id: 'obs-fixture',
    sha256: 'a'.repeat(64),
    target_session_id: 'replay-fixture',
    episode_id: null,
    authority_epoch: 1,
    revision: 0,
    step: 0,
    captured_at: stamp,
    expires_at: new Date(Date.now() + 120000).toISOString(),
    clock_policy: 'replay',
    environment_kind: 'replay',
    embodiment_manifest_id: 'fixture-arm',
    scene_id: 'fixture',
    joints: { shoulder: 0 },
    units: 'radians',
    image_sha256: null,
    source: 'synthetic_fixture',
  };
  const saved = {
    id: 'agent-fixture',
    project_id: 'agent-project',
    provider_id: 'mock',
    provider_revision: 'mock-v1',
    task: 'Move shoulder by 0.03 radians',
    target_kind: 'replay',
    teaching_job_id: null,
    mode: 'supervised',
    clock_policy: 'replay',
    status: 'ready',
    authority_epoch: 1,
    created_at: stamp,
    updated_at: stamp,
    observation,
    pending_proposal: null as Record<string, unknown> | null,
    receipts: [] as Record<string, unknown>[],
    calls_used: 0,
    input_tokens: 0,
    output_tokens: 0,
    max_calls: 4,
    max_seconds: 120,
    max_output_tokens: 1024,
    max_total_tokens: 16000,
    message: 'Synthetic replay fixture; no motion',
    task_success_verified: false,
  };
  let sessions: (typeof saved)[] = [];
  let profiles = [
    {
      id: 'mock',
      provider: 'mock',
      model: 'deterministic-mock',
      reasoning_effort: 'low',
      revision: 'mock-v1',
      saved: false,
      checked: true,
      capabilities: ['named_joint_correction'],
      message: 'Built-in replay provider',
    },
  ];
  await page.route('**/api/v1/**', async (route) => {
    if (await fulfillApplicationUpdateRead(route)) return;
    const request = route.request(),
      path = new URL(request.url()).pathname.replace('/api/v1', '');
    if (request.method() !== 'GET') requests.push({ path, body: request.postData() ? request.postDataJSON() : {} });
    if (path === '/agent/providers') return route.fulfill({ json: profiles });
    if (path === '/agent/providers/openai' && request.method() === 'PUT') {
      profiles = [
        ...profiles.filter((p) => p.id !== 'openai'),
        {
          id: 'openai',
          provider: 'openai_responses',
          model: 'gpt-6-astra',
          reasoning_effort: 'medium',
          revision: 'profile-v1',
          saved: true,
          checked: false,
          capabilities: [],
          message: 'Saved; access unverified',
        },
      ];
      return route.fulfill({ json: profiles.at(-1) });
    }
    if (path === '/projects/agent-project/agent-sessions') {
      if (request.method() === 'POST') sessions = [saved];
      return route.fulfill({ json: request.method() === 'POST' ? saved : sessions });
    }
    if (path === '/agent-sessions/agent-fixture/proposals') {
      saved.calls_used += 1;
      saved.status = 'waiting_for_acceptance';
      saved.pending_proposal = {
        id: 'proposal-exact',
        request_id: request.postDataJSON().request_id,
        agent_session_id: saved.id,
        observation_id: observation.id,
        observation_sha256: observation.sha256,
        authority_epoch: 1,
        expected_revision: 0,
        provider_revision: 'mock-v1',
        model: 'deterministic-mock',
        skill: 'joint_correction',
        arguments: { joint: 'shoulder', delta_rad: 0.03 },
        status: 'proposed',
        created_at: stamp,
        expires_at: observation.expires_at,
      };
      return route.fulfill({ status: 202, json: saved });
    }
    if (path === '/agent-sessions/agent-fixture/approve') {
      saved.status = 'ready';
      saved.receipts = [
        {
          id: 'receipt-fixture',
          proposal_id: 'proposal-exact',
          status: 'replayed',
          arguments: { joint: 'shoulder', delta_rad: 0.03 },
          applied_targets: {},
          pre_observation_id: observation.id,
          post_observation_id: null,
          elapsed_seconds: 0,
          reason: 'Recorded proposal replay; no actuator dispatch',
          task_success_verified: false,
        },
      ];
      saved.pending_proposal = null;
      return route.fulfill({ json: saved });
    }
    if (path === '/agent-sessions/agent-fixture') return route.fulfill({ json: saved });
    if (path === '/agent-sessions/agent-fixture/events') return route.fulfill({ json: [] });
    const replies: Record<string, unknown> = {
      '/health': { status: 'ok', version: 'test' },
      '/capabilities': [],
      '/modules': installedModules,
      '/projects': [{ id: 'agent-project', name: 'Agent project', created_at: stamp }],
      '/projects/agent-project/jobs': [],
      '/datasets': [],
      '/teaching/state': { connected: false, state: null, message: 'Fixture disconnected', voice_configured: false },
      '/teaching/voice/status': {
        broker_reachable: false,
        configuration_present: false,
        dependencies_present: false,
        message: 'Not configured',
      },
      '/teaching/intelligence/status': { broker_reachable: false, configured: false, busy: false },
      '/teaching/intelligence/settings': { saved: false, revision: null, voice_model: null, message: 'Not saved' },
      '/projects/agent-project/recordings/options': {
        configured: false,
        runtime_verified: false,
        configuration_sha256: null,
        max_episodes: 100,
        max_source_bytes: 8589934592,
        setup_message: 'Not configured',
      },
    };
    return route.fulfill(
      path in replies
        ? { json: replies[path] }
        : { status: 404, json: { detail: 'Not configured in this browser fixture' } },
    );
  });
  return { requests, saved };
}

async function startReplay(page: Page) {
  await page.goto('/teaching/');
  await page.getByText('Agent Control', { exact: true }).click();
  await page.getByLabel('Task goal', { exact: true }).fill('Move shoulder by 0.03 radians');
  await page.getByRole('button', { name: 'Create agent session', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Agent session evidence' })).toBeVisible();
  await page.getByRole('button', { name: 'Plan one correction', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Pending agent proposal' })).toContainText('shoulder: +0.03 rad');
}

test('replay requires acceptance of the exact proposal and labels execution evidence truthfully', async ({ page }) => {
  const { requests } = await fixture(page);
  await startReplay(page);
  await expect(page.getByRole('button', { name: 'Accept this proposal', exact: true })).toBeEnabled();
  expect(requests.filter((request) => request.path.endsWith('/approve'))).toEqual([]);
  await page.getByRole('button', { name: 'Accept this proposal', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Latest execution receipt' })).toContainText('Replay receipt');
  await expect(page.getByRole('region', { name: 'Latest execution receipt' })).toContainText(
    'task success remains unverified',
  );
  expect(requests.find((request) => request.path.endsWith('/approve'))?.body).toEqual({
    proposal_id: 'proposal-exact',
    authority_epoch: 1,
  });
  expect(requests.filter((request) => request.path.endsWith('/proposals'))).toHaveLength(1);
  const dimensions = await page.evaluate(() => [
    document.documentElement.scrollWidth,
    document.documentElement.clientWidth,
  ]);
  expect(dimensions[0]).toBeLessThanOrEqual(dimensions[1] + 1);
});

test('a stale controller epoch disables acceptance even if the old proposal remains visible', async ({ page }) => {
  const { requests, saved } = await fixture(page);
  await startReplay(page);
  saved.authority_epoch = 2;
  saved.status = 'paused';
  await page.getByRole('button', { name: 'Refresh agent sessions', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Accept this proposal', exact: true })).toBeDisabled();
  await expect(page.getByRole('region', { name: 'Pending agent proposal' })).toContainText('Expired or superseded');
  expect(requests.filter((request) => request.path.endsWith('/approve'))).toEqual([]);
});

test('provider save clears the password and never implies a capability check or stores the key in browser storage', async ({
  page,
}) => {
  const { requests } = await fixture(page);
  await page.goto('/settings/');
  await page.getByRole('button', { name: 'AI providers', exact: true }).click();
  const panel = page.getByRole('region', { name: 'Agent provider settings', exact: true });
  const key = panel.getByLabel('OpenAI API key', { exact: true });
  const fakeKey = 'sk-BrowserFixtureOnlyNotARealKey12345678';
  await expect(key).toHaveAttribute('type', 'password');
  await key.fill(fakeKey);
  await panel.getByRole('button', { name: 'Save provider profile', exact: true }).click();
  await expect(key).toHaveValue('');
  await expect(panel.getByRole('region', { name: 'Provider openai', exact: true })).toContainText(
    'Capabilities unverified',
  );
  await expect(panel.getByRole('button', { name: 'Check openai', exact: true })).toBeDisabled();
  expect(requests.filter((request) => request.path.endsWith('/check'))).toEqual([]);
  expect(requests.find((request) => request.path === '/agent/providers/openai')?.body.api_key).toBe(fakeKey);
  expect(
    await page.evaluate(() => JSON.stringify({ local: { ...localStorage }, session: { ...sessionStorage } })),
  ).not.toContain(fakeKey);
  await expect(panel).not.toContainText(fakeKey);
});
