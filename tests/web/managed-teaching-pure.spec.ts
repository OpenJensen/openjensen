import { expect, test } from '@playwright/test';
import type { Job } from '../../apps/web/src/lib/api';
import {
  managedTeachingHistory, managedTeachingJob, managedTeachingOptions, managedTeachingRequest,
  managedTeachingStatus, managedTeachingTransport, publishedCapture, readManagedOptions,
  requireManagedProfile, retainedManagedTeachingReceipt, stopManagedSession,
} from '../../apps/web/src/lib/managed-teaching';

const recipe = () => ({ operation: 'teaching.capture' as const, profile_id: 'isaac-local', profile_sha256: 'a'.repeat(64), timeout_seconds: 300 });
const profile = () => ({ id: 'isaac-local', label: 'Configured local Isaac', profile_sha256: 'a'.repeat(64), max_seconds: 300, max_capture_bytes: 1048576, runtime_verified: false, transport: 'local_owned_process' });
const options = () => ({ configured: true, available: true, profiles: [profile()], message: 'Runtime readiness is checked after explicit start.' });
const result = () => ({ schema_version: 1, operation: 'teaching.capture', profile_id: 'isaac-local', profile_sha256: 'a'.repeat(64), session_id: 'b'.repeat(32), session_sha256: 'c'.repeat(64), inventory_sha256: 'd'.repeat(64), recording_configuration_sha256: 'e'.repeat(64),
  episodes: [{ episode_id: '1'.repeat(32), receipt_sha256: '2'.repeat(64), frames: 3, termination: 'finish', outcome: 'unknown' }], origin: 'recorded', lineage_group: 'session-a', published: true, content_verified: true, process_cleanup_verified: true, simulator_coordinates: true, physical_calibration_verified: false, task_success_claimed: false });
const job = (status = 'running') => ({ id: 'job-a', project_id: 'project-a', kind: 'teaching.capture', status, request: recipe(), result: status === 'succeeded' ? result() : null, stage: 'running', error: null, created_at: '2026-09-28T00:00:00Z', updated_at: '2026-09-28T00:00:01Z', compute_target: null, simulation_target: null }) as Job;
const status = () => ({ job: job(), ready: true, session_id: 'b'.repeat(32), stop_requested: false });
const state = () => ({ mode: 'running', episode_id: '1'.repeat(32), revision: 4, session_id: 'b'.repeat(32), instruction: 'Teach one motion', outcome: 'unknown', steps: 3, sim_time: .15, joints: ['joint'], state_rad: [.1], fault: null });
const fetchOriginal = globalThis.fetch;
test.afterEach(() => { globalThis.fetch = fetchOriginal; });
function reply(value: unknown) { return new Response(JSON.stringify(value), { headers: { 'content-type': 'application/json' } }); }

test('configured but unavailable local profiles never admit a start', () => {
  const mac = managedTeachingOptions({ ...options(), available: false });
  expect(mac.configured).toBe(true); expect(mac.profiles[0].runtime_verified).toBe(false);
  expect(() => requireManagedProfile(mac, recipe())).toThrow(/unavailable/);
  const current = managedTeachingOptions(options()); requireManagedProfile(current, recipe());
  expect(() => requireManagedProfile(current, { ...recipe(), timeout_seconds: 301 })).toThrow();
  expect(() => requireManagedProfile(current, { ...recipe(), profile_sha256: 'f'.repeat(64) })).toThrow();
});
for (const field of ['available', 'configured', 'max_seconds', 'max_capture_bytes', 'runtime_verified', 'transport', 'profile_sha256']) {
  test(`malformed teaching profile ${field} fails closed`, () => {
    const value = options();
    if (field === 'available' || field === 'configured') (value as Record<string, unknown>)[field] = 'true';
    else (value.profiles[0] as Record<string, unknown>)[field] = field === 'runtime_verified' ? true : field === 'transport' ? 'remote' : false;
    expect(() => managedTeachingOptions(value)).toThrow();
  });
}
test('requests carry only exact reviewed profile and bounded time, never private paths', () => {
  for (const change of [{ timeout_seconds: true }, { timeout_seconds: 0 }, { timeout_seconds: 3601 }, { timeout_seconds: 1.5 }, { profile_sha256: null }, { profile_id: '../profile' }, { operation: 'dataset.inspect' }, { path: '/private/config' }]) expect(() => managedTeachingRequest({ ...recipe(), ...change })).toThrow();
  expect(managedTeachingRequest(recipe())).toEqual(recipe());
});
test('teaching acknowledgements bind project, job, original request and actual timestamps', () => {
  expect(managedTeachingJob(job(), 'project-a', recipe(), 'job-a').id).toBe('job-a');
  for (const change of [{ project_id: 'project-b' }, { id: 'job-b' }, { kind: 'policy.finetune' }, { status: ['running'] }, { request: { ...recipe(), timeout_seconds: 299 } }, { updated_at: '2020-01-01T00:00:00Z' }]) expect(() => managedTeachingJob({ ...job(), ...change }, 'project-a', recipe(), 'job-a')).toThrow();
});
test('restart-interrupted sessions remain visible terminal history without control readiness', () => {
  const interrupted = job('interrupted');
  expect(managedTeachingHistory([interrupted], 'project-a')).toEqual([interrupted]);
  expect(managedTeachingStatus({ job: interrupted, ready: false, session_id: null, stop_requested: false }, 'project-a', interrupted.id).job.status).toBe('interrupted');
  expect(() => managedTeachingStatus({ job: interrupted, ready: true, session_id: 'b'.repeat(32), stop_requested: false }, 'project-a', interrupted.id)).toThrow();
});
test('a retained succeeded acknowledgement keeps its identity without proving publication', () => {
  const minimal = { ...job('succeeded'), result: null, stage: null, error: null };
  expect(retainedManagedTeachingReceipt(minimal, 'project-a').id).toBe('job-a');
  expect(() => publishedCapture(minimal)).toThrow();
  expect(() => managedTeachingJob(minimal, 'project-a')).toThrow();
  expect(() => retainedManagedTeachingReceipt(minimal, 'project-b')).toThrow();
});
test('transport identity cannot collide across slash-containing project and job names', () => {
  const one = managedTeachingTransport('project/a', 'job', 'b'.repeat(32), recipe());
  const two = managedTeachingTransport('project', 'a/job', 'b'.repeat(32), recipe());
  expect(one.identity).not.toBe(two.identity);
  expect(one.frameUrl).toContain('/projects/project%2Fa/teaching/sessions/job/frame');
});
test('ready requires the exact running session and rejects stop or terminal contradictions', () => {
  expect(managedTeachingStatus(status(), 'project-a', 'job-a').ready).toBe(true);
  for (const change of [{ ready: 1 }, { session_id: null }, { session_id: 'bad' }, { stop_requested: true }, { job: job('queued') }, { job: job('cancelled') }, { job: job('succeeded') }]) expect(() => managedTeachingStatus({ ...status(), ...change }, 'project-a', 'job-a')).toThrow();
});
test('published capture preserves exact provenance without manufacturing calibration or quality', () => {
  const value = managedTeachingJob(job('succeeded'), 'project-a');
  const capture = publishedCapture(value);
  expect(capture).toMatchObject({ project_id: 'project-a', job_id: 'job-a', session_id: 'b'.repeat(32), inventory_sha256: 'd'.repeat(64), recording_configuration_sha256: 'e'.repeat(64), origin: 'recorded', lineage_group: 'session-a', episodes: result().episodes });
  capture.episodes.pop(); expect((value.result as ReturnType<typeof result>).episodes).toHaveLength(1);
  expect(() => publishedCapture(job('running'))).toThrow();
});
for (const field of ['profile_sha256', 'session_id', 'session_sha256', 'inventory_sha256', 'recording_configuration_sha256', 'published', 'content_verified', 'process_cleanup_verified', 'simulator_coordinates', 'physical_calibration_verified', 'task_success_claimed']) {
  test(`publication rejects unsupported ${field} claim`, () => {
    const saved = job('succeeded'), changed = result();
    (changed as Record<string, unknown>)[field] = field === 'physical_calibration_verified' || field === 'task_success_claimed' ? true : false;
    expect(() => managedTeachingJob({ ...saved, result: changed }, 'project-a')).toThrow();
  });
}
test('duplicate or malformed episode receipts cannot become dataset handoffs', () => {
  for (const episodes of [[], [result().episodes[0], result().episodes[0]], [{ ...result().episodes[0], frames: true }], [{ ...result().episodes[0], receipt_sha256: null }], [{ ...result().episodes[0], outcome: ['unknown'] }]]) expect(() => managedTeachingJob({ ...job('succeeded'), result: { ...result(), episodes } }, 'project-a')).toThrow();
});
test('history refuses mixed project identity and never substitutes for an explicitly requested job', () => {
  expect(() => managedTeachingHistory([job(), { ...job(), id: 'other', project_id: 'project-b' }], 'project-a')).toThrow();
  expect(() => managedTeachingHistory([job(), job()], 'project-a')).toThrow();
  expect(() => managedTeachingStatus(status(), 'project-a', 'missing')).toThrow();
});
test('managed reads and commands use only the exact owned endpoint and final navigation fence', async () => {
  const requests: { url: string; method: string; body: unknown }[] = [];
  globalThis.fetch = async (input, init) => {
    const url = String(input), body = init?.body ? JSON.parse(String(init.body)) : undefined;
    requests.push({ url, method: init?.method ?? 'GET', body });
    if (url.endsWith('/profiles')) return reply(options());
    if (url.endsWith('/sessions/job-a')) return reply(status());
    if (url.endsWith('/state')) return reply(state());
    if (url.endsWith('/commands')) return reply({ command_id: body.command_id, status: 'queued' });
    throw new Error('Unexpected endpoint');
  };
  await readManagedOptions('project-a');
  const transport = managedTeachingTransport('project-a', 'job-a', 'b'.repeat(32), recipe());
  expect(transport.frameUrl).toContain('/api/v1/projects/project-a/teaching/sessions/job-a/frame');
  const command = { session_id: 'b'.repeat(32), command_id: 'command-a', operation: 'pause', expected_revision: 4, episode_id: '1'.repeat(32), arguments: {} };
  await expect(transport.request('/commands', command, () => { throw new Error('Navigation changed'); })).rejects.toThrow('Navigation changed');
  expect(requests.filter(r => r.method === 'POST')).toHaveLength(0);
  await transport.request('/commands', command);
  expect(requests.filter(r => r.method === 'POST')).toEqual([{ url: '/api/v1/projects/project-a/teaching/sessions/job-a/commands', method: 'POST', body: command }]);
  expect(requests.every(r => r.url.startsWith('/api/v1/projects/project-a/teaching/'))).toBe(true);
});
test('managed transport refuses global voice and intelligence before any network request', async () => {
  globalThis.fetch = async () => { throw new Error('Unexpected network'); };
  const transport = managedTeachingTransport('project-a', 'job-a', 'b'.repeat(32), recipe());
  for (const path of ['/voice/status', '/voice/join', '/intelligence/status', '/frame', '//other']) await expect(transport.request(path)).rejects.toThrow(/unavailable/);
});
test('changed executor session or malformed fresh state prevents command submission', async () => {
  let posts = 0;
  globalThis.fetch = async (input, init) => {
    if (init?.method === 'POST') posts += 1;
    return reply(String(input).endsWith('/state') ? { ...state(), session_id: 'c'.repeat(32) } : status());
  };
  await expect(managedTeachingTransport('project-a', 'job-a', 'b'.repeat(32), recipe()).request('/commands', { command_id: 'one', session_id: 'b'.repeat(32) })).rejects.toThrow();
  expect(posts).toBe(0);
});
test('stop acknowledgement is not publication, and ambiguity never automatically retries', async () => {
  let posts = 0;
  globalThis.fetch = async () => { posts += 1; return reply({ ...status(), ready: false, stop_requested: true }); };
  const stopping = await stopManagedSession('project-a', 'job-a', recipe());
  expect(stopping.job.status).toBe('running'); expect(() => publishedCapture(stopping.job)).toThrow(); expect(posts).toBe(1);
  globalThis.fetch = async () => { posts += 1; throw new Error('Lost response'); };
  await expect(stopManagedSession('project-a', 'job-a', recipe())).rejects.toThrow(/unverified/); expect(posts).toBe(2);
  globalThis.fetch = async () => reply(status());
  await expect(stopManagedSession('project-a', 'job-a', recipe())).rejects.toThrow(/did not confirm/);
});
