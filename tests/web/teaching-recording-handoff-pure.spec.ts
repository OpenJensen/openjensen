import { expect, test } from '@playwright/test';
import type { ManagedPublishedCapture } from '../../apps/web/src/lib/managed-teaching';
import type { RecordingCatalog } from '../../apps/web/src/lib/recording-preparation';
import { teachingRecordingRecipe } from '../../apps/web/src/lib/teaching-recording-handoff';

const episodes = [1, 2].map(n => ({ episode_id: String(n).repeat(32), receipt_sha256: String(n).repeat(64), frames: 24, termination: 'finish' as const, outcome: 'unknown' as const }));
const capture = (): ManagedPublishedCapture => ({ project_id: 'project-a', job_id: 'teaching-job', profile_id: 'local-isaac', profile_sha256: 'a'.repeat(64),
  session_id: 'b'.repeat(32), session_sha256: 'c'.repeat(64), inventory_sha256: 'd'.repeat(64), recording_configuration_sha256: 'e'.repeat(64),
  episodes: structuredClone(episodes), origin: 'recorded', lineage_group: 'one-real-session' });
const catalog = (): RecordingCatalog => ({ configuration_sha256: 'e'.repeat(64), message: 'Finalized metadata; contents checked during preparation.', captures: [{
  session_id: 'b'.repeat(32), session_sha256: 'c'.repeat(64), origin: 'recorded', lineage_group: 'one-real-session', controller: 'joint_position_targets',
  state_units: 'radians', action_units: 'radians', timebase: 'simulation_seconds', camera_key: 'observation.images.front', joint_names: ['joint_one'],
  width: 64, height: 64, fps: 20, physics_hz: 60, scene_sha256: 'f'.repeat(64), camera_prim: '/World/Camera', episodes: structuredClone(episodes), content_verified: false }] });

test('explicit teaching selection uses exact published session and episodes without changing input or declaring training quality', () => {
  const source = capture(), current = catalog(), before = JSON.stringify([source, current]);
  const recipe = teachingRecordingRecipe(source, 'project-a', current, 600);
  expect(recipe).toEqual({ schema_version: 1, configuration_sha256: 'e'.repeat(64), timeout_seconds: 600,
    captures: [{ session_id: source.session_id, session_sha256: source.session_sha256, episodes: episodes.map(({ episode_id, receipt_sha256 }) => ({ episode_id, receipt_sha256 })) }] });
  expect(JSON.stringify([source, current])).toBe(before);
  recipe.captures[0].episodes.pop(); expect(source.episodes).toHaveLength(2); expect(current.captures[0].episodes).toHaveLength(2);
});

for (const field of ['project_id', 'job_id', 'profile_sha256', 'inventory_sha256', 'session_id', 'session_sha256', 'recording_configuration_sha256', 'origin', 'lineage_group'] as const) {
  test(`a changed teaching ${field} cannot substitute other recordings`, () => {
    const source = capture(); (source as unknown as Record<string, unknown>)[field] = field === 'project_id' ? 'project-b' : '';
    expect(() => teachingRecordingRecipe(source, 'project-a', catalog(), 600)).toThrow();
  });
}
for (const field of ['episode_id', 'receipt_sha256', 'frames', 'termination', 'outcome'] as const) {
  test(`changed finalized episode ${field} refuses the handoff`, () => {
    const current = catalog(); (current.captures[0].episodes[0] as unknown as Record<string, unknown>)[field] = field === 'frames' ? 25 : field === 'termination' ? 'reset' : field === 'outcome' ? 'operator_reported_failure' : '3'.repeat(field === 'episode_id' ? 32 : 64);
    expect(() => teachingRecordingRecipe(capture(), 'project-a', current, 600)).toThrow();
  });
}
test('missing, extra or duplicate episodes and absent sessions fail without falling back to another capture', () => {
  for (const change of ['missing', 'extra', 'duplicate', 'session'] as const) {
    const source = capture(), current = catalog();
    if (change === 'missing') current.captures[0].episodes.pop();
    if (change === 'extra') current.captures[0].episodes.push({ ...episodes[0], episode_id: '3'.repeat(32) });
    if (change === 'duplicate') source.episodes[1] = { ...source.episodes[0] };
    if (change === 'session') current.captures[0].session_id = '4'.repeat(32);
    expect(() => teachingRecordingRecipe(source, 'project-a', current, 600)).toThrow();
  }
});
test('episode ordering does not invent new lineage and invalid preparation limits remain rejected', () => {
  const source = capture(); source.episodes.reverse();
  expect(teachingRecordingRecipe(source, 'project-a', catalog(), 600).captures[0].episodes.map(row => row.episode_id)).toEqual(episodes.map(row => row.episode_id));
  for (const limit of [0, 59, 1801, 60.1, NaN]) expect(() => teachingRecordingRecipe(source, 'project-a', catalog(), limit)).toThrow();
});
