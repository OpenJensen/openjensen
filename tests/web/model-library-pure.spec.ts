import { expect, test } from '@playwright/test';
import type { Job, PolicyArtifact } from '../../apps/web/src/lib/api';
import { episodePartitions, modelActionIssue, modelLineage, ownedModels } from '../../apps/web/src/lib/model-library';

const model = (id: string, parents: string[] = [], extra: Partial<PolicyArtifact> = {}) => ({ id, project_id: 'alpha', job_id: `run-${id}`, label: id, format: 'native_checkpoint', parent_ids: parents, metadata: { architecture: 'act' }, ...extra }) as PolicyArtifact;
const job = (id: string, request: Record<string, unknown>) => ({ id, project_id: 'alpha', kind: 'policy.finetune', request }) as Job;
test('lineage follows all recorded parents and resumed runs and preserves missing identities', () => {
  const root = model('root'), student = model('student', ['root']), packed = model('packed', ['student', 'missing']);
  const jobs = [job('run-root', { resume_job_id: 'resume-2' }), job('resume-2', { resume_job_id: 'resume-1' }), job('resume-1', {}), job('run-student', { artifact_id: 'root' })];
  expect(modelLineage(packed, [root, student, packed], jobs).map(step => step.id)).toEqual(['job:resume-1', 'job:resume-2', 'root', 'student', 'missing', 'packed']);
  expect(modelLineage(packed, [root, student, packed], jobs).find(step => step.id === 'missing')?.missing).toContain('Parent model unavailable');
});
test('foreign models and execution records cannot enter the collection or satisfy lineage', () => {
  const packed = model('packed', ['foreign']), foreign = model('foreign', [], { project_id: 'beta' });
  expect(ownedModels([packed, packed, foreign, model('record', [], { format: 'native_run_record' })], 'alpha').map(item => item.id)).toEqual(['packed']);
  expect(modelLineage(packed, [packed, foreign], []).at(0)?.missing).toContain('foreign');
});
test('cyclic artifact and resume records are bounded and identified', () => {
  const a = model('a', ['b']), b = model('b', ['a']);
  const result = modelLineage(a, [a, b], [job('run-a', { resume_job_id: 'resume' }), job('resume', { resume_job_id: 'resume' })]);
  expect(result).toHaveLength(5);
  expect(result.filter(step => step.missing?.startsWith('Circular'))).toHaveLength(2);
});
test('episode coverage is derived only from valid recorded membership and known counts', () => {
  expect(episodePartitions({ train: [0, 2], validation: [1], final: [3] }, 6).outside).toEqual([4, 5]);
  expect(episodePartitions({ train: [0, 2], validation: [1] }, undefined).outside).toBeNull();
  expect(episodePartitions({ train: [0, 20], validation: [1] }, 6).outside).toBeNull();
  expect(episodePartitions({ validation_fraction: .2 }, 6).partitions).toEqual([]);
});
test('actions depend on the concrete format and provenance of a saved model', () => {
  expect(modelActionIssue(model('teacher'), 'distill')).toBeNull();
  expect(modelActionIssue(model('training', [], { format: 'training_checkpoint' }), 'distill')).toContain('Export');
  expect(modelActionIssue(model('remote', [], { metadata: { architecture: 'act', storage: 'gcs' } }), 'distill')).not.toBeNull();
  expect(modelActionIssue(model('other', [], { metadata: { architecture: 'pi0' } }), 'quantize')).not.toBeNull();
});
