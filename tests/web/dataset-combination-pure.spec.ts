import { expect, test } from '@playwright/test';
import type { DatasetProfile } from '../../apps/web/src/lib/api';
import { combinationCameraMapping, datasetCombinationIssue } from '../../apps/web/src/lib/dataset-combination';

const camera = 'observation.images.front';
const profile = (repo = 'our/primary'): DatasetProfile => ({ source: 'huggingface', repo_id: repo, revision: 'a'.repeat(40), format: 'lerobot_v3', fps: 30,
  features: { action: { dtype: 'float32', shape: [2], names: ['joint', 'gripper'] }, 'observation.state': { dtype: 'float32', shape: [2] }, [camera]: { dtype: 'video', shape: [480, 640, 3] } },
} as unknown as DatasetProfile);

test('matching pinned datasets combine, but immutable identity, format and timebase must match', () => {
  const primary = profile(), other = profile('our/second');
  expect(datasetCombinationIssue(primary, other, [camera])).toBeNull();
  const unnamed = profile('our/unnamed');
  delete primary.features.action.names;
  unnamed.features.action.names = null;
  expect(datasetCombinationIssue(primary, unnamed, [camera])).toBeNull();
  expect(datasetCombinationIssue(primary, primary, [camera])).toContain('already selected');
  expect(datasetCombinationIssue(primary, { ...other, source: 'local' }, [camera])).toContain('separately');
  expect(datasetCombinationIssue(primary, { ...other, format: 'lerobot_v2' }, [camera])).toContain('LeRobot v3');
  expect(datasetCombinationIssue(primary, { ...other, revision: 'main' }, [camera])).toContain('immutable');
  expect(datasetCombinationIssue(primary, { ...other, fps: 10 }, [camera])).toContain('10 FPS here, 30 FPS');
});

for (const key of ['action', 'observation.state']) test(`${key} dimensions, data types and ordered names prevent incompatible selection`, () => {
  const primary = profile(), other = profile('our/second');
  for (const [patch, message] of [[{ shape: [3] }, 'dimensions'], [{ dtype: 'float64' }, 'data types'], [{ names: ['gripper', 'joint'] }, 'field names']] as const) {
    const candidate = { ...other, features: { ...other.features, [key]: { ...other.features[key] as object, ...patch } } };
    expect(datasetCombinationIssue(primary, candidate, [camera])).toContain(message);
  }
});

test('camera aliases pair distinct matching views and never reuse a camera or accept wrong dimensions', () => {
  const primary = profile(), other = profile('our/second'), wrist = 'observation.images.wrist';
  const renamed = { ...other, features: { ...other.features, [wrist]: other.features[camera] } };
  delete renamed.features[camera];
  expect(combinationCameraMapping(primary, renamed, [camera])).toEqual({ [camera]: wrist });
  expect(datasetCombinationIssue(primary, renamed, [camera])).toBeNull();
  const twoViews = { ...primary, features: { ...primary.features, [wrist]: primary.features[camera] } };
  expect(datasetCombinationIssue(twoViews, renamed, [camera, wrist])).toContain('2 distinct matching cameras');
  expect(datasetCombinationIssue(primary, other, [], {})).toContain('at least one camera');
  expect(datasetCombinationIssue(primary, other, [camera], { [camera]: 'missing' })).toContain('Camera dimensions');
  expect(datasetCombinationIssue(twoViews, { ...other, features: { ...other.features, [wrist]: other.features[camera] } }, [camera, wrist], { [camera]: camera, [wrist]: camera })).toContain('distinct');
});

test('unordered feature-name object keys remain compatible, while their ordered arrays remain exact', () => {
  const primary = profile(), other = profile('our/second');
  primary.features.action = { dtype: 'float32', shape: [2], names: { joints: ['a', 'b'], units: ['radians'] } };
  other.features.action = { dtype: 'float32', shape: [2], names: { units: ['radians'], joints: ['a', 'b'] } };
  expect(datasetCombinationIssue(primary, other, [camera])).toBeNull();
});
