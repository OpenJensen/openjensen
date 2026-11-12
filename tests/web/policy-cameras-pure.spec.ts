import { expect, test } from '@playwright/test';
import { policyCameras, sameCameras, viewName } from '../../apps/web/src/lib/policy-cameras';

const front = 'observation.images.front',
  side = 'observation.images.camera_4';

test('each view of a policy gets its own camera, starting from the recording layout', () => {
  expect(viewName(side)).toBe('camera_4');
  // On the recording computer the second view was camera 4.
  expect(policyCameras([front, side], front, 0, [0, 2, 4])).toEqual({ [front]: 0, [side]: 4 });
  // Elsewhere camera 4 may not exist: take the next free camera, never the first view's.
  expect(policyCameras([front, side], front, 0, [0, 1])).toEqual({ [front]: 0, [side]: 1 });
  expect(policyCameras([front, side], front, 4, [0, 4])).toEqual({ [front]: 4, [side]: 0 });
  // The user's choice wins while that camera is free and still connected.
  expect(policyCameras([front, side], front, 0, [0, 1, 2], { [side]: 2 })).toEqual({ [front]: 0, [side]: 2 });
  expect(policyCameras([front, side], front, 2, [0, 1, 2], { [side]: 2 })).toEqual({ [front]: 2, [side]: 0 });
  expect(policyCameras([front, side], front, 0, [0, 1], { [side]: 7 })).toEqual({ [front]: 0, [side]: 1 });
});

test('too few cameras, or a selected camera that is gone, gives no mapping', () => {
  expect(policyCameras([front, side], front, 0, [0])).toBeNull();
  expect(policyCameras([front, side], front, 3, [0, 1])).toBeNull();
  expect(policyCameras([front, side], 'observation.images.other', 0, [0, 1])).toBeNull();
});

test('a check counts only for exactly the cameras it measured', () => {
  expect(sameCameras({ [front]: 0, [side]: 1 }, { [side]: 1, [front]: 0 })).toBe(true);
  expect(sameCameras({ [front]: 0, [side]: 1 }, { [front]: 1, [side]: 0 })).toBe(false);
  expect(sameCameras({ [front]: 0 }, { [front]: 0, [side]: 1 })).toBe(false);
  expect(sameCameras(undefined, { [front]: 0 })).toBe(false);
});
