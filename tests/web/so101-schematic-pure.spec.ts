import { expect, test } from '@playwright/test';
import { so101Schematic, projectSo101 } from '../../apps/web/src/lib/so101-schematic';

const first = -Math.PI / 2, second = Math.PI / 2;

test('gripper animation keeps the fixed finger and all arm joints still while hinging one jaw', () => {
  const a = so101Schematic('gripper', first), b = so101Schematic('gripper', second);
  expect(a.joints).toEqual(b.joints);
  expect(a.fixedFinger).toEqual(b.fixedFinger);
  expect(a.movingFinger[0]).toEqual(b.movingFinger[0]);
  expect(a.movingFinger[1]).not.toEqual(b.movingFinger[1]);
});

test('base yaw preserves heights while wrist roll rotates fingers without moving arm pivots', () => {
  const a = so101Schematic('shoulder_pan', first), b = so101Schematic('shoulder_pan', second);
  expect(a.axis).toEqual({ x: 0, y: 0, z: 1 });
  for (const joint of Object.keys(a.joints) as (keyof typeof a.joints)[]) expect(a.joints[joint].z).toBeCloseTo(b.joints[joint].z, 10);
  expect(a.joints.gripper.y).not.toBeCloseTo(b.joints.gripper.y, 4);
  const rollA = so101Schematic('wrist_roll', first), rollB = so101Schematic('wrist_roll', second);
  expect(rollA.joints).toEqual(rollB.joints);
  expect(rollA.fixedFinger).not.toEqual(rollB.fixedFinger);
});

test('pitch hinges only move links downstream of their selected pivot', () => {
  for (const joint of ['shoulder_lift', 'elbow_flex', 'wrist_flex'] as const) {
    const a = so101Schematic(joint, first), b = so101Schematic(joint, second);
    expect(a.joints.shoulder_pan).toEqual(b.joints.shoulder_pan);
    expect(a.joints[joint]).toEqual(b.joints[joint]);
    expect(a.joints.gripper).not.toEqual(b.joints.gripper);
  }
});

test('illustration stays within its view box for every joint motion', () => {
  for (const joint of ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'] as const) {
    for (let phase = 0; phase < Math.PI * 2; phase += .2) {
      const pose = so101Schematic(joint, phase);
      for (const point of [...Object.values(pose.joints), ...pose.fixedFinger, ...pose.movingFinger]) {
        const pixel = projectSo101(point);
        expect(pixel.x).toBeGreaterThan(0); expect(pixel.x).toBeLessThan(520);
        expect(pixel.y).toBeGreaterThan(0); expect(pixel.y).toBeLessThan(380);
      }
    }
  }
});
