import { expect, test } from '@playwright/test';
import {
  affectedLink,
  so101LeaderDescription,
  jointOrder,
  jointPose,
  previewRanges,
  projectJoint,
  so101Description,
  transformPoint,
  visualMatrix,
} from '../../apps/web/src/lib/so101-schematic';
import { parseBinaryStl } from '../../apps/web/src/lib/so101-webgl';

const source = '3a65d2d35e68a8d2f0c2cc176d19b884506543c93ba72980145b80abe276022c';
test('CAD uses pinned URDF origins, axes and seven articulated links', () => {
  expect(so101Description.urdfSha256).toBe(source);
  expect(so101Description.joints).toHaveLength(6);
  const zero = Object.fromEntries(jointOrder.map((name) => [name, 0])) as Record<(typeof jointOrder)[number], number>,
    pose = jointPose(undefined, 0, zero);
  expect(Object.keys(pose.links)).toHaveLength(7);
  expect(pose.joints.shoulder_pan.origin[0]).toBeCloseTo(0.0388353, 7);
  expect(pose.joints.shoulder_pan.origin[2]).toBeCloseTo(0.0624, 7);
  expect(pose.joints.shoulder_lift.origin[2]).toBeCloseTo(0.1165999515, 7);
  for (const frame of Object.values(pose.joints)) expect(Math.hypot(...frame.axis)).toBeCloseTo(1, 10);
});
test('gripper changes only the actual hinged jaw link', () => {
  const a = jointPose('gripper', -15),
    b = jointPose('gripper', 30),
    child = so101Description.joints.find((joint) => joint.name === 'gripper')!.child;
  for (const name of Object.keys(a.links)) {
    if (name === child) expect(a.links[name]).not.toEqual(b.links[name]);
    else expect(a.links[name]).toEqual(b.links[name]);
  }
  expect(a.joints.gripper.origin).toEqual(b.joints.gripper.origin);
  const fixed = so101Description.visuals.filter((part) => part.link === 'gripper_link');
  expect(fixed.length).toBeGreaterThan(0);
  for (const visual of fixed) expect(visualMatrix(a, visual)).toEqual(visualMatrix(b, visual));
});
test('rotation keeps the selected pivot fixed and all upstream links stationary', () => {
  for (const selected of jointOrder) {
    const a = jointPose(selected, previewRanges[selected][0]),
      b = jointPose(selected, previewRanges[selected][1]);
    expect(a.joints[selected].origin).toEqual(b.joints[selected].origin);
    expect(a.joints[selected].axis).toEqual(b.joints[selected].axis);
    const index = jointOrder.indexOf(selected);
    for (const earlier of jointOrder.slice(0, index)) {
      const link = so101Description.joints.find((joint) => joint.name === earlier)!.child;
      expect(a.links[link]).toEqual(b.links[link]);
    }
    for (const frame of Object.values(a.joints)) {
      const p = projectJoint(frame.origin);
      expect(p.x).toBeGreaterThan(0);
      expect(p.x).toBeLessThan(600);
      expect(p.y).toBeGreaterThan(0);
      expect(p.y).toBeLessThan(420);
    }
  }
});
test('binary STL parser preserves normals and vertices and rejects malformed meshes', () => {
  const buffer = new ArrayBuffer(134),
    data = new DataView(buffer);
  data.setUint32(80, 1, true);
  data.setFloat32(92, 1, true);
  data.setFloat32(108, 0.1, true);
  data.setFloat32(124, 0.2, true);
  const mesh = parseBinaryStl(buffer);
  expect(mesh.count).toBe(3);
  expect(mesh.vertices).toHaveLength(18);
  expect(mesh.vertices[5]).toBe(1);
  expect(() => parseBinaryStl(buffer.slice(0, 120))).toThrow('Invalid CAD mesh');
  data.setFloat32(96, NaN, true);
  expect(() => parseBinaryStl(buffer)).toThrow('Non-finite CAD vertex');
});
test('URDF visual transforms retain real mesh offsets instead of replacing links with sticks', () => {
  const pose = jointPose(),
    base = so101Description.visuals.find((part) => part.mesh === 'base_so101_v2.stl')!;
  const origin = transformPoint(visualMatrix(pose, base), [0, 0, 0]);
  expect(origin[0]).toBeCloseTo(-0.00636471, 8);
  expect(origin[2]).toBeCloseTo(-0.0024, 8);
  expect(new Set(so101Description.visuals.map((part) => part.mesh)).size).toBe(13);
});

test('leader CAD uses its handle and trigger while retaining the five shared arm joints', () => {
  expect(so101LeaderDescription.urdfSha256).not.toBe(so101Description.urdfSha256);
  for (const joint of so101Description.joints.filter((joint) => joint.name !== 'gripper')) {
    expect(so101LeaderDescription.joints.find((other) => other.name === joint.name)).toEqual(joint);
  }
  const meshes = new Set<string>(so101LeaderDescription.visuals.map((part) => part.mesh));
  for (const mesh of ['Handle_SO101.stl', 'Trigger_SO101.stl', 'Wrist_Roll_SO101.stl'])
    expect(meshes.has(mesh)).toBe(true);
  for (const mesh of ['moving_jaw_so101_v1.stl', 'wrist_roll_follower_so101_v1.stl'])
    expect(meshes.has(mesh)).toBe(false);
  const a = jointPose('gripper', -18, undefined, 'leader');
  const b = jointPose('gripper', 20, undefined, 'leader');
  expect(a.role).toBe('leader');
  expect(a.links.trigger_link).not.toEqual(b.links.trigger_link);
  expect(a.links.moving_jaw_so101_v1_link).toBeUndefined();
  expect(a.joints.gripper.origin).toEqual(b.joints.gripper.origin);
  for (const visual of so101LeaderDescription.visuals.filter((part) => part.link !== 'trigger_link'))
    expect(visualMatrix(a, visual)).toEqual(visualMatrix(b, visual));
  expect(affectedLink('gripper', 'trigger_link', 'leader')).toBe(true);
  expect(affectedLink('gripper', 'gripper_link', 'leader')).toBe(false);
});
