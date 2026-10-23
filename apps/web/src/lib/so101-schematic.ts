import type { RobotJoint } from './robot-api';

export type Point3 = { x: number; y: number; z: number };
export type Point2 = { x: number; y: number };
export type SchematicPose = {
  joints: Record<RobotJoint, Point3>;
  axis: Point3;
  fixedFinger: [Point3, Point3];
  movingFinger: [Point3, Point3];
  palm: [Point3, Point3];
};
const add = (a: Point3, b: Point3): Point3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });
const scale = (a: Point3, value: number): Point3 => ({ x: a.x * value, y: a.y * value, z: a.z * value });
const radians = (degrees: number) => degrees * Math.PI / 180;

/** A schematic, never robot telemetry. SO-101 topology/axes follow the repository's
 * so101_new_calib.urdf; dimensions and neutral pose are simplified for legibility.
 * Pan is vertical; lift/elbow/flex share transverse axes; roll follows the tool;
 * the gripper has one fixed finger and one hinged moving jaw.
 */
export function so101Schematic(selected: RobotJoint, phase: number): SchematicPose {
  const movement = Math.sin(phase);
  const pan = radians(selected === 'shoulder_pan' ? movement * 24 : -12);
  const lift = radians(68 + (selected === 'shoulder_lift' ? movement * 17 : 0));
  const elbow = radians(-72 + (selected === 'elbow_flex' ? movement * 23 : 0));
  const flex = radians(22 + (selected === 'wrist_flex' ? movement * 25 : 0));
  const roll = radians(20 + (selected === 'wrist_roll' ? movement * 70 : 0));
  const jaw = radians(selected === 'gripper' ? 6 + (movement + 1) * 18 : 20);
  const rotatePan = (p: Point3): Point3 => ({ x: p.x * Math.cos(pan) - p.y * Math.sin(pan), y: p.x * Math.sin(pan) + p.y * Math.cos(pan), z: p.z });
  const armDirection = (angle: number) => rotatePan({ x: Math.cos(angle), y: 0, z: Math.sin(angle) });
  const transverse = rotatePan({ x: 0, y: 1, z: 0 });
  const toolDirection = armDirection(lift + elbow + flex);
  const unrolledUp = rotatePan({ x: -Math.sin(lift + elbow + flex), y: 0, z: Math.cos(lift + elbow + flex) });
  const toolSide = add(scale(transverse, Math.cos(roll)), scale(unrolledUp, Math.sin(roll)));
  const toolUp = add(scale(transverse, -Math.sin(roll)), scale(unrolledUp, Math.cos(roll)));
  const shoulder_pan = { x: 0, y: 0, z: .043 };
  const shoulder_lift = { x: 0, y: 0, z: .092 };
  const elbow_flex = add(shoulder_lift, scale(armDirection(lift), .119));
  const wrist_flex = add(elbow_flex, scale(armDirection(lift + elbow), .135));
  const wrist_roll = add(wrist_flex, scale(toolDirection, .045));
  const gripper = add(wrist_roll, scale(toolDirection, .054));
  const fixedRoot = add(gripper, scale(toolSide, .017));
  const movingRoot = add(gripper, scale(toolSide, -.017));
  const fixedTip = add(fixedRoot, scale(toolDirection, .046));
  const movingTip = add(add(movingRoot, scale(toolDirection, .046 * Math.cos(jaw))), scale(toolSide, -.046 * Math.sin(jaw)));
  return {
    joints: { shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper },
    axis: selected === 'shoulder_pan' ? { x: 0, y: 0, z: 1 } : selected === 'wrist_roll' ? toolDirection : selected === 'gripper' ? toolUp : transverse,
    fixedFinger: [fixedRoot, fixedTip], movingFinger: [movingRoot, movingTip], palm: [fixedRoot, movingRoot],
  };
}
export function projectSo101(point: Point3): Point2 { return { x: 90 + point.x * 1050 - point.y * 650, y: 340 - point.z * 1000 + point.x * 280 + point.y * 210 }; }
export function schematicAxis(pose: SchematicPose, selected: RobotJoint): [Point2, Point2] {
  const pivot = pose.joints[selected];
  return [projectSo101(add(pivot, scale(pose.axis, -.038))), projectSo101(add(pivot, scale(pose.axis, .038)))];
}
