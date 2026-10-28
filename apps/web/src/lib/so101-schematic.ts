import type { RobotJoint, RobotRole } from './robot-api';

export type Vec3 = [number, number, number];
export type Mat4 = number[];
export type JointFrame = { origin: Vec3; axis: Vec3; matrix: Mat4 };
export type RobotPose = { role: RobotRole; links: Record<string, Mat4>; joints: Record<RobotJoint, JointFrame> };
export type Camera = { matrix: Mat4; right: Vec3; up: Vec3; forward: Vec3; target: Vec3; span: number };

/** Exact joint/visual transforms extracted from the pinned SO-101 URDF below.
 * Meshes and Apache-2.0 provenance are in public/robots/so101/. These are CAD
 * coordinates, never a calibrated physical pose or motor telemetry.
 */
export const so101Description = {
  source: 'TheRobotStudio/SO-ARM100@5f6d2b876a53a4872e405b991dd925556c9e38a4',
  urdfSha256: '3a65d2d35e68a8d2f0c2cc176d19b884506543c93ba72980145b80abe276022c',
  joints: [
    {
      name: 'gripper',
      parent: 'gripper_link',
      child: 'moving_jaw_so101_v1_link',
      xyz: [0.0202, 0.0188, -0.0234],
      rpy: [1.5708, -5.24284e-8, -1.41553e-15],
      axis: [0.0, 0.0, 1.0],
      lower: -0.174533,
      upper: 1.74533,
    },
    {
      name: 'wrist_roll',
      parent: 'wrist_link',
      child: 'gripper_link',
      xyz: [5.55112e-17, -0.0611, 0.0181],
      rpy: [1.5708, 0.0486795, 3.14159],
      axis: [0.0, 0.0, 1.0],
      lower: -2.74385,
      upper: 2.84121,
    },
    {
      name: 'wrist_flex',
      parent: 'lower_arm_link',
      child: 'wrist_link',
      xyz: [-0.1349, 0.0052, 3.62355e-17],
      rpy: [4.02456e-15, 8.67362e-16, -1.5708],
      axis: [0.0, 0.0, 1.0],
      lower: -1.65806,
      upper: 1.65806,
    },
    {
      name: 'elbow_flex',
      parent: 'upper_arm_link',
      child: 'lower_arm_link',
      xyz: [-0.11257, -0.028, 1.73763e-16],
      rpy: [-3.63608e-16, 8.74301e-16, 1.5708],
      axis: [0.0, 0.0, 1.0],
      lower: -1.69,
      upper: 1.69,
    },
    {
      name: 'shoulder_lift',
      parent: 'shoulder_link',
      child: 'upper_arm_link',
      xyz: [-0.0303992, -0.0182778, -0.0542],
      rpy: [-1.5708, -1.5708, 0.0],
      axis: [0.0, 0.0, 1.0],
      lower: -1.74533,
      upper: 1.74533,
    },
    {
      name: 'shoulder_pan',
      parent: 'base_link',
      child: 'shoulder_link',
      xyz: [0.0388353, -8.97657e-9, 0.0624],
      rpy: [3.14159, 4.18253e-17, -3.14159],
      axis: [0.0, 0.0, 1.0],
      lower: -1.91986,
      upper: 1.91986,
    },
  ],
  visuals: [
    {
      link: 'base_link',
      mesh: 'base_motor_holder_so101_v1.stl',
      xyz: [-0.00636471, -9.94414e-5, -0.0024],
      rpy: [1.5708, -1.67685e-15, 1.5708],
      material: '3d_printed',
    },
    {
      link: 'base_link',
      mesh: 'base_so101_v2.stl',
      xyz: [-0.00636471, -8.97657e-9, -0.0024],
      rpy: [1.5708, -2.78073e-29, 1.5708],
      material: '3d_printed',
    },
    {
      link: 'base_link',
      mesh: 'sts3215_03a_v1.stl',
      xyz: [0.0263353, -8.97657e-9, 0.0437],
      rpy: [-8.21148e-16, 7.84513e-18, 1.249e-15],
      material: 'sts3215',
    },
    {
      link: 'base_link',
      mesh: 'waveshare_mounting_plate_so101_v2.stl',
      xyz: [-0.0309827, -0.000199441, 0.0474],
      rpy: [1.5708, -1.35493e-14, 1.5708],
      material: '3d_printed',
    },
    {
      link: 'shoulder_link',
      mesh: 'sts3215_03a_v1.stl',
      xyz: [-0.0303992, 0.000422241, -0.0417],
      rpy: [1.5708, 1.5708, 0.0],
      material: 'sts3215',
    },
    {
      link: 'shoulder_link',
      mesh: 'motor_holder_so101_base_v1.stl',
      xyz: [-0.0675992, -0.000177759, 0.0158499],
      rpy: [1.5708, -1.5708, 0.0],
      material: '3d_printed',
    },
    {
      link: 'shoulder_link',
      mesh: 'rotation_pitch_so101_v1.stl',
      xyz: [0.0122008, 2.22413e-5, 0.0464],
      rpy: [-1.5708, 2.35221e-33, 0.0],
      material: '3d_printed',
    },
    {
      link: 'upper_arm_link',
      mesh: 'sts3215_03a_v1.stl',
      xyz: [-0.11257, -0.0155, 0.0187],
      rpy: [-3.14159, -5.27356e-16, -1.5708],
      material: 'sts3215',
    },
    {
      link: 'upper_arm_link',
      mesh: 'upper_arm_so101_v1.stl',
      xyz: [-0.065085, 0.012, 0.0182],
      rpy: [3.14159, -0.0, -1.30911e-30],
      material: '3d_printed',
    },
    {
      link: 'lower_arm_link',
      mesh: 'under_arm_so101_v1.stl',
      xyz: [-0.0648499, -0.032, 0.0182],
      rpy: [3.14159, -0.0, 6.67202e-31],
      material: '3d_printed',
    },
    {
      link: 'lower_arm_link',
      mesh: 'motor_holder_so101_wrist_v1.stl',
      xyz: [-0.0648499, -0.032, 0.018],
      rpy: [-3.14159, -2.55351e-15, -1.83387e-30],
      material: '3d_printed',
    },
    {
      link: 'lower_arm_link',
      mesh: 'sts3215_03a_v1.stl',
      xyz: [-0.1224, 0.0052, 0.0187],
      rpy: [-3.14159, -7.88861e-31, -3.14159],
      material: 'sts3215',
    },
    {
      link: 'wrist_link',
      mesh: 'sts3215_03a_no_horn_v1.stl',
      xyz: [8.32667e-17, -0.0424, 0.0306],
      rpy: [1.5708, 1.5708, 0.0],
      material: 'sts3215',
    },
    {
      link: 'wrist_link',
      mesh: 'wrist_roll_pitch_so101_v2.stl',
      xyz: [0.0, -0.028, 0.0181],
      rpy: [-1.5708, -1.5708, 0.0],
      material: '3d_printed',
    },
    {
      link: 'gripper_link',
      mesh: 'sts3215_03a_v1.stl',
      xyz: [0.0077, 0.0001, -0.0234],
      rpy: [-1.5708, -5.19179e-17, -1.66533e-16],
      material: 'sts3215',
    },
    {
      link: 'gripper_link',
      mesh: 'wrist_roll_follower_so101_v1.stl',
      xyz: [8.32667e-17, -0.000218214, 0.000949706],
      rpy: [-3.14159, -5.55112e-17, 0.0],
      material: '3d_printed',
    },
    {
      link: 'moving_jaw_so101_v1_link',
      mesh: 'moving_jaw_so101_v1.stl',
      xyz: [-5.55112e-17, -5.55112e-17, 0.0189],
      rpy: [9.53145e-17, 6.93889e-18, 1.24077e-24],
      material: '3d_printed',
    },
  ],
} as const;

// The leader shares the five arm joints and replaces the follower jaw with a handle and trigger.
export const so101LeaderDescription = {
  ...so101Description,
  urdfSha256: 'bc947e2fea65dfc91b0fc40e4cff0c01c6bf426687995937e62e2b0f41122db5',
  joints: so101Description.joints.map((joint) =>
    joint.name === 'gripper' ? { ...joint, child: 'trigger_link' as const, lower: 0, upper: 0.785398163397 } : joint,
  ),
  visuals: [
    ...so101Description.visuals.filter(
      (visual) => !['wrist_roll_follower_so101_v1.stl', 'moving_jaw_so101_v1.stl'].includes(visual.mesh),
    ),
    ...([
      {
        link: 'gripper_link',
        mesh: 'Wrist_Roll_SO101.stl',
        xyz: [0.0, 0.0, 0.0],
        rpy: [3.14159265359, 0.0, 0.0],
        material: '3d_printed',
      },
      {
        link: 'gripper_link',
        mesh: 'Handle_SO101.stl',
        xyz: [0.0, 0.0, 0.0],
        rpy: [3.14159265359, 0.0, 0.0],
        material: '3d_printed',
      },
      {
        link: 'trigger_link',
        mesh: 'Trigger_SO101.stl',
        xyz: [0.0, 0.0, 0.0],
        rpy: [3.14159265359, 0.0, 2.44346095279],
        material: '3d_printed',
      },
    ] as const),
  ],
} as const;
export const descriptionForRole = (role: RobotRole = 'follower') =>
  role === 'leader' ? so101LeaderDescription : so101Description;
export type CadVisual = {
  link: string;
  mesh: string;
  xyz: readonly number[];
  rpy: readonly number[];
  material: string;
};

export const jointOrder: RobotJoint[] = [
  'shoulder_pan',
  'shoulder_lift',
  'elbow_flex',
  'wrist_flex',
  'wrist_roll',
  'gripper',
];
export const referenceDegrees: Record<RobotJoint, number> = {
  shoulder_pan: 0,
  shoulder_lift: -45,
  elbow_flex: 45,
  wrist_flex: 30,
  wrist_roll: 0,
  gripper: 20,
};
export const previewRanges: Record<RobotJoint, [number, number]> = {
  shoulder_pan: [-25, 25],
  shoulder_lift: [-18, 18],
  elbow_flex: [-22, 22],
  wrist_flex: [-25, 25],
  wrist_roll: [-55, 55],
  gripper: [-18, 30],
};
export const add3 = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scale3 = (a: Vec3, k: number): Vec3 => [a[0] * k, a[1] * k, a[2] * k];
export const dot3 = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross3 = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const normalize3 = (a: Vec3): Vec3 => scale3(a, 1 / Math.max(1e-12, Math.hypot(...a)));
export const identity4 = (): Mat4 => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
export function multiply4(a: Mat4, b: Mat4): Mat4 {
  const out = new Array<number>(16).fill(0);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) out[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
  return out;
}
export function transformPoint(m: Mat4, p: readonly number[]): Vec3 {
  return [
    m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
    m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
    m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
  ];
}
export function transformVector(m: Mat4, p: readonly number[]): Vec3 {
  return [
    m[0] * p[0] + m[4] * p[1] + m[8] * p[2],
    m[1] * p[0] + m[5] * p[1] + m[9] * p[2],
    m[2] * p[0] + m[6] * p[1] + m[10] * p[2],
  ];
}
export function originMatrix(xyz: readonly number[], rpy: readonly number[]): Mat4 {
  const [r, p, y] = rpy,
    cr = Math.cos(r),
    sr = Math.sin(r),
    cp = Math.cos(p),
    sp = Math.sin(p),
    cy = Math.cos(y),
    sy = Math.sin(y);
  return [
    cy * cp,
    sy * cp,
    -sp,
    0,
    cy * sp * sr - sy * cr,
    sy * sp * sr + cy * cr,
    cp * sr,
    0,
    cy * sp * cr + sy * sr,
    sy * sp * cr - cy * sr,
    cp * cr,
    0,
    xyz[0],
    xyz[1],
    xyz[2],
    1,
  ];
}
export function jointPose(
  selected?: RobotJoint,
  offsetDegrees = 0,
  reference = referenceDegrees,
  role: RobotRole = 'follower',
): RobotPose {
  const description = descriptionForRole(role);
  const links: Record<string, Mat4> = { base_link: identity4() },
    frames = {} as Record<RobotJoint, JointFrame>;
  for (const name of jointOrder) {
    const joint = description.joints.find((value) => value.name === name)!;
    const frame = multiply4(links[joint.parent], originMatrix(joint.xyz, joint.rpy));
    const q = ((reference[name] + (name === selected ? offsetDegrees : 0)) * Math.PI) / 180;
    const child = multiply4(frame, originMatrix([0, 0, 0], [0, 0, q]));
    frames[name] = {
      origin: transformPoint(frame, [0, 0, 0]),
      axis: normalize3(transformVector(frame, joint.axis)),
      matrix: frame,
    };
    links[joint.child] = child;
  }
  return { role, links, joints: frames };
}
export function cameraForModel(yaw = 0, pitch = 0, zoom = 1): Camera {
  const azimuth = Math.atan2(1, -0.7) + (yaw * Math.PI) / 180,
    elevation = Math.max(-1.35, Math.min(1.35, Math.atan2(-0.55, Math.hypot(0.7, 1)) + (pitch * Math.PI) / 180));
  const target: Vec3 = [0.12, 0, 0.14],
    forward = normalize3([
      Math.cos(azimuth) * Math.cos(elevation),
      Math.sin(azimuth) * Math.cos(elevation),
      Math.sin(elevation),
    ]),
    right = normalize3(cross3(forward, [0, 0, 1])),
    up = normalize3(cross3(right, forward)),
    span = 0.36 / Math.max(0.65, Math.min(1.8, zoom));
  const matrix: Mat4 = [
    (right[0] * 2) / ((span * 600) / 420),
    (up[0] * 2) / span,
    -forward[0] * 2,
    0,
    (right[1] * 2) / ((span * 600) / 420),
    (up[1] * 2) / span,
    -forward[1] * 2,
    0,
    (right[2] * 2) / ((span * 600) / 420),
    (up[2] * 2) / span,
    -forward[2] * 2,
    0,
    (-dot3(right, target) * 2) / ((span * 600) / 420),
    (-dot3(up, target) * 2) / span,
    dot3(forward, target) * 2,
    1,
  ];
  return { matrix, right, up, forward, target, span };
}
export const so101Camera = cameraForModel();
export function projectJoint(point: Vec3, camera = so101Camera): { x: number; y: number } {
  const p = add3(point, scale3(camera.target, -1));
  return { x: 300 + (dot3(p, camera.right) * 420) / camera.span, y: 210 - (dot3(p, camera.up) * 420) / camera.span };
}
export function affectedLink(selected: RobotJoint, link: string, role: RobotRole = 'follower'): boolean {
  const description = descriptionForRole(role);
  const start = jointOrder.indexOf(selected),
    index = jointOrder.findIndex((name) => description.joints.find((value) => value.name === name)!.child === link);
  return index >= start;
}
export function visualMatrix(pose: RobotPose, visual: CadVisual): Mat4 {
  return multiply4(pose.links[visual.link], originMatrix(visual.xyz, visual.rpy));
}
