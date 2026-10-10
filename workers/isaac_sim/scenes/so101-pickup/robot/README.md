# SO101 follower

`so101.usda` loads offline. Default prim: `/SO101`. Fixed articulation root: `/SO101/joints/root_joint`.

The 13 CAD meshes, seven link masses/inertias, six revolute joint frames and limits come from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100/tree/5f6d2b876a53a4872e405b991dd925556c9e38a4/Simulation/SO101), commit `5f6d2b876a53a4872e405b991dd925556c9e38a4`. Original Apache-2.0 license, URDF, STL files and download hashes are retained in `source/`.

White plastic, dark gripper and two orange tapered covers approximate the recorded hardware. Covers remain outside the grasp gap; their unknown mass is not added to the source inertias. Wiring and undocumented modifications are not reconstructed.

The initial pose is visually fitted: `[9.36, -100, 96.57, 50, 7.96, 1.43]` degrees, ordered `shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll gripper`. Shoulder is clipped to the URDF limit; wrist is adjusted for shelf clearance. The dataset's normalized gripper channel is not calibrated to this angular joint. This is not exact trajectory replay.

Rebuild with Python, `numpy` and `usd-core` installed:

```sh
python robot/build_robot.py
# Optional: rebuild at the URDF zero pose.
python robot/build_robot.py --joint-degrees 0 0 0 0 0 0
```

`fetch_source.py` refreshes pinned public sources with `curl`; normal scene use needs no network. Generated mesh layers contain 322,564 unique-source triangles. The articulated instance has 17 CAD colliders and two cover colliders. CAD uses convex decomposition; covers use convex hulls. PhysX cooks colliders on load.

Stage units are meters. USD joint state, limits and drive targets use degrees. Position drives retain URDF effort/velocity limits; gains are simulation estimates. Place the root 0.002401 m above a support plane to seat the CAD base. Fixed-root joint frames are ignored by PhysX.

Verify references, articulation topology, forward kinematics, inertias and
matching states/targets after rebuilding. Live Isaac checks must confirm all
seven links, six DOFs, fixed-base behavior and initial-pose stability. These
checks do not establish grasp behavior or learned control.
