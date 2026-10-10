# Load and rebuild the SO101 follower

Load `so101.usda` with default prim `/SO101` and fixed articulation root
`/SO101/joints/root_joint`. Keep its source mesh layers together.

The CAD meshes, link masses/inertias and joint frames/limits come from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100/tree/5f6d2b876a53a4872e405b991dd925556c9e38a4/Simulation/SO101),
commit `5f6d2b876a53a4872e405b991dd925556c9e38a4`. Original Apache-2.0 license,
URDF, STL files and download hashes are retained in `source/`.

From the parent scene directory, install `numpy` and `usd-core`, then rebuild:

```sh
python robot/build_robot.py
# Optional: rebuild at the URDF zero pose.
python robot/build_robot.py --joint-degrees 0 0 0 0 0 0
```

Use `fetch_source.py` to refresh the pinned sources with `curl`. For placement,
use meters for stage translations and degrees for USD joint states, limits and
drive targets. Seat the root 0.002401 m above a support plane. The default pose
is `[9.36, -100, 96.57, 50, 7.96, 1.43]` degrees in
`shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll gripper` order.

After rebuilding, check references, articulation topology, forward kinematics,
inertias and matching states/targets. Use [the scene's checks](../README.md#gpu-smoke-test)
to inspect the seven links, six DOFs, fixed base and initial pose in Isaac.
