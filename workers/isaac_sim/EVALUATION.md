# Configure simulation outcome recording

## Criteria supplied before the run

Add this `evaluation` mapping to a separate rollout manifest. Replace the object
prim, support height, clearance, durations and speed limits with your selected
protocol. Supply every field:

```yaml
evaluation:
  task: lift_hold
  object_prim: /World/Props/Cup
  support_height_m: 0.2
  lift_clearance_m: 0.01
  lift_hold_seconds: 0.2
  settle_seconds: 0.2
  maximum_linear_speed_m_s: 0.02
  maximum_angular_speed_rad_s: 0.1
  placement_min_m: null
  placement_max_m: null
```

For `lift_hold`, use a continuous lift interval followed by a terminal settle
interval. For `lift_place`, also supply three-coordinate `placement_min_m` and
`placement_max_m` bounds in world coordinates.

Use a dynamic rigid body in a meter-authored Z-up stage. Keep its supported
leaf geometry under that body and use unscaled, unreflected transforms. Run the
prepared manifest with [the rollout command](ROLLOUT.md). Save every initial,
control-tick and final object observation in `trajectory.jsonl`.

## Geometry and replay

Replay a complete recorded trajectory from the repository root:

```sh
PYTHONPATH=workers/isaac_sim python -m sim_worker.rollout.evaluate_trace \
  --manifest path/to/explicit-rollout.yaml \
  --trajectory path/to/trajectory.jsonl \
  --output path/to/new-evaluation.json
```

Use a new output file. Read its criteria hash, trace hash, sample count,
terminal/peak measurements and `geometric_success`; keep the trajectory with its
job/model/scene manifest and collection receipt. For another criterion, create a
separate manifest and replay output.

References: [NVIDIA tensor poses and velocities](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html),
[OpenUSD extent computation](https://openusd.org/dev/api/class_usd_geom_boundable.html).
