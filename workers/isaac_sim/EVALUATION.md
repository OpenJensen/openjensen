# Explicit simulation outcomes

Run answers whether an exact policy executes and records an episode. Evaluation
answers whether recorded behavior satisfies a declared criterion. Neither a
completed process nor a moving arm establishes that a cup was lifted or placed.

The opt-in rollout evaluator measures a rigid object's geometry and motion.
Existing manifests omit this block and keep their diagnostic behavior. No cloud
profile is changed; the new probe has not yet been qualified in live Isaac GPU
execution. Application selection and score presentation remain separate gates.

## Criteria supplied before the run

Add an `evaluation` mapping to a separate rollout manifest. Every field is
required; no task or physical threshold is inferred from the prompt. These
**illustrative** values are not the accepted demo protocol:

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

`lift_hold` requires the whole object to clear the support plane continuously for
the lift interval, followed by a stable terminal interval. Dropping it resets
the current hold; an earlier lift cannot make a later short lift pass.
`lift_place` requires an earlier qualifying lift and the entire object inside an
explicit world-aligned placement region at the end, with speeds within bounds
throughout the terminal settle interval. Supply three coordinates for each
placement bound. Region coordinates are a protocol input, not an inferred box.

Every control tick, including initial and final observations, is required.
Duration is measured between samples. Missing, duplicate, reordered or
cross-episode observations cannot produce a score. Already-successful reset
geometry is rejected. The whole conservative oriented bounding box must qualify;
this can reject a shape that would pass a more exact mesh predicate.

## Geometry and replay

The probe recomputes supported leaf-geometry extents from USD and transforms them
into rigid-body coordinates. Stale authored cached extents are ignored. The
live translation, xyzw quaternion and linear/angular velocities come from the
owned physics tensor view, explicitly bound to world space. Reading does not
advance physics. Initially this supports dynamic rigid bodies in meter-authored
Z-up stages. Instanced geometry, scaled/reflected bodies, animated ancestry or
geometry, nested rigid bodies and unsupported shapes fail explicitly.

`trajectory.jsonl` retains both adjacent object states and per-tick scores.
`result.json.evaluation` records criteria and SHA-256, sample count, terminal/peak
measurements and `geometric_success`. Replay without starting a GPU:

```sh
PYTHONPATH=workers/isaac_sim python -m sim_worker.rollout.evaluate_trace \
  --manifest path/to/explicit-rollout.yaml \
  --trajectory path/to/trajectory.jsonl \
  --output path/to/new-evaluation.json
```

Output is exclusive-create and hashes the consumed trace. Replay requires
complete object evidence and continuity; historical robot-only traces cannot
be scored from video or joint data. Raw evidence needs its enclosing job/model/
scene manifest and collection receipt verified by the consumer. Changing
criteria changes score identity; post-hoc criteria are not a pre-registered test.

## Evidence limits

`geometric_success` describes sampled simulator geometry and motion. Results
keep `quality_validated: false` and `grasp_and_release_verified: false`. Contact,
grasp, physical calibration, held-out generalization and optimizer promotion need
their separate evidence. Missing evidence raises an error rather than silently
changing the success denominator.

Local checks cover real bundled USD geometry, generated trajectories, replay
and unchanged diagnostic rollout behavior. They do not prove live GPU behavior.
Adapter references:
[NVIDIA tensor poses and velocities](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html),
[OpenUSD geometry extent computation](https://openusd.org/dev/api/class_usd_geom_boundable.html).
