# Runtime warnings

Job 18 passed the passive smoke test, but was not warning-free.

- `Failed to find articulation` on visual/material prims came from the test's
  recursive `/World/Robot/**` lookup. Those prims are not articulation roots.
  The test now targets `/World/Robot/joints/root_joint`. A regression test checks
  that path against the composed USD's sole `ArticulationRootAPI` prim.
  GPU job 19 confirmed one articulation, seven links and six DOFs over five
  simulated seconds with zero articulation lookup warnings. The tested source
  hash matches the delivered `smoke_test.py`; see `isaac-run.json` and
  `isaac-lookup-result.json`.
- Fabric reported unsupported values and an invalid array source during scene
  loading, plus missing rendering attributes. Their cause remains unresolved.
  Visible frames and finite physics states do not prove every attribute is valid.
- The container reported SimReady browser dependency failure, multiple OpenUSD
  builds, and OmniHub launch failures. These features need separate investigation;
  the local scene still loaded and recorded.
- Direct texture-to-host readback was flagged as inefficient.
- Audio was unavailable; Kit used a null output.

No collision-cooking, invalid-inertia or joint-limit errors were found in job 18.
The checks establish five seconds of passive stability, not pickup success.

API references: [tensor views](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html),
[simulation manager](https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.simulation_manager/docs/index.html).
