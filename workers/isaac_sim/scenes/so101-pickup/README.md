# Open and record the SO101 scene

Open `scene.usda` in Isaac Sim **6.1** and select `/World/Cameras/Front` or
`/World/Cameras/Overview`. Keep this directory and its robot meshes/textures
intact so relative references resolve.

## Checks and rebuild

From `workers/isaac_sim/scenes/so101-pickup`, edit `scene_config.json` and rebuild:

```sh
python -m pip install -r requirements-build.txt
python build_scene.py
python validate.py
```

Rebuild the robot with `python robot/build_robot.py`; see
[robot setup](robot/README.md). Check USD references, articulation joints/drives,
colliders, camera dimensions and capture settings after edits.

### Episode 1 profile

Use `scene.episode-001.usda` for the episode-1 starting arrangement. Rebuild it:

```sh
python build_episode.py
python validate.py scene.episode-001.usda
python -m unittest discover -s . -p test_episode.py
```

Copy `workers/skypilot/rollout.experimental.example.yaml` to a local task file.
Set its Isaac `SIM_MANIFEST` to `scenes/so101-pickup/rollout.episode-001.yaml`.
Use `--checkpoint` and `--experimental` with
[the SkyPilot rollout launcher](../../../skypilot/ROLLOUT.md); the manifest
requests 150 control steps at 30 Hz.

## Recording

Set `outputs.uri` in `capture.yaml` to your GCS prefix. From `workers/isaac_sim`:

```sh
python -m sim_worker --manifest scenes/so101-pickup/capture.yaml --validate-only
python -m sim_worker --manifest scenes/so101-pickup/capture.yaml --output-dir /outputs
```

Run the second command in the Isaac container with the whole bundle mounted.
The capture manifest requests 150 frames at 1920×1080 and 30 fps.

## GPU smoke test

In the Isaac container, mount a writable `/probe-output` and run:

```sh
/isaac-sim/python.sh --no-ros-env /opt/sim-worker/scenes/so101-pickup/smoke_test.py
```

Inspect its MP4, saved frames, live joint/body states and `result.json` under
`/probe-output`. Use `check_physics.py` for the articulation check without camera
rendering. Install the build and worker dependencies before running
`test_smoke_test.py`.

From `workers/skypilot`, prepare the remote check:

```sh
cp so101-test.example.yaml so101-test.local.yaml
```

Set `SIM_IMAGE` to the immutable worker digest, then submit:

```sh
SIM_PROJECT_ID=your-project bash sky.sh exec isaac-sim so101-test.local.yaml --env ACCEPT_EULA=Y --detach-run
```

Follow logs and stop the cluster using [SkyPilot commands](../../../skypilot/README.md).

## Sources

- [Dataset metadata](https://huggingface.co/datasets/codywang/so101_pickup_test/blob/ecef85bc07005f771ad86deeff1427f9d72953ed/meta/info.json)
- [Manufacturer SO101 model](https://github.com/TheRobotStudio/SO-ARM100/tree/5f6d2b876a53a4872e405b991dd925556c9e38a4/Simulation/SO101)
- [NVIDIA articulation setup](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/dev_guide/rigid_bodies_articulations/articulations.html)

Source assets and dataset declare Apache-2.0; see `NOTICE.md` and `LICENSE`.
