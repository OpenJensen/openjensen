# SO101 pickup scene

Open **`scene.usda`** in Isaac Sim 6.1. Select **`/World/Cameras/Front`**.
Keep this directory intact: robot meshes and textures use relative paths.
`/World/Cameras/Overview` provides an alternate view.

The scene reconstructs the first frame of
[codywang/so101_pickup_test](https://huggingface.co/datasets/codywang/so101_pickup_test/tree/ecef85bc07005f771ad86deeff1427f9d72953ed).
It contains the SO101 follower, orange finger covers, paper cup, open cardboard
box, wooden shelves, metal frame, gray fabric panels, carpet and visible cables.
The dataset label says “cube”; the inspected video shows a paper cup.

## Fidelity

| Part | Basis |
| --- | --- |
| Robot geometry, joints, masses, inertias | Pinned manufacturer CAD/URDF; see `robot/README.md` |
| Surface textures and printed marks | Cropped dataset frame; original retained in `evidence/` |
| Camera | Estimated perspective, 1920×1080, 30 fps |
| Environment dimensions and placement | Estimated from the first image |
| Cup, box, finger covers, cables | Reconstructed geometry |
| Friction, cup mass, lighting, drive gains | Simulation estimates |
| Initial joint pose | Visual fit within model limits; calibration unavailable |

This is an approximate reconstruction. The dataset provides RGB and joint
samples, but no measured geometry, depth, camera calibration, motor calibration
or object physics. Hidden surfaces and clipped graphics are incomplete. Exact
scene identity and trajectory replay cannot be established from these files.
First-video episodes 0–3 and the first frame of video 7 were visually inspected;
all 30 episode metadata records were checked. This scene represents episode 0,
not every episode's changing cup position.

### Episode 1 profile

`scene.episode-001.usda` matches the successful episode 1 starting arrangement.
`episode-001.json` records its source frame, initial joint state, fitted props,
and **5-second rollout (150 steps at 30 Hz)**. The baseline scene is unchanged.

| Cup measurement | Baseline | Episode 1 fit |
| --- | --- | --- |
| Height | 10.50 cm | 10.22 cm |
| Upper body diameter | 7.40 cm | 8.31 cm |
| Bottom diameter | 5.00 cm | 5.53 cm |
| Horizontal distance from robot base | 31.26 cm | 25.14 cm |

The cup bottom is at `(-0.2247814, -0.0291302, 0.2005)` m. Its visual mesh,
open wall colliders, label surface, center of mass, and inertia are rebuilt
together. The box opening is fitted to 27.17 × 19.17 cm; its estimated height
stays 20 cm. Camera and shelf geometry retain the original shelf fit.

Cup contour error on held-out stationary frames is 0.98 px at 1920×1080
(0.33 px at policy resolution). Box corner RMS improves from 17.94 to 5.91 px
at policy resolution. Measurements and limitations are in `evidence/episode-001/`.
These image fits do **not** establish physical scale: shelf width remains an
assumed 1 m. The rolled cup lip, torn box rim, lighting, and contact properties
remain approximate. The unverified joint map still clips the initial shoulder
lift by 11.42°; matching prop geometry does not validate robot calibration.

Rebuild and validate:

```sh
python build_episode.py
python validate.py scene.episode-001.usda
python -m unittest discover -s . -p test_episode.py
```

For SkyPilot, copy `workers/skypilot/rollout.experimental.example.yaml` to a local
task file and set its Isaac `SIM_MANIFEST` to
`scenes/so101-pickup/rollout.episode-001.yaml`. Launch with `--checkpoint` to select
ACT or SmolVLA and `--experimental`. The selector updates model identity and
action horizon while preserving the 150-step duration. No GPU validation of
this profile is claimed by the local structure tests or Blender preview.

The robot is a fixed-base articulation with six position drives. The cup is
dynamic with separate wall colliders; the box and shelving are static colliders.
The cup and box remain open. The stage holds the reconstructed starting pose;
it contains no recorded pick-and-place animation.

## Checks and rebuild

Opening the supplied USD requires no downloads or Python packages.
To edit dimensions/materials, change `scene_config.json`, then rebuild:

```sh
python -m pip install -r requirements-build.txt
python build_scene.py
python validate.py
```

Rebuild the robot separately with `python robot/build_robot.py`.
Source revisions and file hashes are retained under `robot/source/` and
`evidence/dataset.json`.

Verified locally: USD composition and assets, physics graph, FK, inertias,
joint states/drives, open colliders, camera aspect ratio, and capture manifest.
The base seats on the shelf; the cup starts 0.5 mm above it. `preview.png` renders
the actual USD in Blender with renderer-specific light conversion. It is a
geometry preview. `evidence/isaac-frame.png` is the actual Isaac render.

SkyPilot job 18 passed on GCP `g2-standard-16` / NVIDIA L4 with Isaac Sim 6.1:
150 visible frames at 1280×720/30 fps, one seven-link articulation, six valid
joint states, fixed base, supported cup, and advancing physics. The run took
8m 36s, mostly renderer startup. Reports and image digest are in
`evidence/isaac-result.json` and `evidence/isaac-run.json`. Grasp behavior remains
untested; no pickup controller is included.

Job 19 also passed the corrected articulation lookup without the visual-mesh
lookup warnings. Other runtime warnings remain; see `evidence/warnings.md`.

## Recording

`capture.yaml` requests 150 frames at 1920×1080/30 fps. Set `outputs.uri` to your
GCS prefix. From the existing Isaac worker directory:

```sh
python -m sim_worker --manifest scenes/so101-pickup/capture.yaml --validate-only
python -m sim_worker --manifest scenes/so101-pickup/capture.yaml --output-dir /outputs
```

Run the second command inside the worker's Isaac environment with this entire
bundle mounted. The recording uses the configured initial pose and physics.

## GPU smoke test

`smoke_test.py` uses the worker's renderer and video encoder. In its Isaac
container, run:

```sh
/isaac-sim/python.sh --no-ros-env /opt/sim-worker/scenes/so101-pickup/smoke_test.py
```

Mount a writable `/probe-output`. The test saves a five-second 1280×720 MP4,
three frames, live joint/body states, and `result.json`. It checks visible
frames, advancing physics, joint limits, a fixed base, and cup support. A passing
result verifies passive stability; it does not establish pickup success.

Both GPU checks require the existing worker source and its dependencies.
`check_physics.py` checks the same live articulation lookup without camera
rendering. `test_smoke_test.py` guards the lookup against matching visual meshes;
run it with the build dependencies and worker dependencies installed.

From `workers/skypilot`, copy the test template once:

```sh
cp so101-test.example.yaml so101-test.local.yaml
```

Set `SIM_IMAGE` in the local copy to your worker image digest. Submit to the
existing cluster with the wrapper:

```sh
SIM_PROJECT_ID=your-project bash sky.sh exec isaac-sim so101-test.local.yaml --env ACCEPT_EULA=Y --detach-run
```

## Sources

- [Dataset metadata](https://huggingface.co/datasets/codywang/so101_pickup_test/blob/ecef85bc07005f771ad86deeff1427f9d72953ed/meta/info.json)
- [Manufacturer SO101 model](https://github.com/TheRobotStudio/SO-ARM100/tree/5f6d2b876a53a4872e405b991dd925556c9e38a4/Simulation/SO101)
- [NVIDIA articulation setup](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/dev_guide/rigid_bodies_articulations/articulations.html)

Source assets and dataset declare Apache-2.0; see `NOTICE.md` and `LICENSE`.
