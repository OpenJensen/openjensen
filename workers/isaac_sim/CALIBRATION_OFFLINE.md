# Offline calibration

These commands use recorded data and CPU geometry. They start no Isaac process,
physics, policy server, or cloud job. Active scenes and calibration stay unchanged.

The [latest audit](scenes/so101-pickup/evidence/calibration/offline-audit.json)
rejected every candidate. Clipping can be eliminated without recovering valid
geometry; the current runtime calibration remains unverified.

```text
Recorded states -> joint map -> URDF geometry -> camera projection -> video comparison
Recorded actions -> joint map -> existing motion guard -> diagnostic command trace
Training labels -> constrained fit -> frozen candidate -> held-out evaluation
```

Run from `workers/isaac_sim` with Python 3.12:

```sh
python -m pip install -r calibration-offline.requirements.txt
```

Use the pinned dataset revision recorded in `scenes/so101-pickup/evidence/dataset.json`.
Keep its data Parquet, episode metadata Parquet, and first front-camera video locally.
The video contains four separate episodes; their boundaries must be preserved.

```sh
CAL_SCENE=scenes/so101-pickup
CAL_DATA=/absolute/path/to/so101-data.parquet
CAL_EPISODES=/absolute/path/to/so101-episodes.parquet
CAL_VIDEO=/absolute/path/to/so101-front.mp4
CAL_OUT=/absolute/path/to/calibration-results
```

Project recorded states onto episode 1 video using the current camera and map:

```sh
python -m sim_worker.calibration_fit.replay \
  --scene "$CAL_SCENE/scene.episode-001.usda" \
  --urdf "$CAL_SCENE/robot/source/so101_new_calib.urdf" \
  --calibration "$CAL_SCENE/calibration.experimental.yaml" \
  --observations "$CAL_SCENE/evidence/calibration/landmarks.json" \
  --data "$CAL_DATA" --episodes "$CAL_EPISODES" --video "$CAL_VIDEO" \
  --episode 1 --evaluation-episodes 1 3 --lag-frames -2.6733349882040764 \
  --joints shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll gripper \
  --output-dir "$CAL_OUT/replay"
```

The lag is the existing training-only timing estimate: video frame `i` uses state
`i + lag`. Initial frames lacking same-episode support receive no projection.
Requested and limit-clipped projections are reported separately. Neither includes
actuator dynamics or proves a grasp. Inspect `report.json` and the overlay video.

Fit camera, joint offsets, and gripper mapping without changing deployed inputs:

```sh
python -m sim_worker.calibration_fit.refit \
  --scene-dir "$CAL_SCENE" --dataset "$CAL_DATA" --output "$CAL_OUT/fit"
```

Image labels from episodes 0 and 2 fit the candidates; episodes 1 and 3 evaluate
them afterward. Selection uses training error only. Other episodes constrain
state/action support within the original URDF limits. Fixed-camera, free-camera,
wrist-sign, and bounded jaw-geometry alternatives remain separate diagnostics.
The exported candidate is unverified and requires its paired camera. A low robot
pixel error cannot establish correct shelf geometry, joint signs, or physical scale.

Audit recorded actions through the existing map and motion guard:

```sh
python -m sim_worker.calibration_fit.action_replay \
  --dataset "$CAL_DATA" --provenance "$CAL_SCENE/evidence/dataset.json" \
  --scene "$CAL_SCENE/scene.episode-001.usda" \
  --calibration "$CAL_SCENE/calibration.experimental.yaml" \
  --episode 1 --output "$CAL_OUT/actions"
```

Omit `--episode` to audit all 30 episodes. Source hashes and episode boundaries
are checked. Each guard call uses that frame's recorded state. The resulting
`guarded_target_rad` trace is **not** a schedule for execution.

`RecordedActions` in `sim_worker/calibration_fit/recorded_policy.py` implements
the existing `Policy` interface for a future recorded-action run. It copies one
episode's actions, bounds each chunk, and stops at its end. `Rollout` then applies
the existing guard against live simulator positions. Keep the source ID aligned
with the rollout manifest, initialize the reviewed recorded starting pose, and
limit the rollout to the recorded episode length. This adapter does not launch
Isaac; actual drive tracking and contact validation remain pending.

Offline checks:

```sh
python -m unittest discover -s tests -p 'test_calibration_*.py'
python -m unittest discover -s tests -p test_action_replay.py
python -m unittest discover -s tests -p test_recorded_policy.py
```

## Camera decision and completed trial

Keep the default shelf-fit camera in `scene.usda`. Its shelf-reference RMS is
**6.45 px at 640×360**, versus **61.00 px** for the rejected trial camera.
The paired trial camera/map improves held-out robot RMS from 25.86 px (clipped)
to 13.06 px **at 1920×1080**, still above the 9 px acceptance gate. Its maximum
error is 46.89 px against a 24 px gate; focal length reached the fit bound.
Shelf dimensions remain estimated, and neither camera is measured calibration.

The [historical scenario-1 record](scenes/so101-pickup/evidence/calibration/scenario-001-smolvla.json)
retains the paired input hashes and outcome: one 10-second SmolVLA run, no lift
or placement, 43 shoulder clips, and unavailable contact diagnostics. Both GPU
workers were deleted. The candidate camera and map remain unverified and are
not defaults. Future fitting should constrain shelf, cup, and robot together.

The current 5-second configuration is for future runs; it does not alter this
10-second result. No additional simulation has run.
