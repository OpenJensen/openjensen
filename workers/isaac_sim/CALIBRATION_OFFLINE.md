# Run offline calibration

From `workers/isaac_sim`, install the CPU calibration tools with Python 3.12:

```sh
python -m pip install -r calibration-offline.requirements.txt
```

Use the dataset revision in `scenes/so101-pickup/evidence/dataset.json`. Keep its
state Parquet, episode metadata Parquet and front-camera video locally. Set:

```sh
CAL_SCENE=scenes/so101-pickup
CAL_DATA=/absolute/path/to/so101-data.parquet
CAL_EPISODES=/absolute/path/to/so101-episodes.parquet
CAL_VIDEO=/absolute/path/to/so101-front.mp4
CAL_OUT=/absolute/path/to/calibration-results
```

## Project recorded states

Run the episode-1 projection with the paired scene, calibration and landmarks:

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

Video frame `i` uses state `i + lag`; keep episode boundaries intact. Inspect
`report.json` and the overlay video in the new replay output directory.

## Fit a candidate

```sh
python -m sim_worker.calibration_fit.refit \
  --scene-dir "$CAL_SCENE" --dataset "$CAL_DATA" --output "$CAL_OUT/fit"
```

Use episodes 0 and 2 for fit labels and episodes 1 and 3 for held-out checks.
Keep each candidate paired with its camera; retain the original URDF limits and
review geometry, signs and scale before selecting a replacement runtime map.

## Audit recorded actions

```sh
python -m sim_worker.calibration_fit.action_replay \
  --dataset "$CAL_DATA" --provenance "$CAL_SCENE/evidence/dataset.json" \
  --scene "$CAL_SCENE/scene.episode-001.usda" \
  --calibration "$CAL_SCENE/calibration.experimental.yaml" \
  --episode 1 --output "$CAL_OUT/actions"
```

Omit `--episode` to read every source episode. Inspect the new action report and
`guarded_target_rad` trace. Keep the trace as an offline output, separate from
runtime command inputs.

## Check the commands

```sh
python -m unittest discover -s tests -p 'test_calibration_*.py'
python -m unittest discover -s tests -p test_action_replay.py
python -m unittest discover -s tests -p test_recorded_policy.py
```
