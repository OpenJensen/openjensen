# Prepare dataset calibration

1. Install `calibration.requirements.txt` in a separate Python environment.
2. Prepare recorded video/state samples and separate fit and held-out observations
   using [the offline commands](CALIBRATION_OFFLINE.md).
3. Supply camera geometry, video/state alignment, ordered joint signs/offsets,
   gripper conversion, original URDF limits, and full state/action ranges.
4. Use `CalibrationFit.fit()` to fit a candidate and `validation.evaluate()` to
   check it against the held-out observations and geometry/sign provenance.
5. Keep the candidate separate from the rollout manifest until those checks pass;
   select the resulting calibration explicitly in [the rollout manifest](ROLLOUT.md).

## Check the fitter

From `workers/isaac_sim` in that environment:

```bash
PYTHONPATH=. python -m unittest discover -s tests -p 'test_calibration*.py' -v
```
