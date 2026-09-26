# Dataset calibration

The SO101 candidate remains **unverified**. The rollout manifest still selects
the unverified example, so learned control stays disabled.

The offline fitter uses URDF forward kinematics and manual video landmarks to
estimate camera pose, arm offsets and an affine gripper map. It fixes pan offset
to zero as a camera-pose convention; that zero is not an encoder measurement.

## Recorded-data check

The ACT checkpoint statistics match all 4,500 joint samples in the pinned dataset.
Episodes 0 and 2 supplied 50 fit landmarks; episodes 1 and 3 supplied 51 held-out
landmarks. Parameters and timing were selected by fit residuals. Same-video static
background checks support a shared camera. The last video has a shifted camera
and was excluded from these metrics.

The final fit estimates video/state alignment at -2.673 frames (-89.1 ms): video
frame `i` corresponds approximately to interpolated state `i - 2.673`.

| Check | Result | Acceptance |
|---|---:|---:|
| Held-out reprojection RMS, 1920x1080 | 5.86 px | <=9 px |
| Held-out maximum | 12.21 px | <=24 px |
| Shoulder offset standard error | 2.27 degrees | <=2 degrees |
| Maximum gripper angle standard error | 4.67 degrees | <=2 degrees |
| Mapped shoulder action minimum | -112.61 degrees | URDF minimum -100 degrees |
| Mapped wrist-roll action maximum | 201.47 degrees | URDF maximum 162.79 degrees |

Low image error does not resolve the angle uncertainty or model-limit conflicts.
The covariance assumes exact CAD landmarks; unmeasured jaw covers and manual
annotation bias add uncertainty. All positive joint signs fit best among tested
alternatives, but have no independent encoder reference. Annotated gripper poses
also cover less of its range than the full dataset.

The fit is useful for further investigation, not for enabling ACT control.
Keep the robot model limits and verification gate intact until their conflicts
are resolved. More rigid gripper landmarks and a fresh episode holdout would
strengthen a dataset-only fit. Recording-time calibration or measured geometry
would provide the missing independent reference.

## Files

- `scenes/so101-pickup/calibration.candidate.yaml`: rejected diagnostic map.
- `scenes/so101-pickup/evidence/calibration/report.json`: parameters, covariance,
  thresholds, full support and acceptance failures.
- `scenes/so101-pickup/evidence/calibration/landmarks.json`: original annotations.
- `scenes/so101-pickup/evidence/calibration/aligned-observations.json`: observations
  paired using the estimated timing offset.

The task's `outputs/calibration/` directory retains source provenance, timing and
sign experiments, scripts, original frames and final projection overlays.

## Offline checks

Install `calibration.requirements.txt` in a separate Python environment. From
this directory:

```bash
PYTHONPATH=. python -m unittest discover -s tests -p 'test_calibration*.py' -v
```

`CalibrationFit.fit()` keeps fit and validation observations separate and reports
conditional uncertainty. `validation.evaluate()` checks acceptance without
writing calibration files. It requires separate geometry/sign provenance and
checks the full observed state/action range against URDF limits. It never changes
the rollout manifest. Synthetic recovery, independent FK, held-out errors,
rank deficiency and rejection checks passed in 21 tests.
