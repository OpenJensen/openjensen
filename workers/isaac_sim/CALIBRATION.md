# Dataset calibration

For CPU geometry fitting, projection and guarded action auditing, see
[offline calibration](CALIBRATION_OFFLINE.md).

The SO101 candidate remains **unverified**. The rollout manifest still selects
the unverified example, so learned control stays disabled.

The offline fitter uses URDF forward kinematics and manual video landmarks to
estimate camera pose, arm offsets and an affine gripper map. It fixes pan offset
to zero as a camera-pose convention; that zero is not an encoder measurement.

## Calibration requirements

The candidate mapping is a diagnostic input and must not enable learned control.
Verify camera pose and scale, video/state alignment, joint order/signs/offsets,
gripper mapping, the full state/action support and independent held-out episodes.
A low projection error alone does not resolve angle uncertainty or model-limit
conflicts. Manual landmarks and estimated geometry require independent checks.

Keep original URDF limits and the rollout verification gate intact. Recording-time
calibration or measured geometry is needed to establish physical correspondence.
The fitter writes candidates separately and never changes the rollout manifest.

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
rank deficiency and rejection are covered by the synthetic regression checks.
