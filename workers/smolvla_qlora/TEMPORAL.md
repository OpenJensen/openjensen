# Configure training horizons

Place these fields inside `training` for an ACT or SmolVLA recipe:

```json
{"prediction_horizon": 8, "execution_horizon": 3, "observation_history": 1, "frame_stride": 1}
```

Use integer horizons from 1 to 1024 with execution no greater than prediction,
`observation_history: 1` and `frame_stride: 1`. Prediction defaults to 100 for ACT
and 50 for SmolVLA; omitted execution uses the prediction horizon. Choose either
these independent fields or the legacy `chunk_size` field.

Resume with the saved recipe and policy configuration. Preserve
`temporal-contract.json` with the checkpoint, including its dataset FPS and
sampling indices.

## Run the native test

In the pinned LeRobot 0.6.2 CPU environment, enable
`FIREBIRD_TEST_NATIVE_TEMPORAL=1` and run `tests/test_temporal_native.py` offline.
For export, packing and HTTP checks, use the
[ACT fixture commands](../act_optimizer/README.md#verify-independent-prediction-and-execution-horizons).
