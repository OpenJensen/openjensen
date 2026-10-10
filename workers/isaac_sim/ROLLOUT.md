# Remote policy adapter

For explicit object-outcome criteria and replay, see [simulation evaluation](EVALUATION.md).

The rollout worker runs Isaac on L4 and a separate ACT/SmolVLA policy server on
H100. [SkyPilot launch instructions](../skypilot/ROLLOUT.md) create one Job Group
with two GPU tasks and a small CPU controller.

```text
CLI → rollout service → simulation interface → Isaac SDK
                     → policy interface     → HTTP → LeRobot
```

The existing recording command is unchanged. Run a policy episode with:

```bash
/isaac-sim/python.sh --no-ros-env -m sim_worker.rollout \
  --manifest scenes/so101-pickup/rollout.local.yaml \
  --output-dir /outputs
```

Use `--validate-only` outside Isaac to check the manifest and calibration without
starting the SDK. `POLICY_ENDPOINT` overrides the manifest endpoint after cloud
discovery. The server has a bounded readiness wait; each inference request has
its own timeout. Neither wait advances simulation time.

## Calibration and inputs

The provided calibration is deliberately `unverified` and cannot control the
robot. Supply recording-time calibration or a separately validated fit. Merely
changing its status does not establish calibration.

Each joint has paired `sim_rad` and `policy` arrays. Simulator coordinates must
increase strictly; policy coordinates must be strictly monotonic in either
direction. Two points define a linear conversion; more points describe a
piecewise-linear gripper or joint mapping. The same curve maps observations and
actions in opposite directions. Values outside the calibrated range fail; they
are never silently clamped. Joint order follows the manifest, not tensor order.

The SO101 export uses one front RGB camera and six measured joint positions.
The supplied ACT step-29000 export expects 640×360 RGB, six actions, and chunks
of up to 100 actions. The local manifest executes one action before replanning.
Calibration is not included in that export; saved normalization statistics do
not replace motor calibration. The inspected dataset also contains shoulder
targets beyond the current URDF's limits.

The original motor calibration JSON is optional for simulation: recorded values
already passed through the hardware's calibration. What Isaac needs is the
dataset-to-URDF coordinate map. Without that file, start from the manufacturer's
joint convention, fit offsets and gripper scaling against several recorded poses,
then validate on other frames. Camera placement must be checked separately.
This fitting workflow is not automated by the adapter. An assumed mapping is not
a verified mapping; retain the validation gate until replay establishes it.

Checkpoint inspection computes a fingerprint over model weights, config and
saved processors/statistics. The server checks this fingerprint at startup and
the client checks it in responses. ACT and SmolVLA use their saved processors;
the adapter does not invent normalization or reuse another model's statistics.
Input image dimensions must match the checkpoint.

## Execution and failure behavior

`Simulation.reset/observe/apply/step` contains all Isaac interaction. `Policy`
contains reset and prediction. The rollout service owns episode IDs, simulation
ticks and action selection; transport objects never reach the Isaac driver.

Only `step()` advances physics. Rendering checks that time did not advance.
Camera and measured joints share one simulation instant. Commands use articulation
position targets in radians. Reset reloads the authored stage, restoring robot
and object state. The inference server clears its policy history on reset.

The service validates a selected action chunk before applying any of it. It
rejects wrong models, old episodes, stale step numbers, wrong dimensions,
nonfinite values and out-of-range targets. A request failure ends the episode
without another physics step. V1 intentionally has no automatic resumption or
asynchronous action queue after Spot preemption.

`result.json` records completion, model/calibration/manifest identities and final
joint state. `trajectory.jsonl` stores observations, actions, resulting measured
states, simulation times and request latency. `video.mp4` shows pre-action frames;
`final.ppm` preserves the terminal view. On failure, the partial trajectory and
diagnostics remain; the existing recorder removes incomplete MP4 output. Results
are committed before Kit closes, then the SkyPilot host uploads them to GCS.

A successful rollout means the control loop executed. Pickup success still needs
a task-specific object-placement check and evaluation across starting conditions.

## Policy server

Install `rollout-server.requirements.txt` in a separate Python 3.12 environment.
Extract the checkpoint archive, preserving processor statistics files, then run:

```bash
python -m sim_worker.rollout.server \
  --host 0.0.0.0 --port 8080 --backend lerobot --device cuda \
  --checkpoint /path/to/export --model-id sha256:CHECKPOINT_FINGERPRINT \
  --state-dim 6 --camera-key observation.images.front --action-steps 1
```

The private HTTP API has three endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Readiness and model identity |
| `POST /reset` | Start a fresh episode; clear policy state |
| `POST /predict` | RGB8/base64 image, state, task, episode and step → action chunk |

The server serializes backend calls and bounds request sizes. Keep it inside the
configured private VPC; it provides no public authentication endpoint. The Isaac
client uses only standard-library transport and does not install LeRobot into Kit.

## Tests

From `workers/isaac_sim`, use the worker's Python 3.12 environment:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

Local tests exercise actual HTTP transport with a mock backend plus simulator
fakes. The GPU probe uses a synthetic identity mapping and a small joint nudge;
it does not claim dataset calibration or trained-policy success:

```bash
/isaac-sim/python.sh --no-ros-env tests/gpu_rollout.py --output-dir /probe-output
```

The probe checks visible RGB, observation clock stability, fixed-step physics,
measured joint motion, and reset accuracy. The regular recording regression tests
remain part of the suite.

Learned control requires validated dataset-to-URDF calibration. A readiness
response or synthetic joint-nudge probe does not qualify trained-policy quality.
