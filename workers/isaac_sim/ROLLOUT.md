# Run the remote policy adapter

Prepare [the SkyPilot Job Group](../skypilot/ROLLOUT.md) or run the policy server
and Isaac worker separately below. Use Isaac **6.1.0**, a separate Python **3.12**
policy environment, and a complete ACT or SmolVLA export with saved processors.

## Calibration and inputs

Copy `scenes/so101-pickup/rollout.example.yaml` to `rollout.local.yaml` beside it.
Set scene/calibration paths, policy endpoint/model ID, camera, joint order,
control FPS, total steps and execution horizon. Match image dimensions and
state/action dimensions to the checkpoint.

Supply joint calibration arrays `sim_rad` and `policy`. Make `sim_rad` strictly
increasing and `policy` strictly monotonic. Use two points for a linear map or
more points for a piecewise-linear map. Cover all requested state/action ranges
with the selected map and retain the URDF limits. Prepare it using
[offline calibration](CALIBRATION_OFFLINE.md), or supply
[a simulator control contract](CONTROL_CONTRACT.md).

## Policy server

Install `rollout-server.requirements.txt` in the isolated policy environment.
Inspect a complete export to obtain its model fingerprint, keeping all saved
processors/statistics together. Serve it on the configured private VPC:

```bash
python -m sim_worker.rollout.server \
  --host 0.0.0.0 --port 8080 --backend lerobot --device cuda \
  --checkpoint /path/to/export --model-id sha256:CHECKPOINT_FINGERPRINT \
  --state-dim 6 --camera-key observation.images.front --action-steps 1
```

Replace the checkpoint/model ID and use its state dimension, camera key and
execution horizon. For packed ACT CPU serving use [the packed recipe](PACKED_ACT.md).

| Endpoint | Request |
| --- | --- |
| `GET /health` | Read model identity and readiness. |
| `POST /reset` | Supply a fresh episode ID to clear policy state. |
| `POST /predict` | Send the episode, step, task, finite state and RGB8/base64 image; read the returned action chunk. |

Keep port 8080 inside the private network or bind it to loopback for local use.

## Run an episode

Validate the prepared manifest with the Python 3.12 worker environment:

```sh
python -m sim_worker.rollout --manifest scenes/so101-pickup/rollout.local.yaml --validate-only
```

Run in the Isaac container:

```bash
/isaac-sim/python.sh --no-ros-env -m sim_worker.rollout \
  --manifest scenes/so101-pickup/rollout.local.yaml \
  --output-dir /outputs
```

Set `POLICY_ENDPOINT` to the discovered server endpoint when using cloud
provisioning. Inspect `result.json`, `trajectory.jsonl`, `video.mp4` and
`final.ppm` in the output directory. Keep partial outputs on failures. Supply
[explicit object criteria](EVALUATION.md) before running an evaluated episode.

## Local checks

From `workers/isaac_sim` in the Python 3.12 environment:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

To run the simulator transport probe in the Isaac container:

```bash
/isaac-sim/python.sh --no-ros-env tests/gpu_rollout.py --output-dir /probe-output
```
