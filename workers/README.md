# Native worker boundary

Workers execute native operations; the Python application owns project/job metadata. Request and result schemas currently live in `vla_platform.contracts`. Protocol version 1 uses fixed operation names, validated JSON request/result files, and registered subprocess entry points. A worker never receives an arbitrary shell command from a dataset.

Only metadata inspection is implemented, via `python -m vla_platform.datasets.worker REQUEST RESULT`. It uses the small application environment and does not import Torch. It writes a result atomically and leaves state transitions to the supervisor. Requests/results and bounded errors survive under the application workspace's job directory. Intake has a 90-second job deadline; cancellation terminates the process before returning.

LeRobot and native OpenVLA-OFT are reserved isolated environments. Their candidate manifests are **not resolved locks or tested capability claims**. Resolve and test each through its task before exposing any operation. They must not be added as core dependencies. Simulator drivers and SkyPilot use their supported separate runtime environments.

## Isaac simulation

[`isaac_sim`](isaac_sim/README.md) contains the YAML-driven Isaac worker;
[`skypilot`](skypilot/README.md) launches it on GCP L4 VMs. It records MP4s and result manifests in
GCS. Its latest GPU recording failed visual acceptance with black frames;
the demo now rejects entirely black output. Rendering acceptance remains open.

Native projects under `workers/*` are excluded from the application uv workspace.
Use the worker's separate environment. This transfer adds no application API,
CLI operation, or advertised simulation capability.
