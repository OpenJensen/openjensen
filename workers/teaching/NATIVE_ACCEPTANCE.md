# Native AI teleoperation acceptance

The finite runner uses the same owner-thread `Session`, `AgentAuthority`,
`MotionGuard`, original observation hash, command mailbox, native actuator and
capture journal as the application. It does not replace Isaac with a mock.
Its fixture tests verify the protocol only. A live native run must be scheduled
through the operator's compute scheduler, with a new private output directory.
Keep results and local machine details outside the repository.

## Prepare an eligible runtime

Use the pinned Linux Isaac Sim **6.1.0.0** runtime and Python **3.12**. NVIDIA's
[requirements](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html)
list 32 GB RAM, an RTX 4080, 16 GB VRAM and 50 GB storage as the minimum.
Allow additional storage for source assets, caches, captures and the runtime;
keep at least 10 GiB unused throughout. The runner blocks when the documented
RAM/VRAM floor is absent. GPU memory alone does not establish compatibility:
verify RT-core support and the full vendor driver/platform requirements too.

Supported package sources are the NVIDIA
[Python installation](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_python.html)
(`isaacsim[all,extscache]==6.1.0.0`, NVIDIA package index) and the
[container](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_container.html)
`nvcr.io/nvidia/isaac-sim:6.1.0`. Accepting NVIDIA's license is an operator action.
The runner never sets license-acceptance variables and requires explicitly
provided `ACCEPT_EULA=Y` and `OMNI_KIT_ACCEPT_EULA=YES` before native startup.
Do not add these values to an automatic install or source-controlled job.

Copy [settings.example.json](settings.example.json) and
[agent-acceptance.example.json](agent-acceptance.example.json) to an operator
configuration directory. Resolve the settings scene there and retain its USD
assets. Review the exact object prim, support height and criterion. The supplied
scene and task configuration are experimental and are not calibrated to physical
hardware. Optional settings `task_evaluation` uses the exact `evaluation` object
from the scenario; a mismatch is rejected.

From the checkout, set `PYTHONPATH=workers/teaching:workers/isaac_sim` and use the
runtime's Python. The preflight does not import or start Kit:

```sh
python -m jensen_teaching.agent_acceptance --settings /operator/teaching.json --scenario /operator/agent-acceptance.json --output /operator/acceptance/preflight --preflight-only
```

A missing runtime, license gate or memory prerequisite writes a blocked receipt
and exits 2. A ready preflight exits 0 while explicitly leaving native execution
unverified. It does not establish scene or task success.

## Run paired finite experiments

Run the declared scripted smoke baseline first, then the selected provider with
exactly the same scenario, seeds and settings, using distinct outputs:

```sh
python -m jensen_teaching.agent_acceptance --settings /operator/teaching.json --scenario /operator/agent-acceptance.json --output /operator/acceptance/scripted --provider scripted
python -m jensen_teaching.agent_acceptance --settings /operator/teaching.json --scenario /operator/agent-acceptance.json --output /operator/acceptance/astra --provider openai_responses --model gpt-6-astra --planner-python /operator/planner/bin/python --planner-source workers/agent_control/src
```

The scripted example is one bounded joint correction and then stops; it is a
transport/physics baseline, not a pickup policy. Replace its declared actions
only with a reviewed baseline. The provider runtime is isolated from Isaac and
receives no motor transport. Supply its credential through the established
private `OPENJENSEN_AGENT_API_KEY` environment, never in a scenario or command
argument. `--provider mock` checks the provider protocol without a network call;
it cannot demonstrate model capability.

Each seed creates a small USD override with deterministic, bounded object XY
jitter and restores the authored scene before the episode. Running both methods
with the same seeds pairs their initial conditions. Zero jitter is explicitly
reported as not providing independent initial conditions. Existing scene assets
are untouched. Scene loading has separate setup timing from the finite control
wall budget. Each episode also has step, turn and token bounds; output tokens
are reserved even if provider accounting is missing. Owned planner processes
are cancelled and reaped on timeout. Outer scheduler wall limits must still
bound startup, shutdown and the sum of all episodes.

For `stepped`, native physics stays frozen while planning. The observation
retains its original capture timestamp, hash, revision, step and epoch, with an
explicit `frozen_step_v1` policy for up to 120 seconds inside the arm budget.
Admission recaptures native state and verifies the clock without advancing it.
Executor-local epoch-bound heartbeats renew the 30-second lease only inside the
fixed session duration; they do not refresh old observations. For `continuous`,
local physics ticks while planning, and the original 2-second observation age
and measured drift limit still apply. Run both clocks separately and retain both
wall planning time and simulated time. A slow continuous proposal may correctly
be rejected as stale.

The planner receives actual named radian joint limits from the native backend.
A correction owns one tick. A named target segment owns 0.1–2 seconds and at
most 60 local ticks; each target stays within 0.15 radians of the original and
current measured state. The guard runs on every tick. A named zero-delta segment
provides a bounded hold. Acknowledgment occurs only after the final measured
tick. Stop, release or manual takeover interrupts the next tick and retains any
partial execution evidence. Retry by polling the original command ID.

## Interpret evidence

The native `ObjectProbe` samples the configured dynamic object on every owned
physics tick. Conservative object bounds and velocity supply the named
`lift_hold` or `lift_place` geometric criterion, independently of the model's
assessment. Repeated frozen observations do not accumulate hold time or samples;
an object change without a physics tick fails closed. The criteria and evidence
are inside the original observation hash. The generic pickup goal still needs
independent grasp/contact verification or explicit operator review.

`result.json` separates native execution, model assessment, geometric success,
and planning/simulation clocks. `summary.json` reports every seed, including
failed task outcomes. Exit 0 establishes that all requested episodes completed
some native executor actions; inspect `geometric_successes` separately. A model
`achieved` assessment never sets `task_success_verified`. Neither native geometry
nor successful protocol execution claims grasp-and-release or physical transfer.
Failed or unexecuted native episodes exit 3 and retain their failure details.

## Cloud preparation, without provisioning

If the available workstation does not meet the vendor requirements, prepare a
single-GPU SkyPilot task on an eligible RTX/RT-core accelerator, for example an
L40S with 48 GB VRAM, at least 64 GB RAM, 8 CPUs and a 150 GB disk. Select an
available region, image/driver combination, account quota, hourly price and
explicit spending/wall budget before launch. This document grants no paid-cloud
authorization and does not select a provider or accept a license.

Use the pinned image above, mount one checkout and the reviewed operator
configuration, keep credentials in private scheduler environment variables, and
export the worker `PYTHONPATH`. Run preflight, the scripted paired seeds, then
Astra for each clock in that order. Use one named job, an outer wall limit,
automatic idle shutdown, no exposed control ports, and persist only reviewed
capture/evidence outputs. Inspect the original job after a client timeout;
do not submit a replacement until it is terminal. Reuse the same runtime and
assets between methods. No model weights are required.
