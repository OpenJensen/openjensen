# Simulator AI teleoperation protocol

This protocol adds bounded simulation actions to an active AI teleoperation demonstration.
The Agent Control service owns provider calls and approval; the simulator owner
owns capture, admission, motion limiting and actual-step evidence. Physical robot
execution is not advertised by this adapter.

All routes share Teaching's loopback bearer-token boundary. A browser and model
never receive that token. The existing manual and voice routes keep their typed
commands. A manual command revokes agent ownership; Stop/pause and a new explicit
start are required before accepting a new agent controller.

## Acquire, observe, execute, release

`GET /state` advertises `agent_capabilities` schema 2, including
`named_joint_correction`, `named_joint_segment`, `epoch_heartbeat`,
`max_segment_ticks:60`, a rate-bound `max_segment_duration_ms`, `stepped_capture`
and `physical_execution:false`. Legacy schema 1 clients remain supported.

POST `/agent/acquire`, `/agent/observe`, `/agent/execute`, `/agent/segment`,
`/agent/heartbeat` or `/agent/release` with:

```json
{
  "command_id": "unique_request_identity",
  "session_id": "selected_executor_session",
  "agent_session_id": "selected_agent_session",
  "authority_epoch": 0,
  "arguments": {}
}
```

Acquire arguments are `clock_policy` (`stepped` or `continuous`),
`expected_revision` and `episode_id` from the selected current state. Acquisition
requires epoch zero and an active running demonstration. Its receipt returns the
new nonzero epoch and owner. Later requests must use that exact owner and epoch.
Observe, heartbeat and release arguments are empty objects. Optional acquire
`max_session_seconds` is an integer from 1 to 600 (default 30), defining a fixed
arm deadline. The 30-second lease renews only through an explicit epoch-bound
heartbeat, capped at that arm deadline. Inference and observation requests do
not renew it. Expiry revokes authority and pauses simulation; a new explicit
acquisition is required. The local supervisor stops renewal at its run budget.
Optional `planning_snapshot_seconds` is an integer from 1 to 120 (default 30),
permitted only in stepped mode.

Observe captures fresh image pixels and measured state on the simulator owner
thread. It returns immutable observation identity, SHA256, original revision,
episode, scene and embodiment identities, actual named joint limits and radians, clock
policy, simulated step/time and a PNG data URL. `observation_age_ms` on the receipt
is the current executor-measured age; absolute executor timestamps must not be
compared with another host's clock. The observation hash covers all observation
fields except `observation_sha256` itself.

Execute arguments are `observation_id`, `observation_sha256`, `joint`,
`delta_rad`, `proposal_id` and `provider_model`. The executor resolves its retained
original observation. It validates the owner, epoch, episode, revision, deadline,
finite nonzero correction no larger than 0.15 rad, actual joint name and measured
state drift no larger than 0.03 rad. A stepped observation retains its original
capture age, is valid for the explicit planning snapshot interval inside the arm
deadline, and requires the original frozen step. Its `frozen_step_v1` policy is
hash-bound; native clock/state are recaptured without stepping at admission.
A continuous observation is
valid for at most two seconds; its step may advance without a command revision.
These are simulator admission limits, not hardware tracking guarantees.

Only one motion may be in flight. Execute permits one physics tick through the
existing MotionGuard. Receipt acknowledgment follows that actual tick and carries
requested/applied targets, next measured state and clipping flags. Another tick
requires another admitted action in stepped mode.

Segment uses the same observation/proposal/model bindings, plus nonempty named
`targets` in native radians and `duration_seconds` from 0.1 to 2. Each absolute
target must stay within 0.15 radians of both the original and current measured
state, with at most 60 local ticks. A named zero-delta target is a bounded hold.
MotionGuard applies every tick; the final acknowledgment aggregates clipping and
includes requested/applied targets, measured result, executed ticks and simulated
duration. Release or manual takeover can interrupt the next tick; stopped partial
execution evidence is retained. Completion never verifies the general task goal.

Optional operator `task_evaluation` settings select the native dynamic object
probe and exact geometric criterion. Its measurements, criteria hash and source
`isaac_object_bounds_v1` are inside the observation hash. Reobserving the same
frozen step neither accumulates hold time nor permits unowned object changes.
Geometric success is separate from model assessment, verified grasp/release and
physical transfer. See [native acceptance](NATIVE_ACCEPTANCE.md).

Poll `GET /agent/executions/{command_id}`; do not invent another command after a
transport timeout. Duplicate identity and identical canonical operation/payload
return the existing receipt; changed content is rejected. The replay cache belongs
to one executor process. A restarted executor has a new session ID, no agent owner
and no authority to resume an old action. Agent Control persists the higher-level
execution identity and marks interrupted or unresolved sessions explicitly.

## Capture provenance

New journals use capture schema 2. Every motion frame and command event explicitly
names `actor` and `environment_kind:simulation`. Agent corrections use
`agent_command` / `agent_joint_position`, retain session/proposal/execution and
observation identities, provider model, epoch, clock policy and skill-manifest
version. Human commands retain their separate source strings and actor.

The converter accepts the previous schema 1 human contract and strictly validates
schema 2 actor/source agreement. It still writes the actual MotionGuard-applied
target as the action column, separately preserves requested targets and measured
next state, and keeps task success unverified. Agent provenance survives conversion
in the demonstrations metadata. Captures still require explicit episode review,
immutable preparation and independent lineage for a held-out training split.
