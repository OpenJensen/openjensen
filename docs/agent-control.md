# AI teleoperation and Agent Control

Agent Control turns a task goal into a sequence of typed proposals above a local executor. The planner receives an admitted image, named measured joints, units, limits and previous execution receipts. The core retains authority and evidence; the simulator or SO101 companion admits and times each action. Teaching advice, voice tools and Decision Lab retain their separate roles.

Open **ROBOT LAB → AI teleoperation**. **Initial setup** contains physical setup, calibration and manual control. The historical Teaching route redirects to AI teleoperation; internal Teaching API and capture identifiers remain compatible.

## Goals, skills and modes

Use **Goal-directed task** to enter an intended outcome without supplying joint names. Astra can select a bounded named correction or coordinated joint segment and then inspect the new observation and measured execution result. **Start task** continues this loop within the displayed call, step, time and token limits. **Plan next action** requests one turn. The planner must return `needs_help` or `cannot_proceed` when evidence or a required skill is missing.

The current skill library provides simulation joint corrections and segments, plus supervised physical joint segments. It does not provide a calibrated Cartesian grasp, inverse kinematics, object localization, collision avoidance, contact-force control or a compatible trained-policy tool. Accepting “pick up the object” as a goal does not establish that the available skills can complete it. Object pickup needs grounded task-specific skills and separate acceptance.

**Single correction** retains the original explicit-relative-radian experiment for replay and simulation. **Advice** and **Shadow proposals** log proposals. **Supervised acceptance** requires approval of each exact proposal. **Bounded execution** runs admitted simulation actions within the selected budget. Physical SO101 sessions require Goal-directed task and Supervised acceptance; they do not offer autonomous execution.

A model assessment such as `achieved` is separate from task verification. **Review task outcome** records an operator's success, failure or unknown label against an exact retained observation. Further motion clears that review. Native simulator task evidence, when configured, remains distinct from physical grasp verification.

## Replay without a provider account

1. Select a project, choose the built-in `mock` profile and **Replay observation**.
2. Leave **Saved observation ID** blank for the labeled synthetic six-joint fixture.
3. Enter `shoulder_pan +0.03 rad`, choose **Single correction** and **Supervised acceptance**, then create a session.
4. Select **Plan one correction**, inspect the context and choose **Accept this proposal**.

The receipt says `replayed`: no actuator is dispatched and no applied training-action label is generated. The deterministic mock parses one explicit relative instruction; it cannot visually ground an abstract manipulation task. A saved observation ID may instead reference immutable evidence in the same project. Replay never refreshes the original pixels or converts a synthetic fixture into recorded evidence.

## Configure a cloud planner

In **Settings → AI providers**, save an **OpenAI Responses** profile with the selected model ID and private API key. `gpt-6-astra` is the initial selection. The password clears on save and credentials stay in the protected server profile store.

Saving a profile does not check access. Explicitly allow its **Check** operation: it sends a small synthetic image, strict tool request and synthetic tool result through the selected paid provider. The result establishes transport behavior, not robot readiness. Profile changes invalidate the checked status and existing session binding. Mock and OpenAI Responses are implemented; OpenRouter and local-model transports are not exposed.

Before a cloud session, review the target, mode and budgets, then consent to sending the task, measured joint state and admitted images to OpenAI. Each turn reserves input and output capacity and records reported usage. Failed or interrupted calls may still incur charges; token limits are not a dollar quotation. No provider fallback happens automatically.

The isolated worker uses the [Responses function-calling protocol](https://developers.openai.com/api/docs/guides/function-calling), strict schemas, `parallel_tool_calls=false` and `store=false`. Tool results retain their original call ID and private continuation. Each new decision identifies the instruction version and exact UTF-8 SHA256; the deterministic mock identifies its decision contract separately.

## Managed simulation

Start a managed simulation in the same project through AI teleoperation, set a task and start the demonstration. Select **Simulation** in Agent Control and enter its **Managed AI teleoperation job ID**. An arbitrary external executor is not an admitted target.

- **Freeze during planning, then step** captures fresh pixels and measured state on the simulator owner thread, holds physics during inference and advances only admitted local action ticks.
- **Continuously running scene** keeps physics running and rejects action proposals whose original context is too old or has drifted.

The local controller lease is renewed by an epoch-bound supervisor only within the session's fixed deadline. Goal planning in a frozen scene can use an explicitly bounded snapshot window up to 120 seconds; continuous observations remain limited to two seconds. Neither setting changes physical freshness. Planning wall time and simulated time are separate evidence.

Named corrections are at most 0.15 radians. Segments stay within the same measured-state delta envelope, use native radians and have at most 60 local ticks. The simulator's MotionGuard limits every tick. A receipt records requested and applied targets, clipping and measured result; it does not independently prove the task succeeded. See [simulator protocol](../workers/teaching/AGENT_CONTROL.md) and [native acceptance](../workers/teaching/NATIVE_ACCEPTANCE.md).

## Physical SO101 preflight and approval

Physical controls appear only when the connected companion advertises the governed agent capability.

1. Complete follower setup and calibration in **Initial setup**.
2. In AI teleoperation select **Physical SO101**, its USB port, **Calibrated follower** and **Physical camera**. Enable **Show the local camera preview**.
3. Confirm operator presence and a clear movement path, then select **Connect arm and hold current pose**. This explicitly enables torque in a full-arm manual hold.
4. Review the task, provider, budgets and cloud-image consent. **Arm AI session** transfers that existing controller to a lease-bound agent owner; it does not implicitly connect hardware.
5. Select **Start task** or **Plan next action**. Review each proposed target and its purpose against the separately identified **Current execution observation**.
6. Confirm that the current image, object, workspace and path remain suitable, then accept that exact proposal.

The original image supplied to the model stays immutable. Physical approval binds a distinct fresh execution observation, retains both identities and checks epoch, calibration, embodiment, units, image hash and state drift. Execution observations expire after one second; stale approval is rejected. The companion rechecks its own local deadline and measured state before dispatch.

Arm joints use degrees and the gripper uses percent of calibrated travel. The planner's envelope is at most 4 degrees or 8 gripper-percent per segment; the companion independently enforces its fixed limits and 0.5–5 second commanded position profile. Requested, transmitted and measured targets remain distinct. See [SO101 execution contract](../workers/robot_lab/AI_CONTROL.md) for speed, acceleration, tracking, observation and evidence bounds.

The local loop checks encoder/controller health and tracking while holding during cloud inference. **Cancel inference** and **Pause planning** stop planning and retain the supervised local hold while operator presence and session budget remain current; an already admitted segment finishes locally. **Stop agent session** and **Stop arm and disable torque** revoke control and disable torque. Support the arm and object before torque-off. Leaving or hiding the physical page, losing focus or withdrawing operator presence requests Stop; the independent local lease also expires.

Host camera read intervals do not establish camera exposure-time synchronization. Firmware watchdog behavior, actual physical speed/acceleration, contact behavior and interruption response still require controlled hardware acceptance. The executor does not claim collision safety.

## Recording and training

Simulation capture schema 2 records actor and environment separately, with agent/proposal/execution/observation IDs, provider model, prompt version/hash, clock policy and skill version. Schema 1 human captures retain their historical interpretation. Older schema 2 captures without prompt metadata remain readable; a present prompt identity must be a complete validated pair and match its original event.

The converter writes the executor's applied action separately from the requested target and measured next state. Provenance survives conversion and immutable snapshot validation. Review and select episodes explicitly, then use the pinned converter and immutable dataset workflow; independent lineage remains necessary for a held-out training split. A model saying “done” does not change a capture's verified task outcome.

Physical agent sessions retain observations, proposals, receipts and operator review as execution evidence. They are blocked from the legacy human recording path. Physical agent demonstration conversion and training require their own capture contract; the simulation converter does not accept degree/percent trajectories.

## Recovery and acceptance

Sessions, immutable observations, proposals, admissions, receipts and ordered events are durable. Duplicate request identities cannot launch a second inference or motion. A lost execution acknowledgement is reconciled by its existing command ID; an unresolved result is `unknown`, never blindly replayed. Restart interrupts old sessions and revokes their authority.

Automated tests exercise typed transport, freshness, ownership, duplicate identities, cancellation, late results, capture provenance and fake executors. They do not prove live Astra access, native Isaac behavior, physical calibration or object pickup. Use the [native acceptance procedure](../workers/teaching/NATIVE_ACCEPTANCE.md) for simulation and a controlled physical fixture for hardware.

For a later pickup acceptance, first provide the missing calibrated manipulation skill or policy and a visible object/workspace fixture. Use a concrete goal such as lifting the selected object a specified distance and holding it for a specified interval. Follow physical preflight above, inspect every fresh-image approval and measured receipt, and save an operator outcome against the final retained observation. If grounding or a required skill is absent, a `needs_help` assessment is the correct result; do not approve guessed joint targets.
