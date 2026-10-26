# Agent Control

Agent Control adds a planning session above the Teaching executor. It retains an observation, asks a provider for one named joint correction, validates the original observation and authority, and saves the proposal and receipt. The model has no serial, shell, arbitrary HTTP, or physical robot tool.

The first action corrects one named simulated joint, in radians, with an absolute delta no greater than `0.15`. It does not implement Cartesian motion, object localization, grasping, collision avoidance, or general manipulation. Teaching advice, voice tools, and Decision Lab retain their existing roles.

## Try replay without a provider account

1. Select a project and open **Teaching → Agent Control**.
2. Choose the built-in `mock` profile and **Replay observation**.
3. Leave **Saved observation ID** blank to use the labeled synthetic six-joint fixture.
4. Enter `shoulder_pan +0.03 rad`, choose **Supervised acceptance**, and create a session.
5. Select **Plan one correction**, inspect the named joint, units, observation and expiry, then choose **Accept this proposal**.

The receipt says `replayed`. No simulator or robot is dispatched, no applied action label is generated, and task success remains unverified. The mock accepts one explicit relative instruction such as `shoulder_pan by 0.03 radians`. Vague or ambiguous tasks complete with a request for a precise instruction.

A saved observation ID can instead reference an immutable observation retained in the same project, including captured simulator evidence. Replay does not refresh the original pixels, change the source snapshot, or turn a synthetic fixture into recorded evidence. Its displayed admission time and validity window describe the replay turn.

## Configure OpenAI Responses

Open **Settings → AI providers**, create a profile, select **OpenAI Responses**, and enter its model ID and API key. `gpt-6-astra` is the initial model selection; another model must support the same admitted image and strict function protocol. The password field clears on save. Credentials remain in the private server profile store and are never returned in profile status, session evidence, or browser storage.

Saving a key does not verify access. Explicitly allow the capability check and choose **Check** for that profile. The OpenAI check sends a small synthetic image, a strict tool request and a synthetic tool result. These are paid provider requests. A successful check establishes the selected transport's image/tool continuation behavior; target readiness and motion admission remain separate. Changing a profile invalidates its checked status and existing session binding.

Before creating a cloud session, review its model, target, mode, call, time and token limits, then explicitly consent to sending the task, measured joints and any admitted camera image to OpenAI. Images remain associated with their original observation hashes. Each turn reserves an input estimate and maximum output before dispatch and records reported token usage. Failed or interrupted cloud calls can still incur charges; token limits are not a dollar quotation.

The adapter follows the [Responses function-calling protocol](https://developers.openai.com/api/docs/guides/function-calling), with strict argument schemas, `parallel_tool_calls=false` and `store=false`. Tool-result continuation preserves the original call ID and provider context privately. No provider fallback happens automatically. Mock and OpenAI Responses transports are included; OpenRouter and local-model transports are not exposed.

## Use a managed Teaching simulator

Start a managed Teaching session in the same project using a configured Teaching profile. Set a task and start recording so the executor has a running demonstration. In Agent Control, choose **Teaching simulation**, enter that managed Teaching job ID, and choose a clock before creating the agent session. A manually configured external executor is not an agent target.

- **Stepped** acquires fresh pixels and measured state on the simulator owner thread, freezes physics during planning, and permits one simulator step for an admitted correction. Planning wall time is separate from simulation time.
- **Continuous** keeps the scene running and admits a correction only while its original observation remains fresh and measured state drift is acceptable.

The executor owns a 30-second controller lease. Stepped observations have at most 30 seconds of validity; continuous observations have at most 2 seconds. Transfer age is deducted. Execution rechecks the executor's local deadline, owner epoch, episode, command revision, embodiment and measured state. Simulation steps may advance in continuous mode. An expired lease requires a new explicit agent session; the model cannot renew it or enlarge the limits.

**Capture fresh observation** invalidates a previous pending proposal. Acceptance identifies the exact proposal and authority epoch. A completed simulator receipt reflects an actual simulator step, including the limited applied target. It does not prove the requested angle or task goal was reached.

## Modes, interruption and evidence

**Advice** and **Shadow proposals** retain proposals without applying motion. **Supervised acceptance** requires separate acceptance of each valid proposal. **Bounded execution** can execute one admitted correction per planning turn; it does not start an unrestricted autonomous loop. Use **Plan one correction** for each turn.

**Cancel inference**, **Pause planning**, and **Stop agent session** revoke the agent context without waiting for model cooperation. Releasing simulator ownership pauses the target; a new agent session is required to reacquire it. Manual Teaching commands take over ownership and invalidate old agent proposals. Keep the parent Teaching controls available, and use its stop/publish workflow to finish and review the recording.

Sessions, immutable observations, proposals, receipts and ordered events are saved separately. Reusing a planning request ID cannot launch a second inference. A lost execution acknowledgement is reconciled by its existing command identity, never by dispatching replacement motion. Unresolved dispatched outcomes are labeled `unknown`. Restart interrupts prior sessions and invalidates their authority; it does not resume a natural-language task or replay a motion.

Version 2 Teaching captures preserve the `llm_agent` actor, simulation environment, agent/proposal/execution identity and action provenance. Version 1 human captures retain their prior parser. Conversion retains the executor's applied action separately from the requested target. Existing episode review, immutable snapshot and independent lineage requirements still apply before training.

## Acceptance boundaries

Core and worker tests cover malformed or stale context, duplicate identities, interruption, restart recovery, provider continuation, isolated process cancellation and mocked simulator execution. Browser fixtures cover exact proposal acceptance, stale epochs, private settings and truthful replay presentation. These tests do not establish live model access, Isaac runtime acceptance, physical calibration, collision safety, verified task success, or trained-policy quality. Those require separate measured experiments.

Physical SO101 execution is not enabled. The companion needs a reviewed lease-bound segment capability, synchronized observations, calibrated limits, measured tracking and interruption acceptance before a physical Agent Control panel or model execution tool can be offered. Existing Robot Lab Stop semantics remain torque-off; Agent Control adds no physical hold capability.
