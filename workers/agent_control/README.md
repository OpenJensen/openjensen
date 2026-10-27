# Agent Control planner worker

This isolated runtime turns an operator task into typed proposals above a separately owned executor. It accepts abstract goals in task mode and selects only the host's admitted skills. The model has no serial port, actuator credential, shell, arbitrary HTTP or generated-code tool.

## Planning modes

- `joint`: one explicit named simulated joint correction in radians, at most 0.15.
- `task`: choose a named correction (simulation), coordinated joint segment, or `finish_task` assessment (`achieved`, `needs_help`, `cannot_proceed`). Physical targets expose segments only, in the declared degrees/percent units.

The host action-space manifest supplies exact joint names, units, calibrated limits, delta and duration bounds, plus the physical commanded speed/acceleration profile. The worker validates proposals against measured state before returning them; the local executor repeats admission at dispatch. The planner must ask for help when images, calibration or registered skills cannot ground an action. There is no Cartesian grasp, object-pose, inverse-kinematics, collision or policy-execution tool.

The deterministic mock parses explicit relative-radian instructions and cannot visually ground object manipulation. It does not call a provider or touch an executor.

## Provider and process protocol

Run `python -m jensen_agent_control` with this directory's `src` on `PYTHONPATH`, or install it in an isolated environment. Send one JSON object on stdin and receive one on stdout. A sanitized failure exits with status 2. The core owns and cancels the process.

Requests include `operation` (`plan`, `continue`, `check`), `provider` (`mock`, `openai_responses`), model, task, admitted observation, joint names, reasoning effort and token/time limits. Task mode additionally requires `planning_mode:task`, the exact `action_space` and completed-step count. Only host-admitted inline PNG/JPEG data becomes image input. Credentials arrive through `OPENJENSEN_AGENT_API_KEY`; the endpoint is fixed to OpenAI Responses.

The Responses adapter uses strict function schemas, `parallel_tool_calls=false` and `store=false`. It preserves opaque provider context privately for tool-result continuation with the original `call_id`. A new observation is supplied explicitly after the measured receipt. Old images are removed from bounded historical turns rather than relabeled as current. Model identity, usage, tool identity, deadlines and token budgets are validated; there is no automatic fallback.

A proposal returns skill, typed arguments, concise purpose, model, usage, call ID and private continuation. Decisions also retain `prompt_version` and the exact instruction UTF-8 `prompt_sha256`; the mock identifies its deterministic decision contract. An assessment or final response is not an execution receipt or independent task verification.

For `continue`, supply the original continuation, measured receipt and new admitted observation. The continuation is bound to model, embodiment and task action space, and must stay private. The core must reserve input/output capacity before each call and account for failures: returned usage cannot prevent charges already incurred.

## Capability checks and acceptance

An explicit OpenAI capability check uses a synthetic image and synthetic tool result. It is a paid transport test for image input and tool continuation, not motion readiness. The mock check does not claim vision or physical capability. Provider keys, configured status, capability checks and target readiness remain separate.

The tests use HTTP mocks and deterministic executors for strict tools, exact prompt fingerprints, image input, continuation, usage, cancellation and malformed responses. No test calls a paid provider or moves hardware.

See [AI teleoperation guide](../../docs/agent-control.md), [simulator protocol](../teaching/AGENT_CONTROL.md), [native simulator acceptance](../teaching/NATIVE_ACCEPTANCE.md), and [physical executor contract](../robot_lab/AI_CONTROL.md).
