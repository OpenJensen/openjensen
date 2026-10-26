# Agent Control planner worker

This isolated, small planning runtime proposes one named simulated joint correction per turn.
It has no actuator, serial-port, shell, arbitrary HTTP, or physical robot tool. Local core and
executor admission remain authoritative. The deterministic mock only proposes an explicit
relative instruction such as `shoulder_pan +0.03 rad`; vague manipulation tasks request help.

The OpenAI adapter uses the Responses API with strict function schemas, inline admitted image
input, `parallel_tool_calls=false` and `store=false`. Provider output items, including opaque
reasoning context, are preserved privately for tool-result continuation with the original
`call_id`. A later observation is supplied explicitly with the receipt. Model identity, token
usage, deadline, maximum output, and returned token budget are validated. No automatic fallback
changes providers. A capability check uses a synthetic image and a synthetic receipt; it proves
only the selected transport's image/tool protocol, never access to a robot or motion readiness.

## Process protocol

Run `python -m jensen_agent_control` with this directory's `src` on `PYTHONPATH`, or install the
package in an isolated environment. Send one JSON object on stdin; receive one JSON object on
stdout. A sanitized error exits with status 2. The caller owns and cancels the process.

Requests accept `operation` (`plan`, `continue`, `check`), `provider` (`mock`,
`openai_responses`), `model`, `task`, `observation`, `joint_names`, `reasoning_effort`,
`max_output_tokens`, `max_total_tokens`, and `timeout_seconds`. The host creates observations;
only an admitted `observation.image_data_url` in PNG or JPEG format becomes image input.
Credentials are accepted only from `OPENJENSEN_AGENT_API_KEY` in the process environment.
The provider endpoint is fixed to the OpenAI Responses API.

A proposed result includes `arguments` (`joint`, `delta_rad`), `call_id`, actual `model`,
`usage`, and private `continuation`. A complete result includes a compact `summary`. Neither
is an execution receipt or verified task outcome. Send `operation=continue`, the original
`continuation`, the host's `receipt`, and the next admitted observation for another turn.
Continuation contains observation and provider context and must never be returned through a
public API or shown as an explanation. The host must bind it to its original session/profile.

The overall timeout includes a capability check's two requests. The caller must reserve the
input and output budget before launching each turn and account for failed calls; returned
usage checks cannot prevent charges already incurred. Capability checks are paid requests for
the OpenAI transport. Mock checks have no network, usage, or image capability.

## Tests

`python -m unittest discover -s workers/agent_control/tests -v` exercises HTTP-mocked
Responses transport, strict action parsing, image input, private continuation, model identity,
replayed call IDs, bounded cancellation, budget errors, and the JSON process boundary. No test
calls a paid provider or touches an executor.

References: [Function calling](https://developers.openai.com/api/docs/guides/function-calling),
[GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra).
