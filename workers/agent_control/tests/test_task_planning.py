"""Abstract-task planning contracts with mocked HTTP, no model call or actuation."""

import copy
import json
import unittest

import httpx

from jensen_agent_control import OpenAIResponsesProvider, ProviderError, run
from jensen_agent_control.providers import CHECK_IMAGE


def simulation_space():
    return {
        "kind": "simulation",
        "joint_units": {"shoulder_pan": "rad", "gripper": "rad"},
        "joint_limits": {"shoulder_pan": [-0.5, 0.5], "gripper": [0, 0.2]},
        "max_delta": {"shoulder_pan": 0.15, "gripper": 0.1},
        "min_duration_seconds": 0.1,
        "max_duration_seconds": 2.0,
    }


def physical_space():
    return {
        "kind": "physical",
        "joint_units": {"shoulder_pan": "degrees", "gripper": "percent"},
        "joint_limits": {"shoulder_pan": [-10, 10], "gripper": [0, 100]},
        "max_delta": {"shoulder_pan": 2.5, "gripper": 4},
        "min_duration_seconds": 0.2,
        "max_duration_seconds": 5.0,
        "trajectory_profile": {
            "kind": "quintic-position",
            "max_speed_units_per_second": 2.0,
            "max_acceleration_units_per_second2": 3.0,
        },
    }


def request(**changes):
    return {
        "provider": "openai_responses",
        "planning_mode": "task",
        "task": "Pick up the object in front of the gripper.",
        "observation": {
            "observation_id": "observation_0",
            "joints": ["shoulder_pan", "gripper"],
            "state_rad": [0.0, 0.05],
            "image_data_url": CHECK_IMAGE,
            "units": "rad",
        },
        "joint_names": ["shoulder_pan", "gripper"],
        "action_space": simulation_space(),
        **changes,
    }


def segment(**changes):
    return {
        "targets": [
            {"joint": "gripper", "position": 0.1},
            {"joint": "shoulder_pan", "position": 0.02},
        ],
        "duration_seconds": 0.5,
        "purpose": "Bring the gripper closer to the visible object.",
        **changes,
    }


def response(name="propose_joint_segment", arguments=None, *, call_id="call_0"):
    return {
        "id": "response_fixture",
        "model": "gpt-6-astra",
        "status": "completed",
        "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
        "output": [
            {
                "type": "reasoning",
                "id": "reason_" + call_id,
                "summary": [],
                "encrypted_content": "opaque_" + call_id,
            },
            {
                "type": "function_call",
                "id": "fc_" + call_id,
                "call_id": call_id,
                "name": name,
                "arguments": json.dumps(segment() if arguments is None else arguments),
            },
        ],
    }


class TaskPlanningTests(unittest.IsolatedAsyncioTestCase):
    async def invoke(self, body, *, supplied=None):
        sent = []

        def handler(http_request):
            sent.append(json.loads(http_request.content))
            self.assertEqual(str(http_request.url), "https://api.openai.com/v1/responses")
            return httpx.Response(200, json=body)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await run(
                request() if supplied is None else supplied,
                provider=OpenAIResponsesProvider("fixture-secret", client=client),
            )
        self.assertEqual(len(sent), 1)
        return result, sent[0]

    async def test_freeform_pickup_keeps_current_image_and_strict_named_segment(self):
        supplied = request()
        original = copy.deepcopy(supplied)
        result, sent = await self.invoke(response(), supplied=supplied)
        self.assertEqual(supplied, original)
        content = sent["input"][0]["content"]
        current = json.loads(content[0]["text"])
        self.assertEqual(current["task"], supplied["task"])
        self.assertEqual(current["observation"]["observation_id"], "observation_0")
        self.assertEqual(current["action_space"], simulation_space())
        self.assertNotIn("image_data_url", current["observation"])
        self.assertEqual(
            content[1], {"type": "input_image", "image_url": CHECK_IMAGE, "detail": "low"}
        )
        self.assertIn("abstract goal", sent["instructions"])
        import hashlib

        self.assertEqual(result["prompt_version"], "agent-goal-v2")
        self.assertEqual(
            result["prompt_sha256"], hashlib.sha256(sent["instructions"].encode()).hexdigest()
        )
        self.assertFalse(sent["parallel_tool_calls"])
        self.assertFalse(sent["store"])
        schemas = {tool["name"]: tool for tool in sent["tools"]}
        self.assertEqual(
            set(schemas), {"propose_joint_correction", "propose_joint_segment", "finish_task"}
        )
        tool = schemas["propose_joint_segment"]
        self.assertTrue(tool["strict"])
        parameters = tool["parameters"]
        self.assertFalse(parameters["additionalProperties"])
        self.assertEqual(set(parameters["required"]), {"targets", "duration_seconds", "purpose"})
        item = parameters["properties"]["targets"]["items"]
        self.assertFalse(item["additionalProperties"])
        self.assertEqual(item["properties"]["joint"]["enum"], supplied["joint_names"])
        self.assertEqual(parameters["properties"]["duration_seconds"]["maximum"], 2)
        self.assertEqual(result["status"], "proposed")
        self.assertEqual(result["skill"], "joint_segment")
        self.assertEqual(
            result["arguments"],
            {"targets": {"gripper": 0.1, "shoulder_pan": 0.02}, "duration_seconds": 0.5},
        )
        self.assertEqual(result["purpose"], segment()["purpose"])
        self.assertNotIn("fixture-secret", json.dumps(result))

    async def test_finish_is_a_model_assessment_without_verified_task_success(self):
        for outcome in ("achieved", "needs_help", "cannot_proceed"):
            with self.subTest(outcome=outcome):
                assessment = {
                    "outcome": outcome,
                    "summary": "An assessment based on current pixels.",
                }
                result, _ = await self.invoke(response("finish_task", assessment))
                self.assertEqual(result["status"], "assessed")
                self.assertEqual(result["assessment"], assessment)
                self.assertNotIn("arguments", result)
                self.assertIsNot(result.get("task_success_verified"), True)
        for arguments in (
            {"outcome": "passed", "summary": "Unknown outcome"},
            {"outcome": "achieved", "summary": "ok", "task_success_verified": True},
            {"outcome": "needs_help", "summary": ""},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ProviderError):
                await self.invoke(response("finish_task", arguments))

    async def test_continuation_uses_original_call_and_latest_observation_with_bounded_history(
        self,
    ):
        sent = []
        duplicate = False

        def handler(http_request):
            payload = json.loads(http_request.content)
            index = len(sent)
            sent.append(payload)
            return httpx.Response(
                200,
                json=response(
                    arguments=segment(
                        targets=[{"joint": "shoulder_pan", "position": (index + 1) / 100}]
                    ),
                    call_id="call_0" if duplicate else f"call_{index}",
                ),
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = OpenAIResponsesProvider("fixture-secret", client=client)
            previous = None
            for index in range(7):
                observation = request()["observation"] | {
                    "observation_id": f"observation_{index}",
                    "state_rad": [index / 100, 0.05],
                }
                supplied = request(observation=observation, completed_steps=index)
                if previous is not None:
                    supplied.update(
                        operation="continue",
                        continuation=previous["continuation"],
                        receipt={
                            "status": "completed",
                            "execution_id": f"execution_{index - 1}",
                            "task_success_verified": False,
                        },
                    )
                previous = await run(supplied, provider=provider)
                payload = sent[-1]
                users = [item for item in payload["input"] if item.get("role") == "user"]
                self.assertEqual(len(users), min(index + 1, 4))
                current = json.loads(users[-1]["content"][0]["text"])
                self.assertEqual(current["observation"]["observation_id"], f"observation_{index}")
                self.assertEqual(current["completed_steps"], index)
                images = [
                    part
                    for user in users
                    for part in user["content"]
                    if part.get("type") == "input_image"
                ]
                self.assertEqual(
                    images, [{"type": "input_image", "image_url": CHECK_IMAGE, "detail": "low"}]
                )
                calls = [
                    item["call_id"]
                    for item in payload["input"]
                    if item.get("type") == "function_call"
                ]
                outputs = [
                    item for item in payload["input"] if item.get("type") == "function_call_output"
                ]
                self.assertEqual(calls, [item["call_id"] for item in outputs])
                self.assertEqual(len(calls), len(set(calls)))
                if index:
                    self.assertEqual(outputs[-1]["call_id"], f"call_{index - 1}")
                    self.assertEqual(
                        json.loads(outputs[-1]["output"])["execution_id"], f"execution_{index - 1}"
                    )
                    reasoning = [
                        item for item in payload["input"] if item.get("type") == "reasoning"
                    ]
                    self.assertEqual(reasoning[-1]["encrypted_content"], f"opaque_call_{index - 1}")
                self.assertEqual(
                    previous["continuation"]["seen_call_ids"],
                    [f"call_{n}" for n in range(index + 1)],
                )
            # Even a call outside the compacted history stays in the replay ledger.
            duplicate = True
            with self.assertRaisesRegex(ProviderError, "reused call identity"):
                await run(
                    request(
                        operation="continue",
                        continuation=previous["continuation"],
                        receipt={"status": "completed"},
                        completed_steps=7,
                    ),
                    provider=provider,
                )

    async def test_changed_action_space_is_rejected_before_another_http_request(self):
        sent = []

        def handler(http_request):
            sent.append(json.loads(http_request.content))
            return httpx.Response(200, json=response())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = OpenAIResponsesProvider("fixture-secret", client=client)
            first = await run(request(), provider=provider)
            for key, value in (
                ("max_delta", {"shoulder_pan": 0.1, "gripper": 0.1}),
                ("joint_limits", {"shoulder_pan": [-0.4, 0.4], "gripper": [0, 0.2]}),
                ("max_duration_seconds", 1.5),
            ):
                changed = simulation_space() | {key: value}
                with (
                    self.subTest(key=key),
                    self.assertRaisesRegex(ProviderError, "original provider context"),
                ):
                    await run(
                        request(
                            operation="continue",
                            continuation=first["continuation"],
                            receipt={},
                            action_space=changed,
                        ),
                        provider=provider,
                    )
            self.assertEqual(len(sent), 1)

    async def test_physical_degrees_percent_and_quintic_segment_normalization(self):
        for unit in ("deg", "degrees"):
            with self.subTest(unit=unit):
                space = physical_space()
                space["joint_units"]["shoulder_pan"] = unit
                supplied = request(
                    action_space=space,
                    observation={
                        "observation_id": "physical_observation",
                        "joints": {"shoulder_pan": 1, "gripper": 50},
                        "joint_units": space["joint_units"],
                        "image_data_url": CHECK_IMAGE,
                    },
                )
                proposed = segment(
                    targets=[
                        {"joint": "shoulder_pan", "position": 2},
                        {"joint": "gripper", "position": 52},
                    ],
                    duration_seconds=2,
                )
                result, sent = await self.invoke(response(arguments=proposed), supplied=supplied)
                self.assertEqual(
                    result["arguments"],
                    {"targets": {"shoulder_pan": 2.0, "gripper": 52.0}, "duration_seconds": 2.0},
                )
                self.assertEqual(
                    {tool["name"] for tool in sent["tools"]},
                    {"propose_joint_segment", "finish_task"},
                )
                units = json.loads(sent["input"][0]["content"][0]["text"])["units"]
                self.assertEqual(units, space["joint_units"])

    async def test_physical_calibrated_delta_limits_duration_and_quintic_peaks_reject(self):
        supplied = request(
            action_space=physical_space(),
            observation={
                "joints": {"shoulder_pan": 1, "gripper": 50},
                "image_data_url": CHECK_IMAGE,
            },
        )
        cases = [
            segment(targets=[{"joint": "shoulder_pan", "position": 3.6}], duration_seconds=4),
            segment(targets=[{"joint": "gripper", "position": 54.1}], duration_seconds=5),
            segment(targets=[{"joint": "shoulder_pan", "position": 2}], duration_seconds=0.1),
            segment(targets=[{"joint": "shoulder_pan", "position": 2}], duration_seconds=5.1),
            # Delta is admitted, but the quintic peak speed exceeds 2 units/s.
            segment(targets=[{"joint": "shoulder_pan", "position": 3}], duration_seconds=1),
            # Peak speed 1.875 is admitted; peak acceleration 5.774 exceeds 3.
            segment(targets=[{"joint": "shoulder_pan", "position": 2}], duration_seconds=1),
        ]
        limited = copy.deepcopy(supplied)
        limited["action_space"]["joint_limits"]["shoulder_pan"] = [-2, 2]
        for case in cases:
            with self.subTest(arguments=case), self.assertRaises(ProviderError):
                await self.invoke(response(arguments=case), supplied=supplied)
        with self.assertRaises(ProviderError):
            await self.invoke(
                response(
                    arguments=segment(
                        targets=[{"joint": "shoulder_pan", "position": 2.1}], duration_seconds=5
                    )
                ),
                supplied=limited,
            )
        with self.assertRaises(ProviderError):
            await self.invoke(
                response(
                    "propose_joint_correction",
                    {"joint": "shoulder_pan", "delta_rad": 0.01, "purpose": "Wrong physical skill"},
                ),
                supplied=supplied,
            )

    async def test_strict_segment_arguments_and_unknown_tools_fail(self):
        cases = [
            segment(targets=[]),
            segment(targets={"shoulder_pan": 0.01}),
            segment(targets=[{"joint": "unknown", "position": 0.01}]),
            segment(
                targets=[
                    {"joint": "shoulder_pan", "position": 0.01},
                    {"joint": "shoulder_pan", "position": 0.02},
                ]
            ),
            segment(targets=[{"joint": "shoulder_pan", "position": 0.01, "velocity": 1}]),
            segment(targets=[{"joint": "gripper", "position": 0.151}]),
            segment(duration_seconds=True),
            segment(duration_seconds=float("inf")),
            segment(purpose=""),
            segment(purpose="x" * 513),
            segment(extra="shell"),
        ]
        for case in cases:
            with self.subTest(arguments=case), self.assertRaises(ProviderError):
                await self.invoke(response(arguments=case))
        for name in ("shell", "run_python", "serial_write", "web_search"):
            with self.subTest(tool=name), self.assertRaisesRegex(ProviderError, "unknown tool"):
                await self.invoke(
                    response(name, {"command": "arbitrary", "purpose": "No such skill"})
                )

    async def test_invalid_physical_manifest_is_rejected_without_http(self):
        calls = []

        def handler(http_request):
            calls.append(http_request)
            raise AssertionError("Invalid host manifest must fail before HTTP")

        cases = []
        for path, value in (
            ("joint_units", {"shoulder_pan": "rad", "gripper": "percent"}),
            ("max_delta", {"shoulder_pan": 5.1, "gripper": 4}),
            ("max_delta", {"shoulder_pan": 2.5, "gripper": 10.1}),
            ("trajectory_profile", None),
            (
                "trajectory_profile",
                {
                    "kind": "linear",
                    "max_speed_units_per_second": 2,
                    "max_acceleration_units_per_second2": 3,
                },
            ),
        ):
            cases.append(physical_space() | {path: value})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = OpenAIResponsesProvider("fixture-secret", client=client)
            for space in cases:
                with self.subTest(space=space), self.assertRaises(ProviderError):
                    await run(request(action_space=space), provider=provider)
        self.assertEqual(calls, [])

    async def test_task_correction_respects_calibrated_delta_and_actual_limit(self):
        proposed = {"joint": "shoulder_pan", "delta_rad": 0.03, "purpose": "Small approach"}
        supplied = request()
        result, _ = await self.invoke(
            response("propose_joint_correction", proposed), supplied=supplied
        )
        self.assertEqual(result["arguments"], {"joint": "shoulder_pan", "delta_rad": 0.03})
        supplied["action_space"]["max_delta"]["shoulder_pan"] = 0.02
        with self.assertRaises(ProviderError):
            await self.invoke(response("propose_joint_correction", proposed), supplied=supplied)
        supplied = request()
        supplied["observation"]["state_rad"][0] = 0.49
        with self.assertRaises(ProviderError):
            await self.invoke(response("propose_joint_correction", proposed), supplied=supplied)

    def position_request(self):
        names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        space = physical_space()
        space.update(
            joint_units={j: "percent" if j == "gripper" else "degrees" for j in names},
            joint_limits={j: [0, 100] if j == "gripper" else [-90, 90] for j in names},
            max_delta={j: 8 if j == "gripper" else 4 for j in names},
            gripper_position={
                "model": "so101-new-calib-3a65d2d35e68a8d2",
                "frame": "base_link",
                "units": "metres",
                "max_step_m": 0.02,
                "tolerance_m": 0.003,
                "evidence": "model_estimate",
                "orientation_controlled": False,
                "camera_to_base_calibrated": False,
            },
        )
        return request(
            joint_names=names,
            action_space=space,
            observation={
                "observation_id": "physical-position",
                "joints": dict.fromkeys(names, 0.0),
                "joint_units": space["joint_units"],
                "image_data_url": CHECK_IMAGE,
                "gripper_position_estimate": {
                    "frame": "base_link",
                    "position_m": [0.391, 0.0, 0.226],
                    "evidence": "model_estimate",
                },
            },
        )

    async def test_gripper_position_tool_is_metric_and_opted_in_only(self):
        supplied = self.position_request()
        proposed = {
            "frame": "base_link",
            "x_m": 0.30,
            "y_m": 0.02,
            "z_m": 0.15,
            "duration_seconds": 2.0,
            "purpose": "Move to the known base-frame endpoint",
        }
        result, sent = await self.invoke(
            response("propose_gripper_position", proposed), supplied=supplied
        )
        self.assertEqual(result["skill"], "gripper_position")
        self.assertEqual(result["arguments"], {k: v for k, v in proposed.items() if k != "purpose"})
        schema = next(t for t in sent["tools"] if t["name"] == "propose_gripper_position")
        self.assertTrue(schema["strict"])
        self.assertEqual(schema["parameters"]["properties"]["frame"]["enum"], ["base_link"])
        del supplied["action_space"]["gripper_position"]
        with self.assertRaises(ProviderError):
            await self.invoke(response("propose_gripper_position", proposed), supplied=supplied)

    async def test_position_tool_rejects_unknown_frames_and_out_of_bounds_values(self):
        supplied = self.position_request()
        baseline = {
            "frame": "base_link",
            "x_m": 0.30,
            "y_m": 0.02,
            "z_m": 0.15,
            "duration_seconds": 2.0,
            "purpose": "Move to a known endpoint",
        }
        for changes in [
            {"frame": "camera"},
            {"z_m": -0.1},
            {"x_m": 0.7},
            {"duration_seconds": 0.1},
            {"gripper": 100},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ProviderError):
                await self.invoke(
                    response("propose_gripper_position", dict(baseline, **changes)),
                    supplied=supplied,
                )

    async def test_position_continuation_accepts_new_measured_estimates(self):
        supplied = self.position_request()
        proposed = {
            "frame": "base_link",
            "x_m": 0.30,
            "y_m": 0.02,
            "z_m": 0.15,
            "duration_seconds": 2.0,
            "purpose": "Continue toward the known endpoint",
        }
        first, _ = await self.invoke(
            response("propose_gripper_position", proposed, call_id="metric-0"), supplied=supplied
        )
        next_request = self.position_request()
        next_request["observation"]["joints"]["shoulder_lift"] = 2.0
        next_request["observation"]["gripper_position_estimate"]["position_m"] = [0.392, 0.0, 0.215]
        next_request.update(
            operation="continue",
            continuation=first["continuation"],
            receipt={
                "status": "completed",
                "execution_id": "metric-execution",
                "task_success_verified": False,
            },
        )
        result, sent = await self.invoke(
            response("propose_gripper_position", proposed, call_id="metric-1"),
            supplied=next_request,
        )
        self.assertEqual(result["skill"], "gripper_position")
        latest = [item for item in sent["input"] if item.get("role") == "user"][-1]
        current = json.loads(latest["content"][0]["text"])
        self.assertEqual(
            current["observation"]["gripper_position_estimate"]["position_m"], [0.392, 0.0, 0.215]
        )


if __name__ == "__main__":
    unittest.main()
