"""Transport and cancellation acceptance without a model, credentials, or actuator."""

import asyncio
import copy
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import httpx

from jensen_agent_control import MockProvider, OpenAIResponsesProvider, ProviderError, run
from jensen_agent_control.contracts import parse_json, validate_arguments
from jensen_agent_control.providers import CHECK_IMAGE


def request(**changes):
    return {
        "provider": "openai_responses",
        "task": "shoulder_pan +0.03 rad",
        "observation": {
            "observation_id": "obs-1",
            "measured_joints": {"shoulder_pan": 0.0},
            "image_data_url": CHECK_IMAGE,
        },
        "joint_names": ["shoulder_pan"],
        **changes,
    }


def response(*, output=None, model="gpt-6-astra", status="completed", **changes):
    if output is None:
        output = [
            {
                "type": "reasoning",
                "id": "reason-1",
                "summary": [],
                "encrypted_content": "opaque-provider-context",
            },
            {
                "type": "function_call",
                "id": "fc-1",
                "call_id": "call-original",
                "name": "propose_joint_correction",
                "arguments": '{"joint":"shoulder_pan","delta_rad":0.03}',
            },
        ]
    return {
        "id": "resp-1",
        "model": model,
        "status": status,
        "output": output,
        "usage": {"input_tokens": 15, "output_tokens": 7, "total_tokens": 22},
        **changes,
    }


def final_response():
    return response(
        output=[
            {"type": "message", "content": [{"type": "output_text", "text": "Receipt received."}]}
        ]
    )


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_mock_only_acts_on_precise_relative_instruction(self):
        proposal = await run(request(provider="mock"))
        self.assertEqual(proposal["arguments"], {"joint": "shoulder_pan", "delta_rad": 0.03})
        for task in (
            "Pick up the block",
            "shoulder_pan to 0.03 rad",
            "shoulder_pan 0.02 rad and shoulder_pan 0.03 rad",
        ):
            with self.subTest(task=task):
                result = await run(request(provider="mock", task=task))
                self.assertEqual(result["status"], "complete")
                self.assertNotIn("arguments", result)
        with self.assertRaises(ProviderError):
            await run(request(provider="mock", task="shoulder_pan +0.3 rad"))

    async def test_mock_check_cannot_claim_vision_or_hardware(self):
        result = await run({"provider": "mock", "operation": "check"})
        self.assertEqual(result["status"], "checked")
        self.assertFalse(result["capabilities"]["images"])
        self.assertFalse(result["capabilities"]["actuation"])

    async def test_response_images_schema_identity_and_tool_result_continuation(self):
        sent = []

        async def handler(http_request):
            sent.append(json.loads(http_request.content))
            self.assertEqual(str(http_request.url), "https://api.openai.com/v1/responses")
            self.assertEqual(http_request.headers["authorization"], "Bearer secret-for-http-only")
            return httpx.Response(200, json=response() if len(sent) == 1 else final_response())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = OpenAIResponsesProvider("secret-for-http-only", client=client)
            first = await run(request(), provider=provider)
            second = await run(
                request(
                    operation="continue",
                    continuation=first["continuation"],
                    receipt={
                        "status": "completed",
                        "execution_id": "exec-1",
                        "task_success_verified": False,
                    },
                ),
                provider=provider,
            )
        self.assertEqual(first["status"], "proposed")
        self.assertEqual(second["status"], "complete")
        self.assertFalse(sent[0]["parallel_tool_calls"])
        self.assertFalse(sent[0]["store"])
        self.assertEqual(sent[0]["include"], ["reasoning.encrypted_content"])
        tool = sent[0]["tools"][0]
        self.assertTrue(tool["strict"])
        self.assertFalse(tool["parameters"]["additionalProperties"])
        self.assertEqual(set(tool["parameters"]["required"]), {"joint", "delta_rad"})
        self.assertEqual(sent[0]["input"][0]["content"][1]["image_url"], CHECK_IMAGE)
        continuation = sent[1]["input"]
        self.assertIn(response()["output"][0], continuation)
        receipt = next(item for item in continuation if item.get("type") == "function_call_output")
        self.assertEqual(receipt["call_id"], "call-original")
        self.assertEqual(json.loads(receipt["output"])["execution_id"], "exec-1")
        self.assertNotIn("secret-for-http-only", json.dumps(first))
        self.assertEqual(first["usage"]["total_tokens"], 22)

    async def test_malformed_or_unsafe_results_are_rejected(self):
        original = response()
        cases = [
            response(model="different-model"),
            response(status="incomplete"),
            response(output=[]),
            response(usage=None),
            response(output=original["output"] + [original["output"][-1]]),
            response(output=[{"type": "web_search_call"}]),
        ]
        for arguments in (
            '{"joint":"other","delta_rad":0.03}',
            '{"joint":"shoulder_pan","delta_rad":NaN}',
            '{"joint":"shoulder_pan","delta_rad":0.3}',
            '{"joint":"shoulder_pan","delta_rad":true}',
            '{"joint":"shoulder_pan","delta_rad":0.03,"extra":1}',
            '{"joint":"shoulder_pan","joint":"shoulder_pan","delta_rad":0.03}',
        ):
            altered = copy.deepcopy(original)
            altered["output"][-1]["arguments"] = arguments
            cases.append(altered)
        for body in cases:
            with self.subTest(body=body):
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
                ) as client:
                    with self.assertRaises(ProviderError):
                        await run(
                            request(), provider=OpenAIResponsesProvider("secret", client=client)
                        )

    async def test_reused_call_identity_and_changed_manifest_rejected(self):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response()))
        ) as client:
            provider = OpenAIResponsesProvider("secret", client=client)
            first = await run(request(), provider=provider)
            with self.assertRaisesRegex(ProviderError, "reused call"):
                await run(
                    request(operation="continue", continuation=first["continuation"], receipt={}),
                    provider=provider,
                )
            with self.assertRaisesRegex(ProviderError, "original provider context"):
                await run(
                    request(
                        operation="continue",
                        continuation=first["continuation"],
                        receipt={},
                        joint_names=["elbow"],
                    ),
                    provider=provider,
                )

    async def test_timeout_cancels_inflight_work_and_no_late_proposal(self):
        cancelled = asyncio.Event()

        async def delayed(_request):
            try:
                await asyncio.sleep(5)
            finally:
                cancelled.set()
            return httpx.Response(200, json=response())

        async with httpx.AsyncClient(transport=httpx.MockTransport(delayed)) as client:
            with self.assertRaisesRegex(ProviderError, "timed out"):
                await run(
                    request(timeout_seconds=0.02),
                    provider=OpenAIResponsesProvider("secret", client=client),
                )
        self.assertTrue(cancelled.is_set())

    async def test_external_cancellation_propagates(self):
        task = asyncio.create_task(
            run(request(provider="mock"), provider=MockProvider(delay_seconds=5))
        )
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_synthetic_capability_probe_requires_both_turns(self):
        seen = []

        def handler(http_request):
            seen.append(json.loads(http_request.content))
            body = final_response() if len(seen) == 2 else response()
            if len(seen) == 1:
                body["output"][-1]["arguments"] = (
                    '{"joint":"capability_check_joint","delta_rad":0.01}'
                )
            return httpx.Response(200, json=body)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await run(
                request(operation="check"),
                provider=OpenAIResponsesProvider("secret", client=client),
            )
        self.assertEqual(len(seen), 2)
        self.assertEqual(result["status"], "checked")
        self.assertEqual(result["usage"]["total_tokens"], 44)
        self.assertTrue(result["capabilities"]["images"])
        self.assertTrue(result["capabilities"]["tool_continuation"])
        self.assertFalse(result["capabilities"]["actuation"])

    async def test_budget_and_secret_errors_are_sanitized(self):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(401, text="secret-credential"))
        ) as client:
            with self.assertRaises(ProviderError) as caught:
                await run(
                    request(), provider=OpenAIResponsesProvider("secret-credential", client=client)
                )
        self.assertNotIn("secret", str(caught.exception))
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    json=response(
                        usage={"input_tokens": 60, "output_tokens": 10, "total_tokens": 70}
                    ),
                )
            )
        ) as client:
            with self.assertRaises(ProviderError) as caught:
                await run(
                    request(max_total_tokens=64, max_output_tokens=64),
                    provider=OpenAIResponsesProvider("secret", client=client),
                )
        self.assertEqual(caught.exception.code, "token_budget_exceeded")
        self.assertEqual(caught.exception.usage["total_tokens"], 70)

    async def test_never_fetches_remote_observation_urls(self):
        with self.assertRaisesRegex(ProviderError, "inline PNG or JPEG"):
            await run(
                request(observation={"image_data_url": "https://example.invalid/private"}),
                provider=OpenAIResponsesProvider("secret"),
            )


class BoundaryTests(unittest.TestCase):
    def test_duplicate_keys_and_nonfinite_arguments_fail(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(ProviderError):
                parse_json(raw)
        for delta in (float("nan"), float("inf"), True):
            with self.assertRaises(ProviderError):
                validate_arguments({"joint": "shoulder_pan", "delta_rad": delta}, ["shoulder_pan"])

    def test_cli_one_json_response_and_no_credential_echo(self):
        source = str(Path(__file__).resolve().parents[1] / "src")
        env = {
            **os.environ,
            "PYTHONPATH": source,
            "OPENJENSEN_AGENT_API_KEY": "secret-process-only",
        }
        completed = subprocess.run(
            [sys.executable, "-m", "jensen_agent_control"],
            input=json.dumps(request(provider="mock")),
            text=True,
            capture_output=True,
            env=env,
            check=True,
        )
        self.assertEqual(json.loads(completed.stdout)["status"], "proposed")
        self.assertEqual(completed.stderr, "")
        self.assertNotIn("secret-process-only", completed.stdout)
        invalid = subprocess.run(
            [sys.executable, "-m", "jensen_agent_control"],
            input='{"provider":"mock","api_key":"secret-process-only"}',
            text=True,
            capture_output=True,
            env=env,
        )
        self.assertEqual(invalid.returncode, 2)
        self.assertEqual(json.loads(invalid.stdout)["code"], "invalid_request")
        self.assertNotIn("secret-process-only", invalid.stdout + invalid.stderr)


if __name__ == "__main__":
    unittest.main()
