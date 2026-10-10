"""Extended terminal reviews against a disposable application and protocol workers.

No fixture result is evidence of model quality, GPU performance or robot success.
"""

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient
from test_act_export import application as act_application  # noqa: F401
from test_lifecycle import configured, project, submit, wait  # noqa: F401
from vla_platform.api import create_app
from vla_platform.cli_client import validate_acknowledgment
from vla_platform.tui_client import ApiClient
from vla_platform.tui_lifecycle import Context, canonical


def reviewed_submission(client, project_id, mode, payload):
    async def scenario():
        calls = []

        async def bridge(request):
            calls.append(request.method)
            response = await asyncio.to_thread(
                client.request,
                request.method,
                request.url.path,
                content=request.content,
                headers={"Content-Type": "application/json"},
            )
            return httpx.Response(response.status_code, content=response.content)

        api = ApiClient("http://testserver", transport=httpx.MockTransport(bridge))
        try:
            initial = await Context.fetch(api, project_id)
            reviewed = initial.review(mode, canonical(payload))
            current = await Context.fetch(api, project_id)
            refreshed = current.review(mode, canonical(reviewed.request))
            assert reviewed.context_sha256 == refreshed.context_sha256
            assert all(method == "GET" for method in calls)
            path = f"/projects/{project_id}/policy-jobs"
            accepted = await api.request("POST", path, reviewed.request)
            validate_acknowledgment(api, path, reviewed.request, accepted)
            assert calls.count("POST") == 1
            return accepted
        finally:
            await api.close()

    return asyncio.run(scenario())


@pytest.mark.parametrize(
    ("mode", "operation"),
    [
        ("evaluate", "policy.evaluate"),
        ("engine_run", "policy.run"),
        ("gguf_quantize", "policy.quantize"),
    ],
)
def test_terminal_engine_contract_executes_same_saved_policy_through_actual_api(
    configured,  # noqa: F811
    mode,
    operation,
):
    with TestClient(create_app(configured)) as client:
        pid = project(client)
        imported = wait(client, submit(client, pid, operation="policy.import"))
        assert imported["status"] == "succeeded"
        source = imported["result"]["artifacts"][0]
        listed = client.get(f"/api/v1/projects/{pid}/artifacts").json()
        saved_source = next(item for item in listed if item["id"] == source["id"])
        payload = {
            "operation": operation,
            "runtime_id": "fixture",
            "artifact_id": source["id"],
            "timeout_seconds": 60,
        }
        if mode == "gguf_quantize":
            payload["precision"] = {"language": "Q8_0", "vision": None}
        else:
            payload["evaluation"] = {
                "mode": "engine",
                "suite": "libero_object",
                "warmups": 1,
                "repetitions": 2,
            }
        accepted = reviewed_submission(client, pid, mode, payload)
        completed = wait(client, accepted["id"])
        assert completed["status"] == "succeeded", completed
        assert completed["request"]["artifact_id"] == source["id"]
        assert completed["result"]["decision"] != "validated"
        artifacts = client.get(f"/api/v1/projects/{pid}/artifacts").json()
        assert next(item for item in artifacts if item["id"] == source["id"]) == saved_source
        assert client.get(f"/api/v1/jobs/{imported['id']}").json() == imported
        for item in completed["result"]["artifacts"]:
            assert item["parent_ids"] == [source["id"]]
            assert item["metadata"]["fixture_only"] is True


def test_terminal_act_export_contract_uses_the_existing_isolated_export_worker(act_application):  # noqa: F811
    _, client, pid, artifact, _ = act_application
    accepted = reviewed_submission(
        client,
        pid,
        "export",
        {
            "operation": "policy.export",
            "runtime_id": "act-cpu",
            "artifact_id": artifact.id,
            "training_method": "full",
            "timeout_seconds": 60,
        },
    )
    completed = wait(client, accepted["id"])
    assert completed["status"] == "succeeded", completed
    output = completed["result"]["artifacts"][0]
    assert output["format"] == "inference_export" and output["parent_ids"] == [artifact.id]
    assert output["metadata"]["task_success"] is None
