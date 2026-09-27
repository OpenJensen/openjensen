"""Real loopback HTTP/CLI subprocess checks with generated job and TAR records."""

import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from typer.testing import CliRunner
from vla_platform import cli, cli_client, tui_client
from vla_platform.tui_client import ApiClient

TIME = "2026-09-27T10:00:00Z"


@pytest.fixture
def server():
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as tar:
        raw = b'{"scope":"generated transport fixture"}\n'
        info = tarfile.TarInfo("receipt.json")
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
    archive = data.getvalue()
    state = {"requests": [], "job_reads": 0, "hold": False, "lost_posts": 0}
    seen = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            state["requests"].append(("GET", self.path))
            if self.path == "/api/v1/jobs/job":
                state["job_reads"] += 1
                seen.set()
                self.reply(
                    {
                        "id": "job",
                        "project_id": "project",
                        "status": "running"
                        if state["hold"] or state["job_reads"] < 3
                        else "succeeded",
                        "request": {
                            "source": "huggingface",
                            "repo_id": "generated/dataset",
                            "revision": "fixture",
                        },
                        "created_at": TIME,
                        "updated_at": TIME,
                    }
                )
            elif self.path == "/api/v1/projects/project/artifacts":
                self.reply(
                    [
                        {
                            "id": "job:policy",
                            "project_id": "project",
                            "job_id": "job",
                            "label": "Generated artifact",
                            "format": "native_quantized",
                            "path": "private/server-only",
                            "manifest_sha256": "a" * 64,
                            "file_bytes": len(archive),
                            "metadata": {},
                        }
                    ]
                )
            elif self.path == "/api/v1/projects/project/artifacts/job%3Apolicy/download":
                self.send_response(200)
                self.send_header("Content-Type", "application/x-tar")
                self.send_header("Content-Length", str(len(archive)))
                self.end_headers()
                for offset in range(0, len(archive), 257):
                    self.wfile.write(archive[offset : offset + 257])
                    self.wfile.flush()
            else:
                self.send_error(404)

        def do_POST(self):
            state["requests"].append(("POST", self.path))
            state["lost_posts"] += 1
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.close_connection = True  # Accepted bytes but no response: never safe to resubmit.

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", state, archive, seen
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)


def invoke(origin, *arguments):
    return subprocess.run(
        [sys.executable, "-m", "vla_platform.cli", *arguments],
        env={**os.environ, "FIREBIRD_API_URL": origin},
        text=True,
        capture_output=True,
        timeout=10,
    )


def test_real_cli_wait_then_stream_download_without_any_mutation(server, tmp_path):
    origin, state, archive, _ = server
    result = invoke(origin, "jobs", "wait", "job", "--timeout", "5", "--interval", "0.1")
    assert result.returncode == 0 and not result.stderr
    assert json.loads(result.stdout)["status"] == "succeeded" and state["job_reads"] == 3
    destination = tmp_path / "policy.tar"
    digest = hashlib.sha256(archive).hexdigest()
    result = invoke(
        origin,
        "policy",
        "download",
        "project",
        "job:policy",
        "--output",
        str(destination),
        "--expected-sha256",
        digest,
    )
    assert result.returncode == 0 and not result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["sha256"] == digest and receipt["expected_sha256_matched"] is True
    assert destination.read_bytes() == archive and receipt["bytes"] == len(archive)
    assert all(method == "GET" for method, _ in state["requests"])
    assert "private/server-only" not in result.stdout
    result = invoke(
        origin, "policy", "download", "project", "job:policy", "--output", str(destination)
    )
    assert result.returncode == 1 and "already exists" in result.stderr
    assert destination.read_bytes() == archive


def test_real_cli_lost_post_reports_uncertainty_and_sends_once(server):
    origin, state, _, _ = server
    result = invoke(origin, "projects", "create", "Generated lost response")
    assert result.returncode == 1 and not result.stdout
    assert "Outcome unknown" in result.stderr and "sent once" in result.stderr
    assert state["lost_posts"] == 1


def test_real_cli_wait_deadline_exits124_without_cancelling(server):
    origin, state, _, _ = server
    state["hold"] = True
    result = invoke(origin, "jobs", "wait", "job", "--timeout", "1", "--interval", "0.1")
    assert result.returncode == 124 and not result.stdout
    assert "not cancelled" in result.stderr
    assert all(method == "GET" for method, _ in state["requests"])


@pytest.mark.skipif(os.name != "posix", reason="SIGINT child test is POSIX-specific")
def test_real_cli_interrupt_stops_observing_but_never_cancels(server):
    origin, state, _, seen = server
    state["hold"] = True
    child = subprocess.Popen(
        [sys.executable, "-m", "vla_platform.cli", "jobs", "wait", "job"],
        env={**os.environ, "FIREBIRD_API_URL": origin},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert seen.wait(timeout=5)
        child.send_signal(signal.SIGINT)
        stdout, stderr = child.communicate(timeout=5)
        assert child.returncode == 130 and not stdout
        assert "No job cancellation" in stderr
        assert all(method == "GET" for method, _ in state["requests"])
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=3)


@pytest.mark.parametrize(
    "url",
    [
        "http://user:TEST_SECRET@localhost",
        "http://127.0.0.1:TEST_SECRET",
        "http://localhost?secret=TEST_SECRET",
        "file:///private",
    ],
)
def test_bad_endpoint_never_reflects_credentials_or_makes_http(monkeypatch, url):
    monkeypatch.setenv("FIREBIRD_API_URL", url)
    monkeypatch.setattr(
        httpx.AsyncClient, "send", lambda *_a, **_k: pytest.fail("Invalid endpoint reached HTTP")
    )
    result = CliRunner().invoke(cli.app, ["projects", "list"])
    assert result.exit_code == 1 and "API URL" in result.stderr
    assert "TEST_SECRET" not in result.output and "Traceback" not in result.output


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_cli_rejects_redirect_json_without_following(monkeypatch, status):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(
            status,
            json={"detail": "generated redirect"},
            headers={"Location": "https://example.invalid"},
        )

    monkeypatch.setattr(
        cli_client,
        "client",
        lambda: ApiClient("http://127.0.0.1", transport=httpx.MockTransport(handle)),
    )
    result = CliRunner().invoke(cli.app, ["jobs", "show", "job"])
    assert (
        result.exit_code == 1
        and not result.stdout
        and str(status) in result.stderr
        and len(calls) == 1
    )


def test_cli_path_ids_are_encoded_as_single_segments(monkeypatch):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json={"id": "fixture"})

    monkeypatch.setattr(
        cli_client,
        "client",
        lambda: ApiClient("http://127.0.0.1", transport=httpx.MockTransport(handle)),
    )
    result = CliRunner().invoke(cli.app, ["jobs", "show", "x?unexpected=1/#fragment"])
    assert result.exit_code == 0 and len(calls) == 1
    assert calls[0].url.raw_path == b"/api/v1/jobs/x%3Funexpected%3D1%2F%23fragment"
    assert not calls[0].url.query


def test_cli_json_response_limit_is_enforced(monkeypatch):
    monkeypatch.setattr(tui_client, "MAX_RESPONSE", 16)
    monkeypatch.setattr(
        cli_client,
        "client",
        lambda: ApiClient(
            "http://127.0.0.1",
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b" " * 17)),
        ),
    )
    result = CliRunner().invoke(cli.app, ["projects", "list"])
    assert result.exit_code == 1 and not result.stdout and "limit" in result.stderr


def acknowledgment_case(kind):
    """Valid recorded replies include schema defaults, as the actual API does."""
    from vla_platform.contracts import Job, Project

    if kind == "project":
        return (
            "/projects",
            {"name": " Generated "},
            Project(id="created", name="Generated", created_at=TIME).model_dump(mode="json"),
        )
    requests = {
        "policy": {
            "operation": "policy.quantize",
            "runtime_id": "cpu",
            "artifact_id": "source",
            "native_quantization": {"format": "firebird_quant", "bits": 8, "group_size": 64},
        },
        "intake": {"source": "huggingface", "repo_id": "generated/dataset", "revision": "main"},
        "augmentation": {
            "source_job_id": "dataset",
            "episode_indices": [0],
            "camera_key": "observation.images.front",
        },
    }
    request = requests["intake" if kind == "cancel" else kind]
    operation = {"policy": "policy.quantize", "augmentation": "dataset.augment"}.get(
        kind, "dataset.inspect"
    )
    value = Job(
        id="created",
        project_id="expected-project",
        kind=operation,
        request=request,
        created_at=TIME,
        updated_at=TIME,
    ).model_dump(mode="json")
    suffix = {"policy": "policy-jobs", "intake": "intakes", "augmentation": "augmentations"}
    path = (
        "/jobs/created/cancel" if kind == "cancel" else f"/projects/expected-project/{suffix[kind]}"
    )
    return path, None if kind == "cancel" else request, value


@pytest.mark.parametrize("kind", ["project", "policy", "intake", "augmentation", "cancel"])
def test_write_acknowledgment_validates_identity_and_preserves_original_json(monkeypatch, kind):
    import asyncio

    path, payload, acknowledgment = acknowledgment_case(kind)
    calls = []
    monkeypatch.setattr(
        cli_client,
        "client",
        lambda: ApiClient(
            "http://127.0.0.1",
            transport=httpx.MockTransport(
                lambda req: calls.append(req) or httpx.Response(202, json=acknowledgment)
            ),
        ),
    )
    actual = asyncio.run(cli_client.request_json("POST", path, payload))
    assert actual == acknowledgment and len(calls) == 1


@pytest.mark.parametrize("kind", ["project", "policy", "intake", "augmentation", "cancel"])
@pytest.mark.parametrize("fault", ["malformed", "identity", "missing_status_or_name"])
def test_invalid_write_acknowledgment_is_uncertain_and_never_repeated(monkeypatch, kind, fault):
    import asyncio

    from vla_platform.tui_client import ApiError

    path, payload, acknowledgment = acknowledgment_case(kind)
    if fault == "malformed":
        acknowledgment = {}
    elif fault == "identity":
        key = "name" if kind == "project" else "id" if kind == "cancel" else "project_id"
        acknowledgment[key] = "another"
    else:
        del acknowledgment["name" if kind == "project" else "status"]
    calls = []
    monkeypatch.setattr(
        cli_client,
        "client",
        lambda: ApiClient(
            "http://127.0.0.1",
            transport=httpx.MockTransport(
                lambda req: calls.append(req) or httpx.Response(200, json=acknowledgment)
            ),
        ),
    )
    with pytest.raises(ApiError, match="Outcome unknown.*sent once"):
        asyncio.run(cli_client.request_json("POST", path, payload))
    assert len(calls) == 1 and calls[0].method == "POST"


@pytest.mark.parametrize(
    "kind,field,value",
    [
        ("policy", "artifact_id", "another-source"),
        ("policy", "runtime_id", "another-runtime"),
        ("intake", "repo_id", "another/dataset"),
        ("augmentation", "source_job_id", "another"),
    ],
)
def test_valid_but_wrong_source_write_receipt_is_uncertain(monkeypatch, kind, field, value):
    import asyncio

    from vla_platform.tui_client import ApiError

    path, payload, acknowledgment = acknowledgment_case(kind)
    acknowledgment["request"][field] = value
    calls = []
    monkeypatch.setattr(
        cli_client,
        "client",
        lambda: ApiClient(
            "http://127.0.0.1",
            transport=httpx.MockTransport(
                lambda req: calls.append(req) or httpx.Response(200, json=acknowledgment)
            ),
        ),
    )
    with pytest.raises(ApiError, match="Outcome unknown.*sent once"):
        asyncio.run(cli_client.request_json("POST", path, payload))
    assert len(calls) == 1


def test_cli_policy_submit_invalid_acknowledgment_has_no_success_stdout(tmp_path, monkeypatch):
    _, payload, _ = acknowledgment_case("policy")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps(payload))
    calls = []
    monkeypatch.setattr(
        cli_client,
        "client",
        lambda: ApiClient(
            "http://127.0.0.1",
            transport=httpx.MockTransport(
                lambda req: calls.append(req) or httpx.Response(200, json={})
            ),
        ),
    )
    result = CliRunner().invoke(cli.app, ["policy", "submit", "expected-project", str(recipe)])
    assert result.exit_code == 1 and not result.stdout
    assert "Outcome unknown" in result.stderr and "sent once" in result.stderr
    assert len(calls) == 1


@pytest.mark.parametrize(
    "case", ["hub_branch", "local_metadata", "snapshot", "new_training", "resume"]
)
def test_acknowledgment_preserves_legitimate_server_normalization(monkeypatch, case):
    import asyncio

    from vla_platform.contracts import Job

    if case == "hub_branch":
        path, payload, acknowledgment = acknowledgment_case("intake")
        acknowledgment["request"]["revision"] = "a" * 40
    elif case in {"local_metadata", "snapshot"}:
        path = "/projects/expected-project/intakes"
        payload = {
            "source": "local",
            "path": "dataset",
            "snapshot_for_training": case == "snapshot",
        }
        recorded = dict(payload)
        if case == "local_metadata":
            recorded.update(path="/server/datasets/dataset", revision="metadata-sha256:" + "b" * 64)
        acknowledgment = Job(
            id="created",
            project_id="expected-project",
            request=recorded,
            created_at=TIME,
            updated_at=TIME,
        ).model_dump(mode="json")
    else:
        path = "/projects/expected-project/policy-jobs"
        payload = {"operation": "policy.finetune", "runtime_id": "gpu"}
        if case == "resume":
            payload.update(resume_job_id="previous", dataset_job_id="requested-data")
            recorded = {**payload, "dataset_job_id": "saved-data", "training_method": "full"}
        else:
            payload.update(dataset_job_id="data", training={"steps": 10})
            recorded = {
                **payload,
                "training": {"steps": 10, "model_id": "act", "model_revision": "a" * 40},
            }
        acknowledgment = Job(
            id="created",
            project_id="expected-project",
            kind="policy.finetune",
            request=recorded,
            created_at=TIME,
            updated_at=TIME,
        ).model_dump(mode="json")
    calls = []
    monkeypatch.setattr(
        cli_client,
        "client",
        lambda: ApiClient(
            "http://127.0.0.1",
            transport=httpx.MockTransport(
                lambda req: calls.append(req) or httpx.Response(202, json=acknowledgment)
            ),
        ),
    )
    assert asyncio.run(cli_client.request_json("POST", path, payload)) == acknowledgment
    assert len(calls) == 1


def test_pinned_intake_acknowledgment_cannot_change_revision(monkeypatch):
    import asyncio

    from vla_platform.tui_client import ApiError

    path, payload, acknowledgment = acknowledgment_case("intake")
    payload["revision"] = "a" * 40
    acknowledgment["request"]["revision"] = "b" * 40
    monkeypatch.setattr(
        cli_client,
        "client",
        lambda: ApiClient(
            "http://127.0.0.1",
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json=acknowledgment)),
        ),
    )
    with pytest.raises(ApiError, match="Outcome unknown"):
        asyncio.run(cli_client.request_json("POST", path, payload))


@pytest.mark.parametrize(
    "field,value", [("steps", 20000), ("batch_size", 64), ("learning_rate", 0.1)]
)
def test_new_training_receipt_cannot_change_submitted_recipe(monkeypatch, field, value):
    import asyncio

    from vla_platform.contracts import Job
    from vla_platform.tui_client import ApiError

    recipe = {"steps": 12, "batch_size": 1, "learning_rate": 0.0001}
    payload = {
        "operation": "policy.finetune",
        "runtime_id": "gpu",
        "dataset_job_id": "data",
        "training": recipe,
    }
    recorded = {**payload, "training": {**recipe, field: value, "model_id": "act"}}
    acknowledgment = Job(
        id="created",
        project_id="expected-project",
        kind="policy.finetune",
        request=recorded,
        created_at=TIME,
        updated_at=TIME,
    ).model_dump(mode="json")
    calls = []
    monkeypatch.setattr(
        cli_client,
        "client",
        lambda: ApiClient(
            "http://127.0.0.1",
            transport=httpx.MockTransport(
                lambda req: calls.append(req) or httpx.Response(202, json=acknowledgment)
            ),
        ),
    )
    with pytest.raises(ApiError, match="Outcome unknown.*sent once"):
        asyncio.run(
            cli_client.request_json("POST", "/projects/expected-project/policy-jobs", payload)
        )
    assert len(calls) == 1
