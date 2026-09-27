"""Application advisory boundaries; fake workers never load ML or contact a provider."""

import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from vla_platform import decision_api as decision

PAYLOAD = {
    "schema_version": 1,
    "state": "The dataset is ready.",
    "instructions": "Compare only.",
    "criteria": [
        {"id": "train", "text": "Train a policy"},
        {"id": "inspect", "text": "Inspect a dataset"},
    ],
}


def valid_result(payload=PAYLOAD):
    return {
        "schema_version": 1,
        "model": decision.MODEL,
        "revision": decision.REVISION,
        "model_sha256": decision.MODEL_HASH,
        "license": decision.LICENSE,
        "prompt_template": "firebird-experimental-sections-v1",
        "prompt_template_sha256": hashlib.sha256(decision.TEMPLATE.encode()).hexdigest(),
        "request_sha256": hashlib.sha256(decision.canonical(payload)).hexdigest(),
        "device": "cpu",
        "threads": 2,
        "runtime_versions": decision.VERSIONS,
        "advisory_only": True,
        "calibrated": False,
        "selected_id": "train",
        "scores": [
            {"id": row["id"], "logit": 0.0, "relative_weight": 0.5, "tokens": 12}
            for row in payload["criteria"]
        ],
        "timing_ms": {"load": 1.0, "score": 2.0},
        "caveats": decision.CAVEATS,
    }


@pytest.fixture
def client(monkeypatch):
    for key in ("PYTHON", "ROOT", "MODEL_DIR", "ACCEPT_LICENSE"):
        monkeypatch.delenv("FIREBIRD_DECISION_" + key, raising=False)
    app = FastAPI()
    app.include_router(decision.router)
    with TestClient(app) as test:
        yield test


@pytest.fixture
def configured(client, tmp_path, monkeypatch):
    if os.name != "posix":
        pytest.skip("The advisory worker intentionally requires a POSIX host")
    root = tmp_path / "worker"
    package = root / "src/firebird_decision"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    main = package / "__main__.py"
    main.write_text("import json\nprint(" + repr(json.dumps(valid_result())) + ")\n")
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setenv("FIREBIRD_DECISION_PYTHON", sys.executable)
    monkeypatch.setenv("FIREBIRD_DECISION_ROOT", str(root))
    monkeypatch.setenv("FIREBIRD_DECISION_MODEL_DIR", str(model))
    monkeypatch.setenv("FIREBIRD_DECISION_ACCEPT_LICENSE", decision.LICENSE)
    return client, main, root, model


def test_missing_configuration_is_explicit_and_private(client):
    response = client.get("/api/v1/decision/status")
    assert response.json()["configured"] is False and response.json()["available"] is False
    assert response.headers["cache-control"] == "no-store"
    assert client.post("/api/v1/decision/score", json=PAYLOAD).status_code == 503


def test_configured_is_not_a_runtime_or_integrity_claim(configured):
    client, *_ = configured
    result = client.get("/api/v1/decision/status").json()
    assert result["configured"] and result["available"]
    assert "attempt" in result["message"] and "verified on each score" in result["message"]


def test_valid_receipt_matches_request_and_cannot_perform_action(configured):
    client, *_ = configured
    response = client.post("/api/v1/decision/score", json=PAYLOAD)
    assert response.status_code == 200, response.text
    assert response.json() == valid_result()
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": True},
        {"state": " "},
        {"state": "\u0000bad"},
        {"criteria": PAYLOAD["criteria"][:1]},
        {"criteria": PAYLOAD["criteria"] * 2},
        {"python": "/untrusted"},
        {"model_dir": "/private"},
        {"state": "x" * 8001},
    ],
)
def test_rejects_invalid_input_before_launch(configured, change):
    client, main, *_ = configured
    main.write_text("raise AssertionError('must not launch')")
    assert client.post("/api/v1/decision/score", json={**PAYLOAD, **change}).status_code == 422


@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema_version":1,"schema_version":1}',
        b"{}" * 40000,
        b'{"unused":1e309}',
        "{}".encode("utf-16"),
    ],
)
def test_bounded_utf8_unique_finite_json(client, raw):
    assert client.post("/api/v1/decision/score", content=raw).status_code == 422


@pytest.mark.parametrize(
    "change",
    [
        {"request_sha256": "a" * 64},
        {"model_sha256": "a" * 64},
        {"revision": "wrong"},
        {"selected_id": "inspect"},
        {"advisory_only": 1},
        {"calibrated": True},
        {"threads": True},
        {"runtime_versions": {"torch": "other"}},
        {"private_path": "/key"},
        {"timing_ms": {"load": -1.0, "score": 0.0}},
        {
            "scores": [
                {"id": "train", "logit": 0.0, "relative_weight": 0.9, "tokens": 2},
                {"id": "inspect", "logit": 0.0, "relative_weight": 0.1, "tokens": 2},
            ]
        },
    ],
)
def test_forged_receipts_never_cross_api(configured, change):
    client, main, *_ = configured
    result = {**valid_result(), **change}
    main.write_text("print(" + repr(json.dumps(result)) + ")")
    response = client.post("/api/v1/decision/score", json=PAYLOAD)
    assert response.status_code == 503 and "/key" not in response.text


def test_nonfinite_response_and_private_worker_diagnostics_redacted(configured):
    client, main, *_ = configured
    result = valid_result()
    result["timing_ms"]["load"] = float("inf")
    main.write_text("print(" + repr(json.dumps(result).replace("Infinity", "1e309")) + ")")
    assert client.post("/api/v1/decision/score", json=PAYLOAD).status_code == 503
    main.write_text("import sys; print('/private/key credential-fake',file=sys.stderr);sys.exit(2)")
    response = client.post("/api/v1/decision/score", json=PAYLOAD)
    assert response.status_code == 503 and "credential-fake" not in response.text


def test_child_has_no_inherited_credentials_or_user_pythonpath(configured, monkeypatch, tmp_path):
    client, main, *_ = configured
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/private/key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-secret")
    monkeypatch.setenv("PYTHONPATH", "/untrusted")
    path = tmp_path / "environment.json"
    main.write_text(
        "import os,json\nfrom pathlib import Path\nPath("
        + repr(str(path))
        + ").write_text(json.dumps(dict(os.environ)))\nprint("
        + repr(json.dumps(valid_result()))
        + ")"
    )
    assert client.post("/api/v1/decision/score", json=PAYLOAD).status_code == 200
    environment = json.loads(path.read_text())
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in environment
    assert "OPENROUTER_API_KEY" not in environment
    assert environment["PYTHONPATH"].endswith("worker/src")


def test_busy_is_rejected_without_queue_or_retry(configured):
    client, *_ = configured
    assert decision._busy.acquire(blocking=False)
    try:
        assert client.post("/api/v1/decision/score", json=PAYLOAD).status_code == 409
        status = client.get("/api/v1/decision/status").json()
        assert status["busy"] and not status["available"]
    finally:
        decision._busy.release()


@pytest.mark.skipif(os.name != "posix", reason="Decision worker requires POSIX")
@pytest.mark.parametrize("kind", ["timeout", "stdout", "stderr"])
def test_real_child_bounds_reap_child_and_release_slot(configured, monkeypatch, tmp_path, kind):
    client, main, *_ = configured
    pidfile = tmp_path / "pid"
    behavior = (
        "time.sleep(10)"
        if kind == "timeout"
        else f"print('x'*70000,file=sys.{kind});time.sleep(10)"
    )
    main.write_text(
        "import os,time,sys\nfrom pathlib import Path\nPath("
        + repr(str(pidfile))
        + ").write_text(str(os.getpid()))\n"
        + behavior
    )
    monkeypatch.setattr(decision, "DEADLINE", 0.4)
    started = time.monotonic()
    assert client.post("/api/v1/decision/score", json=PAYLOAD).status_code == 503
    assert time.monotonic() - started < 8
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)
    assert not decision._busy.locked()


def test_disconnect_cleans_owned_child(configured, tmp_path):
    _, main, root, model = configured
    pidfile = tmp_path / "pid"
    main.write_text(
        "import os,time\nfrom pathlib import Path\nPath("
        + repr(str(pidfile))
        + ").write_text(str(os.getpid()))\ntime.sleep(10)"
    )

    class Disconnected:
        async def is_disconnected(self):
            return pidfile.exists()

    async def probe():
        with pytest.raises(decision.HTTPException) as error:
            await decision.supervised(PAYLOAD, (Path(sys.executable), root, model), Disconnected())
        assert error.value.status_code == 499

    asyncio.run(probe())
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


def test_openapi_has_bounded_self_contained_request_schema(client):
    schema = client.get("/openapi.json").json()["paths"]["/api/v1/decision/score"]["post"][
        "requestBody"
    ]["content"]["application/json"]["schema"]
    assert "$ref" not in json.dumps(schema)
    assert schema["properties"]["criteria"]["minItems"] == 2
    assert schema["properties"]["criteria"]["maxItems"] == 8
    assert schema["properties"]["state"]["maxLength"] == 8000


def test_canonical_unicode_matches_worker_contract():
    payload = {**PAYLOAD, "state": "Gripper résumé 🤖"}
    assert decision.canonical(payload) == json.dumps(
        payload, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


@pytest.mark.skipif(os.name != "posix", reason="POSIX process cleanup")
def test_cancellation_during_spawn_and_repeated_cleanup_reaps(configured, monkeypatch, tmp_path):
    _, main, root, model = configured
    pidfile = tmp_path / "spawn-pid"
    main.write_text(
        "import os,time\nfrom pathlib import Path\nPath("
        + repr(str(pidfile))
        + ").write_text(str(os.getpid()))\ntime.sleep(20)"
    )
    original = asyncio.create_subprocess_exec

    class Connected:
        async def is_disconnected(self):
            return False

    async def probe():
        registered = asyncio.Event()

        async def delayed(*args, **kwargs):
            child = await original(*args, **kwargs)
            registered.set()
            await asyncio.sleep(0.15)
            return child

        monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed)
        task = asyncio.create_task(
            decision.supervised(PAYLOAD, (Path(sys.executable), root, model), Connected())
        )
        await registered.wait()
        task.cancel()
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(probe())
    if pidfile.exists():
        with pytest.raises(ProcessLookupError):
            os.kill(int(pidfile.read_text()), 0)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process cleanup")
def test_timeout_allows_cli_to_reap_its_model_child(configured, monkeypatch, tmp_path):
    client, main, *_ = configured
    pidfile = tmp_path / "grandchild"
    marker = tmp_path / "reaped"
    main.write_text(
        """import subprocess,sys,signal,time
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'])
Path("""
        + repr(str(pidfile))
        + """).write_text(str(child.pid))
def stop(*args):
    child.kill();child.wait()
    Path("""
        + repr(str(marker))
        + """).write_text('reaped')
    sys.exit(0)
signal.signal(signal.SIGTERM,stop)
time.sleep(20)
"""
    )
    monkeypatch.setattr(decision, "DEADLINE", 0.5)
    assert client.post("/api/v1/decision/score", json=PAYLOAD).status_code == 503
    assert marker.read_text() == "reaped"
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


def test_real_overlapping_requests_admit_only_one(configured, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    client, main, *_ = configured
    marker = tmp_path / "started"
    main.write_text(
        "import time\nfrom pathlib import Path\nPath("
        + repr(str(marker))
        + ").write_text('started')\ntime.sleep(.4)\nprint("
        + repr(json.dumps(valid_result()))
        + ")"
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(client.post, "/api/v1/decision/score", json=PAYLOAD)
        deadline = time.monotonic() + 3
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        assert client.post("/api/v1/decision/score", json=PAYLOAD).status_code == 409
        assert first.result(timeout=3).status_code == 200
    assert not decision._busy.locked()


def test_ignored_termination_forces_owned_child_cleanup(configured, tmp_path, monkeypatch):
    client, main, *_ = configured
    pidfile = tmp_path / "ignoring-pid"
    main.write_text(
        "import os,signal,time\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN)\nPath("
        + repr(str(pidfile))
        + ").write_text(str(os.getpid()))\ntime.sleep(20)"
    )
    monkeypatch.setattr(decision, "DEADLINE", 0.4)
    monkeypatch.setattr(decision, "CLEANUP_SECONDS", 0.1)
    assert client.post("/api/v1/decision/score", json=PAYLOAD).status_code == 503
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


def test_license_opt_in_is_required_even_with_all_paths(configured, monkeypatch):
    client, *_ = configured
    monkeypatch.setenv("FIREBIRD_DECISION_ACCEPT_LICENSE", "yes")
    assert not client.get("/api/v1/decision/status").json()["configured"]
    assert client.post("/api/v1/decision/score", json=PAYLOAD).status_code == 503


def test_temp_alias_is_resolved_for_strict_worker_file_admission(configured, monkeypatch, tmp_path):
    client, main, *_ = configured
    actual = tmp_path / "actual-temp"
    actual.mkdir()
    alias = tmp_path / "temp-alias"
    alias.symlink_to(actual, target_is_directory=True)
    monkeypatch.setattr(decision.tempfile, "tempdir", str(alias))
    main.write_text(
        'import sys\nfrom pathlib import Path\npath=Path(sys.argv[sys.argv.index("--request")+1])\n'
        "assert path == path.resolve()\nassert path.is_file()\nprint("
        + repr(json.dumps(valid_result()))
        + ")"
    )
    assert client.post("/api/v1/decision/score", json=PAYLOAD).status_code == 200
    assert list(actual.iterdir()) == []
