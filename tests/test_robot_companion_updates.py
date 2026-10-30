"""Local service identity and source-only upgrade rollback; no real services run."""

import asyncio
import json

import pytest
from vla_platform import robot_companion_updates as updates
from vla_platform.application_updates import ApplicationUpdates, UpdateError


def private(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o600)
    return path


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setattr(updates.sys, "platform", "linux")
    configuration = tmp_path / "configuration"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(configuration))
    source, desired = tmp_path / "installed", tmp_path / "application"
    for tree, body in ((source, "# old\n"), (desired, "# new\n")):
        tree.mkdir()
        for name in ("server.py", "hardware.py", "assembled.py", "diagnostics.py"):
            (tree / name).write_text(body)
        (tree / "requirements-hardware.txt").write_text("pyserial==3.5\n")
    python = private(tmp_path / "runtime/bin/python", "runtime stays here")
    token = private(tmp_path / "private/companion-token", "x" * 48)
    data = tmp_path / "recordings"
    data.mkdir()
    (data / "saved.json").write_text("saved captures and calibration")
    pairing = {"url": "http://127.0.0.1:18101", "token": "x" * 48}
    pairing_path = private(tmp_path / "workspace/robot-lab-connection.json", json.dumps(pairing))
    arguments = [
        str(python),
        str(source / "server.py"),
        "--host",
        "127.0.0.1",
        "--port",
        "18101",
        "--token-file",
        str(token),
        "--data-dir",
        str(data),
    ]
    unit = private(
        configuration / "systemd/user" / updates.UNIT,
        "[Service]\nExecStart="
        + " ".join(json.dumps(a) for a in arguments)
        + "\nWorkingDirectory="
        + str(source)
        + "\n",
    )
    receipt = {
        "python": str(python),
        "source": str(source),
        "token_file": str(token),
        "data": str(data),
        "platform": "linux",
        "services": [updates.UNIT],
    }
    receipt_path = private(token.parent / "installation.json", json.dumps(receipt))
    info = {
        "FragmentPath": str(unit),
        "MainPID": "123",
        "ActiveState": "active",
        "DropInPaths": "",
        "WorkingDirectory": str(source),
    }
    commands = []

    def command(*args, **kwargs):
        commands.append(args)
        if args[:3] == ("systemctl", "--user", "show"):
            return "\n".join(key + "=" + value for key, value in info.items())
        return ""

    monkeypatch.setattr(updates, "command", command)
    monkeypatch.setattr(updates, "process_matches", lambda pid, args, **kwargs: args == arguments)
    monkeypatch.setattr(
        updates,
        "status",
        lambda pairing: {
            "version": "1",
            "backend": "feetech-sdk",
            "session": {"mode": "idle", "torque_enabled": False},
        },
    )
    return locals()


def controls(fixture):
    return [
        args[2]
        for args in fixture["commands"]
        if args[:2] == ("systemctl", "--user") and args[2] in {"start", "stop"}
    ]


def test_verified_local_upgrade_preserves_runtime_data_token_and_unit(service):
    preserved = {
        name: service[name].read_bytes()
        for name in ("python", "token", "unit", "receipt_path", "pairing_path")
    }
    value = updates.reconcile(service["desired"], service["pairing_path"], allow_update=True)
    assert value["state"] == "updated"
    assert value["managed"]
    assert controls(service) == ["stop", "start"]
    assert (service["source"] / "server.py").read_text() == "# new\n"
    backups = list(service["source"].parent.glob("installed.rollback-*"))
    assert len(backups) == 1 and (backups[0] / "server.py").read_text() == "# old\n"
    assert (service["data"] / "saved.json").read_text() == "saved captures and calibration"
    assert all(service[name].read_bytes() == body for name, body in preserved.items())


def test_health_failure_restores_previous_sources_and_service(service, monkeypatch):
    checks = []

    def health(*args):
        checks.append((service["source"] / "server.py").read_text())
        if len(checks) == 1:
            raise RuntimeError("new service failed")

    monkeypatch.setattr(updates, "wait_healthy", health)
    value = updates.reconcile(service["desired"], service["pairing_path"], allow_update=True)
    assert value["state"] == "needs_update" and "rolled back" in value["message"]
    assert checks == ["# new\n", "# old\n"]
    assert controls(service) == ["stop", "start", "stop", "start"]
    assert (service["source"] / "server.py").read_text() == "# old\n"


def test_uncertain_stop_restarts_and_verifies_previous_service(service, monkeypatch):
    actions = []
    verified = []

    def control(action):
        actions.append(action)
        if action == "stop":
            raise RuntimeError("systemd accepted stop but reply was lost")

    monkeypatch.setattr(updates, "control", control)
    monkeypatch.setattr(updates, "wait_healthy", lambda *args: verified.append(True))
    value = updates.reconcile(service["desired"], service["pairing_path"], allow_update=True)
    assert value["state"] == "needs_update" and "rolled back" in value["message"]
    assert actions == ["stop", "start"]
    assert verified == [True]
    assert (service["source"] / "server.py").read_text() == "# old\n"
    assert not list(service["source"].parent.glob("installed.candidate-*"))


def test_dependency_preflight_failure_never_stops_service_or_leaves_candidate(service, monkeypatch):
    original = updates.command

    def command(*args, **kwargs):
        if args[0] == str(service["python"]):
            raise RuntimeError("incompatible runtime")
        return original(*args, **kwargs)

    monkeypatch.setattr(updates, "command", command)
    value = updates.reconcile(service["desired"], service["pairing_path"], allow_update=True)
    assert value["state"] == "needs_update"
    assert controls(service) == []
    assert not list(service["source"].parent.glob("installed.candidate-*"))


@pytest.mark.parametrize("field,value", [("mode", "teleop"), ("torque_enabled", True)])
def test_active_service_is_not_stopped(service, monkeypatch, field, value):
    status = {
        "version": "1",
        "backend": "feetech-sdk",
        "session": {"mode": "idle", "torque_enabled": False, field: value},
    }
    monkeypatch.setattr(updates, "status", lambda pairing: status)
    result = updates.reconcile(service["desired"], service["pairing_path"], allow_update=True)
    assert result["state"] == "needs_update" and "Stop Robot Lab" in result["message"]
    assert controls(service) == []
    assert (service["source"] / "server.py").read_text() == "# old\n"


def test_other_pairing_token_never_updates_same_named_local_service(service):
    private(service["pairing_path"], json.dumps({**service["pairing"], "token": "y" * 48}))
    value = updates.reconcile(service["desired"], service["pairing_path"], allow_update=True)
    assert value["state"] == "needs_update" and not value["managed"]
    assert controls(service) == []


def test_existing_mismatched_receipt_blocks_automatic_adoption(service):
    private(service["receipt_path"], json.dumps({**service["receipt"], "source": "/unrelated"}))
    value = updates.reconcile(service["desired"], service["pairing_path"], allow_update=True)
    assert value["state"] == "needs_update"
    assert controls(service) == []


@pytest.mark.parametrize(
    "unsafe", ["user-config.json", "captures/recording.mp4", "runtime/lib/dependency.so"]
)
def test_mixed_source_and_user_data_tree_is_never_replaced(service, unsafe):
    private(service["source"] / unsafe, "preserve user data")
    value = updates.reconcile(service["desired"], service["pairing_path"], allow_update=True)
    assert value["state"] == "needs_update"
    assert controls(service) == []
    assert (service["source"] / unsafe).read_text() == "preserve user data"


def test_source_current_never_restarts_service(service):
    value = updates.reconcile(service["source"], service["pairing_path"], allow_update=True)
    assert value["state"] == "current"
    assert controls(service) == []


def test_unmanaged_application_reports_warning_without_source_change(service):
    value = updates.reconcile(service["desired"], service["pairing_path"], allow_update=False)
    assert value["state"] == "needs_update" and value["managed"]
    assert controls(service) == []


@pytest.mark.parametrize("assembled,verification", [("stopped", "stopped"), ("complete", "done")])
def test_completed_and_stopped_setup_allows_idle_update(
    service, monkeypatch, assembled, verification
):
    monkeypatch.setattr(
        updates,
        "status",
        lambda pairing: {
            "version": "1",
            "backend": "feetech-sdk",
            "session": {"mode": "idle", "torque_enabled": False},
            "assembled": {"phase": assembled, "motion_active": False},
            "verification": {"phase": verification, "motion_active": False},
        },
    )
    assert (
        updates.reconcile(service["desired"], service["pairing_path"], allow_update=True)["state"]
        == "updated"
    )


@pytest.mark.parametrize("section", ["assembled", "verification"])
def test_active_motion_cleanup_blocks_update_even_with_idle_session(service, monkeypatch, section):
    monkeypatch.setattr(
        updates,
        "status",
        lambda pairing: {
            "version": "1",
            "backend": "feetech-sdk",
            "session": {"mode": "idle", "torque_enabled": False},
            section: {"phase": "stopped", "motion_active": True},
        },
    )
    assert (
        updates.reconcile(service["desired"], service["pairing_path"], allow_update=True)["state"]
        == "needs_update"
    )
    assert controls(service) == []


def test_legacy_group_wrapper_and_owned_source_override_are_attested(service):
    service["receipt_path"].unlink()
    override = service["unit"].parent / (updates.UNIT + ".d") / "99-local-source.conf"
    inner = " ".join(service["arguments"])
    private(
        override,
        "[Service]\nExecStart=\nExecStart=/usr/bin/sg dialout -c "
        + json.dumps(inner)
        + "\nWorkingDirectory="
        + str(service["source"])
        + "\n",
    )
    service["info"]["DropInPaths"] = str(override)
    value = updates.reconcile(service["desired"], service["pairing_path"], allow_update=True)
    assert value["state"] == "updated"
    assert controls(service) == ["stop", "start"]
    assert override.exists()


def test_unverified_override_does_not_restart_service(service):
    override = service["unit"].parent / (updates.UNIT + ".d") / "99-local-source.conf"
    private(override, "[Service]\nEnvironment=PRIVATE_SETTING=example\n")
    service["info"]["DropInPaths"] = str(override)
    value = updates.reconcile(service["desired"], service["pairing_path"], allow_update=True)
    assert value["state"] == "needs_update"
    assert controls(service) == []


def test_missing_or_different_local_process_never_updates(service, monkeypatch):
    monkeypatch.setattr(updates, "process_matches", lambda *args, **kwargs: False)
    value = updates.reconcile(service["desired"], service["pairing_path"], allow_update=True)
    assert value["state"] == "needs_update"
    assert controls(service) == []


def test_group_wrapper_requires_exact_owned_descendant_arguments(tmp_path):
    expected = ["/runtime/python", "/source/server.py", "--port", "18101"]
    for pid, args, children in (
        (101, ["/usr/bin/sg", "dialout", "-c", "command"], "102"),
        (102, ["/bin/sh", "-c", "command"], "103"),
        (103, expected, ""),
    ):
        folder = tmp_path / str(pid)
        folder.mkdir()
        (folder / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in args) + b"\0")
        child_list = folder / "task" / str(pid) / "children"
        child_list.parent.mkdir(parents=True)
        child_list.write_text(children)
    assert updates.process_matches(101, expected, descendants=True, proc_root=tmp_path)
    assert not updates.process_matches(101, expected, proc_root=tmp_path)
    assert not updates.process_matches(
        101, [*expected, "--unexpected"], descendants=True, proc_root=tmp_path
    )


def test_command_deadline_caps_subprocess_timeout_and_expiration_blocks_launch(monkeypatch):
    calls = []
    monkeypatch.setattr(updates.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(
        updates.subprocess,
        "check_output",
        lambda *args, **kwargs: calls.append(kwargs["timeout"]) or "",
    )
    token = updates.DEADLINE.set(102.0)
    try:
        updates.command("fixture", timeout=20)
        assert calls == [2.0]
        updates.DEADLINE.set(99.0)
        with pytest.raises(RuntimeError, match="time budget"):
            updates.command("fixture", timeout=20)
        assert calls == [2.0]
    finally:
        updates.DEADLINE.reset(token)


def test_startup_warning_is_exposed_in_update_status(tmp_path, monkeypatch):
    pairing = private(tmp_path / "pairing.json", "{}")

    async def check():
        application = ApplicationUpdates(data_dir=tmp_path)
        monkeypatch.setattr(
            application, "configuration", lambda: (_ for _ in ()).throw(UpdateError("not managed"))
        )
        monkeypatch.setattr(
            updates,
            "reconcile",
            lambda *args, **kwargs: updates.result("needs_update", updates.HELP),
        )
        await application.reconcile_companion(pairing)
        assert application.status().companion_update.state == "needs_update"
        await application.close()

    asyncio.run(check())
