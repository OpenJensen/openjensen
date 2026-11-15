"""Local service identity and source-only upgrade rollback; no real services run."""

import asyncio
import json
from types import SimpleNamespace

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
    # No test reads this computer's own LaunchAgents.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
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
    # This service predates describing its own code, and no test may reach a real one.
    monkeypatch.setattr(updates, "reported_files", lambda pairing: None)
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
    "case,eligible",
    [
        ("private", True),
        ("secondary_member", False),
        ("primary_member", False),
        ("foreign_group", False),
        ("world_writable", False),
        ("unknown_group", False),
    ],
)
def test_group_writable_sources_require_proven_private_primary_group(
    service, monkeypatch, case, eligible
):
    import grp
    import pwd

    uid, gid = updates.os.getuid(), service["source"].stat().st_gid
    current = SimpleNamespace(pw_uid=uid, pw_gid=gid)
    other = SimpleNamespace(pw_uid=uid + 1, pw_gid=gid)
    service["source"].chmod(0o777 if case == "world_writable" else 0o775)
    service["unit"].chmod(0o664)
    monkeypatch.setattr(updates.os, "getgid", lambda: gid)
    monkeypatch.setattr(
        pwd,
        "getpwuid",
        lambda value: (
            SimpleNamespace(pw_uid=uid, pw_gid=gid + 1) if case == "foreign_group" else current
        ),
    )
    monkeypatch.setattr(
        pwd, "getpwall", lambda: [current, other] if case == "primary_member" else [current]
    )
    monkeypatch.setattr(pwd, "getpwnam", lambda name: other if name == "other" else current)

    def group(value):
        if case == "unknown_group":
            raise KeyError(value)
        return SimpleNamespace(
            gr_mem=["operator", "other"] if case == "secondary_member" else ["operator"]
        )

    monkeypatch.setattr(grp, "getgrgid", group)
    value = updates.reconcile(service["desired"], service["pairing_path"], allow_update=True)
    assert value["state"] == ("updated" if eligible else "needs_update")
    assert controls(service) == (["stop", "start"] if eligible else [])
    assert service["unit"].stat().st_mode & 0o777 == 0o664


def test_private_pairing_credential_cannot_be_group_readable(service):
    service["token"].chmod(0o660)
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


def reports(service, monkeypatch, files):
    """A Mac with no installed service of its own (a tunnel to another computer, for example):
    the paired service is not one this application may replace; it only answers for itself."""
    monkeypatch.setattr(updates.sys, "platform", "darwin")
    monkeypatch.setattr(updates, "reported_files", lambda pairing: files)


def test_a_service_on_another_platform_running_the_same_code_is_current(service, monkeypatch):
    reports(service, monkeypatch, updates.source_files(service["desired"]))
    value = updates.reconcile(service["desired"], service["pairing_path"], allow_update=True)
    assert value == {
        "state": "current",
        "message": "Robot control service matches this application.",
        "managed": False,
    }
    # Nothing was inspected through systemd and nothing was restarted.
    assert service["commands"] == []


def test_an_installed_service_with_fewer_files_than_the_application_is_still_current(
    service, monkeypatch
):
    # The installer ships the modules the service loads; the application's folder also
    # holds helpers of its own and the pinned requirements.
    (service["desired"] / "collection_mock.py").write_text("# used by the application only\n")
    files = updates.source_files(service["desired"])
    del files["collection_mock.py"], files["requirements-hardware.txt"]
    reports(service, monkeypatch, files)
    value = updates.reconcile(service["desired"], service["pairing_path"])
    assert value["state"] == "current"


def test_a_service_running_other_code_says_what_to_do(service, monkeypatch):
    reports(service, monkeypatch, updates.source_files(service["source"]))
    value = updates.reconcile(service["desired"], service["pairing_path"], allow_update=True)
    assert value["state"] == "needs_update" and not value["managed"]
    assert "different code" in value["message"]
    assert "install_robot_lab.py" in value["message"]
    assert (service["source"] / "server.py").read_text() == "# old\n"
    assert controls(service) == []


def test_a_service_too_old_to_describe_itself_still_needs_an_update(service, monkeypatch):
    reports(service, monkeypatch, None)
    value = updates.reconcile(service["desired"], service["pairing_path"], allow_update=True)
    assert value == updates.result("needs_update", updates.HELP)


def test_a_service_that_cannot_be_reached_is_not_called_out_of_date(service, monkeypatch):
    monkeypatch.setattr(updates.sys, "platform", "darwin")

    def refused(pairing):
        raise ConnectionRefusedError("robot service is not running")

    monkeypatch.setattr(updates, "reported_files", refused)
    assert updates.reconcile(service["desired"], service["pairing_path"]) is None


def test_a_service_answers_over_its_own_address_and_an_older_one_has_no_answer(monkeypatch):
    import io
    from urllib.error import HTTPError

    asked = []

    class Reply(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def answer(request, timeout):
        asked.append((request.full_url, request.get_header("Authorization")))
        return Reply(json.dumps({"files": {"server.py": "a" * 64}}).encode())

    pairing = {"url": "http://127.0.0.1:18101", "token": "secret"}
    monkeypatch.setattr(updates, "urlopen", answer)
    assert updates.reported_files(pairing) == {"server.py": "a" * 64}
    assert asked == [("http://127.0.0.1:18101/v1/source", "Bearer secret")]

    def unknown(request, timeout):
        raise HTTPError(request.full_url, 404, "Not found", {}, None)

    monkeypatch.setattr(updates, "urlopen", unknown)
    assert updates.reported_files(pairing) is None


def test_the_fingerprints_follow_code_and_ignore_notes_tests_and_caches(tmp_path):
    tree = tmp_path / "service"
    (tree / "_vendor").mkdir(parents=True)
    (tree / "server.py").write_text("print('one')\n")
    (tree / "_vendor/tool.py").write_text("VALUE = 1\n")
    (tree / "requirements-hardware.txt").write_text("pyserial==3.5\n")
    first = updates.source_files(tree)
    assert list(first) == ["_vendor/tool.py", "requirements-hardware.txt", "server.py"]
    (tree / "README.md").write_text("notes")
    (tree / "server.py.before-update").write_text("an operator's saved copy")
    (tree / "tests").mkdir()
    (tree / "tests/test_server.py").write_text("assert True\n")
    (tree / "__pycache__").mkdir()
    (tree / "__pycache__/server.cpython-314.pyc").write_bytes(b"cache")
    assert updates.source_files(tree) == first
    (tree / "_vendor/tool.py").write_text("VALUE = 2\n")
    changed = updates.source_files(tree)
    assert changed["_vendor/tool.py"] != first["_vendor/tool.py"]
    assert changed["server.py"] == first["server.py"]


def test_the_service_and_the_application_compute_the_same_fingerprints():
    import importlib.util

    source = updates.Path(__file__).resolve().parents[1] / "workers/robot_lab"
    spec = importlib.util.spec_from_file_location("source_identity", source / "source_identity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = module.source_files(source)
    assert files == updates.source_files(source)
    assert {"server.py", "hardware.py", "source_identity.py"} <= files.keys()


def test_checking_for_updates_asks_a_hand_installed_service_again(tmp_path, monkeypatch):
    """Reinstalling the robot service must clear the warning without restarting the application."""
    pairing = private(tmp_path / "pairing.json", "{}")
    answers = iter(
        [
            updates.result("needs_update", updates.MISMATCH),
            updates.result("current", "Robot control service matches this application."),
        ]
    )

    async def check():
        application = ApplicationUpdates(data_dir=tmp_path)
        monkeypatch.setattr(
            application, "configuration", lambda: (_ for _ in ()).throw(UpdateError("not managed"))
        )
        monkeypatch.setattr(updates, "reconcile", lambda *args, **kwargs: next(answers))
        monkeypatch.setattr(updates, "observe", lambda *args, **kwargs: next(answers))
        await application.reconcile_companion(pairing)
        assert application.status().companion_update.state == "needs_update"
        await application.observe_companion()
        assert application.status().companion_update.state == "current"
        # A service the application manages itself was settled at startup and is left alone.
        application.companion_update.managed = True
        await application.observe_companion()
        assert application.status().companion_update.state == "current"
        await application.close()

    asyncio.run(check())


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


def process_fixture(root, pid, args, children="", uids=None, started=12345):
    folder = root / str(pid)
    folder.mkdir()
    (folder / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in args) + b"\0")
    identities = [updates.os.getuid()] * 4 if uids is None else uids
    (folder / "status").write_text(f"Pid:\t{pid}\nUid:\t" + "\t".join(map(str, identities)) + "\n")
    (folder / "stat").write_text(f"{pid} (fixture process) S " + "0 " * 18 + str(started))
    child_list = folder / "task" / str(pid) / "children"
    child_list.parent.mkdir(parents=True)
    child_list.write_text(children)
    return folder


def test_group_wrapper_requires_exact_owned_descendant_arguments(tmp_path):
    expected = ["/runtime/python", "/source/server.py", "--port", "18101"]
    for pid, args, children in (
        (101, ["/usr/bin/sg", "dialout", "-c", "command"], "102"),
        (102, ["/bin/sh", "-c", "command"], "103"),
        (103, expected, ""),
    ):
        process_fixture(tmp_path, pid, args, children)
    assert updates.process_matches(101, expected, descendants=True, proc_root=tmp_path)
    assert not updates.process_matches(101, expected, proc_root=tmp_path)
    assert not updates.process_matches(
        101, [*expected, "--unexpected"], descendants=True, proc_root=tmp_path
    )


def test_nondumpable_root_owned_proc_directory_can_hold_current_user_process(tmp_path, monkeypatch):
    args = ["/runtime/python", "/source/server.py"]
    folder = process_fixture(tmp_path, 101, args)
    original = updates.Path.stat
    monkeypatch.setattr(
        updates.Path,
        "stat",
        lambda self, **kwargs: (
            SimpleNamespace(st_uid=0) if self == folder else original(self, **kwargs)
        ),
    )
    assert folder.stat().st_uid == 0
    assert updates.process_matches(101, args, proc_root=tmp_path)


def test_exact_managed_setuid_sg_wrapper_can_lead_to_owned_python_child(tmp_path):
    expected = ["/runtime/python", "/source/server.py"]
    wrapper = ["/usr/bin/sg", "dialout", "-c", " ".join(expected)]
    process_fixture(tmp_path, 101, wrapper, "102", uids=[updates.os.getuid(), 0, 0, 0])
    process_fixture(tmp_path, 102, expected)
    assert updates.process_matches(
        101, expected, descendants=True, wrapper_arguments=wrapper, proc_root=tmp_path
    )
    with pytest.raises(ValueError, match="not owned"):
        updates.process_matches(101, expected, descendants=True, proc_root=tmp_path)
    with pytest.raises(ValueError, match="does not match"):
        updates.process_matches(
            101,
            expected,
            descendants=True,
            wrapper_arguments=[*wrapper[:3], "/different/command"],
            proc_root=tmp_path,
        )


def test_privileged_wrapper_never_counts_as_the_owned_python_process(tmp_path):
    wrapper = ["/usr/bin/sg", "dialout", "-c", "command"]
    process_fixture(tmp_path, 101, wrapper, uids=[updates.os.getuid(), 0, 0, 0])
    assert not updates.process_matches(
        101, wrapper, descendants=True, wrapper_arguments=wrapper, proc_root=tmp_path
    )


def test_root_descendant_is_rejected_even_below_attested_sg_wrapper(tmp_path):
    expected = ["/runtime/python", "/source/server.py"]
    wrapper = ["/usr/bin/sg", "dialout", "-c", " ".join(expected)]
    process_fixture(tmp_path, 101, wrapper, "102", uids=[updates.os.getuid(), 0, 0, 0])
    process_fixture(tmp_path, 102, expected, uids=[updates.os.getuid(), 0, 0, 0])
    with pytest.raises(ValueError, match="not owned"):
        updates.process_matches(
            101, expected, descendants=True, wrapper_arguments=wrapper, proc_root=tmp_path
        )


@pytest.mark.parametrize("changed", [0, 1, 2, 3])
def test_foreign_real_effective_saved_or_filesystem_uid_is_rejected(tmp_path, changed):
    args = ["/runtime/python", "/source/server.py"]
    uids = [updates.os.getuid()] * 4
    uids[changed] += 1
    process_fixture(tmp_path, 101, args, uids=uids)
    with pytest.raises(ValueError, match="not owned"):
        updates.process_matches(101, args, proc_root=tmp_path)


def test_pid_reuse_during_command_inspection_is_rejected(tmp_path, monkeypatch):
    args = ["/runtime/python", "/source/server.py"]
    folder = process_fixture(tmp_path, 101, args)
    original = updates.Path.read_bytes

    def read_bytes(path):
        result = original(path)
        if path == folder / "cmdline":
            (folder / "stat").write_text("101 (replacement) S " + "0 " * 18 + "99999")
        return result

    monkeypatch.setattr(updates.Path, "read_bytes", read_bytes)
    with pytest.raises(ValueError, match="changed during inspection"):
        updates.process_matches(101, args, proc_root=tmp_path)


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


@pytest.fixture
def mac_service(service, tmp_path, monkeypatch):
    """The same installed service, registered by the installer as a macOS LaunchAgent."""
    import plistlib

    monkeypatch.setattr(updates.sys, "platform", "darwin")
    agent = tmp_path / "home/Library/LaunchAgents" / (updates.LABEL + ".plist")
    agent.parent.mkdir(parents=True)
    definition = {
        "Label": updates.LABEL,
        "ProgramArguments": service["arguments"],
        "WorkingDirectory": str(service["source"]),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 10,
        "EnvironmentVariables": {"PATH": "/usr/bin:/bin", "PYTHONUNBUFFERED": "1"},
        "StandardOutPath": str(tmp_path / "logs/companion.log"),
        "StandardErrorPath": str(tmp_path / "logs/companion-error.log"),
    }
    agent.write_bytes(plistlib.dumps(definition))
    agent.chmod(0o600)
    receipt = dict(json.loads(service["receipt_path"].read_text()))
    receipt.update(platform="darwin", services=[updates.LABEL, updates.LABEL + ".tunnel"])
    service["receipt_path"].write_text(json.dumps(receipt))
    job = {"loaded": True, "path": str(agent), "pid": 4242, "alive": True, "uid": None}
    commands = []

    def command(*args, **kwargs):
        commands.append(args)
        if args[:2] == ("launchctl", "print"):
            if not job["loaded"]:
                raise updates.subprocess.CalledProcessError(113, args)
            # Nested blocks repeat field names; only the top level describes this job.
            return (
                f"gui/501/{updates.LABEL} = {{\n\tactive count = 1\n\tpath = {job['path']}\n"
                "\tstate = running\n\tendpoints = {\n\t\tstate = active\n\t}\n"
                f"\tpid = {job['pid']}\n"
                "\tinherited environment = {\n\t\tSECRET => hidden\n\t}\n}\n"
            )
        if args[:2] == ("launchctl", "bootout"):
            job.update(loaded=False, alive=False)
        if args[:2] == ("launchctl", "bootstrap"):
            job.update(loaded=True, alive=True, pid=job["pid"] + 1)
        if args[0] == "ps":
            uid = job["uid"] if job["uid"] is not None else updates.os.getuid()
            return f"{uid} {uid} {uid} Sat Nov 14 12:11:55 2026\n"
        return ""

    def kill(pid, signal):
        if pid != job["pid"] or not job["alive"]:
            raise ProcessLookupError(pid)

    monkeypatch.setattr(updates, "command", command)
    monkeypatch.setattr(updates.os, "kill", kill)
    monkeypatch.setattr(
        updates,
        "process_arguments",
        lambda pid: definition["ProgramArguments"] if pid == job["pid"] else [],
    )
    return {**service, "agent": agent, "definition": definition, "job": job, "commands": commands}


def launchd(fixture):
    return [args[1] for args in fixture["commands"] if args[1] in {"bootout", "bootstrap"}]


def test_installer_launch_agent_on_a_mac_updates_with_the_application(mac_service):
    preserved = {
        name: mac_service[name].read_bytes()
        for name in ("python", "token", "agent", "receipt_path", "pairing_path")
    }
    value = updates.reconcile(
        mac_service["desired"], mac_service["pairing_path"], allow_update=True
    )
    assert value == {
        "state": "updated",
        "message": "Robot control service updated and verified with this application.",
        "managed": True,
    }
    assert launchd(mac_service) == ["bootout", "bootstrap"]
    assert (mac_service["source"] / "server.py").read_text() == "# new\n"
    assert (mac_service["data"] / "saved.json").read_text() == "saved captures and calibration"
    assert all(mac_service[name].read_bytes() == body for name, body in preserved.items())


def test_a_mac_service_already_running_this_code_is_left_alone(mac_service):
    value = updates.reconcile(mac_service["source"], mac_service["pairing_path"], allow_update=True)
    assert value["state"] == "current" and value["managed"]
    assert launchd(mac_service) == []


@pytest.mark.parametrize(
    "change",
    [
        "extra environment",
        "alternate program",
        "loaded from another file",
        "another account",
        "different arguments",
        "no receipt",
    ],
)
def test_a_mac_service_that_cannot_be_proven_ours_is_never_restarted(mac_service, change):
    import plistlib

    definition = dict(mac_service["definition"])
    if change == "extra environment":
        definition["EnvironmentVariables"] = {"DYLD_INSERT_LIBRARIES": "/tmp/other.dylib"}
    elif change == "alternate program":
        definition["Program"] = "/bin/sh"
    elif change == "loaded from another file":
        mac_service["job"]["path"] = "/Library/LaunchAgents/other.plist"
    elif change == "another account":
        mac_service["job"]["uid"] = updates.os.getuid() + 1
    elif change == "different arguments":
        definition["ProgramArguments"] = [*definition["ProgramArguments"][:-1], "/elsewhere"]
    else:
        mac_service["receipt_path"].unlink()
    mac_service["agent"].write_bytes(plistlib.dumps(definition))
    value = updates.reconcile(
        mac_service["desired"], mac_service["pairing_path"], allow_update=True
    )
    assert value == updates.result("needs_update", updates.HELP)
    assert launchd(mac_service) == []
    assert (mac_service["source"] / "server.py").read_text() == "# old\n"


def test_a_mac_service_that_fails_to_start_gets_its_previous_code_back(mac_service, monkeypatch):
    checks = []

    def health(*args):
        checks.append((mac_service["source"] / "server.py").read_text())
        if len(checks) == 1:
            raise RuntimeError("new service failed")

    monkeypatch.setattr(updates, "wait_healthy", health)
    value = updates.reconcile(
        mac_service["desired"], mac_service["pairing_path"], allow_update=True
    )
    assert "rolled back" in value["message"]
    assert checks == ["# new\n", "# old\n"]
    assert launchd(mac_service) == ["bootout", "bootstrap", "bootout", "bootstrap"]
    assert (mac_service["source"] / "server.py").read_text() == "# old\n"


@pytest.mark.skipif(updates.sys.platform != "darwin", reason="Reads arguments the macOS way")
def test_mac_process_arguments_keep_spaces_exactly(tmp_path):
    import subprocess
    import sys

    folder = tmp_path / "Application Support"
    folder.mkdir()
    arguments = [sys.executable, "-c", "import time; time.sleep(30)", "--data-dir", str(folder)]
    process = subprocess.Popen(arguments)
    try:
        assert updates.process_arguments(process.pid) == arguments
        assert updates.mac_process_matches(process.pid, arguments)
        assert not updates.mac_process_matches(process.pid, arguments[:-1])
    finally:
        process.terminate()
        process.wait(timeout=5)
