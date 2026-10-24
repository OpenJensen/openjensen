"""Companion installer lifecycle and credential checks use no real services or packages."""

import ast
import importlib.util
import os
import plistlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "robot_lab_installer", Path(__file__).parents[1] / "deploy/install_robot_lab.py"
)
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


def test_private_token_refuses_symlinks_public_permissions_and_invalid_contents(tmp_path):
    token = tmp_path / "token"
    token.write_text("t" * 48)
    token.chmod(0o600)
    assert installer.read_private_token(token) == "t" * 48
    token.chmod(0o644)
    with pytest.raises(ValueError, match="owned private"):
        installer.read_private_token(token)
    token.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(token)
    with pytest.raises(OSError):
        installer.read_private_token(link)
    token.write_text("short")
    with pytest.raises(ValueError, match="invalid format"):
        installer.read_private_token(token)


@pytest.mark.parametrize("host_platform", ["darwin", "linux"])
def test_reinstall_stops_only_managed_services_before_runtime_mutation(
    tmp_path, monkeypatch, capsys, host_platform
):
    home = tmp_path / "home with $dollars and %percent"
    source = tmp_path / "source"
    source.mkdir()
    for name in (*installer.SOURCE_MODULES, *installer.SUPPORT_FILES):
        (source / name).write_text("# supplied source module\n")
    vendor = source / "_vendor/servotools"
    vendor.mkdir(parents=True)
    for name in installer.VENDOR_FILES:
        (vendor / name).write_text("licensed source\n")
    (vendor / "__pycache__").mkdir()
    (vendor / "__pycache__/ignored.pyc").touch()
    prefix = tmp_path / "component"
    private = prefix / "private"
    private.mkdir(parents=True)
    private.chmod(0o700)
    token = private / "companion-token"
    token.write_text("t" * 48 + "\n")
    token.chmod(0o600)
    runtime = tmp_path / "runtime"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "bin/python").touch()
    arrow = tmp_path / "reader-site"
    (arrow / "pyarrow").mkdir(parents=True)
    site = runtime / "site-packages"
    site.mkdir()
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="loaded" if kwargs.get("text") else b"")

    monkeypatch.setattr(sys, "platform", host_platform)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr("shutil.which", lambda command: "/bin/" + command)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr("shutil.disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr("subprocess.run", run)
    monkeypatch.setattr("subprocess.check_output", lambda *a, **k: str(site) + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "install_robot_lab.py",
            "--source",
            str(source),
            "--prefix",
            str(prefix),
            "--runtime",
            str(runtime),
            "--offline",
            "--arrow-site-packages",
            str(arrow),
            "--ssh-host",
            "gpu-server",
            "--no-start",
        ],
    )
    installer.main()
    pip = next(
        index for index, command in enumerate(calls) if command[:3] == ["/bin/uv", "pip", "install"]
    )
    if host_platform == "darwin":
        stops = [
            index for index, command in enumerate(calls) if command[:2] == ["launchctl", "bootout"]
        ]
        assert {calls[index][-1] for index in stops} == {
            f"gui/{os.getuid()}/" + label for label in installer.SERVICE_LABELS
        }
    else:
        stops = [
            index
            for index, command in enumerate(calls)
            if command[:3] == ["/bin/systemctl", "--user", "stop"]
        ]
        assert {calls[index][-1] for index in stops} == set(installer.LINUX_SERVICES)
        unit = (home / ".config/systemd/user" / installer.LINUX_SERVICES[0]).read_text()
        assert "127.0.0.1" in unit and "UMask=0077" in unit and "Restart=on-failure" in unit
        assert "$$dollars" in unit and "%%percent" in unit
        tunnel = (home / ".config/systemd/user" / installer.LINUX_SERVICES[1]).read_text()
        assert "127.0.0.1:18101:127.0.0.1:18101" in tunnel
        assert "127.0.0.1:18096:127.0.0.1:8096" in tunnel
        receipt = __import__("json").loads((private / "installation.json").read_text())
        assert receipt["data"] == str(home / ".local/share/OpenJensen/robot-lab")
    assert len(stops) == 2 and max(stops) < pip
    assert "--offline" in calls[pip]
    assert not any(
        command[:2] == ["launchctl", "bootstrap"] or "--now" in command for command in calls
    )
    assert token.read_text() == "t" * 48 + "\n" and "t" * 48 not in capsys.readouterr().out
    assert all((prefix / "source" / name).is_file() for name in installer.SOURCE_MODULES)
    assert all(
        (prefix / "source/_vendor/servotools" / name).is_file() for name in installer.VENDOR_FILES
    )
    assert not (prefix / "source/_vendor/servotools/__pycache__").exists()
    if host_platform == "darwin":
        with (home / "Library/LaunchAgents/com.openjensen.robot-lab.plist").open("rb") as stream:
            service = plistlib.load(stream)
        assert service["ProgramArguments"][1].endswith("/source/server.py")


def test_missing_dispatch_module_fails_before_creating_files_or_stopping_services(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "server.py").touch()
    prefix = tmp_path / "never-created"
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("shutil.disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: pytest.fail("Invalid source stopped a service")
    )
    monkeypatch.setattr(
        sys, "argv", ["install_robot_lab.py", "--source", str(source), "--prefix", str(prefix)]
    )
    with pytest.raises(SystemExit) as error:
        installer.main()
    assert error.value.code == 2 and not prefix.exists()


@pytest.mark.parametrize("failure", ["missing-license", "symlink-module", "symlink-vendor"])
def test_vendor_sources_are_complete_regular_and_validated_before_any_mutation(
    tmp_path, monkeypatch, failure
):
    source = tmp_path / "source"
    source.mkdir()
    for name in (*installer.SOURCE_MODULES, *installer.SUPPORT_FILES):
        (source / name).write_text("# source\n")
    vendor = source / "_vendor/servotools"
    vendor.mkdir(parents=True)
    for name in installer.VENDOR_FILES:
        (vendor / name).write_text("licensed source\n")
    if failure == "missing-license":
        (vendor / "LICENSE").unlink()
    elif failure == "symlink-module":
        (source / "assembled.py").unlink()
        (source / "assembled.py").symlink_to(source / "server.py")
    else:
        vendor.rename(source / "original-vendor")
        vendor.symlink_to(source / "original-vendor", target_is_directory=True)
    prefix = tmp_path / "must-not-exist"
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("shutil.disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: pytest.fail("Invalid vendor source stopped services")
    )
    monkeypatch.setattr(
        sys, "argv", ["install_robot_lab.py", "--source", str(source), "--prefix", str(prefix)]
    )
    with pytest.raises(SystemExit):
        installer.main()
    assert not prefix.exists()


def test_installer_includes_local_dependencies_of_shipped_modules():
    source = Path(__file__).parents[1] / "workers/robot_lab"
    shipped = set(installer.SOURCE_MODULES)
    for name in shipped:
        tree = ast.parse((source / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                dependency = node.module.split(".")[0] + ".py"
                if (source / dependency).is_file():
                    assert dependency in shipped, f"{name} requires {dependency} in the installer"


def test_linux_first_offline_preparation_does_not_require_a_user_bus(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "linux_units_dir", lambda: tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: None)
    assert installer.stop_linux_services(no_start=True) is None
    with pytest.raises(ValueError, match="user manager"):
        installer.stop_linux_services(no_start=False)
    (tmp_path / installer.LINUX_SERVICES[0]).touch()
    with pytest.raises(ValueError, match="user manager"):
        installer.stop_linux_services(no_start=True)


def test_linux_service_enable_and_tunnel_removal(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "linux_units_dir", lambda: tmp_path / "units")
    calls = []
    monkeypatch.setattr("subprocess.run", lambda args, **kw: calls.append(args))
    arguments = [
        tmp_path / "python",
        tmp_path / "source",
        tmp_path / "token",
        tmp_path / "data",
        tmp_path / "logs",
    ]
    installer.install_linux_services(*arguments, "xbox-360", False, "/bin/systemctl")
    assert calls[-1] == ["/bin/systemctl", "--user", "enable", "--now", *installer.LINUX_SERVICES]
    installer.install_linux_services(*arguments, None, False, "/bin/systemctl")
    assert ["/bin/systemctl", "--user", "disable", installer.LINUX_SERVICES[1]] in calls
    assert not (tmp_path / "units" / installer.LINUX_SERVICES[1]).exists()
    assert (tmp_path / "units" / installer.LINUX_SERVICES[0]).stat().st_mode & 0o777 == 0o600


def test_service_paths_cannot_inject_unit_directives():
    with pytest.raises(ValueError, match="control characters"):
        installer.unit_quote("/tmp/source\nExecStart=/bin/false")
    quoted = installer.unit_quote('/tmp/with "quotes" and \\slashes', command=True)
    assert __import__("json").loads(quoted) == '/tmp/with "quotes" and \\slashes'


def test_linux_missing_units_do_not_require_success_exit_status(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("shutil.which", lambda _: "/bin/systemctl")

    def run(arguments, **kwargs):
        calls.append(arguments)
        return (
            SimpleNamespace(returncode=1, stdout="not-found\n")
            if "show" in arguments
            else SimpleNamespace(returncode=0)
        )

    monkeypatch.setattr("subprocess.run", run)
    assert installer.stop_linux_services(False) == "/bin/systemctl"
    assert not any("stop" in command for command in calls)


@pytest.mark.skipif(sys.platform != "linux", reason="systemd is Linux-only")
def test_generated_linux_units_pass_native_systemd_validation(tmp_path, monkeypatch):
    import shutil
    import subprocess

    analyzer = shutil.which("systemd-analyze")
    if not analyzer:
        pytest.skip("systemd-analyze is unavailable")
    source = tmp_path / 'source with $literal %percent "quotes"'
    source.mkdir()
    (source / "server.py").touch()
    runtime = tmp_path / "runtime-dir"
    runtime.mkdir(mode=0o700)
    units = tmp_path / "units"
    monkeypatch.setattr(installer, "linux_units_dir", lambda: units)
    installer.install_linux_services(
        Path(sys.executable),
        source,
        tmp_path / "token",
        tmp_path / "data",
        tmp_path / "logs",
        "xbox-360",
        True,
        None,
    )
    checked = subprocess.run(
        [analyzer, "--user", "verify", *[str(units / name) for name in installer.LINUX_SERVICES]],
        env={**os.environ, "XDG_RUNTIME_DIR": str(runtime)},
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr


def test_installer_bootstraps_with_distribution_python_310():
    # uv installs the companion Python; the installer must parse before that exists.
    root = Path(__file__).parents[1]
    for relative in (
        "deploy/install_robot_lab.py",
        "deploy/linux/install_robot_lab.py",
        "deploy/macos/install_robot_lab.py",
    ):
        ast.parse((root / relative).read_text(), feature_version=(3, 10))
