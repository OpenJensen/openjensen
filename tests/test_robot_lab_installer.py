"""Mac installer lifecycle and credential checks use no real services or packages."""

import importlib.util
import os
import plistlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "robot_lab_installer", Path(__file__).parents[1] / "deploy/macos/install_robot_lab.py"
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


def test_reinstall_stops_only_managed_services_before_runtime_mutation(
    tmp_path, monkeypatch, capsys
):
    home = tmp_path / "home"
    source = tmp_path / "source"
    source.mkdir()
    for name in installer.SOURCE_MODULES:
        (source / name).write_text("# supplied source module\n")
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
        return SimpleNamespace(returncode=0, stdout=b"")

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/uv")
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
            "xbox-360",
            "--no-start",
        ],
    )
    installer.main()
    bootouts = [
        index for index, command in enumerate(calls) if command[:2] == ["launchctl", "bootout"]
    ]
    pip = next(
        index for index, command in enumerate(calls) if command[:3] == ["/bin/uv", "pip", "install"]
    )
    assert len(bootouts) == 2 and max(bootouts) < pip
    assert {calls[index][-1] for index in bootouts} == {
        f"gui/{os.getuid()}/" + label for label in installer.SERVICE_LABELS
    }
    assert "--offline" in calls[pip]
    assert not any(command[:2] == ["launchctl", "bootstrap"] for command in calls)
    assert token.read_text() == "t" * 48 + "\n" and "t" * 48 not in capsys.readouterr().out
    assert all((prefix / "source" / name).is_file() for name in installer.SOURCE_MODULES)
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
