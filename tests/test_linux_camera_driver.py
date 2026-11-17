"""Exercise the DKMS preparation step with real per-kernel files and no hardware."""

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

BUILD = Path(__file__).resolve().parents[1] / "deploy/linux/uvc/build_driver.py"


def prepare(tmp_path, kernel, driver_text):
    modules = tmp_path / "modules"
    source = modules / kernel / "source/drivers/media/usb/uvc"
    source.mkdir(parents=True)
    (source / "uvc_video.c").write_text(driver_text)
    (source / "uvcvideo.h").write_text("/* header */")
    (source / "Makefile").write_text("obj-m += uvcvideo.o")
    output = tmp_path / ("build-" + kernel)
    result = subprocess.run(
        [sys.executable, str(BUILD), kernel, "--modules", str(modules), "--prepare", str(output)],
        text=True,
        capture_output=True,
    )
    assert (source / "uvc_video.c").read_text() == driver_text
    return result, output


def test_dkms_prepares_the_requested_kernel_and_leaves_stock_driver_alone(tmp_path):
    for kernel in ("6.8.0-a", "7.0.0-b"):
        original = (
            f'/* {kernel} */\n#include "uvcvideo.h"\n'
            "\t\tbandwidth = stream->ctrl.dwMaxPayloadTransferSize;\n"
        )
        result, output = prepare(tmp_path, kernel, original)
        assert result.returncode == 0, result.stderr
        patched = (output / "uvc_video.c").read_text()
        assert kernel in patched
        assert "0x0abd" in patched and "0x8050" in patched
        assert "V4L2_PIX_FMT_MJPEG" in patched and "USB_SPEED_HIGH" in patched
        assert "1536" in patched
        assert (output / "Makefile").is_file()


@pytest.mark.parametrize("sites", [0, 2])
def test_unknown_upstream_layout_aborts_without_touching_stock(tmp_path, sites):
    original = (
        '#include "uvcvideo.h"\n'
        + "\t\tbandwidth = stream->ctrl.dwMaxPayloadTransferSize;\n" * sites
    )
    result, output = prepare(tmp_path, "7.1.0-test", original)
    assert result.returncode and "unsupported" in result.stderr.lower()
    assert not (output / "Makefile").exists()


def test_stock_module_signature_can_reuse_sources_across_kernel_updates(tmp_path, monkeypatch):
    driver = runpy.run_path(str(BUILD))
    stock = tmp_path / "kernel/drivers/media/usb/uvc/uvcvideo.ko.zst"
    stock.parent.mkdir(parents=True)
    stock.touch()
    for signature in ("0123456789ABCDEF0123456", "0123456789ABCDEF01234567"):
        monkeypatch.setattr(subprocess, "check_output", lambda *a, sig=signature, **kw: sig + "\n")
        assert driver["stock_cache"](tmp_path).name == signature
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: "\n")
    assert driver["stock_cache"](tmp_path) is None


def test_camera_enrollment_carries_across_old_updater_and_skips_rollback(tmp_path, monkeypatch):
    import asyncio
    import json
    from types import SimpleNamespace

    from vla_platform import robot_companion_updates
    from vla_platform.application_updates import ApplicationUpdates

    operation = SimpleNamespace(id="a" * 32, target_commit="b" * 40, phase="restarting")
    pairing = tmp_path / "pair"
    pairing.write_text(json.dumps({"url": "local"}))
    pairing.chmod(0o600)
    launched = []

    async def launch(*args):
        launched.append(args)
        return SimpleNamespace(wait=wait)

    async def wait():
        return 0

    async def exercise(next_phase, idle=True):
        states = iter([operation, SimpleNamespace(**{**vars(operation), "phase": next_phase})])
        obj = SimpleNamespace(
            installed=SimpleNamespace(commit="b" * 40), operation=lambda: next(states)
        )
        monkeypatch.setattr(robot_companion_updates, "idle", lambda _: idle)
        await ApplicationUpdates.install_camera_driver(obj, pairing, tmp_path, operation)

    async def advance(_):
        return

    monkeypatch.setattr(asyncio, "sleep", advance)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    asyncio.run(exercise("rolled_back"))
    asyncio.run(exercise("complete", idle=False))
    assert not launched
    asyncio.run(exercise("complete"))
    assert len(launched) == 1 and launched[0][-3:] == (
        "camera-driver-update",
        str(tmp_path),
        operation.id,
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Linux camera driver")
def test_camera_update_prompts_once_per_update_and_only_for_matching_hardware(
    tmp_path, monkeypatch
):
    import importlib.util
    import json
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "installer_update", BUILD.parents[2] / "install_robot_lab.py"
    )
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    op = "c" * 32
    calls = []
    monkeypatch.setattr(installer, "camera_pair_on_usb2", lambda: False)
    installer.camera_update(tmp_path, op)
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(installer, "camera_pair_on_usb2", lambda: True)
    enrolled = iter([False, True])
    monkeypatch.setattr(installer, "camera_driver_installed", lambda: next(enrolled))
    monkeypatch.setattr(installer.os, "geteuid", lambda: 501)
    monkeypatch.setenv("DISPLAY", ":1")
    monkeypatch.setattr(installer.shutil, "which", lambda tool: "/usr/bin/" + tool)
    monkeypatch.setattr(
        installer.subprocess,
        "run",
        lambda args, **kwargs: calls.append(args) or SimpleNamespace(returncode=0),
    )
    installer.camera_update(tmp_path, op)
    installer.camera_update(tmp_path, op)
    assert len(calls) == 1 and calls[0][0] == "gnome-terminal"
    assert calls[0][-1] == "--activate-if-unused"
    result = json.loads((tmp_path / ("camera-driver-" + op + ".json")).read_text())
    assert result["finished"] and "installed" in result["message"]
