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
