#!/usr/bin/env python3
"""Prepare the target kernel's UVC sources in a private DKMS build directory."""

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

UVC = Path("drivers/media/usb/uvc")
SOURCE_NAME = re.compile(r"(?:Makefile|Kbuild|[A-Za-z0-9_-]+\.[ch])")
INCLUDE = '#include "uvcvideo.h"'
RESERVE = "\t\tbandwidth = stream->ctrl.dwMaxPayloadTransferSize;"
CAP = """
/* These USB 2 cameras advertise the largest alternate setting for every MJPEG mode. */
static unsigned int mjpeg_bandwidth_cap = 1536;
module_param(mjpeg_bandwidth_cap, uint, 0444);
MODULE_PARM_DESC(mjpeg_bandwidth_cap, "0abd:8050 MJPEG bytes per microframe (0 disables)");
"""
LIMIT = """
\t\tif (mjpeg_bandwidth_cap && stream->dev->udev->speed == USB_SPEED_HIGH &&
\t\t    le16_to_cpu(stream->dev->udev->descriptor.idVendor) == 0x0abd &&
\t\t    le16_to_cpu(stream->dev->udev->descriptor.idProduct) == 0x8050 &&
\t\t    stream->cur_format->fcc == V4L2_PIX_FMT_MJPEG)
\t\t\tbandwidth = min_t(u32, bandwidth, mjpeg_bandwidth_cap);"""


def read_url(url):
    with urlopen(url, timeout=25) as response:
        data = response.read(2_000_001)
    if len(data) > 2_000_000:
        raise ValueError("Unexpectedly large Ubuntu UVC source file")
    return data


def ubuntu_sources(kernel, output):
    # Use the target binary package, including its ABI and flavour, even when DKMS
    # runs before the new kernel has booted. Never fetch the tip of a branch.
    release = {}
    for line in Path("/etc/os-release").read_text().splitlines():
        key, _, value = line.partition("=")
        release[key] = value.strip('"')
    suite = release.get("UBUNTU_CODENAME") or release.get("VERSION_CODENAME", "")
    if release.get("ID") != "ubuntu" or not re.fullmatch("[a-z]+", suite):
        raise ValueError("Install matching kernel source at /usr/src/openjensen-uvc/" + kernel)
    package, version = subprocess.check_output(
        ["dpkg-query", "-W", "-f=${source:Package} ${source:Version}", "linux-modules-" + kernel],
        text=True,
    ).split()
    if not re.fullmatch(r"linux(?:-[a-z0-9.-]+)?", package) or not re.fullmatch(
        r"[0-9][0-9.+~a-z-]*", version
    ):
        raise ValueError("Unsupported Ubuntu kernel package")
    tag = "Ubuntu" + package.removeprefix("linux") + "-" + version.replace("~", "_")
    repo = "https://git.launchpad.net/~ubuntu-kernel/ubuntu/+source/linux/+git/" + suite
    ref = "refs/tags/" + tag
    listing = subprocess.check_output(
        ["git", "ls-remote", "--tags", repo, ref, ref + "^{}"], text=True, timeout=45
    )
    refs = dict(line.split()[::-1] for line in listing.splitlines())
    commit = refs.get(ref + "^{}", refs.get(ref, ""))
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("No exact Ubuntu kernel source tag " + tag)
    tree = read_url(f"{repo}/tree/{UVC}?id={commit}").decode()
    names = sorted(
        {
            n
            for n in re.findall(r"class='ls-blob[^']*'[^>]*>([^<]+)</a>", tree)
            if SOURCE_NAME.fullmatch(n)
        }
    )
    if not {"Makefile", "uvc_video.c", "uvcvideo.h"}.issubset(names):
        raise ValueError("Incomplete Ubuntu UVC source listing")

    def fetch(name):
        (output / name).write_bytes(read_url(f"{repo}/plain/{UVC}/{quote(name)}?id={commit}"))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, names))


def stock_cache(module_dir):
    candidates = list((module_dir / "kernel" / UVC).glob("uvcvideo.ko*"))
    if len(candidates) != 1:
        return None
    sig = subprocess.check_output(
        ["modinfo", "-F", "srcversion", str(candidates[0])], text=True
    ).strip()
    if re.fullmatch("[0-9A-Fa-f]{23,24}", sig):
        return Path("/var/cache/openjensen-uvc") / sig
    return None


def copy_sources(source, dest):
    for path in source.iterdir():
        if path.is_file() and SOURCE_NAME.fullmatch(path.name):
            shutil.copyfile(path, dest / path.name)


def stage_sources(kernel, modules, output):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._+-]*", kernel):
        raise ValueError("Invalid kernel release")
    module_dir = modules / kernel
    source_dirs = [
        module_dir / "source" / UVC,
        module_dir / "build" / UVC,
        Path("/usr/src/openjensen-uvc") / kernel / UVC,
    ]
    source = next((path for path in source_dirs if (path / "uvc_video.c").is_file()), None)
    output.mkdir(parents=True, exist_ok=True)
    if source:
        copy_sources(source, output)
    else:
        cache = stock_cache(module_dir)
        if cache and (cache / "uvc_video.c").is_file():
            copy_sources(cache, output)
        else:
            if os.environ.get("OPENJENSEN_UVC_OFFLINE") == "1":
                raise ValueError("Offline build needs matching local UVC source for " + kernel)
            ubuntu_sources(kernel, output)
            if cache and os.geteuid() == 0:
                cache.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=cache.parent) as tmp:
                    staging = Path(tmp) / "source"
                    staging.mkdir()
                    copy_sources(output, staging)
                    if not cache.exists():
                        staging.rename(cache)


def patch_sources(output):
    path = output / "uvc_video.c"
    code = path.read_text()
    if code.count(INCLUDE) != 1 or code.count(RESERVE) != 1 or "mjpeg_bandwidth_cap" in code:
        raise ValueError("Unsupported UVC source layout; keeping the stock driver")
    path.write_text(code.replace(INCLUDE, INCLUDE + "\n" + CAP).replace(RESERVE, RESERVE + LIMIT))


def prepare(kernel, modules, output):
    # DKMS versions differ in how they propagate a failed pre-build hook.
    # Publish the Makefile only after the target source has been patched.
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError(f"Use a fresh DKMS build directory: {output}")
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        staging = Path(tmp) / "uvc"
        stage_sources(kernel, modules, staging)
        patch_sources(staging)
        staging.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel")
    parser.add_argument("--modules", type=Path, default=Path("/lib/modules"))
    parser.add_argument("--prepare", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.kernel, args.modules, args.prepare)
