"""Detached managed-service updater. Only confirmed canonical-main commits are accepted.

This module intentionally uses only the standard library so it survives switching
application releases. Its log/config/status stay private and outside the workspace.
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from urllib.request import urlopen

ACTIVE_PHASES = {"queued", "building", "backing_up", "validating", "restarting"}

CANONICAL_REMOTES = {
    "https://github.com/OpenJensen/openjensen.git",
    "https://github.com/OpenJensen/openjensen",
    "git@github.com:OpenJensen/openjensen.git",
    "ssh://git@github.com/OpenJensen/openjensen.git",
}


def load_config(path):
    value = json.loads(Path(path).read_text())
    if isinstance(value, dict) and value.get("kind") == "native":
        return load_native_config(path)
    required = {
        "root",
        "repository",
        "supervisorctl",
        "supervisor_config",
        "service",
        "port",
        "uv",
        "pnpm",
        "node_bin",
    }
    if not isinstance(value, dict) or set(value) != required or os.name != "posix":
        raise ValueError("Updates require a configured POSIX Supervisor deployment.")
    for key in required - {"service", "port"}:
        item = Path(value[key])
        if not item.is_absolute() or not item.exists():
            raise ValueError("Update paths must exist and be absolute.")
    root = Path(value["root"])
    if root.is_symlink() or not (root / "current").is_symlink() or (root / "data").is_symlink():
        raise ValueError(
            "The deployment requires current -> releases and a persistent data directory."
        )
    (root / "current").resolve().relative_to((root / "releases").resolve())
    if not (root / "data").is_dir() or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value["service"]):
        raise ValueError("Invalid service or workspace.")
    if type(value["port"]) is not int or not 1024 <= value["port"] <= 65535:
        raise ValueError("Invalid service port.")
    return value


def load_native_config(path):
    from vla_platform.native_launch import read_private

    value = read_private(path)
    required = {
        "kind",
        "root",
        "workspace",
        "repository",
        "source",
        "python",
        "origin_source",
        "origin_commit",
        "origin_signature",
        "port",
        "launcher_pid",
        "session",
        "uv",
        "pnpm",
        "node_bin",
    }
    if set(value) != required or os.name != "posix":
        raise ValueError("Invalid application launcher configuration.")
    for key in ("root", "workspace", "repository", "source", "python", "origin_source"):
        if not isinstance(value[key], str) or not Path(value[key]).is_absolute():
            raise ValueError("Application paths must be absolute.")
    root, workspace = Path(value["root"]), Path(value["workspace"])
    if (
        root.is_symlink()
        or workspace.is_symlink()
        or not workspace.is_dir()
        or root.stat().st_uid != os.getuid()
        or root.stat().st_mode & 0o077
        or not (root / "current").is_symlink()
        or type(value["launcher_pid"]) is not int
        or value["launcher_pid"] <= 1
        or type(value["port"]) is not int
        or not 1 <= value["port"] <= 65535
        or not isinstance(value["session"], str)
        or not re.fullmatch(r"[0-9a-f]{32}", value["session"])
    ):
        raise ValueError("Invalid application launcher or workspace.")
    if root == workspace or workspace in root.parents:
        raise ValueError("The update cache must be outside the workspace.")
    os.kill(value["launcher_pid"], 0)
    return value


def update_tools(config):
    for name in ("uv", "pnpm", "node_bin"):
        if not config.get(name) or not Path(config[name]).exists():
            raise ValueError(
                "The update needs "
                + ("Node.js" if name == "node_bin" else name)
                + " on this computer. Install the normal application prerequisites and retry."
            )


def write_state(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        os.chmod(temporary, 0o600)
        json.dump(value, handle)
        handle.write("\n")
    os.replace(temporary, path)


def run(args, *, cwd=None, env=None, timeout=1800, capture=False):
    print("Running " + str(args[0]) + " " + str(args[1:3]), flush=True)
    process = subprocess.Popen(
        args,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE if capture else None,
        text=capture,
        start_new_session=True,
    )
    try:
        stdout, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Every command owns a new process group. Do not leave build descendants running.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                continue
        process.wait(timeout=5)
        raise
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, args, output=stdout)
    return stdout.strip() if capture else None


def git(config, *args):
    return run(["git", "-C", config["repository"], *args], timeout=120, capture=True)


def verify_target(config, target, installed):
    if (
        not re.fullmatch(r"[0-9a-f]{40}", target)
        or (installed and not re.fullmatch(r"[0-9a-f]{40}", installed))
        or (not installed and config.get("kind") != "native")
    ):
        raise ValueError("Invalid commit.")
    if config.get("kind") == "native" and not Path(config["repository"]).exists():
        run(
            [
                "git",
                "clone",
                "--bare",
                "--filter=blob:none",
                "https://github.com/OpenJensen/openjensen.git",
                config["repository"],
            ],
            timeout=120,
        )
    if git(config, "remote", "get-url", "origin") not in CANONICAL_REMOTES:
        raise ValueError("The update source must be OpenJensen/openjensen on GitHub.")
    git(config, "fetch", "--no-tags", "origin", "+refs/heads/main:refs/remotes/origin/main")
    if git(config, "rev-parse", "refs/remotes/origin/main") != target:
        raise ValueError("Upstream main changed. Check and confirm the new version.")
    if installed:
        git(config, "merge-base", "--is-ancestor", installed, target)
    if installed == target:
        raise ValueError("This version is already installed.")


def pnpm_command(config):
    """Use Node for JavaScript launchers and the pnpm package's shebang-less stub."""
    executable = Path(config["pnpm"])
    node = Path(config["node_bin"]) / "node"
    if executable.name == "corepack":
        return [str(executable), "pnpm"]
    if executable.suffix in {".js", ".cjs", ".mjs"}:
        return [str(node), str(executable)]
    manifest = executable.parent / "package.json"
    entry = executable.parent / "bin/pnpm.mjs"
    if executable.name == "pnpm" and entry.is_file() and manifest.is_file():
        if json.loads(manifest.read_text()).get("name") == "pnpm":
            return [str(node), str(entry)]
    return [str(executable)]


def prepare_release(config, target, operation):
    root = Path(config["root"])
    release = root / "releases" / ("update-" + operation + "-" + target[:12])
    release.mkdir(mode=0o700)
    archive = root / "run" / ("update-" + operation + ".tar")
    try:
        git(config, "archive", "--format=tar", "--output=" + str(archive), target)
        with tarfile.open(archive) as bundle:
            bundle.extractall(release, filter="data")
    finally:
        archive.unlink(missing_ok=True)
    env = dict(
        os.environ,
        PATH=os.pathsep.join(
            [config["node_bin"], str(Path(config["pnpm"]).parent), os.environ.get("PATH", "")]
        ),
        UV_PROJECT_ENVIRONMENT=str(release / ".venv"),
    )
    # Worker environments and model weights are deliberately outside this update.
    # A new application environment keeps editable imports tied to this release.
    run(
        [config["uv"], "sync", "--frozen", "--all-packages", "--extra", "tui", "--no-dev"],
        cwd=release,
        env=env,
    )
    pnpm = pnpm_command(config)
    run([*pnpm, "install", "--frozen-lockfile", "--prefer-offline"], cwd=release, env=env)
    # Invoke the web package directly: no second pnpm executable is required on PATH.
    run([*pnpm, "--filter", "@openjensen/web", "build"], cwd=release, env=env)
    if not (release / "apps/web/out/dashboard/index.html").is_file():
        raise ValueError("The new web export is incomplete.")
    write_state(release / ".openjensen-build.json", {"commit": target, "channel": "main"})
    return release


def copy_workspace(source, destination):
    if sys.platform.startswith("linux"):
        run(["cp", "-a", "--reflink=auto", str(source), str(destination)])
    else:
        shutil.copytree(source, destination, symlinks=True)

    # Do not follow external dataset/runtime links. Their targets stay in place.
    def inventory(root):
        found = {}
        for directory, dirs, files in os.walk(root, followlinks=False):
            for name in dirs + files:
                path = Path(directory) / name
                if path.is_symlink():
                    found[str(path.relative_to(root))] = ("link", os.readlink(path))
                elif path.is_file():
                    found[str(path.relative_to(root))] = ("file", path.stat().st_size)
        return found

    if inventory(source) != inventory(destination):
        raise ValueError("Workspace backup verification failed.")


def migrate_candidate(release, workspace):
    # Exercise migrations on the candidate without starting workers or contacting hardware.
    code = """import asyncio,sys
from pathlib import Path
from vla_platform.storage import Storage
async def main():
    storage=Storage(Path(sys.argv[1]))
    try: await storage.initialize()
    finally: await storage.close()
asyncio.run(main())
"""
    env = dict(os.environ, PYTHONPATH=str(release / "packages/core/src"))
    run(
        [str(release / ".venv/bin/python"), "-c", code, str(workspace)],
        cwd=release,
        env=env,
        timeout=120,
    )


def service(config, action):
    if config.get("kind") == "native":
        from vla_platform.native_launch import command

        return command(config, action)
    args = [config["supervisorctl"], "-c", config["supervisor_config"]]
    try:
        run([*args, action, config["service"]], timeout=150)
    except subprocess.CalledProcessError:
        if action != "stop":
            raise
        result = subprocess.run(
            [*args, "status", config["service"]], capture_output=True, text=True, timeout=10
        )
        fields = result.stdout.split()
        if (
            len(fields) < 2
            or fields[0] != config["service"]
            or fields[1] not in {"STOPPED", "FATAL"}
        ):
            raise


def activate(root, release):
    link = root / (".current-update-" + str(os.getpid()))
    link.symlink_to(release)
    os.replace(link, root / "current")


def health(config, commit):
    for _ in range(60):
        try:
            with urlopen(
                f"http://127.0.0.1:{config['port']}/api/v1/updates", timeout=3
            ) as response:
                value = json.load(response)
            if value["installed"]["commit"] == commit:
                return
        except OSError, ValueError, KeyError:
            pass
        time.sleep(1)
    raise RuntimeError("The selected application did not become healthy.")


def project_ids(config):
    with urlopen(f"http://127.0.0.1:{config['port']}/api/v1/projects", timeout=10) as response:
        values = json.load(response)
    if not isinstance(values, list):
        raise ValueError("Could not verify workspace projects.")
    return {value["id"] for value in values}


def update(config, target, installed, operation):
    import fcntl

    root = Path(config["root"])
    path = root / "run/application-update.json"
    state = dict(
        id=operation,
        phase="building",
        target_commit=target,
        created=time.time(),
        pid=os.getpid(),
        message="Building the new version…",
    )

    def phase(name, message):
        state.update(phase=name, message=message)
        write_state(path, state)

    with (root / "run/application-update.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        phase("building", "Building the new version…")
        stopped = False
        switched = False
        moved = False
        original_moved = False
        previous = (root / "current").resolve()
        workspace = Path(config.get("workspace", root / "data"))
        private = root / "private/application-updates" / operation
        private.mkdir(mode=0o700, parents=True)
        try:
            update_tools(config)
            # Allow for a complete backup plus candidate and bounded application build output.
            data_bytes = sum(
                p.stat().st_size for p in workspace.rglob("*") if p.is_file() and not p.is_symlink()
            )
            if shutil.disk_usage(root).free < 10 * 1024**3 + data_bytes * 2 + 4 * 1024**3:
                raise ValueError("Insufficient disk space for a preserved backup and new release.")
            verify_target(config, target, installed)
            release = prepare_release(config, target, operation)
            if (root / "current").resolve() != previous:
                raise ValueError("The installed release changed during preparation.")
            with urlopen(
                f"http://127.0.0.1:{config['port']}/api/v1/updates/readiness", timeout=40
            ) as response:
                if json.load(response) != {"idle": True}:
                    raise ValueError("The server is no longer idle.")
            preserved_projects = project_ids(config)
            phase("backing_up", "Stopping the server and backing up the workspace…")
            service(config, "stop")
            stopped = True
            # Respect any other application writer. This lock lives in the stopped workspace.
            with (workspace / "owner.lock").open("a") as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                copy_workspace(workspace, private / "backup")
                copy_workspace(private / "backup", private / "candidate")
                phase("validating", "Validating the workspace with the new version…")
                migrate_candidate(release, private / "candidate")
                os.rename(workspace, private / "previous-data")
                original_moved = True
                os.rename(private / "candidate", workspace)
                moved = True
            phase("restarting", "Restarting the server…")
            activate(root, release)
            switched = True
            service(config, "start")
            stopped = False
            health(config, target)
            if project_ids(config) != preserved_projects:
                raise RuntimeError("Workspace project verification failed.")
            phase("complete", "Update installed. Reconnecting to the new version…")
        except Exception:
            import traceback

            traceback.print_exc()
            try:
                if switched:
                    service(config, "stop")
                    stopped = True
                if moved:
                    os.rename(workspace, private / "failed-data")
                if original_moved:
                    os.rename(private / "previous-data", workspace)
                if switched:
                    activate(root, previous)
                if stopped:
                    service(config, "start")
                    health(config, installed)
                phase(
                    "rolled_back" if switched or original_moved else "failed",
                    "Update failed; the previous version and workspace are retained. "
                    "Inspect the server update log.",
                )
            except Exception:
                traceback.print_exc()
                phase(
                    "recovery_required",
                    "Automatic recovery failed. Keep the server stopped "
                    "and restore the recorded backup.",
                )
            raise


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "target", "installed", "operation"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{32}", args.operation):
        raise ValueError("Invalid update operation.")
    update(load_config(args.config), args.target, args.installed, args.operation)


if __name__ == "__main__":
    main()
