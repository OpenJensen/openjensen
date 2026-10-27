"""Private launcher for ordinary installations; no operator update configuration is needed.

The launcher keeps the original terminal/service process alive while application
children are replaced. The detached updater can therefore restore the old child
even when a new version fails before its HTTP server starts.
"""

import argparse
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

CONTEXT = "JENSEN_NATIVE_LAUNCH"
ROOT = Path(__file__).resolve().parents[4]


def private_directory(path):
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("The application update directory must be private and owned by this user.")
    return path


def read_private(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor) as handle:
        info = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
            or info.st_size > 32768
        ):
            raise ValueError("Invalid private launcher file.")
        return json.load(handle)


def state_root(workspace):
    identity = hashlib.sha256(os.fsencode(workspace)).hexdigest()[:24]
    state = (
        Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")).expanduser().resolve()
    )
    root = state / "openjensen/application-updates" / identity
    # Even a workspace covering the home directory must not recursively back up its cache.
    if root == workspace or workspace in root.parents:
        root = workspace.parent / (".openjensen-updates-" + identity)
    return root


def executable(name, *alternatives):
    for value in (name, *alternatives):
        if found := shutil.which(value):
            return str(Path(found).absolute())
    return None


def configuration(path, workspace, source=ROOT):
    from vla_platform import update_runner

    config = update_runner.load_config(path)
    if config.get("kind") != "native":
        raise ValueError("Invalid application launcher.")
    if Path(config["workspace"]) != workspace or Path(config["source"]) != source:
        raise ValueError("The launcher belongs to a different application or workspace.")
    if config["launcher_pid"] != os.getppid():
        raise ValueError("The application launcher is no longer connected.")
    return config


def command(config, action):
    from vla_platform.update_runner import write_state

    root = Path(config["root"])
    identifier = uuid4().hex
    write_state(
        root / "run/launcher-command.json",
        dict(id=identifier, session=config["session"], action=action),
    )
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        os.kill(config["launcher_pid"], 0)
        try:
            value = read_private(root / "run/launcher-status.json")
            if value.get("command") == identifier:
                if value.get("error"):
                    raise RuntimeError(value["error"])
                return
        except FileNotFoundError:
            pass
        time.sleep(0.1)
    raise TimeoutError("The application launcher did not respond.")


class Launcher:
    def __init__(self, source, workspace, port, *, python=None, root=None):
        self.root = private_directory(root or state_root(workspace))
        for name in ("run", "releases", "private"):
            private_directory(self.root / name)
        self.path = self.root / "run/native-launch.json"
        self.source = source
        self.python = Path(python or sys.executable).absolute()
        self.workspace = workspace
        self.child = None
        self.stopping = False
        self.last_command = None
        self.env = dict(os.environ)
        self.config = dict(
            kind="native",
            root=str(self.root),
            workspace=str(workspace),
            repository=str(self.root / "repository"),
            source=str(source),
            python=str(self.python),
            origin_source=str(source),
            origin_commit=self.source_commit(source),
            origin_signature=self.source_signature(source),
            port=port,
            launcher_pid=os.getpid(),
            session=uuid4().hex,
            uv=executable("uv"),
            pnpm=executable("pnpm", "corepack"),
            node_bin=str(Path(node).parent) if (node := executable("node")) else None,
        )

    @staticmethod
    def source_commit(source):
        from vla_platform.application_updates import installed_commit

        return installed_commit(source)

    @staticmethod
    def source_signature(source):
        digest = hashlib.sha256()
        # Ignore environments/build output while noticing edits to archive installations.
        trees = [*sorted((source / "packages").glob("*/src")), source / "apps/web/src"]
        for tree in trees:
            for path in sorted(tree.rglob("*")):
                if path.is_file() and path.suffix in {".py", ".ts", ".tsx", ".css", ".json"}:
                    digest.update(os.fsencode(path.relative_to(source)))
                    digest.update(path.read_bytes())
        return digest.hexdigest()

    def selected_source(self):
        # Future starts from the same unchanged checkout use its installed update.
        # Developer edits or a different checkout retain their explicitly selected source.
        try:
            previous = read_private(self.path)
            selected = (self.root / "current").resolve(strict=True)
            selected.relative_to(self.root / "releases")
            if (
                previous["origin_source"] == str(self.source)
                and previous["origin_commit"] == self.config["origin_commit"]
                and previous["origin_signature"] == self.config["origin_signature"]
                and (selected / ".venv/bin/python").is_file()
                and (selected / ".openjensen-build.json").is_file()
            ):
                if (self.source / ".git").exists():
                    result = subprocess.run(
                        [
                            "git",
                            "-C",
                            str(self.source),
                            "status",
                            "--porcelain",
                            "--untracked-files=no",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        check=True,
                    )
                    if result.stdout.strip():
                        return self.source
                return selected
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            pass
        return self.source

    def publish(self, *, error=None):
        from vla_platform.update_runner import write_state

        write_state(
            self.root / "run/launcher-status.json",
            dict(
                session=self.config["session"],
                command=self.last_command,
                pid=self.child.pid if self.child and self.child.poll() is None else None,
                error=error,
            ),
        )

    def start(self):
        from vla_platform.update_runner import write_state

        source = (self.root / "current").resolve(strict=True)
        if source == self.source:
            python = self.python
        else:
            source.relative_to(self.root / "releases")
            python = source / ".venv/bin/python"
        self.config.update(source=str(source), python=str(python))
        write_state(self.path, self.config)
        web = self.env.get("JENSEN_WEB_DIR")
        if not web or Path(web).resolve() == self.source / "apps/web/out":
            web = str(source / "apps/web/out")
        env = dict(
            self.env,
            JENSEN_DATA_DIR=str(self.workspace),
            JENSEN_WEB_DIR=web,
            JENSEN_NATIVE_LAUNCH=str(self.path),
            PYTHONPATH=str(source / "packages/core/src"),
        )
        self.child = subprocess.Popen(
            [
                str(python),
                "-m",
                "vla_platform.native_launch",
                "--child",
                "--port",
                str(self.config["port"]),
            ],
            cwd=source,
            env=env,
            start_new_session=True,
        )

    def stop(self):
        if self.child is None or self.child.poll() is not None:
            return
        self.child.terminate()
        try:
            self.child.wait(timeout=45)
        except subprocess.TimeoutExpired:
            self.child.kill()
            self.child.wait(timeout=5)

    def updating(self):
        from vla_platform.update_runner import ACTIVE_PHASES

        try:
            return read_private(self.root / "run/application-update.json")["phase"] in ACTIVE_PHASES
        except (OSError, ValueError, KeyError):
            return False

    def handle_command(self):
        try:
            value = read_private(self.root / "run/launcher-command.json")
            if value["session"] != self.config["session"] or value["id"] == self.last_command:
                return
            if value["action"] not in {"start", "stop"}:
                return
            self.last_command = value["id"]
            if value["action"] == "stop":
                self.stop()
            elif self.child is None or self.child.poll() is not None:
                self.start()
            self.publish()
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            self.publish(error="The application could not restart: " + str(exc))

    def run(self):
        import fcntl

        from vla_platform.update_runner import activate

        with (self.root / "run/launcher.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            activate(self.root, self.selected_source())
            previous_handlers = {}

            def finish(*_):
                self.stopping = True

            for sig in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[sig] = signal.signal(sig, finish)
            try:
                self.start()
                self.publish()
                while not self.stopping:
                    self.handle_command()
                    if self.child.poll() is not None and not self.updating():
                        return self.child.returncode
                    time.sleep(0.1)
                return 0
            finally:
                self.stop()
                self.publish()
                for sig, handler in previous_handlers.items():
                    signal.signal(sig, handler)


def serve(port):
    from vla_platform.environment import getenv
    from vla_platform.settings import Settings

    settings = Settings.from_env()
    if os.name != "posix":
        # The UI still offers the update action and explains platform limitations.
        child(port)
        return
    try:
        launcher = Launcher(ROOT, settings.data_dir, port)
    except (OSError, ValueError):
        os.environ["JENSEN_UPDATE_UNAVAILABLE"] = (
            "The application can run, but updates cannot write their private cache. "
            "Allow this user to write the application state directory, then restart and retry."
        )
        child(port)
        return
    if (configured_web := getenv("JENSEN_WEB_DIR")) is not None:
        launcher.env["JENSEN_WEB_DIR"] = str(Path(configured_web).expanduser().resolve())
    # Relative configuration must keep pointing to the same files after a release switch.
    for field, name in (
        ("local_root", "JENSEN_LOCAL_DATA_ROOT"),
        ("runtime_config", "JENSEN_RUNTIME_CONFIG"),
        ("cloud_runs_dir", "JENSEN_CLOUD_RUNS_DIR"),
        ("simulation_config", "JENSEN_SIMULATION_CONFIG"),
        ("recording_config", "JENSEN_RECORDING_CONFIG"),
        ("teaching_session_config", "JENSEN_TEACHING_SESSION_CONFIG"),
    ):
        if value := getattr(settings, field):
            launcher.env[name] = str(value.absolute())
    raise SystemExit(launcher.run())


def child(port):
    import uvicorn

    uvicorn.run("vla_platform.api:create_app", factory=True, host="127.0.0.1", port=port, workers=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true", required=True)
    parser.add_argument("--port", type=int, required=True)
    child(parser.parse_args().port)
