"""Explicit local CPU setup; importing or planning never installs or activates anything."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

VERSION = "0.1.0"
UV_VERSION = "0.12.19"
UPSTREAM = "e595b7902714ba51f91e47523f66f89c5181b649"
UPSTREAM_SHA = "a750c65130a5ebf2cd72b98189f1ab5b5918c00b1713111ee93188b8065e8c77"
UPSTREAM_URL = (
    f"https://github.com/huggingface/lerobot/archive/{UPSTREAM}.tar.gz#sha256={UPSTREAM_SHA}"
)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def digest(path: Path) -> str:
    """Hash a small pinned setup input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def absolute(value: str | Path) -> Path:
    """Require lexical absolute paths; do not silently resolve links or traversal."""
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Use an absolute path without parent traversal")
    return path


def no_links(path: Path) -> None:
    """Reject links in trusted operator-owned destination paths."""
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"Destination/source directory must not contain symlinks: {part}")


def platform_key(system: str | None = None, machine: str | None = None) -> str:
    """Limit installation to the two CPU platforms with reviewed wheel choices."""
    pair = (system or platform.system(), machine or platform.machine())
    supported = {("Darwin", "arm64"): "macos-arm64", ("Linux", "x86_64"): "linux-x86_64"}
    if pair not in supported:
        raise ValueError(
            "Supported setup: Apple Silicon macOS or Linux x86_64; Windows is deferred"
        )
    return supported[pair]


def write_new(path: Path, value: Any) -> None:
    """Publish owner-only JSON without replacing an existing file."""
    path = absolute(path)
    no_links(path)
    if not path.parent.is_dir():
        raise ValueError("Create the output parent directory explicitly first")
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    fd, name = tempfile.mkstemp(prefix=".local-cpu-", dir=path.parent)
    temporary = Path(name)
    owned = os.fstat(fd)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            current = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            # Covers an interruption immediately after link, before Python could record it.
            # Only roll back this inode, not an observed concurrent replacement.
            if (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
                path.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    """Read only a bounded regular operator file, without accepting duplicate keys."""
    no_links(path)
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("Expected a regular JSON file of at most 1 MiB")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    with path.open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("JSON file grew beyond its 1 MiB limit")

    def reject_constant(value: str) -> None:
        raise ValueError("Nonfinite JSON is not accepted")

    def finite_float(raw_value: str) -> float:
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError("Nonfinite JSON is not accepted")
        return value

    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject_constant,
        parse_float=finite_float,
    )
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def cache_path(root: Path, cache: Path | None) -> Path:
    """Admit an explicit existing uv cache without treating a venv as a cache."""
    if cache is None:
        return root / "cache"
    cache = absolute(cache)
    no_links(cache)
    if root == cache or root.is_relative_to(cache) or cache.is_relative_to(root):
        raise ValueError("Shared cache and new installation root must be disjoint")
    if not cache.is_dir():
        raise ValueError("Shared cache must be an existing uv cache directory")
    if any((part / "pyvenv.cfg").exists() for part in (cache, *cache.parents)):
        raise ValueError("An installed Python environment cannot be used as a shared cache")
    tag = cache / "CACHEDIR.TAG"
    no_links(tag)
    if not tag.is_file():
        raise ValueError("Shared cache must contain the existing uv CACHEDIR.TAG")
    with tag.open("rb") as handle:
        content = handle.read(1025)
    if len(content) > 1024 or not content.startswith(
        b"Signature: 8a477f597d28d172789f06886806bc55"
    ):
        raise ValueError("Shared cache has an invalid cache directory marker")
    return cache


def environment(root: Path, *, offline: bool, cache: Path | None = None) -> dict[str, str]:
    """Do not forward credentials, arbitrary Python paths or user package indexes."""
    env = {key: os.environ[key] for key in ("PATH", "TMPDIR", "LANG") if key in os.environ}
    env.update(
        UV_CACHE_DIR=str(cache_path(root, cache)),
        UV_PYTHON_DOWNLOADS="never",
        UV_PROJECT_ENVIRONMENT=str(root / "act-model"),
        UV_NO_CONFIG="1",
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        HF_HOME=str(root / "cache/huggingface"),
    )
    if cache is not None:
        # Copies remain usable independently of a later cache cleanup.
        env["UV_LINK_MODE"] = "copy"
    if offline:
        env.update(UV_OFFLINE="1", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    return env


def run(command: list[str], *, env: dict[str, str], log: Path, timeout: float = 900) -> str:
    """Bound one owned installer group; repeated signals cannot skip cleanup."""
    interrupted: list[int] = []
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def record_signal(number: int, frame: Any) -> None:
        interrupted.append(number)

    for sig in previous:
        signal.signal(sig, record_signal)
    process = None
    try:
        with log.open("ab") as stream:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=stream,
                env=env,
                start_new_session=True,
            )
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                if interrupted:
                    raise KeyboardInterrupt
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(command, timeout)
                time.sleep(0.05)
            if interrupted:
                raise KeyboardInterrupt
            if process.returncode != 0:
                raise ValueError(f"Setup step failed ({process.returncode}); inspect {log}")
    finally:
        try:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                # The leader may exit before an inherited child. Never leave that group alive.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return ""


def setup_plan(root: Path, python: Path, uv: Path, *, cache: Path | None = None) -> dict[str, Any]:
    """Return exact commands and manifests; never execute tools or create directories."""
    root, python, uv = absolute(root), absolute(python), absolute(uv)
    no_links(root)
    selected_cache = cache_path(root, cache)
    key = platform_key()
    act = REPO / "workers/act_optimizer"
    reader = HERE / f"reader-{key}.lock"
    required = [act / "pyproject.toml", act / "uv.lock", reader, HERE / "probe.py"]
    if not all(path.is_file() for path in required):
        raise ValueError("Setup requires the complete checked-out worker source and lockfiles")
    base = [str(uv), "--no-config", "--no-python-downloads"]
    commands = [
        [
            *base,
            "sync",
            "--locked",
            "--project",
            str(root / "act-project"),
            "--no-dev",
            "--no-install-project",
            "--extra",
            "cpu",
            "--python",
            str(python),
        ],
        [*base, "venv", "--python", str(python), str(root / "dataset-reader")],
        [
            *base,
            "pip",
            "install",
            "--python",
            str(root / "dataset-reader/bin/python"),
            "--require-hashes",
            "--no-deps",
            "--only-binary",
            ":all:",
            "-r",
            str(reader),
        ],
        [
            *base,
            "pip",
            "install",
            "--python",
            str(root / "dataset-reader/bin/python"),
            "--no-deps",
            "--no-build-isolation",
            "lerobot @ " + UPSTREAM_URL,
        ],
        [*base, "pip", "check", "--python", str(root / "act-model/bin/python")],
        [*base, "pip", "check", "--python", str(root / "dataset-reader/bin/python")],
    ]
    return {
        "schema_version": 1,
        "platform": key,
        "installation_root": str(root),
        "repository": str(REPO),
        "uv_version": UV_VERSION,
        "cache": {
            "path": str(selected_cache),
            "shared": cache is not None,
            "scope": "uv_distribution_artifacts_only",
            "link_mode": "copy" if cache is not None else "uv-default",
        },
        "python": str(python),
        "inputs": {str(p.relative_to(REPO)): digest(p) for p in required},
        "commands": commands,
        "network_required": True,
        "activation": "None. Explicit config output and application restart are separate.",
    }


def verify(
    root: Path, *, model_python: Path | None = None, reader_python: Path | None = None
) -> dict[str, Any]:
    """Check native imports/versions/source hashes without a model, data or network."""
    key = platform_key()
    root = absolute(root)
    outputs = {}
    for role, python in (
        ("model", model_python or root / "act-model/bin/python"),
        ("reader", reader_python or root / "dataset-reader/bin/python"),
    ):
        python = absolute(python)
        if not python.is_file() or not os.access(python, os.X_OK):
            raise ValueError(f"Missing executable for {role}; run the explicit install first")
        # Bounded JSON stdout, no keys are forwarded and probe forbids socket activity.
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            try:
                result = subprocess.run(
                    [str(python), "-I", str(HERE / "probe.py"), role, str(REPO)],
                    env=environment(root, offline=True),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=60,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                stdout.seek(0)
                raw = stdout.read(64 * 1024 + 1)
                complete_json = False
                if len(raw) <= 64 * 1024:
                    try:
                        complete_json = isinstance(json.loads(raw.decode("utf-8")), dict)
                    except (ValueError, UnicodeError):
                        pass
                stderr.seek(0, os.SEEK_END)
                stderr.seek(max(0, stderr.tell() - 2000))
                tail = stderr.read(2000).decode("utf-8", errors="replace")
                progress = (
                    "a complete JSON object was captured but the process did not exit"
                    if complete_json
                    else "no complete JSON object was captured"
                )
                raise ValueError(
                    f"{role} runtime verification exceeded its 60-second deadline; "
                    f"{progress} (not accepted). stderr tail: {tail or '<empty>'}"
                ) from error
            stdout.seek(0)
            raw = stdout.read(64 * 1024 + 1)
            if result.returncode or len(raw) > 64 * 1024:
                stderr.seek(0, os.SEEK_END)
                stderr.seek(max(0, stderr.tell() - 2000))
                tail = stderr.read(2000).decode("utf-8", errors="replace")
                raise ValueError(
                    f"{role} runtime verification failed or response too large: {tail}"
                )
            outputs[role] = json.loads(raw)

    return {
        "schema_version": 1,
        "platform": key,
        "verified": outputs,
        "scope": "CPU runtime imports and native source pins only; no model/task acceptance",
    }


def check_tools(python: Path, uv: Path, env: dict[str, str]) -> None:
    """Check only the explicitly selected existing tool executables."""
    uv_version = subprocess.check_output([str(uv), "--version"], env=env, text=True, timeout=5)
    if not uv_version.startswith(f"uv {UV_VERSION} "):
        raise ValueError(f"Use existing uv {UV_VERSION}; this command never installs tools")
    version = subprocess.check_output(
        [str(python), "-I", "-c", "import sys;print('.'.join(map(str,sys.version_info[:2])))"],
        env=env,
        text=True,
        timeout=5,
    ).strip()
    if version != "3.12":
        raise ValueError("Use existing Python3.12; managed Python downloads are disabled")


def install(root: Path, python: Path, uv: Path, *, cache: Path | None = None) -> dict[str, Any]:
    """Install only into a new explicitly selected persistent directory."""
    plan = setup_plan(root, python, uv, cache=cache)
    if root.exists() or not root.parent.is_dir():
        raise ValueError("Installation root must be new; explicitly create its parent first")
    if not python.is_file() or not uv.is_file():
        raise ValueError("Provide existing Python3.12 and uv0.12.19 executables")
    env = environment(root, offline=False, cache=cache)
    check_tools(python, uv, env)
    root.mkdir(mode=0o700)
    write_new(root / "installation-plan.json", plan)
    project = root / "act-project"
    project.mkdir()
    for name in ("pyproject.toml", "uv.lock"):
        (project / name).write_bytes((REPO / "workers/act_optimizer" / name).read_bytes())
    # Preserve partial environments/logs on failure for diagnosis; never upgrade or overwrite them.
    for command in plan["commands"]:
        print("Installing explicit isolated CPU runtime step…", file=sys.stderr)
        run(command, env=env, log=root / "install.log")
    checked = verify(root)
    if any(digest(REPO / name) != value for name, value in plan["inputs"].items()):
        raise ValueError("Setup source changed during installation")
    receipt = {
        **checked,
        "installation_root": str(root),
        "inputs": plan["inputs"],
        "cache": plan["cache"],
    }
    write_new(root / "installation.json", receipt)
    return receipt


def completion_plan(
    root: Path, python: Path, uv: Path, *, cache: Path | None = None
) -> dict[str, Any]:
    """Validate a preserved partial attempt without executing its saved commands."""
    root, python, uv = absolute(root), absolute(python), absolute(uv)
    expected = setup_plan(root, python, uv, cache=cache)
    if os.path.lexists(root / "installation.json"):
        raise FileExistsError("Installation already has a receipt; use verify instead")
    saved = read_json(root / "installation-plan.json")
    # Canonical JSON comparison distinguishes true/1 and other JSON type changes.
    if json.dumps(saved, sort_keys=True, allow_nan=False) != json.dumps(
        expected, sort_keys=True, allow_nan=False
    ):
        raise ValueError("Preserved installation plan differs from the selected tools/source/root")
    for name in ("pyproject.toml", "uv.lock"):
        copied = root / "act-project" / name
        no_links(copied)
        if not copied.is_file() or copied.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("Missing or oversized copied ACT project input")
        if digest(copied) != expected["inputs"][f"workers/act_optimizer/{name}"]:
            raise ValueError("Copied ACT project input differs from the preserved plan")
    base = [str(uv), "--no-config", "--no-python-downloads", "pip", "check", "--python"]
    return {
        "schema_version": 1,
        "mode": "verify_existing_partial_installation",
        "installation_root": str(root),
        "installation_plan_sha256": digest(root / "installation-plan.json"),
        "inputs": expected["inputs"],
        "cache": expected["cache"],
        "commands": [
            base + [str(root / folder / "bin/python")] for folder in ("act-model", "dataset-reader")
        ],
        "probes": [
            [str(root / folder / "bin/python"), "-I", str(HERE / "probe.py"), role, str(REPO)]
            for role, folder in (("model", "act-model"), ("reader", "dataset-reader"))
        ],
        "network_required": False,
        "publication": "Only complete --execute publishes a new installation.json after all checks",
        "activation": "None; no package installation, repair or configuration activation",
    }


def complete(root: Path, python: Path, uv: Path, *, cache: Path | None = None) -> dict[str, Any]:
    """Explicitly verify a coherent partial attempt; never install or repair it."""
    root, python, uv = absolute(root), absolute(python), absolute(uv)
    planned = completion_plan(root, python, uv, cache=cache)
    env = environment(root, offline=True, cache=cache)
    check_tools(python, uv, env)
    descriptor, name = tempfile.mkstemp(prefix="completion-", suffix=".log", dir=root)
    os.close(descriptor)
    log = Path(name)
    for command in planned["commands"]:
        run(command, env=env, log=log)
    checked = verify(root)
    if completion_plan(root, python, uv, cache=cache) != planned:
        raise ValueError("Preserved installation inputs changed during completion")
    receipt = {
        **checked,
        "installation_root": str(root),
        "inputs": planned["inputs"],
        "cache": planned["cache"],
        "completion": {
            "mode": "verified_existing_partial_installation",
            "installation_plan_sha256": planned["installation_plan_sha256"],
            "log": log.name,
            "packages_installed_or_repaired": False,
        },
    }
    write_new(root / "installation.json", receipt)
    return receipt


def config(root: Path, output: Path, base: Path | None) -> dict[str, Any]:
    """Write a new registry preserving existing entries; never activate it."""
    root, output = absolute(root), absolute(output)
    if root == output or output.is_relative_to(root):
        raise ValueError("Keep application configuration outside the installation directory")
    receipt = read_json(root / "installation.json")
    if receipt.get("installation_root") != str(root):
        raise ValueError("Installation identity mismatch")
    verify(root)
    value = read_json(absolute(base)) if base else {"runtimes": [], "sources": []}
    if set(value) - {"runtimes", "sources"}:
        raise ValueError("Base must be a runtime registry with runtimes/sources fields")
    for collection in ("runtimes", "sources"):
        entries = value.setdefault(collection, [])
        if not isinstance(entries, list) or any(
            not isinstance(x, dict) or not isinstance(x.get("id"), str) for x in entries
        ):
            raise ValueError("Invalid base registry entries")
        if len({x["id"] for x in entries}) != len(entries):
            raise ValueError("Duplicate base registry IDs")
    ids = {x["id"] for x in value["runtimes"]}
    for name, folder in (
        ("replay", "isaac_sim"),
        ("distillation", "policy_distillation"),
        ("quantization", "firebird_quant"),
    ):
        identifier = f"local-act-{name}-cpu"
        if identifier in ids:
            raise ValueError(f"Runtime already exists: {identifier}; no entry will be overwritten")
        value["runtimes"].append(
            {
                "id": identifier,
                "label": f"Local ACT {name} · CPU",
                "execution": "native",
                "provider": "local",
                "device": "cpu",
                "python": str(root / "act-model/bin/python"),
                f"native_{name}_only": True,
                f"native_{name}_python": str(root / "act-model/bin/python"),
                f"native_{name}_root": str(REPO / "workers" / folder),
                **(
                    {f"native_{name}_dataset_python": str(root / "dataset-reader/bin/python")}
                    if name != "quantization"
                    else {}
                ),
            }
        )
    write_new(output, value)
    return {
        "output": str(output),
        "runtime_count": len(value["runtimes"]),
        "activated": False,
        "next_step": (
            "Review the new file; explicitly set FIREBIRD_RUNTIME_CONFIG "
            "and restart the app when ready."
        ),
    }


def parser() -> argparse.ArgumentParser:
    """Describe safe planning, explicit installation and separate configuration."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--version", action="version", version=VERSION)
    sub = cli.add_subparsers(dest="action", required=True)
    for action in ("plan", "install", "complete"):
        cmd = sub.add_parser(
            action,
            help="Show commands only"
            if action == "plan"
            else "Opt-in network installation into a NEW directory"
            if action == "install"
            else "Verify a preserved partial installation; no package repair or activation",
        )
        cmd.add_argument("--root", type=Path, required=True)
        cmd.add_argument("--python", type=Path, required=True)
        cmd.add_argument("--uv", type=Path, required=True)
        cmd.add_argument(
            "--cache",
            type=Path,
            help="Explicit existing uv cache to reuse; separate new environments still required",
        )
        if action in {"install", "complete"}:
            cmd.add_argument(
                "--execute",
                action="store_true",
                help="Explicitly authorize downloads and isolated environment creation"
                if action == "install"
                else "Run offline checks and publish a receipt; never install packages",
            )
    cmd = sub.add_parser("verify", help="Offline import/source verification; no model execution")
    cmd.add_argument("--root", type=Path, required=True)
    cmd = sub.add_parser("config", help="Write a NEW runtime registry; never activate it")
    cmd.add_argument("--root", type=Path, required=True)
    cmd.add_argument("--output", type=Path, required=True)
    cmd.add_argument("--base", type=Path, help="Existing registry to preserve unchanged")
    return cli


def main(argv: list[str] | None = None) -> int:
    """Run only the operator-selected action; JSON output is free of credentials."""
    args = parser().parse_args(argv)
    try:
        if args.action == "plan" or (args.action == "install" and not args.execute):
            result = setup_plan(args.root, args.python, args.uv, cache=args.cache)
        elif args.action == "install":
            result = install(
                absolute(args.root), absolute(args.python), absolute(args.uv), cache=args.cache
            )
        elif args.action == "complete":
            action = complete if args.execute else completion_plan
            result = action(args.root, args.python, args.uv, cache=args.cache)
        elif args.action == "verify":
            result = verify(args.root)
        else:
            result = config(args.root, args.output, args.base)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Local CPU setup: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "Local CPU setup interrupted; partial installation retained, not activated.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
