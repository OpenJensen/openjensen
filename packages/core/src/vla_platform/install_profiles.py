"""Independent, source-based installation plans for incompatible robotics runtimes."""

import json
import os
import platform
import shutil
import subprocess
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

GIB = 1024**3


@dataclass(frozen=True)
class InstallProfile:
    id: str
    name: str
    description: str
    python: str
    platforms: tuple[str, ...]
    directory: str
    distributions: tuple[str, ...]
    external: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    estimated_install_gib: int = 1
    manual: bool = False


PROFILES = (
    InstallProfile(
        "core",
        "Core",
        "Web workspace, API and command line; no ML or hardware libraries.",
        "3.14.7",
        ("Darwin", "Linux", "Windows"),
        ".",
        ("vla-platform", "fastapi", "pydantic", "uvicorn", "typer"),
        ("uv", "node", "pnpm"),
        (
            "Web source build requires Node >=24.21,<25 and pnpm 12.6.0.",
            "The existing default uv sync --frozen remains supported.",
        ),
        2,
    ),
    InstallProfile(
        "robot",
        "Robot Lab",
        "SO-101 USB setup, calibration, camera and local recording companion.",
        "3.14.7",
        ("Darwin", "Linux"),
        "workers/robot_lab",
        ("pyserial", "feetech-servo-sdk", "opencv-python-headless", "numpy", "Pillow", "pyarrow"),
        ("uv",),
        (
            "Install on the computer holding the arm and camera USB cables.",
            "Includes hardware and LeRobot v3 dataset tools; no Torch, models or training stack.",
            "macOS camera permission, physical motor isolation and calibration remain required.",
            "See deploy/macos/README.md for the managed Mac service and private pairing.",
        ),
    ),
    InstallProfile(
        "datasets",
        "Dataset tools",
        "Isolated Parquet, HDF5, image and dataset conversion readers.",
        "3.14.7",
        ("Darwin", "Linux", "Windows"),
        "workers/_cpu_readers",
        ("pyarrow", "numpy", "h5py", "Pillow"),
        ("uv", "ffmpeg", "ffprobe"),
        ("Install on the API host; configure JENSEN_CPU_READER_PYTHON for a reused environment.",),
    ),
    InstallProfile(
        "training",
        "Training",
        "Isolated LeRobot 0.4.4 / SmolVLA worker and CUDA training dependencies.",
        "3.11",
        ("Linux",),
        "workers/smolvla_qlora",
        (
            "jensen-vla",
            "lerobot",
            "torch",
            "torchvision",
            "torchcodec",
            "transformers",
            "peft",
            "bitsandbytes",
            "accelerate",
            "huggingface-hub",
            "safetensors",
        ),
        ("uv", "nvidia-smi"),
        (
            "Use the documented CUDA environment; package checks do not prove GPU execution.",
            "Other native workers use separate Python 3.12 environments and their own guides.",
            "Installation excludes model weights and training jobs; scheduled jobs use SkyPilot.",
        ),
        20,
    ),
    InstallProfile(
        "decision",
        "Decision scorer",
        "CPU dependencies for the optional Muose scorer.",
        "3.12.14",
        ("Darwin", "Linux"),
        "workers/decision",
        ("jensen-decision-worker", "torch", "safetensors", "tokenizers"),
        ("uv",),
        (
            "Model files and explicit CC-BY-NC-SA-4.0 acceptance are configured separately.",
            "No model weights are downloaded by dependency installation.",
        ),
        4,
    ),
    InstallProfile(
        "simulation",
        "Simulation",
        "External NVIDIA Isaac Sim 6.1 runtime and its isolated worker.",
        "Isaac Sim 6.1 bundled Python",
        ("Linux",),
        "workers/isaac_sim",
        (),
        ("docker", "nvidia-smi"),
        (
            "Install NVIDIA drivers/container runtime and obtain Isaac Sim under NVIDIA's terms.",
            "Follow workers/isaac_sim/README.md to install worker dependencies inside Isaac.",
            "This manual setup plan does not install or validate the simulator through pip.",
            "No image pulls, container launches or paid resources are started by this installer.",
        ),
        0,
        True,
    ),
)


def profile(ident):
    for value in PROFILES:
        if value.id == ident:
            return value
    raise ValueError(
        "Unknown profile. Choose core, robot, datasets, training, decision or simulation."
    )


def public_profiles():
    return [asdict(value) for value in PROFILES]


def source_root(source=None):
    candidates = (
        [Path(source)] if source is not None else [Path.cwd(), Path(__file__).resolve().parents[4]]
    )
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "pyproject.toml").is_file() and (candidate / "workers").is_dir():
            return candidate
    raise ValueError("Run from an OpenJensen source checkout or supply --source /path/to/checkout.")


def environment_path(value, root, environment=None):
    return (
        Path(environment).expanduser() if environment else root / value.directory / ".venv"
    ).resolve()


def interpreter(environment):
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def install_plan(value, root, environment=None):
    if value.manual:
        return []
    target = environment_path(value, root, environment)
    if value.id == "core":
        if environment:
            raise ValueError(
                "Core uses the checkout's .venv; use separate checkouts for separate Core installs."
            )
        return [
            {"argv": ["uv", "sync", "--frozen"], "cwd": str(root)},
            {"argv": ["pnpm", "install", "--frozen-lockfile"], "cwd": str(root)},
            {"argv": ["pnpm", "build:web"], "cwd": str(root)},
        ]
    commands = []
    if not target.exists():
        commands.append(
            {"argv": ["uv", "venv", "--python", value.python, str(target)], "cwd": str(root)}
        )
    command = ["uv", "pip", "install", "--python", str(interpreter(target))]
    if value.id in {"robot", "datasets"}:
        requirements = "requirements-hardware.txt" if value.id == "robot" else "requirements.txt"
        command += ["--no-deps", "-r", str(root / value.directory / requirements)]
    elif value.id == "decision":
        if platform.system() == "Linux":
            commands.append(
                {
                    "argv": [
                        *command,
                        "--index-url",
                        "https://download.pytorch.org/whl/cpu",
                        "torch==2.11.0",
                    ],
                    "cwd": str(root),
                }
            )
        command += [str(root / value.directory)]
    elif value.id == "training":
        command += ["-e", str(root / value.directory) + "[smolvla]"]
    commands.append({"argv": command, "cwd": str(root)})
    return commands


def expected_versions(value, root):
    if value.id in {"training", "decision"}:
        project = tomllib.loads((root / value.directory / "pyproject.toml").read_text())["project"]
        result = {project["name"].lower(): project["version"]}
        requirements = (
            project["dependencies"]
            if value.id == "decision"
            else project["optional-dependencies"]["smolvla"]
        )
        for requirement in requirements:
            requirement = requirement.split(";", 1)[0].strip()
            if "==" in requirement:
                package, version = requirement.split("==", 1)
                result[package.split("[", 1)[0].lower()] = version
        return result
    if value.id not in {"robot", "datasets"}:
        return {}
    name = "requirements-hardware.txt" if value.id == "robot" else "requirements.txt"
    result = {}
    for line in (root / value.directory / name).read_text().splitlines():
        if line.strip() and not line.startswith("#") and "==" in line:
            package, version = line.split("==", 1)
            result[package.lower()] = version.strip()
    return result


def check_profile(value, root, environment=None, *, runner=subprocess.run, which=shutil.which):
    target = environment_path(value, root, environment)
    external = {name: which(name) is not None for name in value.external}
    supported = platform.system() in value.platforms
    result = {
        "id": value.id,
        "name": value.name,
        "supported_platform": supported,
        "environment": str(target),
        "python_required": value.python,
        "external_tools": external,
        "manual": value.manual,
        "ready": False,
        "notes": list(value.notes),
        "packages": {},
    }
    if value.manual:
        result["message"] = (
            "External Isaac setup and runtime validation are required; no simulator was installed."
        )
        return result
    python = interpreter(target)
    if not python.is_file():
        result["message"] = "The selected environment is not installed."
        return result
    code = (
        "import importlib.metadata as m,json,sys; names=json.loads(sys.argv[1]); "
        "installed={d.metadata['Name'].lower():d.version for d in m.distributions()}; "
        "print(json.dumps({'python':list(sys.version_info[:3]),"
        "'packages':{n:installed.get(n.lower()) for n in names}}))"
    )
    try:
        response = runner(
            [str(python), "-I", "-c", code, json.dumps(value.distributions)],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        details = json.loads(response.stdout)
        if response.returncode or not isinstance(details, dict):
            raise ValueError
        result.update(details)
        expected = expected_versions(value, root)
        packages = details["packages"]
        versions_ok = all(
            packages.get(name) is not None
            and (
                name.lower() not in expected
                or (
                    packages[name].removesuffix("+cpu")
                    if value.id == "decision" and name == "torch"
                    else packages[name]
                )
                == expected[name.lower()]
            )
            for name in value.distributions
        )
        wanted = tuple(map(int, value.python.split(".")[:2]))
        python_ok = tuple(details["python"][:2]) == wanted
        result["python_matches"] = python_ok
        result["package_versions_match"] = versions_ok
        result["ready"] = supported and python_ok and versions_ok and all(external.values())
        result["message"] = (
            "Installed package metadata and external tools are available."
            if result["ready"]
            else "Some dependencies, Python version or external tools need setup."
        )
    except OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError, AttributeError:
        result["message"] = "The selected interpreter could not report its installed dependencies."
    if value.id == "core":
        versions_ok = False
        try:
            node = runner(
                ["node", "--version"], capture_output=True, text=True, timeout=5, check=False
            )
            pnpm = runner(
                ["pnpm", "--version"], capture_output=True, text=True, timeout=5, check=False
            )
            version = tuple(map(int, node.stdout.strip().lstrip("v").split(".")[:3]))
            versions_ok = (
                node.returncode == pnpm.returncode == 0
                and (24, 21, 0) <= version < (25, 0, 0)
                and pnpm.stdout.strip() == "12.6.0"
            )
            result["web_tool_versions"] = {"node": node.stdout.strip(), "pnpm": pnpm.stdout.strip()}
        except OSError, ValueError, subprocess.TimeoutExpired:
            result["web_tool_versions"] = {"message": "Node/pnpm versions need setup."}
        result["web_tool_versions_match"] = versions_ok
        result["web_export_available"] = (root / "apps/web/out/index.html").is_file()
        result["ready"] = result["ready"] and versions_ok and result["web_export_available"]
        if not result["ready"]:
            result["message"] = "Core dependencies, web build tools or web export still need setup."
    return result


def execute_install(value, root, environment=None, *, runner=subprocess.run):
    if platform.system() not in value.platforms:
        raise ValueError(
            f"{value.name} supports {', '.join(value.platforms)}; use a compatible host."
        )
    if value.manual:
        raise ValueError(
            "Simulation requires the external Isaac runtime. Review --dry-run and the worker guide."
        )
    if any(shutil.which(command) is None for command in value.external):
        raise ValueError("Install the profile's listed external tools before running this plan.")
    needed = (10 + value.estimated_install_gib) * GIB
    target = environment_path(value, root, environment)
    destination = target.parent
    while not destination.exists():
        destination = destination.parent
    if any(shutil.disk_usage(path).free < needed for path in {root, destination}):
        raise ValueError(
            f"Keep 10 GiB free after installation; start with at least {needed // GIB} GiB free."
        )
    if value.id != "core" and target.exists():
        state = check_profile(value, root, environment)
        if state["ready"]:
            return {"profile": value.id, "already_installed": True, "environment": str(target)}
        raise ValueError(
            "An existing environment was preserved. Use --check or --environment /new/path."
        )
    for command in install_plan(value, root, environment):
        response = runner(command["argv"], cwd=command["cwd"], check=False)
        if response.returncode:
            raise ValueError(
                "Installation stopped after a command failed. The environment was retained."
            )
    verification = check_profile(value, root, environment)
    return {
        "profile": value.id,
        "installed": verification["ready"],
        "environment": str(target),
        "verification": verification,
    }
