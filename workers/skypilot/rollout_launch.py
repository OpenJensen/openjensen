"""Validate local rollout inputs before submitting a managed GPU Job Group."""

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager, nullcontext
from enum import Enum
from pathlib import Path
from urllib.parse import urlsplit

import rollout_sdk
import yaml

_ROOT = Path(__file__).resolve().parent
_WORKER = (_ROOT / "../isaac_sim").resolve()
_REGION = "us-central1"
_IMAGE_PATTERN = r"[^\s]+@sha256:[0-9a-f]{64}"
_READY_RUN_SECONDS = 1800
_TEST_WALL_SECONDS = 7200
_EXPERIMENTAL_MAX_STEPS = 300
_EXPERIMENTAL_L4_TYPES = {"g2-standard-12", "g2-standard-16", "g2-standard-32"}
_EXPERIMENTAL_ZONES = {f"{_REGION}-{suffix}" for suffix in ("a", "b", "c", "f")}
_CANCEL_SECONDS = 120
_CHECKPOINT_MOUNT = "~/vla-checkpoint"
_REMOTE_WORKDIR = "~/sky_workdir"
_RESOURCE_TYPES = {
    "isaac": ("g2-standard-16", "L4:1", False),
    "vla": ("a3-highgpu-1g", "H100:1", True),
}


class _Mode(Enum):
    ROLLOUT = "rollout"
    READINESS = "readiness"
    EXPERIMENTAL = "experimental"


def _read(path):
    text = path.read_text()
    if "CHANGE_ME" in text:
        raise ValueError(f"Replace CHANGE_ME placeholders in {path}")
    return list(yaml.safe_load_all(text))


def _inside(value, root, label):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be relative to the worker directory")
    path = (root / path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"{label} is not a worker file: {value}")
    return path


def _network(path):
    documents = _read(path)
    if len(documents) != 1 or not isinstance(documents[0], dict):
        raise ValueError("config.yaml must contain one mapping")
    gcp = documents[0].get("gcp", {})
    proxy = gcp.get("ssh_proxy_command", {})
    if (
        not gcp.get("vpc_name")
        or not gcp.get("subnet_names")
        or gcp.get("use_internal_ips") is not True
        or not isinstance(proxy, dict)
        or _REGION not in proxy
    ):
        raise ValueError(
            "Configure the shared VPC, central subnet, internal IPs and us-central1 IAP proxy first"
        )
    jobs = documents[0].get("jobs", {})
    controller = jobs.get("controller", {}).get("resources", {})
    if (
        controller.get("infra") != f"gcp/{_REGION}"
        or jobs.get("force_disable_cloud_bucket") is not True
    ):
        raise ValueError(
            "Pin jobs.controller.resources.infra to gcp/us-central1 and set "
            "jobs.force_disable_cloud_bucket: true"
        )


def _variants(resources, name, mode):
    if "any_of" in resources:
        raise ValueError("Use experimental ordered L4 instance_type fallbacks")
    if "ordered" not in resources:
        return [resources]
    if name != "isaac" or mode != _Mode.EXPERIMENTAL:
        raise ValueError("Ordered resources require experimental Isaac motion")
    ordered = resources["ordered"]
    if not isinstance(ordered, list) or not ordered:
        raise ValueError("Ordered resources must list instance_type fallbacks")
    if any(not isinstance(item, dict) or set(item) != {"instance_type"} for item in ordered):
        raise ValueError("Ordered L4 fallbacks may override only instance_type")
    return [resources | item for item in ordered]


def _valid_infra(value, name, mode):
    region = f"gcp/{_REGION}"
    if value == region:
        return True
    if name != "isaac" or mode != _Mode.EXPERIMENTAL or not isinstance(value, str):
        return False
    parts = value.split("/")
    return len(parts) == 3 and "/".join(parts[:2]) == region and parts[2] in _EXPERIMENTAL_ZONES


def _tasks(documents, mode=_Mode.ROLLOUT):
    if len(documents) != 3 or any(not isinstance(doc, dict) for doc in documents):
        raise ValueError("Expected a Job Group header and two task mappings")
    header = documents[0]
    if header.get("execution") != "parallel" or header.get("primary_tasks") != ["isaac"]:
        raise ValueError("The Job Group must run in parallel with primary_tasks: [isaac]")
    tasks = {doc.get("name"): doc for doc in documents[1:]}
    if set(tasks) != set(_RESOURCE_TYPES):
        raise ValueError("Expected tasks named isaac and vla")
    for name, (machine, accelerator, spot) in _RESOURCE_TYPES.items():
        task = tasks[name]
        resources = task.get("resources", {})
        if "zone" in resources:
            raise ValueError("Encode the zone in infra instead of setting both infra and zone")
        machines = (
            _EXPERIMENTAL_L4_TYPES if name == "isaac" and mode == _Mode.EXPERIMENTAL else {machine}
        )
        for resource in _variants(resources, name, mode):
            if (
                resource.get("instance_type") not in machines
                or resource.get("accelerators") != accelerator
                or resource.get("use_spot") is not spot
                or not _valid_infra(resource.get("infra"), name, mode)
            ):
                raise ValueError(f"{name} must use its configured {_REGION} GPU resource")
        if resources.get("ports"):
            raise ValueError("Do not expose the policy server through public SkyPilot ports")
        if Path(task.get("workdir", "")).resolve() != _WORKER:
            raise ValueError(f"{name}.workdir must point to ../isaac_sim")
    return tasks


def _manifest(task, mode):
    envs = task["envs"]
    if envs.get("SIM_EXPERIMENTAL") and mode != _Mode.EXPERIMENTAL:
        raise ValueError("SIM_EXPERIMENTAL requires --experimental")
    image = envs.get("SIM_IMAGE", "")
    if not re.fullmatch(_IMAGE_PATTERN, image):
        raise ValueError("SIM_IMAGE must use an immutable @sha256 digest")
    destination = urlsplit(envs.get("SIM_RESULTS_URI", ""))
    if (
        destination.scheme != "gs"
        or not destination.netloc
        or destination.query
        or destination.fragment
    ):
        raise ValueError("SIM_RESULTS_URI must name a gs:// bucket prefix")
    acceptance = os.environ.get("ACCEPT_EULA", envs.get("ACCEPT_EULA"))
    if acceptance != "Y":
        raise ValueError("Set ACCEPT_EULA=Y after accepting NVIDIA's container license")
    envs["ACCEPT_EULA"] = acceptance
    manifest = _inside(envs.get("SIM_MANIFEST", ""), _WORKER, "SIM_MANIFEST")
    data = yaml.safe_load(manifest.read_text())
    if not isinstance(data, dict):
        raise ValueError("The rollout manifest must be a mapping")
    # Paths must remain valid after syncing the worker into the container.
    if "control_contract" not in data:
        _inside(
            str(manifest.parent.relative_to(_WORKER) / data["calibration"]), _WORKER, "calibration"
        )
    elif data.get("calibration") is not None:
        raise ValueError("Simulator-native policy requires calibration: null")
    _inside(str(manifest.parent.relative_to(_WORKER) / data["scene"]["uri"]), _WORKER, "scene.uri")
    environment = os.environ | {"PYTHONPATH": str(_WORKER)}
    command = [
        sys.executable,
        "-m",
        "sim_worker.rollout",
        "--manifest",
        str(manifest),
        "--validate-only",
    ]
    if mode == _Mode.EXPERIMENTAL:
        steps = data["control"]["steps"]
        if type(steps) is not int or not 0 < steps <= _EXPERIMENTAL_MAX_STEPS:
            raise ValueError(f"Experimental steps must be within 1..{_EXPERIMENTAL_MAX_STEPS}")
        command.append("--experimental")
    if mode == _Mode.READINESS:
        # Readiness validates schema and exports without authorizing robot motion.
        evidence = manifest.parent / "evidence"
        for name in ("dataset.json", "front-frame-000.jpg"):
            _inside(str(evidence.relative_to(_WORKER) / name), _WORKER, "readiness evidence")
        command = [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; "
            "from sim_worker.rollout.config import load; load(Path(sys.argv[1]))",
            str(manifest),
        ]
    subprocess.run(command, env=environment, check=True)
    return data


def _inspect(source):
    # Inspect exports with the shared CPU-only artifact validator.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "../isaac_sim"))
    from sim_worker.rollout.checkpoint import inspect_checkpoint

    return inspect_checkpoint(Path(source).expanduser().resolve())


def _bind_model(tasks, data, checkpoint, execute_steps, *, manifest_path=None):
    artifact = _inspect(checkpoint)
    joints = len(data["scene"]["joints"])
    if (artifact.state_dim, artifact.action_dim) != (joints, joints):
        raise ValueError("Checkpoint state/actions must match the scene joint count")

    if artifact.control_contract is not None:
        from sim_worker.rollout.contracts import SimSpec
        from sim_worker.rollout.control_contract import admit
        from sim_worker.rollout.control_schema import canonical

        if manifest_path is None:
            raise ValueError("Simulator contract admission requires the scene manifest path")
        contract = {"record": artifact.control_contract, "sha256": artifact.control_contract_sha256}
        if "control_contract" in data and canonical(data["control_contract"]) != canonical(
            contract
        ):
            raise ValueError("Rollout simulator contract differs from the selected checkpoint")
        scene = _inside(
            str(manifest_path.parent.relative_to(_WORKER) / data["scene"]["uri"]),
            _WORKER,
            "scene.uri",
        )
        admit(
            contract,
            SimSpec(
                str(scene),
                data["scene"]["camera"],
                data["scene"]["articulation"],
                tuple(data["scene"]["joints"]),
                data["capture"]["width"],
                data["capture"]["height"],
                data["control"]["fps"],
                data["control"]["physics_hz"],
            ),
        )
        data["control_contract"] = contract
        data["calibration"] = None
    elif "control_contract" in data:
        raise ValueError("Checkpoint has no simulator control contract")

    steps = data["control"]["execute_steps"] if execute_steps is None else execute_steps
    if type(steps) is not int or steps < 1:
        raise ValueError("execute-steps must be a positive integer")
    if execute_steps is None:
        steps = min(steps, artifact.action_steps)
    if steps > artifact.chunk_size:
        raise ValueError("execute-steps exceeds the checkpoint chunk size")

    # Keep robot geometry, joint order and calibration under scenario control.
    data["policy"]["model_id"] = artifact.model_id
    data["capture"].update(width=artifact.width, height=artifact.height)
    data["control"]["execute_steps"] = steps
    policy = tasks["vla"]
    policy["file_mounts"][_CHECKPOINT_MOUNT] = str(checkpoint)
    policy["envs"].update(
        MODEL_ID=artifact.model_id,
        POLICY_STATE_DIM=str(artifact.state_dim),
        POLICY_CAMERA_KEY=artifact.camera_key,
        POLICY_ACTION_STEPS=str(steps),
    )


@contextmanager
def _select_checkpoint(path, checkpoint, mode, execute_steps=None):
    if checkpoint is None:
        if execute_steps is not None:
            raise ValueError("--execute-steps requires --checkpoint")
        yield path
        return

    # Replace model placeholders before the existing complete-input validation.
    documents = list(yaml.safe_load_all(path.read_text()))
    tasks = _tasks(documents, mode)
    isaac = tasks["isaac"]
    manifest = _inside(isaac["envs"]["SIM_MANIFEST"], _WORKER, "SIM_MANIFEST")
    data = yaml.safe_load(manifest.read_text())
    _bind_model(
        tasks, data, checkpoint.expanduser().resolve(), execute_steps, manifest_path=manifest
    )

    # Unique snapshots preserve templates and isolate overlapping submissions.
    with (
        tempfile.NamedTemporaryFile(
            mode="w", prefix=".rollout-", suffix=".local.yaml", dir=manifest.parent
        ) as snapshot,
        tempfile.NamedTemporaryFile(
            mode="w", prefix=".rollout-", suffix=".yaml", dir=_ROOT
        ) as task,
    ):
        yaml.safe_dump(data, snapshot, sort_keys=False)
        snapshot.flush()
        relative = Path(snapshot.name).relative_to(_WORKER).as_posix()
        isaac["envs"]["SIM_MANIFEST"] = relative
        # Explicit mounting includes the snapshot even when workdir ignores it.
        isaac.setdefault("file_mounts", {})[f"{_REMOTE_WORKDIR}/{relative}"] = snapshot.name
        yaml.safe_dump_all(documents, task, sort_keys=False)
        task.flush()
        yield Path(task.name)


def _checkpoint(task, manifest):
    source = task.get("file_mounts", {}).get(_CHECKPOINT_MOUNT)
    if not isinstance(source, str):
        raise ValueError("Mount a local exported checkpoint at ~/vla-checkpoint")
    checkpoint = Path(source).expanduser().resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint}")
    artifact = _inspect(checkpoint)
    expected_control = (
        None
        if artifact.control_contract is None
        else {"record": artifact.control_contract, "sha256": artifact.control_contract_sha256}
    )
    from sim_worker.rollout.control_schema import canonical

    if canonical(manifest.get("control_contract")) != canonical(expected_control):
        raise ValueError("Rollout control contract is missing or differs from checkpoint")
    envs = task["envs"]
    if envs.get("MODEL_ID") != manifest["policy"]["model_id"]:
        raise ValueError("VLA MODEL_ID must match the rollout policy.model_id")
    if envs["MODEL_ID"] != artifact.model_id:
        raise ValueError(
            f"MODEL_ID must match the exported checkpoint fingerprint: {artifact.model_id}"
        )
    state_dim = int(envs.get("POLICY_STATE_DIM", 0))
    if state_dim != len(manifest["scene"]["joints"]) or (
        artifact.state_dim,
        artifact.action_dim,
    ) != (state_dim, state_dim):
        raise ValueError("POLICY_STATE_DIM, checkpoint state/actions and rollout joints must match")
    if envs.get("POLICY_CAMERA_KEY") != artifact.camera_key:
        raise ValueError("POLICY_CAMERA_KEY must match the checkpoint camera")
    if (manifest["capture"]["width"], manifest["capture"]["height"]) != (
        artifact.width,
        artifact.height,
    ):
        raise ValueError(
            f"Rollout capture must match checkpoint dimensions: {artifact.width}x{artifact.height}"
        )
    action_steps = int(envs.get("POLICY_ACTION_STEPS", 0))
    if not manifest["control"]["execute_steps"] <= action_steps <= artifact.chunk_size:
        raise ValueError(
            "POLICY_ACTION_STEPS must cover execute_steps "
            "without exceeding the checkpoint chunk size"
        )
    print(
        f"Checkpoint: {artifact.policy_type}, {artifact.width}x{artifact.height}, "
        f"{artifact.model_id}"
    )
    print(f"Execute {manifest['control']['execute_steps']} steps per policy request")
    return artifact


def _prepare(path, mode=_Mode.ROLLOUT):
    if not os.environ.get("SIM_PROJECT_ID"):
        raise ValueError("Set SIM_PROJECT_ID before launching")
    _network(_ROOT / "config.yaml")
    documents = _read(path)
    tasks = _tasks(documents, mode)
    manifest = _manifest(tasks["isaac"], mode)
    artifact = _checkpoint(tasks["vla"], manifest)
    # Discovery uses this submission's labels, never another experiment's server.
    run_id = uuid.uuid4().hex
    for name, task in tasks.items():
        task["resources"].setdefault("labels", {}).update(
            {"rollout-id": run_id, "rollout-role": name}
        )
        task["envs"]["ROLLOUT_ID"] = run_id
    if mode == _Mode.READINESS:
        documents[0]["name"] = f"isaac-ready-{run_id[:8]}"
        tasks["isaac"]["envs"]["SIM_READY_TIMEOUT"] = str(_READY_RUN_SECONDS)
        tasks["isaac"]["run"] = (
            "set -euo pipefail\n"
            'timeout --signal=TERM --kill-after=60 "$SIM_READY_TIMEOUT" '
            "bash ~/sim-control/check_ready.sh\n"
        )
    if mode == _Mode.EXPERIMENTAL:
        documents[0]["name"] = f"isaac-{artifact.policy_type}-test-{run_id[:8]}"
        tasks["isaac"]["envs"]["SIM_EXPERIMENTAL"] = "1"
    if mode != _Mode.ROLLOUT:
        for task in tasks.values():
            task["resources"]["job_recovery"] = {"max_restarts_on_errors": 0}
    return documents


def _run_sdk(command, *, timeout, check=False):
    # Stop the adapter gracefully so it can reap an in-flight observation.
    # This bounded local cleanup precedes any cloud cancellation request; it is
    # not an exact remote runtime or billing cap.
    try:
        return rollout_sdk.run_owned(command, timeout=timeout, terminate_grace=5)
    except subprocess.CalledProcessError as error:
        return subprocess.CompletedProcess(command, error.returncode)


def _launch(documents, path, mode, options=(), receipt_dir=None):
    receipt_dir = receipt_dir or Path(tempfile.mkdtemp(prefix="firebird-rollout-"))
    receipt_path = receipt_dir / "submission.json"
    command = [
        "bash",
        str(_ROOT / "sky.sh"),
        "rollout-sdk",
        "launch",
        str(path),
        "--receipt",
        str(receipt_path),
        *options,
    ]
    timeout = _TEST_WALL_SECONDS if mode != _Mode.ROLLOUT else None
    if mode != _Mode.ROLLOUT and "--detach-run" in options:
        name = documents[0]["name"]
        print(
            f"Detached {mode.value} requires monitoring: "
            f"bash sky.sh jobs cancel --name {name} --yes",
            flush=True,
        )
    try:
        return _run_sdk(command, check=False, timeout=timeout).returncode
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        if mode != _Mode.ROLLOUT:
            name = documents[0]["name"]
            target = ["--name", name]
            attempt = {
                "group_name": name,
                "job_id": None,
                "reason": type(error).__name__,
                "status": "requesting",
                "resource_deletion": "unverified",
            }
            try:
                receipt = rollout_sdk.validate_receipt(rollout_sdk.read_json(receipt_path))
                if receipt["group_name"] == name:
                    target = [str(receipt["job_id"])]
                    attempt["job_id"] = receipt["job_id"]
            except (OSError, ValueError, TypeError, KeyError) as _error:
                pass  # Submission may not yet have returned an ID; the unique name is best effort.
            cancellation_path = receipt_dir / "cancellation.json"

            # A full disk must not prevent the already-authorized timeout cancellation.
            def record_attempt():
                try:
                    rollout_sdk.write_json(cancellation_path, attempt)
                except (OSError, ValueError) as record_error:
                    print(
                        f"Cancellation receipt unavailable ({type(record_error).__name__}).",
                        file=sys.stderr,
                    )

            record_attempt()
            print(
                f"Requesting cancellation of {mode.value} group {name}; "
                "resource deletion remains unverified.",
                flush=True,
            )
            try:
                subprocess.run(
                    ["bash", str(_ROOT / "sky.sh"), "jobs", "cancel", *target, "--yes"],
                    check=True,
                    timeout=_CANCEL_SECONDS,
                )
            except (subprocess.SubprocessError, OSError, KeyboardInterrupt) as cancel_error:
                attempt.update(status="failed_or_unknown", error=type(cancel_error).__name__)
                record_attempt()
                raise
            attempt["status"] = "requested"
            record_attempt()
        raise


def _submit(args, path):
    documents = _prepare(path, args.mode)
    expected_model = getattr(args, "expected_model_id", None)
    if expected_model is not None:
        tasks = {task["name"]: task for task in documents[1:]}
        if (
            not re.fullmatch(r"sha256:[a-f0-9]{64}", expected_model)
            or tasks["vla"]["envs"]["MODEL_ID"] != expected_model
        ):
            raise ValueError("Checkpoint differs from the application accepted model fingerprint")
    expected_prefix = getattr(args, "expected_results_prefix", None)
    if expected_prefix is not None:
        tasks = {task["name"]: task for task in documents[1:]}
        if tasks["isaac"]["envs"]["SIM_RESULTS_URI"].rstrip("/") != expected_prefix:
            raise ValueError("Result prefix differs from the application accepted profile")
    if args.validate_only:
        print("Local rollout inputs validated. Cloud network, quota and capacity are not checked.")
        return 0
    receipt_dir = getattr(args, "receipt_dir", None)
    if receipt_dir is not None:
        if args.mode != _Mode.EXPERIMENTAL:
            raise ValueError("Persistent application receipts require experimental mode")
        receipt_dir.mkdir(parents=True, exist_ok=False)
        tasks = {task["name"]: task for task in documents[1:]}
        env = tasks["isaac"]["envs"]
        manifest = _inside(env["SIM_MANIFEST"], _WORKER, "SIM_MANIFEST")
        data = yaml.safe_load(manifest.read_text())
        # Each app job owns a distinct prefix even if the operator's template
        # is shared. The remote runner adds its own episode UUID beneath it.
        env["SIM_RESULTS_URI"] = env["SIM_RESULTS_URI"].rstrip("/") + "/" + env["ROLLOUT_ID"]
        rollout_sdk.write_json(
            receipt_dir / "launch-context.json",
            {
                "schema_version": 1,
                "group_name": documents[0]["name"],
                "rollout_id": env["ROLLOUT_ID"],
                "results_prefix": env["SIM_RESULTS_URI"],
                "model_id": data["policy"]["model_id"],
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            },
        )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix=".rollout-", dir=_ROOT
    ) as task:
        yaml.safe_dump_all(documents, task, sort_keys=False)
        task.flush()
        options = tuple(
            flag
            for enabled, flag in ((args.yes, "--yes"), (args.detach_run, "--detach-run"))
            if enabled
        )
        return _launch(documents, task.name, args.mode, options, receipt_dir)


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", type=Path, default=Path("rollout.local.yaml"))
    checkpoints = parser.add_mutually_exclusive_group()
    checkpoints.add_argument(
        "--checkpoint", type=Path, help="Select a complete local ACT or SmolVLA export"
    )
    checkpoints.add_argument(
        "--checkpoint-archive", type=Path, help="Select a bounded ACT or SmolVLA TAR package"
    )
    parser.add_argument(
        "--execute-steps",
        type=int,
        help="Actions to execute before replanning; requires a checkpoint directory or archive",
    )
    parser.add_argument(
        "--receipt-dir", type=Path, help="New private directory for application ownership receipts"
    )
    parser.add_argument(
        "--expected-model-id", help="Require the accepted model fingerprint before submission"
    )
    parser.add_argument(
        "--expected-results-prefix", help="Require the configured result prefix before submission"
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip SkyPilot's launch confirmation")
    parser.add_argument("--detach-run", action="store_true", help="Return after submitting the job")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--check-ready",
        dest="mode",
        action="store_const",
        const=_Mode.READINESS,
        default=_Mode.ROLLOUT,
        help="Test actual model inference across VMs without moving the robot",
    )
    modes.add_argument(
        "--experimental",
        dest="mode",
        action="store_const",
        const=_Mode.EXPERIMENTAL,
        help="Allow candidate calibration for an experimental rollout of at most 300 steps",
    )
    args = parser.parse_args()
    source = args.checkpoint or args.checkpoint_archive
    if source is not None:
        # Resolve relative inputs before the historical task-directory change.
        source = source.expanduser().absolute()
    if args.receipt_dir is not None:
        args.receipt_dir = args.receipt_dir.expanduser().resolve()
    os.chdir(_ROOT)
    try:
        sys.path.insert(0, str(_WORKER))
        from sim_worker.rollout.checkpoint_package import resolve_checkpoint

        lifetime = (
            resolve_checkpoint(source, archive=args.checkpoint_archive is not None)
            if source is not None
            else nullcontext(None)
        )
        with lifetime as resolved:
            checkpoint = resolved.directory if resolved is not None else None
            with _select_checkpoint(
                args.task, checkpoint, args.mode, args.execute_steps
            ) as selected:
                return _submit(args, selected)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"Rollout launch refused: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
