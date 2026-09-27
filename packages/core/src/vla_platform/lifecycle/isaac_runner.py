"""Owned, bounded native ACT/SmolVLA Isaac runs through the existing launcher.

Only the isolated runner Python imports cloud libraries. Local completion is
execution evidence, never a pickup score or proof that cloud resources vanished.
"""

import asyncio
import hashlib
import json
import math
import os
import re
import signal
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit

JSON_LIMIT = 1024 * 1024
OUTPUT_LIMIT = 128 * 1024
POLL_SECONDS = 5
COMMAND_SECONDS = 60
CANCEL_SECONDS = 120
MAX_ARTIFACT_BYTES = 512 * 1024**2
GROUP = re.compile(r"isaac-(?:act|smolvla)-test-[a-f0-9]{8}\Z")
HEX = re.compile(r"[a-f0-9]{64}\Z")
UUID = re.compile(r"[a-f0-9]{32}\Z")
ACTIVE = {"PENDING", "SUBMITTED", "STARTING", "RUNNING", "WINDING_DOWN", "RECOVERING", "CANCELLING"}
TERMINAL = {
    "SUCCEEDED",
    "CANCELLED",
    "FAILED",
    "FAILED_SETUP",
    "FAILED_PRECHECKS",
    "FAILED_NO_RESOURCE",
    "FAILED_CONTROLLER",
}
ARTIFACTS = {
    "job-result.json": JSON_LIMIT,
    "outputs/result.json": JSON_LIMIT,
    "outputs/trajectory.jsonl": 64 * 1024**2,
    "outputs/video.mp4": 400 * 1024**2,
}


def _read(path, limit=JSON_LIMIT):
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("Invalid or oversized simulation record")
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError("Simulation record exceeds its limit")
    return content


def _json(path):
    def pairs(entries):
        result = {}
        for key, value in entries:
            if key in result:
                raise ValueError("Duplicate simulation JSON key")
            result[key] = value
        return result

    def bad(value):
        raise ValueError("Non-finite simulation JSON")

    def floating(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Non-finite simulation JSON")
        return result

    return json.loads(
        _read(path).decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=bad,
        parse_float=floating,
    )


def _write(path, record):
    data = (json.dumps(record, allow_nan=False, sort_keys=True) + "\n").encode()
    if len(data) > JSON_LIMIT or path.is_symlink():
        raise ValueError("Invalid simulation record destination")
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".simulation-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024**2), b""):
            result.update(part)
    return result.hexdigest()


def _uri(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"gs://[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]/[A-Za-z0-9/_-]+", value
    ):
        raise ValueError("Simulation results need a fixed private gs:// prefix")
    if any(part in {"", ".", ".."} for part in urlsplit(value).path.strip("/").split("/")):
        raise ValueError("Invalid simulation result prefix")
    return value.rstrip("/")


@dataclass(frozen=True)
class SimulationProfile:
    id: str
    label: str
    runner_root: Path
    task: Path
    project_id: str
    credential_file: Path
    results_uri: str
    task_sha256: str
    accept_eula: bool
    sky_api_endpoint: str | None = None
    gcloud_config: Path | None = None

    @property
    def skypilot(self):
        return self.runner_root / "workers" / "skypilot"

    @property
    def python(self):
        return self.skypilot / ".venv" / "bin" / "python"

    def public(self):
        return {
            "id": self.id,
            "label": self.label,
            "architectures": ["act", "smolvla"],
            "experimental": True,
            "task_object": "cup",
            "provider": "gcp",
            "accelerators": ["L4", "H100"],
            "task_success": None,
        }

    def identity_hash(self):
        if self.task.is_symlink() or _sha(self.task) != self.task_sha256:
            raise ValueError("Registered simulation task changed")
        # Hash reviewed scripts, scene assets and runtime configuration, never
        # credentials, environments, models, caches or hidden receipt files.
        inventory = {}
        total = 0
        for folder in (self.skypilot, self.runner_root / "workers" / "isaac_sim"):
            for current, dirs, files in os.walk(folder, followlinks=False):
                dirs[:] = sorted(
                    name for name in dirs if not name.startswith(".") and name != "__pycache__"
                )
                if any((Path(current) / name).is_symlink() for name in dirs):
                    raise ValueError("Simulation source snapshot contains a symlink directory")
                for name in sorted(files):
                    path = Path(current) / name
                    if path.resolve() == self.credential_file.resolve():
                        continue
                    if name.startswith(".") or path.suffix in {
                        ".pyc",
                        ".safetensors",
                        ".mp4",
                        ".parquet",
                    }:
                        continue
                    if path.is_symlink() or not path.is_file():
                        raise ValueError("Simulation source snapshot contains a nonregular file")
                    total += path.stat().st_size
                    if total > 1024**3 or len(inventory) >= 10000:
                        raise ValueError("Simulation source snapshot exceeds its bound")
                    inventory[str(path.relative_to(self.runner_root))] = _sha(path)
        for name in (
            "rollout_launch.py",
            "rollout_sdk.py",
            "sky.sh",
            "launch-rollout.sh",
            "config.yaml",
        ):
            if f"workers/skypilot/{name}" not in inventory:
                raise ValueError("Simulation runner is missing reviewed source/configuration")
        record = {
            "id": self.id,
            "project_id": self.project_id,
            "results_uri": self.results_uri,
            "task_sha256": self.task_sha256,
            "sky_api_endpoint": self.sky_api_endpoint,
            "gcloud_config": str(self.gcloud_config) if self.gcloud_config else None,
            "accept_eula": self.accept_eula,
            "files": inventory,
        }
        return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()

    def environment(self):
        if not self.accept_eula:
            raise ValueError("The operator must accept the NVIDIA container license")
        if self.credential_file.is_symlink() or not self.credential_file.is_file():
            raise ValueError("Simulation credentials are not configured")
        # Do not inherit unrelated provider keys, proxy settings or training's
        # SkyPilot endpoint/configuration. Credentials remain server-side.
        env = {
            key: os.environ[key]
            for key in ("HOME", "PATH", "TMPDIR", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR")
            if key in os.environ
        }
        env.update(
            SIM_PROJECT_ID=self.project_id,
            ACCEPT_EULA="Y",
            GOOGLE_APPLICATION_CREDENTIALS=str(self.credential_file),
            CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE=str(self.credential_file),
            CLOUDSDK_CORE_PROJECT=self.project_id,
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(self.runner_root / "workers" / "isaac_sim"),
        )
        if self.gcloud_config is not None:
            env["CLOUDSDK_CONFIG"] = str(self.gcloud_config)
        if self.sky_api_endpoint:
            env["SKYPILOT_API_SERVER_ENDPOINT"] = self.sky_api_endpoint
        return env


def load_profiles(path):
    if path is None:
        return ()
    document = _json(Path(path))
    if (
        not isinstance(document, dict)
        or set(document) != {"schema_version", "profiles"}
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
    ):
        raise ValueError("Invalid simulation profile configuration")
    entries = document["profiles"]
    if not isinstance(entries, list) or len(entries) > 10:
        raise ValueError("Invalid simulation profiles")
    profiles = []
    required = {
        "id",
        "label",
        "runner_root",
        "task",
        "project_id",
        "credential_file",
        "results_uri",
        "task_sha256",
        "accept_eula",
    }
    for item in entries:
        if (
            not isinstance(item, dict)
            or not required <= set(item)
            or set(item) - required - {"sky_api_endpoint", "gcloud_config"}
        ):
            raise ValueError("Invalid simulation profile fields")
        if not isinstance(item["id"], str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", item["id"]):
            raise ValueError("Invalid simulation profile identity")
        if not isinstance(item["label"], str) or not 1 <= len(item["label"]) <= 120:
            raise ValueError("Invalid simulation label")
        if not isinstance(item["project_id"], str) or not re.fullmatch(
            r"[a-z][a-z0-9-]{4,28}[a-z0-9]", item["project_id"]
        ):
            raise ValueError("Invalid simulation project")
        if (
            not isinstance(item["task_sha256"], str)
            or not HEX.fullmatch(item["task_sha256"])
            or type(item["accept_eula"]) is not bool
        ):
            raise ValueError("Simulation profile requires its pinned task and EULA choice")
        values = dict(item)
        for name in ("runner_root", "task", "credential_file"):
            if not isinstance(item[name], str) or not Path(item[name]).is_absolute():
                raise ValueError("Simulation paths must be operator-owned absolute paths")
            values[name] = Path(item[name])
        if (
            not values["task"]
            .resolve()
            .is_relative_to((values["runner_root"] / "workers/skypilot").resolve())
        ):
            raise ValueError("Simulation task must stay in its registered runner")
        if item.get("gcloud_config") is not None:
            if (
                not isinstance(item["gcloud_config"], str)
                or not Path(item["gcloud_config"]).is_absolute()
            ):
                raise ValueError("Cloud SDK configuration must be an operator-owned absolute path")
            values["gcloud_config"] = Path(item["gcloud_config"])
        values["results_uri"] = _uri(item["results_uri"])
        endpoint = item.get("sky_api_endpoint")
        if endpoint is not None:
            if (
                not isinstance(endpoint, str)
                or not 1 <= len(endpoint) <= 2048
                or any(character.isspace() for character in endpoint)
            ):
                raise ValueError("Invalid operator SkyPilot endpoint")
            parsed = urlsplit(endpoint)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("Invalid operator SkyPilot endpoint")
        profiles.append(SimulationProfile(**values))
    if len({p.id for p in profiles}) != len(profiles):
        raise ValueError("Duplicate simulation profile")
    return tuple(profiles)


def admit(profile, metadata):
    checkpoint = metadata.get("checkpoint", metadata) if isinstance(metadata, dict) else {}
    if checkpoint.get("policy_type") not in {"act", "smolvla"}:
        raise ValueError("Native simulation supports ACT and SmolVLA exports")
    if checkpoint.get("state_dim") != 6 or checkpoint.get("action_dim") != 6:
        raise ValueError("The SO101 cup profile requires six state and action coordinates")
    for name in ("width", "height", "chunk_size", "action_steps"):
        if type(checkpoint.get(name)) is not int or checkpoint[name] < 1:
            raise ValueError("Invalid native checkpoint dimensions")
    if (
        checkpoint["width"] % 2
        or checkpoint["height"] % 2
        or min(checkpoint["width"], checkpoint["height"]) < 2
        or max(checkpoint["width"], checkpoint["height"]) > 1920
        or checkpoint["action_steps"] > checkpoint["chunk_size"]
    ):
        raise ValueError("Checkpoint camera or action horizon is unsupported")
    if not isinstance(checkpoint.get("camera_key"), str) or not checkpoint["camera_key"].startswith(
        "observation.images."
    ):
        raise ValueError("Checkpoint needs one named camera")
    if not isinstance(checkpoint.get("model_id"), str) or not re.fullmatch(
        r"sha256:[a-f0-9]{64}", checkpoint["model_id"]
    ):
        raise ValueError("Checkpoint has no verified model fingerprint")
    return {
        "profile_id": profile.id,
        "profile_sha256": profile.identity_hash(),
        "model_id": checkpoint["model_id"],
        "experimental": True,
        "calibration_verified": False,
        "task_success": None,
    }


async def _finish(task):
    interrupted = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            interrupted = True
    result = task.result()
    if interrupted:
        raise asyncio.CancelledError
    return result


async def _stop(process):
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif process.returncode is None:
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), 3)
    except TimeoutError:
        pass
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.returncode is None:
        process.kill()
    await process.wait()


async def _command(profile, args, *, timeout=COMMAND_SECONDS, capture=False):
    spawning = asyncio.create_task(
        asyncio.create_subprocess_exec(
            *args,
            cwd=profile.skypilot,
            env=profile.environment(),
            stdout=asyncio.subprocess.PIPE if capture else asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=os.name == "posix",
        )
    )
    try:
        process = await asyncio.shield(spawning)
    except asyncio.CancelledError:

        async def spawned_cleanup():
            process = await spawning
            await _stop(process)

        await _finish(asyncio.create_task(spawned_cleanup()))
        raise

    async def consume():
        if not capture:
            return await process.wait(), b""
        content = bytearray()
        while chunk := await process.stdout.read(16384):
            content.extend(chunk)
            if len(content) > OUTPUT_LIMIT:
                raise ValueError("Simulation subprocess exceeded its output limit")
        return await process.wait(), bytes(content)

    try:
        return await asyncio.wait_for(consume(), timeout)
    finally:
        await _finish(asyncio.create_task(_stop(process)))


def _context(profile, job_directory):
    context = _json(job_directory / "receipts" / "launch-context.json")
    if (
        not isinstance(context, dict)
        or type(context.get("schema_version")) is not int
        or context.get("schema_version") != 1
        or not GROUP.fullmatch(str(context.get("group_name", "")))
        or not UUID.fullmatch(str(context.get("rollout_id", "")))
    ):
        raise ValueError("Invalid owned simulation launch context")
    if context.get("results_prefix") != profile.results_uri + "/" + context["rollout_id"]:
        raise ValueError("Simulation result destination differs from its registered profile")
    if context["group_name"][-8:] != context["rollout_id"][:8]:
        raise ValueError("Simulation launch identity differs")
    return context


def _receipt(context, path):
    value = _json(path)
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("skypilot") != "0.13.0"
        or value.get("group_name") != context["group_name"]
        or type(value.get("job_id")) is not int
        or value["job_id"] < 1
    ):
        raise ValueError("Simulation submission identity is incomplete or mismatched")
    return value


def _states(receipt, rows):
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("Both owned simulation tasks must be observed")
    if not isinstance(receipt.get("tasks"), list) or len(receipt["tasks"]) != 2:
        raise ValueError("Invalid simulation task identities")
    expected = {item["task_name"]: item for item in receipt["tasks"]}
    if set(expected) != {"isaac", "vla"}:
        raise ValueError("Invalid simulation task identities")
    states = {}
    for row in rows:
        name = row.get("task_name")
        if (
            name not in expected
            or name in states
            or row.get("job_id") != receipt["job_id"]
            or row.get("job_name") != receipt["group_name"]
            or row.get("task_id") != expected[name]["task_id"]
            or row.get("is_primary_in_job_group") is not (name == "isaac")
            or row.get("is_job_group") is not True
            or row.get("execution") != "parallel"
            or row.get("status") not in ACTIVE | TERMINAL
        ):
            raise ValueError("Simulation queue identity/status mismatch")
        states[name] = row["status"]
    return states


async def recover(profile, job_directory):
    """Cancel only the saved owned identity. Never resubmit or adopt late outputs."""
    directory = Path(job_directory)
    record = {"status": "cleanup_unknown", "resource_deletion": "unverified", "resubmitted": False}

    def save_receipt():
        # A failed disk must not prevent cancellation of an already owned group.
        record["receipt_saved"] = True
        try:
            _write(directory / "recovery.json", record)
        except (OSError, ValueError) as _error:
            record["receipt_saved"] = False

    try:
        context = _context(profile, directory)
        accepted = _json(directory / "request.json")
        if (
            accepted.get("profile_id") != profile.id
            or accepted.get("profile_sha256") != await asyncio.to_thread(profile.identity_hash)
            or accepted.get("model_id") != context.get("model_id")
        ):
            raise ValueError("Saved simulation identity differs from the current control target")
    except (OSError, ValueError, KeyError, TypeError) as _error:
        record["reason"] = "No valid durable launch identity; operator reconciliation required"
        save_receipt()
        return record
    target = ["--name", context["group_name"]]
    try:
        receipt = _receipt(context, directory / "receipts" / "submission.json")
        target = [str(receipt["job_id"])]
        record["job_id"] = receipt["job_id"]
    except (OSError, ValueError, KeyError, TypeError) as _error:
        pass
    record["group_name"] = context["group_name"]
    save_receipt()
    try:
        code, _ = await _command(
            profile,
            ["bash", str(profile.skypilot / "sky.sh"), "jobs", "cancel", *target, "--yes"],
            timeout=CANCEL_SECONDS,
        )
        record["status"] = "cancellation_requested" if code == 0 else "cleanup_unknown"
    except (OSError, ValueError, TimeoutError) as _error:
        record["status"] = "cleanup_unknown"
    save_receipt()
    return record


async def run(
    profile,
    checkpoint_directory,
    job_directory,
    event,
    timeout_seconds,
    *,
    expected_profile_sha256=None,
    expected_model_id=None,
):
    """Run one experimental profile; all cloud calls use the fixed runner identity."""
    directory = Path(job_directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "request.json").exists() or (directory / "receipts").exists():
        raise ValueError("Simulation already has dispatch evidence; never resubmit this job")
    identity = await asyncio.to_thread(profile.identity_hash)
    if expected_profile_sha256 is not None and identity != expected_profile_sha256:
        raise ValueError("Simulation profile changed after this job was accepted")
    if type(timeout_seconds) is not int or not 30 <= timeout_seconds <= 7200:
        raise ValueError("Simulation deadline must be between 30 and 7200 seconds")
    profile.environment()
    code, output = await _command(
        profile,
        [
            str(profile.python),
            "-c",
            (
                "import json,sys; from dataclasses import asdict; from pathlib import Path; "
                "from sim_worker.rollout.checkpoint import inspect_checkpoint; "
                "print(json.dumps(asdict(inspect_checkpoint(Path(sys.argv[1])))))"
            ),
            str(checkpoint_directory),
        ],
        capture=True,
    )
    if code:
        raise ValueError("Native checkpoint inspection failed before simulation launch")
    admission = admit(profile, json.loads(output))
    if expected_model_id is not None and admission["model_id"] != expected_model_id:
        raise ValueError("Native checkpoint changed after this job was accepted")
    if admission["profile_sha256"] != identity:
        raise ValueError("Simulation source changed during admission")
    _write(directory / "request.json", admission)
    completed = False
    try:
        async with asyncio.timeout(timeout_seconds):
            await event(
                "preparing", "Preparing the experimental cup simulation", {"profile_id": profile.id}
            )
            if await asyncio.to_thread(profile.identity_hash) != identity:
                raise ValueError("Simulation profile changed before submission")
            code, _ = await _command(
                profile,
                [
                    "bash",
                    str(profile.skypilot / "launch-rollout.sh"),
                    str(profile.task),
                    "--checkpoint",
                    str(checkpoint_directory),
                    "--experimental",
                    "--yes",
                    "--detach-run",
                    "--receipt-dir",
                    str(directory / "receipts"),
                    "--expected-model-id",
                    admission["model_id"],
                    "--expected-results-prefix",
                    profile.results_uri,
                ],
                timeout=min(timeout_seconds, 600),
            )
            if code:
                raise ValueError(
                    "Simulation submission did not complete; inspect its saved ownership receipt"
                )
            context = _context(profile, directory)
            if context.get("model_id") != admission["model_id"]:
                raise ValueError("Launched simulation checkpoint identity changed")
            receipt_path = directory / "receipts" / "submission.json"
            receipt = _receipt(context, receipt_path)
            await event(
                "running",
                "Simulation submitted; waiting for both owned tasks",
                {"job_id": receipt["job_id"], "group_name": receipt["group_name"]},
            )
            previous = None
            while True:
                observation = directory / "observation.json"
                code, _ = await _command(
                    profile,
                    [
                        "bash",
                        str(profile.skypilot / "sky.sh"),
                        "rollout-sdk",
                        "observe",
                        "--receipt",
                        str(receipt_path),
                        "--output",
                        str(observation),
                    ],
                )
                if code:
                    raise ValueError("The owned simulation status could not be verified")
                states = _states(receipt, _json(observation))
                if states != previous:
                    await event("running", "Simulation task status updated", {"tasks": states})
                    previous = states
                if states["isaac"] in TERMINAL and states["vla"] in TERMINAL:
                    break
                await asyncio.sleep(POLL_SECONDS)
            if states["isaac"] != "SUCCEEDED" or states["vla"] not in {"SUCCEEDED", "CANCELLED"}:
                raise ValueError("Simulation task execution failed; task success was not measured")
            await event("saving", "Collecting the owned simulation video and trajectory", None)
            code, _ = await _command(
                profile,
                [str(profile.python), str(Path(__file__).resolve()), "--collect", str(directory)],
                timeout=180,
            )
            if code:
                raise ValueError(
                    "Simulation artifacts could not be verified; remote evidence is retained"
                )
            result = _json(directory / "artifacts.json")
            completed = True
            report = {
                "schema_version": 1,
                "kind": "isaac_simulation",
                "profile_id": profile.id,
                "profile_sha256": identity,
                "model_id": admission["model_id"],
                "execution_status": "succeeded",
                "task_success": None,
                "calibration_verified": False,
                "experimental": True,
                "resource_deletion": "unverified",
                "tasks": states,
                "artifacts": result["artifacts"],
                "run_id": result["run_id"],
                "note": "Experimental cup rollout completed; pickup success was not measured.",
            }
            _write(directory / "report.json", report)
            return report
    finally:
        if not completed:
            await _finish(asyncio.create_task(recover(profile, directory)))


def _response_json(response):
    response.raise_for_status()
    if response.status_code != 200:
        raise ValueError("Unexpected artifact response status")
    content = bytearray()
    for part in response.iter_content(16384):
        content.extend(part)
        if len(content) > JSON_LIMIT:
            raise ValueError("Artifact metadata exceeds its limit")
    return json.loads(content)


def _collect(directory):
    """Isolated read-only GCS helper; exact per-job prefix, generations and bounds."""
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    context = _json(directory / "receipts" / "launch-context.json")
    request = _json(directory / "request.json")
    prefix = _uri(context["results_prefix"])
    if context.get("model_id") != request["model_id"] or not UUID.fullmatch(
        prefix.rsplit("/", 1)[-1]
    ):
        raise ValueError("Artifact collection identity mismatch")
    target = urlsplit(prefix)
    base = "https://storage.googleapis.com/storage/v1/b/" + quote(target.netloc, safe="") + "/o"
    name_prefix = target.path.strip("/") + "/"
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/devstorage.read_only"]
    )
    with AuthorizedSession(credentials) as session:
        with session.get(
            base,
            params={"prefix": name_prefix, "maxResults": 32},
            timeout=30,
            stream=True,
            allow_redirects=False,
        ) as response:
            listing = _response_json(response)
        if listing.get("nextPageToken"):
            raise ValueError("Too many simulation result objects")
        objects = listing.get("items", [])
        if (
            not isinstance(objects, list)
            or len(objects) > 32
            or any(
                not isinstance(item, dict) or not isinstance(item.get("name"), str)
                for item in objects
            )
            or len({item["name"] for item in objects}) != len(objects)
        ):
            raise ValueError("Invalid simulation result listing")
        completions = [
            item for item in objects if item.get("name", "").endswith("/job-result.json")
        ]
        if len(completions) != 1:
            raise ValueError("Exactly one committed simulation completion is required")
        root = completions[0]["name"].removesuffix("job-result.json")
        run_id = root.removeprefix(name_prefix).rstrip("/")
        if not root.startswith(name_prefix) or not UUID.fullmatch(run_id):
            raise ValueError("Result object is outside the owned simulation prefix")
        inventory = {
            item["name"][len(root) :]: item
            for item in objects
            if item.get("name", "").startswith(root)
        }
        artifacts = []
        total = 0
        output = directory / "artifacts"
        output.mkdir(exist_ok=False)
        for name, maximum in ARTIFACTS.items():
            item = inventory.get(name)
            if (
                not item
                or not str(item.get("generation", "")).isdigit()
                or not str(item.get("size", "")).isdigit()
            ):
                raise ValueError("Simulation result is missing required generation-bound artifacts")
            size = int(item["size"])
            total += size
            if not 0 < size <= maximum or total > MAX_ARTIFACT_BYTES:
                raise ValueError("Simulation artifact exceeds its byte limit")
            url = base + "/" + quote(item["name"], safe="")
            local = output / name
            local.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            count = 0
            with session.get(
                url,
                params={
                    "alt": "media",
                    "generation": item["generation"],
                    "ifGenerationMatch": item["generation"],
                },
                timeout=30,
                stream=True,
                allow_redirects=False,
            ) as data:
                data.raise_for_status()
                if data.status_code != 200:
                    raise ValueError("Artifact download refused a redirect or partial response")
                with local.open("xb") as stream:
                    for block in data.iter_content(1024**2):
                        count += len(block)
                        if count > size:
                            raise ValueError("Simulation object grew during download")
                        stream.write(block)
                        digest.update(block)
            if count != size:
                raise ValueError("Simulation object was truncated")
            artifacts.append(
                {
                    "path": str(local.relative_to(directory)),
                    "sha256": digest.hexdigest(),
                    "bytes": size,
                    "generation": item["generation"],
                }
            )
    job = _json(output / "job-result.json")
    rollout = _json(output / "outputs/result.json")
    if (
        job.get("run_id") != run_id
        or job.get("status") != "succeeded"
        or job.get("exit_code") != 0
        or job.get("mode") != "experimental"
        or job.get("rollout_result") != rollout
        or rollout.get("status") != "succeeded"
        or rollout.get("model_id") != request["model_id"]
        or rollout.get("manifest_sha256") != context.get("manifest_sha256")
    ):
        raise ValueError("Downloaded completion does not match this simulation's model/manifest")
    _write(directory / "artifacts.json", {"run_id": run_id, "artifacts": artifacts})


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--collect":
        raise SystemExit("Only the fixed isolated result collector is supported")
    try:
        _collect(Path(sys.argv[2]))
    except Exception as error:
        print("Simulation artifact collection failed: " + type(error).__name__, file=sys.stderr)
        raise SystemExit(1) from None
