"""Offline, single same-format handoff; not exposed by entrypoint or native IPC.

Trusted same-user filesystem and accepted payload receipts, not an authorization sandbox.
Every ambiguous publication leaves a fixed journal that blocks native startup. No API,
Execution, migration, payload execution, automatic recovery or post-start rollback.
"""

import hashlib
import os
import re
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path

import workspace_maintenance as m
from payload_inventory import MAX_MANIFEST, verify
from sidecar_resources import load_resources

JOURNAL = ".desktop-handoff"
PROOFS = (
    "source_and_relocated_payloads_unchanged",
    "workspace_owner_exclusion",
    "shutdown_and_parent_eof_exit_zero",
    "restart_preserved_project_and_job",
    "parent_eof_before_start_created_no_workspace",
    "tampered_static_rejected_before_workspace",
    "wrong_control_nonce_shut_down_owned_listener",
)


@dataclass(frozen=True)
class PayloadFiles:
    root: Path
    manifest: Path
    acceptance: Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def raw_file(path, maximum):
    with m.opened(path) as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise m.MaintenanceError("Transaction file exceeds its bound")
    return raw


def accepted_payload(files, budget):
    """Rehash full payload and independently recorded frozen acceptance; never run it."""
    budget.check()
    root = m.directory(files.root)
    for path in (files.manifest, files.acceptance):
        m.directory(path.parent)
    raw = raw_file(files.manifest, MAX_MANIFEST)
    manifest = m.parse_json(raw)
    verify(root, manifest)
    report_raw = raw_file(files.acceptance, 1024 * 1024)
    report = m.parse_json(report_raw)
    resources = load_resources(root / "_internal/sidecar-resources")
    if not isinstance(report, dict) or any(report.get(key) is not True for key in PROOFS):
        raise m.MaintenanceError("Completed frozen payload acceptance is required")
    native = {
        name: item["macho_cpu_types"]
        for name, item in manifest["entries"].items()
        if item["kind"] == "file" and item["macho_cpu_types"] is not None
    }
    executable = manifest["entries"].get("firebird-sidecar", {})
    if (
        report.get("source_build_id") != resources.build_id
        or report.get("resources_sha256") != resources.manifest_sha256
        or report.get("payload_identity_sha256") != manifest["identity_sha256"]
        or not native
        or any(arch != [0x100000C] for arch in native.values())
        or report.get("native_files") != native
        or executable.get("kind") != "file"
        or not executable.get("mode", 0) & 0o111
    ):
        raise m.MaintenanceError("Accepted ARM64 payload identity differs")
    budget.check()
    return {
        "paths": {
            "root": str(root),
            "manifest": str(files.manifest),
            "acceptance": str(files.acceptance),
        },
        "manifest_sha256": sha(raw),
        "acceptance_sha256": sha(report_raw),
        "marker": {
            "schema_version": 2,
            "owner": "openjensen-desktop",
            "build_id": resources.build_id,
            "resources_sha256": resources.manifest_sha256,
            "payload_identity_sha256": manifest["identity_sha256"],
        },
    }


def publish(path, value):
    """No-replace receipt. Failed flush removes only this call's still-owned inode."""
    data = m.canonical(value) + b"\n"
    if len(data) > m.Limits().json_bytes:
        raise m.MaintenanceError("Transaction receipt exceeds its bound")
    stream = None
    owned = None
    try:
        stream = path.open("xb")
        owned = os.fstat(stream.fileno())
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
        m.sync_directory(path.parent)
    except BaseException:
        if owned is not None:
            try:
                now = path.lstat()
                if stat.S_ISREG(now.st_mode) and (now.st_dev, now.st_ino) == (
                    owned.st_dev,
                    owned.st_ino,
                ):
                    path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        if stream is not None:
            stream.close()
    return sha(data)


def digest_inventory(files):
    return sha(m.canonical(files))


def exact_inventory(root, expected, budget):
    if m.canonical(m.inventory(root, budget)) != m.canonical(expected):
        raise m.MaintenanceError("Transaction workspace or backup inventory changed")


def prepare_handoff(workspace, backup_output, old_payload, new_payload, *, limits=m.Limits()):
    workspace = m.directory(workspace)
    if workspace.name != "desktop-v1":
        raise m.MaintenanceError("Only the dedicated desktop-v1 workspace is supported")
    journal = workspace.parent / JOURNAL
    backup_output = Path(backup_output)
    m.directory(backup_output.parent)
    if (
        backup_output == journal
        or backup_output.is_relative_to(journal)
        or journal.is_relative_to(backup_output)
    ):
        raise m.MaintenanceError("Backup and journal must be disjoint")
    budget = m.Budget(limits)
    with m.locks(workspace):
        if journal.exists() or journal.is_symlink():
            raise m.MaintenanceError("An existing transaction requires explicit inspection")
        old = accepted_payload(old_payload, budget)
        new = accepted_payload(new_payload, budget)
        for payload in (old_payload, new_payload):
            for path in (payload.root, payload.manifest, payload.acceptance):
                if (
                    path == journal
                    or path.is_relative_to(journal)
                    or journal.is_relative_to(path)
                    or path == backup_output
                    or path.is_relative_to(backup_output)
                    or backup_output.is_relative_to(path)
                ):
                    raise m.MaintenanceError(
                        "Payload inputs must be disjoint from journal and backup outputs"
                    )
        if old["marker"] == new["marker"]:
            raise m.MaintenanceError("A distinct verified target payload is required")
        before = m.inventory(workspace, budget)
        old_raw = raw_file(workspace / "desktop-owner.json", 4096)
        if (
            m.marker(workspace, budget) != old["marker"]
            or sha(old_raw) != before["desktop-owner.json"]["sha256"]
        ):
            raise m.MaintenanceError("Current workspace is not owned by the verified old payload")
        backup = m._prepare_backup_locked(workspace, backup_output, budget)
        if m.canonical(backup["files"]) != m.canonical(before):
            raise m.MaintenanceError("Workspace changed before backup")
        backup_raw = raw_file(backup_output / "receipt.json", limits.json_bytes)
        if backup_raw != m.canonical(backup) + b"\n":
            raise m.MaintenanceError("Backup receipt changed")
        exact_inventory(workspace, before, budget)
        prepared = {
            "schema_version": 1,
            "operation": "desktop.workspace.handoff",
            "transaction_id": uuid.uuid4().hex,
            "workspace": str(workspace),
            "workspace_path_sha256": sha(os.fsencode(workspace)),
            "old_payload": old,
            "new_payload": new,
            "old_marker_hex": old_raw.hex(),
            "files": before,
            "backup": str(backup_output),
            "backup_receipt_sha256": sha(backup_raw),
            "backup_inventory": m.inventory(backup_output, budget),
            "limits": m.asdict(limits),
        }
        journal.mkdir(mode=0o700)
        m.sync_directory(journal.parent)
        publish(journal / "prepared.json", prepared)
        return prepared


def load_prepared(workspace, budget):
    journal = m.directory(workspace.parent / JOURNAL)
    raw = raw_file(journal / "prepared.json", budget.limits.json_bytes)
    value = m.parse_json(raw)
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema_version",
            "operation",
            "transaction_id",
            "workspace",
            "workspace_path_sha256",
            "old_payload",
            "new_payload",
            "old_marker_hex",
            "files",
            "backup",
            "backup_receipt_sha256",
            "backup_inventory",
            "limits",
        }
        or not isinstance(value.get("transaction_id"), str)
        or not re.fullmatch(r"[a-f0-9]{32}", value["transaction_id"])
        or value.get("schema_version") != 1
        or type(value["schema_version"]) is not int
        or value.get("operation") != "desktop.workspace.handoff"
        or value.get("workspace") != str(workspace)
        or value.get("workspace_path_sha256") != sha(os.fsencode(workspace))
    ):
        raise m.MaintenanceError("Transaction path or schema differs")
    if (
        not isinstance(value["files"], dict)
        or not isinstance(value["backup_inventory"], dict)
        or not isinstance(value["old_marker_hex"], str)
        or not re.fullmatch(r"(?:[a-f0-9]{2}){1,4096}", value["old_marker_hex"])
    ):
        raise m.MaintenanceError("Invalid transaction inventory or marker")
    try:
        m.Limits(**value["limits"])
    except (TypeError, ValueError) as exc:
        raise m.MaintenanceError("Invalid transaction limits") from exc
    # All derived records are checked against source bytes again before any marker mutation.
    for key in ("old_payload", "new_payload"):
        try:
            paths = value[key]["paths"]
            checked = accepted_payload(
                PayloadFiles(**{k: Path(v) for k, v in paths.items()}), budget
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise m.MaintenanceError("Invalid transaction payload record") from exc
        if m.canonical(checked) != m.canonical(value[key]):
            raise m.MaintenanceError("Transaction payload changed")
    backup = m.directory(Path(value["backup"]))
    exact_inventory(backup, value["backup_inventory"], budget)
    receipt = raw_file(backup / "receipt.json", budget.limits.json_bytes)
    if sha(receipt) != value["backup_receipt_sha256"] or m.canonical(
        m.parse_json(receipt)["files"]
    ) != m.canonical(value["files"]):
        raise m.MaintenanceError("Transaction backup binding differs")
    old_raw = bytes.fromhex(value["old_marker_hex"])
    if sha(old_raw) != value["files"]["desktop-owner.json"]["sha256"] or m.canonical(
        m.parse_json(old_raw)
    ) != m.canonical(value["old_payload"]["marker"]):
        raise m.MaintenanceError("Transaction original marker binding differs")
    return journal, value, sha(raw)


def journal_names(journal, expected):
    expected = set(expected)
    names = set()
    with os.scandir(journal) as entries:
        for entry in entries:
            if entry.name not in expected:
                raise m.MaintenanceError(
                    "Transaction publication is incomplete, started or unrecognized"
                )
            names.add(entry.name)
    if names != expected:
        raise m.MaintenanceError("Transaction publication is incomplete, started or unrecognized")


def post_files(prepared):
    result = dict(prepared["files"])
    raw = m.canonical(prepared["new_payload"]["marker"]) + b"\n"
    result["desktop-owner.json"] = {
        **result["desktop-owner.json"],
        "sha256": sha(raw),
        "bytes": len(raw),
    }
    return result


def replace_marker(workspace, journal, before, raw, budget):
    """Only cooperating owners are excluded; same-user hostile mutation is not a sandbox."""
    temporary = journal / "marker.pending"
    with temporary.open("xb") as stream:
        stream.write(raw)
        os.fchmod(stream.fileno(), before["desktop-owner.json"]["mode"])
        stream.flush()
        os.fsync(stream.fileno())
    exact_inventory(workspace, before, budget)
    os.replace(temporary, workspace / "desktop-owner.json")
    m.sync_directory(workspace)
    m.sync_directory(journal)


def commit_handoff(workspace, *, limits=m.Limits()):
    workspace = m.directory(workspace)
    budget = m.Budget(limits)
    with m.locks(workspace):
        journal, prepared, digest = load_prepared(workspace, budget)
        journal_names(journal, ["prepared.json"])
        exact_inventory(workspace, prepared["files"], budget)
        intent = {"schema_version": 1, "prepared_sha256": digest}
        publish(journal / "commit-intent.json", intent)
        replace_marker(
            workspace,
            journal,
            prepared["files"],
            m.canonical(prepared["new_payload"]["marker"]) + b"\n",
            budget,
        )
        after = post_files(prepared)
        exact_inventory(workspace, after, budget)
        load_prepared(workspace, budget)  # recheck both payloads and backup after mutation
        result = {
            "schema_version": 1,
            "prepared_sha256": digest,
            "workspace_path_sha256": prepared["workspace_path_sha256"],
            "marker": prepared["new_payload"]["marker"],
            "inventory_sha256": digest_inventory(after),
        }
        publish(journal / "committed.json", result)
        return result


def reverse_unstarted_handoff(workspace, *, limits=m.Limits()):
    workspace = m.directory(workspace)
    budget = m.Budget(limits)
    with m.locks(workspace):
        journal, prepared, digest = load_prepared(workspace, budget)
        journal_names(journal, ["prepared.json", "commit-intent.json", "committed.json"])
        committed = m.read_json(journal / "committed.json", limits.json_bytes)
        after = post_files(prepared)
        expected = {
            "schema_version": 1,
            "prepared_sha256": digest,
            "workspace_path_sha256": prepared["workspace_path_sha256"],
            "marker": prepared["new_payload"]["marker"],
            "inventory_sha256": digest_inventory(after),
        }
        if m.canonical(committed) != m.canonical(expected) or m.canonical(
            m.read_json(journal / "commit-intent.json", limits.json_bytes)
        ) != m.canonical({"schema_version": 1, "prepared_sha256": digest}):
            raise m.MaintenanceError("Committed transaction record differs")
        exact_inventory(workspace, after, budget)
        publish(journal / "reverse-intent.json", {"schema_version": 1, "prepared_sha256": digest})
        replace_marker(workspace, journal, after, bytes.fromhex(prepared["old_marker_hex"]), budget)
        exact_inventory(workspace, prepared["files"], budget)
        load_prepared(workspace, budget)
        result = {
            **expected,
            "marker": prepared["old_payload"]["marker"],
            "inventory_sha256": digest_inventory(prepared["files"]),
        }
        publish(journal / "reversed.json", result)
        return result
