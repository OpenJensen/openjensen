"""Prepare a default-off local Tauri candidate from an explicitly accepted frozen payload.

No build, activation, environment install, model execution or network operation occurs.
"""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

from payload_inventory import MAX_MANIFEST, verify
from sidecar_resources import json_object, load_resources, read_regular


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def prepare(payload: Path, manifest: Path, acceptance: Path, output: Path) -> dict:
    if not all(p.is_absolute() for p in (payload, manifest, acceptance, output)):
        raise ValueError("All paths must be absolute")
    if payload.is_symlink() or not payload.is_dir():
        raise ValueError("Payload must be a real directory")
    payload = payload.resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("Candidate output must not exist")
    resolved_output = output.resolve()
    if resolved_output.is_relative_to(payload) or payload.is_relative_to(resolved_output):
        raise ValueError("Candidate output and payload must be disjoint")
    raw = read_regular(manifest, MAX_MANIFEST)
    recorded = json_object(raw)
    verify(payload, recorded)
    report_raw = read_regular(acceptance, 1024 * 1024)
    report = json_object(report_raw)
    resources = load_resources(payload / "_internal/sidecar-resources")
    required = [
        "source_and_relocated_payloads_unchanged",
        "workspace_owner_exclusion",
        "shutdown_and_parent_eof_exit_zero",
        "restart_preserved_project_and_job",
        "parent_eof_before_start_created_no_workspace",
        "tampered_static_rejected_before_workspace",
        "wrong_control_nonce_shut_down_owned_listener",
    ]
    if any(report.get(key) is not True for key in required):
        raise ValueError("A completed frozen acceptance receipt is required")
    if (
        report.get("source_build_id") != resources.build_id
        or report.get("resources_sha256") != resources.manifest_sha256
        or report.get("payload_identity_sha256") != recorded.get("identity_sha256")
        or not re.fullmatch(r"[a-f0-9]{64}", str(recorded.get("identity_sha256")))
    ):
        raise ValueError("Frozen acceptance identity does not match the payload")
    native = {
        name: item["macho_cpu_types"]
        for name, item in recorded["entries"].items()
        if item["kind"] == "file" and item["macho_cpu_types"] is not None
    }
    executable = recorded["entries"].get("firebird-sidecar", {})
    if (
        not native
        or any(arch != [0x100000C] for arch in native.values())
        or executable.get("kind") != "file"
        or not executable.get("mode", 0) & 0o111
        or report.get("native_files") != native
    ):
        raise ValueError("The complete accepted ARM64 payload is required")
    identifier = "dev.firebird.workbench.experiment.p" + recorded["identity_sha256"][:16]
    # All values interpolated into Rust are bounded digests/integers from independently
    # rehashed files and strict resource validation. No operator text becomes Rust code.
    pin = (
        "// Generated local experiment pin. No production activation.\n"
        f'const LOCAL_EXPERIMENT_IDENTIFIER: &str = "{identifier}";\n'
        "const LOCAL_EXPERIMENT_PAYLOAD: PayloadPin = PayloadPin {\n"
        f' executable_sha256: "{executable["sha256"]}",\n'
        f" executable_bytes: {executable['bytes']},\n"
        f' resources_sha256: "{resources.manifest_sha256}",\n'
        f' build_id: "{resources.build_id}",\n'
        " manifest: crate::payload_manifest::ManifestPin {\n"
        f'  sha256: "{sha(raw)}", bytes: {len(raw)},\n'
        f'  identity: "{recorded["identity_sha256"]}",\n'
        f"  entries: {recorded['entry_count']}, file_bytes: {recorded['file_bytes']},\n"
        " },\n};\n"
    )
    output.mkdir()  # exclusive creation; preserve partial copies on failure
    staged = output / "resources/firebird-sidecar"
    staged.parent.mkdir()
    shutil.copytree(payload, staged, symlinks=True)
    verify(staged, recorded)
    manifest_copy = output / "resources/payload-manifest.json"
    manifest_copy.write_bytes(raw)
    (output / "local-payload-pin.rs").write_text(pin)
    repo = Path(__file__).resolve().parents[3]
    overlay = {
        "productName": "OPEN JENSEN Local Experiment",
        "identifier": identifier,
        "build": {"frontendDist": str(repo / "apps/desktop/frontend")},
        "bundle": {
            "resources": {
                str(staged): "sidecar/firebird-sidecar",
                str(manifest_copy): "sidecar/payload-manifest.json",
            }
        },
    }
    (output / "tauri.local.json").write_text(json.dumps(overlay, indent=2) + "\n")
    verify(payload, recorded)
    result = {
        "schema_version": 1,
        "child_source_commit": resources.build_id,
        "payload_identity_sha256": recorded["identity_sha256"],
        "manifest_sha256": sha(raw),
        "acceptance_receipt_sha256": sha(report_raw),
        "pin_sha256": sha(pin.encode()),
        "identifier": identifier,
        "overlay_sha256": sha((output / "tauri.local.json").read_bytes()),
        "production_activation": False,
        "native_candidate_built": False,
        "payload_and_copy_unchanged": True,
    }
    (output / "preparation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("payload", "manifest", "acceptance", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.payload, args.manifest, args.acceptance, args.output)))


if __name__ == "__main__":
    main()
