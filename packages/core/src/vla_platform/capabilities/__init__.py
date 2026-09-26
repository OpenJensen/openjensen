"""Operation availability and scoped evidence; never probes GPUs or imports ML libraries."""

import platform
from pathlib import Path

from vla_platform.contracts import Capability, CapabilityTarget

BASELINE = "prep:docs/tasks/evidence/2026-09-26-foundation-intake.md"
WINDOWS_API = "prep:docs/tasks/evidence/2026-09-26-windows-phase1-API-001.md"
WINDOWS_INTAKE = "prep:docs/tasks/evidence/2026-09-26-windows-phase1-INT-001.md"
APPLICATION_CI = "prep:docs/tasks/evidence/2026-09-26-windows-phase1-FND-003.md"
OPERATING_SYSTEMS = ("linux", "windows", "macos")
DEVICES = ("cpu", "cuda", "mps")
HOST_SYSTEMS = {"Linux", "Windows", "Darwin"}


def _targets(*, metadata: bool, local: bool = False) -> list[CapabilityTarget]:
    targets = []
    for operating_system in OPERATING_SYSTEMS:
        for device in DEVICES:
            support, evidence, refs = "unsupported", "untested", []
            reason = "No operation backend is implemented or selected."
            if metadata and device != "cpu":
                reason = "The metadata adapter runs on CPU only; no accelerator backend is used."
            elif metadata:
                support = "untested"
                reason = "CPU metadata adapter registered; no recorded run on this OS/source path."
                if operating_system == "macos":
                    support, evidence, refs = (
                        "supported",
                        "fixture" if local else "live_source",
                        [BASELINE],
                    )
                    reason = "Metadata-only macOS run; no media, policy or GPU validation."
                elif operating_system == "windows" and local:
                    support, evidence, refs = "supported", "fixture", [WINDOWS_API]
                    reason = "Windows CPU metadata worker ran with synthetic local fixture data."
                elif operating_system == "windows":
                    support, evidence, refs = "supported", "live_source", [WINDOWS_INTAKE]
                    reason = (
                        "Pinned public HF metadata intake on Windows CPU at f7f9a41; "
                        "web transport, CLI and fresh-process persistence verified. "
                        "No interactive-browser, media, policy or GPU validation."
                    )
                elif operating_system == "linux" and local:
                    support, evidence, refs = "supported", "fixture", [APPLICATION_CI]
                    reason = (
                        "Linux CPU local metadata fixture passed CI at 6319c09. "
                        "Later Parquet preview/local-hardening revisions are unverified on Linux; "
                        "no public-HF, policy or GPU validation."
                    )
            targets.append(
                CapabilityTarget(
                    operating_system=operating_system,
                    device=device,
                    support=support,
                    evidence_state=evidence,
                    evidence_refs=refs,
                    reason=reason,
                )
            )
    return targets


def registry(*, local_root: Path | None = None) -> list[Capability]:
    """Available means configured adapter, not proof of a run on the current host.

    Local configuration does not prove that any particular path exists or is valid.
    Neither registration nor a fixture run establishes native policy/GPU support.
    """
    host_supported = platform.system() in HOST_SYSTEMS
    items = [
        Capability(
            stage="Dataset",
            operation="dataset.inspect",
            status="available" if host_supported else "planned",
            description=(
                "Public HF LeRobot metadata only; revision-pinned bounded reads. "
                "No semantic or media validation."
            ),
            backend="lerobot.metadata",
            implementation="registered",
            runnable=host_supported,
            targets=_targets(metadata=True),
        ),
        *[
            Capability(
                stage=stage,
                operation=operation,
                status="planned",
                description=description,
                targets=_targets(metadata=False),
            )
            for stage, operation, description in [
                ("Fine-tune", "policy.finetune", "Native training recipe integration is planned."),
                (
                    "Distill",
                    "policy.distill",
                    "A supported teacher/student pair is not selected yet.",
                ),
                (
                    "Quantize",
                    "policy.quantize",
                    "Packed conversion and reload remain unimplemented.",
                ),
                (
                    "Evaluate",
                    "policy.evaluate",
                    "Closed-loop simulation requires a compatible runtime.",
                ),
                (
                    "Run",
                    "policy.run",
                    "Exact tested policy export and native execution are planned.",
                ),
            ]
        ],
        Capability(
            stage="Dataset",
            operation="dataset.inspect.local",
            status="available" if local_root is not None and host_supported else "planned",
            description=(
                "Local metadata intake is enabled within the configured dataset root."
                if local_root is not None
                else "Local intake is disabled; configure FIREBIRD_LOCAL_DATA_ROOT."
            ),
            backend="lerobot.metadata",
            implementation="registered",
            runnable=local_root is not None and host_supported,
            targets=_targets(metadata=True, local=True),
        ),
    ]
    if not host_supported:
        for item in items:
            if item.implementation == "registered":
                item.description += " The current operating system is not a registered target."
    return items
