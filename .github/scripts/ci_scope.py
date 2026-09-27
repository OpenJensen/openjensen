"""Select affected CI jobs using a local, NUL-delimited Git diff; fail open to all checks."""

import json
import os
import re
import subprocess
from pathlib import Path

SCOPES = (
    "application",
    "quantization",
    "unified_quantization",
    "training",
    "benchmark",
    "act",
    "teaching",
    "decision",
)
WORKERS = {
    "quantization": "workers/vla_cpp/",
    "unified_quantization": "workers/firebird_quant/",
    "training": "workers/smolvla_qlora/",
    "benchmark": "workers/benchmark_gpu/",
    "act": "workers/act_optimizer/",
    "teaching": "workers/teaching/",
    "decision": "workers/decision/",
}


def select(paths):
    scopes = set()
    for path in paths:
        if path.startswith((".github/", "scripts/ci_", "tests/test_ci_scope")):
            return set(SCOPES)
        if path.startswith(("packages/core/", "apps/web/", "tests/", "scripts/")) or path in {
            "pyproject.toml",
            "uv.lock",
            "package.json",
            "pnpm-lock.yaml",
            "pnpm-workspace.yaml",
            ".node-version",
            "playwright.config.ts",
            "playwright.diagnostics.config.ts",
            "docs/workspace-guide.md",  # Rendered into the application at build time.
        }:
            scopes.add("application")
        if path.startswith("packages/core/src/vla_platform/lifecycle/") or path in {
            "packages/core/src/vla_platform/contracts.py",
            "packages/core/src/vla_platform/worker.py",
        }:
            scopes.update(WORKERS)
        for name, prefix in WORKERS.items():
            if path.startswith(prefix):
                scopes.add(name)
                scopes.add("application")  # Application adapters exercise worker boundaries.
        if path.startswith(("workers/_cpu_readers/", "workers/isaac_sim/", "workers/skypilot/")):
            scopes.add("application")
            if path.startswith("workers/isaac_sim/"):
                scopes.add("teaching")
        if path.startswith("workers/smolvla_qlora/"):
            scopes.update(("act", "teaching"))  # Native-source export and local dataset producer.
        if path.startswith("workers/vla_cpp/"):
            scopes.add("benchmark")  # Cross-stack benchmark imports policykit.spatial_runtime.
        if path.startswith("packages/core/src/vla_platform/datasets/"):
            scopes.update(("training", "act", "teaching"))
        known = path.startswith(
            (
                "packages/core/",
                "apps/web/",
                "tests/",
                "scripts/",
                "workers/",
                "docs/",
                "deploy/",
                "apps/desktop/",
            )
        ) or path in {
            "README.md",
            "LICENSE",
            ".gitignore",
            ".gitattributes",
            ".node-version",
            "pyproject.toml",
            "uv.lock",
            "package.json",
            "pnpm-lock.yaml",
            "pnpm-workspace.yaml",
            "playwright.config.ts",
            "playwright.diagnostics.config.ts",
        }
        known_worker = (
            path.startswith(
                tuple(WORKERS.values())
                + (
                    "workers/_cpu_readers/",
                    "workers/isaac_sim/",
                    "workers/skypilot/",
                )
            )
            or path == "workers/README.md"
        )
        if not known or (path.startswith("workers/") and not known_worker):
            return set(SCOPES)
        if path.startswith(("deploy/", "apps/desktop/")):
            scopes.add("application")
    return scopes


def changed_paths(event, document):
    if event == "pull_request":
        base = document["pull_request"]["base"]["sha"]
        head = document["pull_request"]["head"]["sha"]
        separator = "..."
    elif event == "push":
        base, head, separator = document["before"], document["after"], ".."
    elif event == "merge_group":
        base = document["merge_group"]["base_sha"]
        head = document["merge_group"]["head_sha"]
        separator = ".."
    else:
        raise ValueError("Manual or unknown event requests complete verification")
    for revision in (base, head):
        if not isinstance(revision, str) or not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
            raise ValueError("Invalid Git revision")
        if set(revision) == {"0"}:
            raise ValueError("New branch requires complete verification")
    result = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", "-z", base + separator + head, "--"],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return [os.fsdecode(path) for path in result.stdout.split(b"\0") if path]


def main():
    try:
        document = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        selected = select(changed_paths(os.environ["GITHUB_EVENT_NAME"], document))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        selected = set(SCOPES)
        print("Change detection unavailable or complete run requested: running every scope.")
    lines = [f"{name}={'true' if name in selected else 'false'}" for name in SCOPES]
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write("\n".join(lines) + "\n")
    print("Selected checks: " + (", ".join(sorted(selected)) or "documentation only"))


if __name__ == "__main__":
    main()
