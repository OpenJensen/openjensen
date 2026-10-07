"""Select affected CI work from a local, NUL-delimited Git diff.

Pull requests run a fast lane: only the checks the diff can break. The complete lane
(worker adapter tests, mobile browser project) runs after merge to main, on tags, on
manual dispatch and on schedule. Edits to CI itself run everything.

Outputs (all single-line, written to GITHUB_OUTPUT):

* the eight scope flags in ``SCOPES`` (unchanged names; ``application`` = Python job);
* ``desktop_lane`` / ``mobile_lane`` / ``web_lane``: which Playwright projects run;
* ``docs_only``: a valid diff that needs no job at all;
* ``full``: true for main pushes, tags, manual dispatch and schedule;
* ``py_shards`` / ``web_matrix``: JSON matrices consumed by ``application.yml``.
"""

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
LANES = ("desktop_lane", "mobile_lane")
WORKERS = {
    "quantization": "workers/vla_cpp/",
    "unified_quantization": "workers/firebird_quant/",
    "training": "workers/smolvla_qlora/",
    "benchmark": "workers/benchmark_gpu/",
    "act": "workers/act_optimizer/",
    "teaching": "workers/teaching/",
    "decision": "workers/decision/",
}
ALL = set(SCOPES) | set(LANES)
# What a pull request runs when a path cannot be classified: Python + desktop browser.
PR_LANE = {"application", "desktop_lane"}
WEB_BOTH = {"desktop_lane", "mobile_lane"}
CI_PREFIXES = (".github/workflows/", ".github/scripts/", ".github/actions/")
# Directories whose tests are exercised by the Python suite (sky_*, isaac_* tests).
PYTHON_WORKER_DIRS = ("workers/_cpu_readers/", "workers/isaac_sim/", "workers/skypilot/")
PYTHON_AND_DESKTOP_PREFIXES = (
    "packages/core/",
    "scripts/",
    "deploy/",
    "apps/desktop/",
    "tests/fixtures/",  # Browser fixture worker used by tests/web/serve.py.
)
PYTHON_AND_DESKTOP_FILES = {"pyproject.toml", "uv.lock"}
DESKTOP_FILES = {
    "package.json",
    "pnpm-lock.yaml",
    "pnpm-workspace.yaml",
    ".node-version",
    "docs/workspace-guide.md",  # Rendered into the application at build time.
}
MOBILE_FILES = {"playwright.config.ts", "playwright.diagnostics.config.ts"}
DOC_FILES = {"README.md", "LICENSE", ".gitignore", ".gitattributes", "workers/README.md"}
DEFAULT_SHARDS = {"CI_PY_SHARDS": 2, "CI_DESKTOP_SHARDS": 4, "CI_MOBILE_SHARDS": 6}


def path_scopes(path, adapters):
    """Scopes and lanes one changed path needs; None when the path is not recognised."""
    if path.startswith(CI_PREFIXES) or path.startswith(("scripts/ci_", "tests/test_ci_scope")):
        return set(ALL)
    scopes = set()
    if path.startswith(("apps/web/", "tests/web/")) or path in MOBILE_FILES:
        scopes |= WEB_BOTH
    elif path in DESKTOP_FILES:
        scopes.add("desktop_lane")
    elif path.startswith(PYTHON_AND_DESKTOP_PREFIXES) or path in PYTHON_AND_DESKTOP_FILES:
        scopes |= PR_LANE
        if path.startswith("packages/core/src/vla_platform/lifecycle/") or path in {
            "packages/core/src/vla_platform/contracts.py",
            "packages/core/src/vla_platform/worker.py",
        }:
            scopes.update(WORKERS)
        if path.startswith("packages/core/src/vla_platform/datasets/"):
            scopes.update(("training", "act", "teaching"))
    elif path.startswith("tests/"):
        scopes.add("application")
    elif path.startswith(PYTHON_WORKER_DIRS):
        scopes.add("application")
        if path.startswith("workers/isaac_sim/"):
            scopes.add("teaching")
    elif any(path.startswith(prefix) for prefix in WORKERS.values()):
        for name, prefix in WORKERS.items():
            if path.startswith(prefix):
                scopes.add(name)
        if adapters:
            scopes.add("application")  # Adapters exercise worker boundaries; PRs skip this.
        if path.startswith("workers/smolvla_qlora/"):
            scopes.update(("act", "teaching"))  # Native-source export, local dataset producer.
        if path.startswith("workers/vla_cpp/"):
            scopes.add("benchmark")  # Cross-stack benchmark imports policykit.spatial_runtime.
    elif path in DOC_FILES or path.startswith(("docs/", ".github/")):
        pass  # Documentation and repository metadata need no job.
    else:
        return None
    return scopes


def select(paths, adapters=False):
    """Return the scope and lane names a set of changed paths needs.

    Unknown paths fail open to the pull-request lane (Python + desktop browser).
    """
    selected = set()
    for path in paths:
        scopes = path_scopes(path, adapters)
        selected |= PR_LANE if scopes is None else scopes
    if "mobile_lane" in selected:
        selected.add("desktop_lane")
    return selected


def changed_paths(event, document):
    if event == "pull_request":
        base = document["pull_request"]["base"]["sha"]
        head = document["pull_request"]["head"]["sha"]
        separator = "..."
    elif event == "push":
        base, head, separator = document["before"], document["after"], ".."
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


def classify(event, ref, document):
    """Return (selected, full, docs_only) for one workflow event."""
    if event in {"workflow_dispatch", "schedule"} or (
        event == "push" and ref.startswith("refs/tags/")
    ):
        return set(ALL), True, False
    try:
        paths = changed_paths(event, document)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        if event == "pull_request":
            print("Change detection unavailable: running the pull-request lane.")
            return set(PR_LANE), False, False
        print("Change detection unavailable or unknown event: running everything.")
        return set(ALL), event == "push", False
    if event == "push":
        # Main is scoped by its own diff, but gets the mobile project and worker adapters.
        selected = select(paths, adapters=True)
        if "desktop_lane" in selected:
            selected.add("mobile_lane")
        return selected, True, not selected
    selected = select(paths)
    return selected, False, not selected


def shard_count(name):
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return DEFAULT_SHARDS[name]
    return value if 1 <= value <= 12 else DEFAULT_SHARDS[name]


def shards(count):
    return [f"{index}/{count}" for index in range(1, count + 1)]


def web_matrix(selected):
    """One 'core' leg (OpenAPI/type/build checks, pure/openapi/workflow, diagnostics) plus
    desktop and mobile slices. Playwright splits one project by test count and every test
    in a project costs about the same, so shard counts follow measured project cost."""
    if "desktop_lane" not in selected:
        return []
    legs = [{"name": "core", "kind": "core", "project": "", "shard": ""}]
    for project, lane, variable in (
        ("desktop", "desktop_lane", "CI_DESKTOP_SHARDS"),
        ("mobile", "mobile_lane", "CI_MOBILE_SHARDS"),
    ):
        if lane in selected:
            legs += [
                {"name": f"{project} {item}", "kind": "shard", "project": project, "shard": item}
                for item in shards(shard_count(variable))
            ]
    return legs


def main():
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    ref = os.environ.get("GITHUB_REF", "")
    try:
        document = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    except (OSError, ValueError, KeyError):
        document = {}
    selected, full, docs_only = classify(event, ref, document)
    flags = {name: name in selected for name in (*SCOPES, *LANES)}
    flags["web_lane"] = flags["desktop_lane"] or flags["mobile_lane"]
    flags["docs_only"] = docs_only
    flags["full"] = full
    legs = web_matrix(selected)
    lines = [f"{name}={'true' if value else 'false'}" for name, value in flags.items()]
    lines.append("py_shards=" + json.dumps(shards(shard_count("CI_PY_SHARDS"))))
    lines.append("web_matrix=" + json.dumps(legs, separators=(",", ":")))
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write("\n".join(lines) + "\n")
    print("Selected checks: " + (", ".join(sorted(selected)) or "documentation only"))
    print(f"event={event} full={full} docs_only={docs_only}")
    for leg in legs:
        print("  web leg: " + leg["name"])


if __name__ == "__main__":
    main()
