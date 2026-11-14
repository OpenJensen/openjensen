"""Exactly which code is this process running?

A version number alone cannot answer that. Release automation writes the next patch
number into every open pull request, so several unmerged branches legitimately carry the
same number at once. Only one commit ever becomes that release.

This module gives every build one honest name, captured once at startup:

* the published release of ``0.1.30`` is ``0.1.30``;
* anything else is ``0.1.31-dev+62227c5`` (``.dirty`` with uncommitted source changes):
  the number says which release it will become, the suffix says it is not that release.

Two machines run the same code exactly when their labels are equal.
"""

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

from vla_platform import __version__

ROOT = Path(__file__).resolve().parents[4]
SHA = re.compile(r"[0-9a-f]{40}\Z")
# Written into installations: by the updater (channel main) or by a hand-built preview.
MARKERS = (".openjensen-build.json", ".openjensen-release.json")


@dataclass(frozen=True)
class Build:
    version: str
    commit: str | None
    branch: str | None
    dirty: bool
    released: bool

    @property
    def label(self) -> str:
        if self.released:
            return self.version
        suffix = self.commit[:7] if self.commit else "unknown"
        return f"{self.version}-dev+{suffix}" + (".dirty" if self.dirty else "")

    def public(self) -> dict:
        return {**asdict(self), "label": self.label}


def _git(root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            timeout=3,
            stdin=subprocess.DEVNULL,
        )
    except OSError, subprocess.SubprocessError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _marker(root: Path) -> tuple[str, bool] | None:
    """The commit an installation records, and whether it came from the release channel.

    Hand-built previews record their commit too, under another channel. They are not
    the release, however complete they look.
    """
    for name in MARKERS:
        try:
            record = json.loads((root / name).read_text())
            value = record["commit"]
        except OSError, ValueError, KeyError, TypeError:
            continue
        if isinstance(value, str) and SHA.fullmatch(value):
            return value, record.get("channel", "main") == "main"
    return None


def identify(root: Path = ROOT, version: str = __version__) -> Build:
    """Read the identity of the source tree at ``root``. Never raises."""
    if marker := _marker(root):
        commit, released = marker
        return Build(version, commit, "main" if released else None, dirty=False, released=released)
    if not (root / ".git").exists():
        # An unpacked copy with no record of where it came from is not a release.
        return Build(version, None, None, dirty=False, released=False)
    commit = _git(root, "rev-parse", "HEAD")
    if commit is None or not SHA.fullmatch(commit):
        return Build(version, None, None, dirty=False, released=False)
    branch = _git(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    # Tracked files only: logs, caches and local data never change what code runs.
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    dirty = status is None or bool(status)
    tags = (_git(root, "tag", "--points-at", "HEAD") or "").split()
    on_main = (
        _git(root, "merge-base", "--is-ancestor", commit, "refs/remotes/origin/main") is not None
    )
    released = not dirty and (f"v{version}" in tags or on_main)
    return Build(version, commit, branch, dirty=dirty, released=released)


@cache
def current() -> Build:
    """The running process's build. A checkout that moves later cannot relabel it."""
    return identify()


def same_code(first: str, second: str) -> bool:
    """Equal labels are the only evidence that two installations match."""
    return bool(first) and first == second and not first.endswith(".dirty")
