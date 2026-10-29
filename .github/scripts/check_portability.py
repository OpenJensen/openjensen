"""Check shipped text for developer-home paths and private default endpoints.

Only tracked first-party source, tests, configuration and documentation are
inspected. Attribution and third-party licenses are deliberately outside the
check. Diagnostics report locations and categories without echoing content.
"""

import ipaddress
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP_PARTS = {"vendor", "third_party", "third-party", "node_modules", ".venv", "out", ".next"}
GENERIC_USERS = {
    "user",
    "username",
    "example",
    "operator",
    "runner",
    "ubuntu",
    "your-user",
    "private",
}
HOME = re.compile(
    r"(?:/Users/|/home/)([A-Za-z0-9_.-]+)|[A-Za-z]:[\\/]+Users[\\/]+([A-Za-z0-9_.-]+)"
)
ENDPOINT = re.compile(
    r"\b(?:https?://|ssh\s+(?:[A-Za-z0-9_.-]+@)?)(\d{1,3}(?:\.\d{1,3}){3})(?=[:/\s'\"]|$)"
)


def first_party(relative):
    path = Path(relative)
    if SKIP_PARTS.intersection(path.parts):
        return False
    name = path.name.upper()
    if "LICENSE" in name or name in {"NOTICE", "COPYING", "AUTHORS", "CONTRIBUTORS"}:
        return False
    return True


def violations(text):
    for number, line in enumerate(text.splitlines(), 1):
        for match in HOME.finditer(line):
            username = next(value for value in match.groups() if value is not None)
            if username.lower() not in GENERIC_USERS:
                yield number, "developer-home-path"
        for match in ENDPOINT.finditer(line):
            try:
                address = ipaddress.ip_address(match.group(1))
            except ValueError:
                continue
            # Test networks are public documentation space, not personal infrastructure.
            if (
                address.is_private
                and not address.is_loopback
                and not any(
                    address in ipaddress.ip_network(network)
                    for network in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")
                )
            ):
                yield number, "private-default-endpoint"


def tracked_files(root=ROOT):
    result = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True)
    return [name for name in result.stdout.decode().split("\0") if name]


def check(root=ROOT, files=None):
    found = []
    for relative in tracked_files(root) if files is None else files:
        if not first_party(relative):
            continue
        path = root / relative
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 2 * 1024**2:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeError, OSError):
            continue
        if "\0" in text:
            continue
        synthetic_tests = "tests" in Path(relative).parts or Path(relative).name.startswith("test_")
        found.extend(
            (relative, number, category)
            for number, category in violations(text)
            # Synthetic private-network topologies are valid protocol test fixtures.
            if not (synthetic_tests and category == "private-default-endpoint")
        )
    return found


def main():
    found = check()
    for relative, number, category in found:
        print(f"{relative}:{number}: {category}")
    if found:
        print("Use user-configured destinations and generic documentation examples.")
        return 1
    print("Portability checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
