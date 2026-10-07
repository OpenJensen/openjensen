"""Print the pytest files for one CI shard (``--shard 1/3``), balanced by file size.

Every test file is assigned to exactly one shard by greedy largest-first packing, so the
union over all shards is the complete suite. File size is a stable, dependency-free proxy
for cost; pytest-xdist balances tests inside each shard.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATTERNS = ("test_*.py", "*_test.py")


def discover_tests(root=ROOT):
    found = {path for pattern in PATTERNS for path in (root / "tests").rglob(pattern)}
    return sorted(path.relative_to(root).as_posix() for path in found)


def partition(files, count, root=ROOT):
    """Assign each file to the currently lightest bucket, largest files first."""
    weighted = sorted((-(root / name).stat().st_size, name) for name in files)
    buckets = [(0, index, []) for index in range(count)]
    for negative_size, name in weighted:
        weight, index, members = min(buckets)
        members.append(name)
        buckets[index] = (weight - negative_size, index, members)
    return [sorted(members) for _, _, members in buckets]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", required=True, help="CURRENT/TOTAL, for example 2/3")
    args = parser.parse_args(argv)
    current, _, total = args.shard.partition("/")
    if not (current.isdigit() and total.isdigit() and 1 <= int(current) <= int(total)):
        raise SystemExit(f"Invalid shard {args.shard!r}; expected CURRENT/TOTAL")
    selected = partition(discover_tests(), int(total))[int(current) - 1]
    if not selected:
        raise SystemExit(f"Shard {args.shard} has no test files")
    sys.stdout.buffer.write(("\n".join(selected) + "\n").encode())


if __name__ == "__main__":
    main()
