"""Fail when a Markdown file links to a missing file or heading.

Checks relative links and #anchors (GitHub heading slugs) in every tracked Markdown
file. External links are not fetched. Standard library only, so CI can run it first.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {".git", ".venv", "node_modules", "out", ".next", ".claude", "vendor", "third_party"}
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE = re.compile(r"^(```|~~~).*?^\1", re.M | re.S)


def markdown_files(root=ROOT):
    for path in sorted(root.rglob("*.md")):
        if not SKIP_DIRS.intersection(path.relative_to(root).parts):
            yield path


def heading_anchors(path):
    """GitHub anchors: lowercase, drop punctuation except - and _, spaces to hyphens."""
    anchors, seen = set(), {}
    text = FENCE.sub("", path.read_text(encoding="utf-8"))
    for line in text.splitlines():
        match = re.match(r"#{1,6}\s+(.*?)\s*#*\s*$", line)
        if not match:
            continue
        slug = re.sub(r"[^\w\- ]", "", match.group(1).lower()).replace(" ", "-")
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(slug if count == 0 else f"{slug}-{count}")
    for name in re.findall(r"<a\s+(?:name|id)=\"([^\"]+)\"", text):
        anchors.add(name)
    return anchors


def broken_links(path, root=ROOT):
    """Yield (target, reason) for each relative link in path that does not resolve."""
    text = FENCE.sub("", path.read_text(encoding="utf-8"))
    text = re.sub(r"`[^`\n]*`", "", text)
    for target in LINK.findall(text):
        if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("//"):
            continue
        file_part, _, anchor = target.partition("#")
        dest = (path.parent / file_part).resolve() if file_part else path
        if file_part.startswith("/"):
            dest = (root / file_part.lstrip("/")).resolve()
        if not dest.exists():
            yield target, "missing file"
        elif anchor and dest.suffix == ".md" and anchor not in heading_anchors(dest):
            yield target, "missing heading"


def main():
    problems = [
        f"{path.relative_to(ROOT)}: {target} ({reason})"
        for path in markdown_files()
        for target, reason in broken_links(path)
    ]
    for line in problems:
        print(line)
    print(f"{len(problems)} broken documentation links")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
