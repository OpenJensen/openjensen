"""Documentation links resolve to files and headings in the repository."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/check_doc_links.py"
spec = importlib.util.spec_from_file_location("check_doc_links", SCRIPT)
check_doc_links = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_doc_links)


def test_repository_documentation_links_resolve():
    problems = [
        (path.relative_to(check_doc_links.ROOT).as_posix(), target, reason)
        for path in check_doc_links.markdown_files()
        for target, reason in check_doc_links.broken_links(path)
    ]
    assert problems == []


def test_broken_files_and_headings_are_reported(tmp_path):
    (tmp_path / "guide.md").write_text("# Set up\n\n## 2. Open a dataset\n\n## Setup\n\n## Setup\n")
    page = tmp_path / "page.md"
    page.write_text(
        "[ok](guide.md#2-open-a-dataset) [dup](guide.md#setup-1) [web](https://example.com)\n"
        "[gone](missing.md) [stale](guide.md#old-heading) `[code](ignored.md)`\n"
        "```\n[fenced](ignored.md)\n```\n"
    )
    assert sorted(check_doc_links.broken_links(page, tmp_path)) == [
        ("guide.md#old-heading", "missing heading"),
        ("missing.md", "missing file"),
    ]
