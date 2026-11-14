"""Run the checks your change can break, and say why each one was chosen.

    uv run python scripts/check.py              # only what the current change affects
    uv run python scripts/check.py --list       # show the selection, run nothing
    uv run python scripts/check.py --all        # every application check (slow)
    uv run python scripts/check.py --record     # full Python run; refresh the impact map
    uv run python scripts/check.py --base REF   # compare with something other than origin/main

"Your change" is everything that differs from the merge base with ``origin/main``:
commits on this branch, staged and unstaged edits, and new untracked files.

How Python tests are chosen for a changed source file, most precise first:

1. tests that *executed* the file in the last ``--record`` run (the impact map);
2. tests that import the module by name;
3. tests that mention the file by name (it may run as a separate process).

A changed file nothing accounts for runs the whole Python suite: selection fails open,
never silently narrow. Browser specs are matched to changed web files by name, plus a
small smoke set for shared files. CI still runs the complete lanes on the pull request.

Nothing here reaches the network or the cloud; ``tests/conftest.py`` enforces that.
"""

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPACT = ROOT / ".cache/test-impact.json"
PYTHON_ROOTS = ("packages/core/src/", "scripts/")
# A change here can affect any Python test.
PYTHON_SHARED = ("tests/conftest.py", "tests/fixtures/", "pyproject.toml", "uv.lock")
WEB_ROOTS = ("apps/web/", "tests/web/")
# Used by every page: run the smoke set and leave the full suite to CI.
WEB_SHARED = (
    "apps/web/src/app/",
    "apps/web/src/lib/api.ts",
    "apps/web/src/lib/api.generated.ts",
    "apps/web/src/components/workspace-state.ts",
    "apps/web/src/components/workspace-context.tsx",
    "apps/web/src/components/workspace-ui.tsx",
    "apps/web/package.json",
    "pnpm-lock.yaml",
    "playwright.config.ts",
)
WEB_SMOKE = ("workspace-routes.spec.ts", "workspace-ux.spec.ts", "workflow.spec.ts")
RUFF_ROOTS = ("packages/core/", "scripts/", "tests/", ".github/scripts/")
PRETTIER_ROOTS = ("apps/web/src/", "tests/web/")


def git(*arguments):
    result = subprocess.run(
        ["git", "-C", str(ROOT), *arguments], capture_output=True, text=True, check=False
    )
    return result.stdout if result.returncode == 0 else ""


def changed_files(base):
    merge_base = git("merge-base", "HEAD", base).strip()
    if not merge_base:
        raise SystemExit(f"Cannot find {base}. Run: git fetch origin main")
    names = set(git("diff", "--name-only", "--no-renames", merge_base).split("\n"))
    names |= set(git("ls-files", "--others", "--exclude-standard").split("\n"))
    return sorted(name for name in names if name), merge_base


def module_name(path):
    """``packages/core/src/vla_platform/datasets/worker.py`` -> ``vla_platform.datasets.worker``."""
    if not path.startswith("packages/core/src/") or not path.endswith(".py"):
        return None
    parts = path.removeprefix("packages/core/src/").removesuffix(".py").split("/")
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def test_files():
    return sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / "tests").glob("test_*.py"))


def imported_modules(test):
    """Every dotted name a test file imports, including ``from package import module``."""
    try:
        tree = ast.parse((ROOT / test).read_text())
    except OSError, SyntaxError:
        return set()
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.add(node.module)
            found |= {f"{node.module}.{alias.name}" for alias in node.names}
    return found


def load_impact():
    try:
        document = json.loads(IMPACT.read_text())
        return document["tests"], document.get("recorded_at", "unknown time")
    except OSError, ValueError, KeyError:
        return {}, None


def select_python(changed):
    """Return ``{test file: reason}``, or ``None`` for the whole suite, plus the notes."""
    tests = test_files()
    notes = []
    for name in changed:
        if name.startswith(PYTHON_SHARED) or name in PYTHON_SHARED:
            return None, [f"{name} is shared by every Python test"]
    selected = {name: "the test itself changed" for name in changed if name in tests}
    sources = [name for name in changed if name.startswith(PYTHON_ROOTS) and name.endswith(".py")]
    sources += [name for name in changed if name.startswith("workers/") and name.endswith(".py")]
    if not sources:
        return selected, notes
    impact, recorded = load_impact()
    executed_by = {}
    for test, files in impact.items():
        for file in files:
            executed_by.setdefault(file, set()).add(test)
    imports = {test: imported_modules(test) for test in tests}
    text = {test: (ROOT / test).read_text(errors="replace") for test in tests}
    core = {
        path.relative_to(ROOT).as_posix(): path.read_text(errors="replace")
        for path in (ROOT / "packages/core/src").rglob("*.py")
    }
    for source in sources:
        found = {}
        for test in executed_by.get(source, ()):
            if test in tests:
                found[test] = f"executed {source} in the recorded run"
        # Code that runs as its own process is invisible to the recording. Whoever
        # names the file or module starts it, so their tests depend on it too.
        named = [f'"{Path(source).name}"', *([module_name(source)] if module_name(source) else [])]
        for launcher, body in core.items():
            if launcher != source and any(name in body for name in named):
                for test in executed_by.get(launcher, ()):
                    if test in tests:
                        found.setdefault(test, f"runs it through {Path(launcher).name}")
        module = module_name(source)
        for test in tests:
            if module and module in imports[test]:
                found.setdefault(test, f"imports {module}")
            elif Path(source).name in text[test] or (
                source.startswith("workers/") and source.split("/")[1] in text[test]
            ):
                found.setdefault(test, f"mentions {Path(source).name}")
        if not found:
            return None, [f"nothing accounts for {source}; running every Python test"]
        for test, reason in found.items():
            selected.setdefault(test, reason)
    if not impact:
        notes.append(
            "No impact map yet: selection used imports and names only. "
            "Run scripts/check.py --record once for a precise selection."
        )
    else:
        notes.append(f"Impact map recorded {recorded}.")
    return selected, notes


def spec_stem(name):
    return Path(name).name.removesuffix(".spec.ts").removesuffix("-pure")


def tokens(stem):
    return [part.removesuffix("s") for part in stem.split("-") if part]


def select_web(changed):
    """Return ``{spec: reason}`` for the changed web files."""
    specs = sorted(path.name for path in (ROOT / "tests/web").glob("*.spec.ts"))
    selected = {}
    for name in changed:
        if name.startswith("tests/web/") and name.endswith(".spec.ts") and Path(name).name in specs:
            selected[Path(name).name] = "the spec itself changed"
    sources = [
        name
        for name in changed
        if name.startswith("apps/web/src/") or name in WEB_SHARED or name.startswith("tests/web/")
    ]
    sources = [name for name in sources if not name.endswith(".spec.ts")]
    smoke = False
    for source in sources:
        if source.startswith(WEB_SHARED) or source in WEB_SHARED:
            smoke = True
            continue
        wanted = tokens(Path(source).name.split(".")[0])
        best, matches = 0, []
        for spec in specs:
            have = tokens(spec_stem(spec))
            common = 0
            for left, right in zip(wanted, have, strict=False):
                if left != right:
                    break
                common += 1
            if common > best:
                best, matches = common, [spec]
            elif common == best and common:
                matches.append(spec)
        if not matches:
            smoke = True
        for spec in matches:
            selected.setdefault(spec, f"named after {Path(source).name}")
    if smoke:
        for spec in WEB_SMOKE:
            if spec in specs:
                selected.setdefault(spec, "smoke set for a shared web file")
    return selected


class Runner:
    def __init__(self, dry):
        self.dry, self.results = dry, []

    def run(self, title, command, *, env=None):
        printable = " ".join(str(part) for part in command)
        if self.dry:
            print(f"  would run  {title}: {printable[:160]}")
            return True
        print(f"\n== {title}\n   {printable[:200]}", flush=True)
        started = time.monotonic()
        code = subprocess.run(command, cwd=ROOT, env=env).returncode
        self.results.append((title, code == 0, time.monotonic() - started))
        return code == 0

    def summary(self):
        if self.dry:
            return 0
        print("\n== Summary")
        for title, passed, seconds in self.results:
            print(f"   {'ok  ' if passed else 'FAIL'}  {seconds:6.1f}s  {title}")
        failed = [title for title, passed, _ in self.results if not passed]
        total = sum(seconds for _, _, seconds in self.results)
        print(f"   {total:.0f}s total" + (f"; failed: {', '.join(failed)}" if failed else ""))
        return 1 if failed else 0


def merge_impact(directory):
    tests = {}
    for part in sorted(Path(directory).glob("*.json")):
        for test, files in json.loads(part.read_text()).items():
            tests.setdefault(test, set()).update(files)
    if not tests:
        return False
    IMPACT.parent.mkdir(parents=True, exist_ok=True)
    IMPACT.write_text(
        json.dumps(
            {
                "recorded_at": time.strftime("%Y-%m-%d %H:%M"),
                "commit": git("rev-parse", "HEAD").strip(),
                "tests": {test: sorted(files) for test, files in sorted(tests.items())},
            }
        )
    )
    shutil.rmtree(directory, ignore_errors=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--all", action="store_true", help="run every application check")
    parser.add_argument("--list", action="store_true", help="show the selection only")
    parser.add_argument("--record", action="store_true", help="full run; refresh the impact map")
    parser.add_argument("--no-web", action="store_true", help="skip the browser lane")
    args = parser.parse_args()

    python = sys.executable
    changed, merge_base = changed_files(args.base)
    runner = Runner(args.list)
    print(f"Comparing with {args.base} ({merge_base[:9]}): {len(changed)} changed file(s).")
    if not changed and not (args.all or args.record):
        print("Nothing changed; nothing to check.")
        return 0

    # --- Python ---------------------------------------------------------------------
    everything = args.all or args.record
    selected, notes = (None, ["--all"]) if everything else select_python(changed)
    pytest = [python, "-m", "pytest", "-q", "-n", "auto", "--dist", "loadfile"]
    ruff_targets = [
        name
        for name in changed
        if name.endswith(".py") and name.startswith(RUFF_ROOTS) and (ROOT / name).exists()
    ]
    if everything:
        ruff_targets = ["packages/core", "scripts", "tests", ".github/scripts"]
    if ruff_targets:
        runner.run("ruff check", [python, "-m", "ruff", "check", *ruff_targets])
        runner.run("ruff format", [python, "-m", "ruff", "format", "--check", *ruff_targets])
    if any(name in changed for name in ("pyproject.toml", "packages/core/pyproject.toml")) or (
        everything
    ):
        runner.run("versions agree", [python, ".github/scripts/version_files.py", "check"])
    for note in notes:
        print(f"  note: {note}")
    if selected is None:
        environment = dict(os.environ)
        staging = ROOT / ".cache/test-impact.d"
        if args.record:
            shutil.rmtree(staging, ignore_errors=True)
            environment["JENSEN_TEST_IMPACT"] = str(staging)
        print(f"  Python: the whole suite ({len(test_files())} files).")
        passed = runner.run("pytest (all)", [*pytest, "tests"], env=environment)
        if args.record and not args.list:
            written = merge_impact(staging)
            where = f"written to {IMPACT.relative_to(ROOT)}" if written else "not written"
            caveat = "" if passed else " (some tests failed; it still reflects what ran)"
            print(f"  Impact map {where}{caveat}")
    elif selected:
        print(f"  Python: {len(selected)} of {len(test_files())} test files.")
        for test, reason in sorted(selected.items()):
            print(f"    {test}  <- {reason}")
        runner.run(f"pytest ({len(selected)} files)", [*pytest, *sorted(selected)])
    else:
        print("  Python: no test is affected.")

    # --- Web ------------------------------------------------------------------------
    web_changed = [name for name in changed if name.startswith(WEB_ROOTS) or name in WEB_SHARED]
    contract = any(name.startswith("packages/core/src/") for name in changed)
    pnpm = shutil.which("pnpm")
    if args.no_web or not pnpm:
        if (web_changed or contract) and not pnpm:
            print("  Web: pnpm is not installed; skipped.")
    elif web_changed or contract or args.all:
        if contract or args.all:
            # The API schema and the typed web client are generated from the Python
            # contracts. Regenerate them here so they can never drift from the code.
            generated = ("packages/core/openapi.json", "apps/web/src/lib/api.generated.ts")
            before = [(ROOT / name).read_bytes() for name in generated]
            runner.run("export OpenAPI", [python, "scripts/export_openapi.py"])
            runner.run("generate client", [pnpm, "generate:client"])
            if not args.list and before != [(ROOT / name).read_bytes() for name in generated]:
                print("  note: the API schema or typed client was regenerated; include both files.")
                web_changed = web_changed or list(generated)
        if web_changed or args.all:
            runner.run("typecheck", [pnpm, "check:web"])
            prettier = [
                name
                for name in changed
                if name.startswith(PRETTIER_ROOTS)
                and name.endswith((".ts", ".tsx", ".css", ".json"))
                and not name.endswith("api.generated.ts")
                and (ROOT / name).exists()
            ]
            if args.all:
                runner.run("prettier", [pnpm, "format:check"])
                runner.run("eslint", [pnpm, "lint:web"])
            else:
                if prettier:
                    runner.run("prettier", [pnpm, "exec", "prettier", "--check", *prettier])
                lint = [
                    name
                    for name in prettier
                    if name.startswith("apps/web/src/") and name.endswith((".ts", ".tsx"))
                ]
                if lint:
                    runner.run("eslint", [pnpm, "exec", "eslint", *lint])
            specs = (
                {path.name: "--all" for path in (ROOT / "tests/web").glob("*.spec.ts")}
                if args.all
                else select_web(changed)
            )
            if specs:
                total = len(list((ROOT / "tests/web").glob("*.spec.ts")))
                print(f"  Browser: {len(specs)} of {total} specs (desktop only; CI adds mobile).")
                for spec, reason in sorted(specs.items()):
                    print(f"    {spec}  <- {reason}")
                runner.run("build web", [pnpm, "build:web"])
                pure = sorted(
                    f"tests/web/{name}" for name in specs if name.endswith("-pure.spec.ts")
                )
                rest = sorted(
                    f"tests/web/{name}"
                    for name in specs
                    if not name.endswith("-pure.spec.ts")
                    and name not in ("workflow.spec.ts", "openapi.spec.ts")
                )
                if pure:
                    runner.run("browser (pure)", [pnpm, "test:web", "--project=pure", *pure])
                if "workflow.spec.ts" in specs:
                    runner.run("browser (workflow)", [pnpm, "test:web", "--project=workflow"])
                if rest:
                    runner.run("browser (desktop)", [pnpm, "test:web", "--project=desktop", *rest])
            else:
                print("  Browser: no spec is affected.")

    # --- Documentation --------------------------------------------------------------
    if any(name.endswith(".md") for name in changed) or args.all:
        runner.run("documentation links", [python, ".github/scripts/check_doc_links.py"])
    return runner.summary()


if __name__ == "__main__":
    sys.exit(main())
