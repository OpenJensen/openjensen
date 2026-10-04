"""CI must skip unrelated expensive environments while retaining cross-boundary checks."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/ci_scope.py"
spec = importlib.util.spec_from_file_location("ci_scope", SCRIPT)
ci_scope = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci_scope)


@pytest.mark.parametrize(
    "paths,expected",
    [
        (["README.md", "docs/terminal.md"], set()),
        (["docs/workspace-guide.md"], {"application"}),
        (["new-component/code.py"], set(ci_scope.SCOPES)),
        (["workers/new-worker/code.py"], set(ci_scope.SCOPES)),
        (["apps/web/src/app/page.tsx"], {"application"}),
        (["packages/core/src/vla_platform/tui.py", "uv.lock"], {"application"}),
        (["workers/act_optimizer/src/firebird_act/application.py"], {"application", "act"}),
        (["workers/vla_cpp/src/quantize.py"], {"application", "quantization", "benchmark"}),
        (
            ["workers/firebird_quant/src/firebird_quant/codec.py"],
            {"application", "unified_quantization"},
        ),
        (
            ["workers/smolvla_qlora/src/firebird_vla/local_dataset.py"],
            {"application", "training", "act", "teaching"},
        ),
        (["workers/decision/src/firebird_decision/contracts.py"], {"application", "decision"}),
        (["workers/teaching/requirements-voice.txt"], {"application", "teaching"}),
        (["packages/core/src/vla_platform/lifecycle/contracts.py"], set(ci_scope.SCOPES)),
        (
            ["packages/core/src/vla_platform/datasets/snapshots.py"],
            {"application", "training", "act", "teaching"},
        ),
        ([".github/workflows/application.yml"], set(ci_scope.SCOPES)),
    ],
)
def test_selects_changed_area_and_consumers(paths, expected):
    assert ci_scope.select(paths) == expected


def test_rename_runs_both_old_and_new_scopes(tmp_path, monkeypatch):
    # The real Git diff is tested: -M rename output would otherwise hide the old consumer.
    monkeypatch.chdir(tmp_path)

    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()

    git("init", "-q")
    git("config", "user.name", "CI fixture")
    git("config", "user.email", "fixture@example.test")
    source = tmp_path / "workers/vla_cpp/file with\nnewline.py"
    source.parent.mkdir(parents=True)
    source.write_text("fixture")
    git("add", ".")
    git("commit", "-qm", "before")
    before = git("rev-parse", "HEAD")
    target = tmp_path / "workers/teaching/file.py"
    target.parent.mkdir(parents=True)
    source.rename(target)
    git("add", "-A")
    git("commit", "-qm", "after")
    paths = ci_scope.changed_paths("push", {"before": before, "after": git("rev-parse", "HEAD")})
    assert len(paths) == 2
    assert ci_scope.select(paths) == {"application", "quantization", "benchmark", "teaching"}


@pytest.mark.parametrize(
    "event,document",
    [
        ("workflow_dispatch", {}),
        ("push", {"before": "0" * 40, "after": "a" * 40}),
        ("push", {"before": "--help", "after": "a" * 40}),
        ("pull_request", {}),
        ("push", {"before": "a" * 40, "after": "b" * 40}),
    ],
)
def test_uncertain_history_never_skips_checks(tmp_path, monkeypatch, event, document):
    event_path, output = tmp_path / "event.json", tmp_path / "output"
    event_path.write_text(json.dumps(document))
    monkeypatch.setenv("GITHUB_EVENT_NAME", event)
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    ci_scope.main()
    assert output.read_text().splitlines() == [f"{name}=true" for name in ci_scope.SCOPES]


@pytest.mark.parametrize("gate", ["application-verification", "native-worker-verification"])
@pytest.mark.parametrize(
    "case",
    ["pass", "skip", "failure", "cancelled", "unexpected_skip", "missing_output", "scope_failure"],
)
def test_stable_gate_does_not_hide_failed_or_missing_checks(gate, case):
    # Execute the exact inline gate shipped to Actions, not a test reimplementation.
    import ast
    import re

    text = (SCRIPT.parents[1] / "workflows/application.yml").read_text()
    text = text.split(f"  {gate}:\n", 1)[1]
    block = text.split("python3 - <<'PY_GATE'\n", 1)[1].split("          PY_GATE", 1)[0]
    script = "\n".join(line[10:] for line in block.splitlines())
    mapping = ast.literal_eval(re.search(r"required = (.*)", script)[1])
    results = {
        "scope": {"result": "success", "outputs": {scope: "true" for scope in mapping.values()}}
    }
    results.update({job: {"result": "success"} for job in mapping})
    job, scope = next(iter(mapping.items()))
    if case == "skip":
        results["scope"]["outputs"][scope] = "false"
        results[job]["result"] = "skipped"
    elif case in {"failure", "cancelled"}:
        results[job]["result"] = case
    elif case in {"unexpected_skip", "missing_output"}:
        results[job]["result"] = "skipped"
        if case == "missing_output":
            del results["scope"]["outputs"][scope]
    elif case == "scope_failure":
        results["scope"]["result"] = "failure"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=os.environ | {"CHECK_RESULTS": json.dumps(results)},
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert (completed.returncode == 0) == (case in {"pass", "skip"}), completed.stderr


@pytest.mark.parametrize("event", ["push", "pull_request", "merge_group", "workflow_dispatch"])
@pytest.mark.parametrize("windows", [False, True])
def test_windows_matrix_requires_explicit_manual_opt_in(event, windows):
    # Evaluate the exact workflow expression's limited boolean/string subset.
    # This checks all automatic events even if an input is accidentally supplied.
    import ast
    import re

    text = (SCRIPT.parents[1] / "workflows/application.yml").read_text()
    expression = re.search(r"os: \$\{\{ fromJSON\((.+)\) \}\}", text)[1]
    expression = (
        expression.replace("github.event_name", "event")
        .replace("inputs.windows_browser", "windows")
        .replace("&&", "and")
        .replace("||", "or")
    )
    parsed = ast.parse(expression, mode="eval")
    allowed = (
        ast.Expression,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.Compare,
        ast.Eq,
        ast.Name,
        ast.Load,
        ast.Constant,
    )
    assert all(isinstance(node, allowed) for node in ast.walk(parsed))
    assert {node.id for node in ast.walk(parsed) if isinstance(node, ast.Name)} == {
        "event",
        "windows",
    }
    matrix = json.loads(
        eval(
            compile(parsed, "<workflow-matrix>", "eval"),
            {"__builtins__": {}},
            {"event": event, "windows": windows},
        )
    )
    expected = ["ubuntu-latest"]
    if event == "workflow_dispatch" and windows:
        expected.append("windows-latest")
    assert matrix == expected
    assert "windows_browser:" in text and "default: false" in text
    for command in (
        "pytest -q",
        "scripts/export_openapi.py",
        "pnpm generate:client",
        "pnpm check:web",
        "pnpm build:web",
        "pnpm test:web",
        "pnpm test:diagnostics",
    ):
        assert command in text


def test_unified_quantization_keeps_two_explicit_compatible_dependency_pairs():
    import re

    workflow = (SCRIPT.parents[1] / "workflows/application.yml").read_text()
    job = workflow.split("  unified-quantization:\n", 1)[1].split("  training:\n", 1)[0]
    pairs = re.findall(
        r"^          - torch: '([^']+)'\n            numpy: '([^']+)'$", job, re.MULTILINE
    )
    assert pairs == [("2.2.2", "1.26.4"), ("2.11.0", "2.2.6")]
    assert "matrix:\n        include:" in job
    assert "numpy==${{ matrix.numpy }}" in job
    assert "uv pip check --python .venv/bin/python" in job


@pytest.mark.parametrize("name", ["application", "simulation"])
def test_linux_routing_defaults_to_gcp(name):
    import yaml

    document = yaml.safe_load((SCRIPT.parents[1] / f"workflows/{name}.yml").read_text())
    for job in document["jobs"].values():
        routing = job["runs-on"]
        assert "vars.GCP_RUNNERS_ENABLED" not in routing
        assert "github.event.pull_request.head.repo.full_name == github.repository" in routing
        assert "firebird-gcp" in routing
        assert "ubuntu-latest" in routing
    if name == "application":
        assert "matrix.os == 'ubuntu-latest'" in document["jobs"]["application"]["runs-on"]


def test_simulation_requires_dispatch():
    import yaml

    document = yaml.safe_load((SCRIPT.parents[1] / "workflows/simulation.yml").read_text())
    assert set(document[True]) == {"workflow_dispatch"}


def test_push_runs_every_scope(tmp_path, monkeypatch):
    output = tmp_path / "output"
    event = tmp_path / "event.json"
    event.write_text("{}")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(ci_scope, "changed_paths", lambda *_: ["README.md"])

    ci_scope.main()

    assert output.read_text().splitlines() == [f"{name}=true" for name in ci_scope.SCOPES]


def test_pnpm_setup_does_not_cache():
    import yaml

    document = yaml.safe_load((SCRIPT.parents[1] / "workflows/application.yml").read_text())
    steps = document["jobs"]["web"]["steps"]
    setup = next(step for step in steps if step.get("uses") == "pnpm/action-setup@v4")
    assert setup["with"]["cache"] is False


def test_python_and_browser_budgets():
    import yaml

    jobs = yaml.safe_load((SCRIPT.parents[1] / "workflows/application.yml").read_text())["jobs"]
    core = jobs["application"]
    web = jobs["web"]
    assert "pytest -q" in str(core["steps"])
    assert "pnpm test:web" not in str(core["steps"])
    assert web["strategy"]["matrix"]["shard"] == ["1/2", "2/2"]
    assert web["strategy"]["matrix"]["os"] == core["strategy"]["matrix"]["os"]
    assert "--shard=${{ matrix.shard }}" in str(web["steps"])
    diagnostics = next(step for step in web["steps"] if step.get("run") == "pnpm test:diagnostics")
    assert diagnostics["if"] == "matrix.shard == '1/2'"
    assert set(jobs["application-verification"]["needs"]) == {"scope", "application", "web"}


@pytest.mark.parametrize("status", ["failure", "cancelled", "skipped"])
def test_web_failure_blocks_gate(status):
    text = (SCRIPT.parents[1] / "workflows/application.yml").read_text()
    gate = text.split("  application-verification:\n", 1)[1]
    block = gate.split("python3 - <<'PY_GATE'\n", 1)[1].split("          PY_GATE", 1)[0]
    script = "\n".join(line[10:] for line in block.splitlines())
    results = {
        "scope": {"result": "success", "outputs": {"application": "true"}},
        "application": {"result": "success"},
        "web": {"result": status},
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=os.environ | {"CHECK_RESULTS": json.dumps(results)},
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0
    assert f"web: {status}" in result.stderr
