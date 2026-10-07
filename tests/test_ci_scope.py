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


PR_LANE = {"application", "desktop_lane"}
WEB_BOTH = {"desktop_lane", "mobile_lane"}


@pytest.mark.parametrize(
    "paths,expected",
    [
        (["README.md", "docs/terminal.md"], set()),
        ([".github/CODEOWNERS", ".github/ISSUE_TEMPLATE/bug.md"], set()),
        (["docs/workspace-guide.md"], {"desktop_lane"}),
        # Unknown paths fail open to the pull-request lane, never to everything.
        (["new-component/code.py"], PR_LANE),
        (["workers/new-worker/code.py"], PR_LANE),
        (["apps/web/src/app/page.tsx"], WEB_BOTH),
        (["tests/web/workflow.spec.ts"], WEB_BOTH),
        (["playwright.config.ts"], WEB_BOTH),
        (["package.json", "pnpm-lock.yaml"], {"desktop_lane"}),
        (["packages/core/src/vla_platform/tui.py", "uv.lock"], PR_LANE),
        (["scripts/export_openapi.py"], PR_LANE),
        (["tests/test_tui.py"], {"application"}),
        (["tests/fixtures/browser_worker/worker.py"], PR_LANE),
        # Worker-only pull requests skip the Python adapter suite.
        (["workers/act_optimizer/src/firebird_act/application.py"], {"act"}),
        (["workers/vla_cpp/src/quantize.py"], {"quantization", "benchmark"}),
        (["workers/firebird_quant/src/firebird_quant/codec.py"], {"unified_quantization"}),
        (
            ["workers/smolvla_qlora/src/firebird_vla/local_dataset.py"],
            {"training", "act", "teaching"},
        ),
        (["workers/decision/src/firebird_decision/contracts.py"], {"decision"}),
        (["workers/teaching/requirements-voice.txt"], {"teaching"}),
        (["workers/skypilot/launch.py"], {"application"}),
        (["workers/isaac_sim/run.py"], {"application", "teaching"}),
        (
            ["packages/core/src/vla_platform/lifecycle/contracts.py"],
            PR_LANE | set(ci_scope.WORKERS),
        ),
        (
            ["packages/core/src/vla_platform/datasets/snapshots.py"],
            PR_LANE | {"training", "act", "teaching"},
        ),
        ([".github/workflows/application.yml"], set(ci_scope.ALL)),
        ([".github/scripts/ci_scope.py"], set(ci_scope.ALL)),
    ],
)
def test_selects_changed_area_and_consumers(paths, expected):
    assert ci_scope.select(paths) == expected


def test_worker_adapters_run_after_merge_only():
    paths = ["workers/act_optimizer/src/firebird_act/application.py"]
    assert ci_scope.select(paths) == {"act"}
    assert ci_scope.select(paths, adapters=True) == {"application", "act"}


def test_mobile_always_implies_desktop():
    for path in ["apps/web/x.tsx", "tests/web/x.spec.ts", "playwright.config.ts"]:
        selected = ci_scope.select([path])
        assert "mobile_lane" in selected and "desktop_lane" in selected


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
    assert ci_scope.select(paths, adapters=True) == {
        "application",
        "quantization",
        "benchmark",
        "teaching",
    }


def run_scope(tmp_path, monkeypatch, event, document, ref="refs/heads/main", paths=None):
    run = len(list(tmp_path.glob("output-*")))
    event_path, output = tmp_path / "event.json", tmp_path / f"output-{run}"
    event_path.write_text(json.dumps(document))
    monkeypatch.setenv("GITHUB_EVENT_NAME", event)
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_REF", ref)
    for name in ("CI_PY_SHARDS", "CI_DESKTOP_SHARDS", "CI_MOBILE_SHARDS"):
        monkeypatch.delenv(name, raising=False)
    if paths is not None:
        monkeypatch.setattr(ci_scope, "changed_paths", lambda *_: paths)
    ci_scope.main()
    return dict(line.split("=", 1) for line in output.read_text().splitlines())


@pytest.mark.parametrize(
    "event,document",
    [
        ("workflow_dispatch", {}),
        ("push", {"before": "0" * 40, "after": "a" * 40}),
        ("push", {"before": "--help", "after": "a" * 40}),
        ("push", {"before": "a" * 40, "after": "b" * 40}),
        ("merge_group", {}),
    ],
)
def test_uncertain_main_history_runs_everything(tmp_path, monkeypatch, event, document):
    out = run_scope(tmp_path, monkeypatch, event, document)
    assert all(out[name] == "true" for name in (*ci_scope.SCOPES, *ci_scope.LANES))
    assert out["docs_only"] == "false"


def test_uncertain_pull_request_history_runs_pr_lane_not_everything(tmp_path, monkeypatch):
    out = run_scope(tmp_path, monkeypatch, "pull_request", {})
    assert out["application"] == "true" and out["desktop_lane"] == "true"
    assert out["mobile_lane"] == "false" and out["full"] == "false"
    assert all(out[name] == "false" for name in ci_scope.SCOPES[1:])


def test_output_names_are_preserved_and_extended(tmp_path, monkeypatch):
    out = run_scope(tmp_path, monkeypatch, "workflow_dispatch", {})
    assert list(out)[: len(ci_scope.SCOPES)] == list(ci_scope.SCOPES)
    assert set(out) >= {"desktop_lane", "mobile_lane", "web_lane", "docs_only", "full"}
    assert json.loads(out["py_shards"]) == ["1/2", "2/2"]


@pytest.mark.parametrize(
    "event,ref,paths,flags",
    [
        # Pull requests: docs -> nothing, worker -> that worker, web -> desktop (+mobile).
        ("pull_request", "refs/pull/1/merge", ["docs/a.md"], {"docs_only": "true"}),
        (
            "pull_request",
            "refs/pull/1/merge",
            ["workers/decision/src/x.py"],
            {"decision": "true", "application": "false", "web_lane": "false", "full": "false"},
        ),
        (
            "pull_request",
            "refs/pull/1/merge",
            ["apps/web/src/a.tsx"],
            {"application": "false", "desktop_lane": "true", "mobile_lane": "true"},
        ),
        (
            "pull_request",
            "refs/pull/1/merge",
            ["packages/core/src/vla_platform/api.py"],
            {"application": "true", "desktop_lane": "true", "mobile_lane": "false"},
        ),
        # Main is scoped by diff but always full (mobile + adapters).
        ("push", "refs/heads/main", ["docs/a.md"], {"docs_only": "true", "full": "true"}),
        (
            "push",
            "refs/heads/main",
            ["workers/decision/src/x.py"],
            {"decision": "true", "application": "true", "web_lane": "false", "full": "true"},
        ),
        (
            "push",
            "refs/heads/main",
            ["packages/core/src/vla_platform/api.py"],
            {"desktop_lane": "true", "mobile_lane": "true", "full": "true"},
        ),
        # Tags run everything.
        (
            "push",
            "refs/tags/v1.0.0",
            ["docs/a.md"],
            {"quantization": "true", "mobile_lane": "true"},
        ),
    ],
)
def test_lane_outputs(tmp_path, monkeypatch, event, ref, paths, flags):
    document = {"before": "a" * 40, "after": "b" * 40}
    if event == "pull_request":
        document = {"pull_request": {"base": {"sha": "a" * 40}, "head": {"sha": "b" * 40}}}
    out = run_scope(tmp_path, monkeypatch, event, document, ref=ref, paths=paths)
    for name, value in flags.items():
        assert out[name] == value, (name, out)
    assert out["web_lane"] == (
        "true" if "true" in (out["desktop_lane"], out["mobile_lane"]) else "false"
    )


def test_web_matrix_balances_projects_and_runs_extras_once(tmp_path, monkeypatch):
    pr = run_scope(tmp_path, monkeypatch, "pull_request", {}, paths=["packages/core/a.py"])
    legs = json.loads(pr["web_matrix"])
    slices = [leg for leg in legs if leg["kind"] == "shard"]
    assert [leg["kind"] for leg in legs].count("core") == 1
    assert {leg["project"] for leg in slices} == {"desktop"}
    assert [leg["shard"] for leg in slices] == ["1/4", "2/4", "3/4", "4/4"]
    full = json.loads(run_scope(tmp_path, monkeypatch, "workflow_dispatch", {})["web_matrix"])
    assert [leg["project"] for leg in full if leg["kind"] == "shard"].count("mobile") == 6
    docs = run_scope(tmp_path, monkeypatch, "pull_request", {}, paths=["README.md"])
    assert json.loads(docs["web_matrix"]) == []


def test_shard_counts_come_from_repository_variables(tmp_path, monkeypatch):
    out = run_scope(tmp_path, monkeypatch, "workflow_dispatch", {})
    assert len(json.loads(out["web_matrix"])) == 1 + 4 + 6
    monkeypatch.setenv("CI_DESKTOP_SHARDS", "2")
    monkeypatch.setenv("CI_MOBILE_SHARDS", "nonsense")
    monkeypatch.setenv("CI_PY_SHARDS", "99")
    assert len(ci_scope.web_matrix(set(ci_scope.ALL))) == 1 + 2 + 6
    assert ci_scope.shard_count("CI_PY_SHARDS") == 2


@pytest.mark.parametrize("count", [1, 2, 3, 5])
def test_python_shards_cover_every_test_file_exactly_once(count):
    spec = importlib.util.spec_from_file_location("ci_pyshard", SCRIPT.with_name("ci_pyshard.py"))
    shard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shard)
    files = shard.discover_tests()
    assert "tests/test_ci_scope.py" in files and files
    buckets = shard.partition(files, count)
    assert sorted(name for bucket in buckets for name in bucket) == files
    assert all(bucket for bucket in buckets)


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


@pytest.mark.parametrize("event", ["push", "pull_request", "workflow_dispatch"])
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
    expected = ["ubuntu-24.04"]
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
def test_linux_routing_has_a_repository_variable_kill_switch(name):
    import yaml

    document = yaml.safe_load((SCRIPT.parents[1] / f"workflows/{name}.yml").read_text())
    switch = {"application": "vars.CI_RUNNER == 'gcp'", "simulation": "vars.SIM_RUNNER"}[name]
    for job in document["jobs"].values():
        routing = job["runs-on"]
        assert "vars.GCP_RUNNERS_ENABLED" not in routing
        assert switch in routing
        assert "firebird-gcp" in routing
        assert "ubuntu-latest" not in routing
        if name == "application":
            assert "github.event.pull_request.head.repo.full_name == github.repository" in routing
    if name == "application":
        assert "ubuntu-24.04" in str(document["jobs"]["scope"]["runs-on"])
        assert "matrix.os == 'ubuntu-24.04'" in document["jobs"]["application"]["runs-on"]
        assert "ubuntu-latest" not in (SCRIPT.parents[1] / "workflows/application.yml").read_text()


def test_simulation_requires_dispatch():
    import yaml

    document = yaml.safe_load((SCRIPT.parents[1] / "workflows/simulation.yml").read_text())
    assert set(document[True]) == {"workflow_dispatch"}


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
    core_steps, web_steps = str(core["steps"]), str(web["steps"])
    assert "pytest -q -n auto" in core_steps and "ci_pyshard.py" in core_steps
    assert "pnpm test:web" not in core_steps
    assert core["strategy"]["matrix"]["shard"].startswith(
        "${{ fromJSON(needs.scope.outputs.py_shards"
    )
    assert web["strategy"]["matrix"]["os"] == core["strategy"]["matrix"]["os"]
    assert "needs.scope.outputs.web_matrix" in web["strategy"]["matrix"]["leg"]
    assert "--project=${{ matrix.leg.project }} --shard=${{ matrix.leg.shard }}" in web_steps
    # Contract/type/OpenAPI checks and diagnostics run once, on the core leg only.
    for command in (
        "scripts/export_openapi.py",
        "pnpm generate:client",
        "pnpm check:web",
        "pnpm test:diagnostics",
        "--project=pure --project=openapi --project=workflow",
    ):
        step = next(step for step in web["steps"] if command in step.get("run", ""))
        assert "matrix.leg.kind == 'core'" in step["if"]
    build = next(step for step in web["steps"] if step.get("run") == "pnpm build:web")
    assert "matrix.leg.kind" not in build["if"]  # Every leg serves the static export.
    assert set(jobs["application-verification"]["needs"]) == {"scope", "application", "web"}
    assert "web_lane" in str(jobs["application-verification"]["steps"])
    assert web["if"].count("web_lane") == 1


def test_caches_are_keyed_on_real_inputs():
    import yaml

    text = (SCRIPT.parents[1] / "workflows/application.yml").read_text()
    document = yaml.safe_load(text)
    assert "merge_group" not in text
    for job in document["jobs"].values():
        for step in job.get("steps", []):
            glob = str(step.get("with", {}).get("cache-dependency-glob", ""))
            assert ".github/workflows" not in glob
    steps = document["jobs"]["web"]["steps"]
    caches = [step for step in steps if str(step.get("uses", "")).startswith("actions/cache@")]
    keys = " ".join(step["with"]["key"] for step in caches)
    assert "hashFiles('pnpm-lock.yaml')" in keys and "ms-playwright" in str(caches)
    assert any("restore-keys" in step["with"] for step in caches)
    # xdist comes from uv.lock (pytest-xdist==3.8.0 in the dev dependency group).
    assert "-n auto --dist loadfile" in str(document["jobs"]["application"]["steps"])


@pytest.mark.parametrize("status", ["failure", "cancelled", "skipped"])
def test_web_failure_blocks_gate(status):
    text = (SCRIPT.parents[1] / "workflows/application.yml").read_text()
    gate = text.split("  application-verification:\n", 1)[1]
    block = gate.split("python3 - <<'PY_GATE'\n", 1)[1].split("          PY_GATE", 1)[0]
    script = "\n".join(line[10:] for line in block.splitlines())
    results = {
        "scope": {"result": "success", "outputs": {"application": "true", "web_lane": "true"}},
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
