"""Patch preparation and source publication preserve reviewed, verified commit identity."""

import base64
import copy
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts"


def script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


versions = script("version_files")
# The trusted runner imports its sibling; restore the import table after loading.
previous = sys.modules.get("version_files")
sys.modules["version_files"] = versions
try:
    release = script("patch_release")
finally:
    if previous is None:
        del sys.modules["version_files"]
    else:
        sys.modules["version_files"] = previous


@pytest.mark.parametrize(
    "before,after", [("0.1.1", "0.1.2"), ("0.1.9", "0.1.10"), ("0.1.999", "0.1.1000")]
)
def test_patch_never_rolls_into_minor(before, after):
    assert versions.next_patch(before) == after


@pytest.mark.parametrize("value", ["v0.1.1", "0.01.1", "0.1", "0.1.1rc1", "1.0.0\n"])
def test_only_stable_unambiguous_versions(value):
    with pytest.raises(ValueError):
        versions.parse(value)


def test_real_manifests_are_consistent_and_rewrite_is_idempotent():
    original = versions.load(ROOT)
    before = versions.current(original)
    after = versions.next_patch(before)
    changed = versions.rewrite(original, after)
    assert versions.validate(changed, before) == after
    assert len(versions.versions(changed)) == 11
    assert versions.rewrite(changed, after) == changed
    # Third-party lock entries and every other dependency stay byte-for-byte identical.
    restored = versions.rewrite(changed, versions.current(original))
    assert restored == original
    mismatched = dict(changed)
    mismatched["apps/web/package.json"] = original["apps/web/package.json"]
    with pytest.raises(ValueError, match="disagree"):
        versions.validate(mismatched)
    with pytest.raises(ValueError, match="Expected patch"):
        versions.validate(changed, after)
    major, minor, _ = versions.parse(before)
    manual_version = f"{major}.{minor + 1}.0"
    manual = versions.rewrite(original, manual_version)
    assert versions.validate(manual, before, manual=True) == manual_version
    with pytest.raises(ValueError):
        versions.validate(manual, before)


def pr():
    return {
        "number": 7,
        "draft": False,
        "labels": [],
        "head": {"sha": "head", "ref": "feature", "repo": {"full_name": "org/repo"}},
        "base": {"ref": "main", "sha": "base"},
        "merged_at": "date",
        "merge_commit_sha": "merged",
    }


class FakeGitHub:
    repository = "org/repo"

    def __init__(self):
        self.pr = pr()
        self.base = "base"
        self.behind = 0
        self.values = versions.rewrite(versions.load(ROOT), "0.1.1")
        self.base_values = dict(self.values)
        self.calls = []
        self.dispatched = []
        self.writes = []
        self.tag = None
        self.published = None
        self.merged = True
        self.checks = []
        self.run = {
            "conclusion": "success",
            "event": "push",
            "head_branch": "main",
            "head_repository": {"full_name": self.repository},
            "path": ".github/workflows/application.yml",
            "head_sha": "merged",
        }

    def main(self):
        return self.base

    def pages(self, path):
        assert path == "/pulls?state=open&base=main"
        return [copy.deepcopy(self.pr)]

    def contents(self, sha):
        return {"tree": {"sha": "tree"}}, dict(
            self.base_values if sha == self.base else self.values
        )

    def write_versions(self, pull, commit, before, after):
        if before == after:
            return pull["head"]["sha"]
        self.writes.append(after)
        self.values = after
        return "versioned"

    def dispatch_ci(self, pull, sha):
        self.dispatched.append(sha)

    def request(self, method, path, data=None, **kwargs):
        self.calls.append((method, path, data))
        if method == "GET" and path.startswith("/compare/"):
            return {"behind_by": self.behind}
        if method == "PUT" and path.endswith("/update-branch"):
            assert data == {"expected_head_sha": "head"}
            self.pr["head"]["sha"] = "refreshed"
            return {}
        if path == "/pulls/7":
            return copy.deepcopy(self.pr)
        if path.startswith("/commits/") and "/check-runs" in path:
            return {"check_runs": self.checks}
        if path.startswith("/actions/runs/"):
            return self.run
        if path == f"/commits/{self.run['head_sha']}/pulls":
            return [self.pr] if self.merged else []
        if path.startswith("/contents/pyproject.toml"):
            return {"content": base64.b64encode(b'[project]\nversion = "0.1.1"').decode()}
        if path.startswith("/git/ref/tags/"):
            return self.tag
        if path.startswith("/releases/tags/"):
            return self.published
        if method == "POST" and path == "/git/refs":
            self.tag = {"object": {"type": "commit", "sha": data["sha"]}}
            return self.tag
        if method == "POST" and path == "/releases":
            self.published = data
            return data
        raise AssertionError((method, path))


def test_prepares_patch_and_refreshes_after_another_pr_merges():
    api = FakeGitHub()
    release.prepare_pr(api, api.pr, "base", "0.1.1")
    assert versions.current(api.writes[-1]) == "0.1.2"
    assert api.dispatched == ["versioned"]
    # A second PR still carries the first number; main now contains that patch.
    api.base = "next-base"
    api.behind = 1
    release.prepare_pr(api, api.pr, "next-base", "0.1.2")
    assert versions.current(api.writes[-1]) == "0.1.3"
    assert any(path == "/pulls/7/update-branch" for _, path, _ in api.calls)


@pytest.mark.parametrize("kind", ["draft", "fork"])
def test_drafts_and_forks_are_never_written(kind):
    api = FakeGitHub()
    if kind == "draft":
        api.pr["draft"] = True
    else:
        api.pr["head"]["repo"]["full_name"] = "contributor/fork"
    release.prepare_pr(api, api.pr, "base", "0.1.1")
    assert not api.calls and not api.writes and not api.dispatched


def test_unchanged_version_does_not_create_bot_loop():
    api = FakeGitHub()
    api.values = versions.rewrite(api.values, "0.1.2")
    api.checks = [{"name": "application-verification"}]
    release.prepare_pr(api, api.pr, "base", "0.1.1")
    assert not api.writes and not api.dispatched


def test_main_advance_aborts_preparation_and_main_is_never_written():
    api = FakeGitHub()
    api.base = "new-main"
    release.prepare_pr(api, api.pr, "base", "0.1.1")
    assert not api.writes and not api.dispatched
    api.pr["head"]["ref"] = "main"
    with pytest.raises(ValueError, match="Refusing"):
        release.prepare_pr(api, api.pr, "new-main", "0.1.1")


def test_ref_update_cannot_overwrite_a_concurrent_author_push(monkeypatch):
    api = release.GitHub("org/repo", "fixture-token")
    calls = []

    def request(method, path, data):
        calls.append((method, path, data))
        if method == "PATCH":
            assert data == {"sha": "new", "force": False}
            raise release.APIError("Non-fast-forward")
        return {"sha": "new"}

    monkeypatch.setattr(api, "request", request)
    before = versions.rewrite(versions.load(ROOT), "0.1.1")
    with pytest.raises(release.APIError, match="Non-fast-forward"):
        api.write_versions(
            pr(), {"tree": {"sha": "tree"}}, before, versions.rewrite(before, "0.1.2")
        )
    assert all("heads/main" not in path for _, path, _ in calls)


def test_stale_dispatch_cannot_validate_another_head(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "outdated")
    with pytest.raises(ValueError, match="advanced"):
        release.check_ci(FakeGitHub(), {"inputs": {"version_pr": "7"}}, ROOT)


@pytest.mark.parametrize(
    "field,value",
    [
        ("conclusion", "failure"),
        ("event", "pull_request"),
        ("event", "workflow_dispatch"),
        ("head_branch", "feature"),
        ("head_repository", {"full_name": "fork/repo"}),
        ("path", ".github/workflows/other.yml"),
    ],
)
def test_unverified_or_noncanonical_ci_cannot_publish(field, value):
    api = FakeGitHub()
    api.run[field] = value
    with pytest.raises(ValueError, match="canonical"):
        release.publish(api, 123)
    assert not api.published and not api.tag


def test_only_merged_pr_commits_on_main_are_published():
    api = FakeGitHub()
    api.values = versions.rewrite(api.values, "0.1.2")
    api.merged = False
    release.publish(api, 123)
    assert not api.published
    api.merged = True
    api.behind = 1
    with pytest.raises(ValueError, match="no longer on main"):
        release.publish(api, 123)
    assert not api.published


def test_publishes_the_verified_sha_once_without_retargeting_tags():
    api = FakeGitHub()
    api.values = versions.rewrite(api.values, "0.1.2")
    release.publish(api, 123)
    assert api.published["tag_name"] == "v0.1.2"
    assert api.published["target_commitish"] == "merged"
    assert api.published["make_latest"] == "legacy"
    assert not api.published["draft"] and not api.published["prerelease"]
    release.publish(api, 123)
    assert sum(path == "/releases" for _, path, _ in api.calls) == 1
    api.tag["object"]["sha"] = "different"
    with pytest.raises(ValueError, match="Immutable"):
        release.publish(api, 123)


def test_historical_release_is_never_moved():
    api = FakeGitHub()
    release.publish(api, 123)
    assert not api.tag and not api.published


def test_privileged_workflows_execute_only_trusted_main():
    import yaml

    for name in ("patch-versions.yml", "patch-release.yml"):
        value = yaml.safe_load((ROOT / ".github/workflows" / name).read_text())
        job = next(iter(value["jobs"].values()))
        checkout = job["steps"][0]
        assert checkout["with"] == {"ref": "main", "persist-credentials": False}
        assert "pull_request_target" not in value.get(True, {}) or name == "patch-versions.yml"
        assert all("download-artifact" not in step.get("uses", "") for step in job["steps"])


def test_version_gate_pins_python_instead_of_relying_on_the_runner_system():
    import yaml

    workflow = yaml.safe_load((ROOT / ".github/workflows/application.yml").read_text())
    steps = workflow["jobs"]["scope"]["steps"]
    gate = next(
        i
        for i, step in enumerate(steps)
        if step.get("name") == "Validate the application patch version"
    )
    assert any(
        step.get("uses") == "actions/setup-python@v6" and step["with"]["python-version"] == "3.14.7"
        for step in steps[:gate]
    )


def test_draft_cannot_keep_a_green_version_gate_before_becoming_ready(monkeypatch):
    api = FakeGitHub()
    api.pr["draft"] = True
    monkeypatch.setattr(release, "load", lambda root: api.values)
    with pytest.raises(ValueError, match="Expected patch"):
        release.check_ci(api, {"pull_request": api.pr}, ROOT)
    api.values = versions.rewrite(api.values, "0.1.2")
    release.check_ci(api, {"pull_request": api.pr}, ROOT)


def test_prs_targeting_other_branches_do_not_require_a_main_patch():
    api = FakeGitHub()
    api.pr["base"]["ref"] = "feature"
    release.check_ci(api, {"pull_request": api.pr}, ROOT)
    assert not api.calls


@pytest.mark.parametrize("before,after", [("0.1.2", "0.1.3"), ("0.1.3", "0.1.4")])
def test_main_push_validates_the_increment_from_previous_main(monkeypatch, before, after):
    api = FakeGitHub()
    api.run["head_sha"] = "b" * 40
    api.pr["merge_commit_sha"] = api.run["head_sha"]
    values = versions.rewrite(api.values, after)
    monkeypatch.setattr(release, "load", lambda root: values)
    monkeypatch.setattr(
        release,
        "version_at",
        lambda client, sha: before if sha == "a" * 40 else pytest.fail("wrong base"),
    )
    monkeypatch.setenv("GITHUB_SHA", api.run["head_sha"])
    # Rebase can have several feature commits before its tip; event.before stays the prior main.
    event = {"ref": "refs/heads/main", "before": "a" * 40, "after": api.run["head_sha"]}
    release.check_ci(api, event, ROOT)
    monkeypatch.setattr(release, "load", lambda root: versions.rewrite(values, before))
    with pytest.raises(ValueError, match="Expected patch"):
        release.check_ci(api, event, ROOT)


def test_pr_event_with_an_old_main_snapshot_cannot_validate(monkeypatch):
    api = FakeGitHub()
    api.base = "new-main"
    monkeypatch.setattr(release, "load", lambda root: versions.rewrite(api.values, "0.1.2"))
    with pytest.raises(ValueError, match="Main advanced"):
        release.check_ci(api, {"pull_request": api.pr}, ROOT)


def test_pr_behind_main_cannot_validate_even_with_the_right_number(monkeypatch):
    api = FakeGitHub()
    api.behind = 1
    monkeypatch.setattr(release, "load", lambda root: versions.rewrite(api.values, "0.1.2"))
    with pytest.raises(ValueError, match="behind main"):
        release.check_ci(api, {"pull_request": api.pr}, ROOT)


def test_retarget_to_main_refreshes_ci_even_if_the_version_is_prepared():
    api = FakeGitHub()
    api.values = versions.rewrite(api.values, "0.1.2")
    api.checks = [{"name": "application-verification"}]
    release.prepare_all(
        api, {"action": "edited", "changes": {"base": {"ref": {"from": "feature"}}}}
    )
    assert not api.writes
    assert api.dispatched == ["head"]
    api.calls.clear()
    api.dispatched.clear()
    release.prepare_all(api, {"action": "edited", "changes": {"title": {"from": "old title"}}})
    assert not api.calls and not api.dispatched


def test_main_gate_requires_the_exact_push_and_previous_main(monkeypatch):
    api = FakeGitHub()
    event = {"ref": "refs/heads/main", "before": "a" * 40, "after": "b" * 40}
    monkeypatch.setenv("GITHUB_SHA", "c" * 40)
    with pytest.raises(ValueError, match="checked-out"):
        release.check_ci(api, event, ROOT)
    monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    event["before"] = "0" * 40
    with pytest.raises(ValueError, match="previous commit"):
        release.check_ci(api, event, ROOT)


def test_workflows_cover_retargets_and_main_push_version_validation():
    import yaml

    prepare = yaml.safe_load((ROOT / ".github/workflows/patch-versions.yml").read_text())
    assert "edited" in prepare[True]["pull_request_target"]["types"]
    application = yaml.safe_load((ROOT / ".github/workflows/application.yml").read_text())
    steps = application["jobs"]["scope"]["steps"]
    gate = next(
        step for step in steps if step.get("name") == "Validate the application patch version"
    )
    setup = next(step for step in steps if step.get("uses") == "actions/setup-python@v6")
    assert gate["if"] == setup["if"]
    assert "github.event_name == 'push'" in gate["if"] and "refs/heads/main" in gate["if"]


def test_main_advance_while_the_version_is_checked_rejects_the_run(monkeypatch):
    api = FakeGitHub()
    reads = iter(["base", "next-base"])
    monkeypatch.setattr(api, "main", lambda: next(reads))
    monkeypatch.setattr(release, "load", lambda root: versions.rewrite(api.values, "0.1.2"))
    with pytest.raises(ValueError, match="during version validation"):
        release.check_ci(api, {"pull_request": api.pr}, ROOT)


def test_main_version_gate_does_not_depend_on_merged_pr_metadata(monkeypatch):
    api = FakeGitHub()
    api.merged = False
    api.run["head_sha"] = "b" * 40
    monkeypatch.setenv("GITHUB_SHA", api.run["head_sha"])
    monkeypatch.setattr(release, "version_at", lambda client, sha: "0.1.3")
    event = {"ref": "refs/heads/main", "before": "a" * 40, "after": api.run["head_sha"]}
    monkeypatch.setattr(release, "load", lambda root: versions.rewrite(api.values, "0.1.3"))
    with pytest.raises(ValueError, match="Expected patch"):
        release.check_ci(api, event, ROOT)
    monkeypatch.setattr(release, "load", lambda root: versions.rewrite(api.values, "0.1.4"))
    release.check_ci(api, event, ROOT)
