"""One version number can sit on many unmerged branches; a build label cannot.

Every case uses a real throwaway Git repository: the rule under test is what Git says
about a commit, so nothing here is mocked.
"""

import json
import subprocess

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner
from vla_platform import __version__, build
from vla_platform.api import create_app
from vla_platform.build import Build, identify, same_code
from vla_platform.cli import app
from vla_platform.settings import Settings


def git(root, *arguments):
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.invalid",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "PATH": __import__("os").environ["PATH"],
        },
    ).stdout.strip()


@pytest.fixture
def checkout(tmp_path):
    """A clone whose ``origin/main`` holds one released commit."""
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "-q", "-b", "main")
    (origin / "app.py").write_text("release = 1\n")
    git(origin, "add", "app.py")
    git(origin, "commit", "-q", "-m", "release")
    clone = tmp_path / "clone"
    git(tmp_path, "clone", "-q", str(origin), str(clone))
    return clone


def commit(root, text):
    (root / "app.py").write_text(text)
    git(root, "commit", "-q", "-am", "change")
    return git(root, "rev-parse", "HEAD")


def test_a_commit_on_main_is_the_release(checkout):
    value = identify(checkout, "0.1.30")
    assert value.released and not value.dirty and value.branch == "main"
    assert value.label == "0.1.30"


def test_two_branches_with_one_number_get_two_names(checkout):
    """The situation that caused the confusion: several open PRs, all stamped 0.1.31."""
    git(checkout, "switch", "-q", "-c", "feature/camera")
    first = commit(checkout, "camera = 1\n")
    camera = identify(checkout, "0.1.31")
    git(checkout, "switch", "-q", "main")
    git(checkout, "switch", "-q", "-c", "feature/policy")
    second = commit(checkout, "policy = 1\n")
    policy = identify(checkout, "0.1.31")

    assert camera.version == policy.version == "0.1.31"
    assert not camera.released and not policy.released
    assert camera.label == f"0.1.31-dev+{first[:7]}"
    assert policy.label == f"0.1.31-dev+{second[:7]}"
    assert camera.label != policy.label
    assert not same_code(camera.label, policy.label)
    assert camera.branch == "feature/camera"


def test_the_tagged_commit_is_the_release_even_before_main_is_fetched(checkout):
    git(checkout, "switch", "-q", "-c", "release-candidate")
    commit(checkout, "candidate = 1\n")
    assert identify(checkout, "0.1.31").label.startswith("0.1.31-dev+")
    git(checkout, "tag", "v0.1.31")
    assert identify(checkout, "0.1.31").label == "0.1.31"
    # A tag for another number proves nothing about this one.
    assert identify(checkout, "0.1.32").label.startswith("0.1.32-dev+")


def test_uncommitted_source_changes_are_never_a_release(checkout):
    (checkout / "app.py").write_text("edited = 1\n")
    value = identify(checkout, "0.1.30")
    assert value.dirty and not value.released
    assert value.label.endswith(".dirty")
    assert not same_code(value.label, value.label)
    # Untracked files (logs, caches, local data) do not change what code runs.
    git(checkout, "checkout", "-q", "--", "app.py")
    (checkout / "scratch.log").write_text("noise\n")
    assert identify(checkout, "0.1.30").label == "0.1.30"


def test_a_managed_installation_reports_its_recorded_main_commit(tmp_path):
    revision = "a" * 40
    (tmp_path / ".openjensen-build.json").write_text(
        json.dumps({"commit": revision, "channel": "main"})
    )
    value = identify(tmp_path, "0.1.30")
    assert value == Build("0.1.30", revision, "main", dirty=False, released=True)
    assert value.label == "0.1.30"


def test_a_copy_with_no_history_is_not_a_release(tmp_path):
    value = identify(tmp_path, "0.1.30")
    assert value.commit is None and not value.released
    assert value.label == "0.1.30-dev+unknown"


def test_health_updates_and_cli_report_the_same_label(tmp_path):
    label = build.current().label
    assert label.startswith(__version__)
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        health = client.get("/api/v1/health").json()
        assert health == {"status": "ok", "version": __version__, "build": label}
        installed = client.app.state.application_updates.installed
        assert installed.label == label
        assert installed.released == build.current().released
    assert CliRunner().invoke(app, ["build"]).stdout.strip() == label
    printed = json.loads(CliRunner().invoke(app, ["build", "--json"]).stdout)
    assert printed["label"] == label and printed["version"] == __version__
    # ``--version`` stays the bare number for scripts that parse it.
    assert CliRunner().invoke(app, ["--version"]).stdout.strip() == __version__
