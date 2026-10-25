"""Trusted GitHub automation: prepare PR metadata and publish verified main releases.

No PR code, build artifact or PR-provided workflow is executed by this script.
Only the fixed version-manifest paths may be written. Main is never pushed.
"""

import argparse
import base64
import json
import os
import re
import time
import tomllib
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from version_files import FILES, current, load, next_patch, parse, rewrite, validate

MANUAL = "version:manual"
START_AFTER = "0.1.1"


class APIError(RuntimeError):
    pass


class GitHub:
    def __init__(self, repository=None, token=None):
        self.repository = repository or os.environ["GITHUB_REPOSITORY"]
        self.token = token or os.environ["GH_TOKEN"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.repository):
            raise ValueError("Invalid repository.")
        self.prefix = "/repos/" + self.repository

    def request(self, method, path, data=None, missing=False):
        request = Request(
            "https://api.github.com" + self.prefix + path,
            data=json.dumps(data).encode() if data is not None else None,
            method=method,
            headers={
                "Authorization": "Bearer " + self.token,
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=40) as response:
                body = response.read(12 * 1024**2 + 1)
                if len(body) > 12 * 1024**2:
                    raise APIError("GitHub response exceeds its bound.")
                return json.loads(body) if body else None
        except HTTPError as exc:
            if missing and exc.code == 404:
                return None
            raise APIError(f"GitHub {method} {path}: HTTP {exc.code}") from exc

    def pages(self, path):
        page = 1
        while True:
            values = self.request(
                "GET", path + ("&" if "?" in path else "?") + f"per_page=100&page={page}"
            )
            yield from values
            if len(values) < 100:
                break
            page += 1

    def main(self):
        return self.request("GET", "/git/ref/heads/main")["object"]["sha"]

    def contents(self, sha):
        commit = self.request("GET", "/git/commits/" + sha)
        tree = self.request("GET", "/git/trees/" + commit["tree"]["sha"] + "?recursive=1")
        if tree.get("truncated"):
            raise ValueError("Refusing an incomplete source tree.")
        entries = {entry["path"]: entry for entry in tree["tree"]}
        result = {}
        for path in FILES:
            entry = entries.get(path, {})
            if entry.get("type") != "blob" or entry.get("mode") not in {"100644", "100755"}:
                raise ValueError(f"Manifest must be a regular Git blob: {path}")
            blob = self.request("GET", "/git/blobs/" + entry["sha"])
            if blob["encoding"] != "base64" or blob["size"] > 2 * 1024**2:
                raise ValueError("Version manifest exceeds its bound.")
            result[path] = base64.b64decode(blob["content"], validate=False).decode("utf-8")
        return commit, result

    def write_versions(self, pr, commit, before, after):
        entries = [
            {"path": path, "mode": "100644", "type": "blob", "content": after[path]}
            for path in FILES
            if before[path] != after[path]
        ]
        if not entries:
            return pr["head"]["sha"]
        tree = self.request(
            "POST", "/git/trees", {"base_tree": commit["tree"]["sha"], "tree": entries}
        )
        new_commit = self.request(
            "POST",
            "/git/commits",
            {
                "message": "Prepare patch version " + current(after),
                "tree": tree["sha"],
                "parents": [pr["head"]["sha"]],
            },
        )
        # A concurrent author push makes this non-fast-forward and fails safely.
        self.request(
            "PATCH",
            "/git/refs/heads/" + quote(pr["head"]["ref"], safe=""),
            {"sha": new_commit["sha"], "force": False},
        )
        return new_commit["sha"]

    def dispatch_ci(self, pr, sha):
        self.request(
            "POST",
            "/actions/workflows/application.yml/dispatches",
            {
                "ref": pr["head"]["ref"],
                "inputs": {"windows_browser": "false", "version_pr": str(pr["number"])},
            },
        )
        print(f"PR #{pr['number']}: CI requested for {sha[:12]}")


def manual(pr):
    return any(label["name"] == MANUAL for label in pr.get("labels", []))


def prepare_pr(api, pr, base_sha, base_version, force_ci=False):
    number = pr["number"]
    if pr.get("draft") or pr["base"]["ref"] != "main":
        return
    if pr["head"]["repo"]["full_name"] != api.repository:
        print(f"PR #{number}: fork author must run version_files.py apply --base-ref origin/main.")
        return
    if pr["head"]["ref"] == "main":
        raise ValueError("Refusing to write main.")
    updated = False
    comparison = api.request("GET", f"/compare/{base_sha}...{pr['head']['sha']}")
    if comparison["behind_by"]:
        previous = pr["head"]["sha"]
        api.request("PUT", f"/pulls/{number}/update-branch", {"expected_head_sha": previous})
        for _ in range(30):
            pr = api.request("GET", f"/pulls/{number}")
            if pr["head"]["sha"] != previous:
                break
            time.sleep(1)
        else:
            raise APIError("Base refresh is still pending; rerun preparation.")
        updated = True
    if api.main() != base_sha:
        print("Main advanced; the queued preparation run will use the new version.")
        return
    commit, contents = api.contents(pr["head"]["sha"])
    result = contents if manual(pr) else rewrite(contents, next_patch(base_version))
    version = validate(result, base_version, manual(pr))
    sha = api.write_versions(pr, commit, contents, result)
    checks = api.request("GET", f"/commits/{sha}/check-runs?per_page=100")["check_runs"]
    has_ci = any(check["name"] in {"scope", "application-verification"} for check in checks)
    if updated or sha != pr["head"]["sha"] or force_ci or not has_ci:
        api.dispatch_ci(pr, sha)
    print(f"PR #{number}: prepared {version}")


def prepare_all(api, event):
    if (
        event.get("action") in {"labeled", "unlabeled"}
        and event.get("label", {}).get("name") != MANUAL
    ):
        return
    base_sha = api.main()
    _, contents = api.contents(base_sha)
    base_version = validate(contents)
    failed = []
    for pr in api.pages("/pulls?state=open&base=main"):
        try:
            prepare_pr(
                api, pr, base_sha, base_version, event.get("action") in {"labeled", "unlabeled"}
            )
        except Exception as exc:
            failed.append(pr["number"])
            print(f"PR #{pr['number']}: {exc}")
    if failed:
        raise ValueError("Preparation needs attention for PRs: " + ", ".join(map(str, failed)))


def check_ci(api, event, root):
    pr = event.get("pull_request")
    if pr is None:
        number = event.get("inputs", {}).get("version_pr", "")
        if not number:
            print("No PR version check is required for this event.")
            return
        if not str(number).isdigit():
            raise ValueError("Invalid PR number.")
        pr = api.request("GET", f"/pulls/{number}")
        if pr["head"]["sha"] != os.environ["GITHUB_SHA"]:
            raise ValueError("PR advanced after CI dispatch; verify its current head.")
    if pr.get("draft"):
        print("Patch version is prepared when the draft is ready for review.")
        return
    # Read the base snapshot; no fork files or artifacts are executed.
    base_file = api.request("GET", "/contents/pyproject.toml?ref=" + pr["base"]["sha"])
    base_version = tomllib.loads(base64.b64decode(base_file["content"]).decode())["project"][
        "version"
    ]
    version = validate(load(root), base_version, manual(pr))
    print("Validated PR version " + version)


def publish(api, run_id):
    run = api.request("GET", f"/actions/runs/{run_id}")
    if (
        run["conclusion"] != "success"
        or run["event"] != "push"
        or run["head_branch"] != "main"
        or run["head_repository"]["full_name"] != api.repository
        or run["path"].split("@", 1)[0] != ".github/workflows/application.yml"
    ):
        raise ValueError("Only successful canonical main CI may publish a release.")
    sha = run["head_sha"]
    relation = api.request("GET", f"/compare/{sha}...{api.main()}")
    if relation["behind_by"]:
        raise ValueError("Release commit is no longer on main.")
    prs = api.request("GET", f"/commits/{sha}/pulls")
    merged = [
        pr
        for pr in prs
        if pr.get("merged_at") and pr["base"]["ref"] == "main" and pr["merge_commit_sha"] == sha
    ]
    if not merged:
        print("Main push is not a merged PR; no automatic release.")
        return
    _, contents = api.contents(sha)
    version = validate(contents)
    if parse(version) <= parse(START_AFTER):
        print("Historical release is outside automatic patch publication.")
        return
    tag = "v" + version
    existing = api.request("GET", "/git/ref/tags/" + tag, missing=True)
    if existing:
        obj = existing["object"]
        if obj["type"] == "tag":
            obj = api.request("GET", "/git/tags/" + obj["sha"])["object"]
        if obj["type"] != "commit" or obj["sha"] != sha:
            raise ValueError("Immutable release tag already points at another commit.")
    else:
        api.request("POST", "/git/refs", {"ref": "refs/tags/" + tag, "sha": sha})
    release = api.request("GET", "/releases/tags/" + tag, missing=True)
    if release:
        if release["draft"] or release["prerelease"]:
            raise ValueError("An existing draft/prerelease needs maintainer review.")
        print(tag + " is already published.")
        return
    api.request(
        "POST",
        "/releases",
        {
            "tag_name": tag,
            "target_commitish": sha,
            "name": "OpenJensen " + version,
            "draft": False,
            "prerelease": False,
            "generate_release_notes": True,
            "make_latest": "legacy",
            "body": f"Source release from PR #{merged[0]['number']}. "
            "Published after main CI passed. "
            "Install application updates with confirmation in Settings. "
            "Native desktop installers, worker runtimes and model files are separate.",
        },
    )
    print("Published " + tag)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "check-ci", "publish"])
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    api = GitHub()
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    if args.command == "prepare":
        prepare_all(api, event)
    elif args.command == "check-ci":
        check_ci(api, event, args.root)
    else:
        publish(api, event["workflow_run"]["id"])


if __name__ == "__main__":
    main()
