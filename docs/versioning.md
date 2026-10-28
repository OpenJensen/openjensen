# Versions and update channels

OpenJensen records a package version and an exact source commit for every build. This page explains how versions are assigned, when releases are published and what the browser updater follows.

## Check your version

Select the version button in the header. The **OpenJensen updates** dialog shows:

| Field | Shows |
| --- | --- |
| **Installed** | The installed server's version and commit |
| **Server** | The server being controlled |
| **Upstream main** | The newest commit on `main` |
| **Published release** | The latest [GitHub release](https://github.com/OpenJensen/openjensen/releases/latest) |

An installed source build can be newer than the latest published release.

## Automatic patch versions

- Each ready pull request targeting `main` receives the next patch version: `0.1.1 → 0.1.2 → 0.1.3`. The patch number keeps counting: `0.1.9 → 0.1.10`.
- Automation never increments the minor or major number.
- The preparation workflow updates same-repository PR branches against `main`, then commits the version in the workspace, Core, web, desktop, lockfiles and OpenAPI metadata. Worker packages keep their independent versions.
- CI checks that the manifests agree and that the PR contains exactly the next patch. Drafts are prepared when marked ready for review.
- Parallel PRs can start with the same proposed version. When one merges, the others refresh against `main` and receive the following patch.

`main` must require the `application-verification` check **and up-to-date branches**, so a stale proposal cannot merge. Existing reviews and checks still apply; the bot does not approve or merge PRs. Resolve conflicting branch updates before merging.

Bot commits explicitly dispatch CI, because token-generated events do not reliably start another workflow. Both ordinary PR CI and the dispatched run validate the version; a dispatch for an outdated head fails.

### Forks

The privileged preparation workflow runs trusted `main` code and reads a fixed set of PR manifests as data. It never executes fork code or writes to fork branches. Fork contributors update their own branch and run:

```sh
git fetch https://github.com/OpenJensen/openjensen.git main
uv run --no-project --python 3.14.7 .github/scripts/version_files.py apply --base-ref FETCH_HEAD
```

### Minor and major milestones

A maintainer can deliberately choose a new minor or major version: apply the `version:manual` label and edit all application versions consistently. That version must be newer than `main`. The label is an explicit override, not an automatic minor-version rule. Removing it restores patch-only preparation.

## Releases

- After the merged `main` commit passes CI, publication creates its immutable `vX.Y.Z` tag and a GitHub source release with generated notes.
- Failed, cancelled, PR-only and manual CI runs cannot publish.
- Rerunning publication is safe. A tag already pointing elsewhere fails instead of being moved.
- A newer release remains latest if CI finishes out of order.
- Native installers, worker runtimes and model files are separate from source releases.

[Version 0.1.1](releases/0.1.1.md) follows v0.0.1; no v0.1.0 release was published. Existing release tags remain immutable.

## Updates from main

- The browser updater follows **upstream main**, a development channel, rather than GitHub releases.
- It compares commit ancestry, so it detects changes even when publication is still pending. An installed version is not proof that its release has been published.
- A build ahead of or diverged from `main` is shown as a development build and cannot automatically replace `main` with older code.
- The confirmed commit is pinned. If `main` advances during preparation, the update fails without changing the running application; check and confirm the new target.

See [managed updates](upgrading.md#confirmed-browser-updates) for deployment support, backups and rollback behavior.
