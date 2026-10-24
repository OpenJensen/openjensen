# Versions and update channels

OpenJensen has two identifiers: a package version and an exact source commit. The
version button in the header shows both, the server being controlled, upstream
`main`, and the latest published GitHub release.

| Identifier | Value |
| --- | --- |
| Latest published release | [GitHub releases](https://github.com/OpenJensen/openjensen/releases/latest) |
| Current core, web and desktop package version | `0.1.1` |
| Current development build | The commit displayed by the running server |

A package containing `0.1.1` is not evidence that a `v0.1.1` release exists.
Source builds between releases are identified by their commit. Release tags are
immutable. [Version 0.1.1](releases/0.1.1.md) follows v0.0.1; no v0.1.0 release was published.

## Release policy

Use [Semantic Versioning](https://semver.org/). During the `0.x` development phase,
record compatibility changes explicitly. Use patch releases for fixes and minor
releases for feature milestones; keep the Python workspace/core, web package,
desktop Cargo/Tauri version and desktop resource manifest consistent. Before a
release, validate the selected commit in CI, write release notes and upgrade
instructions, and create a new `vX.Y.Z` tag and GitHub release from that commit.
Never move an existing release tag to newer code.

## Main updates

The browser updater follows **upstream main**, a development channel, rather than
GitHub releases. It compares commit ancestry, so it detects changes even when the
package number stays the same. A build ahead of or diverged from main is shown as
a development build and cannot be automatically replaced by older main code.

The confirmed commit is pinned. If main advances during preparation, the update
fails without changing the running application; check and confirm the new target.
See [managed updates](upgrading.md#confirmed-browser-updates) for deployment support,
backups and rollback behavior.
