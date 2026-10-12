# Backup, restore and alpha upgrades

Keep the workspace in a persistent directory outside a disposable checkout. Set `JENSEN_DATA_DIR` to its absolute path when starting the application. One application process owns each workspace.

## Alpha compatibility

The canonical command is `jensen`, settings use `JENSEN_*`, and new workspaces default to `.jensen/`. The alpha command `firebird` and `FIREBIRD_*` settings remain compatibility aliases. When `.firebird/` already exists and `.jensen/` does not, the application reuses the alpha workspace rather than creating a replacement. Use an explicit `JENSEN_DATA_DIR` when multiple directories are present or when launching from another location.

Existing dataset metadata names such as `meta/firebird-lineage.json` and `meta/firebird-demonstrations.json`, stored format discriminators and model identities are compatibility protocols. Keep their saved bytes and names. Do not bulk-replace product names inside databases, manifests, recipes or datasets. Use the current worker imports and paths when configuring a new installation.

## Make a stopped-workspace backup

1. Finish active jobs or cancel each selected job and inspect its cleanup status. Confirm separately that cloud runs and paid resources have stopped before shutting down the API.
2. Stop the API with `Ctrl+C`, or stop its service and prevent automatic restart. Stop any desktop-owned backend and standalone writer using the same workspace. Confirm the server process has exited before copying.
3. Copy the **whole** workspace to a new private directory. Include hidden files, the database and any SQLite sidecars, job records, local artifacts, settings and submission journals. Do not copy only the database.

On macOS/Linux, set `JENSEN_WORKSPACE` to the actual stopped directory. Choose an unused backup path; `cp -pR` preserves the workspace contents beneath it:

```sh
JENSEN_WORKSPACE=/absolute/path/to/workspace
JENSEN_BACKUP=/absolute/private/backups/workspace-before-upgrade
test -d "$JENSEN_WORKSPACE" && test ! -e "$JENSEN_BACKUP" && \
  cp -pR "$JENSEN_WORKSPACE" "$JENSEN_BACKUP"
```

Run the copy only after the checks succeed. In PowerShell use `Copy-Item -LiteralPath 'C:\workspace' -Destination 'C:\private\workspace-before-upgrade' -Recurse` after confirming the destination does not exist. Check that the copied files are readable and complete. Keep the original stopped workspace and backup until the upgraded copy has been reviewed.

The backup preserves remote artifact references, not the referenced GCS objects or external dataset/model directories. Back up those locations separately if you need an offline or independent recovery copy. A capture catalog or runtime can depend on paths outside the workspace; retain those dependencies and their configuration. Provider credentials and API tokens may also live outside it. Store all backups privately.

## Try an upgrade on a copy

Record the old source revision, configuration and worker versions. Preserve the old checkout/runtime and build the new release in a separate checkout. Install its locked application dependencies and rebuild its web export; update worker environments only according to the selected worker's guide.

Copy the stopped backup to another new candidate directory. From the new checkout, point the application at the candidate using an absolute path:

```sh
JENSEN_DATA_DIR=/absolute/private/upgrade-candidate uv run --frozen jensen serve --port 8001
```

Open [localhost:8001](http://127.0.0.1:8001). Inspect projects, saved jobs, settings and representative local artifact downloads/previews. Keep cloud or hardware actions idle during this check. Stop the candidate, restart it with the same directory, and confirm that those records remain available.

Database schema changes can occur on startup. There is no general promise that an older application can open a workspace already modified by a newer version. The experimental desktop maintenance tools are not a certified two-package upgrade or rollback workflow.

## Restore

Stop all writers first. Restore by copying the **unchanged stopped backup** into a new private directory and launch the preserved old checkout/runtime with its supported data-directory setting pointing there. For an alpha version that predates the rename, use `FIREBIRD_DATA_DIR`; current releases accept `JENSEN_DATA_DIR` and its alpha alias. Keep the failed candidate for diagnosis. Do not merge database files from different versions or overwrite the only copy of a workspace.

Confirm project/job records and representative artifacts after launch and again after a restart. Restore external data, model paths and cloud access separately when needed. If a check fails, retain the original/backup and report the exact source revisions with redacted errors through [GitHub issues](https://github.com/OpenJensen/openjensen/issues).
