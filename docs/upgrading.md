# Back up, upgrade and restore

Keep the workspace in a lasting directory outside the source checkout, and start the server with `JENSEN_DATA_DIR` set to its absolute path. Only one server process may use a workspace at a time.

Always back up before an upgrade. Upgrades can change the database on startup, and an older version may not open a workspace a newer one has modified.

## Back up

1. **Finish or cancel jobs.** Cancel each active job and check that its cleanup finished. Confirm separately that cloud runs and paid resources have stopped.
2. **Stop every writer.** Stop the server (`Ctrl+C`, or stop its service and prevent restarts), plus any desktop-owned backend or other process using the workspace. Confirm the server process has exited.
3. **Copy the whole workspace** to a new private directory: hidden files, the database and its SQLite sidecar files, job records, local artifacts, settings and submission journals. Never copy only the database.

On macOS or Linux, using an unused backup path:

```sh
JENSEN_WORKSPACE=/absolute/path/to/workspace
JENSEN_BACKUP=/absolute/private/backups/workspace-before-upgrade
test -d "$JENSEN_WORKSPACE" && test ! -e "$JENSEN_BACKUP" && \
  cp -pR "$JENSEN_WORKSPACE" "$JENSEN_BACKUP"
```

In PowerShell, after confirming the destination does not exist:

```powershell
Copy-Item -LiteralPath 'C:\workspace' -Destination 'C:\private\workspace-before-upgrade' -Recurse
```

Check that the copy is complete and readable. Keep both the stopped original and the backup until the upgrade is confirmed.

**Not included in the backup:** objects in Google Cloud Storage (only their references are kept), dataset or model directories outside the workspace, paths a recording catalog or runtime depends on, and provider credentials stored elsewhere. Back those up separately if you need them. Store every backup privately; backups can contain credentials.

## Upgrade with Update & restart

Servers deployed with the release-directory layout below and Supervisor can update from the browser; see [browser updates](#confirmed-browser-updates). The update takes the backup itself and rolls back on failure.

## Upgrade by hand

Test the new version on a copy first.

1. Record the current source revision, configuration and worker versions, and keep the old checkout and runtime.
2. Build the new version in a separate checkout: install its locked dependencies and rebuild the web interface. Update worker environments only as each worker's guide says.
3. Copy the stopped backup into a new candidate directory and start the new version on it:

   ```sh
   JENSEN_DATA_DIR=/absolute/private/upgrade-candidate uv run --frozen jensen serve --port 8001
   ```

4. Open [localhost:8001](http://127.0.0.1:8001) and check projects, saved jobs, settings and a few local artifact downloads and previews. Leave cloud and hardware actions idle.
5. Stop the candidate, start it again on the same directory, and confirm everything is still there.

The experimental desktop maintenance tools are not a supported upgrade or rollback path.

## Restore

1. Stop every writer.
2. Copy the **unchanged** stopped backup into a new private directory.
3. Start the old checkout and runtime with its data directory pointing there: `JENSEN_DATA_DIR`, or `FIREBIRD_DATA_DIR` for an alpha version from before the rename.
4. Check projects, jobs and a few artifacts, then restart and check again. Restore external data, model paths and cloud access separately if needed.

Keep the failed candidate for diagnosis. Never merge database files from different versions or overwrite the only copy of a workspace. If a check fails, keep the original and backup, and report the exact source revisions with redacted errors in a [GitHub issue](https://github.com/OpenJensen/openjensen/issues).

## Confirmed browser updates

The version button in the header checks upstream `OpenJensen/openjensen` main every ten minutes and offers **Update & restart** when main is ahead of the installed commit. The dialog shows the exact commit and server. Checking never installs anything, and there is no unattended update.

An update from any browser updates the **server** for all its users; it installs nothing on the browser's computer.

### Requirements

- A macOS or Linux Supervisor deployment with a release-directory layout (see `deploy/linux-server/supervisord.conf`): `current` points into `releases/`, with separate `data/`, `run/`, `private/` and `supervisord.conf`.
- The service loads code and web files from `current` and reads `JENSEN_DATA_DIR` (or its alpha alias) from the persistent `data/` directory.
- Python 3.14, Node 24, uv, the pinned pnpm (or Corepack) and Supervisor are installed.
- Free disk for the build and two full workspace copies, plus at least 10 GiB.
- The public gateway is installed from the same release before updates are enabled on an older deployment.

### Set up

On the server, with its real paths:

```sh
uv run --frozen jensen configure-updates \
  --deployment-root /absolute/deployment \
  --repository /absolute/source/openjensen \
  --node-bin /absolute/node/bin
```

`--uv` and `--pnpm` optionally select existing tool paths. The command writes a private `private/application-updates.json` and prints a `JENSEN_UPDATE_CONFIG` setting: add it to the Supervisor program's environment and restart the service. Keep this configuration outside the source checkout. The repository's `origin` must be the canonical GitHub repository; updates never reset or check out local edits or feature branches in that checkout.

### What an update does

1. Blocks new writes, and refuses while jobs or Robot Lab sessions are active.
2. Fetches the confirmed main commit and builds a separate release.
3. Stops the server and takes a complete workspace backup.
4. Validates database migrations on a copy, then activates that copy and the new release.
5. Restarts the service and checks its reported commit and that project identities are preserved. The browser reconnects and reloads.

If migration or startup fails, the previous release **and its original workspace** (configuration, datasets and model references) are kept or restored. Worker environments are left alone; updates never download models or change the Robot Lab companion.

- Progress and logs: `run/application-update.json` and `run/application-update.log`.
- Backups, the original workspace and any failed candidate: `private/application-updates/<operation-id>/`. These can contain credentials; keep them private, and retire old ones only after checking nothing still depends on them.
- A **recovery required** status blocks further writes until an administrator restores the server as described in [Restore](#restore).

Update controls are private to the local or SSH-tunnelled workspace; the public gateway blocks them. Source-only installs show version information but upgrade by hand. Containers are rebuilt by their deployment manager, and desktop packages need their own verified installer.

## Alpha compatibility

The command is `jensen`, settings use `JENSEN_*`, and new workspaces default to `.jensen/`. The alpha `firebird` command and `FIREBIRD_*` settings still work. If `.firebird/` exists and `.jensen/` does not, the app reuses the alpha workspace instead of creating a new one; set `JENSEN_DATA_DIR` explicitly when both exist or when starting from another directory.

Saved names such as `meta/firebird-lineage.json`, `meta/firebird-demonstrations.json`, stored format identifiers and model identities are part of the data format. Keep them as they are, and never bulk-rename product names inside databases, manifests, recipes or datasets. New installations use the current worker imports and paths.
