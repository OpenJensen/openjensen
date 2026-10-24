# Deploy on a Linux server

Use Python 3.14.7 and a dedicated service user on the host. Build the web client locally, then copy the source and export into a timestamped release under `~/openjensen/releases/`.

Use the canonical public route below for a new deployment. The supplied Supervisor
template retains the alpha service labels `firebird` and `firebird-public` for
operational compatibility; use those labels in control commands. Existing alpha
installations keep their configured service names, paths, username, credentials
and cloud identities until their administrator explicitly migrates them. Updating
source does not rename deployed resources. Use [stop-and-copy backups](../../docs/upgrading.md)
before changing a release; rollback requires the preserved pre-upgrade workspace,
not only the old application symlink.

## Set up the service

1. Install the application dependencies in the release with `uv sync --frozen`.
2. Install the [isolated CPU reader](../../workers/_cpu_readers/README.md) in the release checkout, and install FFmpeg/ffprobe on the host for dataset import and video previews.
3. Select the release using the `~/openjensen/current` symlink.
4. Create `data/`, `run/` and `logs/` under `~/openjensen`.
5. Install Supervisor into `~/openjensen/.supervisor` and copy `supervisord.conf` into the deployment root.
6. Start Supervisor using that configuration. Add an `@reboot` user crontab entry to start it when WSL starts.
7. Configure [cloud credentials and workers](../../docs/cloud-connections.md) on the service host.

The configuration uses `data/` for application data, `current/apps/web/out` for web files and `127.0.0.1:8096` for the backend. Keep `data/` across release changes.

## Check and restart

Run on the deployment host:

```sh
~/openjensen/.supervisor/bin/supervisorctl -c ~/openjensen/supervisord.conf status
~/openjensen/.supervisor/bin/supervisorctl -c ~/openjensen/supervisord.conf restart firebird
curl -fsS http://127.0.0.1:8096/api/v1/health
```

Finish or cancel active application jobs and confirm their cleanup before stopping
the service. Back up the whole stopped workspace before an upgrade. For rollback,
use the preserved release/runtime with a restored copy of its pre-upgrade
workspace. Changing `current` alone cannot reverse a database migration.

## Browser updates

Record the exact source commit in each archived release as
`.openjensen-build.json` with `{"commit":"FULL_40_CHARACTER_SHA","channel":"main"}`.
Existing `.openjensen-release.json` receipts containing `commit` are also supported.
The header displays the package version, this commit and upstream main separately.
Configure [confirmed browser updates](../../docs/upgrading.md#confirmed-browser-updates)
to build, back up and restart this service from a private browser session.

## Connect through SSH

Replace `gpu-server` with the SSH host alias of the server:

```sh
ssh -N -L 127.0.0.1:18096:127.0.0.1:8096 gpu-server
```

Open [localhost:18096](http://127.0.0.1:18096). Read service logs under `~/openjensen/logs/`.

## Public Funnel route

1. Build the public frontend with `NEXT_PUBLIC_BASE_PATH=/jensen`.
2. Place the export under `public-web/funnel-20260926/` and copy `public_gateway.py` into the deployment root.
3. Create `private/public-gateway.json` for the service user, using mode `0600` and a salted password hash. Keep the generated login details outside the repository.
4. Start or restart the template's `firebird-public` Supervisor service on `127.0.0.1:8097`.
5. Preview through a loopback SSH tunnel at `http://127.0.0.1:18097/jensen/`.
6. Have the host administrator activate the scoped route using the command below.

Configure the gateway JSON with these fields, replacing the host, service-user path and password-derived values:

```json
{
  "public_origin": "https://host.tailnet.ts.net",
  "static_dir": "/home/service-user/openjensen/public-web/funnel-20260926",
  "username": "jensen",
  "path_prefix": "/jensen",
  "password_salt_hex": "REPLACE_WITH_16_BYTE_SALT_AS_HEX",
  "password_hash_hex": "REPLACE_WITH_PBKDF2_SHA256_HASH_AS_HEX",
  "password_iterations": 310000,
  "preview_origins": ["http://127.0.0.1:18097"]
}
```

Generate a random 16-byte salt and derive a 32-byte PBKDF2-HMAC-SHA256 hash from the chosen password using 310000 iterations. Encode both as hexadecimal. Use an absolute path to the built public export for `static_dir`.

`JENSEN_PUBLIC_CONFIG` selects the private configuration file; the alpha
`FIREBIRD_PUBLIC_CONFIG` setting remains accepted. The template supplies both
settings, and both application data/web settings, with identical paths so a
preserved alpha runtime can still use them during rollback. New gateway configs
use `username: "jensen"` and `path_prefix: "/jensen"`. An existing config without
`path_prefix` retains `/firebird`; its existing username, route, frontend base path
and Funnel mapping must be migrated together if the administrator chooses to
change them.

Activate the route on the host:

```sh
sudo tailscale funnel --bg --https=443 --set-path=/jensen http://127.0.0.1:8097/jensen
```

Open `https://<host>.<tailnet>.ts.net/jensen/`. Check login, authenticated API requests, `/jensen/docs/` and the existing root route. To rotate the gateway password, update the private configuration and restart `firebird-public`.
