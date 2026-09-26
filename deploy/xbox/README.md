# Xbox deployment

Run the application under a dedicated user on the Xbox host with Python 3.14.7.
The Supervisor example expands that user’s `HOME` and uses `~/ollamaforvlas`.
The web client is built locally and copied with the application source; Node and
GPU training dependencies are not needed to serve it.

Keep timestamped releases under `~/ollamaforvlas/releases/` and select the active
release with `current`. The application and cloud jobs run independently of the
browser connection. Connect Google Cloud on the server through its CLI and
application-default credentials; SkyPilot needs both compute and storage access.
New jobs store checkpoints in private GCS and keep only small descriptors and
telemetry on the application host.

Real SmolVLA and ACT training and SmolVLA quantization were verified using this
layout. See [the verification record](../../docs/cloud-training-verification.md)
for measured losses, artifact checks and validation limits.

- Root: `~/ollamaforvlas`
- Releases: `releases/<timestamp>`; `current` selects the active release.
- Application workspace: `data/` (separate from the Mac workspace).
- Listener: `127.0.0.1:8096`.
- Supervisor: `.supervisor/bin/supervisord`, config `supervisord.conf`.
- Logs: `logs/`, capped at 1 MiB per file with two backups per stream.
- A user crontab `@reboot` entry starts Supervisor when WSL starts; Supervisor
  restarts the app after unexpected failures. It does not start WSL when Windows
  itself is shut down.

Service commands over SSH:

```sh
~/ollamaforvlas/.supervisor/bin/supervisorctl -c ~/ollamaforvlas/supervisord.conf status
~/ollamaforvlas/.supervisor/bin/supervisorctl -c ~/ollamaforvlas/supervisord.conf restart firebird
curl -fsS http://127.0.0.1:8096/api/v1/health
```

Restart only when no active run depends on the application process. To roll back,
select a preserved release with the `current` symlink and restart the service.
Do not remove `data/` or another experiment's files during deployment.

The listener stays on loopback. Use an SSH tunnel to reach it, replacing `xbox-360` with the
configured SSH host alias:

```sh
ssh -N -L 127.0.0.1:18096:127.0.0.1:8096 xbox-360
```

Then open `http://127.0.0.1:18096`. The application has no public hosted login.
Cloud account connections and training workers are configured independently on
Xbox; deployment does not copy the Mac's authentication credentials or checkpoints.

## Public Funnel route

The public frontend is built with `NEXT_PUBLIC_BASE_PATH=/firebird`. It is served
by `firebird-public`, a separate password-protected gateway on `127.0.0.1:8097`.
This gateway serves the prefixed export and proxies authenticated API requests to
the existing backend on `8096`; it does not start a second workspace owner.
The normal root deployment remains available independently.
Gateway credential loading requires POSIX ownership, mode and no-follow checks
(as provided by Linux/WSL); it explicitly refuses native Windows configuration
loading. Non-regular files, including FIFOs, are rejected without blocking startup.

- Intended URL: `https://<host>.<tailnet>.ts.net/firebird/`
- Public export: `public-web/funnel-20260926/`
- Gateway code: `public_gateway.py`
- Private credential configuration: `private/public-gateway.json`, owned by the
  service user with mode `0600`. It contains a salted password hash, not plaintext.
- Store generated login details outside the repository. Never commit the private
  gateway configuration or plaintext password.
- Protected preview through an SSH tunnel: `http://127.0.0.1:18097/firebird/`.

Public activation requires the Xbox administrator to run:

```sh
sudo tailscale funnel --bg --https=443 --set-path=/firebird http://127.0.0.1:8097/firebird
```

The existing `/` service and other Funnel ports remain in place. Tailscale strips
the mount prefix; the `/firebird` target suffix adds it back for the gateway.
The `--bg` configuration persists across Tailscale restarts. Do not use `reset`.

The public route is **prepared, not published** until the administrator activates
it. Use only the scoped route above; do not grant broad operator privileges to
work around missing administrator access.

After activation, verify that unauthenticated public page/API requests return
401, authenticated page/API requests succeed, `/firebird/docs` redirects to
`/firebird/docs/`, and the pre-existing root site is still reachable. Change the
private credential configuration and restart `firebird-public` to rotate its
password; the gateway configuration is immutable for a running process.
