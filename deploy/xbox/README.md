# Deploy on Xbox

Use Python 3.14.7 and a dedicated service user on the host. Build the web client locally, then copy the source and export into a timestamped release under `~/ollamaforvlas/releases/`.

## Set up the service

1. Install the application dependencies in the release with `uv sync --frozen`.
2. Install the [isolated CPU reader](../../workers/_cpu_readers/README.md) in the release checkout, and install FFmpeg/ffprobe on the host for dataset import and video previews.
3. Select the release using the `~/ollamaforvlas/current` symlink.
4. Create `data/`, `run/` and `logs/` under `~/ollamaforvlas`.
5. Install Supervisor into `~/ollamaforvlas/.supervisor` and copy `supervisord.conf` into the deployment root.
6. Start Supervisor using that configuration. Add an `@reboot` user crontab entry to start it when WSL starts.
7. Configure [cloud credentials and workers](../../docs/cloud-connections.md) on the service host.

The configuration uses `data/` for application data, `current/apps/web/out` for web files and `127.0.0.1:8096` for the backend. Keep `data/` across release changes.

## Check and restart

Run on the deployment host:

```sh
~/ollamaforvlas/.supervisor/bin/supervisorctl -c ~/ollamaforvlas/supervisord.conf status
~/ollamaforvlas/.supervisor/bin/supervisorctl -c ~/ollamaforvlas/supervisord.conf restart firebird
curl -fsS http://127.0.0.1:8096/api/v1/health
```

Finish active application jobs before restarting. To roll back, point `current` at a preserved release and restart the service.

## Connect through SSH

Replace `xbox-360` with the configured SSH host alias:

```sh
ssh -N -L 127.0.0.1:18096:127.0.0.1:8096 xbox-360
```

Open [localhost:18096](http://127.0.0.1:18096). Read service logs under `~/ollamaforvlas/logs/`.

## Public Funnel route

1. Build the public frontend with `NEXT_PUBLIC_BASE_PATH=/firebird`.
2. Place the export under `public-web/funnel-20260926/` and copy `public_gateway.py` into the deployment root.
3. Create `private/public-gateway.json` for the service user, using mode `0600` and a salted password hash. Keep the generated login details outside the repository.
4. Start or restart the `firebird-public` Supervisor service on `127.0.0.1:8097`.
5. Preview through a loopback SSH tunnel at `http://127.0.0.1:18097/firebird/`.
6. Have the host administrator activate the scoped route using the command below.

Configure the gateway JSON with these fields, replacing the host, service-user path and password-derived values:

```json
{
  "public_origin": "https://host.tailnet.ts.net",
  "static_dir": "/home/service-user/ollamaforvlas/public-web/funnel-20260926",
  "username": "firebird",
  "password_salt_hex": "REPLACE_WITH_16_BYTE_SALT_AS_HEX",
  "password_hash_hex": "REPLACE_WITH_PBKDF2_SHA256_HASH_AS_HEX",
  "password_iterations": 310000,
  "preview_origins": ["http://127.0.0.1:18097"]
}
```

Generate a random 16-byte salt and derive a 32-byte PBKDF2-HMAC-SHA256 hash from the chosen password using 310000 iterations. Encode both as hexadecimal. Use an absolute path to the built public export for `static_dir`.

Activate the route on the host:

```sh
sudo tailscale funnel --bg --https=443 --set-path=/firebird http://127.0.0.1:8097/firebird
```

Open `https://<host>.<tailnet>.ts.net/firebird/`. Check login, authenticated API requests, `/firebird/docs/` and the existing root route. To rotate the gateway password, update the private configuration and restart `firebird-public`.
