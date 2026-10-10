# Run the OPEN JENSEN web client

Use the repository's pinned Node and pnpm versions. Complete the [application setup](../../docs/getting-started.md), then start the API in one terminal:

```sh
uv run --frozen firebird serve
```

## Develop

In a second terminal at the repository root:

```sh
pnpm install --frozen-lockfile
pnpm generate:client
pnpm dev:web
```

Open [localhost:3000](http://127.0.0.1:3000). Development requests use the API at `http://127.0.0.1:8000/api/v1`. Set `NEXT_PUBLIC_API_URL` before starting or building the frontend to use another API origin.

## Build and serve

```sh
pnpm build:web
uv run --frozen firebird serve
```

The export is written to `apps/web/out/`. Open [localhost:8000](http://127.0.0.1:8000); use `/guide/` for workspace instructions and `/docs/` for the API reference.

For a prefixed installation, set `NEXT_PUBLIC_BASE_PATH` during the build:

```sh
NEXT_PUBLIC_BASE_PATH=/firebird pnpm build:web
```

## Keep the workspace and API reference aligned

Edit shared layout and styling in `WorkspaceShell`, `Icon`, `ThemeToggle` and `globals.css`. Update section pages under `src/app/(workspace)/` and their components under `src/components/workspace-sections/`.

After changing the API, regenerate the schema and client from the repository root:

```sh
uv run --frozen python scripts/export_openapi.py
pnpm generate:client
```

Run the web checks:

```sh
pnpm check:web
pnpm build:web
pnpm exec playwright install chromium
pnpm test:web
```

See [development instructions](../../docs/workbench-design.md) and [local checks](../../docs/ci.md).
