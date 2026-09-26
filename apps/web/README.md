# Static web client

This client uses the same Python application API as the CLI. It provides project creation/selection, public Hugging Face or allowed local-directory metadata intake, job history/polling/cancellation, and source-derived results. The six lifecycle scopes stay visible; only metadata intake is implemented here.

From the repository root, install the workspace dependencies and generate the API client before type checking or building:

```sh
pnpm install --frozen-lockfile
pnpm generate:client
pnpm dev:web
pnpm check:web
pnpm build:web
```

The Python application must run separately during development. The development client uses `http://127.0.0.1:8000/api/v1`. Production static exports use `/api/v1` on the origin serving the website. Set `NEXT_PUBLIC_API_URL` at build/development time only when a separate API origin is intended; it is public client configuration, not a place for credentials. The API host must allow that client origin.

`next build` exports files to `out/`. Serve those with the Python application for a same-origin local install. There is no Next.js server, request-time rendering, or Server Action dependency. The routes are `/` for the workspace and `/docs/` for the API reference; project selection and results are client state. No remote fonts, hosted image assets, authentication keys, or model weights are included.

`src/lib/api.generated.ts` is generated from the core OpenAPI schema. Regenerate it when the API contract changes; do not maintain a second manual schema. Layout/page remain static server components; the interactive workspace is a client component because the static export has no server runtime. TanStack Query handles current API state and polling.

Local dataset paths refer to the computer running the API, not necessarily the browser's computer. Server-configured allowed roots govern access; the local source option is disabled until the API reports `dataset.inspect.local` as available. Inspections are metadata-only; the UI does not infer successful training, video decoding, action semantics, or simulator compatibility from metadata.

Styling uses local CSS for this foundation. Tailwind and shadcn remain optional selected tools in the plan and are not installed by this client.

The workspace uses a neutral light theme by default. The appearance control switches between light and dark, remembers the choice in `firebird.theme`, and applies it before the page paints. The six lifecycle stages live in the sidebar; planned stages show their current availability while the dataset form remains mounted so navigation preserves an unfinished intake. On small screens, navigation and project controls move above the workspace.

## Keep the workspace and API reference aligned

Both routes render `WorkspaceShell`, `Icon`, and `ThemeToggle` and inherit the same `globals.css` tokens. Change branding, navigation chrome, typography, spacing, and appearance controls in those shared sources. Keep reference-specific styles limited to endpoint and schema content.

The reference reads the running server's `/openapi.json` on each visit. New operations, parameters, response codes, and models appear from that contract; there is no second hand-maintained endpoint list. It supports search, method filtering, model links, endpoint permalinks, and copyable cURL commands. Swagger/ReDoc assets and CDN dependencies are not used. The JSON schema remains available when the web export has not been built.

After changing either the main page or reference, run:

```sh
pnpm check:web
pnpm build:web
pnpm exec playwright install chromium  # once per browser version
pnpm test:web
```

Browser tests start an isolated Python API on port 8765 with a disposable workspace; they never reuse the running development application's database. They compare actual shell styles and every theme token between routes in light/dark mode and desktop/mobile layouts, verify theme persistence, exercise keyboard navigation, filters, schemas, copy behavior and error recovery, and inject a new operation to check automatic documentation updates. API tests check export routing, schema parity and access boundaries. These checks run on every push and pull request in the application CI workflow; failure traces and screenshots are uploaded for diagnosis.
