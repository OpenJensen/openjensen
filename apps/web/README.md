# OPEN JENSEN web client

The static web client uses the same Python application API as the CLI. It provides project and dataset management, fine-tuning, distillation, quantization, evaluation, simulation and replay, augmentation, teaching, model history, cloud runs, and compute settings. Workflows require their corresponding configured workers or provider connections; inspection and inference results retain their documented scope.

From the repository root, install the workspace dependencies and generate the API client before type checking or building:

```sh
pnpm install --frozen-lockfile
pnpm generate:client
pnpm dev:web
pnpm check:web
pnpm build:web
```

The Python application must run separately during development. The development client uses `http://127.0.0.1:8000/api/v1`. Production static exports use `/api/v1` on the origin serving the website. Set `NEXT_PUBLIC_API_URL` at build/development time only when a separate API origin is intended; it is public client configuration, not a place for credentials. The API host must allow that client origin.

`next build` exports files to `out/`. Serve those with the Python application for a same-origin local install. There is no Next.js server, request-time rendering, or Server Action dependency. `/` opens `/dashboard/`; `/docs/` remains the API reference and `/guide/` the workspace guide. No remote fonts, hosted image assets, authentication keys, or model weights are included.

Every workspace destination has an exported route:

| Route | Destination |
| --- | --- |
| `/dashboard/` | Dashboard |
| `/datasets/` | Dataset library, imports and inspection |
| `/training/` | Fine-tuning setup and saved runs |
| `/distillation/` | Distill |
| `/quantization/` | Quantize |
| `/evaluation/` | Evaluate |
| `/simulation/` | Run: simulation, replay and inference checks |
| `/augmentation/` | Augmentation |
| `/teaching/` | Teaching |
| `/decision-lab/` | Decision lab |
| `/models/` | My models |
| `/cloud-runs/` | Cloud runs |
| `/settings/` | Compute, workflow settings and diagnostics |

Sidebar destinations are normal links, with client navigation, browser Back/Forward and new-tab support. Direct visits and refresh load the same section. The Python static-file mount serves the actual exported `section/index.html` files and redirects slashless section URLs while retaining their query string; unknown paths remain 404s. All routes also support `NEXT_PUBLIC_BASE_PATH`.

The `(workspace)` layout owns the shell, project selection, React Query cache and workflow handoffs. Section bodies live in `components/workspace-sections/`; the URL determines the active section. Dataset intake mounts on the first dataset visit and remains hidden during other workspace visits so drafts, uploads and pending-request guards survive navigation. Training remains mounted during a compute-settings detour, with inactive polling disabled. Other sections mount through their own page modules. A full refresh restores the section and the saved project/theme; unsaved form drafts remain session state. Navigating never submits or starts work.

`src/lib/api.generated.ts` is generated from the core OpenAPI schema. Regenerate it when the API contract changes; do not maintain a second manual schema. Layout/page remain static server components; the interactive workspace is a client component because the static export has no server runtime. TanStack Query handles current API state and polling.

Local dataset paths refer to the computer running the API, not necessarily the browser's computer. Server-configured allowed roots govern access; the local source option is disabled until the API reports `dataset.inspect.local` as available. Inspections are metadata-only; the UI does not infer successful training, video decoding, action semantics, or simulator compatibility from metadata.

Styling uses local CSS shared by the workspace, guide, and API reference.

The workspace uses a neutral light theme by default. The appearance control switches between light and dark, remembers the choice in `firebird.theme`, and applies it before the page paints. The grouped sidebar links to each workspace destination. On small screens, navigation and project controls move above the workspace.

## Keep the workspace and API reference aligned

Workspace routes, the guide and API reference render `WorkspaceShell`, `Icon`, and `ThemeToggle` and inherit the same `globals.css` tokens. Change branding, navigation chrome, typography, spacing, and appearance controls in those shared sources. Keep reference-specific styles limited to endpoint and schema content.

The reference reads the running server's `/openapi.json` on each visit. New operations, parameters, response codes, and models appear from that contract; there is no second hand-maintained endpoint list. It supports search, method filtering, model links, endpoint permalinks, and copyable cURL commands. Swagger/ReDoc assets and CDN dependencies are not used. The JSON schema remains available when the web export has not been built.

After changing either the main page or reference, run:

```sh
pnpm check:web
pnpm build:web
pnpm exec playwright install chromium  # once per browser version
pnpm test:web
```

Browser tests start an isolated Python API on port 8765 with a disposable workspace; they never reuse the running development application's database. They compare actual shell styles and every theme token between routes in light/dark mode and desktop/mobile layouts, verify theme persistence, exercise keyboard navigation, filters, schemas, copy behavior and error recovery, and inject a new operation to check automatic documentation updates. API tests check export routing, schema parity and access boundaries. Run the applicable checks locally before publishing changes and retain their source commit and results. Hosted GitHub Actions is unavailable for this project; checked-in workflows document the intended checks and are not an acceptance gate.

## Visual dataset explorer

Sources provides real, attributed stills for two pinned starters. Inspection opens a full-width dataset overview and an explicit visual-preview action. The explorer uses generated episode API types, retains selected-episode state across workspace views, and pauses hidden media. Videos use source camera dimensions and episode-relative shared controls; v3 endpoints are treated as exclusive to avoid showing the following episode. Native source codec support is required in the browser.

The first-frame table shows a bounded sample of actual selected-episode state/action values, with source channel names and display rounding. The original metadata-only warnings are retained under inspection provenance and distinguished from the additional preview operation.
