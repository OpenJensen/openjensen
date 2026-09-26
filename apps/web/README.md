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

`next build` exports files to `out/`. Serve those with the Python application for a same-origin local install. There is no Next.js server, request-time rendering, or Server Action dependency. The only route is `/`; project selection and results are client state. No remote fonts, hosted image assets, authentication keys, or model weights are included.

`src/lib/api.generated.ts` is generated from the core OpenAPI schema. Regenerate it when the API contract changes; do not maintain a second manual schema. Layout/page remain static server components; the interactive workspace is a client component because the static export has no server runtime. TanStack Query handles current API state and polling.

Local dataset paths refer to the computer running the API, not necessarily the browser's computer. Server-configured allowed roots govern access; the local source option is disabled until the API reports `dataset.inspect.local` as available. Inspections are metadata-only; the UI does not infer successful training, video decoding, action semantics, or simulator compatibility from metadata.

Styling uses local CSS for this foundation. Tailwind and shadcn remain optional selected tools in the plan and are not installed by this client.
