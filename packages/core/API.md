# Local application API

The web transport in `apps/web/src/lib/api.ts` and the `firebird` CLI use the same
`/api/v1` routes. Only `firebird serve` owns a workspace and schedules workers.
OpenAPI is available at `/openapi.json`; generated client records come from the
coordinated contract export, not a separate frontend model.

| Method | Route | Result |
| --- | --- | --- |
| GET | `/api/v1/health` | Application status and version |
| GET | `/api/v1/capabilities` | Runnable and explicitly planned operations |
| GET / POST | `/api/v1/projects` | List projects / create one (201) |
| GET | `/api/v1/projects/{id}/jobs` | Current persisted project jobs |
| POST | `/api/v1/projects/{id}/intakes` | Validated intake, initially queued (202) |
| GET | `/api/v1/jobs/{id}` | Current persisted job |
| POST | `/api/v1/jobs/{id}/cancel` | Cancel active work; terminal jobs stay unchanged |

Poll the project jobs or individual job route while status is `queued` or
`running`; stop at `succeeded`, `failed`, `cancelled`, or `interrupted`. Responses
under `/api/v1/` use `Cache-Control: no-store` so caches do not hide state changes.
Polling returns snapshots: intermediate transitions may occur between polls.
There is no percentage estimate, durable event sequence, SSE endpoint, or
`Last-Event-ID` replay. Resumable SSE remains an explicit future gap.

Invalid input returns 422 with FastAPI validation details (or a string explaining
disabled local intake); missing project/job IDs return 404 with a string `detail`.
Worker failures remain terminal job records with `error` and no successful result.
Submitting work is not a claim that inspection succeeded. Only local/Hugging Face
metadata intake is accepted by this endpoint; training and other planned stages
have no runnable route.

`firebird serve` binds to `127.0.0.1`. Host validation and browser Origin checks
restrict local access. The documented development origins on ports 3000/8000 and
the server's own origin are allowed; other origins, the opaque `null` origin, and
cross-site browser requests without an Origin are rejected before mutations.
CLI requests without browser headers are allowed. Development CORS permits GET,
POST, and the Content-Type header. This is a local application boundary, not
authentication for remote hosting.

Set `FIREBIRD_LOCAL_DATA_ROOT` before starting the owner to enable local intake.
The dataset reader enforces that configured root. CPU metadata checks do not
establish device support, training compatibility, or verified action semantics.

Run `uv run --frozen pytest -q tests/test_api.py` with repository Node 24 on PATH.
The integration check starts a real loopback owner, executes the unchanged web
TypeScript transport with Node's native type stripping, invokes real CLI
subprocesses, and polls a real metadata worker using synthetic local metadata.
This is transport/fixture evidence, not browser rendering, live-source, or GPU
evidence. The web-client test explicitly skips when Node is absent.
