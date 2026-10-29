"""Unified private job and compute routes retain the gateway access boundary."""

import httpx
import pytest
from starlette.testclient import TestClient
from test_public_gateway import AUTH, ORIGIN, gateway
from test_public_gateway import config as config


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/projects/project/work-jobs",
        "/api/v1/projects/project/work-jobs/core:job",
        "/api/v1/projects/project/work-jobs/core:job/logs",
        "/api/v1/projects/project/work-jobs/core:job/failure",
        "/api/v1/projects/project/work-jobs/core:job/diagnostics.zip",
        "/api/v1/compute-targets",
        "/api/v1/compute-targets/target/check",
        "/api/v1/projects/project/%77ork-jobs/core:job/logs",
        "/api/v1/projects/project/%2577ork-jobs/core:job/logs",
        "/api/v1/%63ompute-targets",
    ],
)
@pytest.mark.parametrize("method", ["GET", "POST"])
def test_private_job_and_compute_routes_never_reach_public_upstream(config, path, method):
    def forbidden(request):
        raise AssertionError("Private workspace routes must not reach the public upstream")

    with TestClient(
        gateway.PublicGateway(config[0], transport=httpx.MockTransport(forbidden)),
        base_url=ORIGIN,
        headers={"Authorization": AUTH, "Origin": ORIGIN},
    ) as client:
        response = client.request(method, "/firebird" + path)
        assert response.status_code == 403
        assert "private workspace" in response.text
        assert response.headers["cache-control"] == "no-store"
