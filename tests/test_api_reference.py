import json
from pathlib import Path

import pytest
from fastapi.routing import APIRoute, iter_route_contexts
from fastapi.testclient import TestClient
from vla_platform.api import create_app
from vla_platform.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options", "trace"}


@pytest.fixture
def exported_web(tmp_path: Path) -> Path:
    """Model Next's static export without requiring a Node build for Python tests."""
    directory = tmp_path / "web"
    docs = directory / "docs"
    assets = directory / "_next" / "static"
    docs.mkdir(parents=True)
    assets.mkdir(parents=True)
    shared_stylesheet = '<link rel="stylesheet" href="/_next/static/site.css">'
    (directory / "index.html").write_text(
        f"<!doctype html><html><head>{shared_stylesheet}</head>"
        "<body><h1>Firebird workspace</h1></body></html>",
        encoding="utf-8",
    )
    (docs / "index.html").write_text(
        f"<!doctype html><html><head>{shared_stylesheet}</head>"
        "<body><h1>Firebird API reference</h1></body></html>",
        encoding="utf-8",
    )
    for section in (
        "dashboard",
        "datasets",
        "training",
        "distillation",
        "quantization",
        "evaluation",
        "simulation",
        "settings",
        "cloud-runs",
        "augmentation",
        "teaching",
        "decision-lab",
        "models",
    ):
        page = directory / section
        page.mkdir()
        (page / "index.html").write_text(
            f"<!doctype html><html><head>{shared_stylesheet}</head>"
            f"<body><h1>{section}</h1></body></html>",
            encoding="utf-8",
        )
    (assets / "site.css").write_text("body { color: #eef0f3; }", encoding="utf-8")
    # The API's schema must take precedence over even a stale exported copy.
    (directory / "openapi.json").write_text('{"stale": true}', encoding="utf-8")
    return directory


@pytest.fixture
def web_client(tmp_path: Path, exported_web: Path):
    with TestClient(
        create_app(Settings(data_dir=tmp_path / "workspace", static_dir=exported_web))
    ) as client:
        yield client


@pytest.mark.parametrize(
    ("url", "exported_file"),
    [("/", "dashboard/index.html"), ("/docs/", "docs/index.html")],
)
def test_home_and_api_reference_serve_the_web_export(web_client, exported_web, url, exported_file):
    response = web_client.get(url)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.text == (exported_web / exported_file).read_text(encoding="utf-8")
    assert "swagger-ui" not in response.text.lower()
    assert "cdn.jsdelivr.net" not in response.text.lower()


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_home_redirects_before_rendering_and_preserves_query(web_client, method):
    response = web_client.request(method, "/?project=example", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "http://testserver/dashboard/?project=example"


def test_home_redirect_preserves_proxy_mount_prefix(tmp_path, exported_web):
    app = create_app(Settings(data_dir=tmp_path / "workspace", static_dir=exported_web))
    with TestClient(app, root_path="/firebird") as client:
        response = client.get("/firebird/?project=example", follow_redirects=False)

        assert response.status_code == 307
        assert (
            response.headers["location"] == "http://testserver/firebird/dashboard/?project=example"
        )
        assert client.get(response.headers["location"]).status_code == 200


def test_api_reference_redirect_preserves_query_string(web_client):
    response = web_client.get("/docs?search=projects", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "http://testserver/docs/?search=projects"
    assert web_client.get(response.headers["location"]).status_code == 200


@pytest.mark.parametrize(
    "section",
    [
        "dashboard",
        "datasets",
        "training",
        "distillation",
        "quantization",
        "evaluation",
        "simulation",
        "settings",
        "cloud-runs",
        "augmentation",
        "teaching",
        "decision-lab",
        "models",
    ],
)
def test_workspace_sections_serve_exported_pages_and_preserve_redirect_queries(
    web_client,
    exported_web,
    section,
):
    redirect = web_client.get(f"/{section}?project=example", follow_redirects=False)
    assert redirect.status_code == 307
    assert redirect.headers["location"] == f"http://testserver/{section}/?project=example"
    response = web_client.get(redirect.headers["location"])
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.text == (exported_web / section / "index.html").read_text()
    assert web_client.get(f"/{section}/missing-page/").status_code == 404


def test_shared_web_assets_are_available_from_api_reference(web_client, exported_web):
    response = web_client.get("/_next/static/site.css")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")
    assert response.text == (exported_web / "_next/static/site.css").read_text(encoding="utf-8")


@pytest.mark.parametrize("url", ["/redoc", "/docs/oauth2-redirect", "/docs/missing-page"])
def test_no_legacy_documentation_or_fallback_page(web_client, url):
    assert web_client.get(url).status_code == 404


@pytest.mark.parametrize("url", ["/docs", "/docs/", "/redoc", "/docs/oauth2-redirect"])
def test_documentation_requires_the_web_export(tmp_path, url):
    with TestClient(create_app(Settings(data_dir=tmp_path / "workspace"))) as client:
        response = client.get(url)

    assert response.status_code == 404
    assert "swagger-ui" not in response.text.lower()


@pytest.mark.parametrize("url", ["/", "/docs/", "/openapi.json"])
def test_reference_keeps_the_local_host_boundary(web_client, url):
    assert web_client.get(url, headers={"Host": "attacker.invalid"}).status_code == 400


@pytest.mark.parametrize("url", ["/", "/docs/", "/openapi.json"])
def test_reference_keeps_the_local_origin_boundary(web_client, url):
    assert web_client.get(url, headers={"Origin": "https://attacker.invalid"}).status_code == 403
    assert web_client.get(url, headers={"Origin": "http://testserver"}).status_code == 200


def test_runtime_schema_takes_precedence_over_exported_files(web_client):
    response = web_client.get("/openapi.json")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == web_client.app.openapi()
    assert "stale" not in response.json()


def test_api_routes_work_with_the_web_export_mounted(web_client):
    assert web_client.get("/api/v1/health").json()["status"] == "ok"
    created = web_client.post("/api/v1/projects", json={"name": "Documentation smoke test"})

    assert created.status_code == 201
    assert web_client.get("/api/v1/projects").json() == [created.json()]


def test_openapi_remains_available_without_a_web_build(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path / "workspace"))) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json() == client.app.openapi()


def test_web_mount_does_not_change_api_contract(tmp_path, exported_web):
    without_web = create_app(Settings(data_dir=tmp_path / "without-web"))
    with_web = create_app(Settings(data_dir=tmp_path / "with-web", static_dir=exported_web))

    assert with_web.openapi() == without_web.openapi()
    assert all(path.startswith("/api/v1/") for path in with_web.openapi()["paths"])


def test_checked_in_openapi_matches_runtime_contract(tmp_path):
    schema = create_app(Settings(data_dir=tmp_path / "workspace")).openapi()
    checked_in = json.loads((ROOT / "packages/core/openapi.json").read_text(encoding="utf-8"))

    assert schema == checked_in, (
        "The web reference's OpenAPI contract is stale. Run "
        "uv run python scripts/export_openapi.py and pnpm generate:client, "
        "then include the generated schema and TypeScript changes."
    )


def test_openapi_covers_every_public_api_route_and_method(tmp_path):
    app = create_app(Settings(data_dir=tmp_path / "workspace"))
    registered = {
        (route.path, method.lower())
        for route in iter_route_contexts(app.routes)
        if isinstance(route.original_route, APIRoute) and route.path.startswith("/api/")
        for method in route.methods
    }
    documented = {
        (path, method)
        for path, operations in app.openapi()["paths"].items()
        for method in operations
        if method in HTTP_METHODS
    }

    assert registered, "The reference must document a nonempty public API."
    assert documented == registered
    operation_ids = [
        operation["operationId"]
        for operations in app.openapi()["paths"].values()
        for method, operation in operations.items()
        if method in HTTP_METHODS
    ]
    assert len(operation_ids) == len(set(operation_ids)), "Operation links must remain unique."


def test_openapi_schema_references_resolve(tmp_path):
    schema = create_app(Settings(data_dir=tmp_path / "workspace")).openapi()

    def validate_references(value):
        if isinstance(value, dict):
            if "$ref" in value:
                reference = value["$ref"]
                assert reference.startswith("#/"), "The local reference must be self-contained."
                target = schema
                for key in reference[2:].split("/"):
                    target = target[key.replace("~1", "/").replace("~0", "~")]
                assert isinstance(target, dict), f"Invalid schema reference: {reference}"
            for nested in value.values():
                validate_references(nested)
        elif isinstance(value, list):
            for nested in value:
                validate_references(nested)

    validate_references(schema)
