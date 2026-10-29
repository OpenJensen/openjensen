"""Static route prefetches must receive component payloads, never HTML."""

import pytest
from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.routing import Mount
from vla_platform.api import create_app
from vla_platform.settings import Settings
from vla_platform.web_assets import WorkspaceStaticFiles


@pytest.fixture
def exported(tmp_path):
    root = tmp_path / "export"
    page = root / "settings"
    page.mkdir(parents=True)
    (page / "index.html").write_text("<html>Settings</html>")
    (page / "index.txt").write_text('0:{"page":"settings"}')
    (page / "__next._tree.txt").write_text('0:{"tree":"settings"}')
    (page / "README.txt").write_text("Ordinary text")
    (root / "app.js").write_text("window.example=true")
    return root


@pytest.mark.parametrize("prefix", ["", "/workbench"])
def test_route_headers_distinguish_html_and_component_payloads(exported, prefix):
    app = Starlette(
        routes=[Mount(prefix or "/", app=WorkspaceStaticFiles(directory=exported, html=True))]
    )
    with TestClient(app) as client:
        html = client.get(prefix + "/settings/")
        assert html.status_code == 200 and html.text == "<html>Settings</html>"
        assert html.headers["content-type"].startswith("text/html")
        assert "RSC" in html.headers["vary"]
        payload = client.get(prefix + "/settings/?_rsc=fixture", headers={"RSC": "1"})
        assert payload.status_code == 200 and payload.text == '0:{"page":"settings"}'
        assert payload.headers["content-type"].startswith("text/x-component")
        assert payload.headers["x-nextjs-rewritten-path"] == "/settings/"
        assert "RSC" in payload.headers["vary"]
        for name in ["index.txt", "__next._tree.txt"]:
            response = client.get(prefix + "/settings/" + name, headers={"RSC": "1"})
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/x-component")
            assert response.headers["x-nextjs-rewritten-path"] == "/settings/"
        assert client.head(prefix + "/settings/", headers={"RSC": "1"}).content == b""
        ordinary = client.get(prefix + "/settings/README.txt")
        assert ordinary.text == "Ordinary text" and ordinary.headers["content-type"].startswith(
            "text/plain"
        )
        assert client.get(prefix + "/missing/", headers={"RSC": "1"}).status_code == 404


def test_conditional_component_requests_keep_valid_etags(exported):
    with TestClient(
        Starlette(routes=[Mount("/", app=WorkspaceStaticFiles(directory=exported, html=True))])
    ) as client:
        response = client.get("/settings/", headers={"RSC": "1"})
        cached = client.get(
            "/settings/", headers={"RSC": "1", "If-None-Match": response.headers["etag"]}
        )
        assert cached.status_code == 304
        assert cached.headers["content-type"].startswith("text/x-component")
        assert cached.headers["x-nextjs-rewritten-path"] == "/settings/"


def test_component_headers_do_not_change_api_responses(exported, tmp_path):
    with TestClient(
        create_app(Settings(data_dir=tmp_path / "workspace", static_dir=exported))
    ) as client:
        response = client.get("/api/v1/health", headers={"RSC": "1"})
        assert response.status_code == 200 and response.json()["status"] == "ok"
        assert response.headers["content-type"].startswith("application/json")
        assert response.headers["cache-control"] == "no-store"


def test_component_payloads_cannot_escape_the_export_root(exported, tmp_path):
    outside = tmp_path / "private"
    outside.mkdir()
    (outside / "index.txt").write_text("private content")
    (exported / "escape").symlink_to(outside, target_is_directory=True)
    with TestClient(
        Starlette(routes=[Mount("/", app=WorkspaceStaticFiles(directory=exported, html=True))])
    ) as client:
        response = client.get("/escape/", headers={"RSC": "1"})
        assert response.status_code == 404 and "private content" not in response.text
