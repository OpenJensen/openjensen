from unittest.mock import patch

from main import app


def test_locked_request():
    with patch("main.Settings"), patch("main.Lease") as lease, patch("main.GitHub") as github:
        lease.return_value.acquire.return_value = False
        result = app.test_client().post("/reconcile")
        assert result.json == {"status": "locked"}
        github.assert_not_called()


def test_redacted_failure(caplog):
    with patch("main.Settings"), patch("main.Lease") as lease, patch("main.GitHub") as github:
        github.side_effect = RuntimeError("PRIVATE-KEY")
        result = app.test_client().post("/reconcile")
        assert result.status_code == 503
        lease.return_value.release.assert_called_once()
        assert "PRIVATE-KEY" not in caplog.text
        assert "PRIVATE-KEY" not in result.text


def test_endpoint_requires_post():
    assert app.test_client().get("/reconcile").status_code == 405


def test_redacted_release(caplog):
    with (
        patch("main.Settings"),
        patch("main.Lease") as lease,
        patch("main.GitHub"),
        patch("main.Compute"),
        patch("main.Fleet") as fleet,
    ):
        fleet.return_value.reconcile.return_value = {"started": 0}
        lease.return_value.release.side_effect = RuntimeError("PRIVATE-KEY")
        result = app.test_client().post("/reconcile")
        assert result.status_code == 200
        assert "PRIVATE-KEY" not in caplog.text
