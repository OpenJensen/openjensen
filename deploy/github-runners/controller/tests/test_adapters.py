import json
import time
from http import HTTPStatus
from unittest.mock import Mock, patch

import pytest
from adapters import ApiError, Compute, GitHub, Lease
from config import MAX_LIFETIME_SECONDS, PAGE_SIZE, Settings
from google.api_core.exceptions import NotFound, PreconditionFailed

SETTINGS = Settings("project", "zone", "image", "subnet", "owner/repo", "app", "install", "bucket")


def response(status=HTTPStatus.OK, payload=None):
    result = Mock(status_code=status, ok=status < HTTPStatus.BAD_REQUEST)
    result.json.return_value = payload or {}
    return result


def compute():
    with (
        patch("adapters.google.auth.default", return_value=(Mock(), "project")),
        patch("adapters.AuthorizedSession") as session,
    ):
        return Compute(SETTINGS), session.return_value


def github():
    with (
        patch("adapters.PRIVATE_KEY") as key,
        patch("adapters.jwt.encode"),
        patch("adapters.requests.Session") as session,
    ):
        key.read_text.return_value = "private"
        session.return_value.request.return_value = response(payload={"token": "installation"})
        return GitHub(SETTINGS), session.return_value


def test_worker_payload():
    driver, session = compute()
    session.post.return_value = response()
    driver.create("firebird-ci-0", "runner-name", "jit-secret")
    payload = session.post.call_args.kwargs["json"]
    assert payload["serviceAccounts"] == []
    assert payload["scheduling"]["provisioningModel"] == "STANDARD"
    assert payload["scheduling"]["maxRunDuration"]["seconds"] == str(MAX_LIFETIME_SECONDS)
    assert payload["scheduling"]["instanceTerminationAction"] == "DELETE"
    assert payload["disks"][0]["autoDelete"]
    assert payload["disks"][0]["initializeParams"]["sourceImage"] == "image"
    assert "private" not in json.dumps(payload)


def test_unmanaged_slot():
    driver, session = compute()
    session.get.return_value = response(payload={"labels": {}})
    with pytest.raises(ApiError, match="unmanaged"):
        driver.instances()
    session.delete.assert_not_called()


def test_empty_slots():
    driver, session = compute()
    session.get.return_value = response(HTTPStatus.NOT_FOUND)
    assert driver.instances() == []
    assert session.get.call_count == 3


def test_paginated_runner_list():
    driver, session = github()
    session.request.side_effect = [
        response(payload={"runners": [{}] * PAGE_SIZE}),
        response(payload={"runners": [{"id": 101}]}),
    ]
    assert len(driver.runners()) == PAGE_SIZE + 1
    assert "page=2" in session.request.call_args.args[1]


def test_partial_snapshot_rejected():
    driver, session = github()
    session.request.return_value = response(payload={"runners": [{}] * PAGE_SIZE})
    with patch("adapters.MAX_PAGES", 1), pytest.raises(ApiError, match="partial"):
        driver.runners()


@pytest.mark.parametrize(
    "status,expected",
    [
        (HTTPStatus.NO_CONTENT, True),
        (HTTPStatus.NOT_FOUND, True),
        (HTTPStatus.CONFLICT, False),
        (HTTPStatus.UNPROCESSABLE_ENTITY, False),
    ],
)
def test_retirement_races(status, expected):
    driver, session = github()
    session.delete.return_value = response(status)
    assert driver.retire(42) is expected


def test_redacted_api_error():
    driver, session = github()
    session.request.return_value = response(HTTPStatus.FORBIDDEN, {"token": "SECRET"})
    with pytest.raises(ApiError) as error:
        driver.runners()
    assert "SECRET" not in str(error.value)


def lease(blob):
    with patch("adapters.storage.Client") as client:
        client.return_value.bucket.return_value.blob.return_value = blob
        return Lease("bucket")


def test_active_lease():
    blob = Mock(generation=7)
    blob.download_as_text.return_value = json.dumps({"expires": time.time() + 100})
    lock = lease(blob)
    assert not lock.acquire()
    lock.release()
    blob.upload_from_string.assert_not_called()
    blob.delete.assert_not_called()


def test_expired_lease():
    blob = Mock(generation=7)
    blob.download_as_text.return_value = json.dumps({"expires": 0})
    lock = lease(blob)
    assert lock.acquire()
    assert blob.upload_from_string.call_args.kwargs["if_generation_match"] == 7
    lock.release()
    assert blob.delete.call_args.kwargs["if_generation_match"] == 7


def test_lease_creation_race_is_safe():
    blob = Mock()
    blob.reload.side_effect = NotFound("absent")
    blob.upload_from_string.side_effect = PreconditionFailed("lost race")
    lock = lease(blob)
    assert not lock.acquire()
    assert blob.upload_from_string.call_args.kwargs["if_generation_match"] == 0
    lock.release()
    blob.delete.assert_not_called()


def test_replaced_lease():
    blob = Mock(generation=7)
    blob.reload.side_effect = NotFound("absent")
    lock = lease(blob)
    assert lock.acquire()
    blob.delete.side_effect = PreconditionFailed("new generation")
    lock.release()
    assert blob.delete.call_args.kwargs["if_generation_match"] == 7


def test_truncated_api_is_rejected():
    driver, session = github()
    session.request.return_value = response(payload={"total_count": 300, "runners": []})
    with pytest.raises(ApiError, match="partial"):
        driver.runners()
