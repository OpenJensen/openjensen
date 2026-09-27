import copy

import pytest

from firebird_decision.contracts import (
    DecisionError,
    canonical,
    parse_json,
    request_hash,
    validate_request,
    validate_response,
    weights,
)


@pytest.mark.parametrize(
    "raw",
    [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b"\xff", b"[", b"[" * 2000, b" " * 65537],
)
def test_rejects_malformed_or_unbounded_json(raw):
    with pytest.raises(DecisionError):
        parse_json(raw)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("state", ""),
        ("state", "x" * 8001),
        ("state", "x\x1b[31m"),
        ("state", "\ud800"),
        ("instructions", 2),
        ("criteria", []),
        ("criteria", [{"id": "x", "text": "hi"}] * 2),
        ("criteria", [{"id": "../x", "text": "hi"}, {"id": "ok", "text": "other"}]),
        ("criteria", [{"id": "a", "text": "same"}, {"id": "b", "text": "same "}]),
    ],
)
def test_invalid_requests(field, value, request_data):
    request_data[field] = value
    with pytest.raises(DecisionError):
        validate_request(request_data)


def test_contract_preserves_text_and_binds_hash(request_data):
    assert parse_json(canonical(validate_request(request_data))) == request_data
    initial = request_hash(request_data)
    request_data["criteria"].reverse()
    assert request_hash(request_data) != initial
    request_data["extra"] = "not allowed"
    with pytest.raises(DecisionError):
        validate_request(request_data)


def test_response_complete_identity_and_ties(response_data, request_data):
    assert validate_response(response_data, request_data) == response_data
    changed = copy.deepcopy(request_data)
    changed["state"] = "A different request"
    with pytest.raises(DecisionError):
        validate_response(response_data, changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "main"),
        ("model_sha256", "0" * 64),
        ("schema_version", True),
        ("threads", True),
        ("selected_id", "cash"),
        ("calibrated", True),
        ("advisory_only", False),
        ("timing_ms", {"load": float("nan"), "score": 0}),
        ("scores", []),
    ],
)
def test_rejects_invalid_receipt(field, value, response_data, request_data):
    response_data[field] = value
    with pytest.raises(DecisionError):
        validate_response(response_data, request_data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("logit", float("nan")),
        ("logit", True),
        ("relative_weight", 1.0),
        ("tokens", 513),
        ("tokens", True),
        ("id", "wrong"),
    ],
)
def test_rejects_invalid_score(field, value, response_data, request_data):
    response_data["scores"][0][field] = value
    with pytest.raises(DecisionError):
        validate_response(response_data, request_data)


def test_stable_softmax_and_finite_guard():
    assert weights([10000.0, 10000.0]) == [0.5, 0.5]
    with pytest.raises(DecisionError):
        weights([float("inf"), 2])


@pytest.mark.parametrize("encoding", ["utf-16", "utf-32"])
def test_interchange_rejects_non_utf8_encoding(encoding):
    with pytest.raises(DecisionError):
        parse_json('{"schema_version": 1}'.encode(encoding))
