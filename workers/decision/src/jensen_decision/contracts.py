"""Small bounded interchange contract; importing this module never imports ML packages."""

from __future__ import annotations

import hashlib
import json
import math
import re

LICENSE = "CC-BY-NC-SA-4.0"
REVISION = "5afb8eeff127621fea2d66fc63f56798ada12eda"
REPOSITORY = "muose/Muose-50M-Decision"
MODEL_SHA256 = "3d81f0712ea1e9495a5996258dbc9a41e2fc2e1912dd87464683b20950e5c76f"
VERSIONS = {"torch": "2.11.0", "safetensors": "0.8.0", "tokenizers": "0.23.2"}
MAX_REQUEST_BYTES = 65_536
MAX_RESPONSE_BYTES = 65_536
MAX_TOKENS = 512
TEMPLATE_VERSION = "firebird-experimental-sections-v1"
TEMPLATE = "STATE:\n{state}\n\nINSTRUCTIONS:\n{instructions}\n\nCRITERION:\n{criterion}"
CAVEATS = [
    "Relative softmax weights are not calibrated correctness probabilities.",
    "Experimental Firebird prompt; not a reproduction of the upstream benchmark.",
    "Banking-tuned text scorer; out-of-domain and robotics quality are unverified.",
    "Advisory only: this output cannot authorize a job, tool, payment, or robot action.",
]


ERROR_MESSAGES = {
    "invalid_input": "Decision worker rejected invalid input.",
    "input_file": "Cannot read a bounded regular input file without symlinks.",
    "model_integrity": "Pinned model checksum or size validation failed; nothing was loaded.",
    "runtime_unavailable": "Install the worker's exact pinned CPU runtime dependencies.",
    "token_limit": "Each full criterion prompt must fit 512 tokens; nothing was truncated.",
    "worker_failed": "Decision worker failed; no result was accepted.",
}


class DecisionError(ValueError):
    """A bounded message and a stable code; raw child diagnostics never reach callers."""

    def __init__(self, message, code="invalid_input"):
        if code not in ERROR_MESSAGES:
            raise ValueError("Unknown decision error code")
        self.code = code
        super().__init__(message)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DecisionError("Duplicate JSON object keys are not allowed.")
        result[key] = value
    return result


def parse_json(raw: bytes):
    if len(raw) > MAX_REQUEST_BYTES:
        raise DecisionError("JSON input exceeds 64 KiB.")
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise DecisionError(
            "Invalid JSON; use UTF-8 without duplicate keys or non-finite values."
        ) from exc


def _text(value, field, limit):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise DecisionError(f"{field} must be nonempty text of at most {limit} characters.")
    if any(ord(char) < 32 and char not in "\n\t" for char in value):
        raise DecisionError(f"{field} contains unsupported control characters.")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise DecisionError(f"{field} must contain valid Unicode.") from exc
    return value


def validate_request(value):
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "state",
        "instructions",
        "criteria",
    }:
        raise DecisionError(
            "Request requires only schema_version, state, instructions, and criteria."
        )
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise DecisionError("Unsupported request schema_version; expected 1.")
    _text(value["state"], "state", 8_000)
    _text(value["instructions"], "instructions", 2_000)
    criteria = value["criteria"]
    if not isinstance(criteria, list) or not 2 <= len(criteria) <= 8:
        raise DecisionError("Supply between 2 and 8 criteria.")
    ids, texts = set(), set()
    for criterion in criteria:
        if not isinstance(criterion, dict) or set(criterion) != {"id", "text"}:
            raise DecisionError("Each criterion requires only id and text.")
        identifier = criterion["id"]
        if not isinstance(identifier, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", identifier
        ):
            raise DecisionError(
                "Criterion IDs must be 1-64 ASCII letters, digits, underscores, or hyphens."
            )
        text = _text(criterion["text"], "criterion text", 2_000)
        if identifier in ids or text.strip() in texts:
            raise DecisionError("Criterion IDs and texts must be unique.")
        ids.add(identifier)
        texts.add(text.strip())
    if len(canonical(value)) > MAX_REQUEST_BYTES:
        raise DecisionError("JSON input exceeds 64 KiB.")
    return value


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def request_hash(value):
    return hashlib.sha256(canonical(validate_request(value))).hexdigest()


def prompt(request, criterion):
    return TEMPLATE.format(
        state=request["state"], instructions=request["instructions"], criterion=criterion["text"]
    )


def weights(logits):
    if not logits or any(not math.isfinite(value) for value in logits):
        raise DecisionError("Model returned a non-finite score.")
    values = [math.exp(value - max(logits)) for value in logits]
    total = sum(values)
    return [value / total for value in values]


def validate_response(response, request):
    """Bind every accepted receipt to this input, implementation, and scored option set."""
    expected_fields = {
        "schema_version",
        "model",
        "revision",
        "model_sha256",
        "license",
        "prompt_template",
        "prompt_template_sha256",
        "request_sha256",
        "device",
        "threads",
        "runtime_versions",
        "advisory_only",
        "calibrated",
        "selected_id",
        "scores",
        "timing_ms",
        "caveats",
    }
    fail = "Worker response does not match the submitted request and pinned contract."
    if not isinstance(response, dict) or set(response) != expected_fields:
        raise DecisionError(fail)
    expected = {
        "schema_version": 1,
        "model": REPOSITORY,
        "revision": REVISION,
        "model_sha256": MODEL_SHA256,
        "license": LICENSE,
        "prompt_template": TEMPLATE_VERSION,
        "prompt_template_sha256": hashlib.sha256(TEMPLATE.encode()).hexdigest(),
        "request_sha256": request_hash(request),
        "device": "cpu",
        "threads": 2,
        "runtime_versions": VERSIONS,
        "caveats": CAVEATS,
    }
    if any(response[key] != value for key, value in expected.items()):
        raise DecisionError(fail)
    if (
        type(response["schema_version"]) is not int
        or type(response["threads"]) is not int
        or response["advisory_only"] is not True
        or response["calibrated"] is not False
    ):
        raise DecisionError(fail)
    scores = response["scores"]
    if not isinstance(scores, list) or len(scores) != len(request["criteria"]):
        raise DecisionError(fail)
    for score, criterion in zip(scores, request["criteria"], strict=True):
        if (
            not isinstance(score, dict)
            or set(score) != {"id", "logit", "relative_weight", "tokens"}
            or score["id"] != criterion["id"]
            or type(score["tokens"]) is not int
            or not 1 <= score["tokens"] <= MAX_TOKENS
        ):
            raise DecisionError(fail)
        for key in ("logit", "relative_weight"):
            if type(score[key]) not in (int, float) or not math.isfinite(score[key]):
                raise DecisionError(fail)
    logits = [score["logit"] for score in scores]
    expected_weights = weights(logits)
    if any(
        abs(score["relative_weight"] - expected_weight) > 1e-10
        for score, expected_weight in zip(scores, expected_weights, strict=True)
    ):
        raise DecisionError(fail)
    selected = scores[max(range(len(scores)), key=lambda index: logits[index])]["id"]
    if response["selected_id"] != selected:
        raise DecisionError(fail)
    timing = response["timing_ms"]
    if not isinstance(timing, dict) or set(timing) != {"load", "score"}:
        raise DecisionError(fail)
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0
        for value in timing.values()
    ):
        raise DecisionError(fail)
    return response
