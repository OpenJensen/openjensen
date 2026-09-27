import hashlib

import pytest

from firebird_decision.contracts import (
    CAVEATS,
    LICENSE,
    MODEL_SHA256,
    REPOSITORY,
    REVISION,
    TEMPLATE,
    TEMPLATE_VERSION,
    VERSIONS,
    request_hash,
)


@pytest.fixture
def request_data():
    return {
        "schema_version": 1,
        "state": "A user lost a bank card.",
        "instructions": "Select the matching intent.",
        "criteria": [
            {"id": "lost", "text": "The user lost a card."},
            {"id": "cash", "text": "The user wants cash."},
        ],
    }


@pytest.fixture
def response_data(request_data):
    return {
        "schema_version": 1,
        "model": REPOSITORY,
        "revision": REVISION,
        "model_sha256": MODEL_SHA256,
        "license": LICENSE,
        "prompt_template": TEMPLATE_VERSION,
        "prompt_template_sha256": hashlib.sha256(TEMPLATE.encode()).hexdigest(),
        "request_sha256": request_hash(request_data),
        "device": "cpu",
        "threads": 2,
        "runtime_versions": VERSIONS,
        "advisory_only": True,
        "calibrated": False,
        "selected_id": "lost",
        "scores": [
            {"id": "lost", "logit": 0.0, "relative_weight": 0.5, "tokens": 20},
            {"id": "cash", "logit": 0.0, "relative_weight": 0.5, "tokens": 20},
        ],
        "timing_ms": {"load": 123.0, "score": 2.0},
        "caveats": CAVEATS,
    }
