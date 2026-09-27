"""Real model is an explicit local opt-in, never fetched by tests or required for core CI."""

import json
import os
import sys
from pathlib import Path

import pytest

from firebird_decision.contracts import LICENSE, canonical
from firebird_decision.supervisor import run_owned


@pytest.mark.skipif(
    not os.environ.get("FIREBIRD_DECISION_MODEL"), reason="Explicit local model opt-in required"
)
def test_pinned_real_model_offline(tmp_path):
    assert os.environ.get("FIREBIRD_DECISION_ACCEPT_LICENSE") == LICENSE, (
        "Set FIREBIRD_DECISION_ACCEPT_LICENSE=CC-BY-NC-SA-4.0 to opt in explicitly."
    )
    root = Path(__file__).parents[1]
    raw = run_owned(
        [
            sys.executable,
            str(Path(__file__).with_name("real_probe.py")),
            os.environ["FIREBIRD_DECISION_MODEL"],
            str(root / "fixtures/scoring-v1.json"),
        ],
        timeout=120,
    )
    report = json.loads(raw)
    assert report["sample_count"] == 12
    assert report["source_unchanged"] and report["deterministic_repeat"]
    assert report["criterion_order_invariant"] and report["oversized_tokens_rejected"]
    output = os.environ.get("FIREBIRD_DECISION_RECEIPT")
    if output:
        with Path(output).open("xb") as stream:
            stream.write(raw)
    # Also exercise the actual public supervised CLI, not only the direct scorer.
    request = json.loads((root / "fixtures/scoring-v1.json").read_text())["cases"][0]["request"]
    path = tmp_path / "request.json"
    path.write_bytes(canonical(request))
    raw_cli = run_owned(
        [
            sys.executable,
            "-m",
            "firebird_decision",
            "--model-dir",
            os.environ["FIREBIRD_DECISION_MODEL"],
            "--request",
            str(path),
            "--accept-license",
            LICENSE,
        ],
        timeout=40,
    )
    response = json.loads(raw_cli)
    assert response["scores"] == report["results"][0]["response"]["scores"]
