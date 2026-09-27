"""Opt-in real offline probe; supervisor in test_real.py supplies a 120-second wall bound."""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import statistics
import sys
from pathlib import Path

from firebird_decision.contracts import LICENSE, DecisionError, canonical, validate_response
from firebird_decision.integrity import FILES, read_file
from firebird_decision.scorer import DecisionScorer, offline_guard


def inventory(root):
    return {
        name: hashlib.sha256(read_file(root / name, size, exact_size=size)).hexdigest()
        for name, (size, _) in FILES.items()
    }


def run(root, fixture):
    before = inventory(root)
    fixture_raw = read_file(fixture, 65_536)
    cases = json.loads(fixture_raw)["cases"]
    offline_guard()
    scorer = DecisionScorer(root, LICENSE)
    warm = scorer.score(cases[0]["request"])
    results = []
    for case in cases:
        response = scorer.score(case["request"])
        validate_response(response, case["request"])
        results.append(
            {
                "id": case["id"],
                "domain": case["domain"],
                "expected_id": case["expected_id"],
                "correct": response["selected_id"] == case["expected_id"],
                "response": response,
            }
        )
    assert warm["scores"] == results[0]["response"]["scores"]
    permuted = copy.deepcopy(cases[0]["request"])
    permuted["criteria"].reverse()
    permutation = scorer.score(permuted)
    assert {x["id"]: x for x in permutation["scores"]} == {x["id"]: x for x in warm["scores"]}
    assert permutation["selected_id"] == warm["selected_id"]
    too_long = copy.deepcopy(cases[0]["request"])
    too_long["state"] = "word " * 1000
    try:
        scorer.score(too_long)
    except DecisionError as exc:
        assert "512 tokens" in str(exc)
    else:
        raise AssertionError("Oversized prompt was accepted")
    # Verify every checksum again, after execution, without modifying the model source.
    after = inventory(root)
    assert before == after
    timings = sorted(result["response"]["timing_ms"]["score"] for result in results)
    domains = sorted({result["domain"] for result in results})
    return {
        "schema_version": 1,
        "fixture_sha256": hashlib.sha256(fixture_raw).hexdigest(),
        "platform": "-".join((platform.system(), platform.release(), platform.machine())),
        "python": platform.python_version(),
        "model_inventory_before": before,
        "model_inventory_after": after,
        "source_unchanged": True,
        "network_guard": "Python audit only, not OS sandbox",
        "load_ms": scorer.load_ms,
        "warmup_excluded_score_ms": warm["timing_ms"]["score"],
        "sample_count": len(results),
        "warm_score_median_ms": statistics.median(timings),
        "warm_score_p95_nearest_rank_ms": timings[-1],
        "accuracy_by_domain": {
            domain: {
                "correct": sum(r["correct"] for r in results if r["domain"] == domain),
                "total": sum(r["domain"] == domain for r in results),
            }
            for domain in domains
        },
        "deterministic_repeat": True,
        "criterion_order_invariant": True,
        "oversized_tokens_rejected": True,
        "quality_scope": (
            "Small hand-authored fixture only; not upstream benchmark or robotics acceptance."
        ),
        "results": results,
    }


if __name__ == "__main__":
    print(canonical(run(Path(sys.argv[1]), Path(sys.argv[2]))).decode())
