from __future__ import annotations

from typing import Any


def choose_candidates(results: list[dict[str, Any]], max_success_drop_pp: float = 5.0) -> dict[str, dict[str, Any]]:
    """Select the smallest successful artifact that retains BF16 task quality.

    Results from different policy architectures are never compared; each model has
    its own BF16 reference.  An incomplete/failed candidate is evidence, not a
    deployable artifact.
    """
    selected: dict[str, dict[str, Any]] = {}
    models = {item["model"] for item in results}
    for model in models:
        group = [item for item in results if item["model"] == model]
        baseline = next((item for item in group if item["preset"] == "bf16" and item.get("status") == "complete"), None)
        if not baseline or baseline.get("success_rate") is None:
            selected[model] = {"status": "blocked", "reason": "BF16 reference did not complete", "selected_preset": None}
            continue
        cutoff = float(baseline["success_rate"]) - max_success_drop_pp / 100
        eligible = [
            item for item in group
            if item.get("status") == "complete"
            and item.get("success_rate") is not None
            and float(item["success_rate"]) >= cutoff
            and item.get("artifact_bytes") is not None
        ]
        winner = min(eligible, key=lambda item: (int(item["artifact_bytes"]), item.get("p50_ms") or float("inf"))) if eligible else baseline
        selected[model] = {
            "status": "selected",
            "selected_preset": winner["preset"],
            "artifact": winner.get("artifact"),
            "success_rate": winner.get("success_rate"),
            "bf16_success_rate": baseline["success_rate"],
            "max_success_drop_pp": max_success_drop_pp,
            "reason": "smallest candidate within BF16 quality gate" if winner is not baseline else "BF16 fallback: no compressed candidate passed the quality gate",
        }
    return selected
