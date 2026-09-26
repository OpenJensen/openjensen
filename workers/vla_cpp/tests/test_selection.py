from policykit.selection import choose_candidates


def test_selects_smallest_candidate_within_bf16_quality_gate():
    results = [
        {"model": "smolvla", "preset": "bf16", "status": "complete", "success_rate": 0.80, "artifact_bytes": 1000, "artifact": "bf16.gguf"},
        {"model": "smolvla", "preset": "q8_0", "status": "complete", "success_rate": 0.78, "artifact_bytes": 600, "artifact": "q8.gguf"},
        {"model": "smolvla", "preset": "q4_0", "status": "complete", "success_rate": 0.70, "artifact_bytes": 400, "artifact": "q4.gguf"},
    ]
    winner = choose_candidates(results)["smolvla"]
    assert winner["selected_preset"] == "q8_0"


def test_uses_bf16_fallback_when_compressed_candidates_fail():
    results = [{"model": "pi0", "preset": "bf16", "status": "complete", "success_rate": 0.5, "artifact_bytes": 1000}]
    assert choose_candidates(results)["pi0"]["selected_preset"] == "bf16"
