import pytest
from compare_results import paired_quality


def episode(task, success, seed=42):
    return dict(task=task, init_state=0, seed=seed, noise_seed=42, success=success)


def test_pairs_by_identity_and_exposes_opposite_outcomes():
    reference = [episode(0, True), episode(1, False)]
    candidate = [episode(1, True), episode(0, False)]
    result = paired_quality(reference, candidate)
    assert result["candidate_only"] == result["reference_only"] == 1
    assert result["both_succeed"] == result["both_fail"] == 0


def test_different_seeds_and_duplicate_episodes_are_rejected():
    with pytest.raises(ValueError, match="identical"):
        paired_quality([episode(0, True)], [episode(0, True, seed=43)])
    with pytest.raises(ValueError, match="Duplicate"):
        paired_quality([episode(0, True)] * 2, [episode(0, True)])
