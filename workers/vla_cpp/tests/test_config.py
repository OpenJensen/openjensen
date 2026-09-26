from pathlib import Path

from policykit.config import load_config


def test_matrix_contains_core_and_vision_candidate_rows():
    config = load_config(Path(__file__).parents[1] / "configs/benchmark.v1.yaml")
    assert config.matrix() == [
        ("smolvla", "bf16"), ("smolvla", "q8_0"), ("smolvla", "q4_0"), ("smolvla", "q8_0_vision"), ("smolvla", "q4_0_vision"),
        ("pi0", "bf16"), ("pi0", "q8_0"), ("pi0", "q4_0"), ("pi0", "q8_0_vision"), ("pi0", "q4_0_vision"),
    ]
