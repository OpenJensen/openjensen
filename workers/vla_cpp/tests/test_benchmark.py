from pathlib import Path

from policykit.benchmark import BenchmarkRunner, Result
from policykit.config import load_config


def test_eval_command_carries_task_seed_and_action_chunking():
    config = load_config(Path(__file__).parents[1] / "configs/benchmark.v1.yaml")
    command = BenchmarkRunner(config, "test").evaluation_command(Result("pi0", "q4_0"), task_id=7, seed=314, episodes=1, port=5555)
    assert command[0].endswith("/libero_uv/.venv/bin/python")
    assert command[command.index("--task-id") + 1] == "7"
    assert command[command.index("--seed") + 1] == "314"
    assert command[command.index("--n-action-steps") + 1] == "50"
