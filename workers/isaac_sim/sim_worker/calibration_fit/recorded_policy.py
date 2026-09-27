"""An immutable recorded episode behind the existing rollout Policy contract."""

from sim_worker.rollout.config import number
from sim_worker.rollout.contracts import ActionChunk, Observation


class RecordedActions:
    def __init__(self, rows, source_episode: int, source_id: str, chunk_size: int):
        if type(source_episode) is not int or source_episode < 0:
            raise ValueError("Source episode must be a nonnegative integer")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("Recorded actions need a source/model ID")
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("Recorded action chunk size must be positive")
        selected = [row for row in rows if row["episode_index"] == source_episode]
        if (
            not selected
            or any(
                type(row[key]) is not int
                for row in selected
                for key in ("episode_index", "frame_index")
            )
            or [row["frame_index"] for row in selected] != list(range(len(selected)))
        ):
            raise ValueError("Source episode must contain contiguous frames starting at zero")
        actions = []
        for row in selected:
            action = row["action"]
            if not isinstance(action, (list, tuple)) or not action:
                raise ValueError("Each recorded action must be a nonempty vector")
            values = tuple(number(value, "recorded action") for value in action)
            if actions and len(values) != len(actions[0]):
                raise ValueError("Recorded action dimensions differ")
            actions.append(values)
        self._actions = tuple(actions)
        self._source_id = source_id
        self._chunk_size = chunk_size
        self._episode = None
        self._last_step = None

    def reset(self, episode_id: str) -> None:
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("Rollout episode ID must be nonempty")
        self._episode = episode_id
        self._last_step = None

    def predict(self, observation: Observation) -> ActionChunk:
        if self._episode is None or observation.episode_id != self._episode:
            raise ValueError("Reset the policy for the current rollout episode")
        step = observation.step
        if type(step) is not int or not 0 <= step < len(self._actions):
            raise ValueError("Recorded episode exhausted or step invalid")
        if self._last_step is None and step != 0:
            raise ValueError("Recorded replay must start at step zero after reset")
        if self._last_step is not None and step < self._last_step:
            raise ValueError("Recorded replay cannot move backwards without reset")
        self._last_step = step
        return ActionChunk(
            observation.episode_id,
            step,
            self._source_id,
            self._actions[step : step + self._chunk_size],
        )
