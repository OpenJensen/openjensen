"""Dependency-free temporal admission shared by the API and isolated workers.

Only ACT and SmolVLA have verified independent prediction/execution controls.
Legacy recipes remain coupled; checkpoint resumes use their saved configuration.
"""

import math

TEMPORAL_FIELDS = frozenset(
    {"prediction_horizon", "execution_horizon", "observation_history", "frame_stride"}
)
SPLIT_DEFAULTS = {"act": 100, "smolvla": 50}
IGNORED_CHUNK = frozenset({"diffusion", "multi_task_dit", "vqbet"})


def validate_temporal(recipe, family):
    """Reject unsupported requests before allocation; never mutate the recipe."""
    supplied = TEMPORAL_FIELDS.intersection(recipe)
    if family in IGNORED_CHUNK and "chunk_size" in recipe:
        raise ValueError(
            f"{family} does not support chunk_size; its native temporal contract differs"
        )
    if supplied and family not in SPLIT_DEFAULTS:
        raise ValueError(f"Independent temporal controls are not verified for {family}")
    for key in supplied:
        value = recipe[key]
        if type(value) is not int or not 1 <= value <= 1024:
            raise ValueError(f"{key} must be an integer from 1 to 1024")
    if "chunk_size" in recipe:
        value = recipe["chunk_size"]
        if type(value) is not int or value < 1:
            raise ValueError("chunk_size must be a positive integer")
        if {"prediction_horizon", "execution_horizon"}.intersection(supplied):
            raise ValueError("Use either legacy chunk_size or independent horizons, not both")
        if family in {"psi0", "psi0_base"} and value != 30:
            raise ValueError("Psi-Zero requires chunk_size=30")
    if family not in SPLIT_DEFAULTS:
        return None
    if recipe.get("observation_history", 1) != 1 or recipe.get("frame_stride", 1) != 1:
        raise ValueError(f"{family} currently requires observation_history=1 and frame_stride=1")
    prediction = recipe.get("prediction_horizon", recipe.get("chunk_size", SPLIT_DEFAULTS[family]))
    execution = recipe.get("execution_horizon", prediction)
    if execution > prediction:
        raise ValueError("execution_horizon must be no greater than prediction_horizon")
    return {
        "prediction_horizon": prediction,
        "execution_horizon": execution,
        "observation_history": 1,
        "frame_stride": 1,
    }


def action_timestamps(prediction_horizon, fps):
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Dataset action FPS must be a positive finite number")
    return [index / fps for index in range(prediction_horizon)]


def resolved_temporal(config, fps, family, recipe=None):
    """Record actual policy indices/FPS; requested settings alone are not evidence.

    recipe=None is reserved for checkpoint-owned resume. Other families expose
    native fields and indices without claiming generic split-horizon support.
    """
    action_timestamps(0, fps)
    fields = {}
    for name in (
        "chunk_size",
        "n_action_steps",
        "n_obs_steps",
        "horizon",
        "action_chunk_size",
        "n_action_pred_token",
    ):
        if hasattr(config, name):
            value = getattr(config, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"Resolved policy {name} must be a positive integer")
            fields[name] = value
    action = config.action_delta_indices
    observation = config.observation_delta_indices

    def indices(value, name):
        if value is None:
            return None
        if not isinstance(value, list) or not value or any(type(i) is not int for i in value):
            raise ValueError(f"Resolved {name} indices must be a nonempty integer list")
        return list(value)

    action = indices(action, "action")
    observation = indices(observation, "observation")
    result = {
        "schema_version": 1,
        "family": family,
        "action_fps": fps,
        "policy_fields": fields,
        "action_delta_indices": action,
        "observation_delta_indices": observation,
        "action_delta_timestamps": None if action is None else [i / fps for i in action],
        "observation_delta_timestamps": None
        if observation is None
        else [i / fps for i in observation],
    }
    if family in SPLIT_DEFAULTS:
        prediction, execution = fields.get("chunk_size"), fields.get("n_action_steps")
        if (
            prediction is None
            or execution is None
            or execution > prediction
            or fields.get("n_obs_steps") != 1
        ):
            raise ValueError("Resolved policy has an incompatible temporal configuration")
        if action != list(range(prediction)) or observation not in (None, [0]):
            raise ValueError("Resolved policy temporal indices differ from supported sampling")
        result.update(
            prediction_horizon=prediction,
            execution_horizon=execution,
            observation_history=1,
            frame_stride=1,
        )
        if recipe is not None:
            requested = validate_temporal(recipe, family)
            if any(result[key] != value for key, value in requested.items()):
                raise ValueError("Resolved policy temporal configuration differs from the recipe")
    return result


def check_dataset_temporal(record, dataset):
    """Bind the record to the timestamps actually installed in the dataset loader."""
    if dataset.meta.fps != record["action_fps"]:
        raise ValueError("Dataset FPS differs between temporal contract and split")
    timestamps = dataset.delta_timestamps or {}
    expected = record["action_delta_timestamps"]
    if expected is not None and timestamps.get("action") != expected:
        raise ValueError("Dataset action timestamps differ from the resolved policy")
    expected = record["observation_delta_timestamps"]
    if expected is not None:
        for key in dataset.features:
            if key.startswith("observation.") and timestamps.get(key, [0.0]) != expected:
                raise ValueError("Dataset observation timestamps differ from the resolved policy")
