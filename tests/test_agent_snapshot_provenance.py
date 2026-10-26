"""V2 actor metadata admission stays separate from units and task outcome claims."""

import copy

import pytest
from vla_platform.datasets.snapshot_reader import validate_demonstration_provenance


def fixture():
    agent = {
        "step": 1,
        "actor": "llm_agent",
        "environment_kind": "simulation",
        "agent_session_id": "agent_fixture",
        "proposal_id": "proposal_fixture",
        "execution_id": "execution_fixture",
        "provider_model": "mock-v1",
        "observation_id": "observation_fixture",
        "observation_sha256": "a" * 64,
        "authority_epoch": 1,
        "clock_policy": "stepped",
        "skill_manifest_version": 1,
    }
    event = {key: value for key, value in agent.items() if key != "step"}
    event.update(command_id=agent["execution_id"], source="agent_command", operation="correct")
    provenance = {
        "schema_version": 1,
        "task_success_verified": False,
        "episodes": [
            {
                "episode_index": 0,
                "capture_schema_version": 2,
                "environment_kind": "simulation",
                "actors": ["human_operator", "llm_agent"],
                "events": [event],
                "action_provenance": [
                    {"step": 0, "actor": "human_operator", "environment_kind": "simulation"},
                    agent,
                ],
            }
        ],
    }
    return provenance, [{"episode_index": 0, "length": 2}]


def test_exact_agent_and_human_frames_admitted_without_success_claim():
    provenance, episodes = fixture()
    validate_demonstration_provenance(provenance, episodes)
    assert provenance["task_success_verified"] is False


@pytest.mark.parametrize(
    "damage",
    [
        "unknown_actor",
        "environment",
        "frames",
        "sequence",
        "missing_identity",
        "bool_epoch",
        "model",
        "event",
        "actor_summary",
        "success",
        "downgrade",
        "missing_episode",
        "duplicate",
    ],
)
def test_actor_metadata_cannot_be_dropped_or_reinterpreted(damage):
    provenance, episodes = fixture()
    episode = provenance["episodes"][0]
    action = episode["action_provenance"][1]
    if damage == "unknown_actor":
        action["actor"] = "operator"
    elif damage == "environment":
        action["environment_kind"] = "physical"
    elif damage == "frames":
        episode["action_provenance"].pop()
    elif damage == "sequence":
        action["step"] = 0
    elif damage == "missing_identity":
        del action["proposal_id"]
    elif damage == "bool_epoch":
        action["authority_epoch"] = True
    elif damage == "model":
        action["provider_model"] = ""
    elif damage == "event":
        episode["events"][0]["proposal_id"] = "other_proposal"
    elif damage == "actor_summary":
        episode["actors"] = ["human_operator"]
    elif damage == "success":
        provenance["task_success_verified"] = True
    elif damage == "downgrade":
        del episode["capture_schema_version"]
    elif damage == "missing_episode":
        provenance["episodes"].clear()
    else:
        provenance["episodes"].append(copy.deepcopy(episode))
        episodes.append({"episode_index": 1, "length": 2})
    with pytest.raises(ValueError):
        validate_demonstration_provenance(provenance, episodes)


def test_legacy_sidecar_retains_previous_interpretation():
    validate_demonstration_provenance({"schema_version": 1}, [{"length": 2}])
    validate_demonstration_provenance({"episodes": [{"episode_index": 0}]}, [{"length": 2}])


@pytest.mark.parametrize("fault", [None, "partial", "hash", "version", "event"])
def test_prompt_provenance_survives_snapshot_and_matches_execution(fault):
    provenance, episodes = fixture()
    action = provenance["episodes"][0]["action_provenance"][1]
    event = provenance["episodes"][0]["events"][0]
    metadata = {"prompt_version": "agent-goal-v1", "prompt_sha256": "d" * 64}
    action.update(metadata)
    event.update(metadata)
    if fault == "partial":
        del action["prompt_sha256"]
    elif fault == "hash":
        action["prompt_sha256"] = "invalid"
    elif fault == "version":
        action["prompt_version"] = "bad version"
    elif fault == "event":
        event["prompt_sha256"] = "e" * 64
    if fault:
        with pytest.raises(ValueError):
            validate_demonstration_provenance(provenance, episodes)
    else:
        validate_demonstration_provenance(provenance, episodes)
