import pytest
from vla_platform.lifecycle.contracts import PolicyArtifact
from vla_platform.lifecycle.model_names import ModelNames, descriptive_model_name


def model(id="one", **changes):
    return PolicyArtifact(
        id=id,
        project_id="project",
        job_id="run",
        label="Saved model",
        format="training_checkpoint",
        path="owned",
        manifest_sha256="a" * 64,
        file_bytes=1,
        **changes,
    )


@pytest.mark.parametrize(
    "metadata,expected",
    [
        ({"architecture": "smolvla", "step": 100, "precision": "Q4_0"}, "smolvla-100-q4"),
        (
            {"architecture": "smolvla", "step": 100, "precision": {"language": "Q8_0"}},
            "smolvla-100-q8",
        ),
        ({"architecture": "smolvla", "step": 20, "method": "lora"}, "smolvla-20-lora"),
        ({"architecture": "smolvla", "step": 0, "method": "full"}, "smolvla-0-sft"),
        ({"architecture": "act", "step": 100, "precision": "int8"}, "act-100-int8"),
        ({}, "policy"),
    ],
)
def test_automatic_names_describe_only_recorded_model_properties(metadata, expected):
    assert descriptive_model_name(model(metadata=metadata)) == expected


def test_derived_names_follow_only_same_project_parents_and_preserve_explicit_renames(tmp_path):
    parent = model("parent", metadata={"architecture": "smolvla", "step": 100})
    packed = model("packed", metadata={"precision": "Q4_0"}, parent_ids=["parent"])
    names = ModelNames(tmp_path)
    assert names.decorate(packed, [parent, packed]).run_name == "smolvla-100-q4"
    assert (
        names.decorate(packed, [parent.model_copy(update={"project_id": "foreign"})]).run_name
        == "policy-q4"
    )
    names.rename("project", "run", "Battery pickup")
    renamed = ModelNames(tmp_path).decorate(packed, [parent])
    assert renamed.run_name == "Battery pickup"
    assert renamed.metadata == packed.metadata
    assert renamed.manifest_sha256 == packed.manifest_sha256
    assert (
        names.decorate(packed.model_copy(update={"job_id": "another"}), [parent]).run_name
        == "smolvla-100-q4"
    )
