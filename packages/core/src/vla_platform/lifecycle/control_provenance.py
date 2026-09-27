"""Bind optional simulator coordinates to exact native artifacts without importing ML."""

from pathlib import Path

from .control_schema import FILE, canonical
from .control_schema import metadata as file_metadata

KEYS = ("control_contract", "control_contract_sha256")


def records(metadata):
    yield metadata
    nested = metadata.get("checkpoint")
    if isinstance(nested, dict):
        yield nested


def has_claims(metadata):
    return any(record.get(key) is not None for record in records(metadata) for key in KEYS)


def policy_claims(directory, metadata):
    from .simulation import strict_json

    directory = Path(directory)
    expected = file_metadata(directory)
    if expected:
        from .control_schema import validate

        validate(expected["control_contract"], strict_json(directory / "config.json"))
    matched = False
    for record in records(metadata):
        if not any(record.get(key) is not None for key in KEYS):
            continue
        if not expected or any(
            canonical(record.get(key)) != canonical(expected[key]) for key in KEYS
        ):
            raise ValueError("Simulator control metadata differs from its policy file")
        matched = True
    if expected and not matched:
        raise ValueError("Policy simulator control contract is missing from artifact metadata")
    return expected


def training_claims(directory, metadata):
    from .simulation import strict_json

    checkpoint = Path(directory) / "checkpoint"
    claims = policy_claims(checkpoint / "pretrained_model", metadata)
    outer = file_metadata(checkpoint)
    if canonical(outer) != canonical(claims):
        raise ValueError("Training and policy simulator control contracts differ")
    manifest = strict_json(checkpoint / "manifest.json")
    files = manifest.get("files", {})
    recipe_path = checkpoint / "recipe.json"
    recipe = strict_json(recipe_path) if recipe_path.exists() else {}
    if not isinstance(files, dict):
        raise ValueError("Training manifest files must be an object")
    if not claims:
        if recipe.get("control_contract") is not None or any(
            name in files for name in (FILE, "pretrained_model/" + FILE)
        ):
            raise ValueError("Training checkpoint lost its simulator control contract")
        return {}
    digest = claims["control_contract_sha256"]
    if any(files.get(name) != digest for name in (FILE, "pretrained_model/" + FILE)):
        raise ValueError("Simulator control contract differs from the training manifest")
    if canonical(recipe.get("control_contract")) != canonical(claims["control_contract"]):
        raise ValueError("Simulator control contract differs from the training recipe")
    source = claims["control_contract"]["source"]
    if (
        recipe.get("dataset_source") != "local"
        or recipe.get("dataset_revision") != source["dataset_snapshot_id"]
        or recipe.get("dataset_manifest_sha256") != source["dataset_manifest_sha256"]
    ):
        raise ValueError("Simulator control contract differs from the training dataset")
    return claims


def transform_claims(metadata: dict, admitted: dict) -> dict:
    """Require paired claims when declared; retain omitted historical metadata."""
    if any(key in metadata or admitted.get(key) is not None for key in KEYS):
        return {key: admitted.get(key) for key in KEYS}
    return {}
